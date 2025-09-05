use arrow_schema::{DataType, Field};
use indicatif::ProgressBar;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{
    env, fs,
    time::{Duration, Instant},
};

use lancedb::embeddings::EmbeddingFunction;

use crate::constants::{GEMINI_EMBEDDING_DIM, OPENAI_EMBEDDING_DIM, VOYAGE_EMBEDDING_DIM};
use crate::embedding::voyage::VoyageAIClient;
use crate::llm::errors::LLMError;
use crate::llm::gemini::GeminiClient;
use crate::llm::http_client::{HttpClient, ReqwestClient};
use crate::llm::openai::OpenAIClient;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FailedTexts {
    pub embedding_provider: String,
    pub texts: Vec<String>,
}

/// Returns the embedding dimension given an embedding provider. Note that for Anthropic, we
/// actually use OpenAI's embeddings.
///
/// # Arguments
///
/// * `embedding_name` - Embedding provider name. Must be one of "openai", "anthropic", or
///   "voyageai".
///
/// # Returns
///
/// The dimensions of the embedding provider.
pub fn get_embedding_dims_by_provider(embedding_name: &str) -> u32 {
    match embedding_name {
        "openai" => OPENAI_EMBEDDING_DIM,
        "anthropic" => OPENAI_EMBEDDING_DIM,
        "voyageai" => VOYAGE_EMBEDDING_DIM,
        "gemini" => GEMINI_EMBEDDING_DIM,
        _ => panic!("Invalid embedding provider."),
    }
}

/// Gets an embedding provider, i.e., an atomically reference-counted, heap-allocated
/// `EmbeddingFunction` implementation.
///
/// # Arguments
///
/// * `embedding_name`: Embedding provider name. Must be one of "openai", "anthropic", or
///   "voyageai".
///
/// # Returns
///
/// An thread-safe object that can compute query embeddings.
pub fn get_embedding_provider(
    embedding_name: &str,
) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
    match embedding_name {
        "openai" => Ok(Arc::new(OpenAIClient::<ReqwestClient>::default())),
        "anthropic" => Ok(Arc::new(OpenAIClient::<ReqwestClient>::default())),
        "voyageai" => Ok(Arc::new(VoyageAIClient::<ReqwestClient>::default())),
        "gemini" => Ok(Arc::new(GeminiClient::<ReqwestClient>::default())),
        _ => Err(LLMError::InvalidProviderError(embedding_name.to_string())),
    }
}

/// A trait indicating reranking capabilities. This is made generic since it is expected that users
/// will pass in whatever type they convert `RecordBatch` to, as long as we can convert it into a
/// string in a non-consuming way. A user may also choose to `map` their `Vec<RecordBatch>` with
/// custom logic if they prefer (or if, for some reason, their struct's `AsRef<str>` is implemented
/// with a different purpose, but the resulting string isn't useful for reranking purposes).
pub trait Rerank<T: AsRef<str>> {
    fn rerank(items: Vec<T>, query: &str) -> Vec<T>;
}

pub trait EmbeddingApiRequestTexts<T> {
    fn from_texts(texts: Vec<String>) -> T;
}

pub trait EmbeddingApiResponse {
    type Success;
    type Error;

    fn is_success(&self) -> bool;
    fn get_embeddings(self) -> Option<Vec<Vec<f32>>>;
    fn get_error_message(self) -> Option<String>;
}

pub async fn compute_embeddings_async<
    T: EmbeddingApiResponse + for<'de> Deserialize<'de>,
    U: EmbeddingApiRequestTexts<U> + Serialize + Send + Sync + std::fmt::Debug,
>(
    source: Arc<dyn arrow_array::Array>,
    api_url: &'static str,
    api_key_var: &'static str,
    api_client: impl HttpClient,
    request_fn: fn(Vec<String>) -> U,
    embedding_provider: String,
    batch_size: usize,
    wait_after_request_s: u64,
    embedding_dim: usize,
) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
    let source_array = arrow_array::cast::as_string_array(&source);
    let texts: Vec<Option<String>> = source_array
        .iter()
        .map(|s| s.map(|s| s.to_owned()))
        .collect();

    log::info!("Processing {} input texts.", texts.len());
    let bar = ProgressBar::new(texts.len().try_into().unwrap());

    let api_key = env::var(api_key_var)?;

    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();

    // Gather failed texts
    let mut fail_count = 0;
    let mut total_masked = 0;
    let mut failed_texts: Vec<String> = Vec::new();

    let chunks = texts.chunks(batch_size);
    let num_chunks = chunks.len();

    for (i, batch) in chunks.enumerate() {
        // Build mask for non-empty strings
        let mask: Vec<bool> = batch
            .iter()
            .map(|opt| opt.as_ref().is_some_and(|s| !s.trim().is_empty()))
            .collect();

        // Extract non-empty current texts
        let cur_texts: Vec<String> = batch
            .iter()
            .filter_map(|opt| opt.clone().filter(|s| !s.trim().is_empty()))
            .collect();

        // If none are real, just push zeros for whole batch
        if cur_texts.is_empty() {
            all_embeddings.extend(std::iter::repeat_n(vec![0.0; embedding_dim], batch.len()));
        } else {
            let mut headers = HeaderMap::new();
            headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
            headers.insert("Content-Type", "application/json".parse()?);
            headers.insert("Accept", "application/json".parse()?);

            let start_time = Instant::now();
            let request = request_fn(cur_texts);
            let response = api_client.post_json(api_url, headers, &request).await?;

            let body = response.text().await?;
            log::debug!("Cohere embedding request took {:.1?}", start_time.elapsed());

            let cohere_response: T = serde_json::from_str(&body)?;

            if cohere_response.is_success() {
                let mut it = cohere_response.get_embeddings().unwrap().into_iter();

                // 4. Weave the real embeddings back into the right spots, zero‐padding empties
                let mut batch_embs = Vec::with_capacity(batch.len());
                for &is_real in &mask {
                    if is_real && let Some(embedding) = it.next() {
                        batch_embs.push(embedding);
                    } else {
                        batch_embs.push(vec![0.0_f32; embedding_dim]);
                    }
                }
                all_embeddings.extend(batch_embs);

                total_masked += mask.iter().filter(|mask| !**mask).count();
            } else {
                eprintln!(
                    "Got a 4xx response from the {} API: {}\n",
                    embedding_provider,
                    cohere_response
                        .get_error_message()
                        .unwrap_or(String::from("No error found."))
                );
                eprintln!("We tried sending the request: {request:#?}\n");

                fail_count += batch.len();
                failed_texts.extend(batch.iter().filter_map(|text| text.as_ref()).cloned());

                let zeros: Vec<Vec<f32>> =
                    std::iter::repeat_n(vec![0.0; embedding_dim], batch.len()).collect();
                all_embeddings.extend(zeros);
            }
        }

        bar.inc(batch_size as u64);

        if i < num_chunks - 1 {
            tokio::time::sleep(Duration::from_secs(wait_after_request_s)).await;
        }
    }

    if fail_count > 0 {
        let failed = FailedTexts {
            embedding_provider,
            texts: failed_texts,
        };
        let encoded = serde_json::to_string_pretty(&failed)?;

        if let Err(e) = fs::write("failed.json", encoded) {
            eprintln!("We could not write out the failed texts to 'failed.json': {e}");
        } else {
            println!(
                "We have written the failed texts to 'failed.json'. Consider using /repair to fix this."
            );
        }
    }

    log::info!(
        "Processing finished. Statistics:\n{fail_count} items failed.\n{total_masked} items were empty."
    );

    // Convert to Arrow FixedSizeListArray
    let flattened: Vec<f32> = all_embeddings.iter().flatten().copied().collect();
    let values = arrow_array::Float32Array::from(flattened);

    let list_array = arrow_array::FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        embedding_dim as i32,
        Arc::new(values),
        None,
    )
    .map_err(|e| LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {e}")))?;

    Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
}
