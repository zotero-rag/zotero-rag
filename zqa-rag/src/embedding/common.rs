//! Structs, functions, and traits shared by embedding clients and other embedding-related code in
//! this crate.

use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_schema::{DataType, Field};
use futures::{StreamExt, stream};
use indicatif::ProgressBar;
use lancedb::embeddings::EmbeddingFunction;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::capabilities::EmbeddingProvider;
use crate::constants::{
    DEFAULT_COHERE_EMBEDDING_DIM, DEFAULT_GEMINI_EMBEDDING_DIM, DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_OLLAMA_EMBEDDING_DIM, DEFAULT_OPENAI_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_DIM,
    DEFAULT_ZEROENTROPY_EMBEDDING_DIM,
};
use crate::http_client::HttpClient;
use crate::llm::errors::LLMError;
use crate::providers::ProviderId;
use crate::providers::registry::provider_registry;

/// A struct containing information about texts that failed to embed.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct FailedTexts {
    /// The embedding provider that was used.
    pub embedding_provider: String,
    /// The texts that failed to embed.
    pub texts: Vec<String>,
}

impl std::fmt::Display for FailedTexts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Embedding provider: {}\n\nTexts that failed:\n",
            &self.embedding_provider
        ))?;

        for text in &self.texts {
            let words = text
                .split_whitespace()
                .take(10)
                .collect::<Vec<_>>()
                .join(" ");

            f.write_fmt(format_args!("\t{words}\n"))?;
        }

        Ok(())
    }
}

/// Returns the embedding dimension given an embedding provider.
///
/// # Arguments
///
/// * `embedding_provider` - Embedding provider enum.
///
/// # Returns
///
/// The dimensions of the embedding provider.
#[must_use]
pub fn get_embedding_dims_by_provider(embedding_provider: EmbeddingProvider) -> u32 {
    match embedding_provider {
        EmbeddingProvider::OpenAI => DEFAULT_OPENAI_EMBEDDING_DIM,
        EmbeddingProvider::VoyageAI => DEFAULT_VOYAGE_EMBEDDING_DIM,
        EmbeddingProvider::Gemini => DEFAULT_GEMINI_EMBEDDING_DIM,
        EmbeddingProvider::Ollama => DEFAULT_OLLAMA_EMBEDDING_DIM as u32,
        EmbeddingProvider::Cohere => DEFAULT_COHERE_EMBEDDING_DIM,
        EmbeddingProvider::ZeroEntropy => DEFAULT_ZEROENTROPY_EMBEDDING_DIM,
    }
}

/// Gets an embedding provider with configuration
///
/// # Arguments
///
/// * `config`: Provider-specific configuration
///
/// # Returns
///
/// A thread-safe object that can compute query embeddings
///
/// # Errors
///
/// Returns an error if provider configuration is invalid or initialization fails.
pub fn get_embedding_provider_with_config(
    config: &EmbeddingProviderConfig,
) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
    provider_registry().create_embedding(config)
}

/// Configuration enum for embedding providers
#[derive(Debug, Clone)]
pub enum EmbeddingProviderConfig {
    /// Configuration for OpenAI embedding provider
    OpenAI(crate::config::OpenAIConfig),
    /// Configuration for VoyageAI embedding provider
    VoyageAI(crate::config::VoyageAIConfig),
    /// Configuration for Gemini embedding provider
    Gemini(crate::config::GeminiConfig),
    /// Configuration for Cohere embedding provider
    Cohere(crate::config::CohereConfig),
    /// Configuration for `ollama` embedding provider
    Ollama(crate::config::OllamaConfig),
    /// Configuration for ZeroEntropy embedding provider
    ZeroEntropy(crate::config::ZeroEntropyConfig),
}

impl EmbeddingProviderConfig {
    /// Return the canonical provider ID.
    #[must_use]
    pub const fn provider_id(&self) -> ProviderId {
        match self {
            Self::Ollama(_) => ProviderId::Ollama,
            Self::OpenAI(_) => ProviderId::OpenAI,
            Self::Gemini(_) => ProviderId::Gemini,
            Self::VoyageAI(_) => ProviderId::VoyageAI,
            Self::Cohere(_) => ProviderId::Cohere,
            Self::ZeroEntropy(_) => ProviderId::ZeroEntropy,
        }
    }

    /// Returns the embedding provider enum
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn provider(&self) -> EmbeddingProvider {
        self.provider_id()
            .try_into()
            .expect("Embedding configs always map to embedding providers")
    }

    /// Returns the name of the embedding provider.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        self.provider_id().as_str()
    }

    /// Returns the embedding model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        match self {
            Self::OpenAI(c) => &c.embedding_model,
            Self::VoyageAI(c) => &c.embedding_model,
            Self::Gemini(c) => &c.embedding_model,
            Self::Cohere(c) => &c.embedding_model,
            Self::Ollama(c) => &c.embedding_model,
            Self::ZeroEntropy(c) => &c.embedding_model,
        }
    }
}

/// A trait intended to be used for responses from embedding provider APIs. Typically, these APIs
/// return different structures for successful (200) vs. non-successful (4xx or 5xx) requests. The
/// pattern this repo uses is to have an untagged enum as the response struct that `serde`
/// deserializes to.
pub trait EmbeddingApiResponse {
    /// The type of the successful response.
    type Success;
    /// The type of the error response.
    type Error;

    /// Returns whether the request was successful.
    fn is_success(&self) -> bool;

    /// Returns a list of embedding vectors for each text passed in the request, or `None` if the
    /// request was unsuccessful.
    fn get_embeddings(self) -> Option<Vec<Vec<f32>>>;

    /// Returns the error message from the API, or `None` if it was successful.
    fn get_error_message(self) -> Option<String>;
}

/// A generic version of a function that sends a request to an embedding provider. This allows you
/// to simply define the types of the request and response, write the corresponding traits for
/// those types, and then use this function to handle the details of the request batching and error
/// handling.
///
/// Note that in case of failed requests, this function adds zero vectors in place of texts that
/// failed. This is currently implemented this way because Arrow arrays are somewhat obnoxious to
/// work with and it's not the easiest thing to remove values by index. This is, admittedly, not
/// the best UX, and the workaround provided is the `/embed` command that retries these at a later
/// time.
///
/// TODO: Consider making this more robust later.
///
/// # Arguments:
///
/// * `source` - The source Arrow array containing the texts. This is expected to be an `Arc<dyn
///   Array>`, since that is what LanceDB gives you; as such this is the "native" type. This might
///   be made more general via an extension trait in the future.
/// * `api_url` - The embedding API endpoint.
/// * `api_key` - The API key for the service.
/// * `api_client` - The `HttpClient` trait implementation to use. For real use, you almost
///   certainly want a `ReqwestClient`; the trait allows for easy testing.
/// * `make_request` - A function that transforms a raw `Vec<String>` inputs to the API's expected
///   request format
/// * `embedding_provider` - An owned type containing the name of the embedding provider. This does
///   *not* have to be the same as the value in the `EmbeddingProviders` enum, though it is
///   recommended that you use that value. This is mainly for logging purposes.
/// * `batch_size` - To account for RPM and TPM limits imposed by APIs, you should calculate
///   reasonable values of a "batch size"--the number of texts to send at once, and the time to
///   wait between requests in seconds.
/// * `wait_after_request_s` - See `batch_size`.
/// * `embedding_dim` - The embedding dimensions you expect to receive.
///
/// # Returns
///
/// If successful, an Arrow array containing the embeddings.
///
/// # Errors
///
/// * `LLMError::TimeoutError` - If the HTTP request times out
/// * `LLMError::CredentialError` - If the API returns 401 or 403 status
/// * `LLMError::HttpStatusError` - If the API returns other unsuccessful HTTP status codes
/// * `LLMError::NetworkError` - If a network connectivity error occurs
/// * `LLMError::DeserializationError` - If the API response cannot be parsed
/// * `LLMError::InvalidHeaderError` - If header values cannot be parsed
/// * `LLMError::GenericLLMError` - If other HTTP errors occur or Arrow array creation fails
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) async fn compute_embeddings_async<T, U, F>(
    source: Arc<dyn arrow_array::Array>,
    api_url: &str,
    api_key: &str,
    api_client: impl HttpClient + Clone,
    make_request: F,
    embedding_provider: String,
    batch_size: usize,
    wait_after_request_s: u64,
    embedding_dim: usize,
) -> Result<Arc<dyn arrow_array::Array>, LLMError>
where
    T: Serialize + Send + Sync + std::fmt::Debug,
    U: EmbeddingApiResponse + for<'de> Deserialize<'de> + std::fmt::Debug + Send,
    F: Fn(Vec<String>) -> T + Send + Clone,
{
    let source_array = arrow_array::cast::as_string_array(&source);
    let texts: Vec<Option<String>> = source_array.iter().map(|s| s.map(str::to_owned)).collect();

    log::info!("Processing {} input texts.", texts.len());
    let bar = ProgressBar::new(texts.len() as u64);

    let max_concurrent = env::var("MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT_REQUESTS)
        .max(1);

    let api_url = api_url.to_string();
    let api_key = api_key.to_string();

    let batches: Vec<Vec<Option<String>>> = texts.chunks(batch_size).map(<[_]>::to_vec).collect();
    let num_batches = batches.len();

    let futures = batches.into_iter().enumerate().map(|(i, batch)| {
        let api_url = api_url.clone();
        let api_key = api_key.clone();
        let api_client = api_client.clone();
        let make_request = make_request.clone();
        let embedding_provider = embedding_provider.clone();
        let bar = bar.clone();

        async move {
            // Build a mask of "real" vs "empty" slots to handle providers that reject empty strings.
            let mask: Vec<bool> = batch
                .iter()
                .map(|opt| opt.as_ref().is_some_and(|s| !s.trim().is_empty()))
                .collect();

            let cur_texts: Vec<String> = batch
                .iter()
                .filter_map(|opt| opt.clone().filter(|s| !s.trim().is_empty()))
                .collect();

            // (embeddings_for_batch, fail_count, failed_texts, masked_count)
            type BatchResult = (Vec<Vec<f32>>, usize, Vec<String>, usize);

            if cur_texts.is_empty() {
                let embeddings =
                    std::iter::repeat_n(vec![0.0f32; embedding_dim], batch.len()).collect();
                if wait_after_request_s > 0 && i < num_batches - 1 {
                    tokio::time::sleep(Duration::from_secs(wait_after_request_s)).await;
                }
                return Ok::<BatchResult, LLMError>((embeddings, 0, vec![], mask.len()));
            }

            let mut headers = HeaderMap::new();
            headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
            headers.insert("Content-Type", "application/json".parse()?);
            headers.insert("Accept", "application/json".parse()?);

            let start_time = Instant::now();
            let request = make_request(cur_texts);
            let response = api_client.post_json(&api_url, headers, &request).await?;

            let body = response.text().await?;
            log::debug!(
                "{embedding_provider} embedding request took {:.1?}",
                start_time.elapsed()
            );

            let api_response: U = serde_json::from_str(&body)?;

            let (embeddings_opt, error_message) = if api_response.is_success() {
                (api_response.get_embeddings(), None)
            } else {
                (None, api_response.get_error_message())
            };

            let result: BatchResult = if let Some(emb) = embeddings_opt {
                let mut it = emb.into_iter();
                let mut batch_embs = Vec::with_capacity(batch.len());
                let masked_count = mask.iter().filter(|&&m| !m).count();
                for &is_real in &mask {
                    if is_real && let Some(embedding) = it.next() {
                        batch_embs.push(embedding);
                    } else {
                        batch_embs.push(vec![0.0_f32; embedding_dim]);
                    }
                }
                (batch_embs, 0, vec![], masked_count)
            } else {
                let error_msg = error_message.unwrap_or_else(|| String::from("No error found."));
                log::error!("Got a 4xx response from the {embedding_provider} API: {error_msg}\n");
                log::error!("We tried sending the request: {request:#?}\n");
                let fail_texts: Vec<String> =
                    batch.iter().filter_map(|t| t.as_ref()).cloned().collect();
                let zeros = std::iter::repeat_n(vec![0.0f32; embedding_dim], batch.len()).collect();
                (zeros, batch.len(), fail_texts, 0)
            };

            if wait_after_request_s > 0 && i < num_batches - 1 {
                tokio::time::sleep(Duration::from_secs(wait_after_request_s)).await;
            }

            bar.inc(batch.len() as u64);

            Ok::<BatchResult, LLMError>(result)
        }
    });

    let results = stream::iter(futures)
        .buffered(max_concurrent)
        .collect::<Vec<_>>()
        .await;

    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut fail_count = 0;
    let mut total_masked = 0;
    let mut failed_texts: Vec<String> = Vec::new();

    for result in results {
        match result {
            Ok((batch_embs, batch_fail_count, batch_failed_texts, batch_masked)) => {
                all_embeddings.extend(batch_embs);
                fail_count += batch_fail_count;
                total_masked += batch_masked;
                failed_texts.extend(batch_failed_texts);
            }
            Err(e) => return Err(e),
        }
    }

    if fail_count > 0 {
        let failed = FailedTexts {
            embedding_provider,
            texts: failed_texts,
        };

        log::error!("Some texts failed to embed:\n{failed}");
    }

    log::info!(
        "Processing finished. Statistics:\n{fail_count} items failed.\n{total_masked} items were empty."
    );

    // Convert to Arrow FixedSizeListArray
    let flattened = all_embeddings.iter().flatten().copied();
    let values = arrow_array::Float32Array::from_iter_values(flattened);

    let list_array = arrow_array::FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        embedding_dim as i32,
        Arc::new(values),
        None,
    )
    .map_err(|e| LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {e}")))?;

    Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
}
