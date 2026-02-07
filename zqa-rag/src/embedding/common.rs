//! Structs, functions, and traits shared by embedding clients and other embedding-related code in
//! this crate.

use arrow_schema::{DataType, Field};
use indicatif::ProgressBar;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{
    fs,
    future::Future,
    pin::Pin,
    time::{Duration, Instant},
};

use lancedb::embeddings::EmbeddingFunction;

use crate::constants::{
    DEFAULT_COHERE_EMBEDDING_DIM, DEFAULT_GEMINI_EMBEDDING_DIM, DEFAULT_OPENAI_EMBEDDING_DIM,
    DEFAULT_VOYAGE_EMBEDDING_DIM,
};
use crate::embedding::cohere::CohereClient;
use crate::embedding::voyage::VoyageAIClient;
use crate::llm::errors::LLMError;
use crate::llm::gemini::GeminiClient;
use crate::llm::http_client::{HttpClient, ReqwestClient};
use crate::llm::openai::OpenAIClient;

/// A struct containing information about texts that failed to embed.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FailedTexts {
    /// The embedding provider that was used.
    pub embedding_provider: String,
    /// The texts that failed to embed.
    pub texts: Vec<String>,
}

/// Returns the embedding dimension given an embedding provider. Note that for Anthropic, we
/// actually use OpenAI's embeddings.
///
/// # Arguments
///
/// * `embedding_name` - Embedding provider name. Must be one of "openai", "anthropic",
///   "voyageai", or "cohere".
///
/// # Returns
///
/// The dimensions of the embedding provider.
///
/// # Panics
///
/// * If an invalid embedding provider name is provided.
#[must_use]
pub fn get_embedding_dims_by_provider(embedding_name: &str) -> u32 {
    match embedding_name {
        "openai" | "anthropic" => DEFAULT_OPENAI_EMBEDDING_DIM,
        "voyageai" => DEFAULT_VOYAGE_EMBEDDING_DIM,
        "gemini" => DEFAULT_GEMINI_EMBEDDING_DIM,
        "cohere" => DEFAULT_COHERE_EMBEDDING_DIM,
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
///
/// # Errors
///
/// Returns an error if the embedding provider name is not recognized.
pub fn get_embedding_provider(
    embedding_name: &str,
) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
    match embedding_name {
        "openai" | "anthropic" => Ok(Arc::new(OpenAIClient::<ReqwestClient>::default())),
        "voyageai" => Ok(Arc::new(VoyageAIClient::<ReqwestClient>::default())),
        "gemini" => Ok(Arc::new(GeminiClient::<ReqwestClient>::default())),
        "cohere" => Ok(Arc::new(CohereClient::<ReqwestClient>::default())),
        _ => Err(LLMError::InvalidProviderError(embedding_name.to_string())),
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
    config: EmbeddingProviderConfig,
) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
    match config {
        EmbeddingProviderConfig::OpenAI(cfg) => {
            Ok(Arc::new(OpenAIClient::<ReqwestClient>::with_config(cfg)))
        }
        EmbeddingProviderConfig::VoyageAI(cfg) => {
            Ok(Arc::new(VoyageAIClient::<ReqwestClient>::with_config(cfg)))
        }
        EmbeddingProviderConfig::Gemini(cfg) => {
            Ok(Arc::new(GeminiClient::<ReqwestClient>::with_config(cfg)))
        }
        EmbeddingProviderConfig::Cohere(cfg) => {
            Ok(Arc::new(CohereClient::<ReqwestClient>::with_config(cfg)))
        }
    }
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
}

impl EmbeddingProviderConfig {
    /// Returns the name of the embedding provider.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        match self {
            EmbeddingProviderConfig::OpenAI(_) => "openai",
            EmbeddingProviderConfig::VoyageAI(_) => "voyageai",
            EmbeddingProviderConfig::Gemini(_) => "gemini",
            EmbeddingProviderConfig::Cohere(_) => "cohere",
        }
    }
}

/// A trait indicating reranking capabilities. This is made generic since it is expected that users
/// will pass in whatever type they convert `RecordBatch` to, as long as we can convert it into a
/// string in a non-consuming way. A user may also choose to `map` their `Vec<RecordBatch>` with
/// custom logic if they prefer (or if, for some reason, their struct's `AsRef<str>` is implemented
/// with a different purpose, but the resulting string isn't useful for reranking purposes).
pub trait Rerank<T: AsRef<str>> {
    /// Rerank items using the provider.
    ///
    /// # Arguments:
    ///
    /// * `items` - The items to rerank.
    /// * `query` - The query to rerank against.
    ///
    /// # Returns
    ///
    /// A vector of items reranked using the provider.
    fn rerank<'a>(
        &'a self,
        items: Vec<T>,
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<T>, LLMError>> + Send + 'a>>
    where
        T: 'a;
}

/// A factory method for getting a reranking provider.
///
/// # Arguments:
///
/// * `provider` - The name of the provider to get.
///
/// # Returns
///
/// An `Arc<dyn Rerank<T>>` object, or an `LLMError` if the provider is not supported.
///
/// # Errors
///
/// Returns an error if the provider name is not recognized.
pub fn get_reranking_provider<T: AsRef<str> + Send + Clone>(
    provider: &str,
) -> Result<Arc<dyn Rerank<T>>, LLMError> {
    match provider {
        "voyageai" => Ok(Arc::new(VoyageAIClient::<ReqwestClient>::default())),
        "cohere" => Ok(Arc::new(CohereClient::<ReqwestClient>::default())),
        _ => Err(LLMError::InvalidProviderError(provider.to_string())),
    }
}

/// Gets a reranking provider with configuration
///
/// # Arguments
///
/// * `config`: Provider-specific configuration
///
/// # Returns
///
/// A thread-safe object that can rerank items
///
/// # Errors
///
/// Returns an error if provider configuration is invalid or initialization fails.
pub fn get_reranking_provider_with_config<T: AsRef<str> + Send + Clone>(
    config: RerankProviderConfig,
) -> Result<Arc<dyn Rerank<T>>, LLMError> {
    match config {
        RerankProviderConfig::VoyageAI(cfg) => {
            Ok(Arc::new(VoyageAIClient::<ReqwestClient>::with_config(cfg)))
        }
        RerankProviderConfig::Cohere(cfg) => {
            Ok(Arc::new(CohereClient::<ReqwestClient>::with_config(cfg)))
        }
    }
}

/// Configuration enum for reranking providers
#[derive(Debug, Clone)]
pub enum RerankProviderConfig {
    /// Configuration for VoyageAI reranking provider
    VoyageAI(crate::config::VoyageAIConfig),
    /// Configuration for Cohere reranking provider
    Cohere(crate::config::CohereConfig),
}

/// A trait expected to be implemented by requests to embedding providers. Typically, you want to
/// use this by making the struct generic over itself, like so:
///
/// ```rs
/// struct ExampleEmbeddingRequest {
///     texts: Vec<String>,
///     model: String
/// }
///
/// impl EmbeddingApiRequestTexts<ExampleEmbeddingRequest> for ExampleEmbeddingRequest {
///     fn from_texts(texts: Vec<String>) -> Self {
///         Self {
///             texts,
///             model: "example-model".into()
///         }
///     }
/// }
/// ```
///
pub trait EmbeddingApiRequestTexts<T> {
    /// Create an instance of `T` from a vector of texts.
    fn from_texts(texts: Vec<String>) -> T;
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
pub async fn compute_embeddings_async<
    T: EmbeddingApiRequestTexts<T> + Serialize + Send + Sync + std::fmt::Debug,
    U: EmbeddingApiResponse + for<'de> Deserialize<'de> + std::fmt::Debug,
>(
    source: Arc<dyn arrow_array::Array>,
    api_url: &str,
    api_key: &str,
    api_client: impl HttpClient,
    embedding_provider: String,
    batch_size: usize,
    wait_after_request_s: u64,
    embedding_dim: usize,
) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
    let source_array = arrow_array::cast::as_string_array(&source);
    let texts: Vec<Option<String>> = source_array.iter().map(|s| s.map(str::to_owned)).collect();

    log::info!("Processing {} input texts.", texts.len());
    let bar = ProgressBar::new(texts.len() as u64);

    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();

    // Gather failed texts
    let mut fail_count = 0;
    let mut total_masked = 0;
    let mut failed_texts: Vec<String> = Vec::new();

    let chunks = texts.chunks(batch_size);
    let num_chunks = chunks.len();

    for (i, batch) in chunks.enumerate() {
        // For every batch, we need to handle the case of empty/whitespace strings, since some
        // providers (and by some I mean one and by one I mean Voyage) do not like handling them.
        // 1. Build a mask of "real" vs "empty" slots
        let mask: Vec<bool> = batch
            .iter()
            .map(|opt| opt.as_ref().is_some_and(|s| !s.trim().is_empty()))
            .collect();

        // 2. Extract only the non-empty strings to send
        let cur_texts: Vec<String> = batch
            .iter()
            .filter_map(|opt| opt.clone().filter(|s| !s.trim().is_empty()))
            .collect();

        // 3. If none are real, just push zeros for whole batch
        if cur_texts.is_empty() {
            all_embeddings.extend(std::iter::repeat_n(vec![0.0; embedding_dim], batch.len()));
            bar.inc(batch_size as u64);

            if i < num_chunks - 1 {
                tokio::time::sleep(Duration::from_secs(wait_after_request_s)).await;
            }
            continue;
        }

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
        headers.insert("Content-Type", "application/json".parse()?);
        headers.insert("Accept", "application/json".parse()?);

        let start_time = Instant::now();
        let request = T::from_texts(cur_texts);
        let response = api_client.post_json(api_url, headers, &request).await?;

        let body = response.text().await?;
        log::debug!(
            "{embedding_provider} embedding request took {:.1?}",
            start_time.elapsed()
        );

        let api_response: U = serde_json::from_str(&body)?;

        let (embeddings, error_message) = if api_response.is_success() {
            (api_response.get_embeddings(), None)
        } else {
            (None, api_response.get_error_message())
        };

        if let Some(emb) = embeddings {
            let mut it = emb.into_iter();

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
            let error_message = error_message.unwrap_or_else(|| String::from("No error found."));
            eprintln!("Got a 4xx response from the {embedding_provider} API: {error_message}\n",);
            eprintln!("We tried sending the request: {request:#?}\n");

            fail_count += batch.len();
            failed_texts.extend(batch.iter().filter_map(|text| text.as_ref()).cloned());

            let zeros: Vec<Vec<f32>> =
                std::iter::repeat_n(vec![0.0; embedding_dim], batch.len()).collect();
            all_embeddings.extend(zeros);
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
