//! Structs, functions, and traits shared by embedding clients and other embedding-related code in
//! this crate.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_schema::{DataType, Field};
use futures::{StreamExt, TryStreamExt, stream};
use lancedb::embeddings::EmbeddingFunction;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::capabilities::EmbeddingProvider;
use crate::http_client::HttpClient;
use crate::llm::errors::LLMError;
use crate::providers::ProviderId;
use crate::providers::registry::provider_registry;
use crate::requests::request_with_backoff;

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
            self.embedding_provider
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
#[non_exhaustive]
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
        }
    }

    /// Returns the embedding dimensions.
    #[must_use]
    pub fn dims(&self) -> usize {
        match self {
            Self::OpenAI(c) => c.embedding_dims,
            Self::VoyageAI(c) => c.embedding_dims,
            Self::Gemini(c) => c.embedding_dims,
            Self::Cohere(c) => c.embedding_dims,
            Self::Ollama(c) => c.embedding_dims,
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
/// Rate-limited requests and transient server errors are retried with exponential backoff (see
/// [`request_with_backoff`]) before a batch counts as failed. A batch also counts as failed, rather
/// than returning an error, when the provider returns any other unsuccessful status or a body that
/// cannot be parsed.
///
/// Texts in a failed batch, and texts the provider returned no vector for, are null entries in
/// the returned array, so callers can tell them apart from real embeddings: search should fail on
/// them, and ingestion should skip those rows and retry them later. Empty texts are not sent to
/// the provider and get zero vectors.
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
/// * `max_concurrent` - The maximum number of batches to request at once.
/// * `max_retries` - The maximum number of retries for a batch that fails in a retryable way.
///
/// # Returns
///
/// If successful, an Arrow array containing the embeddings, with a null entry for each text that
/// could not be embedded.
///
/// # Errors
///
/// * `LLMError::TimeoutError` - If the HTTP request times out
/// * `LLMError::NetworkError` - If a network connectivity error occurs
/// * `LLMError::CredentialError` - If the API returns 401 or 403
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
    max_concurrent: usize,
    max_retries: usize,
) -> Result<Arc<dyn arrow_array::Array>, LLMError>
where
    T: Serialize + Send + Sync + std::fmt::Debug,
    U: EmbeddingApiResponse + for<'de> Deserialize<'de> + std::fmt::Debug + Send,
    F: Fn(Vec<String>) -> T + Send + Clone,
{
    let source_array = arrow_array::cast::as_string_array(&source);
    let texts: Vec<Option<String>> = source_array.iter().map(|s| s.map(str::to_owned)).collect();

    log::info!("Processing {} input texts.", texts.len());

    let max_concurrent = max_concurrent.max(1);

    let api_url = api_url.to_string();
    let api_key = api_key.to_string();

    let batches: Vec<Vec<Option<String>>> = texts.chunks(batch_size).map(<[_]>::to_vec).collect();
    let num_batches = batches.len();
    log::debug!(
        "Embedding run: provider={embedding_provider}, dimensions={embedding_dim}, batches={num_batches}, batch_size={batch_size}, concurrency={max_concurrent}, wait_seconds={wait_after_request_s}"
    );

    let futures = batches.into_iter().enumerate().map(|(i, batch)| {
        let api_url = api_url.clone();
        let api_key = api_key.clone();
        let api_client = api_client.clone();
        let make_request = make_request.clone();
        let embedding_provider = embedding_provider.clone();

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
            log::debug!("{embedding_provider} embedding batch {}/{num_batches}: inputs={}, nonempty={}",
                i + 1, batch.len(), cur_texts.len());

            // (embeddings_for_batch, fail_count, failed_texts, masked_count). A `None` embedding
            // marks a text that could not be embedded.
            type BatchResult = (Vec<Option<Vec<f32>>>, usize, Vec<String>, usize);

            if cur_texts.is_empty() {
                log::debug!("{embedding_provider} embedding batch {}: skipping API call, {} empty inputs replaced with zero vectors", i + 1, batch.len());
                let embeddings =
                    std::iter::repeat_n(Some(vec![0.0f32; embedding_dim]), batch.len()).collect();
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
            let outcome: Result<Vec<Vec<f32>>, String> =
                match request_with_backoff(&api_client, &api_url, &headers, &request, max_retries).await {
                    Ok(response) => {
                        let body = response.text().await?;
                        match serde_json::from_str::<U>(&body) {
                            Ok(api_response) if api_response.is_success() => api_response
                                .get_embeddings()
                                .ok_or_else(|| String::from("No embeddings in response.")),
                            Ok(api_response) => Err(api_response
                                .get_error_message()
                                .unwrap_or_else(|| String::from("No error found."))),
                            Err(e) => Err(format!("Could not parse response ({e}): {body}")),
                        }
                    }
                    // The status was not retryable, or retries ran out: record the batch as failed
                    // instead of discarding the batches that succeeded.
                    Err(LLMError::HttpStatusError(body)) => Err(body),
                    Err(e) => return Err(e),
                };
            log::debug!(
                "{embedding_provider} embedding batch {}: succeeded={}, elapsed={:.1?}",
                i + 1, outcome.is_ok(), start_time.elapsed()
            );

            let result: BatchResult = match outcome {
                Ok(emb) => {
                    let expected = mask.iter().filter(|&&is_real| is_real).count();
                    let missing = expected.saturating_sub(emb.len());
                    log::debug!("{embedding_provider} embedding batch {}: expected_vectors={expected}, returned_vectors={}, empty_inputs={}, missing_vectors={missing}",
                        i + 1, emb.len(), batch.len() - expected);
                    let mut it = emb.into_iter();
                    let mut batch_embs = Vec::with_capacity(batch.len());
                    let masked_count = batch.len() - expected;
                    for &is_real in &mask {
                        if is_real {
                            // `None` if the provider returned fewer vectors than texts.
                            batch_embs.push(it.next());
                        } else {
                            batch_embs.push(Some(vec![0.0_f32; embedding_dim]));
                        }
                    }
                    (batch_embs, missing, vec![], masked_count)
                }
                Err(error_msg) => {
                    log::error!("{embedding_provider} embedding batch {} failed: {}", i + 1, crate::logging::preview(&error_msg));

                    // Empty texts were never sent, so they keep their zero vectors.
                    let fail_texts: Vec<String> = batch
                        .iter()
                        .zip(&mask)
                        .filter(|&(_, &is_real)| is_real)
                        .filter_map(|(t, _)| t.clone())
                        .collect();
                    let embs = mask
                        .iter()
                        .map(|&is_real| (!is_real).then(|| vec![0.0_f32; embedding_dim]))
                        .collect();
                    let fail_count = fail_texts.len();
                    (embs, fail_count, fail_texts, batch.len() - fail_count)
                }
            };

            if wait_after_request_s > 0 && i < num_batches - 1 {
                tokio::time::sleep(Duration::from_secs(wait_after_request_s)).await;
            }

            Ok::<BatchResult, LLMError>(result)
        }
    });

    // `try_collect` stops at the first error (such as a rejected API key), so the remaining batches
    // are not sent.
    let results: Vec<_> = stream::iter(futures)
        .buffered(max_concurrent)
        .try_collect()
        .await?;

    let mut all_embeddings: Vec<Option<Vec<f32>>> = Vec::new();
    let mut fail_count = 0;
    let mut total_masked = 0;
    let mut failed_texts: Vec<String> = Vec::new();

    for (batch_embs, batch_fail_count, batch_failed_texts, batch_masked) in results {
        all_embeddings.extend(batch_embs);
        fail_count += batch_fail_count;
        total_masked += batch_masked;
        failed_texts.extend(batch_failed_texts);
    }

    if fail_count > 0 {
        log::error!(
            "{embedding_provider}: {fail_count} texts failed to embed: {}",
            crate::logging::preview(format_args!("{failed_texts:?}"))
        );
    }

    log::info!(
        "Processing finished. Statistics:\n{fail_count} items failed.\n{total_masked} items were empty."
    );

    // Convert to an Arrow FixedSizeListArray, with a null entry for each text that failed.
    let mut builder = FixedSizeListBuilder::with_capacity(
        Float32Builder::with_capacity(all_embeddings.len() * embedding_dim),
        embedding_dim as i32,
        all_embeddings.len(),
    )
    .with_field(Arc::new(Field::new("item", DataType::Float32, true)));
    for embedding in &all_embeddings {
        match embedding {
            Some(embedding) if embedding.len() == embedding_dim => {
                builder.values().append_slice(embedding);
                builder.append(true);
            }
            Some(embedding) => {
                return Err(LLMError::GenericLLMError(format!(
                    "Failed to create FixedSizeListArray: expected {embedding_dim} dimensions, got {}",
                    embedding.len()
                )));
            }
            None => {
                builder.values().append_nulls(embedding_dim);
                builder.append(false);
            }
        }
    }

    let list_array = builder.finish();
    Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
}

/// Public-facing embedding request struct for batch embedding providers. Each
/// [`BatchEmbeddingProvider`] is expected to convert these to their native formats.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchEmbeddingRequest {
    /// Embedding inputs
    pub inputs: Vec<BatchEmbeddingInput>,
    /// The model to use
    pub model: String,
    /// The output dimensions
    pub dims: usize,
}

/// Input to the batch API. This struct corresponds to one of the batch's inputs. Usually, providers
/// expose such functionality by requiring one JSONL line per [`BatchEmbeddingInput`].
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchEmbeddingInput {
    /// A unique ID for this input.
    pub id: String,
    /// The text to embed.
    pub text: String,
}

/// A result of a successful embedding batch submission. The contained ID is used to check on the
/// status of batches.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchSubmission {
    /// The batch's unique ID.
    pub batch_id: String,
    /// The ID of the uploaded file. This is useful to correctly reference batches when actually
    /// submitting a batch request.
    pub file_id: String,
}

/// Results of a completed batch embedding job.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchEmbeddingResults {
    /// The results for successfully embedded inputs
    pub succeeded: Vec<BatchEmbeddingResult>,
    /// Errors for failed inputs
    pub failed: Vec<BatchEmbeddingError>,
}

/// A response part containing the result of a successful embedding, when a batch embedding provider
/// is used.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchEmbeddingResult {
    /// The ID for this input. Corresponds to the `id` in [`BatchEmbeddingInput`].
    pub id: String,
    /// The embedding result
    pub embedding: Vec<f32>,
}

/// Description for a failed batch embedding input.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchEmbeddingError {
    /// The ID for this input. Corresponds to the `id` in [`BatchEmbeddingInput`].
    pub id: String,
    /// The error message
    pub error: String,
}
