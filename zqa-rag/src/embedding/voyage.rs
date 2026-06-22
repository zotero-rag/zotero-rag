//! Functions, structs, and trait implementations for interacting with the VoyageAI API. This module
//! includes support for both embedding only.

use std::{borrow::Cow, env, sync::Arc};

use arrow_schema::{DataType, Field};
use http::HeaderMap;
use lancedb::embeddings::EmbeddingFunction;
use reqwest::multipart::Form;
use serde::{Deserialize, Serialize};
use serde_jsonlines::{json_lines, write_json_lines};

use crate::embedding::common::{
    BatchEmbeddingError, BatchEmbeddingRequest, BatchEmbeddingResult, BatchEmbeddingResults,
    BatchSubmission,
};
use crate::http_client::{HttpClient, ReqwestClient};
use crate::llm::errors::LLMError;
use crate::{
    capabilities::{BatchAPIProvider, BatchJobState, EmbeddingProvider},
    constants::{DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL},
    embedding::common::{EmbeddingApiResponse, compute_embeddings_async},
};

/// A client for Voyage AI's embedding API.
#[derive(Debug, Clone)]
pub struct VoyageAIClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub(crate) client: T,
    /// Optional configuration for the VoyageAI client.
    pub(crate) config: Option<crate::config::VoyageAIConfig>,
}

impl<T: HttpClient + Default + Clone> Default for VoyageAIClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> VoyageAIClient<T>
where
    T: HttpClient + Default + Clone,
{
    /// Creates a new VoyageAIClient instance without configuration
    /// (will fall back to environment variables)
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new VoyageAIClient instance with provided configuration
    #[must_use]
    pub(crate) fn with_config(config: crate::config::VoyageAIConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }

    /// Internal method to compute embeddings that works with LLMError
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If the VOYAGE_AI_API_KEY environment variable is not set
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403 status
    /// * `LLMError::HttpStatusError` - If the API returns other unsuccessful HTTP status codes
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If the API response cannot be parsed
    /// * `LLMError::InvalidHeaderError` - If header values cannot be parsed
    /// * `LLMError::GenericLLMError` - If other HTTP errors occur or Arrow array creation fails
    pub(crate) fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        // Wait for two seconds after each batch to avoid RPM and TPM throttling. At the base
        // tier, Voyage AI has a 3M TPM and a 2K RPM. However, although the context of their
        // models is 32k, we can't just sent 3M / 32k ~= 93 requests at a time and then wait a
        // minute, because there is also a 120k token per request limit. This actually means we
        // can only send floor(120k / 32k) = 3 requests at a time. Now our effective requests
        // per minute is 3M (tokens / min) / 96k (tokens / request) = 31.25 RPM. Rounding down,
        // we can send 30 RPM, so we wait 2s between requests.
        const BATCH_SIZE: usize = 3;
        const WAIT_AFTER_REQUEST_S: u64 = 2;

        let api_key = self.config.as_ref().map_or_else(
            || env::var("VOYAGE_AI_API_KEY"),
            |config| Ok(config.api_key.clone()),
        )?;

        let model = self
            .config
            .as_ref()
            .map_or(DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(), |c| {
                c.embedding_model.clone()
            });
        let output_dimension = self
            .config
            .as_ref()
            .map_or(DEFAULT_VOYAGE_EMBEDDING_DIM, |c| c.embedding_dims as u32);

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(compute_embeddings_async::<
                VoyageAIRequest,
                VoyageAIResponse,
                _,
            >(
                source,
                "https://api.voyageai.com/v1/embeddings",
                &api_key,
                self.client.clone(),
                |texts| VoyageAIRequest {
                    input: texts,
                    model: model.clone(),
                    input_type: None, // Directly convert to vector
                    truncation: true,
                    output_dimension, // Matryoshka embeddings
                    output_dtype: "float".to_string(),
                },
                EmbeddingProvider::VoyageAI.as_str().to_string(),
                BATCH_SIZE,
                WAIT_AFTER_REQUEST_S,
                output_dimension as usize,
            ))
        })
    }
}

/// A request to Voyage AI's embedding endpoint.
#[derive(Serialize, Debug, Deserialize)]
struct VoyageAIRequest {
    input: Vec<String>,
    model: String,
    input_type: Option<String>,
    truncation: bool,
    output_dimension: u32,
    output_dtype: String,
}

/// Embeddings for one text input
#[derive(Serialize, Deserialize, Debug)]
struct VoyageAIEmbedding {
    /// Type of result, usually "embedding"
    object: String,
    /// The embeddings themselves
    embedding: Vec<f32>,
    /// The position of the text. This is more relevant for the Batch API.
    index: u32,
}

/// Token usage stats for a Voyage AI request
#[derive(Serialize, Deserialize, Debug)]
struct VoyageAIUsage {
    /// Tokens consumed
    total_tokens: u32,
}

/// Success response for embedding requests. This is also used to interact with the Batch API.
#[derive(Serialize, Deserialize, Debug)]
struct VoyageAISuccess {
    /// Type of result, usually "list"
    object: String,
    /// The list of embeddings requested
    data: Vec<VoyageAIEmbedding>,
    /// The model used
    model: String,
    /// Token usage stats
    usage: VoyageAIUsage,
}

/// Error response for embedding requests
#[derive(Serialize, Deserialize, Debug)]
struct VoyageAIError {
    /// Error detail. This has the same format as Pydantic validation errors.
    detail: String,
}

/// A response from the Voyage AI API
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
enum VoyageAIResponse {
    Success(VoyageAISuccess),
    Error(VoyageAIError),
}

impl EmbeddingApiResponse for VoyageAIResponse {
    type Success = VoyageAISuccess;
    type Error = VoyageAIError;

    fn is_success(&self) -> bool {
        match self {
            VoyageAIResponse::Success(_) => true,
            VoyageAIResponse::Error(_) => false,
        }
    }

    fn get_embeddings(self) -> Option<Vec<Vec<f32>>> {
        match self {
            VoyageAIResponse::Error(_) => None,
            VoyageAIResponse::Success(res) => {
                Some(res.data.into_iter().map(|v| v.embedding).collect())
            }
        }
    }

    fn get_error_message(self) -> Option<String> {
        match self {
            VoyageAIResponse::Error(err) => Some(err.detail),
            VoyageAIResponse::Success(_) => None,
        }
    }
}

/// Implements the LanceDB EmbeddingFunction trait for VoyageAIClient. Since VoyageAI has the
/// highest token limit for their embedding model (32k instead of OpenAI's 8k), we prefer this
/// instead.
impl<T: HttpClient + Default + Clone + std::fmt::Debug> EmbeddingFunction for VoyageAIClient<T> {
    fn name(&self) -> &'static str {
        "Voyage AI"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        let dim = self
            .config
            .as_ref()
            .map_or(DEFAULT_VOYAGE_EMBEDDING_DIM as i32, |c| {
                c.embedding_dims as i32
            });
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
        )))
    }

    /// The most basic tier of Voyage AI has a TPM of 3M, and an RPM of 2000. Since we have
    /// truncation enabled, we have 3M / 32k ~= 93, so we send 90 requests at a time, before
    /// waiting for one minute.
    fn compute_source_embeddings(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        match self.compute_embeddings_internal(source) {
            Ok(result) => Ok(result),
            Err(e) => Err(lancedb::Error::Other {
                message: e.to_string(),
                source: None,
            }),
        }
    }

    fn compute_query_embeddings(
        &self,
        input: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        match self.compute_embeddings_internal(input) {
            Ok(result) => Ok(result),
            Err(e) => Err(lancedb::Error::Other {
                message: e.to_string(),
                source: None,
            }),
        }
    }
}

/// The `body` field for a request to the Voyage AI Files API.
#[derive(Serialize)]
pub(crate) struct VoyageAIFilesRequestBody {
    /// A list of input texts that belong to one "group".
    pub(crate) input: Vec<String>,
}

/// An input as part of the request to the Voyage AI Files API, used as a precursor to interact
/// with the Batch API. A batch may only have 100K inputs at maximum, and each "request" in a batch
/// is subject to Voyage's 32K-token context window limit and 120K total token limit.
///
/// The documentation seems to use the terms "request" and "input" somewhat interchangeably, and uses
/// "batch" for the full JSONL input. For consistency, in the context of the Batch API, we will use
/// "input" to refer to one single string, and "request" to mean one JSON line, which can have
/// multiple strings (inputs).
///
/// See also: [Documentation](https://docs.voyageai.com/docs/batch-inference#5-retrieve-results)
#[derive(Serialize)]
pub(crate) struct VoyageAIFilesRequest {
    /// A unique ID assigned to each request
    pub(crate) custom_id: String,
    /// One or more inputs as part of this request.
    pub(crate) body: VoyageAIFilesRequestBody,
}

/// A response from the Voyage AI Files API.
#[derive(Deserialize, Serialize, Clone)]
pub(crate) struct VoyageAIFilesResponse {
    /// The file ID, used to refer to input, output, and error files. This starts with "file-".
    pub(crate) id: String,
    /// An ISO 8601 extended format string with time zone offset.
    pub(crate) created_at: String,
    /// An ISO 8601 extended format string with time zone offset.
    pub(crate) expires_at: String,
}

/// The embedding parameters for a batch request.
#[derive(Serialize)]
pub(crate) struct VoyageAIBatchRequestParams<'a> {
    /// The model used. Defaults to [`crate::constants::DEFAULT_VOYAGE_EMBEDDING_MODEL`].
    model: String,
    /// The input type. This is set to "document".
    input_type: &'a str,
    /// The embedding dimension. Defaults to [`crate::constants::DEFAULT_VOYAGE_EMBEDDING_DIM`].
    output_dimension: usize,
}

impl From<BatchEmbeddingRequest> for VoyageAIBatchRequestParams<'_> {
    fn from(value: BatchEmbeddingRequest) -> Self {
        Self {
            model: value.model,
            input_type: "document",
            output_dimension: value.dims,
        }
    }
}

impl Default for VoyageAIBatchRequestParams<'_> {
    fn default() -> Self {
        Self {
            model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
            input_type: "document",
            output_dimension: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
        }
    }
}

/// A response from the Voyage AI Batch API (POST /v1/batches). This is distinct from
/// [`VoyageAIFilesResponse`], which is returned by the Files API.
#[derive(Deserialize, Serialize, Clone)]
pub(crate) struct VoyageAIBatchCreateResponse {
    /// The batch ID, used to check status and retrieve results.
    pub(crate) id: String,
}

/// A batch creation request to the Voyage AI Batch API. This assumes a call to the Files API
/// has been made, uploading the JSONL file in the right format.
#[derive(Serialize)]
pub(crate) struct VoyageAIBatchRequest<'a> {
    /// The Voyage AI embedding endpoint. Defaults to "/v1/embeddings".
    endpoint: &'a str,
    /// The completion window. Defaults to "12h".
    completion_window: &'a str,
    /// Parameters to pass to the embedding endpoint.
    request_params: VoyageAIBatchRequestParams<'a>,
    /// The file ID of the uploaded file.
    input_file_id: String,
}

impl<'a> Default for VoyageAIBatchRequest<'a> {
    fn default() -> Self {
        Self {
            endpoint: "/v1/embeddings",
            completion_window: "12h",
            request_params: VoyageAIBatchRequestParams::<'a>::default(),
            input_file_id: String::new(),
        }
    }
}

/// The status of a batch.
///
/// See also: [Documentation](https://docs.voyageai.com/docs/batch-inference#batch-lifecycle).
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
enum VoyageAIBatchStatus {
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Cancelling,
    Cancelled,
}

impl From<VoyageAIBatchStatus> for BatchJobState {
    fn from(status: VoyageAIBatchStatus) -> Self {
        match status {
            VoyageAIBatchStatus::Validating => BatchJobState::Created,
            VoyageAIBatchStatus::InProgress | VoyageAIBatchStatus::Finalizing => {
                BatchJobState::InProgress
            }
            VoyageAIBatchStatus::Completed => BatchJobState::Completed,
            VoyageAIBatchStatus::Failed => BatchJobState::Failed,
            VoyageAIBatchStatus::Cancelling => BatchJobState::Canceling,
            VoyageAIBatchStatus::Cancelled => BatchJobState::Canceled,
        }
    }
}

/// A response to a request checking the status of a batch.
#[derive(Deserialize, Serialize, Clone)]
pub(crate) struct VoyageAIBatchStatusResponse {
    /// The batch ID.
    id: String,
    /// The file ID used to create this batch.
    input_file_id: String,
    /// The current status
    status: VoyageAIBatchStatus,
    /// ISO 8601 extended format date string with timezone offsets.
    created_at: String,
    /// ISO 8601 extended format date string with timezone offsets.
    in_progress_at: Option<String>,
    /// ISO 8601 extended format date string with timezone offsets.
    completed_at: Option<String>,
    /// ISO 8601 extended format date string with timezone offsets.
    failed_at: Option<String>,
    /// If successfully (even if partially) completed, the file ID containing the successful
    /// results. A separate call to the Files API is necessary to actually obtain the embeddings.
    output_file_id: Option<String>,
    /// If any errors occurred, the file ID containing the failed results. A separate call to the
    /// Files API is necessary to actually obtain the embeddings.
    error_file_id: Option<String>,
}

/// The `response` field of the response from the Batch API.
#[derive(Debug, Deserialize)]
pub(crate) struct VoyageAIBatchResultResponse {
    /// The embeddings.
    body: VoyageAISuccess,
}

/// The response from the Batch API. This is not to be confused as the *raw* output from the Batch
/// API, but the result of calling the *Files API* using the `file_id` obtained from the Batch API.
/// Moreover, this JSONL structure is only contained in the output file, and not errors. For the
/// structure of the lines in the error file, see [`VoyageAIBatchError`].
#[derive(Debug, Deserialize)]
pub(crate) struct VoyageAIBatchResult {
    /// The `custom_id` field of this result
    custom_id: String,
    /// Embeddings and metadata
    response: VoyageAIBatchResultResponse,
}

/// Error details in the JSONL returned by the Files API, when checking the batch status.
///
/// See: [https://docs.voyageai.com/docs/batch-inference#batch-errors]
#[derive(Debug, Deserialize)]
pub(crate) struct VoyageAIBatchErrorDetails {
    /// An error message
    message: String,
}

/// The contents of the file containing error details if some batches fail. This is not the return
/// type of the Batch API itself; it is the response from the Files API.
///
/// See: [https://docs.voyageai.com/docs/batch-inference#batch-errors]
#[derive(Debug, Deserialize)]
pub(crate) struct VoyageAIBatchError {
    custom_id: String,
    error: VoyageAIBatchErrorDetails,
}

impl From<VoyageAIBatchError> for BatchEmbeddingError {
    fn from(value: VoyageAIBatchError) -> Self {
        Self {
            id: value.custom_id,
            error: value.error.message,
        }
    }
}

/// Returns the response body as text, or an [`LLMError::HttpStatusError`] carrying the body if the
/// HTTP status is not a success. The Voyage batch methods below don't use reqwest's
/// `error_for_status`, so we check explicitly here to surface clear failures instead of an opaque
/// deserialization error when the API returns a non-2xx response.
async fn body_or_status_error(res: reqwest::Response, context: &str) -> Result<String, LLMError> {
    if !res.status().is_success() {
        let body = res.text().await.unwrap_or_default();
        return Err(LLMError::HttpStatusError(format!("{context}: {body}")));
    }
    Ok(res.text().await?)
}

impl<T> VoyageAIClient<T>
where
    T: HttpClient,
{
    /// Fetches the current status of a batch job from the Voyage AI Batch API.
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If `VOYAGE_AI_API_KEY` is not set and no config is provided
    /// * `LLMError::InvalidHeaderError` - If the API key cannot be parsed as a header value
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403
    /// * `LLMError::HttpStatusError` - If the API returns another unsuccessful status code
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If the response cannot be parsed
    async fn get_batch_status(
        &self,
        batch_id: &str,
    ) -> Result<VoyageAIBatchStatusResponse, LLMError> {
        let files_api_url = &format!("https://api.voyageai.com/v1/batches/{batch_id}");

        let api_key = if let Some(ref config) = self.config {
            config.api_key.clone()
        } else {
            env::var("VOYAGE_AI_API_KEY")?
        };

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
        headers.insert("Content-Type", "application/json".parse()?);

        let res = self.client.get_json(files_api_url, headers).await?;
        let body = body_or_status_error(res, &format!("Failed to get status for batch {batch_id}"))
            .await?;
        let response: VoyageAIBatchStatusResponse = serde_json::from_str(&body).map_err(|e| {
            log::warn!("Error deserializing Voyage AI batch status response: {e}");
            LLMError::DeserializationError(e.to_string())
        })?;

        Ok(response)
    }

    /// Downloads a file from the Voyage AI Files API and parses it as JSONL batch results.
    ///
    /// The file is written to a temporary path before being parsed, since `json_lines` requires
    /// a file path rather than an in-memory buffer.
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If `VOYAGE_AI_API_KEY` is not set and no config is provided
    /// * `LLMError::InvalidHeaderError` - If the API key cannot be parsed as a header value
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403
    /// * `LLMError::HttpStatusError` - If the API returns another unsuccessful status code
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::GenericLLMError` - If the temporary file cannot be written
    async fn get_file<U>(&self, file_id: &str) -> Result<Vec<U>, LLMError>
    where
        U: for<'a> Deserialize<'a> + Send + 'static,
    {
        let files_api_url = &format!("https://api.voyageai.com/v1/files/{file_id}");

        let api_key = if let Some(ref config) = self.config {
            config.api_key.clone()
        } else {
            env::var("VOYAGE_AI_API_KEY")?
        };

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
        headers.insert("Content-Type", "application/json".parse()?);

        let res = self.client.get_json(files_api_url, headers).await?;
        let body = body_or_status_error(res, &format!("Failed to download file {file_id}")).await?;

        let tmp_file = tempfile::NamedTempFile::new()?;
        tokio::fs::write(tmp_file.path(), body).await?;

        tokio::task::spawn_blocking(move || {
            Ok::<_, LLMError>(
                json_lines::<U, _>(tmp_file)?
                    .filter_map(std::result::Result::ok)
                    .collect::<Vec<_>>(),
            )
        })
        .await
        .map_err(|e| LLMError::GenericLLMError(e.to_string()))?
    }
}

impl<T> BatchAPIProvider for VoyageAIClient<T>
where
    T: HttpClient,
{
    /// Submits a batch embedding job to the Voyage AI Batch API.
    ///
    /// This is a two-step process:
    /// 1. The input requests are serialized as JSONL and uploaded via the Files API.
    /// 2. A batch job is created referencing the uploaded file ID.
    ///
    /// Returns a [`VoyageAIBatchCreateResponse`] containing the batch ID, which can be passed to
    /// [`BatchAPIProvider::get_batch_status`] and [`BatchAPIProvider::get_batch_results`].
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If `VOYAGE_AI_API_KEY` is not set and no config is provided
    /// * `LLMError::InvalidHeaderError` - If the API key cannot be parsed as a header value
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403
    /// * `LLMError::HttpStatusError` - If the API returns another unsuccessful status code
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If either API response cannot be parsed
    /// * `LLMError::GenericLLMError` - If the temporary JSONL file cannot be written
    async fn submit_batch(
        &self,
        request: BatchEmbeddingRequest,
    ) -> Result<BatchSubmission, LLMError> {
        // Part 1 - use the Files API to upload a JSONL file
        const FILES_API_URL: &str = "https://api.voyageai.com/v1/files";

        let inputs = request
            .inputs
            .iter()
            .map(|v| VoyageAIFilesRequest {
                custom_id: v.id.clone(),
                body: VoyageAIFilesRequestBody {
                    input: vec![v.text.clone()],
                },
            })
            .collect::<Vec<_>>();

        let tmp_file = tempfile::NamedTempFile::new()?;
        write_json_lines(&tmp_file, &inputs)?;

        let api_key = if let Some(ref config) = self.config {
            config.api_key.clone()
        } else {
            env::var("VOYAGE_AI_API_KEY")?
        };

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse()?);

        let form_data = Form::new()
            .text("purpose", "batch")
            .file("file", tmp_file)
            .await?;

        let res = self
            .client
            .post_form(FILES_API_URL, headers.clone(), form_data)
            .await?;

        let response = body_or_status_error(res, "Failed to upload batch input file").await?;
        let response: VoyageAIFilesResponse = serde_json::from_str(&response).map_err(|e| {
            log::warn!("Error deserializing Voyage AI Files API response: {e}");
            LLMError::DeserializationError(e.to_string())
        })?;

        // Part 2 - reference the given file ID in our batch embedding request
        const BATCH_API_URL: &str = "https://api.voyageai.com/v1/batches";

        let file_id = &response.id;
        let params = VoyageAIBatchRequestParams::from(request);
        let batch_request = VoyageAIBatchRequest {
            input_file_id: file_id.clone(),
            request_params: params,
            ..Default::default()
        };

        let res = self
            .client
            .post_json(BATCH_API_URL, headers, &batch_request)
            .await?;

        let response = body_or_status_error(res, "Failed to create batch").await?;
        let response: VoyageAIBatchCreateResponse =
            serde_json::from_str(&response).map_err(|e| {
                log::warn!("Error deserializing Voyage AI Batch API response: {e}");
                LLMError::DeserializationError(e.to_string())
            })?;

        Ok(BatchSubmission {
            batch_id: response.id,
            file_id: file_id.clone(),
        })
    }

    /// Returns the current [`BatchJobState`] for the given batch ID.
    ///
    /// # Errors
    ///
    /// See [`VoyageAIClient::get_batch_status`].
    async fn get_batch_status(&self, batch_id: &str) -> Result<BatchJobState, LLMError> {
        let response = self.get_batch_status(batch_id).await?;
        Ok(response.status.into())
    }

    /// Cancel the batch job given its (provider) id.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch id from the provider
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If an API key is not set up
    /// * `LLMError::InvalidHeaderError` - If the API key cannot be parsed as a header value
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403
    /// * `LLMError::HttpStatusError` - If the API returns another unsuccessful status code
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    async fn cancel_batch(&self, batch_id: &str) -> Result<(), LLMError> {
        const BATCH_API_URL: &str = "https://api.voyageai.com/v1/batches";

        let api_key = if let Some(ref config) = self.config {
            config.api_key.clone()
        } else {
            env::var("VOYAGE_AI_API_KEY")?
        };

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse()?);

        let res = self
            .client
            .post_empty(&format!("{BATCH_API_URL}/{batch_id}/cancel"), headers)
            .await?;

        body_or_status_error(res, &format!("Failed to cancel batch {batch_id}")).await?;

        Ok(())
    }

    /// Retrieves the results of a completed or failed batch job.
    ///
    /// Returns a tuple of `(successful_results, error_results)`. Each element is `None` if the
    /// corresponding output file was not produced (e.g. no errors occurred, or the batch produced
    /// no successful outputs).
    ///
    /// # Errors
    ///
    /// * `LLMError::BatchNotCompleted` - If the batch has not yet reached `Completed` or `Failed`
    /// * All errors from [`VoyageAIClient::get_batch_status`] and [`VoyageAIClient::get_file`]
    async fn get_batch_results(&self, batch_id: &str) -> Result<BatchEmbeddingResults, LLMError> {
        let response = self.get_batch_status(batch_id).await?;

        match response.status {
            VoyageAIBatchStatus::Completed | VoyageAIBatchStatus::Failed => {
                let results = if let Some(output_file_id) = response.output_file_id {
                    self.get_file::<VoyageAIBatchResult>(&output_file_id)
                        .await?
                } else {
                    Vec::new()
                };

                let errors = if let Some(err_file_id) = response.error_file_id {
                    self.get_file::<VoyageAIBatchError>(&err_file_id).await?
                } else {
                    Vec::new()
                };

                let success = results
                    .into_iter()
                    .map(|line| {
                        // `line` corresponds to the results for each JSONL input line. The inputs
                        // look like this:
                        //
                        // ```json
                        // {"custom_id": "request_1", "body": {"input": ["Sample text 1", "Sample text 2"]}}
                        // ```
                        BatchEmbeddingResult {
                            id: line.custom_id,
                            // Strictly speaking, this isn't true to the API spec; but callers of
                            // this crate can only access [`super::common::BatchEmbeddingInput`],
                            // which has a 1:1 relationship between each input line and input texts.
                            embedding: line
                                .response
                                .body
                                .data
                                .first()
                                .map(|v| v.embedding.clone())
                                .unwrap_or_default(),
                        }
                    })
                    .collect();

                Ok(BatchEmbeddingResults {
                    succeeded: success,
                    failed: errors.into_iter().map(Into::into).collect(),
                })
            }
            _ => Err(LLMError::BatchNotCompleted(batch_id.into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use dotenv::dotenv;
    use zqa_macros::{test_eq, test_ok};

    use super::*;
    use crate::capabilities::{BatchAPIProvider, BatchJobState};
    use crate::config::VoyageAIConfig;
    use crate::constants::{DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL};
    use crate::embedding::common::BatchEmbeddingInput;
    use crate::http_client::{MockHttpClient, ReqwestClient};

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings() {
        dotenv().ok();

        let array = arrow_array::StringArray::from(vec![
            "Hello, World!",
            "A second string",
            "A third string",
            "A fourth string",
            "A fifth string",
            "A sixth string",
        ]);

        let client = VoyageAIClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        test_eq!(vector.len(), 6);
        test_eq!(vector.value_length(), DEFAULT_VOYAGE_EMBEDDING_DIM as i32);
    }

    #[test]
    fn test_batch_status_validating_maps_to_created() {
        let state: BatchJobState = VoyageAIBatchStatus::Validating.into();
        test_eq!(state, BatchJobState::Created);
    }

    #[test]
    fn test_batch_status_in_progress_maps_to_in_progress() {
        let state: BatchJobState = VoyageAIBatchStatus::InProgress.into();
        test_eq!(state, BatchJobState::InProgress);
    }

    #[test]
    fn test_batch_status_finalizing_maps_to_in_progress() {
        let state: BatchJobState = VoyageAIBatchStatus::Finalizing.into();
        test_eq!(state, BatchJobState::InProgress);
    }

    #[test]
    fn test_batch_status_completed_maps_to_completed() {
        let state: BatchJobState = VoyageAIBatchStatus::Completed.into();
        test_eq!(state, BatchJobState::Completed);
    }

    #[test]
    fn test_batch_status_failed_maps_to_failed() {
        let state: BatchJobState = VoyageAIBatchStatus::Failed.into();
        test_eq!(state, BatchJobState::Failed);
    }

    #[test]
    fn test_batch_status_cancelling_maps_to_canceling() {
        let state: BatchJobState = VoyageAIBatchStatus::Cancelling.into();
        test_eq!(state, BatchJobState::Canceling);
    }

    #[test]
    fn test_batch_status_cancelled_maps_to_canceled() {
        let state: BatchJobState = VoyageAIBatchStatus::Cancelled.into();
        test_eq!(state, BatchJobState::Canceled);
    }

    #[tokio::test]
    async fn test_cancel_batch_success() {
        let config = VoyageAIConfig {
            api_key: "test-key".into(),
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            reranker: "rerank-2.5".into(),
        };
        let client = VoyageAIClient {
            client: MockHttpClient::new(()),
            config: Some(config),
        };

        let result = client.cancel_batch("batch-xyz").await;
        test_ok!(result);
    }

    /// `get_batch_status` deserializes a `VoyageAIBatchStatusResponse` and converts the status
    /// field to a `BatchJobState`.
    #[tokio::test]
    async fn test_get_batch_status_completed() {
        let mock_response = VoyageAIBatchStatusResponse {
            id: "batch-xyz".to_string(),
            input_file_id: "file-abc123".to_string(),
            status: VoyageAIBatchStatus::Completed,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            in_progress_at: None,
            completed_at: Some("2024-01-01T01:00:00Z".to_string()),
            failed_at: None,
            output_file_id: Some("file-out-1".to_string()),
            error_file_id: None,
        };

        let client = VoyageAIClient {
            client: MockHttpClient::new(mock_response),
            config: Some(crate::config::VoyageAIConfig {
                api_key: "test-key".to_string(),
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: "rerank-2".to_string(),
            }),
        };

        let result = BatchAPIProvider::get_batch_status(&client, "batch-xyz").await;
        test_ok!(result);
        test_eq!(result.unwrap(), BatchJobState::Completed);
    }

    #[tokio::test]
    async fn test_get_batch_status_in_progress() {
        let mock_response = VoyageAIBatchStatusResponse {
            id: "batch-xyz".to_string(),
            input_file_id: "file-abc123".to_string(),
            status: VoyageAIBatchStatus::InProgress,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            in_progress_at: Some("2024-01-01T00:05:00Z".to_string()),
            completed_at: None,
            failed_at: None,
            output_file_id: None,
            error_file_id: None,
        };

        let client = VoyageAIClient {
            client: MockHttpClient::new(mock_response),
            config: Some(crate::config::VoyageAIConfig {
                api_key: "test-key".to_string(),
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: "rerank-2".to_string(),
            }),
        };

        let result = BatchAPIProvider::get_batch_status(&client, "batch-xyz").await;
        test_ok!(result);
        test_eq!(result.unwrap(), BatchJobState::InProgress);
    }

    /// When `get_batch_results` is called on a batch that is not yet in a terminal state,
    /// it should return `LLMError::BatchNotCompleted`.
    #[tokio::test]
    async fn test_get_batch_results_not_completed_returns_error() {
        let mock_response = VoyageAIBatchStatusResponse {
            id: "batch-xyz".to_string(),
            input_file_id: "file-abc123".to_string(),
            status: VoyageAIBatchStatus::InProgress,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            in_progress_at: Some("2024-01-01T00:05:00Z".to_string()),
            completed_at: None,
            failed_at: None,
            output_file_id: None,
            error_file_id: None,
        };

        let client = VoyageAIClient {
            client: MockHttpClient::new(mock_response),
            config: Some(crate::config::VoyageAIConfig {
                api_key: "test-key".to_string(),
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: "rerank-2".to_string(),
            }),
        };

        let result = BatchAPIProvider::get_batch_results(&client, "batch-xyz").await;
        assert!(result.is_err(), "Expected BatchNotCompleted error, got Ok");

        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::llm::errors::LLMError::BatchNotCompleted(_)),
            "Expected BatchNotCompleted, got {err:?}"
        );
    }

    /// Submits a small batch job and immediately checks its status. The job should be in a
    /// non-completed state right after submission (Created or InProgress).
    #[tokio::test]
    async fn test_live_submit_batch_and_check_status() {
        dotenv().ok();

        let client = VoyageAIClient::<ReqwestClient>::default();

        let request = BatchEmbeddingRequest {
            inputs: vec![
                BatchEmbeddingInput {
                    id: "live-req-1".into(),
                    text: "Hello, World!".into(),
                },
                BatchEmbeddingInput {
                    id: "live-req-2".into(),
                    text: "A second string".into(),
                },
            ],
            model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
        };

        let submit_result = client.submit_batch(request).await;
        test_ok!(submit_result);

        let batch_id = submit_result.unwrap().batch_id;

        let status_result = BatchAPIProvider::get_batch_status(&client, &batch_id).await;
        test_ok!(status_result);

        let state = status_result.unwrap();
        assert!(
            state == BatchJobState::Created || state == BatchJobState::InProgress,
            "Expected Created or InProgress immediately after submission, got {state:?}"
        );
    }

    /// Same check for `Validating` status (also non-terminal).
    #[tokio::test]
    async fn test_get_batch_results_validating_returns_error() {
        let mock_response = VoyageAIBatchStatusResponse {
            id: "batch-validating".to_string(),
            input_file_id: "file-abc123".to_string(),
            status: VoyageAIBatchStatus::Validating,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            in_progress_at: None,
            completed_at: None,
            failed_at: None,
            output_file_id: None,
            error_file_id: None,
        };

        let client = VoyageAIClient {
            client: MockHttpClient::new(mock_response),
            config: Some(crate::config::VoyageAIConfig {
                api_key: "test-key".to_string(),
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: "rerank-2".to_string(),
            }),
        };

        let result = BatchAPIProvider::get_batch_results(&client, "batch-validating").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::llm::errors::LLMError::BatchNotCompleted(_)
        ));
    }
}
