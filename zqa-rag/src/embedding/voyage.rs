//! Functions, structs, and trait implementations for interacting with the VoyageAI API. This module
//! includes support for both embedding only.

use crate::{
    capabilities::{BatchAPIProvider, BatchJobState, EmbeddingProvider},
    constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    },
    embedding::common::{EmbeddingApiResponse, Rerank, compute_embeddings_async},
};
use std::{borrow::Cow, env, future::Future, pin::Pin, sync::Arc, time::Instant};

use arrow_schema::{DataType, Field};
use http::HeaderMap;
use lancedb::embeddings::EmbeddingFunction;
use reqwest::multipart::Form;
use serde::{Deserialize, Serialize};
use serde_jsonlines::{json_lines, write_json_lines};

use crate::llm::errors::LLMError;
use crate::llm::http_client::{HttpClient, ReqwestClient};

/// A client for Voyage AI's embedding API.
#[derive(Debug, Clone)]
pub struct VoyageAIClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub client: T,
    /// Optional configuration for the VoyageAI client.
    pub config: Option<crate::config::VoyageAIConfig>,
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
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new VoyageAIClient instance with provided configuration
    #[must_use]
    pub fn with_config(config: crate::config::VoyageAIConfig) -> Self {
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
    pub fn compute_embeddings_internal(
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
                    model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
                    input_type: None, // Directly convert to vector
                    truncation: true,
                    output_dimension: DEFAULT_VOYAGE_EMBEDDING_DIM, // Matryoshka embeddings
                    output_dtype: "float".to_string(),
                },
                EmbeddingProvider::VoyageAI.as_str().to_string(),
                BATCH_SIZE,
                WAIT_AFTER_REQUEST_S,
                DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            ))
        })
    }
}

/// A request to Voyage AI's embedding endpoint.
#[derive(Serialize, Debug, Deserialize)]
struct VoyageAIRequest {
    pub input: Vec<String>,
    pub model: String,
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
    pub detail: String,
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
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            DEFAULT_VOYAGE_EMBEDDING_DIM as i32,
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

#[derive(Serialize, Debug)]
struct VoyageAIRerankRequest {
    query: String,
    documents: Vec<String>,
    model: String,
}

#[derive(Clone, Deserialize)]
struct VoyageAIRerankedDoc {
    index: usize,
}

#[derive(Deserialize)]
struct VoyageAIRerankResponse {
    data: Vec<VoyageAIRerankedDoc>,
}

impl<T: HttpClient, U: AsRef<str> + Send + Clone> Rerank<U> for VoyageAIClient<T> {
    fn rerank<'a>(
        &'a self,
        items: Vec<U>,
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<U>, LLMError>> + Send + 'a>>
    where
        U: 'a,
    {
        Box::pin(async move {
            const RERANK_API_URL: &str = "https://api.voyageai.com/v1/rerank";

            // Use config if available, otherwise fall back to env vars
            let (api_key, reranker_model) = if let Some(ref config) = self.config {
                (config.api_key.clone(), config.reranker.clone())
            } else {
                (
                    env::var("VOYAGE_AI_API_KEY")?,
                    env::var("VOYAGE_AI_RERANK_MODEL")
                        .unwrap_or(DEFAULT_VOYAGE_RERANK_MODEL.into()),
                )
            };

            let documents: Vec<String> =
                items.iter().map(|item| item.as_ref().to_string()).collect();
            let request = VoyageAIRerankRequest {
                model: reranker_model,
                query: query.into(),
                documents,
            };

            let mut headers = HeaderMap::new();
            headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
            headers.insert("Content-Type", "application/json".parse()?);
            headers.insert("Accept", "application/json".parse()?);

            let start_time = Instant::now();
            let response = self
                .client
                .post_json(RERANK_API_URL, headers, &request)
                .await?;

            let body = response.text().await?;
            log::debug!("Voyage AI rerank request took {:.1?}", start_time.elapsed());

            let voyage_response: VoyageAIRerankResponse =
                serde_json::from_str(&body).map_err(|e| {
                    log::warn!("Error deserializing Voyage AI reranker response: {e}");
                    LLMError::DeserializationError(e.to_string())
                })?;

            let voyage_response = voyage_response.data;
            let res = voyage_response
                .iter()
                .filter_map(|result| items.get(result.index))
                .cloned()
                .collect::<Vec<_>>();

            Ok(res)
        })
    }
}

/// The `body` field for a request to the Voyage AI Files API.
#[derive(Serialize)]
pub(crate) struct VoyageAIFilesRequestBody {
    /// A list of input texts that belong to one "group".
    input: Vec<String>,
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
pub struct VoyageAIFilesRequest {
    /// A unique ID assigned to each request
    custom_id: String,
    /// One or more inputs as part of this request.
    body: VoyageAIFilesRequestBody,
}

/// A response from the Voyage AI Files API.
#[derive(Deserialize, Serialize, Clone)]
#[allow(dead_code)]
pub struct VoyageAIFilesResponse {
    /// The file ID, used to refer to input, output, and error files. This starts with "file-".
    id: String,
    /// An ISO 8601 extended format string with time zone offset.
    created_at: String,
    /// An ISO 8601 extended format string with time zone offset.
    expires_at: String,
}

/// The embedding parameters for a batch request.
#[derive(Serialize)]
pub struct VoyageAIBatchRequestParams<'a> {
    /// The model used. Defaults to [`crate::constants::DEFAULT_VOYAGE_EMBEDDING_MODEL`].
    model: &'a str,
    /// The input type. This is set to "document".
    input_type: &'a str,
    /// The embedding dimension. Defaults to [`crate::constants::DEFAULT_VOYAGE_EMBEDDING_DIM`].
    output_dimension: usize,
}

impl Default for VoyageAIBatchRequestParams<'_> {
    fn default() -> Self {
        Self {
            model: DEFAULT_VOYAGE_EMBEDDING_MODEL,
            input_type: "document",
            output_dimension: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
        }
    }
}

/// A response from the Voyage AI Batch API (POST /v1/batches). This is distinct from
/// [`VoyageAIFilesResponse`], which is returned by the Files API.
#[derive(Deserialize, Serialize, Clone)]
#[allow(dead_code)]
pub struct VoyageAIBatchCreateResponse {
    /// The batch ID, used to check status and retrieve results.
    pub(crate) id: String,
}

/// A request to the Voyage AI Batch API. This assumes a call to the Files API has been made,
/// uploading the JSONL file in the right format.
///
/// # Construction example
///
/// ```rust
/// # use zqa_rag::embedding::voyage::{VoyageAIBatchRequest, VoyageAIBatchRequestParams};
///
/// let request = VoyageAIBatchRequest::default()
///     .with_file_id("file-123");
/// ```
#[derive(Serialize)]
pub struct VoyageAIBatchRequest<'a> {
    /// The Voyage AI embedding endpoint. Defaults to "/v1/embeddings".
    endpoint: &'a str,
    /// The completion window. Defaults to "12h".
    completion_window: &'a str,
    /// Parameters to pass to the embedding endpoint.
    request_params: VoyageAIBatchRequestParams<'a>,
    /// The file ID of the uploaded file.
    input_file_id: String,
}

impl VoyageAIBatchRequest<'_> {
    /// Add a file ID to the current batch request.
    #[must_use]
    pub fn with_file_id(mut self, file_id: &str) -> Self {
        self.input_file_id = file_id.into();

        self
    }
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
#[allow(dead_code)]
pub struct VoyageAIBatchStatusResponse {
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
#[allow(dead_code)]
pub struct VoyageAIBatchResultResponse {
    /// The HTTP status code from the embedding endpoint
    status_code: u16,
    /// The embeddings
    body: VoyageAISuccess,
}

/// The response from the Batch API. This is not to be confused as the *raw* output from the Batch
/// API, but the result of calling the *Files API* using the `file_id` obtained from the Batch API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct VoyageAIBatchResult {
    /// The ID of the batch containing this result
    batch_id: String,
    /// The `custom_id` field of this result
    custom_id: String,
    /// Embeddings and metadata
    response: VoyageAIBatchResultResponse,
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
        let body = res.text().await?;
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
    async fn get_file(&self, file_id: &str) -> Result<Vec<VoyageAIBatchResult>, LLMError> {
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
        let body = res.text().await?;

        let tmp_file = tempfile::NamedTempFile::new()?;
        tokio::fs::write(tmp_file.path(), body).await?;

        Ok(json_lines::<VoyageAIBatchResult, _>(tmp_file)?
            .filter_map(std::result::Result::ok)
            .collect::<Vec<_>>())
    }
}

impl<T> BatchAPIProvider for VoyageAIClient<T>
where
    T: HttpClient,
{
    type BatchInput = Vec<VoyageAIFilesRequest>;
    type BatchSubmitResponse = VoyageAIBatchCreateResponse;
    type BatchResults = (
        Option<Vec<VoyageAIBatchResult>>,
        Option<Vec<VoyageAIBatchResult>>,
    );

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
        request: Self::BatchInput,
    ) -> Result<Self::BatchSubmitResponse, LLMError> {
        // Part 1 - use the Files API to upload a JSONL file
        const FILES_API_URL: &str = "https://api.voyageai.com/v1/files";

        let tmp_file = tempfile::NamedTempFile::new()?;
        write_json_lines(&tmp_file, &request)?;

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

        let response = res.text().await?;
        let response: VoyageAIFilesResponse = serde_json::from_str(&response).map_err(|e| {
            log::warn!("Error deserializing Voyage AI Files API response: {e}");
            LLMError::DeserializationError(e.to_string())
        })?;

        // Part 2 - reference the given file ID in our batch embedding request
        const BATCH_API_URL: &str = "https://api.voyageai.com/v1/batches";

        let file_id = &response.id;
        let batch_request = VoyageAIBatchRequest::default().with_file_id(file_id);

        let res = self
            .client
            .post_json(BATCH_API_URL, headers, &batch_request)
            .await?;

        let response = res.text().await?;
        let response: VoyageAIBatchCreateResponse =
            serde_json::from_str(&response).map_err(|e| {
                log::warn!("Error deserializing Voyage AI Batch API response: {e}");
                LLMError::DeserializationError(e.to_string())
            })?;

        Ok(response)
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
    async fn get_batch_results(&self, batch_id: &str) -> Result<Self::BatchResults, LLMError> {
        let response = self.get_batch_status(batch_id).await?;

        match response.status {
            VoyageAIBatchStatus::Completed | VoyageAIBatchStatus::Failed => {
                let results = if let Some(output_file_id) = response.output_file_id {
                    Some(self.get_file(&output_file_id).await?)
                } else {
                    None
                };

                let errors = if let Some(err_file_id) = response.error_file_id {
                    Some(self.get_file(&err_file_id).await?)
                } else {
                    None
                };

                Ok((results, errors))
            }
            _ => Err(LLMError::BatchNotCompleted(batch_id.into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::{BatchAPIProvider, BatchJobState};
    use crate::constants::{DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL};
    use crate::embedding::common::Rerank;
    use crate::llm::http_client::{MockHttpClient, ReqwestClient};
    use arrow_array::Array;
    use dotenv::dotenv;
    use std::sync::Arc;
    use zqa_macros::{test_eq, test_ok};

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

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Voyage AI embedding error: {:?}", embeddings.as_ref().err());
        }

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        test_eq!(vector.len(), 6);
        test_eq!(vector.value_length(), DEFAULT_VOYAGE_EMBEDDING_DIM as i32);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_rerank() {
        dotenv().ok();

        let array = vec![
            "Hello, World!".to_string(),
            "A second string".to_string(),
            "A third string".to_string(),
        ];
        let query = "A string";

        let client = VoyageAIClient::<ReqwestClient>::default();
        let reranked = client.rerank(array.clone(), query).await;

        // Debug the error if there is one
        if reranked.is_err() {
            println!("Voyage AI reranker error: {:?}", reranked.as_ref().err());
        }

        test_ok!(reranked);

        let reranked = reranked.unwrap();
        test_eq!(reranked.len(), array.len());
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

        let request = vec![
            VoyageAIFilesRequest {
                custom_id: "live-req-1".to_string(),
                body: VoyageAIFilesRequestBody {
                    input: vec!["Hello, World!".to_string(), "A second string".to_string()],
                },
            },
            VoyageAIFilesRequest {
                custom_id: "live-req-2".to_string(),
                body: VoyageAIFilesRequestBody {
                    input: vec!["A third string".to_string()],
                },
            },
        ];

        let submit_result = client.submit_batch(request).await;
        test_ok!(submit_result);

        let batch_id = submit_result.unwrap().id;

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
