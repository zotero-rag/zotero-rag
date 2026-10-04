//! Functions, structs, and trait implementations for interacting with the Cohere API. This module
//! includes support for embedding only.

use std::borrow::Cow;
use std::env;
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};

use super::common::EmbeddingApiResponse;
use crate::capabilities::EmbeddingProvider;
use crate::constants::{
    DEFAULT_COHERE_EMBEDDING_DIM, DEFAULT_COHERE_EMBEDDING_MODEL, DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_RETRIES,
};
use crate::embedding::common::compute_embeddings_async;
use crate::http_client::{HttpClient, ReqwestClient};
use crate::llm::errors::LLMError;

/// A client for Cohere's embeddings API.
#[derive(Debug, Clone)]
pub(crate) struct CohereClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub(crate) client: T,
    /// Optional configuration for the Cohere client.
    pub(crate) config: Option<crate::config::CohereConfig>,
}

impl<T: HttpClient + Default + Clone> Default for CohereClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CohereClient<T>
where
    T: HttpClient + Default + Clone,
{
    /// Creates a new CohereClient instance without configuration
    /// (will fall back to environment variables)
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new CohereClient instance with provided configuration
    #[must_use]
    pub(crate) fn with_config(config: crate::config::CohereConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }
}

impl<T: HttpClient + Clone> CohereClient<T> {
    fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
        input_type: &'static str,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        // For a non-trial API key, Cohere's RPM is 2000 and the Embed model has a 128k context
        // window. It does not appear that there is a TPM limit. In theory, we could therefore send
        // 2000 requests spread over 1 minute (so 30 requests/second), each with one input.
        const BATCH_SIZE: usize = 30;
        const WAIT_AFTER_REQUEST_S: u64 = 0;

        let api_key = self.config.as_ref().map_or_else(
            || env::var("COHERE_API_KEY"),
            |config| Ok(config.api_key.clone()),
        )?;
        let model = self
            .config
            .as_ref()
            .map_or(DEFAULT_COHERE_EMBEDDING_MODEL, |c| {
                c.embedding_model.as_str()
            });
        let embedding_dims = self
            .config
            .as_ref()
            .map_or(DEFAULT_COHERE_EMBEDDING_DIM, |c| c.embedding_dims as u32);

        // Embed v3 models have fixed widths and do not support output_dimension.
        let fixed_dims = match model {
            "embed-english-v3.0" | "embed-multilingual-v3.0" => Some(1024),
            "embed-english-light-v3.0" | "embed-multilingual-light-v3.0" => Some(384),
            _ => None,
        };
        let output_dimension = match fixed_dims {
            Some(expected_dims) => {
                if embedding_dims != expected_dims {
                    return Err(LLMError::GenericLLMError(format!(
                        "Cohere model {model} requires {expected_dims} embedding dimensions, got {embedding_dims}"
                    )));
                }
                None
            }
            None => Some(embedding_dims),
        };
        let max_concurrent = self
            .config
            .as_ref()
            .map_or(DEFAULT_MAX_CONCURRENT_REQUESTS, |c| {
                c.max_concurrent_requests
            });
        let max_retries = self
            .config
            .as_ref()
            .map_or(DEFAULT_MAX_RETRIES, |c| c.max_retries);

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(compute_embeddings_async::<
                CohereEmbedRequest,
                CohereAIResponse,
                _,
            >(
                source,
                "https://api.cohere.com/v2/embed",
                &api_key,
                self.client.clone(),
                |texts| CohereEmbedRequest {
                    texts,
                    model: model.to_string(),
                    input_type: input_type.into(),
                    output_dimension,
                    // Requesting float vectors explicitly for newer APIs; ignored by older
                    embedding_types: Some(vec!["float".into()]),
                },
                EmbeddingProvider::Cohere.as_str().to_string(),
                BATCH_SIZE,
                WAIT_AFTER_REQUEST_S,
                embedding_dims as usize,
                max_concurrent,
                max_retries,
            ))
        })
    }
}

/// A request to the Cohere embeddings API.
#[derive(Serialize, Debug)]
struct CohereEmbedRequest {
    texts: Vec<String>,
    model: String,
    input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_types: Option<Vec<String>>,
}

/// The embeddings returned by the Cohere API.
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct CohereAIEmbeddings {
    /// The embeddings, returned as a vector of floats for each text.
    float: Vec<Vec<f32>>,
}

/// Represents a successful response from the Cohere embeddings API.
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct CohereAISuccess {
    embeddings: CohereAIEmbeddings,
}

/// Represents an error response from the Cohere embeddings API.
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct CohereAIError {
    message: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
enum CohereAIResponse {
    Success(CohereAISuccess),
    Error(CohereAIError),
}

impl EmbeddingApiResponse for CohereAIResponse {
    type Success = CohereAISuccess;
    type Error = CohereAIError;

    fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    fn get_embeddings(self) -> Option<Vec<Vec<f32>>> {
        match self {
            CohereAIResponse::Error(_) => None,
            CohereAIResponse::Success(res) => Some(res.embeddings.float),
        }
    }

    fn get_error_message(self) -> Option<String> {
        match self {
            CohereAIResponse::Error(err) => Some(err.message),
            CohereAIResponse::Success(_) => None,
        }
    }
}

impl<T: HttpClient + Clone + std::fmt::Debug> EmbeddingFunction for CohereClient<T> {
    fn name(&self) -> &'static str {
        "Cohere"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        let dim = self
            .config
            .as_ref()
            .map_or(DEFAULT_COHERE_EMBEDDING_DIM as i32, |c| {
                c.embedding_dims as i32
            });

        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
        )))
    }

    fn compute_source_embeddings(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        match self.compute_embeddings_internal(source, "search_document") {
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
        match self.compute_embeddings_internal(input, "search_query") {
            Ok(result) => Ok(result),
            Err(e) => Err(lancedb::Error::Other {
                message: e.to_string(),
                source: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;
    use serde_json::json;
    use zqa_macros::{test_eq, test_ok};

    use super::{CohereClient, DEFAULT_COHERE_EMBEDDING_DIM};
    use crate::config::CohereConfig;
    use crate::http_client::{
        ConcurrencyTrackingMockHttpClient, RecordingSequentialMockHttpClient, ReqwestClient,
    };
    use crate::llm::errors::LLMError;

    /// Build an `embed-v4.0` config with 256 dimensions and the given request limits.
    fn embed_v4_config(max_concurrent_requests: usize, max_retries: usize) -> CohereConfig {
        CohereConfig {
            api_key: "test-key".into(),
            embedding_model: "embed-v4.0".into(),
            embedding_dims: 256,
            reranker: String::new(),
            max_concurrent_requests,
            max_retries,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_failed_batch_keeps_successful_batches() {
        // A failed batch is zero-filled without discarding or reordering the batches that succeeded,
        // whether the provider returns an error status or a body that is not JSON.
        let batch_response = json!({"embeddings": {"float": vec![vec![0.5_f32; 256]; 30]}});
        let http_client = RecordingSequentialMockHttpClient::from_status_bodies([
            (200, batch_response.to_string()),
            (502, String::from("<html>502 Bad Gateway</html>")),
            (200, String::from("<html>not JSON</html>")),
            (
                200,
                json!({"embeddings": {"float": [vec![0.25_f32; 256]]}}).to_string(),
            ),
        ]);
        let client = CohereClient {
            client: http_client.clone(),
            config: Some(embed_v4_config(1, 0)),
        };
        let input = Arc::new(arrow_array::StringArray::from(vec!["configured input"; 91]));
        let embeddings = client.compute_source_embeddings(input).unwrap();
        let vectors = arrow_array::cast::as_fixed_size_list_array(&embeddings);
        let values = vectors
            .values()
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .unwrap()
            .values();
        test_eq!(vectors.len(), 91);
        test_eq!(http_client.requests().len(), 4);
        let (first, rest) = values.split_at(30 * 256);
        let (failed, last) = rest.split_at(60 * 256);
        assert!(first.iter().all(|v| (v - 0.5).abs() < f32::EPSILON));
        assert!(failed.iter().all(|v| v.abs() < f32::EPSILON));
        assert!(last.iter().all(|v| (v - 0.25).abs() < f32::EPSILON));

        // A rejected API key fails every batch, so it stops the run, without sending the remaining
        // batches, instead of zero-filling.
        let http_client = RecordingSequentialMockHttpClient::from_status_bodies([
            (200, batch_response.to_string()),
            (401, String::from("invalid api token")),
            (200, batch_response.to_string()),
        ]);
        let client = CohereClient {
            client: http_client.clone(),
            config: Some(embed_v4_config(1, 0)),
        };
        let input = Arc::new(arrow_array::StringArray::from(vec!["configured input"; 90]));
        let result = client.compute_embeddings_internal(input, "search_document");
        assert!(matches!(result, Err(LLMError::CredentialError(_))));
        test_eq!(http_client.requests().len(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_configured_embeddings() {
        for (embedding_model, embedding_dims, output_dimension) in [
            ("embed-v4.0", 256, Some(json!(256))),
            ("embed-english-v3.0", 1024, None),
            ("embed-english-light-v3.0", 384, None),
        ] {
            let config = CohereConfig {
                api_key: "test-key".into(),
                embedding_model: embedding_model.into(),
                embedding_dims,
                reranker: String::new(),
                max_concurrent_requests: crate::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
                max_retries: crate::constants::DEFAULT_MAX_RETRIES,
            };
            let response = json!({"embeddings": {"float": [vec![0.5; config.embedding_dims]]}});
            let http_client = RecordingSequentialMockHttpClient::new([response.clone(), response]);
            let client = CohereClient {
                client: http_client.clone(),
                config: Some(config.clone()),
            };

            for input_type in ["search_document", "search_query"] {
                let input = Arc::new(arrow_array::StringArray::from(vec!["configured input"]));

                let embeddings = if input_type == "search_document" {
                    client.compute_source_embeddings(input)
                } else {
                    client.compute_query_embeddings(input)
                }
                .unwrap();

                let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);
                let dest_type = client.dest_type().unwrap();

                test_eq!(vector.len(), 1);
                test_eq!(vector.value_length(), config.embedding_dims as i32);
                test_eq!(embeddings.data_type(), dest_type.as_ref());
            }

            let requests = http_client.requests();
            test_eq!(requests.len(), 2);

            for (request, input_type) in requests.iter().zip(["search_document", "search_query"]) {
                test_eq!(request["model"], config.embedding_model);
                test_eq!(request.get("output_dimension"), output_dimension.as_ref());
                test_eq!(request["input_type"], input_type);
            }
        }

        // Batches are requested concurrently, up to the configured limit.
        let batch_response = json!({"embeddings": {"float": vec![vec![0.5_f32; 256]; 30]}});
        let http_client =
            ConcurrencyTrackingMockHttpClient::new(std::iter::repeat_n(batch_response, 4));
        let client = CohereClient {
            client: http_client.clone(),
            config: Some(embed_v4_config(2, crate::constants::DEFAULT_MAX_RETRIES)),
        };
        let input = Arc::new(arrow_array::StringArray::from(vec![
            "configured input";
            120
        ]));
        let embeddings = client.compute_source_embeddings(input).unwrap();
        test_eq!(embeddings.len(), 120);
        test_eq!(http_client.peak_in_flight(), 2);

        let http_client = RecordingSequentialMockHttpClient::new::<serde_json::Value>([]);
        let client = CohereClient {
            client: http_client.clone(),
            config: Some(CohereConfig {
                api_key: "test-key".into(),
                embedding_model: "embed-english-v3.0".into(),
                embedding_dims: 256,
                reranker: String::new(),
                max_concurrent_requests: crate::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
                max_retries: crate::constants::DEFAULT_MAX_RETRIES,
            }),
        };
        let input = Arc::new(arrow_array::StringArray::from(vec!["configured input"]));
        let error = client.compute_source_embeddings(input).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires 1024 embedding dimensions, got 256")
        );
        test_eq!(http_client.requests(), Vec::<serde_json::Value>::new());
    }

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

        let client = CohereClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_internal(Arc::new(array), "search_document");

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        test_eq!(vector.len(), 6);
        test_eq!(vector.value_length(), DEFAULT_COHERE_EMBEDDING_DIM as i32);
    }
}
