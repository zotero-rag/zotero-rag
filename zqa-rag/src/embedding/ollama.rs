use std::sync::Arc;
use std::{borrow::Cow, fmt::Debug};

use arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};

use crate::{
    capabilities::EmbeddingProvider,
    clients::ollama::OllamaClient,
    constants::{
        DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_EMBEDDING_DIM, DEFAULT_OLLAMA_EMBEDDING_MODEL,
    },
    embedding::common::{EmbeddingApiResponse, compute_embeddings_async},
    http_client::HttpClient,
    llm::errors::LLMError,
};

#[derive(Debug, Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingSuccessResponse {
    /// Embeddings result
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingErrorResponse {
    error: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OllamaEmbeddingResponse {
    Success(OllamaEmbeddingSuccessResponse),
    Error(OllamaEmbeddingErrorResponse),
}

impl EmbeddingApiResponse for OllamaEmbeddingResponse {
    type Success = OllamaEmbeddingSuccessResponse;
    type Error = OllamaEmbeddingErrorResponse;

    fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    fn get_embeddings(self) -> Option<Vec<Vec<f32>>> {
        match self {
            Self::Error(_) => None,
            Self::Success(res) => Some(res.embeddings),
        }
    }

    fn get_error_message(self) -> Option<String> {
        match self {
            Self::Success(_) => None,
            Self::Error(e) => Some(e.error),
        }
    }
}

impl<T: HttpClient + Debug + Default + Clone> OllamaClient<T> {
    fn get_embedding_dims(&self) -> Option<usize> {
        self.config.as_ref().map(|c| c.embedding_dims)
    }

    /// Internal method to compute embeddings
    ///
    /// # Errors
    ///
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::HttpStatusError` - If the API returns other unsuccessful HTTP status codes
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If the API response cannot be parsed
    /// * `LLMError::GenericLLMError` - If other HTTP errors occur or Arrow array creation fails
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        const BATCH_SIZE: usize = 32;
        const WAIT_AFTER_REQUEST_S: u64 = 0;

        let model = self
            .config
            .as_ref()
            .map_or(DEFAULT_OLLAMA_EMBEDDING_MODEL.into(), |c| {
                c.embedding_model.clone()
            });
        let base_url = self
            .config
            .as_ref()
            .map_or(DEFAULT_OLLAMA_BASE_URL.into(), |c| c.base_url.clone());
        let base_url = base_url.trim_end_matches('/');

        let url = format!("{base_url}/api/embed");

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(compute_embeddings_async::<
                OllamaEmbeddingRequest,
                OllamaEmbeddingResponse,
                _,
            >(
                source,
                &url,
                "",
                self.client.clone(),
                |input| OllamaEmbeddingRequest {
                    model: model.clone(),
                    input,
                },
                EmbeddingProvider::Ollama.as_str().to_string(),
                BATCH_SIZE,
                WAIT_AFTER_REQUEST_S,
                self.get_embedding_dims()
                    .unwrap_or(DEFAULT_OLLAMA_EMBEDDING_DIM),
            ))
        })
    }
}

impl<T> EmbeddingFunction for OllamaClient<T>
where
    T: HttpClient + Default + Debug + Clone,
{
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn source_type(&self) -> lancedb::Result<std::borrow::Cow<'_, arrow_schema::DataType>> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> lancedb::Result<Cow<'_, DataType>> {
        let embedding_dims = self
            .get_embedding_dims()
            .unwrap_or(DEFAULT_OLLAMA_EMBEDDING_DIM) as i32;

        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            embedding_dims,
        )))
    }

    fn compute_source_embeddings(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> lancedb::Result<Arc<dyn arrow_array::Array>> {
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
    ) -> lancedb::Result<Arc<dyn arrow_array::Array>> {
        match self.compute_embeddings_internal(input) {
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
    use lancedb::embeddings::EmbeddingFunction;
    use zqa_macros::{test_eq, test_ok};

    use super::{
        OllamaClient, OllamaEmbeddingErrorResponse, OllamaEmbeddingResponse,
        OllamaEmbeddingSuccessResponse,
    };
    use crate::{
        config::OllamaConfig,
        constants::DEFAULT_OLLAMA_EMBEDDING_DIM,
        embedding::common::EmbeddingApiResponse,
        http_client::{MockHttpClient, ReqwestClient},
    };

    #[test]
    fn test_success_response_deserializes() {
        let json = r#"{"embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}"#;
        let resp: OllamaEmbeddingResponse = serde_json::from_str(json).unwrap();
        assert!(resp.is_success());
        let embs = resp.get_embeddings().unwrap();
        test_eq!(embs.len(), 2);
        test_eq!(embs[0], vec![1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn test_error_response_deserializes() {
        let json = r#"{"error": "model not found"}"#;
        let resp: OllamaEmbeddingResponse = serde_json::from_str(json).unwrap();
        assert!(!resp.is_success());
        test_eq!(resp.get_error_message().unwrap(), "model not found");
    }

    // -- EmbeddingApiResponse impl --

    #[test]
    fn test_success_variant_get_embeddings_returns_some() {
        let resp = OllamaEmbeddingResponse::Success(OllamaEmbeddingSuccessResponse {
            embeddings: vec![vec![0.1, 0.2]],
        });
        assert!(resp.get_embeddings().is_some());
    }

    #[test]
    fn test_error_variant_get_embeddings_returns_none() {
        let resp = OllamaEmbeddingResponse::Error(OllamaEmbeddingErrorResponse {
            error: "oops".into(),
        });
        assert!(resp.get_embeddings().is_none());
    }

    #[test]
    fn test_success_variant_get_error_message_returns_none() {
        let resp =
            OllamaEmbeddingResponse::Success(OllamaEmbeddingSuccessResponse { embeddings: vec![] });
        assert!(resp.get_error_message().is_none());
    }

    #[test]
    fn test_error_variant_get_error_message_returns_some() {
        let resp = OllamaEmbeddingResponse::Error(OllamaEmbeddingErrorResponse {
            error: "bad model".into(),
        });
        test_eq!(resp.get_error_message().unwrap(), "bad model");
    }

    #[test]
    fn test_dest_type_uses_default_dims_without_config() {
        use arrow_schema::DataType;
        let client = OllamaClient::<ReqwestClient>::default();
        let dt = client.dest_type().unwrap();
        match dt.as_ref() {
            DataType::FixedSizeList(_, size) => {
                test_eq!(*size, DEFAULT_OLLAMA_EMBEDDING_DIM as i32);
            }
            other => panic!("Expected FixedSizeList, got {other:?}"),
        }
    }

    #[test]
    fn test_dest_type_uses_config_dims() {
        use arrow_schema::DataType;
        let config = OllamaConfig {
            embedding_dims: 768,
            ..OllamaConfig::default()
        };
        let client = OllamaClient::<ReqwestClient>::with_config(config);
        let dt = client.dest_type().unwrap();
        match dt.as_ref() {
            DataType::FixedSizeList(_, size) => {
                test_eq!(*size, 768_i32);
            }
            other => panic!("Expected FixedSizeList, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings_with_mock_success_response() {
        let dim = 4_usize;
        let mock_response = serde_json::json!({
            "embeddings": [[0.1_f32, 0.2, 0.3, 0.4], [0.5_f32, 0.6, 0.7, 0.8]]
        });

        let client = OllamaClient {
            client: MockHttpClient::new(mock_response),
            config: Some(OllamaConfig {
                embedding_dims: dim,
                ..OllamaConfig::default()
            }),
        };

        let array = arrow_array::StringArray::from(vec!["hello", "world"]);
        let result = client.compute_embeddings_internal(Arc::new(array));

        test_ok!(result);

        let embeddings = result.unwrap();
        let list_array = arrow_array::cast::as_fixed_size_list_array(&embeddings);
        test_eq!(list_array.len(), 2);
        test_eq!(list_array.value_length(), dim as i32);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings_error_response_yields_zero_vectors() {
        let dim = 4_usize;
        let mock_response = serde_json::json!({"error": "model not found"});

        let client = OllamaClient {
            client: MockHttpClient::new(mock_response),
            config: Some(OllamaConfig {
                embedding_dims: dim,
                ..OllamaConfig::default()
            }),
        };

        let array = arrow_array::StringArray::from(vec!["hello"]);
        let result = client.compute_embeddings_internal(Arc::new(array));

        // Error responses produce zero vectors
        test_ok!(result);

        let embeddings = result.unwrap();
        let list_array = arrow_array::cast::as_fixed_size_list_array(&embeddings);
        test_eq!(list_array.len(), 1);
        test_eq!(list_array.value_length(), dim as i32);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings_config_dispatch_uses_custom_base_url_and_model() {
        // The mock doesn't care about the URL, but we verify that config values are
        // accepted and that the Arrow output has the configured dimension.
        let dim = 8_usize;
        let embeddings_data: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..dim).map(|j| i as f32 * 0.1 + j as f32 * 0.01).collect())
            .collect();

        let mock_response = serde_json::json!({"embeddings": embeddings_data});

        let client = OllamaClient {
            client: MockHttpClient::new(mock_response),
            config: Some(OllamaConfig {
                base_url: "http://custom-ollama:11434".into(),
                embedding_model: "custom-model".into(),
                embedding_dims: dim,
                ..OllamaConfig::default()
            }),
        };

        let array = arrow_array::StringArray::from(vec!["a", "b", "c"]);
        let result = client.compute_embeddings_internal(Arc::new(array));

        test_ok!(result);

        let embeddings = result.unwrap();
        let list_array = arrow_array::cast::as_fixed_size_list_array(&embeddings);
        test_eq!(list_array.len(), 3);
        test_eq!(list_array.value_length(), dim as i32);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings_live() {
        if std::env::var("CI").is_ok() {
            return;
        }

        dotenv::dotenv().ok();

        let array = arrow_array::StringArray::from(vec![
            "Hello, World!",
            "A second string",
            "A third string",
            "A fourth string",
            "A fifth string",
            "A sixth string",
        ]);

        let client = OllamaClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);
        test_eq!(vector.len(), 6);
        test_eq!(vector.value_length(), DEFAULT_OLLAMA_EMBEDDING_DIM as i32);
    }
}
