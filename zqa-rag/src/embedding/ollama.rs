use serde::{Deserialize, Serialize};
use std::{borrow::Cow, env, fmt::Debug};

use arrow_schema::DataType;
use lancedb::embeddings::EmbeddingFunction;
use std::sync::Arc;

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
        const WAIT_AFTER_REQUEST_S: u64 = 1;

        let model = env::var("OLLAMA_EMBEDDING_MODEL")
            .ok()
            .unwrap_or_else(|| DEFAULT_OLLAMA_EMBEDDING_MODEL.to_string());
        let base_url = self
            .config
            .clone()
            .map_or(DEFAULT_OLLAMA_BASE_URL.into(), |c| c.base_url);
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
                DEFAULT_OLLAMA_EMBEDDING_DIM,
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
        Ok(Cow::Owned(DataType::Utf8))
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
    use super::{DEFAULT_OLLAMA_EMBEDDING_DIM, OllamaClient};
    use crate::http_client::ReqwestClient;
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

        let client = OllamaClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Ollama embedding error: {:?}", embeddings.as_ref().err());
        }

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        test_eq!(vector.len(), 6);
        test_eq!(vector.value_length(), DEFAULT_OLLAMA_EMBEDDING_DIM as i32);
    }
}
