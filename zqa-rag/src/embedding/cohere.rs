//! Functions, structs, and trait implementations for interacting with the Cohere API. This module
//! includes support for embedding only.

use super::common::EmbeddingApiResponse;
use crate::{
    capabilities::EmbeddingProvider,
    constants::{DEFAULT_COHERE_EMBEDDING_DIM, DEFAULT_COHERE_EMBEDDING_MODEL},
    embedding::common::compute_embeddings_async,
};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, env, sync::Arc};

use arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;

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

    fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        // For a non-trial API key, Cohere's RPM is 2000 and the Embed model has a 128k context
        // window. It does not appear that there is a TPM limit. In theory, we could therefore send
        // 2000 requests spread over 1 minute (so 30 requests/second), each with one input.
        const BATCH_SIZE: usize = 30;
        const WAIT_AFTER_REQUEST_S: u64 = 1;

        let api_key = self.config.as_ref().map_or_else(
            || env::var("COHERE_API_KEY"),
            |config| Ok(config.api_key.clone()),
        )?;

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
                    model: DEFAULT_COHERE_EMBEDDING_MODEL.to_string(),
                    input_type: "search_document".into(),
                    output_dimension: DEFAULT_COHERE_EMBEDDING_DIM,
                    // Requesting float vectors explicitly for newer APIs; ignored by older
                    embedding_types: Some(vec!["float".into()]),
                },
                EmbeddingProvider::Cohere.as_str().to_string(),
                BATCH_SIZE,
                WAIT_AFTER_REQUEST_S,
                DEFAULT_COHERE_EMBEDDING_DIM as usize,
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
    output_dimension: u32,
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

impl<T: HttpClient + Default + Clone + std::fmt::Debug> EmbeddingFunction for CohereClient<T> {
    fn name(&self) -> &'static str {
        "Cohere"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            DEFAULT_COHERE_EMBEDDING_DIM as i32,
        )))
    }

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

#[cfg(test)]
mod tests {
    use super::{CohereClient, DEFAULT_COHERE_EMBEDDING_DIM};
    use crate::http_client::ReqwestClient;
    use arrow_array::Array;
    use dotenv::dotenv;
    use std::{env, sync::Arc};
    use zqa_macros::{test_eq, test_ok};

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments until we get ollama there
            return;
        }

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
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Cohere embedding error: {:?}", embeddings.as_ref().err());
        }

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        test_eq!(vector.len(), 6);
        test_eq!(vector.value_length(), DEFAULT_COHERE_EMBEDDING_DIM as i32);
    }
}
