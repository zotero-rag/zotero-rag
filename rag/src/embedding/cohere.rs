//! Functions, structs, and trait implementations for interacting with the Cohere API. This module
//! includes support for embedding only.

use super::common::{EmbeddingApiRequestTexts, EmbeddingApiResponse};
use crate::{
    capabilities::EmbeddingProviders,
    constants::{COHERE_EMBEDDING_DIM, COHERE_EMBEDDING_MODEL, DEFAULT_COHERE_RERANK_MODEL},
    embedding::common::{Rerank, compute_embeddings_async},
};
use http::HeaderMap;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, env, future::Future, pin::Pin, sync::Arc, time::Instant};

use arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;

use crate::llm::errors::LLMError;
use crate::llm::http_client::{HttpClient, ReqwestClient};

/// A client for Cohere's embeddings API.
#[derive(Debug, Clone)]
pub struct CohereClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub client: T,
    /// Optional configuration for the Cohere client.
    pub config: Option<crate::config::CohereConfig>,
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
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new CohereClient instance with provided configuration
    pub fn with_config(config: crate::config::CohereConfig) -> Self {
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

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(compute_embeddings_async::<
                CohereEmbedRequest,
                CohereAIResponse,
            >(
                source,
                "https://api.cohere.com/v2/embed",
                "COHERE_API_KEY",
                self.client.clone(),
                EmbeddingProviders::Cohere.as_str().to_string(),
                BATCH_SIZE,
                WAIT_AFTER_REQUEST_S,
                COHERE_EMBEDDING_DIM as usize,
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
    // Requesting float vectors explicitly for newer APIs; ignored by older
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_types: Option<Vec<String>>,
}

impl EmbeddingApiRequestTexts<CohereEmbedRequest> for CohereEmbedRequest {
    fn from_texts(texts: Vec<String>) -> Self {
        Self {
            texts,
            model: COHERE_EMBEDDING_MODEL.to_string(),
            input_type: "search_document".into(),
            output_dimension: COHERE_EMBEDDING_DIM,
            embedding_types: Some(vec!["float".into()]),
        }
    }
}

/// The embeddings returned by the Cohere API.
#[derive(Serialize, Deserialize)]
pub struct CohereAIEmbeddings {
    /// The embeddings, returned as a vector of floats for each text.
    float: Vec<Vec<f32>>,
}

/// Represents a successful response from the Cohere embeddings API.
#[derive(Serialize, Deserialize)]
pub struct CohereAISuccess {
    embeddings: CohereAIEmbeddings,
}

/// Represents an error response from the Cohere embeddings API.
#[derive(Serialize, Deserialize)]
pub struct CohereAIError {
    message: String,
}

#[derive(Serialize, Deserialize)]
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
    fn name(&self) -> &str {
        "Cohere"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            COHERE_EMBEDDING_DIM as i32,
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

#[derive(Serialize)]
struct CohereRerankRequest {
    model: String,
    query: String,
    top_n: Option<usize>,
    documents: Vec<String>,
}

#[derive(Clone, Deserialize)]
struct CohereRerankedDocument {
    index: usize,
}

#[derive(Deserialize)]
struct CohereRerankResponse {
    results: Vec<CohereRerankedDocument>,
}

impl<T: HttpClient, U: AsRef<str> + Send + Clone> Rerank<U> for CohereClient<T> {
    fn rerank<'a>(
        &'a self,
        items: Vec<U>,
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<U>, LLMError>> + Send + 'a>>
    where
        U: 'a,
    {
        Box::pin(async move {
            const RERANK_API_URL: &str = "https://api.cohere.com/v2/rerank";

            // Use config if available, otherwise fall back to env vars
            let (api_key, reranker_model) = if let Some(ref config) = self.config {
                (config.api_key.clone(), config.reranker.clone())
            } else {
                (
                    env::var("COHERE_API_KEY")?,
                    env::var("COHERE_RERANKER").unwrap_or(DEFAULT_COHERE_RERANK_MODEL.into()),
                )
            };

            let documents: Vec<String> =
                items.iter().map(|item| item.as_ref().to_string()).collect();
            let request = CohereRerankRequest {
                model: reranker_model,
                query: query.into(),
                top_n: None,
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
            log::debug!("Cohere rerank request took {:.1?}", start_time.elapsed());

            let cohere_response: CohereRerankResponse =
                serde_json::from_str(&body).map_err(|e| {
                    log::warn!("Error deserializing Cohere reranker response: {e}");
                    LLMError::DeserializationError(e.to_string())
                })?;

            let cohere_response = cohere_response.results;
            let res = cohere_response
                .iter()
                .filter_map(|result| items.get(result.index))
                .cloned()
                .collect::<Vec<_>>();

            Ok(res)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{COHERE_EMBEDDING_DIM, CohereClient};
    use crate::{embedding::common::Rerank, llm::http_client::ReqwestClient};
    use arrow_array::Array;
    use dotenv::dotenv;
    use std::sync::Arc;

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
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Cohere embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), COHERE_EMBEDDING_DIM as i32);
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

        let client = CohereClient::<ReqwestClient>::default();
        let reranked = client.rerank(array.clone(), query).await;

        // Debug the error if there is one
        if reranked.is_err() {
            println!("Cohere reranker error: {:?}", reranked.as_ref().err());
        }

        assert!(reranked.is_ok());

        let reranked = reranked.unwrap();
        assert_eq!(reranked.len(), array.len());
    }
}
