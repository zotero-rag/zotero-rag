use crate::{
    capabilities::EmbeddingProviders,
    constants::{DEFAULT_VOYAGE_RERANK_MODEL, VOYAGE_EMBEDDING_DIM, VOYAGE_EMBEDDING_MODEL},
    embedding::common::{
        EmbeddingApiRequestTexts, EmbeddingApiResponse, Rerank, compute_embeddings_async,
    },
};
use std::{borrow::Cow, env, future::Future, pin::Pin, sync::Arc, time::Instant};

use arrow_schema::{DataType, Field};
use http::HeaderMap;
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};

use crate::llm::errors::LLMError;
use crate::llm::http_client::{HttpClient, ReqwestClient};

/// A client for Voyage AI's embedding API.
#[derive(Debug, Clone)]
pub struct VoyageAIClient<T: HttpClient = ReqwestClient> {
    pub client: T,
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
    /// Creates a new VoyageAIClient instance
    pub fn new() -> Self {
        Self {
            client: T::default(),
        }
    }

    /// Internal method to compute embeddings that works with LLMError
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

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(compute_embeddings_async::<
                VoyageAIRequest,
                VoyageAIResponse,
            >(
                source,
                "https://api.voyageai.com/v1/embeddings",
                "VOYAGE_AI_API_KEY",
                self.client.clone(),
                EmbeddingProviders::VoyageAI.as_str().to_string(),
                BATCH_SIZE,
                WAIT_AFTER_REQUEST_S,
                VOYAGE_EMBEDDING_DIM as usize,
            ))
        })
    }
}

/// A request to Voyage AI's embedding endpoint. This struct should not be created directly.
/// Instead, use `from_texts` for good defaults.
#[derive(Serialize, Debug, Deserialize)]
struct VoyageAIRequest {
    pub input: Vec<String>,
    pub model: String,
    input_type: Option<String>,
    truncation: bool,
    output_dimension: u32,
    output_dtype: String,
}

impl EmbeddingApiRequestTexts<VoyageAIRequest> for VoyageAIRequest {
    fn from_texts(texts: Vec<String>) -> Self {
        Self {
            input: texts,
            model: VOYAGE_EMBEDDING_MODEL.to_string(),
            input_type: None, // Directly convert to vector
            truncation: true,
            output_dimension: VOYAGE_EMBEDDING_DIM, // Matryoshka embeddings
            output_dtype: "float".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct VoyageAIEmbedding {
    object: String,
    embedding: Vec<f32>,
    index: u32,
}

#[derive(Serialize, Deserialize)]
struct VoyageAIUsage {
    total_tokens: u32,
}

#[derive(Serialize, Deserialize)]
struct VoyageAISuccess {
    object: String,
    data: Vec<VoyageAIEmbedding>,
    model: String,
    usage: VoyageAIUsage,
}

#[derive(Serialize, Deserialize)]
struct VoyageAIError {
    pub detail: String,
}

#[derive(Serialize, Deserialize)]
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
                Some(res.data.iter().map(|v| v.embedding.clone()).collect())
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

#[derive(Serialize, Deserialize)]
pub struct FailedTexts {
    pub embedding_provider: String,
    pub texts: Vec<String>,
}

/// Implements the LanceDB EmbeddingFunction trait for VoyageAIClient. Since VoyageAI has the
/// highest token limit for their embedding model (32k instead of OpenAI's 8k), we prefer this
/// instead.
impl<T: HttpClient + Default + Clone + std::fmt::Debug> EmbeddingFunction for VoyageAIClient<T> {
    fn name(&self) -> &str {
        "Voyage AI"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            VOYAGE_EMBEDDING_DIM as i32,
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

#[derive(Serialize)]
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
    type Client = T;

    fn rerank<'a>(
        &'a self,
        client: &'a T,
        items: Vec<U>,
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<U>, LLMError>> + Send + 'a>>
    where
        U: 'a,
    {
        Box::pin(async move {
            const RERANK_API_URL: &str = "https://api.voyageai.com/v1/rerank";
            let api_key = env::var("VOYAGE_AI_API_KEY")?;

            let documents: Vec<String> =
                items.iter().map(|item| item.as_ref().to_string()).collect();
            let request = VoyageAIRerankRequest {
                model: env::var("VOYAGE_AI_RERANK_MODEL")
                    .unwrap_or(DEFAULT_VOYAGE_RERANK_MODEL.into()),
                query: query.into(),
                documents,
            };

            let mut headers = HeaderMap::new();
            headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
            headers.insert("Content-Type", "application/json".parse()?);
            headers.insert("Accept", "application/json".parse()?);

            let start_time = Instant::now();
            let response = client.post_json(RERANK_API_URL, headers, &request).await?;

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

#[cfg(test)]
mod tests {
    use crate::embedding::common::Rerank;
    use crate::embedding::voyage::{VOYAGE_EMBEDDING_DIM, VoyageAIClient};
    use crate::llm::http_client::ReqwestClient;
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

        let client = VoyageAIClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Voyage AI embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), VOYAGE_EMBEDDING_DIM as i32);
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
        let reranked = client.rerank(&client.client, array.clone(), query).await;

        // Debug the error if there is one
        if reranked.is_err() {
            println!("Voyage AI reranker error: {:?}", reranked.as_ref().err());
        }

        assert!(reranked.is_ok());

        let reranked = reranked.unwrap();
        assert_eq!(reranked.len(), array.len());
    }
}
