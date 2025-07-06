use std::{borrow::Cow, env, sync::Arc, time::Duration};

use arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use super::errors::LLMError;
use crate::llm::http_client::{HttpClient, ReqwestClient};

/// A client for Voyage AI's embedding API.
#[derive(Debug, Clone)]
pub struct VoyageAIClient<T: HttpClient = ReqwestClient> {
    pub client: T,
}

impl<T: HttpClient + Default> Default for VoyageAIClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> VoyageAIClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new AnthropicClient instance
    pub fn new() -> Self {
        Self {
            client: T::default(),
        }
    }

    /// Internal method to compute embeddings that can be reused by both source and query embedding functions
    async fn compute_embeddings_async(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        let source_array = arrow_array::cast::as_string_array(&source);
        let texts: Vec<String> = source_array.iter().map(|s| s.unwrap().to_owned()).collect();

        let api_key = env::var("VOYAGE_AI_API_KEY")?;

        let mut all_embeddings = Vec::new();
        let batch_size = 90;

        for batch in texts.chunks(batch_size) {
            let request = VoyageAIRequest::from_texts(batch.to_vec());

            let mut headers = HeaderMap::new();
            headers.insert(
                "Authorization",
                format!("Bearer {}", api_key).parse().unwrap(),
            );
            headers.insert("Content-Type", "application/json".parse().unwrap());

            let response = self
                .client
                .post_json("https://api.voyageai.com/v1/embeddings", headers, &request)
                .await?;

            let body = response.text().await?;

            let voyage_response: VoyageAISuccess = serde_json::from_str(&body)?;

            for embedding_data in voyage_response.data {
                all_embeddings.push(embedding_data.embedding);
            }

            // Wait for one minute after each batch (except the last one)
            if batch.len() == batch_size {
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        }

        // Convert to Arrow FixedSizeListArray
        let embedding_dim = 2048; // VoyageAI 3-large dimension
        let flattened: Vec<f32> = all_embeddings.iter().flatten().copied().collect();
        let values = arrow_array::Float32Array::from(flattened);

        let list_array = arrow_array::FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            embedding_dim,
            Arc::new(values),
            None,
        )
        .map_err(|e| {
            LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {}", e))
        })?;

        Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
    }

    /// Internal method to compute embeddings that works with LLMError
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            LLMError::GenericLLMError(format!("Could not create tokio runtime: {}", e))
        })?;

        rt.block_on(self.compute_embeddings_async(source))
    }
}

/// A request to Voyage AI's embedding endpoint. This struct should not be created directly.
/// Instead, use `from_texts` instead for good defaults.
#[derive(Serialize, Deserialize)]
struct VoyageAIRequest {
    pub input: Vec<String>,
    pub model: String,
    input_type: Option<String>,
    truncation: bool,
    output_dimension: u32,
    output_dtype: String,
}

impl VoyageAIRequest {
    pub fn from_texts(texts: Vec<String>) -> Self {
        Self {
            input: texts,
            model: "voyage-3-large".to_string(),
            input_type: None, // Directly convert to vector
            truncation: true,
            output_dimension: 2048, // Matryoshka embeddings
            output_dtype: "float".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct VoyageAIEmbedding {
    object: String,
    embedding: Vec<f32>,
    index: u32,
}

#[derive(Serialize, Deserialize)]
pub struct VoyageAIUsage {
    total_tokens: u32,
}

#[derive(Serialize, Deserialize)]
pub struct VoyageAISuccess {
    object: String,
    data: Vec<VoyageAIEmbedding>,
    model: String,
    usage: VoyageAIUsage,
}

#[derive(Serialize, Deserialize)]
pub struct VoyageAIError {
    detail: String,
}

#[derive(Serialize, Deserialize)]
pub enum VoyageAIResponse {
    VoyageAISuccess(VoyageAISuccess),
    VoyageAIError(VoyageAIError),
}

/// Implements the LanceDB EmbeddingFunction trait for VoyageAIClient. Since VoyageAI has the
/// highest token limit for their embedding model (32k instead of OpenAI's 8k), we prefer this
/// instead.
impl<T: HttpClient + Default + std::fmt::Debug> EmbeddingFunction for VoyageAIClient<T> {
    fn name(&self) -> &str {
        "Voyage AI"
    }

    fn source_type(&self) -> Result<Cow<DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2048,
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

#[cfg(test)]
mod tests {
    use crate::llm::http_client::ReqwestClient;
    use crate::llm::voyage::VoyageAIClient;
    use arrow_array::Array;
    use dotenv::dotenv;
    use std::sync::Arc;

    #[test]
    fn test_compute_embeddings() {
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
        assert_eq!(vector.value_length(), 2048);
    }
}
