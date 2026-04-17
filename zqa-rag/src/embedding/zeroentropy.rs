//! Functions, structs, and trait implementations for interacting with the ZeroEntropy API. This
//! module includes support for embedding and reranking.

use std::{borrow::Cow, env, sync::Arc};

use arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};

use super::common::EmbeddingApiResponse;
use crate::http_client::{HttpClient, ReqwestClient};
use crate::llm::errors::LLMError;
use crate::{
    capabilities::EmbeddingProvider,
    constants::{DEFAULT_ZEROENTROPY_EMBEDDING_DIM, DEFAULT_ZEROENTROPY_EMBEDDING_MODEL},
    embedding::common::compute_embeddings_async,
};

/// A client for ZeroEntropy's embedding and reranking APIs.
#[derive(Debug, Clone)]
pub(crate) struct ZeroEntropyClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub(crate) client: T,
    /// Optional configuration for the ZeroEntropy client.
    pub(crate) config: Option<crate::config::ZeroEntropyConfig>,
}

impl<T: HttpClient + Default + Clone> Default for ZeroEntropyClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ZeroEntropyClient<T>
where
    T: HttpClient + Default + Clone,
{
    /// Creates a new `ZeroEntropyClient` instance without configuration
    /// (will fall back to environment variables).
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new `ZeroEntropyClient` instance with provided configuration.
    #[must_use]
    pub(crate) fn with_config(config: crate::config::ZeroEntropyConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }

    /// Internal helper for computing embeddings.
    ///
    /// # Arguments
    ///
    /// * `source` - Arrow string array of texts to embed.
    /// * `input_type` - Either `"document"` (for indexing) or `"query"` (for search).
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If `ZEROENTROPY_API_KEY` is not set and no config is provided.
    /// * `LLMError::TimeoutError` - If the HTTP request times out.
    /// * `LLMError::CredentialError` - If the API returns 401 or 403.
    /// * `LLMError::HttpStatusError` - If the API returns other error status codes.
    /// * `LLMError::NetworkError` - If a network error occurs.
    /// * `LLMError::DeserializationError` - If the response cannot be parsed.
    /// * `LLMError::InvalidHeaderError` - If header values cannot be parsed.
    /// * `LLMError::GenericLLMError` - For other HTTP errors or Arrow array creation failures.
    pub(crate) fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
        input_type: &'static str,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        // ZeroEntropy uses a bytes-per-minute (BPM) limit instead of tokens per minute.
        // Default limits: 2.5M BPM, 1000 RPM.
        // If we assume ~30,000 tokens per paper (we have to make *some* guess), and assume that a
        // token is ~4 bytes (we're assuming most papers are mostly written in ASCII characters and
        // are therefore 1 byte in UTF-8), that's ~120k bytes per paper. This gives us ~2.5M/120k ~
        // 20 papers per minute. So a batch of 1 paper every 3s is a half-decent estimate.
        const BATCH_SIZE: usize = 1;
        const WAIT_AFTER_REQUEST_S: u64 = 3;

        let api_key = self.config.as_ref().map_or_else(
            || env::var("ZEROENTROPY_API_KEY"),
            |config| Ok(config.api_key.clone()),
        )?;

        let embedding_model = self.config.as_ref().map_or_else(
            || DEFAULT_ZEROENTROPY_EMBEDDING_MODEL.to_string(),
            |c| c.embedding_model.clone(),
        );

        let embedding_dim_u32 = self
            .config
            .as_ref()
            .map_or(DEFAULT_ZEROENTROPY_EMBEDDING_DIM, |c| {
                c.embedding_dims as u32
            });

        let embedding_dim = embedding_dim_u32 as usize;

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(compute_embeddings_async::<
                ZeroEntropyEmbedRequest,
                ZeroEntropyEmbedResponse,
                _,
            >(
                source,
                "https://api.zeroentropy.dev/v1/models/embed",
                &api_key,
                self.client.clone(),
                // Technically there's also a `latency` option that can be passed, but it's optional
                // and the default is just a nicer behavior anyway.
                move |texts| ZeroEntropyEmbedRequest {
                    model: embedding_model.clone(),
                    input_type: input_type.to_string(),
                    input: texts,
                    dimensions: embedding_dim_u32,
                    encoding_format: "float".to_string(),
                },
                EmbeddingProvider::ZeroEntropy.as_str().to_string(),
                BATCH_SIZE,
                WAIT_AFTER_REQUEST_S,
                embedding_dim,
            ))
        })
    }
}

/// A request to ZeroEntropy's embedding API.
#[derive(Serialize, Debug)]
struct ZeroEntropyEmbedRequest {
    /// The embedding model to use (e.g., "zembed-1").
    model: String,
    /// Whether the inputs are queries or documents ("query" or "document").
    input_type: String,
    /// The texts to embed.
    input: Vec<String>,
    /// The output embedding dimension.
    dimensions: u32,
    /// The encoding format; always "float".
    encoding_format: String,
}

/// A single embedding result from ZeroEntropy.
#[derive(Serialize, Deserialize, Debug)]
struct ZeroEntropyEmbedResult {
    /// The embedding vector.
    embedding: Vec<f32>,
}

/// Usage statistics from a ZeroEntropy embedding response.
#[derive(Serialize, Deserialize, Debug)]
struct ZeroEntropyEmbedUsage {
    /// Total bytes in the request.
    total_bytes: u32,
    /// Total tokens used.
    total_tokens: u32,
}

/// Successful response from the ZeroEntropy embedding API.
#[derive(Serialize, Deserialize, Debug)]
struct ZeroEntropyEmbedSuccess {
    /// The embedding results, one per input text.
    results: Vec<ZeroEntropyEmbedResult>,
    /// Token/byte usage statistics.
    usage: ZeroEntropyEmbedUsage,
}

/// Error response from the ZeroEntropy API.
#[derive(Serialize, Deserialize, Debug)]
struct ZeroEntropyEmbedError {
    /// Human-readable error detail (may be a string or a validation error array).
    detail: Option<serde_json::Value>,
}

/// Response from the ZeroEntropy embedding API.
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
enum ZeroEntropyEmbedResponse {
    /// Successful embedding response.
    Success(ZeroEntropyEmbedSuccess),
    /// Error response.
    Error(ZeroEntropyEmbedError),
}

impl EmbeddingApiResponse for ZeroEntropyEmbedResponse {
    type Success = ZeroEntropyEmbedSuccess;
    type Error = ZeroEntropyEmbedError;

    fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    fn get_embeddings(self) -> Option<Vec<Vec<f32>>> {
        match self {
            ZeroEntropyEmbedResponse::Success(res) => {
                Some(res.results.into_iter().map(|r| r.embedding).collect())
            }
            ZeroEntropyEmbedResponse::Error(_) => None,
        }
    }

    fn get_error_message(self) -> Option<String> {
        match self {
            ZeroEntropyEmbedResponse::Error(err) => Some(err.detail.map_or_else(
                || "Unknown ZeroEntropy error".to_string(),
                |d| d.to_string(),
            )),
            ZeroEntropyEmbedResponse::Success(_) => None,
        }
    }
}

impl<T: HttpClient + Default + Clone + std::fmt::Debug> EmbeddingFunction for ZeroEntropyClient<T> {
    fn name(&self) -> &'static str {
        "ZeroEntropy"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        let dim = self
            .config
            .as_ref()
            .map_or(DEFAULT_ZEROENTROPY_EMBEDDING_DIM, |c| {
                c.embedding_dims as u32
            });
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )))
    }

    fn compute_source_embeddings(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        match self.compute_embeddings_internal(source, "document") {
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
        match self.compute_embeddings_internal(input, "query") {
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
    use zqa_macros::{test_eq, test_ok};

    use super::{DEFAULT_ZEROENTROPY_EMBEDDING_DIM, ZeroEntropyClient};
    use crate::http_client::ReqwestClient;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings() {
        dotenv().ok();

        let array = arrow_array::StringArray::from(vec![
            "Hello, World!",
            "A second string",
            "A third string",
        ]);

        let client = ZeroEntropyClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_internal(Arc::new(array), "document");

        if embeddings.is_err() {
            println!(
                "ZeroEntropy embedding error: {:?}",
                embeddings.as_ref().err()
            );
        }

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        test_eq!(vector.len(), 3);
        test_eq!(
            vector.value_length(),
            DEFAULT_ZEROENTROPY_EMBEDDING_DIM as i32
        );
    }
}
