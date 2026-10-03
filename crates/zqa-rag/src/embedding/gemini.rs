use std::borrow::Cow;
use std::env;
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use futures::{StreamExt, stream};
use http::HeaderMap;
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};

use crate::clients::gemini::{GeminiClient, get_gemini_api_key};
use crate::config::GeminiConfig;
use crate::constants::{
    DEFAULT_GEMINI_EMBEDDING_DIM, DEFAULT_GEMINI_EMBEDDING_MODEL, DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_RETRIES,
};
use crate::http_client::HttpClient;
use crate::llm::errors::LLMError;
use crate::llm::gemini::GeminiPart;
use crate::requests::request_with_backoff;

impl<T> GeminiClient<T>
where
    T: HttpClient,
{
    /// Internal method to compute embeddings that works with LLMError
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variables are set
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403 status
    /// * `LLMError::HttpStatusError` - If the API returns other unsuccessful HTTP status codes
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If the API response cannot be parsed
    /// * `LLMError::GenericLLMError` - If other HTTP errors occur or Arrow array creation fails
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.compute_embeddings_async(source))
        })
    }

    /// Compute embeddings asynchronously using the Gemini API.
    ///
    /// # Arguments:
    ///
    /// * `source`: An Arrow array
    ///
    /// # Returns
    ///
    /// If successful, an Arrow array containing the embeddings for each source text.
    pub(crate) async fn compute_embeddings_async(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        let source_array = arrow_array::cast::as_string_array(&source);
        let texts: Vec<String> = source_array
            .iter()
            .filter_map(|s| Some(s?.to_owned()))
            .collect();
        let embedding_dim = self
            .config
            .as_ref()
            .map_or(DEFAULT_GEMINI_EMBEDDING_DIM as usize, |config| {
                config.embedding_dims
            });

        // Create a stream of futures
        let futures = texts.iter().map(|text| {
            call_gemini_embedding_api(
                &self.client,
                text.clone(),
                self.config.as_ref(),
                embedding_dim,
            )
        });

        // Preserve input order so vectors remain associated with their source rows.
        let max_concurrent = env::var("MAX_CONCURRENT_REQUESTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_CONCURRENT_REQUESTS)
            .max(1);

        // Process futures with limited concurrency
        let results = stream::iter(futures)
            .buffered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        // Process results and construct Arrow array
        let embeddings: Vec<Vec<f32>> = results.into_iter().collect::<Result<_, _>>()?;

        // Convert to Arrow FixedSizeListArray
        let flattened: Vec<f32> = embeddings.iter().flatten().copied().collect();
        let values = arrow_array::Float32Array::from(flattened);

        let list_array = arrow_array::FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            embedding_dim as i32,
            Arc::new(values),
            None,
        )
        .map_err(|e| {
            LLMError::GenericLLMError(format!(
                "Failed to create FixedSizeListArray in Gemini embeddings: {e}"
            ))
        })?;

        Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
    }
}

/// Call the Gemini embeddings API.
///
/// # Arguments:
///
/// * `client`: An `HTTPClient` implementation.
/// * `text`: The text to embed.
/// * `config`: The configured credentials and model, or environment-based defaults.
/// * `embedding_dim`: The requested output width, also used for Arrow conversion.
///
/// # Returns
///
/// An embedding vector if the request was successful.
async fn call_gemini_embedding_api(
    client: &impl HttpClient,
    text: String,
    config: Option<&GeminiConfig>,
    embedding_dim: usize,
) -> Result<Vec<f32>, LLMError> {
    let api_key = config.map_or_else(get_gemini_api_key, |config| Ok(config.api_key.clone()))?;
    let model = config.map_or_else(
        || {
            env::var("GEMINI_EMBEDDING_MODEL")
                .unwrap_or_else(|_| DEFAULT_GEMINI_EMBEDDING_MODEL.to_string())
        },
        |config| config.embedding_model.clone(),
    );
    let model = format!("models/{}", model.strip_prefix("models/").unwrap_or(&model));

    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse()?);
    headers.insert("x-goog-api-key", api_key.parse()?);

    let url = format!("https://generativelanguage.googleapis.com/v1beta/{model}:embedContent");
    let request_body = GeminiEmbeddingRequest::from_text(text, model, embedding_dim);

    let res =
        request_with_backoff(client, &url, &headers, &request_body, DEFAULT_MAX_RETRIES).await?;
    let body = res.text().await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let parsed: GeminiEmbeddingResponse = serde_json::from_value(json)?;
    let mut values = parsed.embedding.values;
    if values.len() != embedding_dim {
        return Err(LLMError::GenericLLMError(format!(
            "Gemini embeddings must have the requested dimension {embedding_dim}, got {}",
            values.len()
        )));
    }

    // Gemini Embedding 001 leaves reduced-width vectors unnormalized.
    if request_body.model == "models/gemini-embedding-001" && embedding_dim < 3072 {
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut values {
                *value /= norm;
            }
        }
    }

    Ok(values)
}

/// Content for an embedding API request
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbeddingRequestContent {
    parts: Vec<GeminiPart>,
}

/// A request to embed texts using the Gemini API
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbeddingRequest {
    model: String,
    content: GeminiEmbeddingRequestContent,
    // Gemini Embedding 001 ignores the nested embedContentConfig version of this field.
    output_dimensionality: usize,
}

impl GeminiEmbeddingRequest {
    fn from_text(text: String, model: String, embedding_dim: usize) -> Self {
        Self {
            model,
            content: GeminiEmbeddingRequestContent {
                parts: vec![GeminiPart::Text {
                    text,
                    thought: None,
                    thought_signature: None,
                }],
            },
            output_dimensionality: embedding_dim,
        }
    }
}

/// A vector containing the embeddings, returned as a nested object by Gemini's embedding API.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbeddingVector {
    values: Vec<f32>,
}

/// The full embedding API response
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbeddingResponse {
    embedding: GeminiEmbeddingVector,
}

/// Implements the LanceDB EmbeddingFunction trait for Gemini client.
impl<T: HttpClient + std::fmt::Debug> EmbeddingFunction for GeminiClient<T> {
    fn name(&self) -> &'static str {
        "Gemini"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        let embedding_dim = self
            .config
            .as_ref()
            .map_or(DEFAULT_GEMINI_EMBEDDING_DIM as usize, |config| {
                config.embedding_dims
            });
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            embedding_dim as i32,
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
