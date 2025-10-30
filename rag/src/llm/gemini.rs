use futures::StreamExt;
use std::borrow::Cow;
use std::env;
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use futures::stream;
use lancedb::embeddings::EmbeddingFunction;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::common::request_with_backoff;
use crate::constants::{
    DEFAULT_GEMINI_EMBEDDING_MODEL, DEFAULT_GEMINI_MODEL, DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_RETRIES, GEMINI_EMBEDDING_DIM,
};
use crate::llm::base::{ChatHistoryContent, ContentType};

use super::base::{ApiClient, ChatRequest, CompletionApiResponse, UserMessage};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};

/// A client for Google's Gemini APIs (chat + embeddings)
#[derive(Debug, Clone)]
pub struct GeminiClient<T: HttpClient = ReqwestClient> {
    pub client: T,
    pub config: Option<crate::config::GeminiConfig>,
}

impl<T: HttpClient + Default> Default for GeminiClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Call the Gemini embeddings API.
///
/// # Arguments:
///
/// * `client`: An `HTTPClient` implementation.
/// * `text`: The text to embed.
///
/// # Returns
///
/// An embedding vector if the request was successful.
async fn call_gemini_embedding_api(
    client: &impl HttpClient,
    text: String,
) -> Result<Vec<f32>, LLMError> {
    let api_key = get_gemini_api_key()?;
    let model = env::var("GEMINI_EMBEDDING_MODEL")
        .ok()
        .unwrap_or_else(|| DEFAULT_GEMINI_EMBEDDING_MODEL.to_string());

    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse()?);
    headers.insert("x-goog-api-key", api_key.parse()?);

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent",
        model
    );
    let request_body = GeminiEmbeddingRequest::from_text(text);

    let res =
        request_with_backoff(client, &url, &headers, &request_body, DEFAULT_MAX_RETRIES).await?;
    let body = res.text().await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let parsed: GeminiEmbeddingResponse = serde_json::from_value(json).map_err(|e| {
        LLMError::GenericLLMError(format!(
            "Failed to deserialize Gemini embedding response: {e}"
        ))
    })?;

    Ok(parsed.embedding.values)
}

impl<T> GeminiClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new GeminiClient instance without configuration
    /// (will fall back to environment variables)
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new GeminiClient instance with provided configuration
    pub fn with_config(config: crate::config::GeminiConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }

    /// Internal method to compute embeddings that works with LLMError
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
    async fn compute_embeddings_async(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        let source_array = arrow_array::cast::as_string_array(&source);
        let texts: Vec<String> = source_array
            .iter()
            .filter_map(|s| Some(s?.to_owned()))
            .collect();

        // Create a stream of futures
        let futures = texts
            .iter()
            .map(|text| call_gemini_embedding_api(&self.client, text.clone()));

        // Convert to a stream and process with buffer_unordered to limit concurrency
        let max_concurrent = env::var("MAX_CONCURRENT_REQUESTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_CONCURRENT_REQUESTS);

        // Process futures with limited concurrency
        let results = stream::iter(futures)
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        // Process results and construct Arrow array
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for result in results {
            match result {
                Ok(embedding) => embeddings.push(embedding),
                Err(e) => return Err(e),
            }
        }

        // Convert to Arrow FixedSizeListArray
        let embedding_dim = if embeddings.is_empty() {
            GEMINI_EMBEDDING_DIM as usize
        } else {
            embeddings[0].len()
        };

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

/// Get the Gemini API key from environment variables.
fn get_gemini_api_key() -> Result<String, LLMError> {
    // Prefer GEMINI_API_KEY, fallback to GOOGLE_API_KEY if present
    match env::var("GEMINI_API_KEY") {
        Ok(v) => Ok(v),
        Err(_) => Ok(env::var("GOOGLE_API_KEY")?),
    }
}

/// A content part in a request to the Gemini API
#[derive(Serialize, Deserialize, Clone)]
struct GeminiPart {
    text: String,
}

/// Content for requests to the Gemini API
#[derive(Serialize, Deserialize, Clone)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

/// Thinking config in case reasoning models are used
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
}

/// Optional text generation configuration
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

/// The request body for text generation
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiRequestBody {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

impl From<&UserMessage> for GeminiRequestBody {
    fn from(msg: &UserMessage) -> Self {
        let model_max = msg.max_tokens.or_else(|| {
            env::var("GEMINI_MAX_TOKENS")
                .ok()
                .and_then(|s| s.parse().ok())
        });

        let mut contents: Vec<GeminiContent> = msg
            .chat_history
            .iter()
            .map(|c| {
                let content = c.content[0].clone();

                GeminiContent {
                    role: map_role(&c.role),
                    parts: vec![GeminiPart {
                        text: match content {
                            ChatHistoryContent::Text(s) => s,
                            _ => "".into(),
                        },
                    }],
                }
            })
            .collect();

        contents.push(GeminiContent {
            role: "user".to_string(),
            parts: vec![GeminiPart {
                text: msg.message.clone(),
            }],
        });

        GeminiRequestBody {
            contents,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: model_max,
                // TODO: Make these configurable
                temperature: Some(1.0),
                top_k: Some(1),
                top_p: Some(1.0),
                thinking_config: Some(GeminiThinkingConfig {
                    include_thoughts: Some(false),
                    thinking_budget: Some(1024),
                }),
            }),
        }
    }
}

/// Helper function to change "assistant" roles to "model" for Gemini's API.
fn map_role(role: &str) -> String {
    match role {
        // Gemini uses "model" instead of "assistant"
        "assistant" => "model".into(),
        _ => "user".into(),
    }
}

/// Usage metadata received from the Gemini text generation response.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    thoughts_token_count: u32,
    candidates_token_count: u32,
    total_token_count: u32,
}

/// One of several response candidates.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseCandidate {
    content: GeminiContent,
    finish_reason: String,
}

/// Text generation response from the Gemini API.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseBody {
    candidates: Vec<GeminiResponseCandidate>,
    usage_metadata: GeminiUsageMetadata,
}

impl<T: HttpClient> ApiClient for GeminiClient<T> {
    async fn send_message<'a>(
        &self,
        request: &'a mut ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, LLMError> {
        // TODO: Implement tool support for Gemini
        let message = request.message;
        let key = get_gemini_api_key()?;
        let model = env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_GEMINI_MODEL.to_string());

        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse()?);
        headers.insert("x-goog-api-key", key.parse()?);

        let req_body: GeminiRequestBody = message.into();

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model
        );

        let res =
            request_with_backoff(&self.client, &url, &headers, &req_body, DEFAULT_MAX_RETRIES)
                .await?;

        let body = res.text().await?;
        let json: serde_json::Value = match serde_json::from_str(&body) {
            Ok(json) => json,
            Err(_) => return Err(LLMError::DeserializationError(body)),
        };

        let response: GeminiResponseBody = match serde_json::from_value(json) {
            Ok(response) => response,
            Err(_) => return Err(LLMError::DeserializationError(body)),
        };

        let first = response
            .candidates
            .first()
            .ok_or_else(|| LLMError::GenericLLMError("No candidates in Gemini response".into()))?;

        let content_text = first
            .content
            .parts
            .iter()
            .map(|p| p.text.clone())
            .collect::<Vec<_>>()
            .join("");

        Ok(CompletionApiResponse {
            content: vec![ContentType::Text(content_text)],
            input_tokens: response.usage_metadata.prompt_token_count,
            output_tokens: response.usage_metadata.candidates_token_count,
        })
    }
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
}

impl GeminiEmbeddingRequest {
    fn from_text(text: String) -> Self {
        Self {
            model: env::var("GEMINI_EMBEDDING_MODEL")
                .ok()
                .unwrap_or_else(|| DEFAULT_GEMINI_EMBEDDING_MODEL.to_string()),
            content: GeminiEmbeddingRequestContent {
                parts: vec![GeminiPart { text }],
            },
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
impl<T: HttpClient + Default + std::fmt::Debug> EmbeddingFunction for GeminiClient<T> {
    fn name(&self) -> &str {
        "Gemini"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            GEMINI_EMBEDDING_DIM as i32,
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
    use super::*;
    use crate::llm::base::{ApiClient, ChatHistoryItem, ChatRequest, UserMessage};
    use crate::llm::http_client::MockHttpClient;
    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;

    #[tokio::test]
    async fn test_send_message_with_mock() {
        dotenv().ok();

        let mock_response = GeminiResponseBody {
            candidates: vec![GeminiResponseCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart {
                        text: "Hello from Gemini!".into(),
                    }],
                },
                finish_reason: "stop".into(),
            }],
            usage_metadata: GeminiUsageMetadata {
                prompt_token_count: 7,
                candidates_token_count: 11,
                total_token_count: 18,
                thoughts_token_count: 0,
            },
        };

        let mock_http = MockHttpClient::new(mock_response);
        let client = GeminiClient {
            client: mock_http,
            config: None,
        };

        let message = UserMessage {
            message: "foo".into(),
            chat_history: vec![ChatHistoryItem {
                role: "assistant".into(),
                content: vec![ChatHistoryContent::Text("Prior".into())],
            }],
            max_tokens: Some(256),
        };
        let mut request = ChatRequest::from(&message);
        let res = client.send_message(&mut request).await;
        assert!(res.is_ok());
        let res = res.unwrap();
        assert_eq!(res.content.len(), 1);
        if let ContentType::Text(text) = &res.content[0] {
            assert_eq!(text, "Hello from Gemini!");
        } else {
            panic!("Expected Text content type");
        }
        assert_eq!(res.input_tokens, 7);
        assert_eq!(res.output_tokens, 11);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings_mock() {
        dotenv().ok();

        // Build a deterministic mock response that returns a 3-length embedding
        #[derive(Debug, Serialize, Deserialize, Clone, Default)]
        struct MockEmbeddingResp {
            embedding: MockEmbeddingVec,
        }
        #[derive(Debug, Serialize, Deserialize, Clone, Default)]
        struct MockEmbeddingVec {
            values: Vec<f32>,
        }

        let mock = MockEmbeddingResp {
            embedding: MockEmbeddingVec {
                values: vec![1.0, 0.0, -1.0],
            },
        };

        let mock_http = MockHttpClient::new(mock);
        let client = GeminiClient {
            client: mock_http,
            config: None,
        };

        let array = arrow_array::StringArray::from(vec!["A", "B", " ", "C"]);
        let embeddings = client.compute_source_embeddings(Arc::new(array));

        assert!(embeddings.is_ok());
        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);
        assert_eq!(vector.len(), 4);
        // With mock 3-length vectors, value_length should be 3
        assert_eq!(vector.value_length(), 3);
    }

    #[tokio::test]
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

        let client = GeminiClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_async(Arc::new(array)).await;

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Gemini embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), GEMINI_EMBEDDING_DIM as i32);
    }

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = GeminiClient::<ReqwestClient>::default();
        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
        };
        let mut request = ChatRequest::from(&message);
        let res = client.send_message(&mut request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Gemini test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
    }
}
