use crate::common;
use std::borrow::Cow;
use std::env;

use arrow_array;
use futures::stream;
use futures::StreamExt;
use http::HeaderMap;
use lancedb::arrow::arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::base::{ApiClient, ApiResponse, ChatHistoryItem, UserMessage};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};

/// Anthropic does not have an embedding model, so we use OpenAI instead.
const OPENAI_EMBEDDING_DIM: u32 = 1536;
const DEFAULT_CLAUDE_MODEL: &str = "claude-sonnet-4-20250514";

/// A generic client class for now. We can add stuff here later if needed, for
/// example, features like Anthropic's native RAG thing
#[derive(Debug, Clone)]
pub struct AnthropicClient<T: HttpClient = ReqwestClient> {
    pub client: T,
}

impl<T: HttpClient + Default> Default for AnthropicClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AnthropicClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new AnthropicClient instance
    pub fn new() -> Self {
        Self {
            client: T::default(),
        }
    }

    // This is our internal implementation that works with LLMError
    // Note that this is also copied in OpenAIClient.
    // Async version to handle the embedding computation
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
            .map(|text| common::get_openai_embedding(text.clone()));

        // Convert to a stream and process with buffer_unordered to limit concurrency
        let max_concurrent = env::var("MAX_CONCURRENT_REQUESTS")
            .unwrap_or_else(|_| "5".to_string())
            .parse::<usize>()
            .unwrap_or(5);

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
            OPENAI_EMBEDDING_DIM as usize // default for text-embedding-3-small
        } else {
            embeddings[0].len()
        };

        let flattened: Vec<f32> = embeddings.iter().flatten().copied().collect();
        let values = arrow_array::Float32Array::from(flattened);

        let list_array = arrow_array::FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            embedding_dim as i32,
            Arc::new(values),
            None,
        )
        .map_err(|e| {
            LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {e}"))
        })?;

        Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
    }

    // Synchronous wrapper for the trait implementation
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        // Convert to a synchronous operation because the trait expects a Result, not a Future
        let rt = tokio::runtime::Runtime::new()
            .map_err(|_| LLMError::GenericLLMError("Could not create tokio runtime".to_string()))?;

        rt.block_on(self.compute_embeddings_async(source))
    }
}

/// Represents a request to the Anthropic API
///
/// * `model` - The model to use for the request (e.g., "claude-3-5-sonnet-20241022")
/// * `max_tokens` - The maximum number of tokens that can be generated in the response
/// * `messages` - The conversation history and current message
#[derive(Serialize, Deserialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ChatHistoryItem>,
}

impl From<UserMessage> for AnthropicRequest {
    fn from(msg: UserMessage) -> AnthropicRequest {
        let n_messages = msg.chat_history.len();
        let mut messages = msg.chat_history.clone();
        messages.insert(
            n_messages,
            ChatHistoryItem {
                role: "user".to_owned(),
                content: msg.message.clone(),
            },
        );

        AnthropicRequest {
            model: env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_CLAUDE_MODEL.to_string()),
            max_tokens: 8192,
            messages,
        }
    }
}

/// Token usage statistics returned by the Anthropic API
///
/// * `input_tokens` - Number of tokens in the input prompt
/// * `output_tokens` - Number of tokens in the generated response
#[derive(Clone, Serialize, Deserialize)]
struct AnthropicUsageStats {
    input_tokens: u32,
    output_tokens: u32,
}

/// Content block in an Anthropic API response
///
/// * `text` - The text content from the model's response
/// * `r#type` - The type of content (usually "text")
#[derive(Clone, Serialize, Deserialize)]
struct AnthropicResponseContent {
    text: String,
    r#type: String,
}

/// Response from the Anthropic API
///
/// * `id` - Unique identifier for the response
/// * `model` - The model that generated the response
/// * `role` - The role of the message (usually "assistant")
/// * `stop_reason` - Why the model stopped generating (e.g., "end_turn")
/// * `stop_sequence` - The stop sequence that caused generation to end, if any
/// * `usage` - Token usage statistics
/// * `r#type` - The type of the response (usually "message")
/// * `content` - The content blocks in the response
#[derive(Clone, Serialize, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    role: String,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: AnthropicUsageStats,
    r#type: String,
    content: Vec<AnthropicResponseContent>,
}

/// We can use hard-coded strings here; I think the resulting
/// locality-of-behavior is worth the loss in pointless generality.
impl<T: HttpClient> ApiClient for AnthropicClient<T> {
    async fn send_message(&self, message: &UserMessage) -> Result<ApiResponse, LLMError> {
        let key = env::var("ANTHROPIC_KEY")?;

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", key.parse()?);
        headers.insert("anthropic-version", "2023-06-01".parse()?);
        headers.insert("content-type", "application/json".parse()?);

        let req_body: AnthropicRequest = message.clone().into();
        let res = self
            .client
            .post_json("https://api.anthropic.com/v1/messages", headers, &req_body)
            .await?;

        // Get the response body as text first for debugging
        let body = res.text().await?;

        let json: serde_json::Value = serde_json::from_str(&body)?;
        let response: AnthropicResponse = serde_json::from_value(json)?;

        Ok(ApiResponse {
            content: response.content[0].text.clone(),
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

/// Implements the LanceDB EmbeddingFunction trait for AnthropicClient. Note that Anthropic
/// does not have their own embeddings model, so we'll just use OpenAI's model instead. This
/// does mean users will need an API key from both--but there's really no other option here.
/// Anthropic's docs recommend Voyage AI--but users are more likely to have an OpenAI key than
/// a Voyage AI key.
///
/// Maintainers should note that any updates here should also be reflected in AnthropicClient.
impl<T: HttpClient + Default + std::fmt::Debug> EmbeddingFunction for AnthropicClient<T> {
    fn name(&self) -> &str {
        "Anthropic"
    }

    fn source_type(&self) -> Result<Cow<DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            OPENAI_EMBEDDING_DIM as i32, // text-embedding-3-small size
        )))
    }

    fn compute_source_embeddings(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        // Call our internal implementation and map LLMError to lancedb::Error
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
        // For queries, we don't need concurrency since it's typically a single query
        // Just reuse the same implementation with the expectation it's usually for one item
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
    use dotenv::dotenv;

    use crate::llm::anthropic::{DEFAULT_CLAUDE_MODEL, OPENAI_EMBEDDING_DIM};
    use crate::llm::base::{ApiClient, UserMessage};
    use crate::llm::http_client::{MockHttpClient, ReqwestClient};

    use super::{
        AnthropicClient, AnthropicResponse, AnthropicResponseContent, AnthropicUsageStats,
    };

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = AnthropicClient::<ReqwestClient>::default();
        let message = UserMessage {
            chat_history: Vec::new(),
            message: "Hello!".to_owned(),
        };

        let res = client.send_message(&message).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Anthropic test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_request_with_mock() {
        // Load environment variables from .env file
        dotenv().ok();

        // Create a proper AnthropicResponse that matches the structure we expect to deserialize
        let mock_response = AnthropicResponse {
            id: "mock-id".to_string(),
            model: DEFAULT_CLAUDE_MODEL.to_string(),
            role: "assistant".to_string(),
            stop_reason: "end_turn".to_string(),
            stop_sequence: None,
            usage: AnthropicUsageStats {
                input_tokens: 9,
                output_tokens: 13,
            },
            r#type: "message".to_string(),
            content: vec![AnthropicResponseContent {
                text: "Hi there! How can I help you today?".to_string(),
                r#type: "text".to_string(),
            }],
        };

        let mock_http_client = MockHttpClient::new(mock_response);
        let mock_client = AnthropicClient {
            client: mock_http_client,
        };

        let message = UserMessage {
            chat_history: Vec::new(),
            message: "Hello!".to_owned(),
        };

        let res = mock_client.send_message(&message).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Anthropic test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());

        let res = res.unwrap();
        assert_eq!(res.input_tokens, 9);
        assert_eq!(res.output_tokens, 13);
        assert_eq!(res.content, "Hi there! How can I help you today?");
    }

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

        let client = AnthropicClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Anthropic embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), OPENAI_EMBEDDING_DIM as i32);
    }
}
