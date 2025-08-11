use std::borrow::Cow;
use std::env;
use std::sync::Arc;

use arrow_array;
use http::HeaderMap;
use lancedb::arrow::arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};

use super::base::{ApiClient, ApiResponse, ChatHistoryItem, UserMessage};
use super::embeddings::{compute_openai_embeddings_sync, get_openai_embedding_dim};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};
use crate::common::request_with_backoff;
use crate::constants::{
    DEFAULT_ANTHROPIC_MAX_TOKENS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_MAX_RETRIES,
};
const DEFAULT_CLAUDE_MODEL: &str = DEFAULT_ANTHROPIC_MODEL;

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

    /// Internal implementation for computing embeddings using shared logic
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        compute_openai_embeddings_sync(source)
    }
}

/// Represents a request to the Anthropic API
#[derive(Serialize, Deserialize)]
struct AnthropicRequest {
    /// The model to use for the request (e.g., "claude-3-5-sonnet-20241022")
    model: String,
    /// The maximum number of tokens that can be generated in the response
    max_tokens: u32,
    /// The conversation history and current message
    messages: Vec<ChatHistoryItem>,
}

impl From<UserMessage> for AnthropicRequest {
    fn from(msg: UserMessage) -> AnthropicRequest {
        let mut messages = msg.chat_history;
        messages.push(ChatHistoryItem {
            role: "user".to_owned(),
            content: msg.message,
        });

        AnthropicRequest {
            model: env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_CLAUDE_MODEL.to_string()),
            max_tokens: msg.max_tokens.unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS),
            messages,
        }
    }
}

/// Token usage statistics returned by the Anthropic API
#[derive(Clone, Serialize, Deserialize)]
struct AnthropicUsageStats {
    /// Number of tokens in the input prompt
    input_tokens: u32,
    /// Number of tokens in the generated response
    output_tokens: u32,
}

/// Content block in an Anthropic API response
#[derive(Clone, Serialize, Deserialize)]
struct AnthropicResponseContent {
    /// The text content from the model's response
    text: String,
    /// The type of content (usually "text")
    r#type: String,
}

/// Response from the Anthropic API
#[derive(Clone, Serialize, Deserialize)]
struct AnthropicResponse {
    /// Unique identifier for the response
    id: String,
    /// The model that generated the response
    model: String,
    /// The role of the message (usually "assistant")
    role: String,
    /// Why the model stopped generating (e.g., "end_turn")
    stop_reason: String,
    /// The stop sequence that caused generation to end, if any
    stop_sequence: Option<String>,
    /// Token usage statistics
    usage: AnthropicUsageStats,
    /// The type of the response (usually "message")
    r#type: String,
    /// The content blocks in the response
    content: Vec<AnthropicResponseContent>,
}

/// We can use hard-coded strings here; I think the resulting
/// locality-of-behavior is worth the loss in pointless generality.
impl<T: HttpClient> ApiClient for AnthropicClient<T> {
    async fn send_message(&self, message: &UserMessage) -> Result<ApiResponse, LLMError> {
        let key = env::var("ANTHROPIC_API_KEY")?;

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", key.parse()?);
        headers.insert("anthropic-version", "2023-06-01".parse()?);
        headers.insert("content-type", "application/json".parse()?);

        let req_body: AnthropicRequest = message.clone().into();
        const MAX_RETRIES: usize = DEFAULT_MAX_RETRIES;
        let res = request_with_backoff(
            &self.client,
            "https://api.anthropic.com/v1/messages",
            &headers,
            req_body,
            MAX_RETRIES,
        )
        .await?;

        // Get the response body as text first for debugging
        let body = res.text().await?;

        let json: serde_json::Value = serde_json::from_str(&body)?;
        let response: AnthropicResponse = serde_json::from_value(json.clone()).map_err(|err| {
            eprintln!("Failed to deserialize Anthropic response: we got the response {json}");

            LLMError::DeserializationError(err.to_string())
        })?;

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
/// Maintainers should note that any updates here should also be reflected in OpenAIClient.
impl<T: HttpClient + Default + std::fmt::Debug> EmbeddingFunction for AnthropicClient<T> {
    fn name(&self) -> &str {
        "Anthropic"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            get_openai_embedding_dim() as i32, // text-embedding-3-small size
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
    use lancedb::embeddings::EmbeddingFunction;

    use crate::llm::anthropic::DEFAULT_CLAUDE_MODEL;
    use crate::llm::base::{ApiClient, UserMessage};
    use crate::llm::embeddings::get_openai_embedding_dim;
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
            max_tokens: Some(1024),
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
            max_tokens: None,
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

        let client = AnthropicClient::<ReqwestClient>::default();
        let embeddings = client.compute_source_embeddings(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Anthropic embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), get_openai_embedding_dim() as i32);
    }
}
