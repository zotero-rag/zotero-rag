use super::base::{ApiClient, ApiResponse, ChatHistoryItem, UserMessage};
use super::errors::LLMError;
use crate::common;
use arrow_array;
use futures::stream;
use futures::StreamExt;
use lancedb::arrow::arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use reqwest;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::env;
use std::sync::Arc;

/// A generic client class for now. We can add stuff here later if needed, for
/// example, features like Anthropic's native RAG thing
#[derive(Debug, Clone)]
pub struct AnthropicClient {}

impl Default for AnthropicClient {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicClient {
    /// Creates a new AnthropicClient instance
    pub fn new() -> Self {
        Self {}
    }

    // This is our internal implementation that works with LLMError
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        // Convert to a synchronous operation because the trait expects a Result, not a Future
        let rt = tokio::runtime::Runtime::new()
            .map_err(|_| LLMError::GenericLLMError("Could not create tokio runtime".to_string()))?;

        rt.block_on(async {
            let source_array = arrow_array::cast::as_string_array(&source);
            let texts: Vec<String> = source_array.iter().map(|s| s.unwrap().to_owned()).collect();

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
                1536 // default for text-embedding-3-small
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
                LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {}", e))
            })?;

            Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
        })
    }
}

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
            model: env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string()),
            max_tokens: 8192,
            messages,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct AnthropicUsageStats {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Serialize, Deserialize)]
struct AnthropicResponseContent {
    text: String,
    r#type: String,
}

#[derive(Serialize, Deserialize)]
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
impl ApiClient for AnthropicClient {
    async fn send_message(&self, message: &UserMessage) -> Result<ApiResponse, LLMError> {
        let key = env::var("ANTHROPIC_KEY")?;

        let client = reqwest::Client::new();
        let req_body: AnthropicRequest = message.clone().into();
        let res = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&req_body)
            .send()
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
impl EmbeddingFunction for AnthropicClient {
    fn name(&self) -> &str {
        "Anthropic"
    }

    fn source_type(&self) -> Result<Cow<DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new(
                "item",
                DataType::Float32,
                false,
            )),
            1536, // text-embedding-3-small size
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
    use std::env;
    use std::sync::Arc;

    use arrow_array::Array;
    use dotenv::dotenv;

    use crate::llm::base::{ApiClient, ApiResponse, UserMessage};
    use crate::llm::errors::LLMError;

    use super::AnthropicClient;

    // Mock implementation of the ApiClient trait
    struct MockAnthropicClient {
        response: Result<ApiResponse, LLMError>,
    }

    impl ApiClient for MockAnthropicClient {
        async fn send_message(&self, _message: &UserMessage) -> Result<ApiResponse, LLMError> {
            self.response.clone()
        }
    }

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        let client = AnthropicClient {};
        let message = UserMessage {
            chat_history: Vec::new(),
            message: "Hello!".to_owned(),
        };

        let res = client.send_message(&message).await;
        dbg!(res.clone().unwrap());

        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_request_with_mock() {
        // TODO: Not fully fleshed out yet--I need a real mocking library
        let mock_response = ApiResponse {
            content: "Hi there! How can I help you today?".to_string(),
            input_tokens: 9,
            output_tokens: 13,
        };

        let mock_client = MockAnthropicClient {
            response: Ok(mock_response),
        };

        let message = UserMessage {
            chat_history: Vec::new(),
            message: "Hello!".to_owned(),
        };

        let res = mock_client.send_message(&message).await;

        assert!(res.is_ok());

        let res = res.unwrap();
        assert_eq!(res.input_tokens, 9);
        assert_eq!(res.output_tokens, 13);
    }

    #[test]
    fn test_compute_embeddings() {
        dotenv().ok();

        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        let array = arrow_array::StringArray::from(vec![
            "Hello, World!",
            "A second string",
            "A third string",
            "A fourth string",
            "A fifth string",
            "A sixth string",
        ]);

        let client = AnthropicClient::new();
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), 1536);
    }
}
