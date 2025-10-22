use std::borrow::Cow;
use std::env;
use std::fmt::Debug;
use std::sync::Arc;

use http::HeaderMap;
use lancedb::arrow::arrow_schema::DataType;
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};

use super::base::{ApiClient, ChatHistoryItem, ChatRequest, CompletionApiResponse, UserMessage};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};
use crate::common::request_with_backoff;
use crate::constants::{DEFAULT_MAX_RETRIES, DEFAULT_OPENAI_MODEL, OPENAI_EMBEDDING_DIM};
use crate::embedding::openai::compute_openai_embeddings_sync;

/// A client for OpenAI's chat completions API
#[derive(Debug, Clone)]
pub struct OpenAIClient<T: HttpClient = ReqwestClient> {
    pub client: T,
    pub config: Option<crate::config::OpenAIConfig>,
}

impl<T: HttpClient + Default> Default for OpenAIClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OpenAIClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new OpenAIClient instance without configuration
    /// (will fall back to environment variables)
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new OpenAIClient instance with provided configuration
    pub fn with_config(config: crate::config::OpenAIConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
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

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIRequest {
    model: String,

    /// The conversation ID returned by the Responses API
    #[serde(skip_serializing_if = "Option::is_none")]
    conversation: Option<String>,

    input: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

impl OpenAIRequest {
    fn from_message(msg: &UserMessage, model: String, max_tokens: Option<u32>) -> Self {
        let mut messages = msg.chat_history.clone();
        messages.push(ChatHistoryItem {
            role: "user".to_owned(),
            content: msg.message.clone(),
        });

        OpenAIRequest {
            model,
            input: msg.message.clone(),
            max_output_tokens: msg.max_tokens.or(max_tokens),
            conversation: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIContent {
    r#type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
enum OpenAIOutput {
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<String>,
    },
    #[serde(rename = "message")]
    Message {
        id: String,
        status: String,
        content: Vec<OpenAIContent>,
        role: String,
    },
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIResponse {
    id: String,
    created_at: u64,
    model: String,
    usage: OpenAIUsage,
    output: Vec<OpenAIOutput>,
}

impl<T: HttpClient> ApiClient for OpenAIClient<T> {
    async fn send_message<'a>(
        &self,
        request: &ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, super::errors::LLMError> {
        // TODO: Implement tool support for OpenAI
        let message = request.message;
        // Use config if available, otherwise fall back to env vars
        let (api_key, model, max_tokens) = if let Some(ref config) = self.config {
            (
                config.api_key.clone(),
                config.model.clone(),
                Some(config.max_tokens),
            )
        } else {
            (
                env::var("OPENAI_API_KEY")?,
                env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string()),
                env::var("OPENAI_MAX_TOKENS")
                    .ok()
                    .and_then(|s| s.parse().ok()),
            )
        };

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
        headers.insert("content-type", "application/json".parse()?);

        let req_body = OpenAIRequest::from_message(message, model, max_tokens);
        dbg!(&req_body);
        const MAX_RETRIES: usize = DEFAULT_MAX_RETRIES;
        let res = request_with_backoff(
            &self.client,
            "https://api.openai.com/v1/responses",
            &headers,
            req_body,
            MAX_RETRIES,
        )
        .await?;

        let body = res.text().await?;

        let json: serde_json::Value = match serde_json::from_str(&body) {
            Ok(json) => json,
            Err(_) => return Err(super::errors::LLMError::DeserializationError(body)),
        };

        let response: OpenAIResponse = match serde_json::from_value(json) {
            Ok(response) => response,
            Err(_) => return Err(super::errors::LLMError::DeserializationError(body)),
        };

        // Find the first message output item
        let message_output = response
            .output
            .iter()
            .find_map(|item| match item {
                OpenAIOutput::Message { content, .. } => Some(content),
                OpenAIOutput::Reasoning { .. } => None,
            })
            .ok_or(LLMError::DeserializationError(
                "No message content found in OpenAI response output.".into(),
            ))?;

        let response_parts = message_output
            .iter()
            .filter_map(|res| res.text.clone())
            .collect::<Vec<_>>();

        let first_text_output = response_parts
            .first()
            .ok_or(LLMError::DeserializationError(
                "No text content found in OpenAI response output.".into(),
            ))?;

        Ok(CompletionApiResponse {
            content: first_text_output.clone(),
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

/// Implements the LanceDB EmbeddingFunction trait for OpenAI client. This is the same code
/// as the one in AnthropicClient verbatim--I made a judgement call that two copies are okay;
/// when we hit a place where we need a third copy, we'll refactor.
///
/// Maintainers should note that any updates here should also be reflected in AnthropicClient.
impl<T: HttpClient + Default + Debug> EmbeddingFunction for OpenAIClient<T> {
    fn name(&self) -> &str {
        "OpenAI"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(lancedb::arrow::arrow_schema::Field::new(
                "item",
                DataType::Float32,
                true,
            )),
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
    use super::{OpenAIClient, OpenAIContent, OpenAIOutput, OpenAIResponse, OpenAIUsage};
    use crate::constants::OPENAI_EMBEDDING_DIM;
    use crate::llm::base::{ApiClient, ChatRequest, UserMessage};
    use crate::llm::http_client::{MockHttpClient, ReqwestClient};
    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = OpenAIClient::<ReqwestClient>::default();
        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
        };
        let request = ChatRequest::from(&message);
        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("OpenAI test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_request_with_mock() {
        // Load environment variables from .env file
        dotenv().ok();

        // Create a proper OpenAIResponse that matches the structure we expect to deserialize
        let mock_response = OpenAIResponse {
            id: "mock-id".to_string(),
            created_at: 1234567890,
            model: "gpt-4.1-2025-04-14".to_string(),
            usage: OpenAIUsage {
                input_tokens: 5,
                output_tokens: 10,
                total_tokens: 15,
            },
            output: vec![OpenAIOutput::Message {
                id: "msg_id".into(),
                status: "completed".into(),
                role: "assistant".into(),
                content: vec![OpenAIContent {
                    r#type: "output_text".into(),
                    text: Some("Hi there!".into()),
                }],
            }],
        };

        let mock_http_client = MockHttpClient::new(mock_response);
        let mock_client = OpenAIClient {
            client: mock_http_client,
            config: None,
        };

        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
        };
        let request = ChatRequest::from(&message);
        let res = mock_client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("OpenAI test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
        let res = res.unwrap();
        assert_eq!(res.content, "Hi there!");
        assert_eq!(res.input_tokens, 5);
        assert_eq!(res.output_tokens, 10);
    }

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

        let client = OpenAIClient::<ReqwestClient>::default();
        let embeddings = client.compute_source_embeddings(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("OpenAI embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), OPENAI_EMBEDDING_DIM as i32);
    }
}
