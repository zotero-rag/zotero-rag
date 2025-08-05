use crate::common::request_with_backoff;
use std::collections::HashMap;
use std::env;

use http::HeaderMap;
use serde::{Deserialize, Serialize};

use super::base::{ApiClient, ApiResponse, ChatHistoryItem, UserMessage};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};

const DEFAULT_MODEL: &str = "openai/gpt-4o";

/// A generic client class for now. We can add stuff here later for
/// all the features OpenRouter supports.
#[derive(Debug, Clone)]
pub struct OpenRouterClient<T: HttpClient = ReqwestClient> {
    pub client: T,
}

impl<T: HttpClient + Default> Default for OpenRouterClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OpenRouterClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new OpenRouterClient instance
    pub fn new() -> Self {
        Self {
            client: T::default(),
        }
    }
}

/// Represents a request to the OpenRouter API
#[derive(Serialize, Deserialize)]
struct OpenRouterRequest {
    /// The model to use for the request (e.g., "google/gemini-2.5-flash")
    model: String,
    /// The conversation history and current message
    messages: Vec<ChatHistoryItem>,
}

impl From<UserMessage> for OpenRouterRequest {
    fn from(msg: UserMessage) -> OpenRouterRequest {
        let n_messages = msg.chat_history.len();
        let mut messages = msg.chat_history.clone();
        messages.insert(
            n_messages,
            ChatHistoryItem {
                role: "user".to_owned(),
                content: msg.message.clone(),
            },
        );

        OpenRouterRequest {
            model: env::var("OPENROUTER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            messages,
        }
    }
}

/// Token usage statistics returned by the OpenRouter API
#[derive(Clone, Serialize, Deserialize)]
struct OpenRouterUsageStats {
    /// Number of tokens in the input prompt
    prompt_tokens: u32,
    /// Number of tokens in the generated response
    completion_tokens: u32,
    /// Total token usage. Includes reasoning tokens
    total_tokens: u32,
}

/// The part of the API response containing the actual content.
#[derive(Clone, Serialize, Deserialize)]
struct OpenRouterResponseMessage {
    /// Usually "assistant"
    role: String,
    /// The model response
    content: String,
    /// Information on why a response was refused, if any
    refusal: Option<String>,
}

/// Content block in an OpenRouter API response
#[derive(Clone, Serialize, Deserialize)]
struct OpenRouterResponseChoices {
    /// The model response and related metadata
    message: OpenRouterResponseMessage,
    /// For multiple responses, a mapping to log-probabilities
    logprobs: Option<HashMap<String, f64>>,
    /// Finish reason, usually "stop", though it might indicate token limit
    finish_reason: String,
    /// Index of the message
    index: usize,
}

/// Response from the OpenRouter API
/// * `content` - The content blocks in the response
#[derive(Clone, Serialize, Deserialize)]
struct OpenRouterResponse {
    /// Unique identifier for the response
    id: String,
    /// The model that generated the response
    model: String,
    /// The model provider (e.g., "OpenAI")
    provider: String,
    /// The type of this message; we expect it to be "chat.completion"
    object: String,
    /// Timestamp this object was created
    created: u64,
    /// Token usage statistics
    usage: OpenRouterUsageStats,
    /// Model response choices
    choices: Vec<OpenRouterResponseChoices>,
}

/// We can use hard-coded strings here; I think the resulting
/// locality-of-behavior is worth the loss in pointless generality.
impl<T: HttpClient> ApiClient for OpenRouterClient<T> {
    async fn send_message(&self, message: &UserMessage) -> Result<ApiResponse, LLMError> {
        let key = env::var("OPENROUTER_API_KEY")?;

        let mut headers = HeaderMap::new();
        let auth = format!("Bearer {key}");
        headers.insert("Authorization", auth.parse()?);
        headers.insert("Content-Type", "application/json".parse()?);

        let req_body: OpenRouterRequest = message.clone().into();
        const MAX_RETRIES: usize = 3;
        let res = request_with_backoff(
            &self.client,
            "https://openrouter.ai/api/v1/chat/completions",
            &headers,
            req_body,
            MAX_RETRIES,
        )
        .await?;

        // Get the response body as text first for debugging
        let body = res.text().await?;

        let json: serde_json::Value = serde_json::from_str(&body)?;
        let response: OpenRouterResponse = serde_json::from_value(json.clone()).map_err(|err| {
            eprintln!("Failed to deserialize OpenRouter response: we got the response {json}");

            LLMError::DeserializationError(err.to_string())
        })?;

        Ok(ApiResponse {
            content: response.choices.clone()[0].message.content.clone(),
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use dotenv::dotenv;

    use super::*;
    use crate::llm::base::{ApiClient, UserMessage};
    use crate::llm::http_client::{MockHttpClient, ReqwestClient};

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = OpenRouterClient::<ReqwestClient>::default();
        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
        };

        let res = client.send_message(&message).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("OpenRouter test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_request_with_mock() {
        // Load environment variables from .env file
        dotenv().ok();

        // Create a proper OpenRouterResponse that matches the structure we expect to deserialize
        let mock_response = OpenRouterResponse {
            id: String::from("test"),
            model: String::from("openai/gpt-3.5-turbo"),
            provider: String::from("OpenAI"),
            object: String::from("chat.completion"),
            created: 1000,
            usage: OpenRouterUsageStats {
                prompt_tokens: 14,
                completion_tokens: 163,
                total_tokens: 177,
            },
            choices: vec![OpenRouterResponseChoices {
                message: OpenRouterResponseMessage {
                    role: String::from("assistant"),
                    content: String::from("Hi there! How can I help you today?"),
                    refusal: Some(String::new()),
                },
                finish_reason: String::from("stop"),
                logprobs: Some(HashMap::new()),
                index: 0,
            }],
        };

        let mock_http_client = MockHttpClient::new(mock_response);
        let mock_client = OpenRouterClient {
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
            println!("OpenRouter test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());

        let res = res.unwrap();
        assert_eq!(res.input_tokens, 14);
        assert_eq!(res.output_tokens, 163);
        assert_eq!(res.content, "Hi there! How can I help you today?");
    }
}
