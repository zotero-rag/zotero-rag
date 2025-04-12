use super::base::{ApiClient, ApiResponse, ChatHistoryItem, UserMessage};
use reqwest;
use serde::{Deserialize, Serialize};
use std::env;

/// A generic client class for now. We can add stuff here later if needed, for
/// example, features like Anthropic's native RAG thing
#[derive(Debug, Clone)]
pub struct AnthropicClient {}

impl AnthropicClient {
    /// Creates a new AnthropicClient instance
    pub fn new() -> Self {
        Self {}
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
    async fn send_message(
        &self,
        message: &UserMessage,
    ) -> Result<ApiResponse, super::errors::LLMError> {
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

        let json: serde_json::Value = match serde_json::from_str(&body) {
            Ok(json) => json,
            Err(_) => return Err(super::errors::LLMError::DeserializationError(body)),
        };

        let response: AnthropicResponse = match serde_json::from_value(json) {
            Ok(response) => response,
            Err(_) => return Err(super::errors::LLMError::DeserializationError(body)),
        };

        Ok(ApiResponse {
            content: response.content[0].text.clone(),
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
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
    #[ignore]
    async fn test_request_works() {
        dotenv().ok();

        let client = AnthropicClient {};
        let message = UserMessage {
            chat_history: Vec::new(),
            message: "Hello!".to_owned(),
        };

        let res = client.send_message(&message).await;

        assert!(res.is_ok());

        let res = res.unwrap();
        dbg!(res);
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
}
