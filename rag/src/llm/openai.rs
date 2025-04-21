use super::base::{ApiClient, ApiResponse, ChatHistoryItem, UserMessage};
use reqwest;
use serde::{Deserialize, Serialize};
use std::env;

/// A client for OpenAI's chat completions API
#[derive(Debug, Clone)]
pub struct OpenAIClient {}

impl Default for OpenAIClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAIClient {
    /// Creates a new OpenAIClient instance
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Serialize, Deserialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<ChatHistoryItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

impl From<UserMessage> for OpenAIRequest {
    fn from(msg: UserMessage) -> OpenAIRequest {
        let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string());
        let max_tokens = env::var("OPENAI_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok());
        let mut messages = msg.chat_history.clone();
        messages.push(ChatHistoryItem {
            role: "user".to_owned(),
            content: msg.message.clone(),
        });

        OpenAIRequest {
            model,
            messages,
            max_tokens,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize, Deserialize)]
struct OpenAIChoiceMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
struct OpenAIChoice {
    message: OpenAIChoiceMessage,
    finish_reason: Option<String>,
    index: u32,
}

#[derive(Serialize, Deserialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    usage: OpenAIUsage,
    choices: Vec<OpenAIChoice>,
}

impl ApiClient for OpenAIClient {
    async fn send_message(
        &self,
        message: &UserMessage,
    ) -> Result<ApiResponse, super::errors::LLMError> {
        let key = env::var("OPENAI_API_KEY")?;

        let client = reqwest::Client::new();
        let req_body: OpenAIRequest = message.clone().into();
        let res = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(key)
            .header("content-type", "application/json")
            .json(&req_body)
            .send()
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

        let choice = &response.choices[0];
        Ok(ApiResponse {
            content: choice.message.content.clone(),
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::OpenAIClient;
    use crate::llm::base::{ApiClient, ApiResponse, UserMessage};
    use crate::llm::errors::LLMError;
    use dotenv::dotenv;

    /// Mock implementation of the ApiClient trait
    struct MockOpenAIClient {
        response: Result<ApiResponse, LLMError>,
    }

    impl ApiClient for MockOpenAIClient {
        async fn send_message(&self, _message: &UserMessage) -> Result<ApiResponse, LLMError> {
            self.response.clone()
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_request_works() {
        dotenv().ok();

        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        let client = OpenAIClient::new();
        let message = UserMessage {
            chat_history: Vec::new(),
            message: "Hello!".to_owned(),
        };
        let res = client.send_message(&message).await;
        assert!(res.is_ok());
        dbg!(res.unwrap());
    }

    #[tokio::test]
    async fn test_request_with_mock() {
        let mock_response = ApiResponse {
            content: "Hi there!".to_string(),
            input_tokens: 5,
            output_tokens: 10,
        };
        let mock_client = MockOpenAIClient {
            response: Ok(mock_response.clone()),
        };
        let message = UserMessage {
            chat_history: Vec::new(),
            message: "Hello!".to_owned(),
        };
        let res = mock_client.send_message(&message).await;
        assert!(res.is_ok());
        let res = res.unwrap();
        assert_eq!(res.content, mock_response.content);
        assert_eq!(res.input_tokens, mock_response.input_tokens);
        assert_eq!(res.output_tokens, mock_response.output_tokens);
    }
}
