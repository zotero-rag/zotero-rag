use std::env;
use super::base::{ApiClient, ApiResponse, UserMessage};
use reqwest;
use serde::{Deserialize, Serialize};

/// A generic client class for now. We can add stuff here later if needed, for
/// example, features like Anthropic's native RAG thing
struct AnthropicClient {}

/// Anthropic-specific message classes
#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
}

/// We can freely use tons of hard-coded strings here; I think the resulting
/// locality-of-behavior is worth the loss in pointless generality.
impl ApiClient for AnthropicClient {
    async fn send_message(
        &self,
        message: &UserMessage,
    ) -> Result<ApiResponse, super::errors::LLMError> {
        let key = env::var("ANTHROPIC_KEY")?;

        let client = reqwest::Client::new();
        let res = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&message)
            .send()
            .await?;
    }
}
