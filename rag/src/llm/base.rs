use super::errors::LLMError;
use serde::{Deserialize, Serialize};

/// Backend-independent struct to represent what's sent to the APIs.
/// Implementations of ApiClient will all need at least this info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    chat_history: Vec<String>,
    message: String,
}

/// Response info from APIs that we are at least interested in.
/// Implementations of ApiClient should return this.
#[derive(Debug)]
pub struct ApiResponse {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A struct to hold secrets from .env
pub struct Secrets {
    endpoint: String,
    api_key: String,
}

/// Trait that clients should implement. This will be fleshed out
/// more as the project progresses.
#[allow(async_fn_in_trait)]
pub trait ApiClient {
    async fn send_message(&self, message: &UserMessage) -> Result<ApiResponse, LLMError>;
}
