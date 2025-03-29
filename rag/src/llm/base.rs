use super::errors::LLMError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatHistoryItem {
    pub role: String,
    pub content: String,
}

/// A user-facing struct that does not carry API-specific information. Clients should
/// convert from this to native message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    pub chat_history: Vec<ChatHistoryItem>,
    pub message: String,
}

/// A user-facing struct representing API responses, containing only information users
/// would be interested in.
#[derive(Debug, Clone)]
pub struct ApiResponse {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[allow(async_fn_in_trait)]
pub trait ApiClient {
    async fn send_message(&self, message: &UserMessage) -> Result<ApiResponse, LLMError>;
}
