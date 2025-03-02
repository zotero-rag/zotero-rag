use super::errors::LLMError;

#[derive(Debug, Clone)]
pub struct UserMessage {
    chat_history: Vec<String>,
    message: String,
}

#[derive(Debug)]
pub struct ApiResponse {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub struct Secrets {
    endpoint: String,
    api_key: String,
}

#[allow(async_fn_in_trait)]
pub trait ApiClient {
    async fn send_message(&self, message: &UserMessage) -> Result<ApiResponse, LLMError>;
}
