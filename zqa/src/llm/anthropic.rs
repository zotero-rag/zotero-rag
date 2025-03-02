use super::base::{ApiClient, ApiResponse, UserMessage};

struct AnthropicClient {}

impl ApiClient for AnthropicClient {
    async fn send_message(
        &self,
        message: &UserMessage,
    ) -> Result<ApiResponse, super::errors::LLMError> {
        Ok(ApiResponse {
            content: String::new(),
            input_tokens: 0,
            output_tokens: 0,
        })
    }
}
