use crate::llm::anthropic::AnthropicClient;
use crate::llm::base::ApiClient;
use crate::llm::errors::LLMError;
use crate::llm::openai::OpenAIClient;

/// Enum representing different LLM client implementations
#[derive(Debug, Clone)]
pub enum LLMClient {
    Anthropic(AnthropicClient),
    OpenAI(OpenAIClient),
}

// Implement ApiClient for LLMClient to delegate to the inner implementations
impl ApiClient for LLMClient {
    async fn send_message(
        &self,
        message: &crate::llm::base::UserMessage,
    ) -> Result<crate::llm::base::ApiResponse, LLMError> {
        match self {
            LLMClient::Anthropic(client) => client.send_message(message).await,
            LLMClient::OpenAI(client) => client.send_message(message).await,
        }
    }
}

/// Returns an ApiClient implementation based on the provider name
///
/// # Errors
/// Returns LLMError::InvalidProviderError if the provider is not supported
pub fn get_client_by_provider(provider: &str) -> Result<LLMClient, LLMError> {
    match provider {
        "anthropic" => Ok(LLMClient::Anthropic(AnthropicClient::new())),
        "openai" => Ok(LLMClient::OpenAI(OpenAIClient::new())),
        _ => Err(LLMError::InvalidProviderError(provider.to_string())),
    }
}
