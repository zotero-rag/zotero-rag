use crate::llm::anthropic::AnthropicClient;
use crate::llm::base::ApiClient;
use crate::llm::errors::LLMError;

/// Returns an ApiClient implementation based on the provider name
///
/// # Errors
/// Returns LLMError::InvalidProviderError if the provider is not supported
pub fn get_client_by_provider(provider: &str) -> Result<impl ApiClient, LLMError> {
    match provider {
        "anthropic" => Ok(AnthropicClient::new()),
        _ => Err(LLMError::InvalidProviderError(provider.to_string())),
    }
}
