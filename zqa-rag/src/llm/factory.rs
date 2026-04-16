//! Factory methods for creating clients based on the provider name.

use crate::clients::anthropic::AnthropicClient;
use crate::clients::gemini::GeminiClient;
use crate::clients::ollama::OllamaClient;
use crate::clients::openai::OpenAIClient;
use crate::clients::openrouter::OpenRouterClient;
use crate::config::LLMClientConfig;
use crate::llm::base::{ApiClient, ChatRequest};
use crate::llm::errors::LLMError;
use crate::providers::registry::provider_registry;

/// Enum representing different LLM client implementations
#[derive(Debug, Clone)]
pub enum LLMClient {
    /// Anthropic client
    Anthropic(AnthropicClient),
    /// Ollama client
    Ollama(OllamaClient),
    /// OpenAI client
    OpenAI(OpenAIClient),
    /// OpenRouter client
    OpenRouter(OpenRouterClient),
    /// Gemini client
    Gemini(GeminiClient),
}

impl LLMClient {
    /// Return the configured model
    #[must_use]
    pub fn get_model_name(&self) -> Option<String> {
        match self {
            LLMClient::Anthropic(client) => client.config.as_ref().map(|c| c.model.clone()),
            LLMClient::Ollama(client) => client.config.as_ref().map(|c| c.model.clone()),
            LLMClient::OpenAI(client) => client.config.as_ref().map(|c| c.model.clone()),
            LLMClient::OpenRouter(client) => client.config.as_ref().map(|c| c.model.clone()),
            LLMClient::Gemini(client) => client.config.as_ref().map(|c| c.model.clone()),
        }
    }
}

// Implement ApiClient for LLMClient to delegate to the inner implementations
impl ApiClient for LLMClient {
    async fn send_message<'a>(
        &self,
        message: &'a ChatRequest<'a>,
    ) -> Result<crate::llm::base::CompletionApiResponse, LLMError> {
        match self {
            LLMClient::Anthropic(client) => client.send_message(message).await,
            LLMClient::Ollama(client) => client.send_message(message).await,
            LLMClient::OpenAI(client) => client.send_message(message).await,
            LLMClient::OpenRouter(client) => client.send_message(message).await,
            LLMClient::Gemini(client) => client.send_message(message).await,
        }
    }
}

/// Returns an ApiClient implementation with provided configuration
///
/// # Errors
///
/// * `LLMError::InvalidProviderError` if the provider is not supported
pub fn get_client_with_config(config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
    provider_registry().create_llm(config)
}
