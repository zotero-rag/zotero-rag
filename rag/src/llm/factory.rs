//! Factory methods for creating clients based on the provider name.

use crate::llm::anthropic::AnthropicClient;
use crate::llm::base::{ApiClient, ChatRequest};
use crate::llm::errors::LLMError;
use crate::llm::gemini::GeminiClient;
use crate::llm::openai::OpenAIClient;
use crate::llm::openrouter::OpenRouterClient;

/// Enum representing different LLM client implementations
#[derive(Debug, Clone)]
pub enum LLMClient {
    /// Anthropic client
    Anthropic(AnthropicClient),
    /// OpenAI client
    OpenAI(OpenAIClient),
    /// OpenRouter client
    OpenRouter(OpenRouterClient),
    /// Gemini client
    Gemini(GeminiClient),
}

// Implement ApiClient for LLMClient to delegate to the inner implementations
impl ApiClient for LLMClient {
    async fn send_message<'a>(
        &self,
        message: &'a ChatRequest<'a>,
    ) -> Result<crate::llm::base::CompletionApiResponse, LLMError> {
        match self {
            LLMClient::Anthropic(client) => client.send_message(message).await,
            LLMClient::OpenAI(client) => client.send_message(message).await,
            LLMClient::OpenRouter(client) => client.send_message(message).await,
            LLMClient::Gemini(client) => client.send_message(message).await,
        }
    }
}

/// Returns an ApiClient implementation based on the provider name
/// without configuration (will fall back to environment variables)
///
/// # Errors
/// Returns LLMError::InvalidProviderError if the provider is not supported
pub fn get_client_by_provider(provider: &str) -> Result<LLMClient, LLMError> {
    match provider {
        "anthropic" => Ok(LLMClient::Anthropic(AnthropicClient::new())),
        "openai" => Ok(LLMClient::OpenAI(OpenAIClient::new())),
        "openrouter" => Ok(LLMClient::OpenRouter(OpenRouterClient::new())),
        "gemini" => Ok(LLMClient::Gemini(GeminiClient::new())),
        _ => Err(LLMError::InvalidProviderError(provider.to_string())),
    }
}

/// Configuration for LLM clients
#[derive(Debug, Clone)]
pub enum LLMClientConfig {
    /// Anthropic client configuration
    Anthropic(crate::config::AnthropicConfig),
    /// OpenAI client configuration
    OpenAI(crate::config::OpenAIConfig),
    /// OpenRouter client configuration
    OpenRouter(crate::config::OpenRouterConfig),
    /// Gemini client configuration
    Gemini(crate::config::GeminiConfig),
}

/// Returns an ApiClient implementation with provided configuration
///
/// # Errors
/// Returns LLMError::InvalidProviderError if the provider is not supported
pub fn get_client_with_config(config: LLMClientConfig) -> Result<LLMClient, LLMError> {
    match config {
        LLMClientConfig::Anthropic(cfg) => {
            Ok(LLMClient::Anthropic(AnthropicClient::with_config(cfg)))
        }
        LLMClientConfig::OpenAI(cfg) => Ok(LLMClient::OpenAI(OpenAIClient::with_config(cfg))),
        LLMClientConfig::OpenRouter(cfg) => {
            Ok(LLMClient::OpenRouter(OpenRouterClient::with_config(cfg)))
        }
        LLMClientConfig::Gemini(cfg) => Ok(LLMClient::Gemini(GeminiClient::with_config(cfg))),
    }
}
