//! Factory methods for creating clients based on the provider name.

use crate::clients::anthropic::AnthropicClient;
use crate::clients::openai::OpenAIClient;
use crate::config::LLMClientConfig;
use crate::llm::base::{ApiClient, ChatRequest};
use crate::llm::errors::LLMError;
use crate::llm::gemini::GeminiClient;
use crate::llm::ollama::OllamaClient;
use crate::llm::openrouter::OpenRouterClient;

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
/// Returns LLMError::InvalidProviderError if the provider is not supported
pub fn get_client_with_config(config: LLMClientConfig) -> Result<LLMClient, LLMError> {
    match config {
        LLMClientConfig::Anthropic(cfg) => {
            Ok(LLMClient::Anthropic(AnthropicClient::with_config(cfg)))
        }
        LLMClientConfig::Ollama(cfg) => Ok(LLMClient::Ollama(OllamaClient::with_config(cfg))),
        LLMClientConfig::OpenAI(cfg) => Ok(LLMClient::OpenAI(OpenAIClient::with_config(cfg))),
        LLMClientConfig::OpenRouter(cfg) => {
            Ok(LLMClient::OpenRouter(OpenRouterClient::with_config(cfg)))
        }
        LLMClientConfig::Gemini(cfg) => Ok(LLMClient::Gemini(GeminiClient::with_config(cfg))),
    }
}
