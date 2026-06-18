//! Factory methods for creating clients based on the provider name.

use crate::clients::anthropic::AnthropicClient;
use crate::clients::gemini::GeminiClient;
use crate::clients::ollama::OllamaClient;
use crate::clients::openai::OpenAIClient;
use crate::clients::openrouter::OpenRouterClient;
#[cfg(any(test, feature = "mock"))]
use crate::clients::test::TestClient;
use crate::config::LLMClientConfig;
use crate::llm::base::{ApiClient, ChatRequest, ReasoningConfig};
use crate::llm::errors::LLMError;
use crate::providers::registry::provider_registry;

/// Enum representing different LLM client implementations
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Mock client (enabled by the `mock` feature)
    #[cfg(any(test, feature = "mock"))]
    Mock(TestClient),
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
            #[cfg(any(test, feature = "mock"))]
            LLMClient::Mock(_) => Some("mock_client".into()),
        }
    }

    /// Get the reasoning config from the client's config.
    #[must_use]
    pub fn get_reasoning_config(&self) -> Option<ReasoningConfig> {
        match self {
            LLMClient::Anthropic(client) => client.config.as_ref().map(|c| ReasoningConfig {
                max_tokens: c.reasoning_budget,
                effort: None,
                summary: None,
            }),
            LLMClient::Ollama(client) => client.config.as_ref().map(|c| ReasoningConfig {
                max_tokens: c.reasoning_budget,
                effort: None,
                summary: None,
            }),
            LLMClient::OpenAI(client) => client.config.as_ref().map(|c| ReasoningConfig {
                max_tokens: None,
                effort: c.reasoning_effort.clone(),
                summary: None,
            }),
            LLMClient::OpenRouter(client) => client.config.as_ref().map(|config| {
                if config.reasoning_effort.is_none() && config.reasoning_budget.is_none() {
                    return None;
                }

                Some(ReasoningConfig {
                    max_tokens: config.reasoning_budget,
                    effort: config.reasoning_effort.clone(),
                    summary: None,
                })
            })?,
            LLMClient::Gemini(client) => client.config.as_ref().map(|c| ReasoningConfig {
                max_tokens: c.reasoning_budget,
                effort: None,
                summary: None,
            }),
            #[cfg(any(test, feature = "mock"))]
            LLMClient::Mock(_) => None,
        }
    }
}

// Implement ApiClient for LLMClient to delegate to the inner implementations
impl ApiClient for LLMClient {
    async fn send_message(
        &self,
        message: &ChatRequest<'_>,
    ) -> Result<crate::llm::base::CompletionApiResponse, LLMError> {
        match self {
            LLMClient::Anthropic(client) => client.send_message(message).await,
            LLMClient::Ollama(client) => client.send_message(message).await,
            LLMClient::OpenAI(client) => client.send_message(message).await,
            LLMClient::OpenRouter(client) => client.send_message(message).await,
            LLMClient::Gemini(client) => client.send_message(message).await,
            #[cfg(any(test, feature = "mock"))]
            LLMClient::Mock(client) => client.send_message(message).await,
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
