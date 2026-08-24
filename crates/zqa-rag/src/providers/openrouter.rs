//! OpenRouter provider implementation
use crate::capabilities::LlmFactory;
use crate::clients::openrouter::OpenRouterClient;
use crate::config::LLMClientConfig;
use crate::llm::errors::LLMError;
use crate::llm::factory::LLMClient;
use crate::providers::provider_id::ProviderId;

/// Provider descriptor for OpenRouter-backed LLM capabilities.
pub struct OpenRouterProvider;

impl LlmFactory for OpenRouterProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::OpenRouter
    }

    fn create_llm(&self, config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
        let LLMClientConfig::OpenRouter(cfg) = config else {
            return Err(LLMError::InvalidProviderError("openrouter".into()));
        };

        Ok(LLMClient::OpenRouter(OpenRouterClient::with_config(
            cfg.clone(),
        )))
    }
}
