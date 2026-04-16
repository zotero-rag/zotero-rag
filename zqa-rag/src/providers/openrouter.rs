use crate::{
    capabilities::LlmFactory,
    clients::openrouter::OpenRouterClient,
    config::LLMClientConfig,
    llm::{errors::LLMError, factory::LLMClient},
    providers::provider_id::ProviderId,
};

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

        Ok(LLMClient::OpenRouter(OpenRouterClient::with_config(cfg.clone())))
    }
}
