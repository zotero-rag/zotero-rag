//! Anthropic provider implementation
use crate::{
    capabilities::LlmFactory,
    clients::anthropic::AnthropicClient,
    config::LLMClientConfig,
    llm::{errors::LLMError, factory::LLMClient},
    providers::provider_id::ProviderId,
};

/// Provider descriptor for Anthropic-backed LLM capabilities.
pub struct AnthropicProvider;

impl LlmFactory for AnthropicProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Anthropic
    }

    fn create_llm(&self, config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
        let LLMClientConfig::Anthropic(cfg) = config else {
            return Err(LLMError::InvalidProviderError("anthropic".into()));
        };

        Ok(LLMClient::Anthropic(AnthropicClient::with_config(
            cfg.clone(),
        )))
    }
}
