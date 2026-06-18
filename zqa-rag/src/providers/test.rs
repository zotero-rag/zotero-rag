use crate::{
    capabilities::LlmFactory,
    clients::test::TestClient,
    config::LLMClientConfig,
    llm::{errors::LLMError, factory::LLMClient},
    providers::ProviderId,
};

/// An [`LlmFactory`] that produces mock LLM clients returning canned responses.
pub struct MockProvider;

impl LlmFactory for MockProvider {
    fn provider_id(&self) -> super::ProviderId {
        ProviderId::Mock
    }

    fn create_llm(&self, config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
        let LLMClientConfig::Mock(cfg) = config else {
            return Err(LLMError::InvalidProviderError("mock".into()));
        };

        Ok(LLMClient::Mock(TestClient::new(&cfg.responses)))
    }
}
