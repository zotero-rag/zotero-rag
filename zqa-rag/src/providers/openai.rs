//! OpenAI provider implementation
use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::{
    capabilities::{EmbeddingFactory, LlmFactory},
    clients::openai::OpenAIClient,
    config::LLMClientConfig,
    embedding::common::EmbeddingProviderConfig,
    http_client::ReqwestClient,
    llm::{errors::LLMError, factory::LLMClient},
    providers::provider_id::ProviderId,
    vector::backends::{backend::VectorBackendRegistrar, lance::{LanceBackend, LanceError}},
};

/// Provider descriptor for OpenAI capabilities.
pub struct OpenAIProvider;

impl LlmFactory for OpenAIProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::OpenAI
    }

    fn create_llm(&self, config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
        let LLMClientConfig::OpenAI(cfg) = config else {
            return Err(LLMError::InvalidProviderError("openai".into()));
        };

        Ok(LLMClient::OpenAI(OpenAIClient::with_config(cfg.clone())))
    }
}

impl EmbeddingFactory for OpenAIProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::OpenAI
    }

    fn create_embedding(
        &self,
        config: &EmbeddingProviderConfig,
    ) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
        let EmbeddingProviderConfig::OpenAI(cfg) = config else {
            return Err(LLMError::InvalidProviderError("openai".into()));
        };

        Ok(Arc::new(OpenAIClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl VectorBackendRegistrar<LanceBackend> for OpenAIProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::OpenAI
    }

    fn register(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError> {
        let EmbeddingProviderConfig::OpenAI(cfg) = config else {
            return Err(LanceError::ParameterError(
                "Expected OpenAI embedding config".into(),
            ));
        };

        db.embedding_registry().register(
            ProviderId::OpenAI.as_str(),
            Arc::new(OpenAIClient::<ReqwestClient>::with_config(cfg.clone())),
        )?;
        Ok(())
    }
}
