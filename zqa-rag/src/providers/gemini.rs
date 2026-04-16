use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::{
    capabilities::{EmbeddingFactory, LlmFactory},
    clients::gemini::GeminiClient,
    config::LLMClientConfig,
    embedding::common::EmbeddingProviderConfig,
    http_client::ReqwestClient,
    llm::{errors::LLMError, factory::LLMClient},
    providers::provider_id::ProviderId,
    vector::lance::{LanceEmbeddingRegistrar, LanceError},
};

/// Provider descriptor for Gemini capabilities.
pub struct GeminiProvider;

impl LlmFactory for GeminiProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Gemini
    }

    fn create_llm(&self, config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
        let LLMClientConfig::Gemini(cfg) = config else {
            return Err(LLMError::InvalidProviderError("gemini".into()));
        };

        Ok(LLMClient::Gemini(GeminiClient::with_config(cfg.clone())))
    }
}

impl EmbeddingFactory for GeminiProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Gemini
    }

    fn create_embedding(
        &self,
        config: &EmbeddingProviderConfig,
    ) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
        let EmbeddingProviderConfig::Gemini(cfg) = config else {
            return Err(LLMError::InvalidProviderError("gemini".into()));
        };

        Ok(Arc::new(GeminiClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl LanceEmbeddingRegistrar for GeminiProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Gemini
    }

    fn register_with_lancedb(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError> {
        let EmbeddingProviderConfig::Gemini(cfg) = config else {
            return Err(LanceError::ParameterError("Expected Gemini embedding config".into()));
        };

        db.embedding_registry().register(
            ProviderId::Gemini.as_str(),
            Arc::new(GeminiClient::<ReqwestClient>::with_config(cfg.clone())),
        )?;
        Ok(())
    }
}
