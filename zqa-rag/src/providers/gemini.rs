//! Gemini provider implementation
use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::capabilities::{EmbeddingFactory, LlmFactory};
use crate::clients::gemini::GeminiClient;
use crate::config::LLMClientConfig;
use crate::embedding::common::EmbeddingProviderConfig;
use crate::http_client::ReqwestClient;
use crate::llm::errors::LLMError;
use crate::llm::factory::LLMClient;
use crate::providers::provider_id::ProviderId;
use crate::vector::backends::backend::VectorBackendRegistrar;
use crate::vector::backends::lance::{LanceBackend, LanceError};

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

impl VectorBackendRegistrar<LanceBackend> for GeminiProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Gemini
    }

    fn register(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError> {
        let EmbeddingProviderConfig::Gemini(cfg) = config else {
            return Err(LanceError::ParameterError(
                "Expected Gemini embedding config".into(),
            ));
        };

        db.embedding_registry().register(
            ProviderId::Gemini.as_str(),
            Arc::new(GeminiClient::<ReqwestClient>::with_config(cfg.clone())),
        )?;
        Ok(())
    }
}
