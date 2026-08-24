//! Ollama provider implementation
use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::capabilities::{EmbeddingFactory, LlmFactory};
use crate::clients::ollama::OllamaClient;
use crate::config::LLMClientConfig;
use crate::embedding::common::EmbeddingProviderConfig;
use crate::http_client::ReqwestClient;
use crate::llm::errors::LLMError;
use crate::llm::factory::LLMClient;
use crate::providers::provider_id::ProviderId;
use crate::vector::backends::backend::VectorBackendRegistrar;
use crate::vector::backends::lance::{LanceBackend, LanceError};

/// Provider descriptor for Ollama capabilities.
pub struct OllamaProvider;

impl LlmFactory for OllamaProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Ollama
    }

    fn create_llm(&self, config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
        let LLMClientConfig::Ollama(cfg) = config else {
            return Err(LLMError::InvalidProviderError("ollama".into()));
        };

        Ok(LLMClient::Ollama(OllamaClient::with_config(cfg.clone())))
    }
}

impl EmbeddingFactory for OllamaProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Ollama
    }

    fn create_embedding(
        &self,
        config: &EmbeddingProviderConfig,
    ) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
        let EmbeddingProviderConfig::Ollama(cfg) = config else {
            return Err(LLMError::InvalidProviderError("ollama".into()));
        };

        Ok(Arc::new(OllamaClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl VectorBackendRegistrar<LanceBackend> for OllamaProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Ollama
    }

    fn register(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError> {
        let EmbeddingProviderConfig::Ollama(cfg) = config else {
            return Err(LanceError::ParameterError(
                "Expected Ollama embedding config".into(),
            ));
        };

        db.embedding_registry().register(
            ProviderId::Ollama.as_str(),
            Arc::new(OllamaClient::<ReqwestClient>::with_config(cfg.clone())),
        )?;
        Ok(())
    }
}
