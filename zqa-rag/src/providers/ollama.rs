//! Ollama provider implementation
use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::{
    capabilities::{EmbeddingFactory, LlmFactory},
    clients::ollama::OllamaClient,
    config::LLMClientConfig,
    embedding::common::EmbeddingProviderConfig,
    http_client::ReqwestClient,
    llm::{errors::LLMError, factory::LLMClient},
    providers::provider_id::ProviderId,
    vector::backends::{
        backend::VectorBackendRegistrar,
        lance::{LanceBackend, LanceError},
    },
};

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
