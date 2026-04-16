//! Voyage AI provider implementation
use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::{
    capabilities::{EmbeddingFactory, RerankFactory},
    embedding::{common::EmbeddingProviderConfig, voyage::VoyageAIClient},
    http_client::ReqwestClient,
    llm::errors::LLMError,
    providers::provider_id::ProviderId,
    reranking::common::{Rerank, RerankProviderConfig},
    vector::lance::{LanceEmbeddingRegistrar, LanceError},
};

/// Provider descriptor for Voyage AI capabilities.
pub struct VoyageAIProvider;

impl EmbeddingFactory for VoyageAIProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::VoyageAI
    }

    fn create_embedding(
        &self,
        config: &EmbeddingProviderConfig,
    ) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
        let EmbeddingProviderConfig::VoyageAI(cfg) = config else {
            return Err(LLMError::InvalidProviderError("voyageai".into()));
        };

        Ok(Arc::new(VoyageAIClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl RerankFactory for VoyageAIProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::VoyageAI
    }

    fn create_reranker(&self, config: &RerankProviderConfig) -> Result<Arc<dyn Rerank>, LLMError> {
        let RerankProviderConfig::VoyageAI(cfg) = config else {
            return Err(LLMError::InvalidProviderError("voyageai".into()));
        };

        Ok(Arc::new(VoyageAIClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl LanceEmbeddingRegistrar for VoyageAIProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::VoyageAI
    }

    fn register_with_lancedb(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError> {
        let EmbeddingProviderConfig::VoyageAI(cfg) = config else {
            return Err(LanceError::ParameterError(
                "Expected VoyageAI embedding config".into(),
            ));
        };

        db.embedding_registry().register(
            ProviderId::VoyageAI.as_str(),
            Arc::new(VoyageAIClient::<ReqwestClient>::with_config(cfg.clone())),
        )?;
        Ok(())
    }
}
