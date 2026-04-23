//! Cohere provider implementation
use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::{
    capabilities::{EmbeddingFactory, RerankFactory},
    embedding::{cohere::CohereClient, common::EmbeddingProviderConfig},
    http_client::ReqwestClient,
    llm::errors::LLMError,
    providers::provider_id::ProviderId,
    reranking::common::{Rerank, RerankProviderConfig},
    vector::backends::{
        backend::VectorBackendRegistrar,
        lance::{LanceBackend, LanceError},
    },
};

/// Provider descriptor for Cohere capabilities.
pub struct CohereProvider;

impl EmbeddingFactory for CohereProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Cohere
    }

    fn create_embedding(
        &self,
        config: &EmbeddingProviderConfig,
    ) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
        let EmbeddingProviderConfig::Cohere(cfg) = config else {
            return Err(LLMError::InvalidProviderError("cohere".into()));
        };

        Ok(Arc::new(CohereClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl RerankFactory for CohereProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Cohere
    }

    fn create_reranker(&self, config: &RerankProviderConfig) -> Result<Arc<dyn Rerank>, LLMError> {
        let RerankProviderConfig::Cohere(cfg) = config else {
            return Err(LLMError::InvalidProviderError("cohere".into()));
        };

        Ok(Arc::new(CohereClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl VectorBackendRegistrar<LanceBackend> for CohereProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Cohere
    }

    fn register(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError> {
        let EmbeddingProviderConfig::Cohere(cfg) = config else {
            return Err(LanceError::ParameterError("expected cohere config".into()));
        };
        db.embedding_registry().register(
            ProviderId::Cohere.as_str(),
            Arc::new(CohereClient::<ReqwestClient>::with_config(cfg.clone())),
        )?;
        Ok(())
    }
}
