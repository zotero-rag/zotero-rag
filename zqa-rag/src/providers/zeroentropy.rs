//! ZeroEntropy provider implementation
use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::{
    capabilities::{EmbeddingFactory, RerankFactory},
    embedding::{common::EmbeddingProviderConfig, zeroentropy::ZeroEntropyClient},
    http_client::ReqwestClient,
    llm::errors::LLMError,
    providers::provider_id::ProviderId,
    reranking::common::{Rerank, RerankProviderConfig},
    vector::lance::{LanceEmbeddingRegistrar, LanceError},
};

/// Provider descriptor for ZeroEntropy capabilities.
pub struct ZeroEntropyProvider;

impl EmbeddingFactory for ZeroEntropyProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::ZeroEntropy
    }

    fn create_embedding(
        &self,
        config: &EmbeddingProviderConfig,
    ) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
        let EmbeddingProviderConfig::ZeroEntropy(cfg) = config else {
            return Err(LLMError::InvalidProviderError("zeroentropy".into()));
        };

        Ok(Arc::new(ZeroEntropyClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl RerankFactory for ZeroEntropyProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::ZeroEntropy
    }

    fn create_reranker(&self, config: &RerankProviderConfig) -> Result<Arc<dyn Rerank>, LLMError> {
        let RerankProviderConfig::ZeroEntropy(cfg) = config else {
            return Err(LLMError::InvalidProviderError("zeroentropy".into()));
        };

        Ok(Arc::new(ZeroEntropyClient::<ReqwestClient>::with_config(
            cfg.clone(),
        )))
    }
}

impl LanceEmbeddingRegistrar for ZeroEntropyProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::ZeroEntropy
    }

    fn register_with_lancedb(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError> {
        let EmbeddingProviderConfig::ZeroEntropy(cfg) = config else {
            return Err(LanceError::ParameterError(
                "Expected ZeroEntropy embedding config".into(),
            ));
        };

        db.embedding_registry().register(
            ProviderId::ZeroEntropy.as_str(),
            Arc::new(ZeroEntropyClient::<ReqwestClient>::with_config(cfg.clone())),
        )?;
        Ok(())
    }
}
