//! Structs, functions, and traits shared by reranking clients

use std::{pin::Pin, sync::Arc};

use crate::{
    capabilities::RerankerProvider,
    llm::errors::LLMError,
    providers::{ProviderId, registry::provider_registry},
};

/// A trait indicating reranking capabilities.
pub trait Rerank: Send + Sync {
    /// Rerank items using the provider.
    ///
    /// # Arguments
    ///
    /// * `items` - The items to rerank.
    /// * `query` - The query to rerank against.
    ///
    /// # Returns
    ///
    /// A vector of indices of the items reranked using the provider.
    fn rerank<'a>(
        &'a self,
        items: &'a [&str],
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<usize>, LLMError>> + Send + 'a>>;
}

/// Gets a reranking provider with configuration
///
/// # Arguments
///
/// * `config`: Provider-specific configuration
///
/// # Returns
///
/// A thread-safe object that can rerank items
///
/// # Errors
///
/// Returns an error if provider configuration is invalid or initialization fails.
pub fn get_reranking_provider_with_config(
    config: &RerankProviderConfig,
) -> Result<Arc<dyn Rerank>, LLMError> {
    provider_registry().create_reranker(config)
}

/// Configuration enum for reranking providers
#[derive(Debug, Clone)]
pub enum RerankProviderConfig {
    /// Configuration for VoyageAI reranking provider
    VoyageAI(crate::config::VoyageAIConfig),
    /// Configuration for Cohere reranking provider
    Cohere(crate::config::CohereConfig),
    /// Configuration for ZeroEntropy reranking provider
    ZeroEntropy(crate::config::ZeroEntropyConfig),
}

impl RerankProviderConfig {
    /// Return the canonical provider ID.
    #[must_use]
    pub const fn provider_id(&self) -> ProviderId {
        match self {
            Self::VoyageAI(_) => ProviderId::VoyageAI,
            Self::Cohere(_) => ProviderId::Cohere,
            Self::ZeroEntropy(_) => ProviderId::ZeroEntropy,
        }
    }

    /// Return the provider (enum)
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn provider(&self) -> RerankerProvider {
        self.provider_id()
            .try_into()
            .expect("Reranking configs always map to valid providers")
    }

    #[must_use]
    /// Return the name of the provider for this config.
    pub fn provider_name(&self) -> &str {
        self.provider_id().as_str()
    }
}
