//! Structs, functions, and traits shared by reranking clients

use std::{pin::Pin, sync::Arc};

use crate::{
    capabilities::RerankerProvider,
    embedding::{cohere::CohereClient, voyage::VoyageAIClient, zeroentropy::ZeroEntropyClient},
    http_client::ReqwestClient,
    llm::errors::LLMError,
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

/// A factory method for getting a reranking provider.
///
/// # Arguments
///
/// * `provider` - The name of the provider to get.
///
/// # Returns
///
/// An `Arc<dyn Rerank>` object, or an `LLMError` if the provider is not supported.
///
/// # Errors
///
/// Returns an error if the provider name is not recognized.
#[must_use]
pub fn get_reranking_provider(provider: RerankerProvider) -> Arc<dyn Rerank> {
    match provider {
        RerankerProvider::VoyageAI => Arc::new(VoyageAIClient::<ReqwestClient>::default()),
        RerankerProvider::Cohere => Arc::new(CohereClient::<ReqwestClient>::default()),
        RerankerProvider::ZeroEntropy => Arc::new(ZeroEntropyClient::<ReqwestClient>::default()),
    }
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
    config: RerankProviderConfig,
) -> Result<Arc<dyn Rerank>, LLMError> {
    match config {
        RerankProviderConfig::VoyageAI(cfg) => {
            Ok(Arc::new(VoyageAIClient::<ReqwestClient>::with_config(cfg)))
        }
        RerankProviderConfig::Cohere(cfg) => {
            Ok(Arc::new(CohereClient::<ReqwestClient>::with_config(cfg)))
        }
        RerankProviderConfig::ZeroEntropy(cfg) => Ok(Arc::new(
            ZeroEntropyClient::<ReqwestClient>::with_config(cfg),
        )),
    }
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
    /// Return the provider (enum)
    #[must_use]
    pub fn provider(&self) -> RerankerProvider {
        match self {
            Self::Cohere(_) => RerankerProvider::Cohere,
            Self::VoyageAI(_) => RerankerProvider::VoyageAI,
            Self::ZeroEntropy(_) => RerankerProvider::ZeroEntropy,
        }
    }

    #[must_use]
    /// Return the name of the provider for this config.
    pub fn provider_name(&self) -> &str {
        match self {
            Self::Cohere(_) => "cohere",
            Self::VoyageAI(_) => "voyageai",
            Self::ZeroEntropy(_) => "zeroentropy",
        }
    }
}
