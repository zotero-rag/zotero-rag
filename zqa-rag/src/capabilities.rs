//! A module describing the capabilities of each provider. This contains enums that show which
//! providers exposed through this crate have which capabilities. Note that it is possible for a
//! provider to not have all the capabilities listed here, if that API endpoint is not (yet) supported.

use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, sync::Arc};
use zqa_pdftools::chunk::ChunkingStrategy;

use crate::{
    config::LLMClientConfig,
    embedding::common::EmbeddingProviderConfig,
    llm::{errors::LLMError, factory::LLMClient},
    providers::provider_id::ProviderId,
    reranking::common::{Rerank, RerankProviderConfig},
};

/// Providers of models that can generate text. Clients for these providers should implement
/// the `ApiClient` trait. Generally speaking, for this reason, these structs and all their trait
/// implementations will be in the `llm/` directory.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelProvider {
    /// Anthropic model provider
    Anthropic,
    /// Ollama model provider
    Ollama,
    /// OpenAI model provider
    OpenAI,
    /// OpenRouter model provider
    OpenRouter,
    /// Gemini model provider
    Gemini,
}

impl ModelProvider {
    /// Returns the string representation of the provider.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        ProviderId::from(self).as_str()
    }

    /// Returns whether the provider is contained in the list of providers.
    #[must_use]
    pub fn contains(provider: &str) -> bool {
        [
            ModelProvider::Anthropic.as_str(),
            ModelProvider::Ollama.as_str(),
            ModelProvider::OpenAI.as_str(),
            ModelProvider::OpenRouter.as_str(),
            ModelProvider::Gemini.as_str(),
        ]
        .contains(&provider)
    }
}

/// Providers of embedding models. Structs corresponding to these should implement LanceDB's
/// `EmbeddingFunction` trait. Depending on if the provider also has models that support text
/// generation, you will find the structs in `llm/` or `embedding/`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    /// Cohere embedding provider
    Cohere,
    /// OpenAI embedding provider
    OpenAI,
    /// Ollama embedding provider
    Ollama,
    /// VoyageAI embedding provider
    VoyageAI,
    /// Gemini embedding provider
    Gemini,
    /// ZeroEntropy embedding provider
    ZeroEntropy,
}

impl EmbeddingProvider {
    /// Returns the string representation of the provider.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        ProviderId::from(self).as_str()
    }

    /// Returns whether the provider is contained in the list of providers.
    #[must_use]
    pub fn contains(provider: &str) -> bool {
        [
            EmbeddingProvider::Cohere.as_str(),
            EmbeddingProvider::OpenAI.as_str(),
            EmbeddingProvider::Ollama.as_str(),
            EmbeddingProvider::VoyageAI.as_str(),
            EmbeddingProvider::Gemini.as_str(),
            EmbeddingProvider::ZeroEntropy.as_str(),
        ]
        .contains(&provider)
    }

    /// Returns the recommended chunking strategy for this provider.
    #[must_use]
    pub fn recommended_chunking_strategy(&self) -> ChunkingStrategy {
        match self {
            EmbeddingProvider::Cohere
            | EmbeddingProvider::VoyageAI
            | EmbeddingProvider::ZeroEntropy => ChunkingStrategy::WholeDocument,
            // include a buffer for approximation errors
            EmbeddingProvider::OpenAI => ChunkingStrategy::SectionBased(7500),
            // include a buffer for approximation errors; actual limit is 2048
            // for local models, 2048 is a reasonable guess for *most* local embedding models
            EmbeddingProvider::Gemini | EmbeddingProvider::Ollama => {
                ChunkingStrategy::SectionBased(1500)
            }
        }
    }
}

/// Providers of batch APIs generally show jobs as being in certain states. Not every such provider
/// may implement all these states, so you should not make such an assumption; however, certain
/// basic assumptions can be made, e.g., a job cannot be completed before it is created. These
/// assumptions are encoded via the [`PartialOrd`] trait.
///
/// Moreover, it is not correct to assume that the states here are a union of all the states that
/// the various batch API providers can produce; [`BatchAPIProvider`]s will sometimes logically
/// merge states; e.g., Voyage AI's [batch
/// lifecycle](https://docs.voyageai.com/docs/batch-inference#batch-lifecycle) has a "finalizing"
/// state, but the [`crate::embedding::voyage::VoyageAIClient`] changes this to
/// [`BatchJobState::InProgress`].
#[derive(Debug, PartialEq, Eq)]
pub enum BatchJobState {
    /// The batch job has been created, but it may not have started processing.
    Created,
    /// The batch job is in progress.
    InProgress,
    ///The results of the batch job are ready.
    Completed,
    /// The batch job has failed validation.
    Failed,
    /// A request has been made to cancel the job, but it has not yet canceled.
    Canceling,
    /// The batch job has been canceled.
    Canceled,
}

impl PartialOrd for BatchJobState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        // Main progression: Created < InProgress < Completed
        // Cancellation branch: Created/InProgress < Canceling < Canceled
        match (self, other) {
            // Failed is incomparable to all other states
            (BatchJobState::Failed, _)
            | (_, BatchJobState::Failed)
            | (BatchJobState::Completed, BatchJobState::Canceling | BatchJobState::Canceled)
            | (BatchJobState::Canceling | BatchJobState::Canceled, BatchJobState::Completed) => {
                None
            }

            (
                BatchJobState::Created,
                BatchJobState::InProgress
                | BatchJobState::Completed
                | BatchJobState::Canceling
                | BatchJobState::Canceled,
            )
            | (
                BatchJobState::InProgress,
                BatchJobState::Completed | BatchJobState::Canceling | BatchJobState::Canceled,
            )
            | (BatchJobState::Canceling, BatchJobState::Canceled) => Some(Ordering::Less),

            (
                BatchJobState::InProgress
                | BatchJobState::Completed
                | BatchJobState::Canceling
                | BatchJobState::Canceled,
                BatchJobState::Created,
            )
            | (
                BatchJobState::Completed | BatchJobState::Canceling | BatchJobState::Canceled,
                BatchJobState::InProgress,
            )
            | (BatchJobState::Canceled, BatchJobState::Canceling) => Some(Ordering::Greater),

            // Equal case already handled above
            _ => unreachable!(),
        }
    }
}

/// A provider of a batch API.
#[allow(async_fn_in_trait)]
pub trait BatchAPIProvider {
    /// The embedding batch creation request
    type BatchInput: Serialize;
    /// The embedding batch creation response
    type BatchSubmitResponse: for<'de> Deserialize<'de>;
    /// The embedding results
    type BatchResults: for<'de> Deserialize<'de>;

    /// Submit a job to the batch API.
    async fn submit_batch(
        &self,
        request: Self::BatchInput,
    ) -> Result<Self::BatchSubmitResponse, LLMError>;

    /// Check the status of a submitted batch job.
    async fn get_batch_status(&self, batch_id: &str) -> Result<BatchJobState, LLMError>;

    /// Get the results of a completed batch job.
    async fn get_batch_results(&self, batch_id: &str) -> Result<Self::BatchResults, LLMError>;
}

/// Providers of batch embedding APIs. Structs corresponding to these should implement
/// [`BatchAPIProvider`]. By definition, this enum is a subset of [`EmbeddingProvider`].
#[derive(Clone, Debug)]
pub enum BatchEmbeddingProvider {
    /// Voyage AI batch API provider
    VoyageAI,
}

impl BatchEmbeddingProvider {
    /// Return a string representation of the provider.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // Oh boy, what enum variant will we get, I wonder
        match self {
            BatchEmbeddingProvider::VoyageAI => "voyageai",
        }
    }
}

/// Providers of reranker (also called cross-encoder) models. Structs corresponding to values here
/// should implement the `Rerank` trait.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RerankerProvider {
    /// Cohere reranking provider
    Cohere,
    /// VoyageAI reranking provider
    VoyageAI,
    /// ZeroEntropy reranking provider
    ZeroEntropy,
}

impl RerankerProvider {
    /// Returns the string representation of the provider.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        ProviderId::from(self).as_str()
    }

    /// Returns whether the provider is contained in the list of providers.
    #[must_use]
    pub fn contains(provider: &str) -> bool {
        [
            RerankerProvider::Cohere.as_str(),
            RerankerProvider::VoyageAI.as_str(),
            RerankerProvider::ZeroEntropy.as_str(),
        ]
        .contains(&provider)
    }
}

/// Factory trait to create an [`LLMClient`] for a provider.
pub trait LlmFactory: Send + Sync {
    /// Return the canonical provider ID
    fn provider_id(&self) -> ProviderId;
    /// Attempt to create an [`LLMClient`] with the provided config.
    ///
    /// # Errors
    ///
    /// [`LLMError`] variant if creating an LLM client fails.
    fn create_llm(&self, config: &LLMClientConfig) -> Result<LLMClient, LLMError>;
}

/// Factory trait to create an [`EmbeddingFunction`] object
pub trait EmbeddingFactory: Send + Sync {
    /// Return the canonical provider ID
    fn provider_id(&self) -> ProviderId;
    /// Attempt to create a [`EmbeddingProvider`] object with the provided config.
    ///
    /// # Errors
    ///
    /// [`LLMError`] variant if creating a trait object fails.
    fn create_embedding(
        &self,
        config: &EmbeddingProviderConfig,
    ) -> Result<Arc<dyn EmbeddingFunction>, LLMError>;
}

/// Factory trait to create an [`Rerank`] object
pub trait RerankFactory: Send + Sync {
    /// Return the canonical provider ID
    fn provider_id(&self) -> ProviderId;
    /// Attempt to create a [`Rerank`] object with the provided config.
    ///
    /// # Errors
    ///
    /// [`LLMError`] variant if creating a reranker fails.
    fn create_reranker(&self, config: &RerankProviderConfig) -> Result<Arc<dyn Rerank>, LLMError>;
}

#[cfg(test)]
mod tests {
    use lancedb::embeddings::EmbeddingFunction;

    use crate::clients::anthropic::AnthropicClient;
    use crate::clients::gemini::GeminiClient;
    use crate::clients::ollama::OllamaClient;
    use crate::clients::openai::OpenAIClient;
    use crate::clients::openrouter::OpenRouterClient;
    use crate::embedding::cohere::CohereClient;
    use crate::embedding::voyage::VoyageAIClient;
    use crate::embedding::zeroentropy::ZeroEntropyClient;
    use crate::http_client::ReqwestClient;
    use crate::llm::base::ApiClient;
    use crate::reranking::common::Rerank;

    use super::BatchAPIProvider;

    fn assert_api_client<T: ApiClient>() {}
    fn assert_embedding_fn<T: EmbeddingFunction>() {}
    fn assert_batch_provider<T: BatchAPIProvider>() {}
    fn assert_reranker<T: Rerank>() {}

    /// Verify that every [`super::ModelProvider`] variant has a corresponding client that
    /// implements [`ApiClient`]. If a client is removed or its trait impl is dropped, this
    /// will fail to compile.
    #[test]
    fn model_providers_implement_api_client() {
        assert_api_client::<AnthropicClient<ReqwestClient>>();
        assert_api_client::<OllamaClient<ReqwestClient>>();
        assert_api_client::<OpenAIClient<ReqwestClient>>();
        assert_api_client::<OpenRouterClient<ReqwestClient>>();
        assert_api_client::<GeminiClient<ReqwestClient>>();
    }

    /// Verify that every [`super::EmbeddingProvider`] variant has a corresponding client that
    /// implements LanceDB's [`EmbeddingFunction`].
    #[test]
    fn embedding_providers_implement_embedding_function() {
        assert_embedding_fn::<OpenAIClient<ReqwestClient>>();
        assert_embedding_fn::<OllamaClient<ReqwestClient>>();
        assert_embedding_fn::<GeminiClient<ReqwestClient>>();
        assert_embedding_fn::<CohereClient<ReqwestClient>>();
        assert_embedding_fn::<VoyageAIClient<ReqwestClient>>();
        assert_embedding_fn::<ZeroEntropyClient<ReqwestClient>>();
    }

    /// Verify that [`super::BatchEmbeddingProvider::VoyageAI`] has a corresponding client that
    /// implements [`BatchAPIProvider`].
    #[test]
    fn batch_embedding_providers_implement_batch_api() {
        assert_batch_provider::<VoyageAIClient<ReqwestClient>>();
    }

    /// Verify that every [`super::RerankerProviders`] variant has a corresponding client that
    /// implements [`Rerank`].
    #[test]
    fn reranker_providers_implement_rerank() {
        assert_reranker::<CohereClient<ReqwestClient>>();
        assert_reranker::<VoyageAIClient<ReqwestClient>>();
        assert_reranker::<ZeroEntropyClient<ReqwestClient>>();
    }
}
