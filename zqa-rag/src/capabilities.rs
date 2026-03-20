//! A module describing the capabilities of each provider. This contains enums that show which
//! providers exposed through this crate have which capabilities. Note that it is possible for a
//! provider to not have all the capabilities listed here, if that API endpoint is not (yet) supported.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use zqa_pdftools::chunk::ChunkingStrategy;

use crate::llm::errors::LLMError;

/// Providers of models that can generate text. Clients for these providers should implement
/// the `ApiClient` trait. Generally speaking, for this reason, these structs and all their trait
/// implementations will be in the `llm/` directory.
#[derive(Clone, Debug)]
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
        match self {
            ModelProvider::Anthropic => "anthropic",
            ModelProvider::Ollama => "ollama",
            ModelProvider::OpenAI => "openai",
            ModelProvider::OpenRouter => "openrouter",
            ModelProvider::Gemini => "gemini",
        }
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
///
/// For legacy reasons, although Anthropic does not have their own embedding model, this is
/// included in this list (and it implements `EmbeddingFunction` by calling OpenAI's embedding
/// model instead.
#[derive(Clone, Debug)]
pub enum EmbeddingProvider {
    /// Cohere embedding provider
    Cohere,
    /// OpenAI embedding provider
    OpenAI,
    /// Anthropic embedding provider, uses the OpenAI API (for now).
    Anthropic,
    /// VoyageAI embedding provider
    VoyageAI,
    /// Gemini embedding provider
    Gemini,
}

impl EmbeddingProvider {
    /// Returns the string representation of the provider.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingProvider::Cohere => "cohere",
            EmbeddingProvider::OpenAI => "openai",
            EmbeddingProvider::Anthropic => "anthropic",
            EmbeddingProvider::VoyageAI => "voyageai",
            EmbeddingProvider::Gemini => "gemini",
        }
    }

    /// Returns whether the provider is contained in the list of providers.
    #[must_use]
    pub fn contains(provider: &str) -> bool {
        [
            EmbeddingProvider::Cohere.as_str(),
            EmbeddingProvider::OpenAI.as_str(),
            EmbeddingProvider::Anthropic.as_str(),
            EmbeddingProvider::VoyageAI.as_str(),
            EmbeddingProvider::Gemini.as_str(),
        ]
        .contains(&provider)
    }

    /// Returns the recommended chunking strategy for this provider.
    #[must_use]
    pub fn recommended_chunking_strategy(&self) -> ChunkingStrategy {
        match self {
            EmbeddingProvider::Cohere | EmbeddingProvider::VoyageAI => {
                ChunkingStrategy::WholeDocument
            }
            // include a buffer for approximation errors
            EmbeddingProvider::OpenAI | EmbeddingProvider::Anthropic => {
                ChunkingStrategy::SectionBased(7500)
            }
            // include a buffer for approximation errors; actual limit is 2048
            EmbeddingProvider::Gemini => ChunkingStrategy::SectionBased(1500),
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
/// should implement the `Rerank` trait. In general, providers that have a reranker model also have
/// an embedding model, so it's likely you will find the structs in `embedding/`.
#[derive(Clone, Debug)]
pub enum RerankerProviders {
    /// Cohere reranking provider
    Cohere,
    /// VoyageAI reranking provider
    VoyageAI,
}

impl RerankerProviders {
    /// Returns the string representation of the provider.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            RerankerProviders::Cohere => "cohere",
            RerankerProviders::VoyageAI => "voyageai",
        }
    }

    /// Returns whether the provider is contained in the list of providers.
    #[must_use]
    pub fn contains(provider: &str) -> bool {
        [
            RerankerProviders::Cohere.as_str(),
            RerankerProviders::VoyageAI.as_str(),
        ]
        .contains(&provider)
    }
}

#[cfg(test)]
mod tests {
    use lancedb::embeddings::EmbeddingFunction;

    use crate::embedding::cohere::CohereClient;
    use crate::embedding::common::Rerank;
    use crate::embedding::voyage::VoyageAIClient;
    use crate::http_client::ReqwestClient;
    use crate::llm::anthropic::AnthropicClient;
    use crate::llm::base::ApiClient;
    use crate::llm::gemini::GeminiClient;
    use crate::llm::ollama::OllamaClient;
    use crate::llm::openai::OpenAIClient;
    use crate::llm::openrouter::OpenRouterClient;

    use super::BatchAPIProvider;

    fn assert_api_client<T: ApiClient>() {}
    fn assert_embedding_fn<T: EmbeddingFunction>() {}
    fn assert_batch_provider<T: BatchAPIProvider>() {}
    fn assert_reranker<T: Rerank<U>, U: AsRef<str>>() {}

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
        assert_embedding_fn::<AnthropicClient<ReqwestClient>>();
        assert_embedding_fn::<OpenAIClient<ReqwestClient>>();
        assert_embedding_fn::<GeminiClient<ReqwestClient>>();
        assert_embedding_fn::<CohereClient<ReqwestClient>>();
        assert_embedding_fn::<VoyageAIClient<ReqwestClient>>();
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
        assert_reranker::<CohereClient<ReqwestClient>, String>();
        assert_reranker::<VoyageAIClient<ReqwestClient>, String>();
    }
}
