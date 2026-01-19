//! A module describing the capabilities of each provider. This contains enums that show which
//! providers exposed through this crate have which capabilities. Note that it is possible for a
//! provider to not have all the capabilities listed here, if that API endpoint is not (yet) supported.

use crate::chunking::ChunkingStrategy;

/// Providers of models that can generate text. Clients for these providers should implement
/// the `ApiClient` trait. Generally speaking, for this reason, these structs and all their trait
/// implementations will be in the `llm/` directory.
#[derive(Clone, Debug)]
pub enum ModelProvider {
    /// OpenAI model provider
    OpenAI,
    /// Anthropic model provider
    Anthropic,
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
            ModelProvider::OpenAI => "openai",
            ModelProvider::Anthropic => "anthropic",
            ModelProvider::OpenRouter => "openrouter",
            ModelProvider::Gemini => "gemini",
        }
    }

    /// Returns whether the provider is contained in the list of providers.
    #[must_use]
    pub fn contains(provider: &str) -> bool {
        [
            ModelProvider::OpenAI.as_str(),
            ModelProvider::Anthropic.as_str(),
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
            EmbeddingProvider::Cohere => ChunkingStrategy::WholeDocument,
            // include a buffer for approximation errors
            EmbeddingProvider::OpenAI => ChunkingStrategy::SectionBased(7500),
            EmbeddingProvider::Anthropic => ChunkingStrategy::SectionBased(7500),
            EmbeddingProvider::VoyageAI => ChunkingStrategy::WholeDocument,
            // include a buffer for approximation errors; actual limit is 2048
            EmbeddingProvider::Gemini => ChunkingStrategy::SectionBased(1500),
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
