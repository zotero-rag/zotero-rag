//! A module describing the capabilities of each provider. This contains enums that show which
//! providers exposed through this crate have which capabilities. Note that it is possible for a
//! provider to not have all the capabilities listed here, if that API endpoint is not (yet) supported.

/// Providers of models that can generate text. Clients for these providers should implement
/// the `ApiClient` trait. Generally speaking, for this reason, these structs and all their trait
/// implementations will be in the `llm/` directory.
#[derive(Clone, Debug)]
pub enum ModelProviders {
    /// OpenAI model provider
    OpenAI,
    /// Anthropic model provider
    Anthropic,
    /// OpenRouter model provider
    OpenRouter,
    /// Gemini model provider
    Gemini,
}

impl ModelProviders {
    /// Returns the string representation of the provider.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelProviders::OpenAI => "openai",
            ModelProviders::Anthropic => "anthropic",
            ModelProviders::OpenRouter => "openrouter",
            ModelProviders::Gemini => "gemini",
        }
    }

    /// Returns whether the provider is contained in the list of providers.
    #[must_use]
    pub fn contains(provider: &str) -> bool {
        [
            ModelProviders::OpenAI.as_str(),
            ModelProviders::Anthropic.as_str(),
            ModelProviders::OpenRouter.as_str(),
            ModelProviders::Gemini.as_str(),
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
pub enum EmbeddingProviders {
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

impl EmbeddingProviders {
    /// Returns the string representation of the provider.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingProviders::Cohere => "cohere",
            EmbeddingProviders::OpenAI => "openai",
            EmbeddingProviders::Anthropic => "anthropic",
            EmbeddingProviders::VoyageAI => "voyageai",
            EmbeddingProviders::Gemini => "gemini",
        }
    }

    /// Returns whether the provider is contained in the list of providers.
    #[must_use]
    pub fn contains(provider: &str) -> bool {
        [
            EmbeddingProviders::Cohere.as_str(),
            EmbeddingProviders::OpenAI.as_str(),
            EmbeddingProviders::Anthropic.as_str(),
            EmbeddingProviders::VoyageAI.as_str(),
            EmbeddingProviders::Gemini.as_str(),
        ]
        .contains(&provider)
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
