/// Providers of models that can generate text. Clients for these providers should implement
/// the `ApiClient` trait. Generally speaking, for this reason, these structs and all their trait
/// implementations will be in the `llm/` directory.
#[derive(Clone, Debug)]
pub enum ModelProviders {
    OpenAI,
    Anthropic,
    OpenRouter,
    Gemini,
}

impl ModelProviders {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelProviders::OpenAI => "openai",
            ModelProviders::Anthropic => "anthropic",
            ModelProviders::OpenRouter => "openrouter",
            ModelProviders::Gemini => "gemini",
        }
    }

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
    OpenAI,
    Anthropic,
    VoyageAI,
    Gemini,
}

impl EmbeddingProviders {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingProviders::OpenAI => "openai",
            EmbeddingProviders::Anthropic => "anthropic",
            EmbeddingProviders::VoyageAI => "voyageai",
            EmbeddingProviders::Gemini => "gemini",
        }
    }

    pub fn contains(provider: &str) -> bool {
        [
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
    Cohere,
    VoyageAI,
}

impl RerankerProviders {
    pub fn as_str(&self) -> &'static str {
        match self {
            RerankerProviders::Cohere => "cohere",
            RerankerProviders::VoyageAI => "voyageai",
        }
    }

    pub fn contains(provider: &str) -> bool {
        [
            RerankerProviders::Cohere.as_str(),
            RerankerProviders::VoyageAI.as_str(),
        ]
        .contains(&provider)
    }
}
