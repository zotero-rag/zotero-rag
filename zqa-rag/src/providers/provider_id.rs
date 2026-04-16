//! Canonical identifiers for each provider

/// The canonical list of providers, regardless of capabilities. The
/// [`super::registry::ProviderRegistry`] is responsible for maintaining that information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum ProviderId {
    Anthropic,
    OpenAI,
    OpenRouter,
    Gemini,
    Ollama,
    VoyageAI,
    Cohere,
    ZeroEntropy,
}

impl ProviderId {
    /// Return a canonical string representation for a provider
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAI => "openai",
            Self::OpenRouter => "openrouter",
            Self::Gemini => "gemini",
            Self::Ollama => "ollama",
            Self::VoyageAI => "voyageai",
            Self::Cohere => "cohere",
            Self::ZeroEntropy => "zeroentropy",
        }
    }
}
