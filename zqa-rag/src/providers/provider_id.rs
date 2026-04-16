//! Canonical identifiers for each provider

use std::str::FromStr;

use crate::capabilities::{EmbeddingProvider, ModelProvider, RerankerProvider};

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

impl std::fmt::Display for ProviderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
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

impl FromStr for ProviderId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "anthropic" => Ok(ProviderId::Anthropic),
            "openai" => Ok(Self::OpenAI),
            "openrouter" => Ok(Self::OpenRouter),
            "gemini" => Ok(Self::Gemini),
            "ollama" => Ok(Self::Ollama),
            "voyageai" => Ok(Self::VoyageAI),
            "cohere" => Ok(Self::Cohere),
            "zeroentropy" => Ok(Self::ZeroEntropy),
            _ => Err(format!("Invalid provider name: {s}")),
        }
    }
}

impl From<ModelProvider> for ProviderId {
    fn from(value: ModelProvider) -> Self {
        Self::from_str(value.as_str())
            .expect("ModelProvider instances contain valid provider names.")
    }
}

impl From<EmbeddingProvider> for ProviderId {
    fn from(value: EmbeddingProvider) -> Self {
        Self::from_str(value.as_str())
            .expect("EmbeddingProvider instances contain valid provider names.")
    }
}

impl From<RerankerProvider> for ProviderId {
    fn from(value: RerankerProvider) -> Self {
        Self::from_str(value.as_str())
            .expect("RerankerProvider instances contain valid provider names.")
    }
}

impl TryFrom<ProviderId> for EmbeddingProvider {
    type Error = String;

    fn try_from(value: ProviderId) -> Result<Self, Self::Error> {
        match value {
            ProviderId::VoyageAI => Ok(EmbeddingProvider::VoyageAI),
            ProviderId::ZeroEntropy => Ok(EmbeddingProvider::ZeroEntropy),
            ProviderId::Gemini => Ok(EmbeddingProvider::Gemini),
            ProviderId::OpenAI => Ok(EmbeddingProvider::OpenAI),
            ProviderId::Ollama => Ok(EmbeddingProvider::Ollama),
            ProviderId::Cohere => Ok(EmbeddingProvider::Cohere),
            _ => Err(format!("Provider {value} does not support embedding.")),
        }
    }
}

impl TryFrom<ProviderId> for RerankerProvider {
    type Error = String;

    fn try_from(value: ProviderId) -> Result<Self, Self::Error> {
        match value {
            ProviderId::VoyageAI => Ok(RerankerProvider::VoyageAI),
            ProviderId::ZeroEntropy => Ok(RerankerProvider::ZeroEntropy),
            ProviderId::Cohere => Ok(RerankerProvider::Cohere),
            _ => Err(format!("Provider {value} does not support reranking.")),
        }
    }
}

impl TryFrom<ProviderId> for ModelProvider {
    type Error = String;

    fn try_from(value: ProviderId) -> Result<Self, Self::Error> {
        match value {
            ProviderId::Gemini => Ok(ModelProvider::Gemini),
            ProviderId::Ollama => Ok(ModelProvider::Ollama),
            ProviderId::OpenAI => Ok(ModelProvider::OpenAI),
            ProviderId::Anthropic => Ok(ModelProvider::Anthropic),
            ProviderId::OpenRouter => Ok(ModelProvider::OpenRouter),
            _ => Err(format!("Provider {value} does not support generation.")),
        }
    }
}
