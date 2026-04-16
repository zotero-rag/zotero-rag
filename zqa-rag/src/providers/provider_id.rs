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

impl From<&ModelProvider> for ProviderId {
    fn from(value: &ModelProvider) -> Self {
        match value {
            ModelProvider::Anthropic => Self::Anthropic,
            ModelProvider::Ollama => Self::Ollama,
            ModelProvider::OpenAI => Self::OpenAI,
            ModelProvider::OpenRouter => Self::OpenRouter,
            ModelProvider::Gemini => Self::Gemini,
        }
    }
}

impl From<&EmbeddingProvider> for ProviderId {
    fn from(value: &EmbeddingProvider) -> Self {
        match value {
            EmbeddingProvider::Cohere => Self::Cohere,
            EmbeddingProvider::OpenAI => Self::OpenAI,
            EmbeddingProvider::Ollama => Self::Ollama,
            EmbeddingProvider::VoyageAI => Self::VoyageAI,
            EmbeddingProvider::Gemini => Self::Gemini,
            EmbeddingProvider::ZeroEntropy => Self::ZeroEntropy,
        }
    }
}

impl From<&RerankerProvider> for ProviderId {
    fn from(value: &RerankerProvider) -> Self {
        match value {
            RerankerProvider::Cohere => Self::Cohere,
            RerankerProvider::VoyageAI => Self::VoyageAI,
            RerankerProvider::ZeroEntropy => Self::ZeroEntropy,
        }
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

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::capabilities::{EmbeddingProvider, ModelProvider, RerankerProvider};

    use super::ProviderId;

    #[test]
    fn provider_id_parses_all_supported_names() {
        assert_eq!(ProviderId::from_str("anthropic"), Ok(ProviderId::Anthropic));
        assert_eq!(ProviderId::from_str("openai"), Ok(ProviderId::OpenAI));
        assert_eq!(
            ProviderId::from_str("openrouter"),
            Ok(ProviderId::OpenRouter)
        );
        assert_eq!(ProviderId::from_str("gemini"), Ok(ProviderId::Gemini));
        assert_eq!(ProviderId::from_str("ollama"), Ok(ProviderId::Ollama));
        assert_eq!(ProviderId::from_str("voyageai"), Ok(ProviderId::VoyageAI));
        assert_eq!(ProviderId::from_str("cohere"), Ok(ProviderId::Cohere));
        assert_eq!(
            ProviderId::from_str("zeroentropy"),
            Ok(ProviderId::ZeroEntropy)
        );
    }

    #[test]
    fn provider_id_rejects_invalid_names() {
        assert!(ProviderId::from_str("not-a-provider").is_err());
    }

    #[test]
    fn capability_enums_convert_to_provider_ids() {
        assert_eq!(ProviderId::from(&ModelProvider::OpenAI), ProviderId::OpenAI);
        assert_eq!(
            ProviderId::from(&EmbeddingProvider::Gemini),
            ProviderId::Gemini
        );
        assert_eq!(
            ProviderId::from(&RerankerProvider::ZeroEntropy),
            ProviderId::ZeroEntropy
        );
    }

    #[test]
    fn provider_ids_convert_to_supported_capability_enums() {
        assert_eq!(
            EmbeddingProvider::try_from(ProviderId::OpenAI),
            Ok(EmbeddingProvider::OpenAI)
        );
        assert_eq!(
            RerankerProvider::try_from(ProviderId::Cohere),
            Ok(RerankerProvider::Cohere)
        );
        assert_eq!(
            ModelProvider::try_from(ProviderId::Anthropic),
            Ok(ModelProvider::Anthropic)
        );
    }

    #[test]
    fn provider_ids_reject_unsupported_capability_conversions() {
        assert!(EmbeddingProvider::try_from(ProviderId::Anthropic).is_err());
        assert!(RerankerProvider::try_from(ProviderId::OpenAI).is_err());
        assert!(ModelProvider::try_from(ProviderId::VoyageAI).is_err());
    }
}
