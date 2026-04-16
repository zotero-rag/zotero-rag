#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
