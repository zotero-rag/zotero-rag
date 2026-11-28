//! Configuration structures for RAG providers
//!
//! Every provider client is built from one of these configs. The crate does not read environment
//! variables or config files; resolving API keys and settings is the caller's job (`zqa` reads
//! them from its TOML config and the environment).

use crate::constants::{
    DEFAULT_ANTHROPIC_MAX_TOKENS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_GEMINI_EMBEDDING_DIM,
    DEFAULT_GEMINI_EMBEDDING_MODEL, DEFAULT_GEMINI_MODEL, DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_RETRIES, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_EMBEDDING_DIM,
    DEFAULT_OLLAMA_EMBEDDING_MODEL, DEFAULT_OLLAMA_MAX_TOKENS, DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_EMBEDDING_DIM, DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_OPENAI_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL, DEFAULT_OPENROUTER_MAX_TOKENS, DEFAULT_OPENROUTER_MODEL,
};
use crate::providers::ProviderId;

/// Configuration for Anthropic LLM provider
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for Anthropic
    pub api_key: String,
    /// Model name (e.g., "claude-sonnet-4-5")
    pub model: String,
    /// Maximum tokens for generation
    pub max_tokens: u32,
    /// Token budget for extended thinking. `None` disables thinking.
    /// Only used on models with manual extended thinking (Claude Opus 4.5 and earlier).
    pub reasoning_budget: Option<u32>,
    /// Reasoning effort level ("low", "medium", "high", "xhigh", "max") for models with
    /// adaptive thinking (Claude Opus 4.6+ and Claude 5 models). `None` uses the model default.
    pub reasoning_effort: Option<String>,
    /// Maximum number of retries for a request that fails in a retryable way
    pub max_retries: usize,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: DEFAULT_ANTHROPIC_MODEL.into(),
            max_tokens: DEFAULT_ANTHROPIC_MAX_TOKENS,
            reasoning_budget: None,
            reasoning_effort: None,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
}

/// Configuration for OpenAI LLM and embedding provider
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API key for OpenAI
    pub api_key: String,
    /// Model name for generation (e.g., "gpt-5.2")
    pub model: String,
    /// Maximum tokens for generation
    pub max_tokens: u32,
    /// Embedding model name
    pub embedding_model: String,
    /// Embedding dimensions
    pub embedding_dims: usize,
    /// Reasoning effort level (e.g., "high"). `None` disables reasoning.
    pub reasoning_effort: Option<String>,
    /// Maximum number of retries for a request that fails in a retryable way
    pub max_retries: usize,
    /// Maximum number of concurrent embedding requests
    pub max_concurrent_requests: usize,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: DEFAULT_OPENAI_MODEL.into(),
            max_tokens: DEFAULT_OPENAI_MAX_TOKENS,
            embedding_model: DEFAULT_OPENAI_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_OPENAI_EMBEDDING_DIM as usize,
            reasoning_effort: None,
            max_retries: DEFAULT_MAX_RETRIES,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }
}

/// Configuration for `ollama` LLM and embedding provider
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Model name for generation (e.g., "qwen3.5")
    pub model: String,
    /// Maximum tokens for generation
    pub max_tokens: u32,
    /// Embedding model name
    pub embedding_model: String,
    /// Embedding dimensions
    pub embedding_dims: usize,
    /// Base URL for the ollama API (e.g., "http://localhost:11434")
    pub base_url: String,
    /// Token budget for extended thinking. `None` disables thinking.
    pub reasoning_budget: Option<u32>,
    /// Maximum number of retries for a request that fails in a retryable way
    pub max_retries: usize,
    /// Maximum number of concurrent embedding requests
    pub max_concurrent_requests: usize,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_OLLAMA_MODEL.into(),
            max_tokens: DEFAULT_OLLAMA_MAX_TOKENS,
            embedding_model: DEFAULT_OLLAMA_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_OLLAMA_EMBEDDING_DIM,
            base_url: DEFAULT_OLLAMA_BASE_URL.into(),
            reasoning_budget: None,
            max_retries: DEFAULT_MAX_RETRIES,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }
}

/// Configuration for Gemini LLM and embedding provider
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    /// API key for Gemini
    pub api_key: String,
    /// Model name for generation (e.g., "gemini-2.5-pro")
    pub model: String,
    /// Embedding model name
    pub embedding_model: String,
    /// Embedding dimensions
    pub embedding_dims: usize,
    /// Reasoning effort, mapped to a Gemini thinking level. `None` uses the model default.
    pub reasoning_effort: Option<String>,
    /// Maximum number of retries for a request that fails in a retryable way
    pub max_retries: usize,
    /// Maximum number of concurrent embedding requests
    pub max_concurrent_requests: usize,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: DEFAULT_GEMINI_MODEL.into(),
            embedding_model: DEFAULT_GEMINI_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_GEMINI_EMBEDDING_DIM as usize,
            reasoning_effort: None,
            max_retries: DEFAULT_MAX_RETRIES,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }
}

/// Configuration for Voyage AI embedding and reranking provider
#[derive(Debug, Clone)]
pub struct VoyageAIConfig {
    /// API key for Voyage AI
    pub api_key: String,
    /// Embedding model name
    pub embedding_model: String,
    /// Embedding dimensions
    pub embedding_dims: usize,
    /// Reranker model name
    pub reranker: String,
    /// Maximum number of concurrent embedding requests
    pub max_concurrent_requests: usize,
    /// Maximum number of retries for a request that fails in a retryable way
    pub max_retries: usize,
}

/// Configuration for Cohere embedding and reranking provider
#[derive(Debug, Clone)]
pub struct CohereConfig {
    /// API key for Cohere
    pub api_key: String,
    /// Embedding model name
    pub embedding_model: String,
    /// Embedding dimensions
    pub embedding_dims: usize,
    /// Reranker model name
    pub reranker: String,
    /// Maximum number of concurrent embedding requests
    pub max_concurrent_requests: usize,
    /// Maximum number of retries for a request that fails in a retryable way
    pub max_retries: usize,
}

/// Configuration for OpenRouter provider
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    /// API key for OpenRouter
    pub api_key: String,
    /// Model name (e.g., "anthropic/claude-sonnet-4.5")
    pub model: String,
    /// Output token budget
    pub max_tokens: u32,
    /// Reasoning effort level (e.g., "high"). `None` disables reasoning.
    pub reasoning_effort: Option<String>,
    /// Token budget for extended thinking. `None` disables thinking.
    pub reasoning_budget: Option<u32>,
    /// Maximum number of retries for a request that fails in a retryable way
    pub max_retries: usize,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            max_tokens: DEFAULT_OPENROUTER_MAX_TOKENS,
            model: DEFAULT_OPENROUTER_MODEL.into(),
            reasoning_effort: None,
            reasoning_budget: None,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
}

/// Configuration for the mock LLM provider.
#[cfg(any(test, feature = "mock"))]
#[derive(Debug, Clone)]
pub struct MockConfig {
    /// Canned responses returned in order, one per request.
    pub responses: Vec<String>,
}

/// Configuration for LLM clients
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LLMClientConfig {
    /// Anthropic client configuration
    Anthropic(crate::config::AnthropicConfig),
    /// Ollama client configuration
    Ollama(crate::config::OllamaConfig),
    /// OpenAI client configuration
    OpenAI(crate::config::OpenAIConfig),
    /// OpenRouter client configuration
    OpenRouter(crate::config::OpenRouterConfig),
    /// Gemini client configuration
    Gemini(crate::config::GeminiConfig),
    /// Mock client configuration (enabled by the `mock` feature)
    #[cfg(any(test, feature = "mock"))]
    Mock(crate::config::MockConfig),
}

impl LLMClientConfig {
    /// Return the canonical provider ID
    #[must_use]
    pub const fn provider_id(&self) -> ProviderId {
        match self {
            Self::Anthropic(_) => ProviderId::Anthropic,
            Self::Ollama(_) => ProviderId::Ollama,
            Self::OpenAI(_) => ProviderId::OpenAI,
            Self::OpenRouter(_) => ProviderId::OpenRouter,
            Self::Gemini(_) => ProviderId::Gemini,
            #[cfg(any(test, feature = "mock"))]
            Self::Mock(_) => ProviderId::Mock,
        }
    }

    /// Get the model name for the client
    #[must_use]
    pub fn model_name(&self) -> &str {
        match self {
            Self::Anthropic(cfg) => &cfg.model,
            Self::Ollama(cfg) => &cfg.model,
            Self::OpenAI(cfg) => &cfg.model,
            Self::OpenRouter(cfg) => &cfg.model,
            Self::Gemini(cfg) => &cfg.model,
            #[cfg(any(test, feature = "mock"))]
            Self::Mock(_) => "",
        }
    }
}

/// Provider configs for tests that call live APIs, with API keys read from the environment.
///
/// The crate never reads the environment itself; callers pass configs explicitly. Live tests still
/// need real keys (and, for Ollama, the server URL), so these helpers read them (typically loaded
/// from `.env` by `dotenv`) and fill every other field with the same defaults that `zqa` uses. A
/// missing key is left empty, so the test fails on the provider's authentication error.
#[cfg(test)]
pub(crate) mod test_env {
    use super::{
        AnthropicConfig, CohereConfig, GeminiConfig, OllamaConfig, OpenAIConfig, OpenRouterConfig,
        VoyageAIConfig,
    };
    use crate::constants::{
        DEFAULT_COHERE_EMBEDDING_DIM, DEFAULT_COHERE_EMBEDDING_MODEL, DEFAULT_COHERE_RERANK_MODEL,
        DEFAULT_MAX_CONCURRENT_REQUESTS, DEFAULT_MAX_RETRIES, DEFAULT_VOYAGE_EMBEDDING_DIM,
        DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };

    fn api_key(var: &str) -> String {
        std::env::var(var).unwrap_or_default()
    }

    /// An Anthropic config using `ANTHROPIC_API_KEY`.
    pub(crate) fn anthropic() -> AnthropicConfig {
        AnthropicConfig {
            api_key: api_key("ANTHROPIC_API_KEY"),
            ..AnthropicConfig::default()
        }
    }

    /// An OpenAI config using `OPENAI_API_KEY`.
    pub(crate) fn openai() -> OpenAIConfig {
        OpenAIConfig {
            api_key: api_key("OPENAI_API_KEY"),
            ..OpenAIConfig::default()
        }
    }

    /// A Gemini config using `GEMINI_API_KEY`, or `GOOGLE_API_KEY` if that is not set.
    pub(crate) fn gemini() -> GeminiConfig {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .unwrap_or_default();

        GeminiConfig {
            api_key,
            ..GeminiConfig::default()
        }
    }

    /// An Ollama config using `OLLAMA_BASE_URL` and `OLLAMA_MODEL` where they are set, and the
    /// defaults otherwise. Contributors often run a server elsewhere or have only one model pulled.
    pub(crate) fn ollama() -> OllamaConfig {
        let mut config = OllamaConfig::default();
        if let Ok(base_url) = std::env::var("OLLAMA_BASE_URL") {
            config.base_url = base_url;
        }

        if let Ok(model) = std::env::var("OLLAMA_MODEL") {
            config.model = model;
        }

        config
    }

    /// An OpenRouter config using `OPENROUTER_API_KEY`.
    pub(crate) fn openrouter() -> OpenRouterConfig {
        OpenRouterConfig {
            api_key: api_key("OPENROUTER_API_KEY"),
            ..OpenRouterConfig::default()
        }
    }

    /// A Voyage AI config using `VOYAGE_AI_API_KEY`.
    pub(crate) fn voyage() -> VoyageAIConfig {
        VoyageAIConfig {
            api_key: api_key("VOYAGE_AI_API_KEY"),
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// A Cohere config using `COHERE_API_KEY`.
    pub(crate) fn cohere() -> CohereConfig {
        CohereConfig {
            api_key: api_key("COHERE_API_KEY"),
            embedding_model: DEFAULT_COHERE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_COHERE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_COHERE_RERANK_MODEL.into(),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
}
