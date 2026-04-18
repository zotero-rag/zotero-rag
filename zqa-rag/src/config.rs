//! Configuration structures for RAG providers
//!
//! This module provides configuration structs that can be passed to clients
//! instead of reading from environment variables or TOML files directly.
//! This makes the rag crate more general and reusable.

use crate::{
    constants::{
        DEFAULT_ANTHROPIC_MAX_TOKENS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL,
        DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_EMBEDDING_DIM,
        DEFAULT_OLLAMA_EMBEDDING_MODEL, DEFAULT_OLLAMA_MAX_TOKENS, DEFAULT_OLLAMA_MODEL,
        DEFAULT_OPENAI_EMBEDDING_DIM, DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_OPENAI_MAX_TOKENS,
        DEFAULT_OPENAI_MODEL, DEFAULT_OPENROUTER_MODEL,
    },
    providers::ProviderId,
};

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
    pub reasoning_budget: Option<u32>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: DEFAULT_ANTHROPIC_MODEL.into(),
            max_tokens: DEFAULT_ANTHROPIC_MAX_TOKENS,
            reasoning_budget: None,
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
    /// Token budget for extended thinking. `None` disables thinking.
    pub reasoning_budget: Option<u32>,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: DEFAULT_GEMINI_MODEL.into(),
            embedding_model: DEFAULT_GEMINI_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_OLLAMA_EMBEDDING_DIM,
            reasoning_budget: None,
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
}

/// Configuration for ZeroEntropy embedding and reranking provider
#[derive(Debug, Clone)]
pub struct ZeroEntropyConfig {
    /// API key for ZeroEntropy
    pub api_key: String,
    /// Embedding model name
    pub embedding_model: String,
    /// Embedding dimensions (one of 2560, 1280, 640, 320, 160, 80, or 40)
    pub embedding_dims: usize,
    /// Reranker model name
    pub reranker: String,
}

/// Configuration for OpenRouter provider
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    /// API key for OpenRouter
    pub api_key: String,
    /// Model name (e.g., "anthropic/claude-sonnet-4.5")
    pub model: String,
    /// Reasoning effort level (e.g., "high"). `None` disables reasoning.
    pub reasoning_effort: Option<String>,
    /// Token budget for extended thinking. `None` disables thinking.
    pub reasoning_budget: Option<u32>,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: DEFAULT_OPENROUTER_MODEL.into(),
            reasoning_effort: None,
            reasoning_budget: None,
        }
    }
}

/// Configuration for LLM clients
#[derive(Debug, Clone)]
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
        }
    }
}
