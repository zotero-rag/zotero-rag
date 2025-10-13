//! Configuration structures for RAG providers
//!
//! This module provides configuration structs that can be passed to clients
//! instead of reading from environment variables or TOML files directly.
//! This makes the rag crate more general and reusable.

/// Configuration for Anthropic LLM provider
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for Anthropic
    pub api_key: String,
    /// Model name (e.g., "claude-sonnet-4-5")
    pub model: String,
    /// Maximum tokens for generation
    pub max_tokens: u32,
}

/// Configuration for OpenAI LLM and embedding provider
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API key for OpenAI
    pub api_key: String,
    /// Model name for generation (e.g., "gpt-4.1-2025-04-14")
    pub model: String,
    /// Maximum tokens for generation
    pub max_tokens: u32,
    /// Embedding model name
    pub embedding_model: String,
    /// Embedding dimensions
    pub embedding_dims: usize,
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

/// Configuration for OpenRouter provider
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    /// API key for OpenRouter
    pub api_key: String,
    /// Model name (e.g., "anthropic/claude-sonnet-4.5")
    pub model: String,
}
