//! Constants used throughout the RAG system

/// Default maximum concurrent requests for embedding processing
pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 5;

/// Maximum number of retries for API requests with backoff
pub const DEFAULT_MAX_RETRIES: usize = 3;

/// Default OpenAI model for chat completions
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-5.2-2025-12-11";

/// Default OpenAI generation max tokens
pub const DEFAULT_OPENAI_MAX_TOKENS: u32 = 8192;

/// OpenAI text-embedding-3-small dimension
pub const DEFAULT_OPENAI_EMBEDDING_DIM: u32 = 1536;

/// Default OpenAI embedding model
pub const DEFAULT_OPENAI_EMBEDDING_MODEL: &str = "text-embedding-3-large";

/// Default Anthropic model for chat completions
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-5";

/// Default maximum tokens for Anthropic requests
pub const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 64000;

/// Default Gemini model for chat completions
pub const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-pro";

/// Gemini gemini-embedding-001 dimension
pub const DEFAULT_GEMINI_EMBEDDING_DIM: u32 = 3072;

/// Default Gemini embedding model
pub const DEFAULT_GEMINI_EMBEDDING_MODEL: &str = "gemini-embedding-001";

/// Default OpenRouter model
pub const DEFAULT_OPENROUTER_MODEL: &str = "anthropic/claude-sonnet-4.5";

/// Default Cohere dimensions
pub const DEFAULT_COHERE_EMBEDDING_DIM: u32 = 1536;

/// Default Cohere embedding model
pub const DEFAULT_COHERE_EMBEDDING_MODEL: &str = "embed-v4.0";

/// Default Cohere rerank model
pub const DEFAULT_COHERE_RERANK_MODEL: &str = "rerank-v3.5";

/// Default Voyage AI rerank model
pub const DEFAULT_VOYAGE_RERANK_MODEL: &str = "rerank-2.5";

/// Default Voyage AI embedding model
pub const DEFAULT_VOYAGE_EMBEDDING_MODEL: &str = "voyage-3-large";

/// Default Voyage AI embedding dimension
pub const DEFAULT_VOYAGE_EMBEDDING_DIM: u32 = 2048;
