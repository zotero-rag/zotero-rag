//! Constants used throughout the RAG system

/// Default OpenAI model for chat completions
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-2025-04-14";

/// Default Anthropic model for chat completions  
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-20250514";

/// Default Gemini model for chat completions
pub const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-pro";

/// Default maximum tokens for Anthropic requests
pub const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 8192;

/// Default maximum concurrent requests for embedding processing
pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 5;

/// Maximum number of retries for API requests with backoff
pub const DEFAULT_MAX_RETRIES: usize = 3;

/// OpenAI text-embedding-3-small dimension
pub const OPENAI_EMBEDDING_DIM: u32 = 1536;

/// VoyageAI voyage-3-large dimension
pub const VOYAGE_EMBEDDING_DIM: u32 = 2048;

/// Gemini gemini-embedding-001 dimension
pub const GEMINI_EMBEDDING_DIM: u32 = 3072;

/// Default VoyageAI embedding model
pub const VOYAGE_EMBEDDING_MODEL: &str = "voyage-3-large";

/// Default Cohere embedding model
pub const COHERE_EMBEDDING_MODEL: &str = "embed-v4.0";

/// Default Cohere dimensions
pub const COHERE_EMBEDDING_DIM: u32 = 1536;

/// Default OpenAI embedding model
pub const DEFAULT_OPENAI_EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// Default Gemini embedding model
pub const DEFAULT_GEMINI_EMBEDDING_MODEL: &str = "gemini-embedding-001";

/// Default Cohere rerank model
pub const DEFAULT_COHERE_RERANK_MODEL: &str = "rerank-v3.5";

/// Default Voyage AI rerank model
pub const DEFAULT_VOYAGE_RERANK_MODEL: &str = "rerank-2.5";
