//! Constants used throughout the RAG system

/// Default OpenAI model for chat completions
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-2025-04-14";

/// Default Anthropic model for chat completions  
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-20250514";

/// Default maximum tokens for Anthropic requests
pub const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 8192;

/// Default maximum concurrent requests for embedding processing
pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 5;

/// Maximum number of retries for API requests with backoff
pub const DEFAULT_MAX_RETRIES: usize = 3;

/// OpenAI text-embedding-3-small dimension
pub const OPENAI_EMBEDDING_DIM: u32 = 1536;

/// Default OpenAI embedding model
pub const DEFAULT_OPENAI_EMBEDDING_MODEL: &str = "text-embedding-3-small";

