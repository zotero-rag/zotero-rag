//! Constants used throughout the RAG system

/// Default maximum concurrent requests for embedding processing
pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 1;

/// Maximum number of retries for API requests with backoff
pub const DEFAULT_MAX_RETRIES: usize = 3;

/// Default `ollama` model for text generation.
pub const DEFAULT_OLLAMA_MODEL: &str = "qwen3.5:latest";

/// Default small `ollama` model for text generation
pub const DEFAULT_OLLAMA_MODEL_SMALL: &str = "qwen3.5:0.8b";

/// Default `ollama` generation max tokens
pub const DEFAULT_OLLAMA_MAX_TOKENS: u32 = 8192;

/// Default `ollama` embedding model.
pub const DEFAULT_OLLAMA_EMBEDDING_MODEL: &str = "qwen3-embedding";

/// Default embedding dimension for `qwen3-embedding`.
pub const DEFAULT_OLLAMA_EMBEDDING_DIM: usize = 4096;

/// Default base URL for the `ollama` API.
pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";

/// Default OpenAI model for chat completions
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4";

/// Default OpenAI model for conversation title generation
pub const DEFAULT_OPENAI_MODEL_SMALL: &str = "gpt-5.4-mini";

/// Default OpenAI generation max tokens
pub const DEFAULT_OPENAI_MAX_TOKENS: u32 = 8192;

/// OpenAI text-embedding-3-small dimension
pub const DEFAULT_OPENAI_EMBEDDING_DIM: u32 = 1536;

/// Default OpenAI embedding model
pub const DEFAULT_OPENAI_EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// Default Anthropic model for chat completions
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-6";

/// Default Anthropic model for conversation title generation
pub const DEFAULT_ANTHROPIC_MODEL_SMALL: &str = "claude-haiku-4-5";

/// Default maximum tokens for Anthropic requests
pub const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 64000;

/// Default Gemini model for chat completions
pub const DEFAULT_GEMINI_MODEL: &str = "gemini-3.1-pro-preview";

/// Default Gemini model for conversation title generation
pub const DEFAULT_GEMINI_MODEL_SMALL: &str = "gemini-3.1-flash-lite-preview";

/// Gemini gemini-embedding-2-preview dimension
pub const DEFAULT_GEMINI_EMBEDDING_DIM: u32 = 3072;

/// Default Gemini embedding model
pub const DEFAULT_GEMINI_EMBEDDING_MODEL: &str = "gemini-embedding-2-preview";

/// Default OpenRouter model
pub const DEFAULT_OPENROUTER_MODEL: &str = "anthropic/claude-sonnet-4.6";

/// Default OpenRouter model for conversation title generation
pub const DEFAULT_OPENROUTER_MODEL_SMALL: &str = "anthropic/claude-haiku-4.5";

/// Default Cohere dimensions
pub const DEFAULT_COHERE_EMBEDDING_DIM: u32 = 1536;

/// Default Cohere embedding model
pub const DEFAULT_COHERE_EMBEDDING_MODEL: &str = "embed-v4.0";

/// Default Cohere rerank model
pub const DEFAULT_COHERE_RERANK_MODEL: &str = "rerank-v4.0-pro";

/// Default Voyage AI rerank model
pub const DEFAULT_VOYAGE_RERANK_MODEL: &str = "rerank-2.5";

/// Default Voyage AI embedding model
pub const DEFAULT_VOYAGE_EMBEDDING_MODEL: &str = "voyage-4-large";

/// Default Voyage AI embedding dimension
pub const DEFAULT_VOYAGE_EMBEDDING_DIM: u32 = 2048;

/// Default ZeroEntropy embedding model
pub const DEFAULT_ZEROENTROPY_EMBEDDING_MODEL: &str = "zembed-1";

/// Default ZeroEntropy embedding dimension
pub const DEFAULT_ZEROENTROPY_EMBEDDING_DIM: u32 = 2560;

/// Default ZeroEntropy rerank model
pub const DEFAULT_ZEROENTROPY_RERANK_MODEL: &str = "zerank-2";
