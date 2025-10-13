use rag::constants::{
    DEFAULT_ANTHROPIC_MAX_TOKENS, DEFAULT_MAX_CONCURRENT_REQUESTS, DEFAULT_MAX_RETRIES,
};
use rag::constants::{DEFAULT_OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIM};
use serde::{Deserialize, Serialize};
use std::{env, num::ParseIntError, path::Path};
use thiserror::Error;

/// TOML config. Below is an example config with all the defaults. The TOML config is
/// overridden by environment variables.
///
/// ```toml
/// model_provider = "anthropic"  # Generation model provider
/// embedding_provider = "voyageai"  # Embedding/reranker model provider
/// reranker_provider = "voyageai"  # Usually this will be the same as your `embedding_provider`
/// max_concurrent_requests = 5  # Max concurrent embedding requests
/// max_retries = 3  # Max retries when network requests fail
///
/// # `log_level` is a CLI-only arg so it isn't applied inadvertently.
///
/// # Provider-specific configs. This allows you to merely change the `model_provider`
/// # above and have the settings for that provider applied.
/// [anthropic]
/// model = "claude-sonnet-4-5"
/// api_key = "sk-ant-..."
/// max_tokens = 64000
///
/// [openai]
/// model = "gpt-5"
/// api_key = "sk-proj-..."
/// max_tokens = 8192
/// embedding_model = "text-embedding-3-small"
/// embedding_dims = 1536
///
/// [gemini]
/// model = "gemini-2.5-pro"
/// api_key = "AI..."
/// embedding_model = "gemini-embedding-001"
/// embedding_dims = 3072
///
/// [voyageai]
/// reranker = "rerank-2.5"
/// embedding_model = "voyage-3-large"
/// embedding_dims = 2048
///
/// [cohere]
/// reranker = "rerank-v3.5"
/// embedding_model = "embed-v4.0"
/// embedding_dims = 1536
///
/// [openrouter]
/// api_key = "..."
/// model = "anthropic/claude-sonnet-4.5"
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Generation model provider (anthropic, openai, gemini, openrouter)
    #[serde(default = "default_model_provider")]
    pub model_provider: String,

    /// Embedding provider (anthropic, openai, voyageai, gemini, cohere)
    #[serde(default = "default_embedding_provider")]
    pub embedding_provider: String,

    /// Reranker provider (voyageai, cohere)
    #[serde(default = "default_reranker_provider")]
    pub reranker_provider: String,

    /// Maximum number of concurrent embedding requests
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,

    /// Maximum number of retries for failed network requests
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,

    /// Anthropic-specific configuration
    #[serde(default)]
    pub anthropic: Option<AnthropicConfig>,

    /// OpenAI-specific configuration
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,

    /// Gemini-specific configuration
    #[serde(default)]
    pub gemini: Option<GeminiConfig>,

    /// Voyage AI-specific configuration
    #[serde(default)]
    pub voyageai: Option<VoyageAIConfig>,

    /// Cohere-specific configuration
    #[serde(default)]
    pub cohere: Option<CohereConfig>,

    /// OpenRouter-specific configuration
    #[serde(default)]
    pub openrouter: Option<OpenRouterConfig>,
}

/// Errors reading or parsing TOML
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse TOML: {0}")]
    Parse(#[from] toml::de::Error),

    #[error("Failed to parse value: {0}")]
    ParseField(#[from] ParseIntError),
}

/// Anthropic provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicConfig {
    /// Model name (e.g., "claude-sonnet-4-5")
    pub model: String,

    /// API key
    pub api_key: Option<String>,

    /// Maximum tokens for generation
    #[serde(default = "default_anthropic_max_tokens")]
    pub max_tokens: u32,
}

/// OpenAI provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIConfig {
    /// Model name (e.g., "gpt-5")
    pub model: String,

    /// API key
    pub api_key: Option<String>,

    /// Maximum tokens for generation
    #[serde(default = "default_openai_max_tokens")]
    pub max_tokens: u32,

    /// Embedding model name
    pub embedding_model: Option<String>,

    /// Embedding dimensions
    pub embedding_dims: Option<usize>,
}

/// Gemini provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GeminiConfig {
    /// Model name (e.g., "gemini-2.5-pro")
    pub model: String,

    /// API key
    pub api_key: Option<String>,

    /// Embedding model name
    pub embedding_model: Option<String>,

    /// Embedding dimensions
    pub embedding_dims: Option<usize>,
}

/// Voyage AI provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VoyageAIConfig {
    /// Reranker model name
    pub reranker: Option<String>,

    /// Embedding model name
    pub embedding_model: Option<String>,

    /// Embedding dimensions
    pub embedding_dims: Option<usize>,

    /// API key
    pub api_key: Option<String>,
}

/// Cohere provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CohereConfig {
    /// Reranker model name
    pub reranker: Option<String>,

    /// Embedding model name
    pub embedding_model: Option<String>,

    /// Embedding dimensions
    pub embedding_dims: Option<usize>,

    /// API key
    pub api_key: Option<String>,
}

/// OpenRouter configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenRouterConfig {
    /// Generation model
    pub model: Option<String>,

    /// API key
    pub api_key: Option<String>,
}

// Default value functions
fn default_model_provider() -> String {
    "anthropic".to_string()
}

fn default_embedding_provider() -> String {
    "voyageai".to_string()
}

fn default_reranker_provider() -> String {
    "voyageai".to_string()
}

fn default_max_concurrent_requests() -> usize {
    DEFAULT_MAX_CONCURRENT_REQUESTS
}

fn default_max_retries() -> usize {
    DEFAULT_MAX_RETRIES
}

fn default_anthropic_max_tokens() -> u32 {
    DEFAULT_ANTHROPIC_MAX_TOKENS
}

fn default_openai_max_tokens() -> u32 {
    8192
}

/// An extension trait for strings to be updated with a value from the environment.
trait OverwriteFromEnv {
    fn replace_with_env(&mut self, _var: &str)
    where
        Self: Sized,
    {
    }
}

impl OverwriteFromEnv for String {
    fn replace_with_env(&mut self, var: &str) {
        if let Ok(env_var) = env::var(var) {
            *self = env_var;
        }
    }
}

impl OverwriteFromEnv for Option<String> {
    fn replace_with_env(&mut self, var: &str) {
        if let Ok(env_var) = env::var(var) {
            *self = Some(env_var);
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Overwrite config from environment variables (higher priority)
    pub fn read_env(&mut self) -> Result<(), ConfigError> {
        // Main options
        // The model providers are not exposed as env options.
        if let Ok(max_concurrent_requests) = env::var("MAX_CONCURRENT_REQUESTS") {
            self.max_concurrent_requests = max_concurrent_requests.parse()?;
        }

        // Anthropic options
        if let Some(anthropic_config) = &mut self.anthropic {
            // The max tokens is not exposed as an env option.
            anthropic_config.model.replace_with_env("ANTHROPIC_MODEL");
            anthropic_config
                .api_key
                .replace_with_env("ANTHROPIC_API_KEY");
        }

        // OpenAI options
        if let Some(openai_config) = &mut self.openai {
            // The max tokens and embedding dims are not exposed as env options.
            openai_config.model.replace_with_env("OPENAI_MODEL");
            openai_config.api_key.replace_with_env("OPENAI_API_KEY");
            openai_config
                .embedding_model
                .replace_with_env("OPENAI_EMBEDDING_MODEL");
        }

        // Gemini options
        if let Some(gemini_config) = &mut self.gemini {
            // embedding_dims is not exposed as an env option
            gemini_config.model.replace_with_env("GEMINI_MODEL");
            // GEMINI_API_KEY has priority over GOOGLE_API_KEY
            gemini_config.api_key.replace_with_env("GOOGLE_API_KEY");
            gemini_config.api_key.replace_with_env("GEMINI_API_KEY");
            gemini_config
                .embedding_model
                .replace_with_env("GEMINI_EMBEDDING_MODEL");
        }

        // VoyageAI options
        if let Some(voyage_config) = &mut self.voyageai {
            // embedding_dims is not exposed as an env option
            voyage_config
                .reranker
                .replace_with_env("VOYAGE_AI_RERANKER");
            voyage_config
                .embedding_model
                .replace_with_env("VOYAGE_AI_MODEL");
        }

        // Cohere options
        if let Some(cohere_config) = &mut self.cohere {
            // embedding_dims is not exposed as an env option
            cohere_config.reranker.replace_with_env("COHERE_RERANKER");
            cohere_config
                .embedding_model
                .replace_with_env("COHERE_MODEL");
        }

        // OpenRouter options
        if let Some(openrouter_config) = &mut self.openrouter {
            openrouter_config.model.replace_with_env("OPENROUTER_MODEL");
            openrouter_config
                .api_key
                .replace_with_env("OPENROUTER_API_KEY");
        }

        Ok(())
    }

    /// Create a default configuration
    pub fn default() -> Self {
        Self {
            model_provider: default_model_provider(),
            embedding_provider: default_embedding_provider(),
            reranker_provider: default_reranker_provider(),
            max_concurrent_requests: default_max_concurrent_requests(),
            max_retries: default_max_retries(),
            anthropic: None,
            openai: None,
            gemini: None,
            voyageai: None,
            cohere: None,
            openrouter: None,
        }
    }
}

// Convert zqa configs to rag configs using From trait
impl From<AnthropicConfig> for rag::config::AnthropicConfig {
    fn from(config: AnthropicConfig) -> Self {
        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("ANTHROPIC_API_KEY").ok())
                .expect("ANTHROPIC_API_KEY must be set"),
            model: config.model,
            max_tokens: config.max_tokens,
        }
    }
}

impl From<OpenAIConfig> for rag::config::OpenAIConfig {
    fn from(config: OpenAIConfig) -> Self {
        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("OPENAI_API_KEY").ok())
                .expect("OPENAI_API_KEY must be set"),
            model: config.model,
            max_tokens: config.max_tokens,
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| DEFAULT_OPENAI_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(OPENAI_EMBEDDING_DIM as usize),
        }
    }
}

impl From<GeminiConfig> for rag::config::GeminiConfig {
    fn from(config: GeminiConfig) -> Self {
        use rag::constants::{DEFAULT_GEMINI_EMBEDDING_MODEL, GEMINI_EMBEDDING_DIM};

        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("GEMINI_API_KEY").ok())
                .or_else(|| env::var("GOOGLE_API_KEY").ok())
                .expect("GEMINI_API_KEY or GOOGLE_API_KEY must be set"),
            model: config.model,
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| DEFAULT_GEMINI_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(GEMINI_EMBEDDING_DIM as usize),
        }
    }
}

impl From<VoyageAIConfig> for rag::config::VoyageAIConfig {
    fn from(config: VoyageAIConfig) -> Self {
        use rag::constants::{
            DEFAULT_VOYAGE_RERANK_MODEL, VOYAGE_EMBEDDING_DIM, VOYAGE_EMBEDDING_MODEL,
        };

        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("VOYAGE_AI_API_KEY").ok())
                .expect("VOYAGE_AI_API_KEY must be set"),
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| VOYAGE_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(VOYAGE_EMBEDDING_DIM as usize),
            reranker: config
                .reranker
                .unwrap_or_else(|| DEFAULT_VOYAGE_RERANK_MODEL.to_string()),
        }
    }
}

impl From<CohereConfig> for rag::config::CohereConfig {
    fn from(config: CohereConfig) -> Self {
        use rag::constants::{
            COHERE_EMBEDDING_DIM, COHERE_EMBEDDING_MODEL, DEFAULT_COHERE_RERANK_MODEL,
        };

        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("COHERE_API_KEY").ok())
                .expect("COHERE_API_KEY must be set"),
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| COHERE_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(COHERE_EMBEDDING_DIM as usize),
            reranker: config
                .reranker
                .unwrap_or_else(|| DEFAULT_COHERE_RERANK_MODEL.to_string()),
        }
    }
}

impl From<OpenRouterConfig> for rag::config::OpenRouterConfig {
    fn from(config: OpenRouterConfig) -> Self {
        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("OPENROUTER_API_KEY").ok())
                .expect("OPENROUTER_API_KEY must be set"),
            model: config
                .model
                .unwrap_or_else(|| "anthropic/claude-sonnet-4.5".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_config() {
        let toml_str = r#"
            model_provider = "anthropic"
            embedding_provider = "voyageai"
            reranker_provider = "voyageai"
            max_concurrent_requests = 5
            max_retries = 3

            [anthropic]
            model = "claude-sonnet-4-5"
            api_key = "sk-ant-test"
            max_tokens = 64000

            [openai]
            model = "gpt-5"
            api_key = "sk-proj-test"
            max_tokens = 8192
            embedding_model = "text-embedding-3-small"
            embedding_dims = 1536

            [voyageai]
            reranker = "rerank-2.5"
            embedding_model = "voyage-3-large"
            embedding_dims = 2048
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.model_provider, "anthropic");
        assert_eq!(config.embedding_provider, "voyageai");
        assert_eq!(config.max_concurrent_requests, 5);
        assert_eq!(config.max_retries, 3);

        let anthropic = config.anthropic.unwrap();
        assert_eq!(anthropic.model, "claude-sonnet-4-5");
        assert_eq!(anthropic.max_tokens, 64000);

        let voyageai = config.voyageai.unwrap();
        assert_eq!(voyageai.reranker.unwrap(), "rerank-2.5");
        assert_eq!(voyageai.embedding_model.unwrap(), "voyage-3-large");
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml_str = r#"
            model_provider = "openai"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.model_provider, "openai");
        assert_eq!(config.embedding_provider, "voyageai"); // default
        assert_eq!(config.max_concurrent_requests, 5); // default
    }
}
