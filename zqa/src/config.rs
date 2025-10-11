use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// TOML config. Below is an example config with all the defaults. The TOML config is
/// overridden by CLI args. In order, the priority is TOML < env < CLI args.
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
}

/// Errors reading or parsing TOML
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse TOML: {0}")]
    Parse(#[from] toml::de::Error),
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
    5
}

fn default_max_retries() -> usize {
    3
}

fn default_anthropic_max_tokens() -> u32 {
    64000
}

fn default_openai_max_tokens() -> u32 {
    8192
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
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
