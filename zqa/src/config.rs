use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::path::PathBuf;
use std::{env, num::ParseIntError, path::Path};
use thiserror;
use thiserror::Error;
use zqa_rag::config::LLMClientConfig;
#[allow(clippy::wildcard_imports)]
use zqa_rag::constants::*;
use zqa_rag::constants::{DEFAULT_OPENAI_EMBEDDING_DIM, DEFAULT_OPENAI_EMBEDDING_MODEL};
use zqa_rag::embedding::common::EmbeddingProviderConfig;

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
/// model_small = "claude-haiku-4-5"
/// api_key = "sk-ant-..."
/// max_tokens = 64000
///
/// [ollama]
/// model = "qwen3.5"  # Defaults to the 9B version
/// model_small = "qwen3.5:0.8b"
/// max_tokens = 8192
/// embedding_model = "qwen3-embedding"
/// embedding_dims = 4096
/// base_url = "http://localhost:11434"  # Defaults to local ollama instance
///
/// [openai]
/// model = "gpt-5.2"
/// model_small = "gpt-5-mini"
/// api_key = "sk-proj-..."
/// max_tokens = 8192
/// embedding_model = "text-embedding-3-small"
/// embedding_dims = 1536
///
/// [gemini]
/// model = "gemini-3.1-pro-preview"
/// model_small = "gemini-3-flash-preview"
/// api_key = "AI..."
/// embedding_model = "gemini-embedding-001"
/// embedding_dims = 3072
///
/// [voyageai]
/// reranker = "rerank-2.5"
/// embedding_model = "voyage-3-large"
/// embedding_dims = 2048
///api_key = "..."
///
/// [cohere]
/// reranker = "rerank-v3.5"
/// embedding_model = "embed-v4.0"
/// embedding_dims = 1536
/// api_key = "..."
///
/// [openrouter]
/// api_key = "..."
/// model = "anthropic/claude-sonnet-4.5"
/// model_small = "anthropic/claude-haiku-4.5"
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

    /// Ollama-specific configuration
    #[serde(default)]
    pub ollama: Option<OllamaConfig>,

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

impl Config {
    /// Load configuration from a TOML file
    ///
    /// # Errors
    ///
    /// * `ConfigError::ParseError` - If TOML parsing fails.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Overwrite config from environment variables (higher priority)
    ///
    /// # Errors
    ///
    /// * `ConfigError::ParseFieldError` - If parsing a field as an integer fails.
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
                .model_small
                .replace_with_env("ANTHROPIC_MODEL_SMALL");
            anthropic_config
                .api_key
                .replace_with_env("ANTHROPIC_API_KEY");
        }

        // Ollama options
        if let Some(ollama_config) = &mut self.ollama {
            ollama_config.model.replace_with_env("OLLAMA_MODEL");
            ollama_config
                .model_small
                .replace_with_env("OLLAMA_MODEL_SMALL");
            ollama_config.base_url.replace_with_env("OLLAMA_BASE_URL");
        }

        // OpenAI options
        if let Some(openai_config) = &mut self.openai {
            // The max tokens and embedding dims are not exposed as env options.
            openai_config.model.replace_with_env("OPENAI_MODEL");
            openai_config
                .model_small
                .replace_with_env("OPENAI_MODEL_SMALL");
            openai_config.api_key.replace_with_env("OPENAI_API_KEY");
            openai_config
                .embedding_model
                .replace_with_env("OPENAI_EMBEDDING_MODEL");
        }

        // Gemini options
        if let Some(gemini_config) = &mut self.gemini {
            // embedding_dims is not exposed as an env option
            gemini_config.model.replace_with_env("GEMINI_MODEL");
            gemini_config
                .model_small
                .replace_with_env("GEMINI_MODEL_SMALL");
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
            voyage_config.api_key.replace_with_env("VOYAGE_AI_API_KEY");
        }

        // Cohere options
        if let Some(cohere_config) = &mut self.cohere {
            // embedding_dims is not exposed as an env option
            cohere_config.reranker.replace_with_env("COHERE_RERANKER");
            cohere_config
                .embedding_model
                .replace_with_env("COHERE_MODEL");
            cohere_config.api_key.replace_with_env("COHERE_API_KEY");
        }

        // OpenRouter options
        if let Some(openrouter_config) = &mut self.openrouter {
            openrouter_config.model.replace_with_env("OPENROUTER_MODEL");
            openrouter_config
                .model_small
                .replace_with_env("OPENROUTER_MODEL_SMALL");
            openrouter_config
                .api_key
                .replace_with_env("OPENROUTER_API_KEY");
        }

        Ok(())
    }

    /// Get the embedding configuration based on the `embedding_provider` value.
    #[must_use]
    pub fn get_embedding_config(&self) -> Option<EmbeddingProviderConfig> {
        match self.embedding_provider.as_str() {
            "openai" => self
                .openai
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::OpenAI(cfg.clone().into())),
            "gemini" => self
                .gemini
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::Gemini(cfg.clone().into())),
            "voyageai" => self
                .voyageai
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::VoyageAI(cfg.clone().into())),
            "cohere" => self
                .cohere
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::Cohere(cfg.clone().into())),
            _ => None,
        }
    }

    #[must_use]
    pub fn get_reranker_config(&self) -> Option<EmbeddingProviderConfig> {
        match self.reranker_provider.as_str() {
            "voyageai" => self
                .voyageai
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::VoyageAI(cfg.clone().into())),
            "cohere" => self
                .cohere
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::Cohere(cfg.clone().into())),
            _ => None,
        }
    }

    /// Get the LLM client configuration using the small model variant. Falls back to the regular
    /// model if no small model is configured.
    #[must_use]
    pub fn get_small_model_config(&self) -> Option<LLMClientConfig> {
        let mut client_config = self.get_generation_config()?;

        let small_model_name = match self.model_provider.as_str() {
            "anthropic" => self.anthropic.as_ref().and_then(|c| c.model_small.as_ref()),
            "ollama" => self.ollama.as_ref().and_then(|c| c.model_small.as_ref()),
            "openai" => self.openai.as_ref().and_then(|c| c.model_small.as_ref()),
            "gemini" => self.gemini.as_ref().and_then(|c| c.model_small.as_ref()),
            "openrouter" => self
                .openrouter
                .as_ref()
                .and_then(|c| c.model_small.as_ref()),
            _ => None,
        };

        if let Some(small_model_name) = small_model_name {
            match &mut client_config {
                LLMClientConfig::Anthropic(c) => c.model.clone_from(small_model_name),
                LLMClientConfig::Ollama(c) => c.model.clone_from(small_model_name),
                LLMClientConfig::OpenAI(c) => c.model.clone_from(small_model_name),
                LLMClientConfig::Gemini(c) => c.model.clone_from(small_model_name),
                LLMClientConfig::OpenRouter(c) => c.model.clone_from(small_model_name),
            }
        }

        Some(client_config)
    }

    #[must_use]
    pub fn get_embedding_model_name(&self) -> Option<String> {
        match self.get_embedding_config() {
            None => None,
            Some(EmbeddingProviderConfig::OpenAI(cfg)) => Some(cfg.embedding_model.clone()),
            Some(EmbeddingProviderConfig::Gemini(cfg)) => Some(cfg.embedding_model.clone()),
            Some(EmbeddingProviderConfig::VoyageAI(cfg)) => Some(cfg.embedding_model.clone()),
            Some(EmbeddingProviderConfig::Cohere(cfg)) => Some(cfg.embedding_model.clone()),
        }
    }

    #[must_use]
    pub fn get_generation_model_name(&self) -> Option<String> {
        match self.get_generation_config() {
            None => None,
            Some(LLMClientConfig::Anthropic(cfg)) => Some(cfg.model.clone()),
            Some(LLMClientConfig::Ollama(cfg)) => Some(cfg.model.clone()),
            Some(LLMClientConfig::OpenAI(cfg)) => Some(cfg.model.clone()),
            Some(LLMClientConfig::Gemini(cfg)) => Some(cfg.model.clone()),
            Some(LLMClientConfig::OpenRouter(cfg)) => Some(cfg.model.clone()),
        }
    }

    #[must_use]
    pub fn get_generation_config(&self) -> Option<LLMClientConfig> {
        match self.model_provider.as_str() {
            "anthropic" => self
                .anthropic
                .as_ref()
                .map(|cfg| LLMClientConfig::Anthropic(cfg.clone().into())),
            "ollama" => self
                .ollama
                .as_ref()
                .map(|cfg| LLMClientConfig::Ollama(cfg.clone().into())),
            "openai" => self
                .openai
                .as_ref()
                .map(|cfg| LLMClientConfig::OpenAI(cfg.clone().into())),
            "gemini" => self
                .gemini
                .as_ref()
                .map(|cfg| LLMClientConfig::Gemini(cfg.clone().into())),
            "openrouter" => self
                .openrouter
                .as_ref()
                .map(|cfg| LLMClientConfig::OpenRouter(cfg.clone().into())),
            _ => None,
        }
    }
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
    pub model: Option<String>,

    /// Small model name, used for generating conversation titles
    pub model_small: Option<String>,

    /// API key
    pub api_key: Option<String>,

    /// Maximum tokens for generation
    #[serde(default = "default_anthropic_max_tokens")]
    pub max_tokens: u32,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            model: Some(DEFAULT_ANTHROPIC_MODEL.into()),
            model_small: Some(DEFAULT_ANTHROPIC_MODEL_SMALL.into()),
            max_tokens: DEFAULT_ANTHROPIC_MAX_TOKENS,
            api_key: None,
        }
    }
}

/// `ollama` provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaConfig {
    /// Model name (e.g., "qwen3.5")
    pub model: Option<String>,
    /// Small model name, used for generating conversation titles
    pub model_small: Option<String>,
    /// Maximum tokens for generation
    pub max_tokens: Option<u32>,
    /// Embedding model name (e.g., "qwen3-embedding")
    pub embedding_model: Option<String>,
    /// Embedding dimension size
    pub embedding_dims: Option<usize>,
    /// Base URL for the ollama API (e.g., "http://localhost:11434")
    pub base_url: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            model: Some(DEFAULT_OLLAMA_MODEL.into()),
            max_tokens: Some(DEFAULT_OLLAMA_MAX_TOKENS),
            model_small: Some(DEFAULT_OLLAMA_MODEL_SMALL.into()),
            embedding_model: Some(DEFAULT_OLLAMA_EMBEDDING_MODEL.into()),
            embedding_dims: Some(DEFAULT_OLLAMA_EMBEDDING_DIM),
            base_url: Some(DEFAULT_OLLAMA_BASE_URL.into()),
        }
    }
}

/// `OpenAI` provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIConfig {
    /// Model name (e.g., "gpt-5.2")
    pub model: Option<String>,

    /// Small model name, used for generating conversation titles
    pub model_small: Option<String>,

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

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            model: Some(DEFAULT_OPENAI_MODEL.into()),
            model_small: Some(DEFAULT_OPENAI_MODEL_SMALL.into()),
            max_tokens: DEFAULT_OPENAI_MAX_TOKENS,
            embedding_model: Some(DEFAULT_OPENAI_EMBEDDING_MODEL.into()),
            embedding_dims: Some(DEFAULT_OPENAI_EMBEDDING_DIM as usize),
            api_key: None,
        }
    }
}

/// Gemini provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GeminiConfig {
    /// Model name (e.g., "gemini-2.5-pro")
    pub model: Option<String>,

    /// Small model name, used for generating conversation titles
    pub model_small: Option<String>,

    /// API key
    pub api_key: Option<String>,

    /// Embedding model name
    pub embedding_model: Option<String>,

    /// Embedding dimensions
    pub embedding_dims: Option<usize>,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            model: Some(DEFAULT_GEMINI_MODEL.into()),
            model_small: Some(DEFAULT_GEMINI_MODEL_SMALL.into()),
            embedding_model: Some(DEFAULT_GEMINI_EMBEDDING_MODEL.into()),
            embedding_dims: Some(DEFAULT_GEMINI_EMBEDDING_DIM as usize),
            api_key: None,
        }
    }
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

impl Default for VoyageAIConfig {
    fn default() -> Self {
        Self {
            reranker: Some(DEFAULT_VOYAGE_RERANK_MODEL.into()),
            embedding_model: Some(DEFAULT_VOYAGE_EMBEDDING_MODEL.into()),
            embedding_dims: Some(DEFAULT_VOYAGE_EMBEDDING_DIM as usize),
            api_key: None,
        }
    }
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

impl Default for CohereConfig {
    fn default() -> Self {
        Self {
            reranker: Some(DEFAULT_COHERE_RERANK_MODEL.into()),
            embedding_model: Some(DEFAULT_COHERE_EMBEDDING_MODEL.into()),
            embedding_dims: Some(DEFAULT_COHERE_EMBEDDING_DIM as usize),
            api_key: None,
        }
    }
}

/// `OpenRouter` configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenRouterConfig {
    /// Generation model
    pub model: Option<String>,

    /// Small model name, used for generating conversation titles
    pub model_small: Option<String>,

    /// API key
    pub api_key: Option<String>,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            model: Some(DEFAULT_OPENROUTER_MODEL.into()),
            model_small: Some(DEFAULT_OPENROUTER_MODEL_SMALL.into()),
            api_key: None,
        }
    }
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
    DEFAULT_OPENAI_MAX_TOKENS
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

impl Default for Config {
    fn default() -> Self {
        Self {
            model_provider: default_model_provider(),
            embedding_provider: default_embedding_provider(),
            reranker_provider: default_reranker_provider(),
            max_concurrent_requests: default_max_concurrent_requests(),
            max_retries: default_max_retries(),
            anthropic: Some(AnthropicConfig::default()),
            ollama: Some(OllamaConfig::default()),
            openai: Some(OpenAIConfig::default()),
            gemini: Some(GeminiConfig::default()),
            voyageai: Some(VoyageAIConfig::default()),
            cohere: Some(CohereConfig::default()),
            openrouter: Some(OpenRouterConfig::default()),
        }
    }
}

// Convert zqa configs to rag configs using From trait
impl From<AnthropicConfig> for zqa_rag::config::AnthropicConfig {
    fn from(config: AnthropicConfig) -> Self {
        Self {
            api_key: config
                .api_key
                .expect("Anthropic API key not found. Please set it in your config file or as ANTHROPIC_API_KEY."),
            model: config.model.unwrap_or(DEFAULT_ANTHROPIC_MODEL.into()),
            max_tokens: config.max_tokens,
        }
    }
}

impl From<OllamaConfig> for zqa_rag::config::OllamaConfig {
    fn from(config: OllamaConfig) -> Self {
        Self {
            model: config.model.unwrap_or(DEFAULT_OLLAMA_MODEL.into()),
            max_tokens: config.max_tokens.unwrap_or(DEFAULT_OLLAMA_MAX_TOKENS),
            embedding_model: config
                .embedding_model
                .unwrap_or(DEFAULT_OLLAMA_EMBEDDING_MODEL.into()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(DEFAULT_OLLAMA_EMBEDDING_DIM),
            base_url: config.base_url.unwrap_or(DEFAULT_OLLAMA_BASE_URL.into()),
        }
    }
}

impl From<OpenAIConfig> for zqa_rag::config::OpenAIConfig {
    fn from(config: OpenAIConfig) -> Self {
        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("OPENAI_API_KEY").ok())
                .expect(
                "OpenAI API key not found. Please set it in your config file or as OPENAI_API_KEY.",
            ),
            model: config.model.unwrap_or(DEFAULT_OPENAI_MODEL.into()),
            max_tokens: config.max_tokens,
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| DEFAULT_OPENAI_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(DEFAULT_OPENAI_EMBEDDING_DIM as usize),
        }
    }
}

impl From<GeminiConfig> for zqa_rag::config::GeminiConfig {
    fn from(config: GeminiConfig) -> Self {
        use zqa_rag::constants::{DEFAULT_GEMINI_EMBEDDING_DIM, DEFAULT_GEMINI_EMBEDDING_MODEL};

        Self {
            api_key: config.api_key.expect(
                "Gemini API key not found. Please set it in your config file or as GEMINI_API_KEY.",
            ),
            model: config.model.unwrap_or(DEFAULT_GEMINI_MODEL.into()),
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| DEFAULT_GEMINI_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(DEFAULT_GEMINI_EMBEDDING_DIM as usize),
        }
    }
}

impl From<VoyageAIConfig> for zqa_rag::config::VoyageAIConfig {
    fn from(config: VoyageAIConfig) -> Self {
        use zqa_rag::constants::{
            DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL,
            DEFAULT_VOYAGE_RERANK_MODEL,
        };

        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("VOYAGE_AI_API_KEY").ok())
                .expect("Voyage API key not found. Please set it in your config file or as VOYAGE_AI_API_KEY."),
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(DEFAULT_VOYAGE_EMBEDDING_DIM as usize),
            reranker: config
                .reranker
                .unwrap_or_else(|| DEFAULT_VOYAGE_RERANK_MODEL.to_string()),
        }
    }
}

impl From<CohereConfig> for zqa_rag::config::CohereConfig {
    fn from(config: CohereConfig) -> Self {
        use zqa_rag::constants::{
            DEFAULT_COHERE_EMBEDDING_DIM, DEFAULT_COHERE_EMBEDDING_MODEL,
            DEFAULT_COHERE_RERANK_MODEL,
        };

        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("COHERE_API_KEY").ok())
                .expect(
                "Cohere API key not found. Please set it in your config file or as COHERE_API_KEY.",
            ),
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| DEFAULT_COHERE_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(DEFAULT_COHERE_EMBEDDING_DIM as usize),
            reranker: config
                .reranker
                .unwrap_or_else(|| DEFAULT_COHERE_RERANK_MODEL.to_string()),
        }
    }
}

impl From<OpenRouterConfig> for zqa_rag::config::OpenRouterConfig {
    fn from(config: OpenRouterConfig) -> Self {
        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("OPENROUTER_API_KEY").ok())
                .expect("OpenRouter API key not found. Please set it in your config file or as OPENROUTER_API_KEY."),
            model: config
                .model
                .unwrap_or(DEFAULT_OPENROUTER_MODEL.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use zqa_macros::test_eq;

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
            model_small = "claude-sonnet-4"
            api_key = "sk-ant-test"
            max_tokens = 64000

            [openai]
            model = "gpt-5.2"
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
        test_eq!(config.model_provider, "anthropic");
        test_eq!(config.embedding_provider, "voyageai");
        test_eq!(config.max_concurrent_requests, 5);
        test_eq!(config.max_retries, 3);

        let anthropic = config.anthropic.unwrap();
        test_eq!(anthropic.model, Some("claude-sonnet-4-5".into()));
        test_eq!(anthropic.model_small, Some("claude-sonnet-4".into()));
        test_eq!(anthropic.max_tokens, 64_000);

        let voyageai = config.voyageai.unwrap();
        test_eq!(voyageai.reranker.unwrap(), "rerank-2.5");
        test_eq!(voyageai.embedding_model.unwrap(), "voyage-3-large");
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml_str = r#"
            model_provider = "openai"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        test_eq!(config.model_provider, "openai");
        test_eq!(config.embedding_provider, "voyageai"); // default
        test_eq!(config.max_concurrent_requests, 5); // default
    }
}

/// An error that represents a failure to read the base user directory. This is only created
/// when `directories::UserDirs::new` fails.
#[derive(Debug, Error)]
pub struct BaseDirError;

impl Display for BaseDirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to read base directory.")
    }
}

/// Get the config directory for the application.
///
/// # Returns
///
/// If successful, a `PathBuf` with the path to the config directory.
///
/// # Errors
///
/// A `BaseDirError` if `directories::UserDirs::new` fails.
pub fn get_config_dir() -> Result<PathBuf, BaseDirError> {
    let base_dir = directories::UserDirs::new().ok_or(BaseDirError)?;
    Ok(base_dir.home_dir().join(".config").join("zqa"))
}
