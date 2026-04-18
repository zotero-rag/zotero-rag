use std::fmt::Display;
use std::path::PathBuf;
use std::{env, num::ParseIntError, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use zqa_rag::capabilities::{EmbeddingProvider, ModelProvider, RerankerProvider};
use zqa_rag::config::LLMClientConfig;
#[allow(clippy::wildcard_imports)]
use zqa_rag::constants::*;
use zqa_rag::constants::{DEFAULT_OPENAI_EMBEDDING_DIM, DEFAULT_OPENAI_EMBEDDING_MODEL};
use zqa_rag::embedding::common::EmbeddingProviderConfig;
use zqa_rag::llm::base::ReasoningConfig;
use zqa_rag::reranking::common::RerankProviderConfig;

/// TOML config. Below is an example config with all the defaults. The TOML config is
/// overridden by environment variables.
///
/// ```toml
/// model_provider = "anthropic"  # Generation model provider
/// embedding_provider = "voyageai"  # Embedding/reranker model provider
/// reranker_provider = "voyageai"  # Omit this to skip reranking
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
/// reasoning_budget = 2048
///
/// [ollama]
/// model = "qwen3.5"  # Defaults to the 9B version
/// model_small = "qwen3.5:0.8b"
/// max_tokens = 8192
/// embedding_model = "qwen3-embedding"
/// embedding_dims = 4096
/// reasoning_budget = 2048
/// base_url = "http://localhost:11434"  # Defaults to local ollama instance
///
/// [openai]
/// model = "gpt-5.2"
/// model_small = "gpt-5-mini"
/// api_key = "sk-proj-..."
/// max_tokens = 8192
/// embedding_model = "text-embedding-3-small"
/// embedding_dims = 1536
/// reasoning_effort = "high"  # Supported by the most models
///
/// [gemini]
/// model = "gemini-3.1-pro-preview"
/// model_small = "gemini-3-flash-preview"
/// api_key = "AI..."
/// embedding_model = "gemini-embedding-001"
/// embedding_dims = 3072
/// reasoning_budget = 2048
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
/// [zeroentropy]
/// reranker = "zerank-2"
/// embedding_model = "zembed-1"
/// embedding_dims = 2560
/// api_key = "..."
///
/// [openrouter]
/// api_key = "..."
/// model = "anthropic/claude-sonnet-4.5"
/// model_small = "anthropic/claude-haiku-4.5"
/// reasoning_effort = "high"
/// reasoning_budget = 2048
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Generation model provider (anthropic, openai, gemini, openrouter)
    #[serde(default = "default_model_provider")]
    pub model_provider: ModelProvider,

    /// Embedding provider (anthropic, openai, voyageai, gemini, cohere, zeroentropy)
    #[serde(default = "default_embedding_provider")]
    pub embedding_provider: EmbeddingProvider,

    /// Reranker provider (voyageai, cohere, zeroentropy). Omit this to disable reranking.
    pub reranker_provider: Option<RerankerProvider>,

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

    /// ZeroEntropy-specific configuration
    #[serde(default)]
    pub zeroentropy: Option<ZeroEntropyConfig>,
}

impl Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.redacted())
    }
}

fn redact(secret: Option<&String>) -> Option<String> {
    let s = secret?;
    let len = s.len();
    if len >= 4
        && let Some((head, _)) = s.split_at_checked(2)
        && let Some((_, tail)) = s.split_at_checked(len - 2)
    {
        return Some(format!("{head}***{tail}"));
    }

    Some("***".into())
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
                .embedding_model
                .replace_with_env("OLLAMA_EMBEDDING_MODEL");
            ollama_config
                .embedding_dims
                .replace_with_env("OLLAMA_EMBEDDING_DIMS");
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

        // ZeroEntropy options
        if let Some(ze_config) = &mut self.zeroentropy {
            // embedding_dims is not exposed as an env option
            ze_config
                .reranker
                .replace_with_env("ZEROENTROPY_RERANK_MODEL");
            ze_config
                .embedding_model
                .replace_with_env("ZEROENTROPY_EMBEDDING_MODEL");
            ze_config.api_key.replace_with_env("ZEROENTROPY_API_KEY");
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
        match self.embedding_provider {
            EmbeddingProvider::OpenAI => self
                .openai
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::OpenAI(cfg.clone().into())),
            EmbeddingProvider::Gemini => self
                .gemini
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::Gemini(cfg.clone().into())),
            EmbeddingProvider::VoyageAI => self
                .voyageai
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::VoyageAI(cfg.clone().into())),
            EmbeddingProvider::Cohere => self
                .cohere
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::Cohere(cfg.clone().into())),
            EmbeddingProvider::Ollama => self
                .ollama
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::Ollama(cfg.clone().into())),
            EmbeddingProvider::ZeroEntropy => self
                .zeroentropy
                .as_ref()
                .map(|cfg| EmbeddingProviderConfig::ZeroEntropy(cfg.clone().into())),
        }
    }

    #[must_use]
    pub fn get_reasoning_config(&self) -> Option<ReasoningConfig> {
        match self.model_provider {
            ModelProvider::Anthropic => self.anthropic.as_ref().and_then(|c| {
                c.reasoning_budget.map(|budget| ReasoningConfig {
                    max_tokens: Some(budget),
                    effort: None,
                    summary: None,
                })
            }),
            ModelProvider::Ollama => self.ollama.as_ref().and_then(|c| {
                c.reasoning_budget.map(|budget| ReasoningConfig {
                    max_tokens: Some(budget),
                    effort: None,
                    summary: None,
                })
            }),
            ModelProvider::OpenAI => self.openai.as_ref().and_then(|c| {
                c.reasoning_effort.as_ref().map(|effort| ReasoningConfig {
                    max_tokens: None,
                    effort: Some(effort.clone()),
                    summary: None,
                })
            }),
            ModelProvider::Gemini => self.gemini.as_ref().and_then(|c| {
                c.reasoning_budget.map(|budget| ReasoningConfig {
                    max_tokens: Some(budget),
                    effort: None,
                    summary: None,
                })
            }),
            ModelProvider::OpenRouter => self.openrouter.as_ref().and_then(|c| {
                if c.reasoning_effort.is_none() && c.reasoning_budget.is_none() {
                    return None;
                }

                Some(ReasoningConfig {
                    max_tokens: c.reasoning_budget,
                    effort: c.reasoning_effort.clone(),
                    summary: None,
                })
            }),
        }
    }

    #[must_use]
    pub fn get_reranker_config(&self) -> Option<RerankProviderConfig> {
        match self.reranker_provider {
            Some(RerankerProvider::VoyageAI) => self
                .voyageai
                .as_ref()
                .map(|cfg| RerankProviderConfig::VoyageAI(cfg.clone().into())),
            Some(RerankerProvider::Cohere) => self
                .cohere
                .as_ref()
                .map(|cfg| RerankProviderConfig::Cohere(cfg.clone().into())),
            Some(RerankerProvider::ZeroEntropy) => self
                .zeroentropy
                .as_ref()
                .map(|cfg| RerankProviderConfig::ZeroEntropy(cfg.clone().into())),
            _ => None,
        }
    }

    /// Get the LLM client configuration using the small model variant. Falls back to the regular
    /// model if no small model is configured.
    #[must_use]
    pub fn get_small_model_config(&self) -> Option<LLMClientConfig> {
        let mut client_config = self.get_generation_config()?;

        let small_model_name = match self.model_provider {
            ModelProvider::Anthropic => {
                self.anthropic.as_ref().and_then(|c| c.model_small.as_ref())
            }
            ModelProvider::Ollama => self.ollama.as_ref().and_then(|c| c.model_small.as_ref()),
            ModelProvider::OpenAI => self.openai.as_ref().and_then(|c| c.model_small.as_ref()),
            ModelProvider::Gemini => self.gemini.as_ref().and_then(|c| c.model_small.as_ref()),
            ModelProvider::OpenRouter => self
                .openrouter
                .as_ref()
                .and_then(|c| c.model_small.as_ref()),
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
        self.get_embedding_config().map(|config| match config {
            EmbeddingProviderConfig::OpenAI(cfg) => cfg.embedding_model,
            EmbeddingProviderConfig::Gemini(cfg) => cfg.embedding_model,
            EmbeddingProviderConfig::VoyageAI(cfg) => cfg.embedding_model,
            EmbeddingProviderConfig::Ollama(cfg) => cfg.embedding_model,
            EmbeddingProviderConfig::Cohere(cfg) => cfg.embedding_model,
            EmbeddingProviderConfig::ZeroEntropy(cfg) => cfg.embedding_model,
        })
    }

    #[must_use]
    pub fn get_generation_model_name(&self) -> Option<String> {
        self.get_generation_config().map(|config| match config {
            LLMClientConfig::Anthropic(cfg) => cfg.model,
            LLMClientConfig::Ollama(cfg) => cfg.model,
            LLMClientConfig::OpenAI(cfg) => cfg.model,
            LLMClientConfig::Gemini(cfg) => cfg.model,
            LLMClientConfig::OpenRouter(cfg) => cfg.model,
        })
    }

    /// Return the configuration object with all API keys redacted.
    #[must_use]
    pub fn redacted(&self) -> Self {
        let mut clone = self.clone();

        if let Some(cfg) = clone.anthropic.as_mut() {
            cfg.api_key = redact(cfg.api_key.as_ref());
        }

        if let Some(cfg) = clone.openai.as_mut() {
            cfg.api_key = redact(cfg.api_key.as_ref());
        }

        if let Some(cfg) = clone.gemini.as_mut() {
            cfg.api_key = redact(cfg.api_key.as_ref());
        }

        if let Some(cfg) = clone.openrouter.as_mut() {
            cfg.api_key = redact(cfg.api_key.as_ref());
        }

        if let Some(cfg) = clone.voyageai.as_mut() {
            cfg.api_key = redact(cfg.api_key.as_ref());
        }

        if let Some(cfg) = clone.cohere.as_mut() {
            cfg.api_key = redact(cfg.api_key.as_ref());
        }

        if let Some(cfg) = clone.zeroentropy.as_mut() {
            cfg.api_key = redact(cfg.api_key.as_ref());
        }

        clone
    }

    #[must_use]
    pub fn get_generation_config(&self) -> Option<LLMClientConfig> {
        match self.model_provider {
            ModelProvider::Anthropic => self
                .anthropic
                .as_ref()
                .map(|cfg| LLMClientConfig::Anthropic(cfg.clone().into())),
            ModelProvider::Ollama => self
                .ollama
                .as_ref()
                .map(|cfg| LLMClientConfig::Ollama(cfg.clone().into())),
            ModelProvider::OpenAI => self
                .openai
                .as_ref()
                .map(|cfg| LLMClientConfig::OpenAI(cfg.clone().into())),
            ModelProvider::Gemini => self
                .gemini
                .as_ref()
                .map(|cfg| LLMClientConfig::Gemini(cfg.clone().into())),
            ModelProvider::OpenRouter => self
                .openrouter
                .as_ref()
                .map(|cfg| LLMClientConfig::OpenRouter(cfg.clone().into())),
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

    /// Token budget for extended thinking. Omit to disable thinking.
    pub reasoning_budget: Option<u32>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            model: Some(DEFAULT_ANTHROPIC_MODEL.into()),
            model_small: Some(DEFAULT_ANTHROPIC_MODEL_SMALL.into()),
            max_tokens: DEFAULT_ANTHROPIC_MAX_TOKENS,
            api_key: None,
            reasoning_budget: None,
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
    /// Token budget for extended thinking. Omit to disable thinking.
    pub reasoning_budget: Option<u32>,
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
            reasoning_budget: None,
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

    /// Reasoning effort level (e.g., "high"). Omit to disable reasoning.
    pub reasoning_effort: Option<String>,
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
            reasoning_effort: None,
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

    /// Token budget for extended thinking. Omit to disable thinking.
    pub reasoning_budget: Option<u32>,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            model: Some(DEFAULT_GEMINI_MODEL.into()),
            model_small: Some(DEFAULT_GEMINI_MODEL_SMALL.into()),
            embedding_model: Some(DEFAULT_GEMINI_EMBEDDING_MODEL.into()),
            embedding_dims: Some(DEFAULT_GEMINI_EMBEDDING_DIM as usize),
            api_key: None,
            reasoning_budget: None,
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

/// ZeroEntropy provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ZeroEntropyConfig {
    /// Reranker model name
    pub reranker: Option<String>,

    /// Embedding model name
    pub embedding_model: Option<String>,

    /// Embedding dimensions (one of 2560, 1280, 640, 320, 160, 80, or 40)
    pub embedding_dims: Option<usize>,

    /// API key
    pub api_key: Option<String>,
}

impl Default for ZeroEntropyConfig {
    fn default() -> Self {
        Self {
            reranker: Some(DEFAULT_ZEROENTROPY_RERANK_MODEL.into()),
            embedding_model: Some(DEFAULT_ZEROENTROPY_EMBEDDING_MODEL.into()),
            embedding_dims: Some(DEFAULT_ZEROENTROPY_EMBEDDING_DIM as usize),
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

    /// Reasoning effort level (e.g., "high"). Omit to disable reasoning.
    pub reasoning_effort: Option<String>,

    /// Token budget for extended thinking. Omit to disable thinking.
    pub reasoning_budget: Option<u32>,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            model: Some(DEFAULT_OPENROUTER_MODEL.into()),
            model_small: Some(DEFAULT_OPENROUTER_MODEL_SMALL.into()),
            api_key: None,
            reasoning_effort: None,
            reasoning_budget: None,
        }
    }
}

// Default value functions
fn default_model_provider() -> ModelProvider {
    ModelProvider::Anthropic
}

fn default_embedding_provider() -> EmbeddingProvider {
    EmbeddingProvider::VoyageAI
}

#[allow(clippy::unnecessary_wraps)]
fn default_reranker_provider() -> Option<RerankerProvider> {
    Some(RerankerProvider::VoyageAI)
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

impl OverwriteFromEnv for Option<usize> {
    fn replace_with_env(&mut self, var: &str)
    where
        Self: Sized,
    {
        if let Ok(env_var) = env::var(var)
            && let Ok(value) = env_var.parse::<usize>()
        {
            *self = Some(value);
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
            zeroentropy: Some(ZeroEntropyConfig::default()),
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
            reasoning_budget: config.reasoning_budget,
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
            reasoning_budget: config.reasoning_budget,
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
            reasoning_effort: config.reasoning_effort,
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
            reasoning_budget: config.reasoning_budget,
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

impl From<ZeroEntropyConfig> for zqa_rag::config::ZeroEntropyConfig {
    fn from(config: ZeroEntropyConfig) -> Self {
        use zqa_rag::constants::{
            DEFAULT_ZEROENTROPY_EMBEDDING_DIM, DEFAULT_ZEROENTROPY_EMBEDDING_MODEL,
            DEFAULT_ZEROENTROPY_RERANK_MODEL,
        };

        Self {
            api_key: config
                .api_key
                .or_else(|| env::var("ZEROENTROPY_API_KEY").ok())
                .expect(
                    "ZeroEntropy API key not found. Please set it in your config file or as ZEROENTROPY_API_KEY.",
                ),
            embedding_model: config
                .embedding_model
                .unwrap_or_else(|| DEFAULT_ZEROENTROPY_EMBEDDING_MODEL.to_string()),
            embedding_dims: config
                .embedding_dims
                .unwrap_or(DEFAULT_ZEROENTROPY_EMBEDDING_DIM as usize),
            reranker: config
                .reranker
                .unwrap_or_else(|| DEFAULT_ZEROENTROPY_RERANK_MODEL.to_string()),
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
            model: config.model.unwrap_or(DEFAULT_OPENROUTER_MODEL.into()),
            reasoning_effort: config.reasoning_effort,
            reasoning_budget: config.reasoning_budget,
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
        test_eq!(config.model_provider, ModelProvider::Anthropic);
        test_eq!(config.embedding_provider, EmbeddingProvider::VoyageAI);
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
        test_eq!(config.model_provider, ModelProvider::OpenAI);
        test_eq!(config.embedding_provider, EmbeddingProvider::VoyageAI); // default
        test_eq!(config.max_concurrent_requests, 1); // default
    }

    #[test]
    fn test_redact_none() {
        assert_eq!(redact(None), None);
    }

    #[test]
    fn test_redact_short_key() {
        // Strings shorter than 4 chars get fully masked
        assert_eq!(redact(Some(&"ab".into())), Some("***".into()));
        assert_eq!(redact(Some(&"abc".into())), Some("***".into()));
    }

    #[test]
    fn test_redact_normal_key() {
        // Shows first 2 and last 2 characters
        assert_eq!(
            redact(Some(&"sk-ant-abc123".into())),
            Some("sk***23".into())
        );
        // Exactly 4 chars: first 2 and last 2 overlap but are still shown
        assert_eq!(redact(Some(&"abcd".into())), Some("ab***cd".into()));
    }

    #[test]
    fn test_redacted_masks_all_provider_keys() {
        let config = Config {
            anthropic: Some(AnthropicConfig {
                api_key: Some("sk-ant-abcdef".into()),
                ..AnthropicConfig::default()
            }),
            openai: Some(OpenAIConfig {
                api_key: Some("sk-proj-abcdef".into()),
                ..OpenAIConfig::default()
            }),
            gemini: Some(GeminiConfig {
                api_key: Some("AIzaabcdef".into()),
                ..GeminiConfig::default()
            }),
            openrouter: Some(OpenRouterConfig {
                api_key: Some("or-abcdef".into()),
                ..OpenRouterConfig::default()
            }),
            voyageai: Some(VoyageAIConfig {
                api_key: Some("pa-abcdef".into()),
                ..VoyageAIConfig::default()
            }),
            cohere: Some(CohereConfig {
                api_key: Some("co-abcdef".into()),
                ..CohereConfig::default()
            }),
            ..Config::default()
        };

        let redacted = config.redacted();

        assert_ne!(
            redacted.anthropic.as_ref().unwrap().api_key,
            Some("sk-ant-abcdef".into())
        );
        assert_ne!(
            redacted.openai.as_ref().unwrap().api_key,
            Some("sk-proj-abcdef".into())
        );
        assert_ne!(
            redacted.gemini.as_ref().unwrap().api_key,
            Some("AIzaabcdef".into())
        );
        assert_ne!(
            redacted.openrouter.as_ref().unwrap().api_key,
            Some("or-abcdef".into())
        );
        assert_ne!(
            redacted.voyageai.as_ref().unwrap().api_key,
            Some("pa-abcdef".into())
        );
        assert_ne!(
            redacted.cohere.as_ref().unwrap().api_key,
            Some("co-abcdef".into())
        );

        // Spot-check the format: first 2 + *** + last 2
        assert_eq!(
            redacted.anthropic.as_ref().unwrap().api_key,
            Some("sk***ef".into())
        );
    }

    #[test]
    fn test_redacted_none_key_stays_none() {
        let config = Config {
            anthropic: Some(AnthropicConfig {
                api_key: None,
                ..AnthropicConfig::default()
            }),
            ..Config::default()
        };

        let redacted = config.redacted();
        assert_eq!(redacted.anthropic.unwrap().api_key, None);
    }

    #[test]
    fn test_redacted_does_not_mutate_original() {
        let config = Config {
            anthropic: Some(AnthropicConfig {
                api_key: Some("sk-ant-secret".into()),
                ..AnthropicConfig::default()
            }),
            ..Config::default()
        };

        let _ = config.redacted();
        assert_eq!(
            config.anthropic.unwrap().api_key,
            Some("sk-ant-secret".into())
        );
    }

    #[test]
    fn test_redacted_preserves_non_key_fields() {
        let config = Config {
            anthropic: Some(AnthropicConfig {
                model: Some("claude-opus-4-6".into()),
                api_key: Some("sk-ant-secret".into()),
                max_tokens: 8192,
                ..AnthropicConfig::default()
            }),
            ..Config::default()
        };

        let redacted = config.redacted();
        let anthropic = redacted.anthropic.unwrap();
        assert_eq!(anthropic.model, Some("claude-opus-4-6".into()));
        assert_eq!(anthropic.max_tokens, 8192);
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
