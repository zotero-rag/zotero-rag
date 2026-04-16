//! Registry for providers
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use lancedb::embeddings::EmbeddingFunction;

use crate::{
    capabilities::{EmbeddingFactory, LlmFactory, RerankFactory},
    config::LLMClientConfig,
    embedding::common::EmbeddingProviderConfig,
    llm::{errors::LLMError, factory::LLMClient},
    providers::{
        anthropic::AnthropicProvider, cohere::CohereProvider, gemini::GeminiProvider,
        ollama::OllamaProvider, openai::OpenAIProvider, openrouter::OpenRouterProvider,
        provider_id::ProviderId, voyage::VoyageAIProvider, zeroentropy::ZeroEntropyProvider,
    },
    reranking::common::{Rerank, RerankProviderConfig},
    vector::lance::{LanceEmbeddingRegistrar, LanceError},
};

/// Registry for provider factories keyed by canonical provider ID.
pub struct ProviderRegistry {
    /// Mappings from LLM providers to corresponding factory methods
    llm: HashMap<ProviderId, Arc<dyn LlmFactory>>,
    /// Mappings from embedding providers to corresponding factory methods
    embedding: HashMap<ProviderId, Arc<dyn EmbeddingFactory>>,
    /// Mappings from reranking providers to corresponding factory methods
    rerank: HashMap<ProviderId, Arc<dyn RerankFactory>>,
    /// Mappings from LanceDB embedding providers to corresponding factory methods
    lance_embedding: HashMap<ProviderId, Arc<dyn LanceEmbeddingRegistrar>>,
}

impl ProviderRegistry {
    /// Create an empty provider registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            llm: HashMap::new(),
            embedding: HashMap::new(),
            rerank: HashMap::new(),
            lance_embedding: HashMap::new(),
        }
    }

    /// Register an LLM factory.
    pub fn register_llm(&mut self, factory: Arc<dyn LlmFactory>) {
        self.llm.insert(factory.provider_id(), factory);
    }

    /// Register an embedding factory.
    pub fn register_embedding(&mut self, factory: Arc<dyn EmbeddingFactory>) {
        self.embedding.insert(factory.provider_id(), factory);
    }

    /// Register a reranking factory.
    pub fn register_rerank(&mut self, factory: Arc<dyn RerankFactory>) {
        self.rerank.insert(factory.provider_id(), factory);
    }

    /// Register a LanceDB embedding registrar.
    pub fn register_lance_embedding(&mut self, registrar: Arc<dyn LanceEmbeddingRegistrar>) {
        self.lance_embedding
            .insert(registrar.provider_id(), registrar);
    }

    /// Create an LLM client from provider-specific config.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider is not registered or client creation fails.
    pub fn create_llm(&self, config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
        let provider = config.provider_id();
        let factory = self
            .llm
            .get(&provider)
            .ok_or_else(|| LLMError::InvalidProviderError(provider.as_str().to_string()))?;
        factory.create_llm(config)
    }

    /// Create an embedding client from provider-specific config.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider is not registered or client creation fails.
    pub fn create_embedding(
        &self,
        config: &EmbeddingProviderConfig,
    ) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
        let provider = config.provider_id();
        let factory = self
            .embedding
            .get(&provider)
            .ok_or_else(|| LLMError::InvalidProviderError(provider.as_str().to_string()))?;
        factory.create_embedding(config)
    }

    /// Create a reranker from provider-specific config.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider is not registered or client creation fails.
    pub fn create_reranker(
        &self,
        config: &RerankProviderConfig,
    ) -> Result<Arc<dyn Rerank>, LLMError> {
        let provider = config.provider_id();
        let factory = self
            .rerank
            .get(&provider)
            .ok_or_else(|| LLMError::InvalidProviderError(provider.as_str().to_string()))?;
        factory.create_reranker(config)
    }

    /// Register the provider's embedding implementation with LanceDB.
    ///
    /// # Errors
    ///
    /// Returns an error if no LanceDB registrar exists for the provider or if registration fails.
    pub fn register_embedding_with_lancedb(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError> {
        let provider = config.provider_id();
        let registrar = self.lance_embedding.get(&provider).ok_or_else(|| {
            LanceError::ParameterError(format!(
                "{} does not support LanceDB embedding registration",
                provider.as_str()
            ))
        })?;

        registrar.register_with_lancedb(db, config)
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Construct the default provider registry used by the crate.
#[must_use]
pub fn default_provider_registry() -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();

    registry.register_llm(Arc::new(AnthropicProvider));
    registry.register_llm(Arc::new(OpenAIProvider));
    registry.register_llm(Arc::new(OpenRouterProvider));
    registry.register_llm(Arc::new(GeminiProvider));
    registry.register_llm(Arc::new(OllamaProvider));

    registry.register_embedding(Arc::new(OpenAIProvider));
    registry.register_embedding(Arc::new(VoyageAIProvider));
    registry.register_embedding(Arc::new(CohereProvider));
    registry.register_embedding(Arc::new(ZeroEntropyProvider));
    registry.register_embedding(Arc::new(GeminiProvider));
    registry.register_embedding(Arc::new(OllamaProvider));

    registry.register_rerank(Arc::new(VoyageAIProvider));
    registry.register_rerank(Arc::new(CohereProvider));
    registry.register_rerank(Arc::new(ZeroEntropyProvider));

    registry.register_lance_embedding(Arc::new(OpenAIProvider));
    registry.register_lance_embedding(Arc::new(VoyageAIProvider));
    registry.register_lance_embedding(Arc::new(CohereProvider));
    registry.register_lance_embedding(Arc::new(ZeroEntropyProvider));
    registry.register_lance_embedding(Arc::new(GeminiProvider));
    registry.register_lance_embedding(Arc::new(OllamaProvider));

    registry
}

static DEFAULT_REGISTRY: LazyLock<ProviderRegistry> = LazyLock::new(default_provider_registry);

/// Get the default provider registry.
#[must_use]
pub fn provider_registry() -> &'static ProviderRegistry {
    &DEFAULT_REGISTRY
}

#[cfg(test)]
mod tests {
    use super::ProviderRegistry;
    use crate::{
        config::{
            AnthropicConfig, CohereConfig, GeminiConfig, OllamaConfig, OpenAIConfig,
            OpenRouterConfig, VoyageAIConfig, ZeroEntropyConfig,
        },
        embedding::common::EmbeddingProviderConfig,
        llm::factory::LLMClient,
        providers::registry::provider_registry,
        reranking::common::RerankProviderConfig,
        vector::lance::LanceError,
    };

    fn openai_llm_config() -> crate::config::LLMClientConfig {
        crate::config::LLMClientConfig::OpenAI(OpenAIConfig {
            api_key: "test-key".into(),
            model: "gpt-test".into(),
            max_tokens: 1024,
            embedding_model: "text-embedding-test".into(),
            embedding_dims: 1536,
        })
    }

    fn anthropic_llm_config() -> crate::config::LLMClientConfig {
        crate::config::LLMClientConfig::Anthropic(AnthropicConfig {
            api_key: "test-key".into(),
            model: "claude-test".into(),
            max_tokens: 1024,
        })
    }

    fn gemini_llm_config() -> crate::config::LLMClientConfig {
        crate::config::LLMClientConfig::Gemini(GeminiConfig {
            api_key: "test-key".into(),
            model: "gemini-test".into(),
            embedding_model: "gemini-embedding-test".into(),
            embedding_dims: 3072,
        })
    }

    fn ollama_llm_config() -> crate::config::LLMClientConfig {
        crate::config::LLMClientConfig::Ollama(OllamaConfig {
            model: "qwen-test".into(),
            max_tokens: 1024,
            embedding_model: "qwen-embedding-test".into(),
            embedding_dims: 768,
            base_url: "http://127.0.0.1:11434".into(),
        })
    }

    fn openrouter_llm_config() -> crate::config::LLMClientConfig {
        crate::config::LLMClientConfig::OpenRouter(OpenRouterConfig {
            api_key: "test-key".into(),
            model: "anthropic/test-model".into(),
        })
    }

    fn openai_embedding_config() -> EmbeddingProviderConfig {
        EmbeddingProviderConfig::OpenAI(OpenAIConfig {
            api_key: "test-key".into(),
            model: "gpt-test".into(),
            max_tokens: 1024,
            embedding_model: "text-embedding-test".into(),
            embedding_dims: 1536,
        })
    }

    fn gemini_embedding_config() -> EmbeddingProviderConfig {
        EmbeddingProviderConfig::Gemini(GeminiConfig {
            api_key: "test-key".into(),
            model: "gemini-test".into(),
            embedding_model: "gemini-embedding-test".into(),
            embedding_dims: 3072,
        })
    }

    fn ollama_embedding_config() -> EmbeddingProviderConfig {
        EmbeddingProviderConfig::Ollama(OllamaConfig {
            model: "qwen-test".into(),
            max_tokens: 1024,
            embedding_model: "qwen-embedding-test".into(),
            embedding_dims: 768,
            base_url: "http://127.0.0.1:11434".into(),
        })
    }

    fn voyage_embedding_config() -> EmbeddingProviderConfig {
        EmbeddingProviderConfig::VoyageAI(VoyageAIConfig {
            api_key: "test-key".into(),
            embedding_model: "voyage-test".into(),
            embedding_dims: 1024,
            reranker: "rerank-test".into(),
        })
    }

    fn cohere_embedding_config() -> EmbeddingProviderConfig {
        EmbeddingProviderConfig::Cohere(CohereConfig {
            api_key: "test-key".into(),
            embedding_model: "embed-test".into(),
            embedding_dims: 1024,
            reranker: "rerank-test".into(),
        })
    }

    fn zeroentropy_embedding_config() -> EmbeddingProviderConfig {
        EmbeddingProviderConfig::ZeroEntropy(ZeroEntropyConfig {
            api_key: "test-key".into(),
            embedding_model: "zembed-test".into(),
            embedding_dims: 1024,
            reranker: "zerank-test".into(),
        })
    }

    fn voyage_rerank_config() -> RerankProviderConfig {
        RerankProviderConfig::VoyageAI(VoyageAIConfig {
            api_key: "test-key".into(),
            embedding_model: "voyage-test".into(),
            embedding_dims: 1024,
            reranker: "rerank-test".into(),
        })
    }

    fn cohere_rerank_config() -> RerankProviderConfig {
        RerankProviderConfig::Cohere(CohereConfig {
            api_key: "test-key".into(),
            embedding_model: "embed-test".into(),
            embedding_dims: 1024,
            reranker: "rerank-test".into(),
        })
    }

    fn zeroentropy_rerank_config() -> RerankProviderConfig {
        RerankProviderConfig::ZeroEntropy(ZeroEntropyConfig {
            api_key: "test-key".into(),
            embedding_model: "zembed-test".into(),
            embedding_dims: 1024,
            reranker: "zerank-test".into(),
        })
    }

    #[test]
    fn default_registry_creates_all_llm_clients() {
        let registry = provider_registry();

        assert!(matches!(
            registry.create_llm(&anthropic_llm_config()),
            Ok(LLMClient::Anthropic(_))
        ));
        assert!(matches!(
            registry.create_llm(&openai_llm_config()),
            Ok(LLMClient::OpenAI(_))
        ));
        assert!(matches!(
            registry.create_llm(&openrouter_llm_config()),
            Ok(LLMClient::OpenRouter(_))
        ));
        assert!(matches!(
            registry.create_llm(&gemini_llm_config()),
            Ok(LLMClient::Gemini(_))
        ));
        assert!(matches!(
            registry.create_llm(&ollama_llm_config()),
            Ok(LLMClient::Ollama(_))
        ));
    }

    #[test]
    fn default_registry_creates_all_embedding_clients() {
        let registry = provider_registry();

        assert!(
            registry
                .create_embedding(&openai_embedding_config())
                .is_ok()
        );
        assert!(
            registry
                .create_embedding(&voyage_embedding_config())
                .is_ok()
        );
        assert!(
            registry
                .create_embedding(&cohere_embedding_config())
                .is_ok()
        );
        assert!(
            registry
                .create_embedding(&zeroentropy_embedding_config())
                .is_ok()
        );
        assert!(
            registry
                .create_embedding(&gemini_embedding_config())
                .is_ok()
        );
        assert!(
            registry
                .create_embedding(&ollama_embedding_config())
                .is_ok()
        );
    }

    #[test]
    fn default_registry_creates_all_rerank_clients() {
        let registry = provider_registry();

        assert!(registry.create_reranker(&voyage_rerank_config()).is_ok());
        assert!(registry.create_reranker(&cohere_rerank_config()).is_ok());
        assert!(
            registry
                .create_reranker(&zeroentropy_rerank_config())
                .is_ok()
        );
    }

    #[tokio::test]
    async fn default_registry_registers_embedding_with_lancedb() {
        let registry = provider_registry();
        let db = lancedb::connect("memory://test-registry-success")
            .execute()
            .await
            .expect("memory db should be created");
        let result = registry.register_embedding_with_lancedb(&db, &openai_embedding_config());
        assert!(result.is_ok());
    }

    #[test]
    fn empty_registry_rejects_unregistered_llm_provider() {
        let registry = ProviderRegistry::new();
        let result = registry.create_llm(&openai_llm_config());

        assert!(matches!(
            result,
            Err(crate::llm::errors::LLMError::InvalidProviderError(provider)) if provider == "openai"
        ));
    }

    #[test]
    fn empty_registry_rejects_unregistered_embedding_provider() {
        let registry = ProviderRegistry::new();
        let result = registry.create_embedding(&openai_embedding_config());

        assert!(matches!(
            result,
            Err(crate::llm::errors::LLMError::InvalidProviderError(provider)) if provider == "openai"
        ));
    }

    #[test]
    fn empty_registry_rejects_unregistered_rerank_provider() {
        let registry = ProviderRegistry::new();
        let result = registry.create_reranker(&voyage_rerank_config());

        assert!(matches!(
            result,
            Err(crate::llm::errors::LLMError::InvalidProviderError(provider)) if provider == "voyageai"
        ));
    }

    #[tokio::test]
    async fn empty_registry_rejects_lancedb_registration_for_unregistered_provider() {
        let registry = ProviderRegistry::new();
        let db = lancedb::connect("memory://test-registry")
            .execute()
            .await
            .expect("memory db should be created");
        let result = registry.register_embedding_with_lancedb(&db, &openai_embedding_config());

        assert!(matches!(
            result,
            Err(LanceError::ParameterError(message)) if message.contains("openai")
        ));
    }
}
