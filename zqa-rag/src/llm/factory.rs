//! Factory methods for creating clients based on the provider name.

use std::sync::Arc;

use crate::capabilities::{BatchAPIProvider, BatchJobState};
use crate::clients::anthropic::AnthropicClient;
use crate::clients::gemini::GeminiClient;
use crate::clients::ollama::OllamaClient;
use crate::clients::openai::OpenAIClient;
use crate::clients::openrouter::OpenRouterClient;
use crate::config::LLMClientConfig;
use crate::embedding::common::{BatchEmbeddingRequest, BatchEmbeddingResults, BatchSubmission};
use crate::embedding::voyage::VoyageAIClient;
use crate::http_client::ReqwestClient;
use crate::llm::base::{ApiClient, ChatRequest, ReasoningConfig};
use crate::llm::errors::LLMError;
use crate::providers::registry::provider_registry;

/// Enum representing different LLM client implementations
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LLMClient {
    /// Anthropic client
    Anthropic(AnthropicClient),
    /// Ollama client
    Ollama(OllamaClient),
    /// OpenAI client
    OpenAI(OpenAIClient),
    /// OpenRouter client
    OpenRouter(OpenRouterClient),
    /// Gemini client
    Gemini(GeminiClient),
}

impl LLMClient {
    /// Return the configured model
    #[must_use]
    pub fn get_model_name(&self) -> Option<String> {
        match self {
            LLMClient::Anthropic(client) => client.config.as_ref().map(|c| c.model.clone()),
            LLMClient::Ollama(client) => client.config.as_ref().map(|c| c.model.clone()),
            LLMClient::OpenAI(client) => client.config.as_ref().map(|c| c.model.clone()),
            LLMClient::OpenRouter(client) => client.config.as_ref().map(|c| c.model.clone()),
            LLMClient::Gemini(client) => client.config.as_ref().map(|c| c.model.clone()),
        }
    }

    /// Get the reasoning config from the client's config.
    #[must_use]
    pub fn get_reasoning_config(&self) -> Option<ReasoningConfig> {
        match self {
            LLMClient::Anthropic(client) => client.config.as_ref().map(|c| ReasoningConfig {
                max_tokens: c.reasoning_budget,
                effort: None,
                summary: None,
            }),
            LLMClient::Ollama(client) => client.config.as_ref().map(|c| ReasoningConfig {
                max_tokens: c.reasoning_budget,
                effort: None,
                summary: None,
            }),
            LLMClient::OpenAI(client) => client.config.as_ref().map(|c| ReasoningConfig {
                max_tokens: None,
                effort: c.reasoning_effort.clone(),
                summary: None,
            }),
            LLMClient::OpenRouter(client) => client.config.as_ref().map(|config| {
                if config.reasoning_effort.is_none() && config.reasoning_budget.is_none() {
                    return None;
                }

                Some(ReasoningConfig {
                    max_tokens: config.reasoning_budget,
                    effort: config.reasoning_effort.clone(),
                    summary: None,
                })
            })?,
            LLMClient::Gemini(client) => client.config.as_ref().map(|c| ReasoningConfig {
                max_tokens: c.reasoning_budget,
                effort: None,
                summary: None,
            }),
        }
    }
}

// Implement ApiClient for LLMClient to delegate to the inner implementations
impl ApiClient for LLMClient {
    async fn send_message(
        &self,
        message: &ChatRequest<'_>,
    ) -> Result<crate::llm::base::CompletionApiResponse, LLMError> {
        match self {
            LLMClient::Anthropic(client) => client.send_message(message).await,
            LLMClient::Ollama(client) => client.send_message(message).await,
            LLMClient::OpenAI(client) => client.send_message(message).await,
            LLMClient::OpenRouter(client) => client.send_message(message).await,
            LLMClient::Gemini(client) => client.send_message(message).await,
        }
    }
}

/// Returns an ApiClient implementation with provided configuration
///
/// # Errors
///
/// * `LLMError::InvalidProviderError` if the provider is not supported
pub fn get_client_with_config(config: &LLMClientConfig) -> Result<LLMClient, LLMError> {
    provider_registry().create_llm(config)
}

/// Enum representing different batch embedding client implementations
///
/// `BatchAPIProvider` is not dyn-compatible, so this acts as a hand-rolled vtable.
#[non_exhaustive]
pub enum BatchEmbeddingClient {
    /// Voyage AI batch embedding client
    VoyageAI(Arc<VoyageAIClient<ReqwestClient>>),
}

impl BatchAPIProvider for BatchEmbeddingClient {
    /// Submit a request to the batch embedding API.
    ///
    /// # Arguments
    ///
    /// * `request` - A [`BatchEmbeddingRequest`] object containing all the texts.
    ///
    /// # Returns
    ///
    /// Details of the submitted batch if succeeded, or an [`LLMError`].
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If the API key is not set, and no config is provided
    /// * `LLMError::InvalidHeaderError` - If the API key cannot be parsed as a header value
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403
    /// * `LLMError::HttpStatusError` - If the API returns another unsuccessful status code
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If either API response cannot be parsed
    /// * `LLMError::GenericLLMError` - If the temporary JSONL file cannot be written
    async fn submit_batch(
        &self,
        request: BatchEmbeddingRequest,
    ) -> Result<BatchSubmission, LLMError> {
        match self {
            Self::VoyageAI(client) => client.submit_batch(request).await,
        }
    }

    /// Check the status of a submitted batch.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The ID of the submitted batch, returned by the provider during submission.
    ///
    /// # Returns
    ///
    /// The state of the batch job
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If no API key was supplied.
    /// * `LLMError::InvalidHeaderError` - If the API key cannot be parsed as a header value
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403
    /// * `LLMError::HttpStatusError` - If the API returns another unsuccessful status code
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If the response cannot be parsed
    async fn get_batch_status(&self, batch_id: &str) -> Result<BatchJobState, LLMError> {
        match self {
            Self::VoyageAI(client) => client.get_batch_status(batch_id).await,
        }
    }

    /// Attempt to fetch the results of a submitted batch job.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The ID of the submitted batch, returned by the provider during submission.
    ///
    /// # Returns
    ///
    /// The results of the batch, including successes and failures.
    ///
    /// # Errors
    ///
    /// * `LLMError::BatchNotCompleted` - If the batch has not yet reached [`BatchJobState::Completed`] or [`BatchJobState::Failed`]
    /// * `LLMError::EnvError` - If an API key is not set up
    /// * `LLMError::InvalidHeaderError` - If the API key cannot be parsed as a header value
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403
    /// * `LLMError::HttpStatusError` - If the API returns another unsuccessful status code
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If the response cannot be parsed
    /// * `LLMError::GenericLLMError` - If a temporary file cannot be written
    async fn get_batch_results(&self, batch_id: &str) -> Result<BatchEmbeddingResults, LLMError> {
        match self {
            Self::VoyageAI(client) => client.get_batch_results(batch_id).await,
        }
    }

    async fn cancel_batch(&self, batch_id: &str) -> Result<(), LLMError> {
        match self {
            Self::VoyageAI(client) => client.cancel_batch(batch_id).await,
        }
    }
}
