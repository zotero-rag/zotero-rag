//! Client definition for the Gemini API.

use std::env;

use crate::{
    http_client::{HttpClient, ReqwestClient},
    llm::errors::LLMError,
};

/// A client for Google's Gemini APIs (chat + embeddings)
#[derive(Debug, Clone)]
pub struct GeminiClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub client: T,
    /// Optional configuration for the Gemini client.
    pub config: Option<crate::config::GeminiConfig>,
}

impl<T: HttpClient + Default> Default for GeminiClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GeminiClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new GeminiClient instance without configuration
    /// (will fall back to environment variables)
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new GeminiClient instance with provided configuration
    #[must_use]
    pub fn with_config(config: crate::config::GeminiConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }
}

/// Get the Gemini API key from environment variables.
pub(crate) fn get_gemini_api_key() -> Result<String, LLMError> {
    // Prefer GEMINI_API_KEY, fallback to GOOGLE_API_KEY if present
    match env::var("GEMINI_API_KEY") {
        Ok(v) => Ok(v),
        Err(_) => Ok(env::var("GOOGLE_API_KEY")?),
    }
}
