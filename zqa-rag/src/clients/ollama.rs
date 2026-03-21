//! Client definition for the ollama API.

use crate::http_client::{HttpClient, ReqwestClient};

/// Client for interacting with the Ollama API
#[derive(Debug, Clone)]
pub struct OllamaClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub(crate) client: T,
    /// Optional configuration for the `ollama` client.
    pub(crate) config: Option<crate::config::OllamaConfig>,
}

impl<T: HttpClient + Default> Default for OllamaClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OllamaClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new `OllamaClient` instance without configuration
    /// (will fall back to environment variables)
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new `OllamaClient` instance with provided configuration
    #[must_use]
    pub fn with_config(config: crate::config::OllamaConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }
}
