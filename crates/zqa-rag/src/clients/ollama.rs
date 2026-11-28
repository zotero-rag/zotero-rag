//! Client definition for the Ollama API.

use crate::http_client::{HttpClient, ReqwestClient};

/// Client for interacting with the Ollama API
#[derive(Debug, Clone)]
pub struct OllamaClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub(crate) client: T,
    /// Configuration for the `ollama` client.
    pub(crate) config: crate::config::OllamaConfig,
}

impl<T> OllamaClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new `OllamaClient` instance with provided configuration
    #[must_use]
    pub fn with_config(config: crate::config::OllamaConfig) -> Self {
        Self {
            client: T::default(),
            config,
        }
    }
}
