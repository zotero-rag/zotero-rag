//! Client definition for the Gemini API.

use crate::http_client::{HttpClient, ReqwestClient};

/// A client for Google's Gemini APIs (chat + embeddings)
#[derive(Debug, Clone)]
pub struct GeminiClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub(crate) client: T,
    /// Configuration for the Gemini client.
    pub(crate) config: crate::config::GeminiConfig,
}

impl<T> GeminiClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new GeminiClient instance with provided configuration
    #[must_use]
    pub fn with_config(config: crate::config::GeminiConfig) -> Self {
        Self {
            client: T::default(),
            config,
        }
    }
}
