//! Client definition for the Anthropic API.

use crate::http_client::{HttpClient, ReqwestClient};

/// A generic client class for now. We can add stuff here later if needed, for
/// example, features like Anthropic's native RAG thing
#[derive(Debug, Clone)]
pub struct AnthropicClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub client: T,
    /// Optional configuration for the Anthropic client.
    pub config: Option<crate::config::AnthropicConfig>,
}

impl<T: HttpClient + Default> Default for AnthropicClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AnthropicClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new AnthropicClient instance without configuration
    /// (will fall back to environment variables)
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new AnthropicClient instance with provided configuration
    #[must_use]
    pub fn with_config(config: crate::config::AnthropicConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }
}
