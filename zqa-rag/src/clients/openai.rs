//! Client definition for the OpenAI API.

use crate::http_client::{HttpClient, ReqwestClient};

/// A client for OpenAI's chat completions (Responses) API.
#[derive(Debug, Clone)]
pub struct OpenAIClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub(crate) client: T,
    /// Optional configuration for the OpenAI client.
    pub(crate) config: Option<crate::config::OpenAIConfig>,
}

impl<T: HttpClient + Default> Default for OpenAIClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OpenAIClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new `OpenAIClient` instance without configuration.
    ///
    /// Uses environment variables for configuration.
    ///
    /// # Returns
    ///
    /// A client initialized with a default HTTP client and no config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new `OpenAIClient` instance with provided configuration.
    ///
    /// # Arguments:
    ///
    /// * `config` - Client configuration including API key and model.
    ///
    /// # Returns
    ///
    /// A client initialized with a default HTTP client and the given config.
    #[must_use]
    pub fn with_config(config: crate::config::OpenAIConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }
}
