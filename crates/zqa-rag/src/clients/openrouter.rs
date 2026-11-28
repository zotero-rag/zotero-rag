//! Client definition for the OpenRouter API.

use crate::http_client::{HttpClient, ReqwestClient};

/// A generic client class for now. We can add stuff here later for
/// all the features OpenRouter supports.
#[derive(Debug, Clone)]
pub struct OpenRouterClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub(crate) client: T,
    /// Configuration for the OpenRouter client.
    pub(crate) config: crate::config::OpenRouterConfig,
}

impl<T> OpenRouterClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new OpenRouterClient instance with provided configuration
    #[must_use]
    pub fn with_config(config: crate::config::OpenRouterConfig) -> Self {
        Self {
            client: T::default(),
            config,
        }
    }
}
