//! This module provides the `LLMError` enum, which is the main error type returned by functions in
//! this crate.

use http::header::InvalidHeaderValue;
use thiserror::Error;

/// A wrapper for all kinds of errors to one enum that tells us what happened.
/// Variant error messages are handled via thiserror.
#[derive(Debug, Error)]
pub enum LLMError {
    /// API errors that likely represent credential errors.
    #[error("Got 4xx response, possible credentials error: {0}")]
    CredentialError(String),
    /// Errors resulting from failed deserialization. This usually means that the API returned
    /// invalid JSON.
    #[error("Failed to deserialize response: {0}")]
    DeserializationError(String),
    /// Errors indicating a necessary environment variable is missing.
    #[error("Environment variable could not be fetched: {0}")]
    EnvError(#[from] std::env::VarError),
    /// A broad "other" error. This is also used for cases when we don't know what went wrong.
    #[error("Unknown error occurred: {0}")]
    GenericLLMError(String),
    /// An unsuccessful HTTP response that is not a credential error or a timeout.
    #[error("Other HTTP status code error: {0}")]
    HttpStatusError(String),
    /// An error indicating an invalid provider was specified.
    #[error("Invalid LLM provider: {0}")]
    InvalidProviderError(String),
    /// An error indicating an invalid header value was specified.
    #[error("Invalid request header value: {0}")]
    InvalidHeaderError(#[from] InvalidHeaderValue),
    /// A wrapper around all LanceDB errors. The wrapped error should contain more details.
    #[error("LanceDB Error: {0}")]
    LanceError(#[from] lancedb::Error),
    /// An error indicating a tool call failed.
    #[error("Error calling a tool: {0}")]
    ToolCallError(String),
    /// A network connectivity error.
    #[error("A network connectivity error occurred")]
    NetworkError,
    /// A request timeout error.
    #[error("Request timed out")]
    TimeoutError,
}

/// From<...> implementations begin here
impl From<reqwest::Error> for LLMError {
    fn from(error: reqwest::Error) -> LLMError {
        if error.is_timeout() {
            return LLMError::TimeoutError;
        } else if let Some(status) = error.status() {
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return LLMError::CredentialError(error.to_string());
            }

            return LLMError::HttpStatusError(error.to_string());
        } else if error.is_connect() {
            return LLMError::NetworkError;
        }

        LLMError::GenericLLMError(error.to_string())
    }
}

impl From<serde_json::Error> for LLMError {
    fn from(error: serde_json::Error) -> LLMError {
        LLMError::DeserializationError(error.to_string())
    }
}
