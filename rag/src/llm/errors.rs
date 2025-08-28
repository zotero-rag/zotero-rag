use http::header::InvalidHeaderValue;
use thiserror::Error;

/// A wrapper for all kinds of errors to one enum that tells us what happened.
/// Variant error messages are handled via thiserror.
#[derive(Clone, Debug, Error)]
pub enum LLMError {
    #[error("Got 4xx response, possible credentials error: {0}")]
    CredentialError(String),
    #[error("Failed to deserialize response: {0}")]
    DeserializationError(String),
    #[error("Environment variable could not be fetched: {0}")]
    EnvError(String),
    #[error("Unknown error occurred: {0}")]
    GenericLLMError(String),
    #[error("Other HTTP status code error")]
    HttpStatusError,
    #[error("Invalid LLM provider: {0}")]
    InvalidProviderError(String),
    #[error("Invalid request header value: {0}")]
    InvalidHeaderError(String),
    #[error("LanceDB Error: {0}")]
    LanceError(String),
    #[error("A network connectivity error occurred")]
    NetworkError,
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

            return LLMError::HttpStatusError;
        } else if error.is_connect() {
            return LLMError::NetworkError;
        }

        LLMError::GenericLLMError(error.to_string())
    }
}

impl From<std::env::VarError> for LLMError {
    fn from(error: std::env::VarError) -> LLMError {
        LLMError::EnvError(error.to_string())
    }
}

impl From<serde_json::Error> for LLMError {
    fn from(error: serde_json::Error) -> LLMError {
        LLMError::DeserializationError(error.to_string())
    }
}

impl From<lancedb::Error> for LLMError {
    fn from(error: lancedb::Error) -> Self {
        LLMError::LanceError(error.to_string())
    }
}

impl From<InvalidHeaderValue> for LLMError {
    fn from(value: InvalidHeaderValue) -> Self {
        LLMError::InvalidHeaderError(value.to_string())
    }
}
