/// A wrapper for all kinds of errors to one enum that tells us what happened.
/// Has implementations of From<...> and Display
#[derive(Debug, Clone)]
pub enum LLMError {
    TimeoutError,
    CredentialError,
    GenericLLMError(String),
    NetworkError,
    HttpStatusError,
    EnvError,
    DeserializationError(String),
    InvalidProviderError(String),
}

impl std::error::Error for LLMError {}
impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LLMError::TimeoutError => write!(f, "Request timed out"),
            LLMError::CredentialError => write!(f, "Got 4xx response, possible credentials error"),
            LLMError::GenericLLMError(msg) => {
                write!(f, "Unknown error occurred: {}", msg)
            }
            LLMError::NetworkError => write!(f, "A network connectivity error occurred"),
            LLMError::HttpStatusError => write!(f, "Other HTTP status code error"),
            LLMError::EnvError => write!(f, "Environment variable could not be fetched"),
            LLMError::DeserializationError(body) => {
                write!(f, "Failed to deserialize response: {}", body)
            }
            LLMError::InvalidProviderError(provider) => {
                write!(f, "Invalid LLM provider: {}", provider)
            }
        }
    }
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
                return LLMError::CredentialError;
            }

            return LLMError::HttpStatusError;
        } else if error.is_connect() {
            return LLMError::NetworkError;
        }

        LLMError::GenericLLMError(error.to_string())
    }
}

impl From<std::env::VarError> for LLMError {
    fn from(_error: std::env::VarError) -> LLMError {
        LLMError::EnvError
    }
}

impl From<serde_json::Error> for LLMError {
    fn from(error: serde_json::Error) -> LLMError {
        LLMError::DeserializationError(error.to_string())
    }
}
