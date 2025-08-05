use http::header::InvalidHeaderValue;

/// A wrapper for all kinds of errors to one enum that tells us what happened.
/// Has implementations of From<...> and Display
#[derive(Clone, Debug)]
pub enum LLMError {
    CredentialError(String),
    DeserializationError(String),
    EnvError(String),
    GenericLLMError(String),
    HttpStatusError,
    InvalidProviderError(String),
    InvalidHeaderError(String),
    LanceError(String),
    NetworkError,
    TimeoutError,
}

impl std::error::Error for LLMError {}
impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LLMError::CredentialError(msg) => {
                write!(f, "Got 4xx response, possible credentials error: {msg}")
            }
            LLMError::DeserializationError(body) => {
                write!(f, "Failed to deserialize response: {body}")
            }
            LLMError::EnvError(msg) => {
                write!(f, "Environment variable could not be fetched: {msg}")
            }
            LLMError::GenericLLMError(msg) => {
                write!(f, "Unknown error occurred: {msg}")
            }
            LLMError::HttpStatusError => write!(f, "Other HTTP status code error"),
            LLMError::InvalidHeaderError(msg) => {
                write!(f, "Invalid request header value: {msg}")
            }
            LLMError::InvalidProviderError(provider) => {
                write!(f, "Invalid LLM provider: {provider}")
            }
            LLMError::LanceError(msg) => write!(f, "LanceDB Error: {msg}"),
            LLMError::NetworkError => write!(f, "A network connectivity error occurred"),
            LLMError::TimeoutError => write!(f, "Request timed out"),
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
