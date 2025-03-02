#[derive(Debug)]
pub enum LLMError {
    NetworkError,
}

impl std::error::Error for LLMError {}
impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LLMError::NetworkError => write!(f, "Could not find .env file"),
        }
    }
}
