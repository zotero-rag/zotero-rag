use std::io;
use thiserror::Error;

use rag::llm::errors::LLMError;
use rustyline::error::ReadlineError;

use crate::utils;

#[derive(Clone, Debug, Error)]
pub enum CLIError {
    #[error("Error parsing library: {0}")]
    ArrowError(String),
    #[error("IO Error: {0}")]
    IOError(String),
    #[error("Error communicating with the LLM: {0}")]
    LLMError(String),
    #[error("Malformed batch in batch_iter.bin")]
    MalformedBatchError,
    #[error("Error from the readline implementation: {0}")]
    ReadlineError(String),
}

impl From<io::Error> for CLIError {
    fn from(value: io::Error) -> Self {
        Self::IOError(value.to_string())
    }
}

impl From<utils::arrow::ArrowError> for CLIError {
    fn from(value: utils::arrow::ArrowError) -> Self {
        Self::ArrowError(value.to_string())
    }
}

impl From<arrow_schema::ArrowError> for CLIError {
    fn from(value: arrow_schema::ArrowError) -> Self {
        Self::ArrowError(value.to_string())
    }
}

impl From<&arrow_schema::ArrowError> for CLIError {
    fn from(value: &arrow_schema::ArrowError) -> Self {
        Self::ArrowError(value.to_string())
    }
}

impl From<LLMError> for CLIError {
    fn from(value: LLMError) -> Self {
        Self::LLMError(value.to_string())
    }
}

impl From<ReadlineError> for CLIError {
    fn from(value: ReadlineError) -> Self {
        Self::ReadlineError(value.to_string())
    }
}
