use std::io;
use thiserror::Error;

use rag::{llm::errors::LLMError, vector::lance::LanceError};
use rustyline::error::ReadlineError;

use crate::utils;

#[derive(Debug, Error)]
pub enum CLIError {
    #[error("Error parsing library: {0}")]
    ArrowError(String),
    #[error("IO Error: {0}")]
    IOError(#[from] io::Error),
    #[error("Error communicating with the LLM: {0}")]
    LLMError(String),
    #[error("Malformed batch in batch_iter.bin")]
    MalformedBatchError,
    #[error("Error from readline: {0}")]
    ReadlineError(String),
    #[error("LanceDB error: {0}")]
    LanceError(String),
}

impl From<LanceError> for CLIError {
    fn from(value: LanceError) -> Self {
        Self::LanceError(value.to_string())
    }
}

impl From<lancedb::Error> for CLIError {
    fn from(value: lancedb::Error) -> Self {
        Self::LanceError(value.to_string())
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
