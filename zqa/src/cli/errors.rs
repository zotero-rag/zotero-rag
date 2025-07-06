use std::io;

use crate::utils;

#[derive(Clone, Debug)]
pub enum CLIError {
    IOError(String),
    ArrowError(String),
}

impl std::error::Error for CLIError {}
impl std::fmt::Display for CLIError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::IOError(msg) => write!(f, "IO Error: {msg}"),
            Self::ArrowError(msg) => write!(f, "Error parsing library: {msg}"),
        }
    }
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
