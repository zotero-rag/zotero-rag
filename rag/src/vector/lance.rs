use crate::llm::{
    anthropic::AnthropicClient, 
    http_client::ReqwestClient, 
    openai::OpenAIClient
};

use super::arrow::{library_to_arrow, ArrowError};
use core::fmt;
use lancedb::{connect, connection::CreateTableMode, Connection, Error as LanceDbError};
use std::{error::Error, sync::Arc};

/// Errors that can occur when working with LanceDB
#[derive(Debug)]
pub enum LanceError {
    /// Error connecting to LanceDB
    ConnectionError(String),
    /// Error creating a table in LanceDB
    TableCreationError(String),
    /// Error with Arrow data
    ArrowError(ArrowError),
    /// Other LanceDB-related errors
    Other(String),
}

impl fmt::Display for LanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionError(msg) => write!(f, "LanceDB connection error: {}", msg),
            Self::TableCreationError(msg) => write!(f, "LanceDB table creation error: {}", msg),
            Self::ArrowError(e) => write!(f, "Arrow error: {}", e),
            Self::Other(msg) => write!(f, "LanceDB error: {}", msg),
        }
    }
}

impl Error for LanceError {}

// Convert from ArrowError to LanceError
impl From<ArrowError> for LanceError {
    fn from(err: ArrowError) -> Self {
        Self::ArrowError(err)
    }
}

// Convert from LanceDB's error to our LanceError
impl From<LanceDbError> for LanceError {
    fn from(err: LanceDbError) -> Self {
        Self::Other(err.to_string())
    }
}

/// Creates and initializes a LanceDB table for vector storage
/// 
/// Connects to LanceDB at the default location, creates a table named "data",
/// and registers embedding functions for both Anthropic and OpenAI.
///
/// # Returns
/// A Connection to the LanceDB database if successful
///
/// # Errors
/// Returns a `LanceError` if connection, table creation, or registering embedding functions fails
pub async fn create_initial_table() -> Result<Connection, LanceError> {
    let uri = "data/lancedb-table";

    // Connect to LanceDB
    let db = connect(uri)
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    // Get Arrow data - error will be automatically converted using From implementation
    let data = library_to_arrow()?;

    // Create the table
    let _tbl = db
        .create_table("data", data)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .map_err(|e| LanceError::TableCreationError(e.to_string()))?;

    db.embedding_registry()
        .register("anthropic", Arc::new(AnthropicClient::<ReqwestClient>::default()))?;

    db.embedding_registry()
        .register("openai", Arc::new(OpenAIClient::default()))?;

    Ok(db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_initial_table() {
        let db = create_initial_table().await;

        assert!(db.is_ok());
        let db = db.unwrap();

        let tbl_names = db.table_names().execute().await;
        assert!(tbl_names.is_ok());
        assert_eq!(tbl_names.unwrap(), vec!["data"]);

        let tbl = db.open_table("data").execute().await;
        assert!(tbl.is_ok());
    }
}
