use crate::llm::{
    anthropic::AnthropicClient, base::ModelProviders, http_client::ReqwestClient,
    openai::OpenAIClient,
};

use arrow_array::{RecordBatch, RecordBatchIterator};
use core::fmt;
use lancedb::{
    arrow::arrow_schema::ArrowError, connect, connection::CreateTableMode,
    embeddings::EmbeddingDefinition, Connection, Error as LanceDbError,
};
use std::{error::Error, sync::Arc, vec::IntoIter};

/// Errors that can occur when working with LanceDB
#[derive(Debug)]
pub enum LanceError {
    /// Error connecting to LanceDB
    ConnectionError(String),
    /// Error creating a table in LanceDB
    TableCreationError(String),
    /// Invalid params
    ParameterError(String),
    /// Other LanceDB-related errors
    Other(String),
}

impl fmt::Display for LanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionError(msg) => write!(f, "LanceDB connection error: {msg}"),
            Self::TableCreationError(msg) => write!(f, "LanceDB table creation error: {msg}"),
            Self::ParameterError(msg) => write!(f, "Invalid parameter: {msg}"),
            Self::Other(msg) => write!(f, "LanceDB error: {msg}"),
        }
    }
}

impl Error for LanceError {}

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
/// # Arguments
///
/// * `data`: An Arrow `RecordBatchIterator` containing data. See
///   https://docs.rs/lancedb/latest/lancedb/index.html for an example of creating this.
/// * `embedding_params`: An `EmbeddingDefinition` object that contains the source column that has
///   the text data, the destination column name, and the embedding function to use.
///
/// # Returns
/// A Connection to the LanceDB database if successful
///
/// # Errors
/// Returns a `LanceError` if connection, table creation, or registering embedding functions fails
pub async fn create_initial_table(
    data: RecordBatchIterator<IntoIter<Result<RecordBatch, ArrowError>>>,
    embedding_params: EmbeddingDefinition,
) -> Result<Connection, LanceError> {
    let uri = "data/lancedb-table";

    if !(ModelProviders::contains(&embedding_params.embedding_name)) {
        return Err(LanceError::ParameterError(format!(
            "{} is not a valid embedding.",
            embedding_params.embedding_name
        )));
    }

    // Connect to LanceDB
    let db = connect(uri)
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    match embedding_params.embedding_name.as_str() {
        "anthropic" => {
            db.embedding_registry().register(
                ModelProviders::Anthropic.as_str(),
                Arc::new(AnthropicClient::<ReqwestClient>::default()),
            )?;
        }
        "openai" => {
            db.embedding_registry().register(
                ModelProviders::OpenAI.as_str(),
                Arc::new(OpenAIClient::default()),
            )?;
        }
        _ => unreachable!(
            "Unknown embedding provider {}",
            embedding_params.embedding_name
        ),
    }

    // Create the table
    let _tbl = db
        .create_table("data", data)
        .mode(CreateTableMode::Overwrite)
        .add_embedding(embedding_params)?
        .execute()
        .await
        .map_err(|e| LanceError::TableCreationError(e.to_string()))?;

    Ok(db)
}

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use dotenv::dotenv;
    use futures::StreamExt;
    use lancedb::query::ExecutableQuery;

    use super::*;

    #[tokio::test]
    async fn test_create_initial_table_with_openai() {
        dotenv().ok();

        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "data_openai",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data)]).unwrap();
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        let db = create_initial_table(
            reader,
            EmbeddingDefinition::new(
                "data_openai",      // source column
                "openai",           // embedding name, either "openai" or "anthropic"
                Some("embeddings"), // dest column
            ),
        )
        .await;

        assert!(db.is_ok());
        let db = db.unwrap();

        let tbl_names = db.table_names().execute().await;
        assert!(tbl_names.is_ok());
        assert_eq!(tbl_names.unwrap(), vec!["data"]);

        let tbl = db.open_table("data").execute().await;
        assert!(tbl.is_ok());

        let tbl = tbl.unwrap();
        let tbl_values = tbl.query().execute().await;

        assert!(tbl_values.is_ok());

        let mut tbl_values = tbl_values.unwrap();
        let row = tbl_values.next().await;

        assert!(row.is_some());
        let row = row.unwrap();

        assert!(row.is_ok());
        let row = row.unwrap();

        for column in ["data_openai", "embeddings"] {
            assert!(row.column_by_name(column).is_some());
        }
    }

    #[tokio::test]
    async fn test_create_initial_table_with_anthropic() {
        dotenv().ok();

        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "data_anthropic",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data)]).unwrap();
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        let db = create_initial_table(
            reader,
            EmbeddingDefinition::new("data_anthropic", "anthropic", Some("embeddings")),
        )
        .await;

        assert!(db.is_ok());
        let db = db.unwrap();

        let tbl_names = db.table_names().execute().await;
        assert!(tbl_names.is_ok());
        assert_eq!(tbl_names.unwrap(), vec!["data"]);

        let tbl = db.open_table("data").execute().await;
        assert!(tbl.is_ok());

        let tbl = tbl.unwrap();
        let tbl_values = tbl.query().execute().await;

        assert!(tbl_values.is_ok());

        let mut tbl_values = tbl_values.unwrap();
        let row = tbl_values.next().await;

        assert!(row.is_some());
        let row = row.unwrap();

        assert!(row.is_ok());
        let row = row.unwrap();

        for column in ["data_anthropic", "embeddings"] {
            assert!(row.column_by_name(column).is_some());
        }
    }

    #[tokio::test]
    async fn test_invalid_embedding_provider_rejected() {
        dotenv().ok();

        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "data",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data)]).unwrap();
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        let db = create_initial_table(
            reader,
            EmbeddingDefinition::new("data", "invalid", Some("embeddings")),
        )
        .await;

        assert!(db.is_err());
    }
}
