//! Functions for working with LanceDB. This module includes all the functionality for using
//! LanceDB's features, including connecting, inserting, querying, and deleting. For certain
//! operations, variants with backup support are provided.

use crate::embedding::common::EmbeddingProviderConfig;
use crate::providers::provider_id::ProviderId;
use crate::providers::registry::provider_registry;
use crate::vector::backup::with_backup;
use crate::vector::checkhealth::get_zero_vectors;

use arrow_array::{
    RecordBatch, RecordBatchIterator, StringArray, cast::AsArray, types::Float32Type,
};

use futures::TryStreamExt;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::{
    Connection, Error as LanceDbError, arrow::arrow_schema::ArrowError, connect,
    database::CreateTableMode, embeddings::EmbeddingDefinition, query::ExecutableQuery,
    query::QueryBase,
};
use std::collections::HashSet;
use std::{fmt::Display, path::PathBuf, sync::Arc, time::Instant};
use thiserror::Error;

// NOTE: Maintainers: ensure that `DB_URI` begins with `TABLE_NAME`

/// The URI for the LanceDB table. This is the default location for the table, and for now cannot
/// be changed.
pub const DB_URI: &str = "data/lancedb-table";

/// Returns the database URI, allowing override via `LANCEDB_URI` environment variable.
#[must_use]
pub fn get_db_uri() -> String {
    std::env::var("LANCEDB_URI").unwrap_or_else(|_| DB_URI.to_string())
}

/// The name of the table. This is the default table name, and for now cannot be changed.
pub const TABLE_NAME: &str = "data";

/// Errors that can occur when working with LanceDB
#[derive(Debug, Error)]
pub enum LanceError {
    /// Error connecting to LanceDB
    #[error("LanceDB connection error: {0}")]
    ConnectionError(String),
    /// Error running some query
    #[error("Failed to execute query: {0}")]
    QueryError(String),
    /// Error creating or updating a table in LanceDB
    #[error("LanceDB table update error: {0}")]
    TableUpdateError(String),
    /// Invalid params
    #[error("Invalid parameter: {0}")]
    ParameterError(String),
    /// The database is in an invalid state
    #[error("The DB is in an invalid state: {0}")]
    InvalidStateError(String),
    /// IO errors, used by repair.rs.
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    /// Other LanceDB-related errors
    #[error("Other LanceDB error: {0}")]
    Other(Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl From<ArrowError> for LanceError {
    fn from(value: ArrowError) -> Self {
        Self::Other(Box::new(value))
    }
}

impl From<LanceDbError> for LanceError {
    fn from(value: LanceDbError) -> Self {
        Self::Other(Box::new(value))
    }
}

/// Table statistics, to be shown when the user calls the `/stats` command.
#[derive(Debug, PartialEq)]
pub struct TableStatistics {
    /// The table version. Each update to a table creates a new version.
    table_version: u64,
    /// Number of rows in the table
    num_rows: usize,
}

impl Display for TableStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Table statistics:\n")?;
        f.write_str(&format!("\tTable version: {}\n", self.table_version))?;
        f.write_str(&format!("\tNumber of rows: {}\n", self.num_rows))?;

        Ok(())
    }
}

/// Registration methods for embedding providers to interface with LanceDB
pub trait LanceEmbeddingRegistrar: Send + Sync {
    /// Get the provider ID for this object
    fn provider_id(&self) -> ProviderId;

    /// Register this provider with LanceDB
    ///
    /// # Errors
    ///
    /// Return a [`LanceError`] if registration fails.
    fn register_with_lancedb(
        &self,
        db: &lancedb::Connection,
        config: &EmbeddingProviderConfig,
    ) -> Result<(), LanceError>;
}

/// Checks if an existing LanceDB exists and has a valid table
pub async fn lancedb_exists() -> bool {
    let uri = get_db_uri();
    if !PathBuf::from(&uri).exists() {
        return false;
    }

    // Check if we can actually connect and open the table
    let db_result = connect(&uri).execute().await;
    if let Ok(db) = db_result {
        db.open_table(TABLE_NAME).execute().await.is_ok()
    } else {
        false
    }
}

/// Create indices for the database. This function creates two indices: an IVF-PQ index on the
/// embedding column, and a FTS index on the text column. This allows for full-text search as well
/// as more efficient vector searches. If indices already exist, they are optimized instead.
///
/// # Arguments:
///
/// * `text_col` - The name of the column containing text
/// * `embedding_col` - The name of the embedding column
///
/// # Errors
///
/// Returns an error if the database connection fails, table operations fail, or index creation fails.
pub async fn create_or_update_indexes(
    text_col: &str,
    embedding_col: &str,
) -> Result<(), LanceError> {
    let db = connect(&get_db_uri())
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    let tbl = db.open_table(TABLE_NAME).execute().await?;

    let indices = tbl.list_indices().await?;
    let has_vector_index = indices
        .iter()
        .any(|i| i.columns.as_slice() == [embedding_col]);
    let has_fts_index = indices.iter().any(|i| i.columns.as_slice() == [text_col]);

    if !has_vector_index {
        tbl.create_index(&[embedding_col], lancedb::index::Index::Auto)
            .execute()
            .await?;
    }

    if !has_fts_index {
        // Note that currently, multi-column indexes are not supported by LanceDB.
        tbl.create_index(
            &[text_col],
            lancedb::index::Index::FTS(FtsIndexBuilder::default()),
        )
        .execute()
        .await?;
    }

    // If both indices already existed before this run, they could be optimized here
    // but optimization is optional and can be done separately.

    Ok(())
}

/// Connects to the database and prints out simple statistics.
///
/// # Errors
/// Returns a `LanceError` if
/// - the DB cannot be found
/// - the DB could not be connected to
/// - the table could not be opened
/// - the table statistics could not be computed
pub async fn db_statistics() -> Result<TableStatistics, LanceError> {
    let db = connect(&get_db_uri())
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    let tbl = db.open_table(TABLE_NAME).execute().await?;
    let table_version = tbl.version().await?;
    let num_rows = tbl.count_rows(None).await?;

    Ok(TableStatistics {
        table_version,
        num_rows,
    })
}

/// Connect to the Lance database and add the embedding `embedding_name` to the database's
/// embedding registry. Note that if we are opening a database that had one defined, we should use
/// the same one here. Since that isn't stored (at least, not that I know of), this onus is on the
/// user.
///
/// # Arguments
/// - `embedding_name`: The embedding method to use. Must be in `EmbeddingProviders`.
///
/// # Returns
/// - `db`: A `Connection` object to the Lance database.
async fn get_db_with_embeddings(
    embedding_config: &EmbeddingProviderConfig,
) -> Result<Connection, LanceError> {
    let db = connect(&get_db_uri()).execute().await?;
    let registry = provider_registry();

    registry.register_embedding_with_lancedb(&db, embedding_config)?;
    Ok(db)
}

/// Given a column name and a list of key values, delete all matching rows from the database.
///
/// # Arguments
///
/// * `column` - The name of the column to match against.
/// * `keys` - The list of values to delete. Rows whose `column` value is in this list are deleted.
/// * `embedding_config` - The embedding provider configuration.
///
/// # Errors
///
/// * `LanceError::ConnectionError` - If the database connection fails
/// * `LanceError::InvalidStateError` - If the table doesn't exist
/// * `LanceError::Other` - If the delete query fails
pub async fn delete_rows(
    column: &str,
    keys: &[impl AsRef<str>],
    embedding_config: &EmbeddingProviderConfig,
) -> Result<(), LanceError> {
    if keys.is_empty() {
        return Ok(());
    }

    let db = get_db_with_embeddings(embedding_config).await?;
    let table = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    for key_chunk in keys.chunks(100) {
        let delete_pred = key_chunk
            .iter()
            .map(|k| format!("{column} = '{}'", k.as_ref().replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(" OR ");

        table.delete(&delete_pred).await?;
    }

    Ok(())
}

/// Return `RecordBatch`es for records with zero embedding vectors.
///
/// # Arguments
///
/// * `embedding_config` - The embedding provider configuration
///
/// # Errors
///
/// * `LanceError::ConnectionError` - If the database connection fails
/// * `LanceError::InvalidStateError` - If the table doesn't exist
/// * `LanceError::Other` - If querying for zero vectors fails
pub async fn get_zero_vector_records(
    embedding_config: &EmbeddingProviderConfig,
) -> Result<Vec<RecordBatch>, LanceError> {
    let db = get_db_with_embeddings(embedding_config).await?;
    let table = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    // Get all records with zero embeddings
    let zero_batches = get_zero_vectors(&table, embedding_config.provider_id(), 10000).await?;
    Ok(zero_batches)
}

/// From a `RecordBatch`, return all values from a specified column as a `Vec<String>`.
///
/// # Arguments
///
/// * `batch`: A reference to a `RecordBatch`.
/// * `column`: The index of the column to use.
///
/// # Returns
///
/// A `Vec<String>` containing all the items in the specified column of the `RecordBatch`.
#[must_use]
pub fn get_column_from_batch(batch: &RecordBatch, column: usize) -> Vec<String> {
    let results = batch.column(column).as_string::<i32>();

    results
        .iter()
        .filter_map(|s| Some(s?.to_string()))
        .collect()
}

/// Deduplicate rows in the LanceDB table based on a 'by' column, keeping only the first
/// occurrence of each unique 'by' value and deleting the duplicates.
///
/// # Arguments
///
/// * `embedding_config` - The embedding provider configuration
/// * `schema` - Arrow schema for the table
/// * `by` - Column name to deduplicate by
/// * `key` - Column name to use as the key for deletion
///
/// # Returns
///
/// Number of rows deleted
///
/// # Errors
///
/// * `LanceError::InvalidStateError` if the table cannot be opened
/// * `LanceError::ConnectionError` if the database connection fails
/// * `LanceError::InvalidStateError` if the table doesn't exist
/// * `LanceError::ParameterError` if the key column is not found in the rows
/// * `LanceError::Other` for other LanceDB errors
pub async fn dedup_rows(
    embedding_config: &EmbeddingProviderConfig,
    schema: arrow_schema::Schema,
    by: &str,
    key: &str,
) -> Result<usize, LanceError> {
    let db = get_db_with_embeddings(embedding_config).await?;
    let table = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    if let Some((by_idx, _)) = schema.column_with_name(by)
        && let Some((key_idx, _)) = schema.column_with_name(key)
    {
        let mut stream = table.query().execute().await?;
        let mut seen_by_values = HashSet::new();
        let mut duplicate_keys = Vec::new();

        while let Some(batch) = stream.try_next().await? {
            let by_values = get_column_from_batch(&batch, by_idx);
            let key_values = get_column_from_batch(&batch, key_idx);

            for (by_val, key_val) in by_values.into_iter().zip(key_values) {
                if !seen_by_values.insert(by_val) {
                    duplicate_keys.push(key_val);
                }
            }
        }

        // Delete duplicate rows
        let deleted_count = duplicate_keys.len();
        if !duplicate_keys.is_empty() {
            delete_rows(key, &duplicate_keys, embedding_config).await?;
        }

        return Ok(deleted_count);
    }

    Ok(0)
}

/// Return all the rows in the LanceDB table, selecting only the columns specified. This is useful
/// for computing what rows do *not* exist in the table, such as in cases where the data source has
/// new items. Note that LanceDB by default seems to chunk by 1024 items, so if you're certain you
/// will receive fewer than 1024 rows, it is safe to assume only one element exists.
///
/// # Arguments
/// * embedding_config: The embedding provider configuration
/// * columns: The set of columns to return. Typically, you do not want to include the full-text
///   column here.
///
/// # Returns
/// * batches: A vector of Arrow `RecordBatch` objects containing the results.
///
/// # Errors
///
/// * `LanceError::ConnectionError` - If the database connection fails
/// * `LanceError::InvalidStateError` - If the table doesn't exist
/// * `LanceError::Other` - If the query execution fails
pub async fn get_lancedb_items(
    embedding_config: &EmbeddingProviderConfig,
    columns: Vec<String>,
) -> Result<Vec<RecordBatch>, LanceError> {
    let db = get_db_with_embeddings(embedding_config).await?;

    let tbl = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    // The installed version of LanceDB has a bug where without the `.limit` call here, it only
    // returns 10 rows; see https://github.com/lancedb/lancedb/issues/1852#issuecomment-2489837804
    let results: Vec<RecordBatch> = tbl
        .query()
        .select(lancedb::query::Select::Columns(columns))
        .limit(tbl.count_rows(None).await?)
        .execute()
        .await?
        .try_collect()
        .await?;

    Ok(results)
}

/// Perform a vector search on the database using the `query` and the `embedding_name` embedding
/// method. Returns a vector of Arrow `RecordBatch` objects containing the results.
///
/// # Arguments
///
/// * `query` - The query string to search for
/// * `embedding_config` - The embedding provider configuration
/// * `limit` - Limit on the number of returned results
///
/// # Returns
///
/// A vector of Arrow `RecordBatch` objects containing the results.
///
/// # Errors
///
/// * `LanceError::ParameterError` - If the embedding provider is not recognized
/// * `LanceError::ConnectionError` - If the database connection fails
/// * `LanceError::InvalidStateError` - If the table doesn't exist or embedding provider not in registry
/// * `LanceError::Other` - If embedding computation or query execution fails
pub async fn vector_search(
    query: String,
    embedding_config: &EmbeddingProviderConfig,
    limit: usize,
) -> Result<Vec<RecordBatch>, LanceError> {
    let start_time = Instant::now();
    let db = get_db_with_embeddings(embedding_config).await?;

    let tbl = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;
    log::debug!("Opening the DB and table took {:.1?}", start_time.elapsed());

    let start_time = Instant::now();
    let embedding = db
        .embedding_registry()
        .get(embedding_config.provider_id().as_str())
        .ok_or(LanceError::InvalidStateError(format!(
            "{} is not in the database embedding registry",
            embedding_config.provider_id().as_str()
        )))?;

    let query_vec = embedding.compute_query_embeddings(Arc::new(StringArray::from(vec![query])))?;
    log::debug!("Computing embeddings took {:.1?}", start_time.elapsed());

    // Convert FixedSizeListArray to Vec<f32>
    // The embedding functions return `FixedSizeListArray` with Float32 elements
    // See https://github.com/apache/arrow-rs/discussions/6087#discussioncomment-10851422 for
    // converting an Arrow Array to a `Vec`.
    let query_vec: Vec<f32> = {
        let list_array = arrow_array::cast::as_fixed_size_list_array(&query_vec);
        let values = list_array.values().as_primitive::<Float32Type>();
        values.iter().map(|v| v.unwrap_or(0.0)).collect()
    };

    let start_time = Instant::now();
    let stream = tbl
        .query()
        .limit(limit)
        .nearest_to(query_vec)?
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    log::debug!("Vector search took {:.1?}", start_time.elapsed());

    Ok(batches)
}

/// Given the name of the key column and the value of a row's key, return the row with that key if
/// it exists, or `None` otherwise. The returned row is returned as a single `RecordBatch`.
///
/// Technically, there's nothing *requiring* you to specify a key as a column name; you could very
/// well specify any arbitrary column and use this, but note that even if multiple rows match, this
/// limits the result to one item, and that returned value may change across LanceDB versions, so
/// you should not rely on that behavior.
///
/// # Arguments
///
/// * `key_col` - The name of the column that's the DB key.
/// * `key` - The value of the key for the row to retrieve.
///
/// # Returns
///
/// The row with the specified `key`, if it exists, or `None` if no matching row was found.
///
/// # Errors
///
/// Returns a [`LanceError`] if the database connection fails, the table cannot be opened,
/// the query cannot be executed, or the result stream cannot be collected.
pub async fn search_by_key(key_col: &str, key: &str) -> Result<Option<RecordBatch>, LanceError> {
    let db = connect(&get_db_uri())
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    let tbl = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    let stream = tbl
        .query()
        .only_if(format!("{key_col} = '{}'", key.replace('\'', "''")))
        .limit(1)
        .execute()
        .await
        .map_err(|e| LanceError::QueryError(e.to_string()))?;

    let batches: Vec<_> = stream
        .try_collect()
        .await
        .map_err(|e| LanceError::QueryError(e.to_string()))?;

    Ok(batches.first().cloned())
}

/// Given the name of a column and some values, return the rows with any matching values.
///
/// # Arguments
///
/// * `col` - The name of the column that's the DB key.
/// * `values` - The values to match.
///
/// # Returns
///
/// The rows with the specified `values`.
///
/// # Errors
///
/// * `LanceError::ConnectionError` - If the DB connection failed.
/// * `LanceError::InvalidStateError` - If opening the table failed.
/// * `LanceError::Other` - If query execution failed.
pub async fn search_by_column(
    col: &str,
    values: &[impl AsRef<str> + Display],
) -> Result<Vec<RecordBatch>, LanceError> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let db = connect(&get_db_uri())
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    let tbl = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    let queries = values
        .iter()
        .map(|key| format!("{col} = '{}'", key.as_ref().replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(" OR ");

    let stream = tbl.query().only_if(queries).execute().await?;
    stream.try_collect().await.map_err(Into::into)
}

/// Creates and initializes a LanceDB table for vector storage.
///
/// Connects to LanceDB at the default location, creates a table named `TABLE_NAME`,
/// and registers embedding functions for OpenAI. If this table already exists,
/// it simply opens it and upserts the data; otherwise, it inserts the data into the newly created
/// table.
///
/// # Arguments
///
/// * `data` -  An Arrow `RecordBatchIterator` containing data. See
///   https://docs.rs/lancedb/latest/lancedb/index.html for an example of creating this.
/// * `merge_on` - `None` if you want to create or overwrite the current database; otherwise, a
///   reference to an array of keys to merge on.
/// * `embedding_config` - The embedding provider configuration
/// * `source_col` - The name of the column containing the source text to embed.
///
/// # Returns
/// A Connection to the LanceDB database if successful
///
/// # Errors
/// Returns a `LanceError` if connection, table creation, or registering embedding functions fails
pub async fn insert_records(
    data: Vec<RecordBatch>,
    merge_on: Option<&[&str]>,
    embedding_config: &EmbeddingProviderConfig,
    source_col: &str,
) -> Result<Connection, LanceError> {
    let db = get_db_with_embeddings(embedding_config).await?;

    if lancedb_exists().await
        && let Some(merge_on) = merge_on
        && let Some(first_batch) = data.first()
    {
        // Add rows if they don't already exist
        let tbl = db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| LanceError::TableUpdateError(e.to_string()))?;

        let schema = first_batch.schema();

        let reader = RecordBatchIterator::new(
            data.into_iter().map(std::result::Result::Ok),
            schema.clone(),
        );
        tbl.merge_insert(merge_on)
            .when_not_matched_insert_all()
            .clone()
            .execute(Box::new(reader))
            .await
            .map_err(|e| LanceError::TableUpdateError(e.to_string()))?;
    } else {
        let embedding_params = EmbeddingDefinition::new(
            source_col,
            embedding_config.provider_id().as_str(),
            Some("embeddings"),
        );

        // Create a new table and add rows
        db.create_table(TABLE_NAME, data)
            .mode(CreateTableMode::Overwrite)
            .add_embedding(embedding_params)?
            .execute()
            .await
            .map_err(|e| LanceError::TableUpdateError(e.to_string()))?;
    }

    Ok(db)
}

/// Insert records with backup support. Creates a backup before the operation and
/// restores it if the operation fails.
///
/// # Arguments
///
/// * `data` - An Arrow `RecordBatchIterator` containing data
/// * `merge_on` - `None` if you want to create or overwrite the current database; otherwise, a
///   reference to an array of keys to merge on.
/// * `embedding_config` - The embedding provider configuration
/// * `source_col` - The name of the column containing the source text to embed.
///
/// # Returns
/// A Connection to the LanceDB database if successful
///
/// # Errors
/// Returns a `LanceError` if backup creation, database operations, or restoration fails
pub async fn insert_records_with_backup(
    data: Vec<RecordBatch>,
    merge_on: Option<&[&str]>,
    embedding_config: &EmbeddingProviderConfig,
    source_col: &str,
) -> Result<Connection, LanceError> {
    with_backup(insert_records(data, merge_on, embedding_config, source_col)).await
}

/// Delete rows with backup support. Creates a backup before the operation and
/// restores it if the operation fails.
///
/// # Arguments
///
/// * `column` - The name of the column to match against.
/// * `keys` - The list of values to delete. Rows whose `column` value is in this list are deleted.
/// * `embedding_config` - The embedding provider configuration.
///
/// # Errors
///
/// Returns a `LanceError` if backup creation, database operations, or restoration fails.
pub async fn delete_rows_with_backup(
    column: &str,
    keys: &[impl AsRef<str>],
    embedding_config: &EmbeddingProviderConfig,
) -> Result<(), LanceError> {
    with_backup(delete_rows(column, keys, embedding_config)).await
}

/// Create or update indexes with backup support. Creates a backup before the operation and
/// restores it if the operation fails.
///
/// # Arguments
/// * `text_col` - The name of the column containing text
/// * `embedding_col` - The name of the embedding column
///
/// # Errors
/// Returns a `LanceError` if backup creation, database operations, or restoration fails
pub async fn create_or_update_indexes_with_backup(
    text_col: &str,
    embedding_col: &str,
) -> Result<(), LanceError> {
    with_backup(create_or_update_indexes(text_col, embedding_col)).await
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::StringArray;
    use arrow_array::cast::as_fixed_size_list_array;
    use arrow_array::cast::as_string_array;
    use arrow_array::types::Float32Type;
    use dotenv::dotenv;
    use futures::StreamExt;
    use lancedb::embeddings::EmbeddingFunction;
    use lancedb::query::ExecutableQuery;
    use serial_test::serial;
    use zqa_macros::{test_eq, test_ok};
    use zqa_pdftools::chunk::Chunker;

    use crate::capabilities::EmbeddingProvider;
    use crate::clients::openai::OpenAIClient;
    use crate::config::{OpenAIConfig, VoyageAIConfig};
    use crate::constants::{
        DEFAULT_OPENAI_EMBEDDING_DIM, DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_OPENAI_MODEL,
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use crate::http_client::ReqwestClient;

    use super::*;

    fn get_test_openai_embedding_config() -> EmbeddingProviderConfig {
        EmbeddingProviderConfig::OpenAI(OpenAIConfig {
            api_key: env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
            model: DEFAULT_OPENAI_MODEL.into(),
            max_tokens: 8192,
            embedding_model: DEFAULT_OPENAI_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_OPENAI_EMBEDDING_DIM as usize,
        })
    }

    #[tokio::test]
    #[serial]
    async fn test_create_initial_table_with_openai() {
        dotenv().ok();

        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "data_openai",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data)]).unwrap();
        let batches = vec![record_batch];

        let embedding_config = get_test_openai_embedding_config();
        let db = insert_records(batches, None, &embedding_config, "data_openai").await;

        test_ok!(db);
        let db = db.unwrap();

        let tbl_names = db.table_names().execute().await;
        test_ok!(tbl_names);
        test_eq!(tbl_names.unwrap(), vec![TABLE_NAME]);

        let tbl = db.open_table(TABLE_NAME).execute().await;
        test_ok!(tbl);

        let tbl = tbl.unwrap();
        let tbl_values = tbl.query().execute().await;

        test_ok!(tbl_values);

        let mut tbl_values = tbl_values.unwrap();
        let row = tbl_values.next().await;

        assert!(row.is_some());
        let row = row.unwrap();

        test_ok!(row);
        let row = row.unwrap();

        for column in ["data_openai", "embeddings"] {
            assert!(row.column_by_name(column).is_some());
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_create_initial_table_with_voyage() {
        dotenv().ok();

        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "data_voyage",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data)]).unwrap();
        let batches = vec![record_batch];

        let db = insert_records(
            batches,
            None,
            &EmbeddingProviderConfig::VoyageAI(VoyageAIConfig {
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                api_key: env::var("VOYAGE_AI_API_KEY").unwrap(),
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            }),
            "data_voyage",
        )
        .await;

        test_ok!(db);
        let db = db.unwrap();

        let tbl_names = db.table_names().execute().await;
        test_ok!(tbl_names);
        test_eq!(tbl_names.unwrap(), vec![TABLE_NAME]);

        let tbl = db.open_table(TABLE_NAME).execute().await;
        test_ok!(tbl);

        let tbl = tbl.unwrap();
        let tbl_values = tbl.query().execute().await;

        test_ok!(tbl_values);

        let mut tbl_values = tbl_values.unwrap();
        let row = tbl_values.next().await;

        assert!(row.is_some());
        let row = row.unwrap();

        test_ok!(row);
        let row = row.unwrap();

        for column in ["data_voyage", "embeddings"] {
            assert!(row.column_by_name(column).is_some());
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_search_by_key() {
        dotenv().ok();

        // First create a table with test data
        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("content", arrow_schema::DataType::Utf8, false),
        ]);
        let id_data = StringArray::from(vec!["key1", "key2", "key3"]);
        let content_data = StringArray::from(vec!["Content 1", "Content 2", "Content 3"]);
        let record_batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_data), Arc::new(content_data)],
        )
        .unwrap();
        let batches = vec![record_batch];

        let embedding_config = get_test_openai_embedding_config();
        let _db = insert_records(batches, None, &embedding_config, "content")
            .await
            .unwrap();

        // Test finding an existing key
        let result = search_by_key("id", "key2").await;
        test_ok!(result);
        let result = result.unwrap();
        assert!(result.is_some());

        let batch = result.unwrap();
        let id_column = batch.column_by_name("id").unwrap();
        let id_array = as_string_array(id_column);
        test_eq!(id_array.value(0), "key2");

        // Test finding a non-existent key
        let result = search_by_key("id", "nonexistent").await;
        test_ok!(result);
        assert!(result.unwrap().is_none());

        // Test with SQL injection attempt (should be escaped)
        let result = search_by_key("id", "'; DROP TABLE data; --").await;
        test_ok!(result);
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    #[serial]
    async fn test_search_by_column() {
        dotenv().ok();

        // First create a table with test data
        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("category", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("item", arrow_schema::DataType::Utf8, false),
        ]);
        let category_data = StringArray::from(vec!["fruit", "vegetable", "fruit", "grain"]);
        let item_data = StringArray::from(vec!["apple", "carrot", "banana", "rice"]);
        let record_batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(category_data), Arc::new(item_data)],
        )
        .unwrap();
        let batches = vec![record_batch];

        let embedding_config = get_test_openai_embedding_config();
        let _db = insert_records(batches, None, &embedding_config, "item")
            .await
            .unwrap();

        // Test finding multiple matching values
        let values = vec!["fruit", "grain"];
        let result = search_by_column("category", &values).await;
        test_ok!(result);
        let batches = result.unwrap();

        // Should find 3 records: 2 fruits and 1 grain
        let total_rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
        test_eq!(total_rows, 3);

        // Test finding single value
        let values = vec!["vegetable"];
        let result = search_by_column("category", &values).await;
        test_ok!(result);
        let batches = result.unwrap();
        let total_rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
        test_eq!(total_rows, 1);

        // Test with empty values array
        let values: Vec<&str> = vec![];
        let result = search_by_column("category", &values).await;
        test_ok!(result);
        let batches = result.unwrap();
        assert!(batches.is_empty());

        // Test with non-existent values
        let values = vec!["nonexistent"];
        let result = search_by_column("category", &values).await;
        test_ok!(result);
        let batches = result.unwrap();
        let total_rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
        test_eq!(total_rows, 0);

        // Test with SQL injection attempt (should be escaped)
        let values = vec!["'; DROP TABLE data; --"];
        let result = search_by_column("category", &values).await;
        test_ok!(result);
    }

    #[tokio::test]
    #[serial]
    async fn test_create_or_update_indexes() {
        dotenv().ok();

        // IVF-PQ index creation requires at least 256 rows
        let row_count = 256;
        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("text", arrow_schema::DataType::Utf8, false),
        ]);
        let id_data: StringArray = (0..row_count).map(|i| Some(i.to_string())).collect();
        let text_data: StringArray = (0..row_count)
            .map(|i| Some(format!("Sample text for row {i}")))
            .collect();
        let record_batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_data), Arc::new(text_data)],
        )
        .unwrap();

        let embedding_config = get_test_openai_embedding_config();
        let _db = insert_records(vec![record_batch], None, &embedding_config, "text")
            .await
            .unwrap();

        // First call — should create both indices
        let result = create_or_update_indexes("text", "embeddings").await;
        test_ok!(result);

        // Second call — should be idempotent when indices already exist
        let result = create_or_update_indexes("text", "embeddings").await;
        test_ok!(result);
    }

    #[tokio::test]
    #[serial]
    async fn test_dedup_rows_removes_duplicates() {
        dotenv().ok();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        ]);
        // "dup_title" appears twice — one duplicate should be removed
        let id_data = StringArray::from(vec!["id1", "id2", "id3", "id4"]);
        let title_data = StringArray::from(vec!["unique_a", "dup_title", "unique_b", "dup_title"]);
        let record_batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(id_data), Arc::new(title_data)],
        )
        .unwrap();

        let embedding_config = get_test_openai_embedding_config();
        let _db = insert_records(vec![record_batch], None, &embedding_config, "title")
            .await
            .unwrap();

        let deleted = dedup_rows(&embedding_config, schema, "title", "id").await;
        test_ok!(deleted);
        test_eq!(deleted.unwrap(), 1);

        // Verify 3 rows remain after dedup
        let remaining = get_lancedb_items(&embedding_config, vec!["id".to_string()])
            .await
            .unwrap();
        let total_rows: usize = remaining.iter().map(RecordBatch::num_rows).sum();
        test_eq!(total_rows, 3);
    }

    #[tokio::test]
    #[serial]
    async fn test_dedup_rows_no_duplicates() {
        dotenv().ok();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        ]);
        let id_data = StringArray::from(vec!["id1", "id2", "id3"]);
        let title_data = StringArray::from(vec!["unique_x", "unique_y", "unique_z"]);
        let record_batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(id_data), Arc::new(title_data)],
        )
        .unwrap();

        let embedding_config = get_test_openai_embedding_config();
        let _db = insert_records(vec![record_batch], None, &embedding_config, "title")
            .await
            .unwrap();

        let deleted = dedup_rows(&embedding_config, schema, "title", "id").await;
        test_ok!(deleted);
        test_eq!(deleted.unwrap(), 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_dedup_rows_missing_columns_returns_zero() {
        dotenv().ok();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        ]);
        let id_data = StringArray::from(vec!["id1", "id2"]);
        let title_data = StringArray::from(vec!["a", "b"]);
        let record_batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_data), Arc::new(title_data)],
        )
        .unwrap();

        let embedding_config = get_test_openai_embedding_config();
        let _db = insert_records(vec![record_batch], None, &embedding_config, "title")
            .await
            .unwrap();

        // Schema with non-existent columns — should return Ok(0) rather than an error
        let missing_schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("nonexistent_by", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("nonexistent_key", arrow_schema::DataType::Utf8, false),
        ]);
        let deleted = dedup_rows(
            &embedding_config,
            missing_schema,
            "nonexistent_by",
            "nonexistent_key",
        )
        .await;
        test_ok!(deleted);
        test_eq!(deleted.unwrap(), 0);
    }

    fn pdf_asset_path() -> String {
        format!("{}/assets/deeply.pdf", env!("CARGO_MANIFEST_DIR"))
    }

    /// Verify that a real PDF is chunked into non-empty, well-formed chunks using the
    /// OpenAI-recommended strategy.
    #[tokio::test]
    #[serial]
    async fn test_pdf_chunking_with_openai_strategy() {
        let extracted =
            zqa_pdftools::parse::extract_text(&pdf_asset_path()).expect("Failed to parse PDF");

        let strategy = EmbeddingProvider::OpenAI.recommended_chunking_strategy();
        let chunks = Chunker::new(extracted, strategy).chunk();

        assert!(!chunks.is_empty(), "Expected at least one chunk from PDF");

        for chunk in &chunks {
            assert!(
                !chunk.content.trim().is_empty(),
                "Chunk {} content must not be empty",
                chunk.chunk_id
            );
        }

        // Chunk IDs must be sequential starting from 1
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, i + 1, "Chunk ID should equal position + 1");
        }

        // chunk_count must agree with the actual number of chunks produced
        let expected_count = chunks.len();
        for chunk in &chunks {
            assert_eq!(
                chunk.chunk_count, expected_count,
                "chunk.chunk_count disagrees with actual chunk count"
            );
        }

        let total_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
        assert!(
            total_chars > 1000,
            "Expected at least 1000 characters of text, got {total_chars}"
        );
    }

    /// Verify that chunks from a real PDF can be embedded with OpenAI and that the resulting
    /// vectors are non-zero.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_pdf_chunk_embeddings_are_non_zero() {
        dotenv().ok();

        let extracted =
            zqa_pdftools::parse::extract_text(&pdf_asset_path()).expect("Failed to parse PDF");

        let strategy = EmbeddingProvider::OpenAI.recommended_chunking_strategy();
        let chunks = Chunker::new(extracted, strategy).chunk();

        // Embed only the first 3 chunks to keep API usage low
        let sample_texts: Vec<&str> = chunks.iter().take(3).map(|c| c.content.as_str()).collect();
        assert!(!sample_texts.is_empty(), "Need at least one chunk to embed");

        let client = OpenAIClient::<ReqwestClient>::default();
        let source: Arc<dyn arrow_array::Array> = Arc::new(StringArray::from(sample_texts.clone()));
        let embeddings = client.compute_source_embeddings(source);

        assert!(
            embeddings.is_ok(),
            "compute_source_embeddings failed: {:?}",
            embeddings.err()
        );
        let embeddings = embeddings.unwrap();
        let list_array = as_fixed_size_list_array(&embeddings);

        assert_eq!(
            list_array.len(),
            sample_texts.len(),
            "Expected one embedding per chunk"
        );
        assert_eq!(
            list_array.value_length(),
            DEFAULT_OPENAI_EMBEDDING_DIM as i32,
            "Embedding dimension mismatch — expected text-embedding-3-small size"
        );

        for i in 0..list_array.len() {
            let all_zero = list_array
                .value(i)
                .as_primitive::<Float32Type>()
                .iter()
                .all(|v| v.is_none_or(|f| f == 0.0));
            assert!(
                !all_zero,
                "Embedding for chunk {i} is all zeros — likely a failed API call"
            );
        }
    }

    /// End-to-end test: chunk a PDF, embed and store the chunks, add an unrelated sentinel
    /// document, then verify that a query matching the sentinel retrieves it as the top result
    /// rather than a PDF chunk.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_retrieval_ranks_relevant_content_higher() {
        dotenv().ok();

        let extracted =
            zqa_pdftools::parse::extract_text(&pdf_asset_path()).expect("Failed to parse PDF");

        let strategy = EmbeddingProvider::OpenAI.recommended_chunking_strategy();
        let chunks = Chunker::new(extracted, strategy).chunk();

        // Build corpus: first 5 PDF chunks + one clearly off-topic sentinel
        let sentinel_id = "cooking_sentinel";
        let sentinel_text = "How to bake chocolate chip cookies: cream together butter and \
            sugar, beat in eggs and vanilla extract, stir in flour, baking soda, and salt, \
            then fold in chocolate chips. Drop rounded tablespoons onto an ungreased baking \
            sheet and bake at 375°F for 9 to 11 minutes until golden brown.";

        let pdf_chunk_ids: Vec<String> = (0..chunks.len().min(5))
            .map(|i| format!("pdf_chunk_{i}"))
            .collect();

        let mut ids: Vec<&str> = pdf_chunk_ids.iter().map(String::as_str).collect();
        ids.push(sentinel_id);

        let mut texts: Vec<&str> = chunks.iter().take(5).map(|c| c.content.as_str()).collect();
        texts.push(sentinel_text);

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("text", arrow_schema::DataType::Utf8, false),
        ]);
        let record_batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(texts)),
            ],
        )
        .unwrap();

        let embedding_config = get_test_openai_embedding_config();
        insert_records(vec![record_batch], None, &embedding_config, "text")
            .await
            .unwrap();

        let results = vector_search(
            "chocolate chip cookie baking recipe ingredients".to_string(),
            &embedding_config,
            10,
        )
        .await;
        test_ok!(results);

        let results = results.unwrap();
        assert!(!results.is_empty(), "Expected at least one search result");

        let returned_ids: Vec<String> = results
            .iter()
            .flat_map(|batch| {
                as_string_array(batch.column_by_name("id").unwrap())
                    .iter()
                    .filter_map(|v| v.map(str::to_owned))
            })
            .collect();

        assert!(
            !returned_ids.is_empty(),
            "No id values found in search results"
        );
        assert_eq!(
            returned_ids[0], sentinel_id,
            "Expected the cooking sentinel to be ranked #1 for a cookie query, \
             but top result was '{}'. Full ranking: {:?}",
            returned_ids[0], returned_ids
        );
    }
}
