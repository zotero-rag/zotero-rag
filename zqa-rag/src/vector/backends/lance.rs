//! LanceDB vector backend implementation.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{ArrowError, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::database::CreateTableMode;
use lancedb::embeddings::EmbeddingDefinition;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Error as LanceDbError, connect, index::scalar::FtsIndexBuilder};
use thiserror::Error;

use crate::{
    embedding::common::EmbeddingProviderConfig, providers::registry::provider_registry,
    vector::backends::backend::VectorBackend,
};

// NOTE: Maintainers: ensure that `LANCEDB_URI` begins with `LANCE_TABLE_NAME`

/// The URI for the LanceDB table. This is the default location for the table, and for now cannot
/// be changed.
pub const LANCEDB_URI: &str = "data/lancedb-table";

/// The name of the table. This is the default table name, and for now cannot be changed.
pub const LANCE_TABLE_NAME: &str = "data";

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
    /// Arrow errors, used by repair.rs.
    #[error(transparent)]
    ArrowError(#[from] ArrowError),
    /// Other LanceDB-related errors
    #[error(transparent)]
    Other(#[from] LanceDbError),
}

/// Backend for LanceDB vector store.
pub struct LanceBackend {
    /// Configuration for the LanceDB embedding provider.
    config: EmbeddingProviderConfig,
    /// Arrow schema for the LanceDB table.
    schema: Arc<Schema>,
    /// Column name containing the source text used to generate embeddings.
    source_col: String,
}

impl LanceBackend {
    /// Creates a new `LanceBackend` with the given embedding provider config, Arrow schema, and
    /// source column name.
    #[must_use]
    pub fn new(config: EmbeddingProviderConfig, schema: Arc<Schema>, source_col: String) -> Self {
        Self {
            config,
            schema,
            source_col,
        }
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
}

#[async_trait]
impl VectorBackend for LanceBackend {
    type Record = RecordBatch;
    type Error = LanceError;
    type Config = EmbeddingProviderConfig;
    type Connection = Connection;

    /// Returns the database URI, allowing override via `LANCEDB_URI` environment variable.
    fn get_db_path(&self) -> String {
        std::env::var("LANCEDB_URI").unwrap_or_else(|_| LANCEDB_URI.to_string())
    }

    /// Checks if an existing LanceDB exists and has a valid table
    async fn db_exists(&self) -> bool {
        let uri = self.get_db_path();
        if !PathBuf::from(&uri).exists() {
            return false;
        }

        // Check if we can actually connect and open the table
        let db_result = connect(&uri).execute().await;
        if let Ok(db) = db_result {
            db.open_table(LANCE_TABLE_NAME).execute().await.is_ok()
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
    async fn create_or_update_indices(
        &self,
        text_col: &str,
        embedding_col: &str,
    ) -> Result<(), Self::Error> {
        let db = self.connect().await?;
        let tbl = db.open_table(LANCE_TABLE_NAME).execute().await?;

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

    /// Connect to the Lance database and add the embedding `embedding_name` to the database's
    /// embedding registry. Note that if we are opening a database that had one defined, we should use
    /// the same one here. Since that isn't stored (at least, not that I know of), this onus is on the
    /// user.
    async fn connect(&self) -> Result<Self::Connection, Self::Error> {
        let db = connect(&self.get_db_path())
            .execute()
            .await
            .map_err(|e| LanceError::ConnectionError(e.to_string()))?;
        let registry = provider_registry();

        registry
            .register_embedding_with_lancedb(&db, &self.config)
            .map_err(|e| LanceError::ConnectionError(e.to_string()))?;
        Ok(db)
    }

    /// Given a column name and a list of key values, delete all matching rows from the database.
    ///
    /// # Arguments
    ///
    /// * `col` - The name of the column to match against.
    /// * `keys` - The list of values to delete. Rows whose `col` value is in this list are deleted.
    ///
    /// # Errors
    ///
    /// * `LanceError::ConnectionError` - If the database connection fails
    /// * `LanceError::InvalidStateError` - If the table doesn't exist
    /// * `LanceError::Other` - If the delete query fails
    async fn delete_rows(&self, col: &str, keys: &[String]) -> Result<(), Self::Error> {
        if keys.is_empty() {
            return Ok(());
        }

        self.schema.field_with_name(col).map_err(|_| {
            LanceError::ParameterError(format!("Column {col} does not exist in the schema"))
        })?;

        let db = self.connect().await?;
        let table = db
            .open_table(LANCE_TABLE_NAME)
            .execute()
            .await
            .map_err(|_| {
                LanceError::InvalidStateError(format!(
                    "The table {LANCE_TABLE_NAME} does not exist"
                ))
            })?;

        for key_chunk in keys.chunks(100) {
            let delete_pred = key_chunk
                .iter()
                .map(|k| format!("{col} = '{}'", k.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(" OR ");

            table.delete(&delete_pred).await?;
        }

        Ok(())
    }

    /// Deduplicate rows in the LanceDB table based on a 'by' column, keeping only the first
    /// occurrence of each unique 'by' value and deleting the duplicates.
    ///
    /// # Arguments
    ///
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
    async fn dedup_rows(&self, by: &str, key: &str) -> Result<usize, Self::Error> {
        let db = self.connect().await?;
        let table = db
            .open_table(LANCE_TABLE_NAME)
            .execute()
            .await
            .map_err(|_| {
                LanceError::InvalidStateError(format!(
                    "The table {LANCE_TABLE_NAME} does not exist"
                ))
            })?;

        if let Some((by_idx, _)) = self.schema.column_with_name(by)
            && let Some((key_idx, _)) = self.schema.column_with_name(key)
        {
            let mut stream = table.query().execute().await?;
            let mut seen_by_values = HashSet::new();
            let mut duplicate_keys = Vec::new();

            while let Some(batch) = stream.try_next().await? {
                let by_values = LanceBackend::get_column_from_batch(&batch, by_idx);
                let key_values = LanceBackend::get_column_from_batch(&batch, key_idx);

                for (by_val, key_val) in by_values.into_iter().zip(key_values) {
                    if !seen_by_values.insert(by_val) {
                        duplicate_keys.push(key_val);
                    }
                }
            }

            // Delete duplicate rows
            let deleted_count = duplicate_keys.len();
            if !duplicate_keys.is_empty() {
                self.delete_rows(key, &duplicate_keys).await?;
            }

            Ok(deleted_count)
        } else {
            Err(LanceError::ParameterError(format!(
                "column '{by}' or '{key}' not found in schema"
            )))
        }
    }

    /// Return all the rows in the LanceDB table, selecting only the columns specified. This is useful
    /// for computing what rows do *not* exist in the table, such as in cases where the data source has
    /// new items. Note that LanceDB by default seems to chunk by 1024 items, so if you're certain you
    /// will receive fewer than 1024 rows, it is safe to assume only one element exists.
    ///
    /// # Arguments
    ///
    /// * `columns`: The set of columns to return. Typically, you do not want to include the full-text
    ///   column here.
    ///
    /// # Returns
    ///
    /// A vector of Arrow `RecordBatch` objects containing the results.
    ///
    /// # Errors
    ///
    /// * `LanceError::ConnectionError` - If the database connection fails
    /// * `LanceError::InvalidStateError` - If the table doesn't exist
    /// * `LanceError::Other` - If the query execution fails
    async fn get_items(&self, columns: &[String]) -> Result<Vec<Self::Record>, Self::Error> {
        let db = self.connect().await?;

        let tbl = db
            .open_table(LANCE_TABLE_NAME)
            .execute()
            .await
            .map_err(|_| {
                LanceError::InvalidStateError(format!(
                    "The table {LANCE_TABLE_NAME} does not exist"
                ))
            })?;

        let string_cols = columns.to_vec();

        // The installed version of LanceDB has a bug where without the `.limit` call here, it only
        // returns 10 rows; see https://github.com/lancedb/lancedb/issues/1852#issuecomment-2489837804
        let results: Vec<RecordBatch> = tbl
            .query()
            .select(lancedb::query::Select::Columns(string_cols))
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
    async fn vector_search(
        &self,
        query: String,
        limit: usize,
    ) -> Result<Vec<Self::Record>, Self::Error> {
        let start_time = Instant::now();
        let db = self.connect().await?;

        let tbl = db
            .open_table(LANCE_TABLE_NAME)
            .execute()
            .await
            .map_err(|_| {
                LanceError::InvalidStateError(format!(
                    "The table {LANCE_TABLE_NAME} does not exist"
                ))
            })?;
        log::debug!("Opening the DB and table took {:.1?}", start_time.elapsed());

        let start_time = Instant::now();
        let embedding = db
            .embedding_registry()
            .get(self.config.provider_id().as_str())
            .ok_or(LanceError::InvalidStateError(format!(
                "{} is not in the database embedding registry",
                self.config.provider_id().as_str()
            )))?;

        let query_vec =
            embedding.compute_query_embeddings(Arc::new(StringArray::from(vec![query])))?;
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
    async fn search_by_key(
        &self,
        key_col: &str,
        key: &str,
    ) -> Result<Option<Self::Record>, Self::Error> {
        self.schema.field_with_name(key_col).map_err(|_| {
            LanceError::ParameterError(format!("Column {key_col} does not exist in the schema"))
        })?;

        let db = self.connect().await?;
        let tbl = db
            .open_table(LANCE_TABLE_NAME)
            .execute()
            .await
            .map_err(|_| {
                LanceError::InvalidStateError(format!(
                    "The table {LANCE_TABLE_NAME} does not exist"
                ))
            })?;

        let stream = tbl
            .query()
            .only_if(format!("{key_col} = '{}'", key.replace('\'', "''")))
            .limit(1)
            .execute()
            .await
            .map_err(|e| LanceError::QueryError(e.to_string()))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| LanceError::QueryError(e.to_string()))?;

        Ok(batches.into_iter().next())
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
    async fn search_by_column(
        &self,
        col: &str,
        values: &[String],
    ) -> Result<Vec<Self::Record>, Self::Error> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        self.schema.field_with_name(col).map_err(|_| {
            LanceError::ParameterError(format!("Column {col} does not exist in the schema"))
        })?;

        let db = self.connect().await?;
        let tbl = db
            .open_table(LANCE_TABLE_NAME)
            .execute()
            .await
            .map_err(|_| {
                LanceError::InvalidStateError(format!(
                    "The table {LANCE_TABLE_NAME} does not exist"
                ))
            })?;

        let queries = values
            .iter()
            .map(|key| format!("{col} = '{}'", key.replace('\'', "''")))
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
    /// * `items` - A vector of `Record` items to insert into the database.
    /// * `merge_on` - `None` if you want to create or overwrite the current database; otherwise, a
    ///   reference to an array of keys to merge on.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the items were inserted successfully
    ///
    /// # Errors
    ///
    /// Returns a `LanceError` if connection, table creation, or registering embedding functions fails
    async fn insert_items(
        &self,
        items: Vec<Self::Record>,
        merge_on: Option<&[&str]>,
    ) -> Result<(), Self::Error> {
        if items.is_empty() {
            return Ok(());
        }

        let db = self.connect().await?;

        if self.db_exists().await
            && let Some(merge_on) = merge_on
            && let Some(first_batch) = items.first()
        {
            // Add rows if they don't already exist
            let tbl = db
                .open_table(LANCE_TABLE_NAME)
                .execute()
                .await
                .map_err(|e| LanceError::TableUpdateError(e.to_string()))?;

            let schema = first_batch.schema();

            let reader = RecordBatchIterator::new(
                items.into_iter().map(std::result::Result::Ok),
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
                self.source_col.as_str(),
                self.config.provider_id().as_str(),
                Some("embeddings"),
            );

            // Create a new table and add rows
            db.create_table(LANCE_TABLE_NAME, items)
                .mode(CreateTableMode::Overwrite)
                .add_embedding(embedding_params)?
                .execute()
                .await
                .map_err(|e| LanceError::TableUpdateError(e.to_string()))?;
        }

        Ok(())
    }
}
