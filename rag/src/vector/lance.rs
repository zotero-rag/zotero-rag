use crate::capabilities::EmbeddingProviders;
use crate::embedding::common::get_embedding_provider;
use crate::embedding::voyage::VoyageAIClient;
use crate::llm::{anthropic::AnthropicClient, http_client::ReqwestClient, openai::OpenAIClient};
use crate::vector::checkhealth::get_zero_vectors;

use arrow_array::RecordBatchReader;
use arrow_array::{
    RecordBatch, RecordBatchIterator, StringArray, cast::AsArray, types::Float32Type,
};
use arrow_schema::Schema;
use futures::TryStreamExt;
use lancedb::{
    Connection, Error as LanceDbError, arrow::arrow_schema::ArrowError, connect,
    database::CreateTableMode, embeddings::EmbeddingDefinition, query::ExecutableQuery,
    query::QueryBase,
};
use std::{fmt::Display, path::PathBuf, sync::Arc, time::Instant, vec::IntoIter};
use thiserror::Error;

// Maintainers: ensure that `DB_URI` begins with `TABLE_NAME`
pub const DB_URI: &str = "data/lancedb-table";
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
    Other(String),
}

impl From<ArrowError> for LanceError {
    fn from(value: ArrowError) -> Self {
        Self::Other(value.to_string())
    }
}

impl From<LanceDbError> for LanceError {
    fn from(value: LanceDbError) -> Self {
        Self::Other(value.to_string())
    }
}

/// Table statistics, to be shown when the user calls the `/stats` command.
pub struct TableStatistics {
    /// The table version. Each update to a table creates a new version.
    table_version: u64,
    /// Number of rows in the table
    num_rows: usize,
}

impl Display for TableStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Table statistics:\n")?;
        f.write_str(&format!("\tTable version: {}", self.table_version))?;
        f.write_str(&format!("\tNumber of rows: {}", self.num_rows))?;

        Ok(())
    }
}

/// Checks if an existing LanceDB exists and has a valid table
pub async fn lancedb_exists() -> bool {
    if !PathBuf::from(DB_URI).exists() {
        return false;
    }

    // Check if we can actually connect and open the table
    let db_result = connect(DB_URI).execute().await;
    if let Ok(db) = db_result {
        db.open_table(TABLE_NAME).execute().await.is_ok()
    } else {
        false
    }
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
    let db = connect(DB_URI)
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
async fn get_db_with_embeddings(embedding_name: &str) -> Result<Connection, LanceError> {
    if !(EmbeddingProviders::contains(embedding_name)) {
        return Err(LanceError::ParameterError(format!(
            "{embedding_name} is not a valid embedding."
        )));
    }

    let db = connect(DB_URI)
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    match embedding_name {
        "anthropic" => {
            db.embedding_registry().register(
                EmbeddingProviders::Anthropic.as_str(),
                Arc::new(AnthropicClient::<ReqwestClient>::default()),
            )?;
        }
        "openai" => {
            db.embedding_registry().register(
                EmbeddingProviders::OpenAI.as_str(),
                Arc::new(OpenAIClient::<ReqwestClient>::default()),
            )?;
        }
        "voyageai" => {
            db.embedding_registry().register(
                EmbeddingProviders::VoyageAI.as_str(),
                Arc::new(VoyageAIClient::<ReqwestClient>::default()),
            )?;
        }
        _ => unreachable!("Unknown embedding provider {}", embedding_name),
    }

    Ok(db)
}

/// A transparent wrapper around LanceDB functionality to update rows in the database. This can be
/// used for fixing embeddings, for example. Note that the `new_data` passed should have the same
/// schema as the table.
///
/// # Arguments:
///
/// * `on` - The set of columns to match on
/// * `embedding_name` - The name of the embedding provider
/// * `new_data` - The new data for all columns
pub async fn update_records(
    on: &[&str],
    embedding_name: &str,
    new_data: Box<dyn RecordBatchReader + Send>,
) -> Result<(), LanceError> {
    let db = get_db_with_embeddings(embedding_name).await?;
    let tbl = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    // TODO: Might just be easier to delete and re-insert
    let mut merge_insert = tbl.merge_insert(on);

    merge_insert
        .when_matched_update_all(None)
        .when_not_matched_insert_all();

    let result = merge_insert.execute(new_data).await?;

    log::info!("Updated {} rows.", result.num_updated_rows);

    Ok(())
}

/// Fix zero embeddings in the database by regenerating embeddings for affected records.
/// This function identifies records with zero embeddings and updates them with proper embeddings.
///
/// # Arguments
///
/// * `embedding_col` - The name of the embedding column
/// * `embedding_name` - The embedding provider to use for generating new embeddings
/// * `merge_key` - The column name to use for matching records (should be a unique identifier)
///
/// # Returns
///
/// Returns `Ok(usize)` with the number of records fixed, or a `LanceError` if the operation failed.
pub async fn fix_zero_embeddings(
    schema: Arc<Schema>,
    embedding_col: &str,
    embedding_name: &str,
    merge_key: &str,
) -> Result<usize, LanceError> {
    let db = get_db_with_embeddings(embedding_name).await?;
    let table = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    let db_columns = schema
        .fields
        .iter()
        .map(|f| f.name().clone())
        .collect::<Vec<String>>();

    // Get all records with zero embeddings
    let zero_batches = get_zero_vectors(&table, embedding_name, 10000).await?;

    if zero_batches.is_empty() {
        return Ok(0);
    }

    let mut total_fixed = 0;

    // Process each batch of zero-embedding records
    for batch in zero_batches {
        let num_rows = batch.num_rows();

        // Get the text column to generate new embeddings
        let text_col_idx = batch.schema().index_of(merge_key).map_err(|_| {
            LanceError::InvalidStateError(format!("Column '{}' not found", merge_key))
        })?;
        let text_array = batch.column(text_col_idx).as_string::<i32>();
        let delete_pred = text_array
            .iter()
            .filter_map(|s| s.map(|id| format!("{} = '{}'", merge_key, id)))
            .collect::<Vec<String>>()
            .join(" OR ");
        dbg!(&delete_pred);

        // Delete the old records
        table
            .delete(&delete_pred)
            .await
            .map_err(|e| LanceError::TableUpdateError(e.to_string()))?;

        // Generate new embeddings
        let embedding_provider = get_embedding_provider(embedding_name)
            .map_err(|e| LanceError::InvalidStateError(e.to_string()))?;
        let embeddings = embedding_provider
            .compute_source_embeddings(Arc::new(text_array.clone()))
            .map_err(|e| LanceError::InvalidStateError(e.to_string()))?;

        // Create new batch with updated embeddings
        let mut columns = Vec::new();
        for (i, field) in batch.schema().fields().iter().enumerate() {
            if field.name() == embedding_col {
                columns.push(Arc::new(embeddings.clone()) as arrow_array::ArrayRef);
            } else if db_columns.contains(field.name()) {
                columns.push(batch.column(i).clone());
            }
        }

        let updated_batch = RecordBatch::try_new(Arc::clone(&schema), columns)
            .map_err(|e| LanceError::InvalidStateError(e.to_string()))?;

        // Create reader for this single batch
        let batches = vec![Ok(updated_batch)];
        let batch_reader = RecordBatchIterator::new(batches.into_iter(), schema.clone());

        let mut merge_insert = table.merge_insert(&[merge_key]);

        merge_insert.when_not_matched_insert_all();

        merge_insert.execute(Box::new(batch_reader)).await?;

        total_fixed += num_rows;
    }

    Ok(total_fixed)
}

/// Return all the rows in the LanceDB table, selecting only the columns specified. This is useful
/// for computing what rows do *not* exist in the table, such as in cases where the data source has
/// new items. Note that LanceDB by default seems to chunk by 1024 items, so if you're certain you
/// will receive fewer than 1024 rows, it is safe to assume only one element exists.
///
/// # Arguments
/// * embedding_name: The embedding method to use. Must be one of `EmbeddingProviders`.
/// * columns: The set of columns to return. Typically, you do not want to include the full-text
///   column here.
///
/// # Returns
/// * batches: A vector of Arrow `RecordBatch` objects containing the results.
pub async fn get_lancedb_items(
    embedding_name: &str,
    columns: Vec<String>,
) -> Result<Vec<RecordBatch>, LanceError> {
    let db = get_db_with_embeddings(embedding_name).await?;

    let tbl = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;

    // The installed version of Lance has a bug where without the `.limit` call here, it only
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
/// * query: The query string to search for
/// * embedding_name: The embedding method to use. Must be one of `EmbeddingProviders`.
///
/// # Returns
/// * batches: A vector of Arrow `RecordBatch` objects containing the results.
pub async fn vector_search(
    query: String,
    embedding_name: &str,
) -> Result<Vec<RecordBatch>, LanceError> {
    let start_time = Instant::now();
    let db = get_db_with_embeddings(embedding_name).await?;

    let tbl = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;
    log::debug!("Opening the DB and table took {:.1?}", start_time.elapsed());

    let start_time = Instant::now();
    let embedding =
        db.embedding_registry()
            .get(embedding_name)
            .ok_or(LanceError::InvalidStateError(format!(
                "{embedding_name} is not in the database embedding registry"
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
        .limit(10)
        .nearest_to(query_vec)?
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    log::debug!("Vector search took {:.1?}", start_time.elapsed());

    Ok(batches)
}

/// Creates and initializes a LanceDB table for vector storage.
///
/// Connects to LanceDB at the default location, creates a table named `TABLE_NAME`,
/// and registers embedding functions for both Anthropic and OpenAI. If this table already exists,
/// it simply opens it and upserts the data; otherwise, it inserts the data into the newly created
/// table.
///
/// # Arguments
/// * `data`: An Arrow `RecordBatchIterator` containing data. See
///   https://docs.rs/lancedb/latest/lancedb/index.html for an example of creating this.
/// * `merge_on`: `None` if you want to create or overwrite the current database; otherwise, a
///   reference to an array of keys to merge on.
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
    merge_on: Option<&[&str]>,
    embedding_params: EmbeddingDefinition,
) -> Result<Connection, LanceError> {
    let db = get_db_with_embeddings(&embedding_params.embedding_name).await?;

    if lancedb_exists().await && merge_on.is_some() {
        // Add rows if they don't already exist
        let tbl = db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| LanceError::TableUpdateError(e.to_string()))?;

        tbl.merge_insert(merge_on.unwrap())
            .when_not_matched_insert_all()
            .clone()
            .execute(Box::new(data))
            .await
            .map_err(|e| LanceError::TableUpdateError(e.to_string()))?;
    } else {
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

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use dotenv::dotenv;
    use futures::StreamExt;
    use lancedb::query::ExecutableQuery;
    use serial_test::serial;

    use super::*;

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
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        let db = create_initial_table(
            reader,
            None,
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
        assert_eq!(tbl_names.unwrap(), vec![TABLE_NAME]);

        let tbl = db.open_table(TABLE_NAME).execute().await;
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
    #[serial]
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
            None,
            EmbeddingDefinition::new("data_anthropic", "anthropic", Some("embeddings")),
        )
        .await;

        assert!(db.is_ok());
        let db = db.unwrap();

        let tbl_names = db.table_names().execute().await;
        assert!(tbl_names.is_ok());
        assert_eq!(tbl_names.unwrap(), vec![TABLE_NAME]);

        let tbl = db.open_table(TABLE_NAME).execute().await;
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
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        let db = create_initial_table(
            reader,
            None,
            EmbeddingDefinition::new("data_voyage", "voyageai", Some("embeddings")),
        )
        .await;

        assert!(db.is_ok());
        let db = db.unwrap();

        let tbl_names = db.table_names().execute().await;
        assert!(tbl_names.is_ok());
        assert_eq!(tbl_names.unwrap(), vec![TABLE_NAME]);

        let tbl = db.open_table(TABLE_NAME).execute().await;
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

        for column in ["data_voyage", "embeddings"] {
            assert!(row.column_by_name(column).is_some());
        }
    }

    #[tokio::test]
    #[serial]
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
            None,
            EmbeddingDefinition::new("data", "invalid", Some("embeddings")),
        )
        .await;

        assert!(db.is_err());
    }
}
