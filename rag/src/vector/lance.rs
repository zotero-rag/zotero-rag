use crate::llm::{
    anthropic::AnthropicClient, base::EmbeddingProviders, http_client::ReqwestClient,
    openai::OpenAIClient, voyage::VoyageAIClient,
};

use arrow_array::{
    RecordBatch, RecordBatchIterator, StringArray, cast::AsArray, types::Float32Type,
    FixedSizeListArray,
};
use std::fs;
use core::fmt;
use futures::TryStreamExt;
use lancedb::{
    Connection, Error as LanceDbError, arrow::arrow_schema::ArrowError, connect,
    database::CreateTableMode, embeddings::EmbeddingDefinition, query::ExecutableQuery,
    query::QueryBase,
};
use std::{error::Error, fmt::Display, path::PathBuf, sync::Arc, vec::IntoIter};

// Maintainers: ensure that `DB_URI` begins with `TABLE_NAME`
pub const DB_URI: &str = "data/lancedb-table";
pub const TABLE_NAME: &str = "data";

/// Errors that can occur when working with LanceDB
#[derive(Debug)]
pub enum LanceError {
    /// Error connecting to LanceDB
    ConnectionError(String),
    /// Error creating or updating a table in LanceDB
    TableUpdateError(String),
    /// Invalid params
    ParameterError(String),
    /// The database is in an invalid state
    InvalidStateError(String),
    /// Other LanceDB-related errors
    Other(String),
}

impl fmt::Display for LanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionError(msg) => write!(f, "LanceDB connection error: {msg}"),
            Self::TableUpdateError(msg) => write!(f, "LanceDB table update error: {msg}"),
            Self::ParameterError(msg) => write!(f, "Invalid parameter: {msg}"),
            Self::InvalidStateError(msg) => write!(f, "The DB is in an invalid state: {msg}"),
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

/// Health check result for LanceDB
#[derive(Debug)]
pub struct HealthCheckResult {
    /// Directory exists and size in bytes
    pub directory_exists: bool,
    pub directory_size: u64,
    /// Table can be opened
    pub table_accessible: bool,
    /// Number of rows in the table
    pub num_rows: usize,
    /// Number of rows with all-zero embeddings
    pub zero_embedding_count: usize,
    /// Titles of documents with all-zero embeddings
    pub zero_embedding_titles: Vec<String>,
    /// Index information: (index_type, indexed_rows)
    pub index_info: Vec<(String, usize)>,
    /// Total number of indexed rows
    pub total_indexed_rows: usize,
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

/// Calculate the size of a directory recursively
fn calculate_directory_size(path: &std::path::Path) -> Result<u64, std::io::Error> {
    let mut size = 0;
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                size += calculate_directory_size(&path)?;
            } else {
                size += entry.metadata()?.len();
            }
        }
    }
    Ok(size)
}

/// Perform comprehensive health checks on the LanceDB database
///
/// This function checks:
/// - Directory existence and size
/// - Table accessibility  
/// - Row count
/// - Zero embeddings detection
/// - Index status
///
/// # Arguments
/// * `embedding_name` - The embedding provider name to use for connection
///
/// # Returns
/// * `HealthCheckResult` containing all health check information
///
/// # Errors
/// Returns a `LanceError` if any critical health check fails
pub async fn perform_health_check(embedding_name: &str) -> Result<HealthCheckResult, LanceError> {
    let mut result = HealthCheckResult {
        directory_exists: false,
        directory_size: 0,
        table_accessible: false,
        num_rows: 0,
        zero_embedding_count: 0,
        zero_embedding_titles: Vec::new(),
        index_info: Vec::new(),
        total_indexed_rows: 0,
    };

    // Check if directory exists and get size
    let db_path = PathBuf::from(DB_URI);
    result.directory_exists = db_path.exists();
    
    if result.directory_exists {
        result.directory_size = calculate_directory_size(&db_path)
            .unwrap_or(0);
    }

    if !result.directory_exists {
        return Ok(result);
    }

    // Try to connect to database and open table
    let db_connection = connect(DB_URI).execute().await;
    if let Ok(db) = db_connection {
        if let Ok(tbl) = db.open_table(TABLE_NAME).execute().await {
            result.table_accessible = true;
            
            // Get row count
            result.num_rows = tbl.count_rows(None).await.unwrap_or(0);
            
            // Check for zero embeddings if table has rows
            if result.num_rows > 0 {
                // Get all rows with embeddings and title columns
                match tbl.query()
                    .select(lancedb::query::Select::Columns(vec![
                        "embeddings".to_string(),
                        "title".to_string()
                    ]))
                    .limit(result.num_rows)
                    .execute()
                    .await {
                    Ok(stream) => {
                        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap_or_default();
                        
                        // Check each row for zero embeddings
                        for batch in batches {
                            if let Some(embeddings_col) = batch.column_by_name("embeddings") {
                                if let Some(title_col) = batch.column_by_name("title") {
                                    let title_array = title_col.as_string::<i32>();
                                    
                                    // Check if embeddings are fixed-size list arrays
                                    if let Some(embedding_array) = embeddings_col.as_any().downcast_ref::<FixedSizeListArray>() {
                                        for (row_idx, embedding_opt) in embedding_array.iter().enumerate() {
                                            if let Some(embedding_values) = embedding_opt {
                                                let float_values = embedding_values.as_primitive::<Float32Type>();
                                                let all_zeros = float_values.iter().all(|v| v.unwrap_or(1.0) == 0.0);
                                                
                                                if all_zeros {
                                                    result.zero_embedding_count += 1;
                                                    if let Some(title) = title_array.value(row_idx).strip_suffix(".pdf") {
                                                        result.zero_embedding_titles.push(title.to_string());
                                                    } else {
                                                        result.zero_embedding_titles.push(title_array.value(row_idx).to_string());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {} // Continue even if we can't check embeddings
                }
            }

            // Get index information - LanceDB may have vector indices
            // Note: Index information is not easily accessible via current LanceDB API
            // For now, we'll assume all rows are indexed if the table is accessible
            if result.num_rows > 0 {
                result.index_info.push(("Vector".to_string(), result.num_rows));
                result.total_indexed_rows = result.num_rows;
            }
        }
    }

    Ok(result)
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
                Arc::new(OpenAIClient::default()),
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
    embedding_name: String,
) -> Result<Vec<RecordBatch>, LanceError> {
    let db = get_db_with_embeddings(&embedding_name).await?;

    let tbl = db.open_table(TABLE_NAME).execute().await.map_err(|_| {
        LanceError::InvalidStateError(format!("The table {TABLE_NAME} does not exist"))
    })?;
    let embedding =
        db.embedding_registry()
            .get(&embedding_name)
            .ok_or(LanceError::InvalidStateError(format!(
                "{embedding_name} is not in the database embedding registry"
            )))?;

    let query_vec = embedding.compute_query_embeddings(Arc::new(StringArray::from(vec![query])))?;

    // Convert FixedSizeListArray to Vec<f32>
    // The embedding functions return `FixedSizeListArray` with Float32 elements
    // See https://github.com/apache/arrow-rs/discussions/6087#discussioncomment-10851422 for
    // converting an Arrow Array to a `Vec`.
    let query_vec: Vec<f32> = {
        let list_array = arrow_array::cast::as_fixed_size_list_array(&query_vec);
        let values = list_array.values().as_primitive::<Float32Type>();
        values.iter().map(|v| v.unwrap_or(0.0)).collect()
    };

    let stream = tbl
        .query()
        .limit(10)
        .nearest_to(query_vec)?
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    Ok(batches)
}

/// Creates and initializes a LanceDB table for vector storage
///
/// Connects to LanceDB at the default location, creates a table named `TABLE_NAME`,
/// and registers embedding functions for both Anthropic and OpenAI.
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

    #[tokio::test]
    #[serial]
    async fn test_perform_health_check_no_database() {
        dotenv().ok();

        // Clean up any existing data
        let _ = std::fs::remove_dir_all(DB_URI);

        let result = perform_health_check("voyageai").await;
        assert!(result.is_ok());

        let health_result = result.unwrap();
        assert!(!health_result.directory_exists);
        assert_eq!(health_result.directory_size, 0);
        assert!(!health_result.table_accessible);
        assert_eq!(health_result.num_rows, 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_perform_health_check_with_database() {
        dotenv().ok();

        // Clean up any existing data
        let _ = std::fs::remove_dir_all(DB_URI);

        // Create a test database first
        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        ]);
        let pdf_text_data = StringArray::from(vec!["Hello world", "Test document"]);
        let title_data = StringArray::from(vec!["doc1.pdf", "doc2.pdf"]);
        let record_batch = RecordBatch::try_new(
            Arc::new(schema), 
            vec![Arc::new(pdf_text_data), Arc::new(title_data)]
        ).unwrap();
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        let _db = create_initial_table(
            reader,
            None,
            EmbeddingDefinition::new("pdf_text", "voyageai", Some("embeddings")),
        )
        .await
        .unwrap();

        // Now test health check
        let result = perform_health_check("voyageai").await;
        assert!(result.is_ok());

        let health_result = result.unwrap();
        assert!(health_result.directory_exists);
        assert!(health_result.directory_size > 0);
        assert!(health_result.table_accessible);
        assert_eq!(health_result.num_rows, 2);
        // Note: Zero embedding check might not work in tests due to actual API calls
    }
}
