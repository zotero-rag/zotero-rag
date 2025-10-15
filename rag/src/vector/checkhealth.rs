use crate::embedding::common::get_embedding_dims_by_provider;

use super::lance::LanceError;
use super::lance::{DB_URI, TABLE_NAME};
use arrow_array::RecordBatch;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Table, connect};
use std::fmt;
use std::fs;
use std::io;
use std::path::PathBuf;

/// ANSI color codes for console output
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const RESET: &str = "\x1b[0m";

/// Health check result for LanceDB
#[derive(Debug)]
#[must_use = "You should probably use this; functions exposing this generally do not have side effects."]
pub struct HealthCheckResult {
    /// Directory exists
    pub directory_exists: bool,
    /// Directory size in bytes. `None` when the check hasn't run, `Some(Ok(size))` if
    /// we could compute the size and `Some(Err(...))` when there was an error connecting
    /// to the DB or opening the table.
    pub directory_size: Option<Result<u64, io::Error>>,
    /// Table can be opened. `None` when the check hasn't run, `Some(Ok())` if it is accessible,
    /// and `Some(Err(...))` when there was an error connecting to the DB or opening the table.
    pub table_accessible: Option<Result<(), LanceError>>,
    /// Number of rows in the table. `None` when the check hasn't run, `Some(Ok())` if it is
    /// accessible, and `Some(Err(...))` when there was an error connecting to the DB or opening
    /// the table.
    pub num_rows: Option<Result<usize, LanceError>>,
    /// Rows with all-zero embeddings. `None` when the check hasn't run, `Some(Ok(batches))` if
    /// we successfully computed the complete record batches with zero embeddings, and `Some(Err(...))` when
    /// there was an error connecting to the DB or opening the table.
    pub zero_embedding_items: Option<Result<Vec<RecordBatch>, LanceError>>,
    /// Index information: (index_name, index_type). `None` when the check hasn't run,
    /// `Some(Ok(index_info))` if we successfully got the index information, and `Some(Err(...))`
    /// when there was an error connecting to the DB or opening the table.
    pub index_info: Option<Result<Vec<(String, String)>, LanceError>>,
}

/// Format file size in a human-readable format
///
/// # Arguments:
/// * `bytes` - Size in bytes
///
/// # Returns:
/// A string with a human-readable file size.
fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

impl fmt::Display for HealthCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LanceDB Health Check Results")?;
        writeln!(f, "===============================")?;

        // Check 1: Directory existence and size
        if self.directory_exists {
            writeln!(f, "{}✓ Database directory exists{}", GREEN, RESET)?;
            match &self.directory_size {
                Some(Ok(size)) => {
                    writeln!(f, "\tSize: {}", format_file_size(*size))?;
                }
                Some(Err(e)) => {
                    writeln!(
                        f,
                        "\t{}Error: Failed to calculate size: {}{}",
                        RED, e, RESET
                    )?;
                }
                None => writeln!(f)?,
            }
        } else {
            writeln!(f, "{}✗ Database directory does not exist{}", RED, RESET)?;
            writeln!(
                f,
                "{}  → Subsequent checks will be skipped{}",
                YELLOW, RESET
            )?;
            return Ok(());
        }
        writeln!(f)?;

        // Check 2: Table accessibility
        match &self.table_accessible {
            Some(Ok(())) => {
                writeln!(f, "{}✓ Table is accessible{}", GREEN, RESET)?;
            }
            Some(Err(e)) => {
                writeln!(f, "{}✗ Table is not accessible: {}{}", RED, e, RESET)?;
                writeln!(
                    f,
                    "{}  → Subsequent checks will be skipped{}",
                    YELLOW, RESET
                )?;
                return Ok(());
            }
            None => {
                writeln!(
                    f,
                    "{}⚠ Table accessibility check was skipped{}",
                    YELLOW, RESET
                )?;
                return Ok(());
            }
        }

        // Check 3: Row count
        match &self.num_rows {
            Some(Ok(count)) => {
                if *count == 0 {
                    writeln!(f, "{}⚠ Table has no rows{}", YELLOW, RESET)?;
                } else {
                    writeln!(f, "\tTable has {} rows{}", count, RESET)?;
                }
            }
            Some(Err(e)) => {
                writeln!(f, "{}✗ Failed to get row count: {}{}", RED, e, RESET)?;
            }
            None => {
                writeln!(f, "{}⚠ Row count check was skipped{}", YELLOW, RESET)?;
            }
        }
        writeln!(f)?;

        // Check 4: Zero embeddings
        match &self.zero_embedding_items {
            Some(Ok(zero_batches)) => {
                let total_zero_rows: usize =
                    zero_batches.iter().map(|batch| batch.num_rows()).sum();
                if total_zero_rows == 0 {
                    writeln!(f, "{}✓ No zero embeddings found{}", GREEN, RESET)?;
                } else {
                    writeln!(
                        f,
                        "{}⚠ Found {} rows with zero embeddings{}. Run /embed to fix.",
                        YELLOW, total_zero_rows, RESET
                    )?;
                }
            }
            Some(Err(e)) => {
                writeln!(
                    f,
                    "{}✗ Failed to check zero embeddings: {}{}",
                    RED, e, RESET
                )?;
            }
            None => {
                writeln!(f, "{}⚠ Zero embeddings check was skipped{}", YELLOW, RESET)?;
            }
        }
        writeln!(f)?;

        // Check 5: Index information
        match &self.index_info {
            Some(Ok(indices)) => {
                if indices.is_empty() {
                    if let Some(Ok(row_count)) = self.num_rows {
                        if row_count > 10000 {
                            writeln!(
                                f,
                                "{}⚠ No indices found (may impact query performance){}",
                                YELLOW, RESET
                            )?;
                        } else {
                            writeln!(
                                f,
                                "{}✓ No indices found. This should not affect performance for your library size.{}",
                                GREEN, RESET
                            )?;
                        }
                    }
                } else {
                    writeln!(f, "{}✓ Found {} index(es):{}", GREEN, indices.len(), RESET)?;
                    for (name, index_type) in indices {
                        writeln!(f, "\t- {} ({})", name, index_type)?;
                    }
                }
            }
            Some(Err(e)) => {
                writeln!(
                    f,
                    "{}✗ Failed to get index information: {}{}",
                    RED, e, RESET
                )?;
            }
            None => {
                writeln!(
                    f,
                    "{}⚠ Index information check was skipped{}",
                    YELLOW, RESET
                )?;
            }
        }

        Ok(())
    }
}

/// Calculate the size of a directory recursively.
///
/// # Arguments:
/// * `path` - The directory whose size needs to be computed
///
/// # Returns
/// The size in bytes, if successful.
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

/// Given a table, an expected schema, and a limit for the number of rows to query in the table,
/// check if any rows have zero embeddings, which is a sign that something went wrong.
///
/// # Arguments:
/// * `tbl` - The LanceDB table to query
/// * `embeddings_provider` - The embedding provider used. Must be one of `EmbeddingProviders`.
/// * `query_limit` - Limit on the number of rows in the table to query
///
/// # Returns:
/// * A list of complete RecordBatches for rows that have zero embeddings, if nothing went wrong
/// * Otherwise, a `LanceError` detailing what went wrong and why:
///     * A `QueryError` if some query failed
///     * An `InvalidStateError` if the table is in some invalid state
pub(crate) async fn get_zero_vectors(
    tbl: &Table,
    embedding_provider: &str,
    query_limit: usize,
) -> Result<Vec<RecordBatch>, LanceError> {
    let embedding_size = get_embedding_dims_by_provider(embedding_provider);

    let stream = tbl
        .query()
        .nearest_to(vec![0.0; embedding_size as usize])?
        .distance_range(Some(0.0), Some(1e-8))
        .limit(query_limit)
        .execute()
        .await
        .map_err(|e| LanceError::QueryError(e.to_string()))?;

    stream
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| LanceError::QueryError(e.to_string()))
}

/// Given a LanceDB table, get basic index information. Note that LanceDB's Rust API does not
/// expose all index information directly, so we'll use what's available.
///
/// # Arguments:
/// * `tbl` - The LanceDB table to get info from
///
/// # Returns:
/// Index information if we were able to list indices successfully.
async fn check_indexes(tbl: &lancedb::table::Table) -> Result<Vec<(String, String)>, LanceError> {
    match tbl.list_indices().await {
        Ok(indices) => {
            if indices.is_empty() {
                return Ok(Vec::new());
            }

            Ok(indices
                .iter()
                .map(|index| (index.name.clone(), index.index_type.to_string()))
                .collect::<Vec<_>>())
        }
        Err(e) => Err(LanceError::QueryError(format!(
            "Failed to get indexes: {e}"
        ))),
    }
}

/// Performs a comprehensive health check on the LanceDB database.
///
/// This function checks for:
/// - Directory existence and reports size
/// - Table accessibility
/// - Row count (errors if zero rows)
/// - All-zero embeddings (errors if found, writes titles to bad_embeddings.txt)
/// - Index information (warns if missing or incomplete)
///
/// # Arguments:
/// * `schema` - The expected schema for the LanceDB table.
/// * `text_col` - The name of the column containing the full texts.
/// * `embedding_provider` - The embedding provider. Must be one of `EmbeddingProviders`.
///
/// # Returns:
///
/// A `Result<HealthCheckResult, LanceError>` indicating success and whether issues were found
#[must_use = "This function has no side-effects, so you likely want to inspect this value."]
pub async fn lancedb_health_check(
    embedding_provider: &str,
) -> Result<HealthCheckResult, LanceError> {
    let mut result = HealthCheckResult {
        directory_exists: false,
        directory_size: None,
        table_accessible: None,
        num_rows: None,
        zero_embedding_items: None,
        index_info: None,
    };

    // Check 1: Directory existence and size
    let db_path = PathBuf::from(DB_URI);
    result.directory_exists = db_path.exists();

    if result.directory_exists {
        result.directory_size = Some(calculate_directory_size(&db_path));
    } else {
        // If the directory doesn't exist, none of the other checks make sense.
        return Ok(result);
    }

    // Check 2: Table connectivity
    let db_connection = connect(DB_URI).execute().await;
    if let Err(e) = db_connection {
        // Capture the error
        result.table_accessible = Some(Err(LanceError::ConnectionError(e.to_string())));

        // None of the future checks make sense here.
        return Ok(result);
    }

    // Check 3: Table opening
    let db = db_connection.unwrap();
    let tbl = db.open_table(TABLE_NAME).execute().await;
    if let Err(e) = tbl {
        // Capture the error
        result.table_accessible = Some(Err(LanceError::ConnectionError(e.to_string())));

        // None of the future checks make sense here.
        return Ok(result);
    }

    let tbl = tbl.unwrap();
    result.table_accessible = Some(Ok(()));

    // Check 4: Row count
    // If we get an error here, it may be a temporary error for just this query, so we won't stop
    // our health checks.
    result.num_rows = match tbl.count_rows(None).await {
        Ok(count) => Some(Ok(count)),
        Err(e) => Some(Err(LanceError::QueryError(e.to_string()))),
    };

    // Check 5: Check indexes
    result.index_info = Some(check_indexes(&tbl).await);

    // Check 6: All-zero embeddings
    // We can't have `result.num_rows` be `None` at this point.
    if let Some(query_limit) = &result.num_rows {
        result.zero_embedding_items = match query_limit {
            Ok(count) => Some(Ok(get_zero_vectors(&tbl, embedding_provider, *count).await?)),
            Err(e) => Some(Err(LanceError::QueryError(e.to_string()))),
        }
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::lancedb_health_check;
    use crate::vector::lance::DB_URI;
    use crate::vector::lance::insert_records;
    use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingDefinition;
    use serial_test::serial;
    use std::sync::Arc;

    #[tokio::test]
    #[serial]
    async fn test_perform_health_check_no_database() {
        dotenv().ok();

        // Clean up any existing data
        let _ = std::fs::remove_dir_all(DB_URI);
        let _ = std::fs::remove_dir_all(format!("rag/{}", DB_URI));

        let result = lancedb_health_check("voyageai").await;
        assert!(result.is_ok());

        let health_result = result.unwrap();
        assert!(!health_result.directory_exists);
        assert!(health_result.directory_size.is_none());
        assert!(health_result.table_accessible.is_none());
        assert!(health_result.num_rows.is_none());
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
            Arc::new(schema.clone()),
            vec![Arc::new(pdf_text_data), Arc::new(title_data)],
        )
        .unwrap();
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        let _db = insert_records(
            reader,
            None,
            EmbeddingDefinition::new("pdf_text", "voyageai", Some("embeddings")),
        )
        .await
        .unwrap();

        // Now test health check
        let result = lancedb_health_check("voyageai").await;
        assert!(result.is_ok());

        let health_result = result.unwrap();
        assert!(health_result.directory_exists);
        assert!(health_result.directory_size.is_some());
        assert!(health_result.directory_size.unwrap().is_ok_and(|x| x > 0));
        assert!(health_result.table_accessible.is_some());
        assert!(health_result.num_rows.is_some());
        assert!(health_result.num_rows.unwrap().is_ok_and(|x| x == 2));
    }
}
