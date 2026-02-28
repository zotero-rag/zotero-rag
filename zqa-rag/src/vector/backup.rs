//! Utilities for performing operations with backups. The backup functionality exposes LanceDB's
//! versioning mechanism.

use std::future::Future;
use thiserror::Error;

use crate::vector::lance::{LanceError, TABLE_NAME, get_db_uri};

/// Errors that can occur during backup operations
#[derive(Debug, Error)]
pub enum BackupError {
    /// LanceDB library error during backup operations
    #[error("LanceDB library error during backup: {0}")]
    LanceDbError(#[from] lancedb::Error),
    /// Backup validation error
    #[error("Backup validation failed: {0}")]
    ValidationError(String),
    /// Backup not found
    #[error("Backup not found: {0}")]
    BackupNotFound(String),
    /// Operation failed error
    #[error("Operation failed: {0}")]
    OperationFailed(Box<dyn std::error::Error + Send + Sync>),
}

/// Metadata about a backup
#[derive(Debug, Clone)]
pub(crate) struct BackupMetadata {
    /// Original table version
    pub original_version: Option<u64>,
}

/// TODO: Refactor this to take in a `Connection` to reduce overhead of repeatedly connecting to
/// the DB.
///
/// Creates a backup using LanceDB's internal versioning mechanism.
///
/// Warning: This function is currently not safe for concurrent use. Specifically, it is possible for
/// another process/thread to modify the table between the time the backup is created and the time
/// the operation is executed, resulting in data loss.
pub(crate) async fn create_backup() -> Result<BackupMetadata, LanceError> {
    // Connect to the database to get current version
    let db = lancedb::connect(&get_db_uri())
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    let table = db
        .open_table(TABLE_NAME)
        .execute()
        .await
        .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

    let current_version = table.version().await?;

    Ok(BackupMetadata {
        original_version: Some(current_version),
    })
}

/// Restores the database from the active backup.
///
/// This method uses LanceDB's `checkout` and `restore` functionalities
/// to revert the table to the version recorded in the backup metadata.
pub(crate) async fn restore_backup(backup_metadata: &BackupMetadata) -> Result<(), BackupError> {
    let original_version = backup_metadata.original_version.ok_or_else(|| {
        BackupError::ValidationError("Version backup missing original version".to_string())
    })?;

    // Connect to database
    let db = lancedb::connect(&get_db_uri())
        .execute()
        .await
        .map_err(BackupError::LanceDbError)?;

    let table = db
        .open_table(TABLE_NAME)
        .execute()
        .await
        .map_err(BackupError::LanceDbError)?;

    // Restore to the previous version
    table
        .checkout(original_version)
        .await
        .map_err(BackupError::LanceDbError)?;
    table.restore().await.map_err(BackupError::LanceDbError)?;

    Ok(())
}

/// A convenience function to wrap asynchronous database operations with a backup and restore mechanism.
///
/// This function will:
/// 1. Create a backup
/// 2. Execute the provided asynchronous `operation`.
/// 3. If the `operation` fails, attempt to restore the database from the backup.
///
/// # Arguments:
///
/// * `operation` - An asynchronous operation that returns a `Result`.
///
/// # Returns
///
/// If successful, the `Ok` variant of the return type of `operation`. If unsuccessful, a
/// `BackupError` with details. In this case, logs are written at the WARN level explaining what
/// stages failed.
///
/// # Errors
///
/// * `LanceError::ConnectionError` - If creating a backup fails due to database connection issues
/// * `LanceError::InvalidStateError` - If both the operation and backup restoration fail
/// * `LanceError::Other` - If the operation fails (wrapping the operation's error)
///
/// # Examples
///
/// ```no_run
/// use zqa_rag::vector::backup::{with_backup };
/// use zqa_rag::vector::lance::LanceError;
/// use std::io;
///
/// async fn my_database_operation() -> Result<(), io::Error> {
///     // Simulate a database operation
///     println!("Performing database operation...");
///     // For demonstration, let's say it always succeeds
///     Ok(())
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), LanceError> {
///     with_backup(my_database_operation()).await?;
///     println!("Operation completed with backup and cleanup.");
///     Ok(())
/// }
/// ```
pub async fn with_backup<F, R, E>(operation: F) -> Result<R, LanceError>
where
    F: Future<Output = Result<R, E>>,
    E: std::error::Error + Send + Sync + 'static,
{
    // Create backup
    let backup_metadata = create_backup().await?;

    match operation.await {
        Ok(result) => Ok(result),
        Err(e) => {
            // Failure: (attempt to) restore backup
            if let Err(restore_error) = restore_backup(&backup_metadata).await {
                log::error!("Failed to restore backup: {restore_error}");

                return Err(LanceError::InvalidStateError(format!(
                    "Operation failed AND restore failed. Database may be in inconsistent state.\nOperation error: {e}.\nRestore error: {restore_error}"
                )));
            }

            Err(LanceError::Other(Box::new(e)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use dotenv::dotenv;
    use lancedb::{connect, embeddings::EmbeddingDefinition};
    use serial_test::serial;
    use std::sync::Arc;
    use zqa_macros::{test_contains, test_eq, test_ok};

    /// Helper function to set up a test database with some initial data
    async fn setup_test_db() -> Result<u64, LanceError> {
        dotenv().ok();

        // Create a simple schema
        let schema = Schema::new(vec![Field::new("test_data", DataType::Utf8, false)]);
        let data = StringArray::from(vec!["test1", "test2"]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data)]).unwrap();
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        // Connect and register embedding function
        let db = connect(&get_db_uri()).execute().await?;

        db.embedding_registry().register(
            "openai",
            Arc::new(crate::llm::openai::OpenAIClient::<
                crate::llm::http_client::ReqwestClient,
            >::default()),
        )?;

        db.create_table(TABLE_NAME, reader)
            .mode(lancedb::database::CreateTableMode::Overwrite)
            .add_embedding(EmbeddingDefinition::new(
                "test_data",
                "openai",
                Some("embeddings"),
            ))?
            .execute()
            .await?;

        // Get the current version
        let table = db.open_table(TABLE_NAME).execute().await?;
        Ok(table.version().await?)
    }

    /// Helper function to add more data to the database (creating a new version)
    async fn add_data_to_db() -> Result<u64, LanceError> {
        let db = connect(&get_db_uri()).execute().await?;

        // Register embedding function
        db.embedding_registry().register(
            "openai",
            Arc::new(crate::llm::openai::OpenAIClient::<
                crate::llm::http_client::ReqwestClient,
            >::default()),
        )?;

        let table = db.open_table(TABLE_NAME).execute().await?;

        // Create new data
        let schema = Schema::new(vec![Field::new("test_data", DataType::Utf8, false)]);
        let data = StringArray::from(vec!["test3", "test4"]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data)]).unwrap();
        let batches = vec![Ok(record_batch.clone())];
        let reader = RecordBatchIterator::new(batches.into_iter(), record_batch.schema());

        // Add data
        table.add(reader).execute().await?;

        Ok(table.version().await?)
    }

    #[tokio::test]
    #[serial]
    async fn test_create_backup_captures_version() {
        let initial_version = setup_test_db().await.expect("Failed to setup test db");

        let backup_metadata = create_backup().await.expect("Failed to create backup");

        assert!(backup_metadata.original_version.is_some());
        test_eq!(backup_metadata.original_version.unwrap(), initial_version);
    }

    #[tokio::test]
    #[serial]
    async fn test_restore_backup_restores_to_previous_version() {
        // Setup initial database with 2 rows
        let initial_version = setup_test_db().await.expect("Failed to setup test db");

        // Get initial row count
        let db = connect(&get_db_uri())
            .execute()
            .await
            .expect("Failed to connect to db");
        let table = db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .expect("Failed to open table");
        let initial_row_count = table
            .count_rows(None)
            .await
            .expect("Failed to get row count");

        // Create backup
        let backup_metadata = create_backup().await.expect("Failed to create backup");
        test_eq!(backup_metadata.original_version.unwrap(), initial_version);

        // Add more data (creates new version and adds 2 more rows)
        let new_version = add_data_to_db().await.expect("Failed to add data");
        assert!(
            new_version > initial_version,
            "New version should be higher"
        );

        // Verify row count increased
        let table = db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .expect("Failed to open table");
        let row_count_after_add = table
            .count_rows(None)
            .await
            .expect("Failed to get row count");
        test_eq!(row_count_after_add, initial_row_count + 2);

        // Restore backup
        restore_backup(&backup_metadata)
            .await
            .expect("Failed to restore backup");

        // Verify row count is restored to original (LanceDB restore creates a new version,
        // but restores the data from the old version)
        let table = db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .expect("Failed to open table");
        let final_row_count = table
            .count_rows(None)
            .await
            .expect("Failed to get row count");

        test_eq!(final_row_count, initial_row_count);
    }

    #[tokio::test]
    #[serial]
    async fn test_restore_backup_fails_with_missing_version() {
        let backup_metadata = BackupMetadata {
            original_version: None,
        };

        let result = restore_backup(&backup_metadata).await;

        assert!(result.is_err());
        match result {
            Err(BackupError::ValidationError(msg)) => {
                assert!(msg.contains("missing original version"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_with_backup_succeeds_on_successful_operation() {
        setup_test_db().await.expect("Failed to setup test db");

        let result = with_backup(async {
            // Simulate a successful operation
            Ok::<_, std::io::Error>(42)
        })
        .await;

        test_ok!(result);
        test_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    #[serial]
    async fn test_with_backup_restores_on_failed_operation() {
        // Setup initial database with 2 rows
        setup_test_db().await.expect("Failed to setup test db");

        // Get initial row count
        let db = connect(&get_db_uri())
            .execute()
            .await
            .expect("Failed to connect to db");
        let table = db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .expect("Failed to open table");
        let initial_row_count = table
            .count_rows(None)
            .await
            .expect("Failed to get row count");

        let result = with_backup(async {
            // Add data inside the operation (adds 2 more rows)
            add_data_to_db().await.expect("Failed to add data");

            // Simulate a failure
            Err::<(), _>(std::io::Error::other("Simulated failure"))
        })
        .await;

        // The operation should fail
        assert!(result.is_err());

        // Verify the database was restored to the original row count
        let table = db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .expect("Failed to open table");
        let final_row_count = table
            .count_rows(None)
            .await
            .expect("Failed to get row count");

        test_eq!(final_row_count, initial_row_count);
    }

    #[tokio::test]
    #[serial]
    async fn test_with_backup_preserves_error_message() {
        setup_test_db().await.expect("Failed to setup test db");

        let error_message = "Custom error message";
        let result =
            with_backup(async { Err::<(), _>(std::io::Error::other(error_message)) }).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        test_contains!(err.to_string(), error_message);
    }

    #[tokio::test]
    #[serial]
    async fn test_backup_metadata_clone() {
        let metadata = BackupMetadata {
            original_version: Some(42),
        };

        let cloned = metadata.clone();

        test_eq!(metadata.original_version, cloned.original_version);
    }

    #[tokio::test]
    #[serial]
    async fn test_backup_metadata_debug() {
        let metadata = BackupMetadata {
            original_version: Some(123),
        };

        let debug_str = format!("{metadata:?}");

        assert!(debug_str.contains("BackupMetadata"));
        assert!(debug_str.contains("123"));
    }
}
