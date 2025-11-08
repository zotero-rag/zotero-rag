use chrono::Utc;
use std::future::Future;
use thiserror::Error;

use crate::vector::lance::{DB_URI, LanceError, TABLE_NAME};

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
    /// Backup identifier
    pub backup_id: String,
    /// Original table version
    pub original_version: Option<u64>,
}

/// Creates a backup using LanceDB's internal versioning mechanism.
///
/// This strategy records the current version of the LanceDB table, allowing
/// for rollback to this specific version later.
pub(crate) async fn create_backup(backup_id: &str) -> Result<BackupMetadata, LanceError> {
    // Connect to the database to get current version
    let db = lancedb::connect(DB_URI)
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
        backup_id: backup_id.to_string(),
        original_version: Some(current_version),
    })
}

/// Restores the database from the active backup.
///
/// This method uses LanceDB's `checkout` and `restore` functionalities
/// to revert the table to the version recorded in the backup metadata.
pub(crate) async fn restore_backup(
    backup_metadata: &BackupMetadata,
    backup_id: &str,
) -> Result<(), BackupError> {
    if backup_metadata.backup_id != backup_id {
        return Err(BackupError::BackupNotFound(backup_id.to_string()));
    }

    let original_version = backup_metadata.original_version.ok_or_else(|| {
        BackupError::ValidationError("Version backup missing original version".to_string())
    })?;

    // Connect to database
    let db = lancedb::connect(DB_URI)
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
/// # Examples
///
/// ```
/// use crate::vector::backup::{with_backup, BackupError};
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
/// async fn main() -> Result<(), BackupError> {
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
    // Create a unique backup ID
    let backup_id = format!("backup_{}", Utc::now().format("%Y%m%d_%H%M%S"));

    // Create backup
    let backup_metadata = create_backup(&backup_id).await?;

    match operation.await {
        Ok(result) => Ok(result),
        Err(e) => {
            // Failure: restore backup
            if let Err(restore_error) = restore_backup(&backup_metadata, &backup_id).await {
                log::error!("Failed to restore backup {}: {}", backup_id, restore_error);

                // Log the restore error but still return the original operation error
                return Err(LanceError::Other(Box::new(e)));
            }

            Err(LanceError::Other(Box::new(e)))
        }
    }
}
