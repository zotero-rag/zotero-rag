use chrono::{DateTime, Utc};
use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::vector::lance::{DB_URI, LanceError, TABLE_NAME};

/// Errors that can occur during backup operations
#[derive(Debug, Error)]
pub enum BackupError {
    /// IO error during backup operations
    #[error("IO error during backup: {0}")]
    IoError(#[from] std::io::Error),
    /// LanceDB error during backup operations
    #[error("LanceDB error during backup: {0}")]
    LanceError(#[from] LanceError),
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

/// Backup strategy to use
#[derive(Debug, Clone)]
pub enum BackupStrategy {
    /// Copy the entire database directory
    DirectoryCopy,
    /// Use LanceDB's built-in versioning (restore to previous version)
    LanceVersion,
}

/// Metadata about a backup
#[derive(Debug, Clone)]
pub struct BackupMetadata {
    /// Backup identifier
    pub backup_id: String,
    /// Timestamp when backup was created
    pub created_at: DateTime<Utc>,
    /// Strategy used for this backup
    pub strategy: BackupStrategy,
    /// Original database path
    pub original_path: PathBuf,
    /// Backup location (for directory copies)
    pub backup_path: Option<PathBuf>,
    /// Original table version (for LanceDB versions)
    pub original_version: Option<u64>,
}

/// Database backup manager
/// Manages database backups, supporting different strategies like directory copying and LanceDB's built-in versioning.
pub struct BackupManager {
    /// Directory where backups are stored
    backup_dir: PathBuf,
    /// Active backup metadata
    active_backup: Option<BackupMetadata>,
}

/// Recursively copies the contents of a source directory to a destination directory.
///
/// If the destination directory does not exist, it will be created.
fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

impl BackupManager {
    /// Creates a new `BackupManager`.
    ///
    /// If `backup_dir` is `None`, it defaults to "data/backups".
    pub fn new(backup_dir: Option<PathBuf>) -> Self {
        let backup_dir = backup_dir.unwrap_or_else(|| PathBuf::from("data/backups"));

        Self {
            backup_dir,
            active_backup: None,
        }
    }

    /// Creates a backup of the database using the specified strategy.
    ///
    /// This method ensures the backup directory exists and then delegates to the
    /// appropriate backup creation method based on the `BackupStrategy`.
    /// The created backup's metadata is stored as the `active_backup`.
    pub async fn create_backup(&mut self, strategy: BackupStrategy) -> Result<String, BackupError> {
        let now = Utc::now();
        let backup_id = format!("backup_{}", now.format("%Y%m%d_%H%M%S"));

        // Ensure backup directory exists
        if !self.backup_dir.exists() {
            fs::create_dir_all(&self.backup_dir)?;
        }

        let metadata = match strategy {
            BackupStrategy::DirectoryCopy => self.create_directory_backup(&backup_id).await?,
            BackupStrategy::LanceVersion => self.create_version_backup(&backup_id).await?,
        };

        self.active_backup = Some(metadata);
        Ok(backup_id)
    }

    /// Creates a backup by recursively copying the entire database directory.
    ///
    /// This strategy is suitable for databases where the entire directory can be
    /// copied to create a consistent snapshot.
    async fn create_directory_backup(
        &self,
        backup_id: &str,
    ) -> Result<BackupMetadata, BackupError> {
        let db_path = PathBuf::from(DB_URI);

        if !db_path.exists() {
            return Err(BackupError::ValidationError(
                "Database directory does not exist".to_string(),
            ));
        }

        let backup_path = self.backup_dir.join(backup_id);

        // Copy directory recursively
        copy_dir_all(&db_path, &backup_path)?;

        Ok(BackupMetadata {
            backup_id: backup_id.to_string(),
            created_at: Utc::now(),
            strategy: BackupStrategy::DirectoryCopy,
            original_path: db_path,
            backup_path: Some(backup_path),
            original_version: None,
        })
    }

    /// Creates a backup using LanceDB's internal versioning mechanism.
    ///
    /// This strategy records the current version of the LanceDB table, allowing
    /// for rollback to this specific version later.
    async fn create_version_backup(&self, backup_id: &str) -> Result<BackupMetadata, BackupError> {
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
            created_at: Utc::now(),
            strategy: BackupStrategy::LanceVersion,
            original_path: PathBuf::from(DB_URI),
            backup_path: None,
            original_version: Some(current_version),
        })
    }

    /// Restores the database from the active backup.
    ///
    /// This method checks the `backup_id` against the active backup and
    /// delegates to the appropriate restore method based on the backup strategy.
    pub async fn restore_backup(&mut self, backup_id: &str) -> Result<(), BackupError> {
        let backup = self
            .active_backup
            .as_ref()
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if backup.backup_id != backup_id {
            return Err(BackupError::BackupNotFound(backup_id.to_string()));
        }

        match &backup.strategy {
            BackupStrategy::DirectoryCopy => {
                self.restore_directory_backup(backup).await?;
            }
            BackupStrategy::LanceVersion => {
                self.restore_version_backup(backup).await?;
            }
        }

        Ok(())
    }

    /// Restores the database from a directory copy backup.
    ///
    /// This involves removing the current database directory and copying the
    /// backup directory back to the original database path.
    async fn restore_directory_backup(&self, backup: &BackupMetadata) -> Result<(), BackupError> {
        let backup_path = backup.backup_path.as_ref().ok_or_else(|| {
            BackupError::ValidationError("Directory backup missing backup path".to_string())
        })?;

        if !backup_path.exists() {
            return Err(BackupError::BackupNotFound(format!(
                "Backup directory not found: {}",
                backup_path.display()
            )));
        }

        // Remove current database directory
        if backup.original_path.exists() {
            fs::remove_dir_all(&backup.original_path)?;
        }

        // Copy backup back to original location
        copy_dir_all(backup_path, &backup.original_path)?;

        Ok(())
    }

    /// Restores the LanceDB table to a previous version.
    ///
    /// This method uses LanceDB's `checkout` and `restore` functionalities
    /// to revert the table to the version recorded in the backup metadata.
    async fn restore_version_backup(&self, backup: &BackupMetadata) -> Result<(), BackupError> {
        let original_version = backup.original_version.ok_or_else(|| {
            BackupError::ValidationError("Version backup missing original version".to_string())
        })?;

        // Connect to database
        let db = lancedb::connect(DB_URI)
            .execute()
            .await
            .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

        let table = db
            .open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

        // Restore to the previous version
        match table.checkout(original_version).await {
            Ok(_) => {
                table.restore().await?;
            }
            Err(e) => {
                // `checkout` only creates a "detached head state", so it failing is not (yet) a
                // problem.
                log::warn!(
                    "Restoring version backup failed: {e}; we will checkout the latest version."
                );

                // These two failing *is* an issue, so we propagate the error up.
                table.checkout_latest().await?;
                table.restore().await?;
            }
        }

        Ok(())
    }

    /// Cleans up a successful backup by deleting the backup files/data.
    ///
    /// For `DirectoryCopy` strategy, it removes the copied directory.
    /// For `LanceVersion` strategy, no explicit cleanup is needed as versions
    /// are managed internally by LanceDB.
    pub async fn cleanup_backup(&mut self, backup_id: &str) -> Result<(), BackupError> {
        let backup = self
            .active_backup
            .as_ref()
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if backup.backup_id != backup_id {
            return Err(BackupError::BackupNotFound(backup_id.to_string()));
        }

        match &backup.strategy {
            BackupStrategy::DirectoryCopy => {
                if let Some(backup_path) = &backup.backup_path
                    && backup_path.exists()
                {
                    fs::remove_dir_all(backup_path)?;
                }
            }
            BackupStrategy::LanceVersion => {
                // For version-based backups, we don't need to clean up anything
                // The versions are managed by LanceDB internally
            }
        }

        self.active_backup = None;
        Ok(())
    }

    /// Returns a reference to the active backup's metadata, if one exists.
    pub fn get_active_backup(&self) -> Option<&BackupMetadata> {
        self.active_backup.as_ref()
    }
}

/// A convenience function to wrap asynchronous database operations with a backup and restore mechanism.
///
/// This function will:
/// 1. Create a backup using the specified `BackupStrategy`.
/// 2. Execute the provided asynchronous `operation`.
/// 3. If the `operation` succeeds, clean up the created backup.
/// 4. If the `operation` fails, attempt to restore the database from the backup.
///
/// # Arguments:
///
/// * `strategy` - The `BackupStrategy` to use for creating the backup.
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
/// use crate::vector::backup::{with_backup, BackupStrategy, BackupError};
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
///     with_backup(BackupStrategy::DirectoryCopy, my_database_operation()).await?;
///     println!("Operation completed with backup and cleanup.");
///     Ok(())
/// }
/// ```
pub async fn with_backup<F, R, E>(strategy: BackupStrategy, operation: F) -> Result<R, BackupError>
where
    F: Future<Output = Result<R, E>>,
    E: std::error::Error + Send + Sync + 'static,
{
    let mut backup_manager = BackupManager::new(None);

    // Create backup
    let backup_id = backup_manager.create_backup(strategy).await?;

    // Execute operation
    match operation.await {
        Ok(result) => {
            // Success: cleanup backup
            backup_manager.cleanup_backup(&backup_id).await?;
            Ok(result)
        }
        Err(error) => {
            // Failure: restore backup
            if let Err(restore_error) = backup_manager.restore_backup(&backup_id).await {
                log::error!("Failed to restore backup: {}", restore_error);
                // Log the restore error but still return the original operation error
            }

            // Convert the operation error to our error type
            Err(BackupError::OperationFailed(Box::new(error)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_backup_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let backup_manager = BackupManager::new(Some(temp_dir.path().to_path_buf()));

        assert!(backup_manager.active_backup.is_none());
        assert_eq!(backup_manager.backup_dir, temp_dir.path());
    }

    #[tokio::test]
    async fn test_directory_copy_backup() {
        let temp_dir = TempDir::new().unwrap();
        let db_dir = temp_dir.path().join("test_db");
        let backup_dir = temp_dir.path().join("backups");

        // Create a mock database directory with some files
        fs::create_dir_all(&db_dir).unwrap();
        fs::write(db_dir.join("test_file.txt"), "test data").unwrap();

        let mut backup_manager = BackupManager::new(Some(backup_dir.clone()));

        // Verify the backup manager is initialized correctly
        assert!(backup_manager.active_backup.is_none());
        assert_eq!(backup_manager.backup_dir, backup_dir);

        let res = backup_manager
            .create_backup(BackupStrategy::DirectoryCopy)
            .await;
        assert!(res.is_ok());

        let res = res.unwrap();
        assert!(res.starts_with("backup"));

        // Check the copying was done
        assert!(backup_manager.backup_dir.exists());
        assert!(backup_manager.backup_dir.join("test_file.txt").exists());
    }

    #[test]
    fn test_copy_dir_all() {
        let temp_dir = TempDir::new().unwrap();
        let src_dir = temp_dir.path().join("src");
        let dst_dir = temp_dir.path().join("dst");

        // Create source directory structure
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(src_dir.join("file1.txt"), "content1").unwrap();
        fs::create_dir_all(src_dir.join("subdir")).unwrap();
        fs::write(src_dir.join("subdir").join("file2.txt"), "content2").unwrap();

        copy_dir_all(&src_dir, &dst_dir).unwrap();

        // Verify files were copied
        assert!(dst_dir.join("file1.txt").exists());
        assert!(dst_dir.join("subdir").join("file2.txt").exists());
        assert_eq!(
            fs::read_to_string(dst_dir.join("file1.txt")).unwrap(),
            "content1"
        );
        assert_eq!(
            fs::read_to_string(dst_dir.join("subdir").join("file2.txt")).unwrap(),
            "content2"
        );
    }
}
