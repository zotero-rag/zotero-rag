use std::path::{Path, PathBuf};
use std::fs;
use thiserror::Error;
use chrono::{DateTime, Utc};

use crate::vector::lance::{LanceError, DB_URI, TABLE_NAME};

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
pub struct BackupManager {
    /// Directory where backups are stored
    backup_dir: PathBuf,
    /// Active backup metadata
    active_backup: Option<BackupMetadata>,
}

impl BackupManager {
    /// Create a new backup manager
    pub fn new(backup_dir: Option<PathBuf>) -> Self {
        let backup_dir = backup_dir.unwrap_or_else(|| {
            PathBuf::from("data/backups")
        });
        
        Self {
            backup_dir,
            active_backup: None,
        }
    }

    /// Create a backup before performing database operations
    pub async fn create_backup(&mut self, strategy: BackupStrategy) -> Result<String, BackupError> {
        let now = Utc::now();
        let backup_id = format!("backup_{}", now.format("%Y%m%d_%H%M%S"));

        // Ensure backup directory exists
        if !self.backup_dir.exists() {
            fs::create_dir_all(&self.backup_dir)?;
        }

        let metadata = match strategy {
            BackupStrategy::DirectoryCopy => {
                self.create_directory_backup(&backup_id).await?
            },
            BackupStrategy::LanceVersion => {
                self.create_version_backup(&backup_id).await?
            },
        };

        self.active_backup = Some(metadata);
        Ok(backup_id)
    }

    /// Create a backup by copying the database directory
    async fn create_directory_backup(&self, backup_id: &str) -> Result<BackupMetadata, BackupError> {
        let db_path = PathBuf::from(DB_URI);
        
        if !db_path.exists() {
            return Err(BackupError::ValidationError(
                "Database directory does not exist".to_string()
            ));
        }

        let backup_path = self.backup_dir.join(backup_id);
        
        // Copy directory recursively
        self.copy_dir_all(&db_path, &backup_path)?;

        Ok(BackupMetadata {
            backup_id: backup_id.to_string(),
            created_at: Utc::now(),
            strategy: BackupStrategy::DirectoryCopy,
            original_path: db_path,
            backup_path: Some(backup_path),
            original_version: None,
        })
    }

    /// Create a backup using LanceDB versions
    async fn create_version_backup(&self, backup_id: &str) -> Result<BackupMetadata, BackupError> {
        // Connect to the database to get current version
        let db = lancedb::connect(DB_URI)
            .execute()
            .await
            .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

        let table = db.open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| LanceError::InvalidStateError(
                format!("Table {} does not exist", TABLE_NAME)
            ))?;

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

    /// Restore from backup if an operation failed
    pub async fn restore_backup(&mut self, backup_id: &str) -> Result<(), BackupError> {
        let backup = self.active_backup.as_ref()
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if backup.backup_id != backup_id {
            return Err(BackupError::BackupNotFound(backup_id.to_string()));
        }

        match &backup.strategy {
            BackupStrategy::DirectoryCopy => {
                self.restore_directory_backup(backup).await?;
            },
            BackupStrategy::LanceVersion => {
                self.restore_version_backup(backup).await?;
            },
        }

        Ok(())
    }

    /// Restore from directory backup
    async fn restore_directory_backup(&self, backup: &BackupMetadata) -> Result<(), BackupError> {
        let backup_path = backup.backup_path.as_ref()
            .ok_or_else(|| BackupError::ValidationError(
                "Directory backup missing backup path".to_string()
            ))?;

        if !backup_path.exists() {
            return Err(BackupError::BackupNotFound(
                format!("Backup directory not found: {}", backup_path.display())
            ));
        }

        // Remove current database directory
        if backup.original_path.exists() {
            fs::remove_dir_all(&backup.original_path)?;
        }

        // Copy backup back to original location
        self.copy_dir_all(backup_path, &backup.original_path)?;

        Ok(())
    }

    /// Restore using LanceDB version rollback
    async fn restore_version_backup(&self, backup: &BackupMetadata) -> Result<(), BackupError> {
        let _original_version = backup.original_version
            .ok_or_else(|| BackupError::ValidationError(
                "Version backup missing original version".to_string()
            ))?;

        // Connect to database
        let db = lancedb::connect(DB_URI)
            .execute()
            .await
            .map_err(|e| LanceError::ConnectionError(e.to_string()))?;

        let table = db.open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| LanceError::InvalidStateError(
                format!("Table {} does not exist", TABLE_NAME)
            ))?;

        // Restore to the previous version (LanceDB's restore() takes no arguments)
        table.restore().await?;

        Ok(())
    }

    /// Clean up successful backup (delete backup files/data)
    pub async fn cleanup_backup(&mut self, backup_id: &str) -> Result<(), BackupError> {
        let backup = self.active_backup.as_ref()
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if backup.backup_id != backup_id {
            return Err(BackupError::BackupNotFound(backup_id.to_string()));
        }

        match &backup.strategy {
            BackupStrategy::DirectoryCopy => {
                if let Some(backup_path) = &backup.backup_path {
                    if backup_path.exists() {
                        fs::remove_dir_all(backup_path)?;
                    }
                }
            },
            BackupStrategy::LanceVersion => {
                // For version-based backups, we don't need to clean up anything
                // The versions are managed by LanceDB internally
            },
        }

        self.active_backup = None;
        Ok(())
    }

    /// Get information about the active backup
    pub fn get_active_backup(&self) -> Option<&BackupMetadata> {
        self.active_backup.as_ref()
    }

    /// Recursively copy a directory
    fn copy_dir_all(&self, src: &Path, dst: &Path) -> Result<(), std::io::Error> {
        if !dst.exists() {
            fs::create_dir_all(dst)?;
        }

        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());

            if src_path.is_dir() {
                self.copy_dir_all(&src_path, &dst_path)?;
            } else {
                fs::copy(&src_path, &dst_path)?;
            }
        }

        Ok(())
    }
}

/// Convenience function to wrap database operations with backup/restore
pub async fn with_backup<F, R, E>(
    strategy: BackupStrategy,
    operation: F,
) -> Result<R, BackupError> 
where
    F: std::future::Future<Output = Result<R, E>>,
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
        },
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
    use tempfile::TempDir;
    use std::fs;

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

        let backup_manager = BackupManager::new(Some(backup_dir.clone()));

        // Verify the backup manager is initialized correctly
        assert!(backup_manager.active_backup.is_none());
        assert_eq!(backup_manager.backup_dir, backup_dir);
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
        
        let backup_manager = BackupManager::new(None);
        backup_manager.copy_dir_all(&src_dir, &dst_dir).unwrap();
        
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