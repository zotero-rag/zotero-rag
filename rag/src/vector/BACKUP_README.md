# Database Backup System

This module provides automatic backup and restore functionality for LanceDB operations in the `rag` crate.

## Features

- **Two backup strategies:**
  - **Directory Copy**: Creates a complete copy of the database directory
  - **LanceDB Versions**: Uses LanceDB's built-in versioning for rollback
  
- **Automatic rollback**: Failed operations automatically restore from backup
- **Cleanup**: Successful operations automatically clean up backup data
- **Error handling**: Comprehensive error types and logging

## Usage

### Basic Usage

```rust
use crate::vector::backup::BackupStrategy;
use crate::vector::lance::{insert_records_with_backup, delete_rows_with_backup};

// Insert records with directory backup
let connection = insert_records_with_backup(
    data,
    merge_on,
    embedding_params,
    BackupStrategy::DirectoryCopy,
).await?;

// Delete rows with version backup
delete_rows_with_backup(
    rows,
    key,
    embedding_name,
    BackupStrategy::LanceVersion,
).await?;
```

### Direct Backup Manager Usage

```rust
use crate::vector::backup::{BackupManager, BackupStrategy};

let mut backup_manager = BackupManager::new(None);

// Create backup
let backup_id = backup_manager.create_backup(BackupStrategy::DirectoryCopy).await?;

// Perform your operations...
let result = perform_database_operation().await;

match result {
    Ok(_) => {
        // Success: cleanup backup
        backup_manager.cleanup_backup(&backup_id).await?;
    },
    Err(_) => {
        // Failure: restore backup
        backup_manager.restore_backup(&backup_id).await?;
    }
}
```

## Backup Strategies

### Directory Copy (`BackupStrategy::DirectoryCopy`)

- **How it works**: Creates a complete copy of the `data/lancedb-table` directory
- **Storage**: Backups stored in `data/backups/backup_<timestamp>/`
- **Pros**: 
  - Complete data protection
  - Works with any database state
  - Independent of LanceDB version support
- **Cons**: 
  - Higher disk usage
  - Slower for large databases
- **Best for**: Critical operations, large structural changes

### LanceDB Versions (`BackupStrategy::LanceVersion`)

- **How it works**: Records current table version, restores to that version on failure
- **Storage**: Uses LanceDB's internal versioning (no extra disk usage)
- **Pros**: 
  - Fast and efficient
  - No additional disk space required
  - Leverages database's built-in capabilities
- **Cons**: 
  - Depends on LanceDB version support
  - May not protect against directory-level corruption
- **Best for**: Routine operations, small changes

## Error Handling

The backup system provides comprehensive error handling:

```rust
use crate::vector::backup::BackupError;

match insert_records_with_backup(data, merge_on, embedding_params, strategy).await {
    Ok(connection) => {
        println!("Operation successful, backup cleaned up");
    },
    Err(BackupError::IoError(e)) => {
        eprintln!("File system error during backup: {}", e);
    },
    Err(BackupError::LanceError(e)) => {
        eprintln!("Database error: {}", e);
    },
    Err(BackupError::ValidationError(msg)) => {
        eprintln!("Validation failed: {}", msg);
    },
    Err(BackupError::BackupNotFound(id)) => {
        eprintln!("Backup {} not found", id);
    },
}
```

## Configuration

### Custom Backup Directory

```rust
let backup_manager = BackupManager::new(Some(PathBuf::from("/custom/backup/path")));
```

### Default Configuration

- **Backup directory**: `data/backups/`
- **Backup naming**: `backup_<unix_timestamp>`
- **Database location**: `data/lancedb-table` (from `DB_URI` constant)

## Implementation Notes

- All backup operations are logged using the `log` crate
- Backup cleanup failures are logged as warnings but don't fail the operation
- Restore failures are logged as errors and propagate the error
- The system ensures backups are created before operations begin
- Failed operations automatically trigger restore attempts

## Testing

The module includes comprehensive tests covering:
- Backup manager creation and configuration
- Directory copying functionality  
- Error handling for various failure scenarios
- Integration with existing database operations

Run tests with:
```bash
cargo test backup
```