//! Backend-agnostic utilities for checking the health of a vector store. This module defines
//! the neutral [`HealthCheckResult`] type and the [`HealthCheckable`] trait; each backend is
//! responsible for implementing the checks that make sense for it.

use std::{fmt, io};

use async_trait::async_trait;

use crate::vector::backends::backend::VectorBackend;

/// ANSI color codes for console output
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const RESET: &str = "\x1b[0m";

/// Errors from running health checks or diagnostics through [`HealthCheckable`] implementations.
/// Individual check failures are captured inside [`HealthCheckResult`] rather than returned, so
/// this error is only used when a fast path fails before the result object exists.
#[derive(Debug, thiserror::Error)]
pub enum HealthCheckError {
    /// An IO error, e.g. writing diagnostic output.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The store is in a state the diagnostics did not expect (likely a bug).
    #[error("The store is in an invalid state: {0}")]
    InvalidState(String),
}

/// A record type that can report how many rows it contains. Backends implement this for their
/// record type so that health-check reporting can count zero-embedding rows without knowing
/// anything about the record's shape.
pub trait RowCount {
    /// The number of rows in this record.
    fn row_count(&self) -> usize;
}

/// Backend-agnostic health check result. Not every check applies to every backend: file-based
/// stores can report storage size, while remote stores cannot, so checks are reported as
/// `Option`s where `None` means "not run" or "not applicable" and `Some(Err(_))` means the check
/// ran and failed.
///
/// The type is generic over the backend's record type `R` (used for the zero-embedding items,
/// which callers may want to inspect or repair) and the backend's error type `E`.
#[derive(Debug)]
#[must_use = "You should probably use this; functions exposing this generally do not have side effects."]
pub struct HealthCheckResult<R, E> {
    /// The store's storage exists and is reachable. For file-based backends this is directory
    /// existence; remote backends report whether the store endpoint/collection could be found.
    pub storage_exists: bool,
    /// Storage size in bytes, where meaningful. `None` when the check hasn't run or is not
    /// applicable to the backend, `Some(Ok(size))` if the size was computed, and
    /// `Some(Err(...))` on failure.
    pub storage_size: Option<Result<u64, io::Error>>,
    /// The store's primary table/collection can be opened. `None` when the check hasn't run,
    /// `Some(Ok(()))` if it is accessible, and `Some(Err(...))` when it could not be opened.
    pub table_accessible: Option<Result<(), E>>,
    /// Number of rows in the store. `None` when the check hasn't run, `Some(Ok(count))` on
    /// success, and `Some(Err(...))` on failure.
    pub num_rows: Option<Result<usize, E>>,
    /// Records with all-zero embeddings. `None` when the check hasn't run, `Some(Ok(records))`
    /// with the complete records containing zero embeddings, and `Some(Err(...))` on failure.
    pub zero_embedding_items: Option<Result<Vec<R>, E>>,
    /// Index information: (index_name, index_type). `None` when the check hasn't run or is not
    /// applicable, `Some(Ok(index_info))` on success, and `Some(Err(...))` on failure.
    pub index_info: Option<Result<Vec<(String, String)>, E>>,
    /// Metadata version drift, as `(stored, live)`, for backends that version writes.
    /// `None` when the check hasn't run or is not applicable, `Some(Ok((stored, live)))` with the
    /// two versions (equal means in sync), and `Some(Err(...))` when the versions could not be
    /// read.
    pub version_drift: Option<Result<(u64, u64), E>>,
}

/// A vector store backend that can report on its own health. Implementors should run every
/// check that is meaningful for them, capture per-check failures in the corresponding
/// [`HealthCheckResult`] field, and always produce a result object.
#[async_trait]
pub trait HealthCheckable: VectorBackend
where
    Self::Record: RowCount,
{
    /// Run health checks on the store and return the collected results. Per-check failures are
    /// captured in the corresponding [`HealthCheckResult`] field rather than returned, so this
    /// is infallible by contract: implementations should always produce a result object.
    ///
    /// # Returns
    ///
    /// A [`HealthCheckResult`] describing each check that ran.
    async fn health_check(&self) -> HealthCheckResult<Self::Record, Self::Error>;
}

/// Format file size in a human-readable format
///
/// # Arguments
///
/// * `bytes` - Size in bytes
///
/// # Returns
///
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

impl<R: RowCount, E: fmt::Display> fmt::Display for HealthCheckResult<R, E> {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Vector Store Health Check Results")?;
        writeln!(f, "=================================")?;

        // Check 1: Storage existence and size
        if self.storage_exists {
            writeln!(f, "{GREEN}✓ Database storage exists{RESET}")?;
            match &self.storage_size {
                Some(Ok(size)) => {
                    let size_str = format_file_size(*size);
                    writeln!(f, "\tSize: {size_str}")?;
                }
                Some(Err(e)) => {
                    writeln!(f, "\t{RED}Error: Failed to calculate size: {e}{RESET}")?;
                }
                None => writeln!(f)?,
            }
        } else {
            writeln!(f, "{RED}✗ Database storage does not exist{RESET}")?;
            writeln!(f, "{YELLOW}  → Subsequent checks will be skipped{RESET}")?;
            return Ok(());
        }
        writeln!(f)?;

        // Check 2: Table accessibility
        match &self.table_accessible {
            Some(Ok(())) => {
                writeln!(f, "{GREEN}✓ Table is accessible{RESET}")?;
            }
            Some(Err(e)) => {
                writeln!(f, "{RED}✗ Table is not accessible: {e}{RESET}")?;
                writeln!(f, "{YELLOW}  → Subsequent checks will be skipped{RESET}")?;
                return Ok(());
            }
            None => {
                writeln!(f, "{YELLOW}⚠ Table accessibility check was skipped{RESET}")?;
                return Ok(());
            }
        }

        // Check 3: Row count
        match &self.num_rows {
            Some(Ok(count)) => {
                if *count == 0 {
                    writeln!(f, "{YELLOW}⚠ Table has no rows{RESET}")?;
                } else {
                    writeln!(f, "\tTable has {count} rows{RESET}")?;
                }
            }
            Some(Err(e)) => {
                writeln!(f, "{RED}✗ Failed to get row count: {e}{RESET}")?;
            }
            None => {
                writeln!(f, "{YELLOW}⚠ Row count check was skipped{RESET}")?;
            }
        }
        writeln!(f)?;

        // Check 4: Zero embeddings
        match &self.zero_embedding_items {
            Some(Ok(zero_records)) => {
                let total_zero_rows: usize = zero_records.iter().map(RowCount::row_count).sum();
                if total_zero_rows == 0 {
                    writeln!(f, "{GREEN}✓ No zero embeddings found{RESET}")?;
                } else {
                    writeln!(
                        f,
                        "{YELLOW}⚠ Found {total_zero_rows} rows with zero embeddings{RESET}. Run `/embed fix` to fix."
                    )?;
                }
            }
            Some(Err(e)) => {
                writeln!(f, "{RED}✗ Failed to check zero embeddings: {e}{RESET}")?;
            }
            None => {
                writeln!(f, "{YELLOW}⚠ Zero embeddings check was skipped{RESET}")?;
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
                                "{YELLOW}⚠ No indices found (may impact query performance){RESET}"
                            )?;
                        } else {
                            writeln!(
                                f,
                                "{GREEN}✓ No indices found. This should not affect performance for your library size.{RESET}"
                            )?;
                        }
                    }
                } else {
                    writeln!(f, "{}✓ Found {} index(es):{}", GREEN, indices.len(), RESET)?;
                    for (name, index_type) in indices {
                        writeln!(f, "\t- {name} ({index_type})")?;
                    }
                }
            }
            Some(Err(e)) => {
                writeln!(f, "{RED}✗ Failed to get index information: {e}{RESET}")?;
            }
            None => {
                writeln!(f, "{YELLOW}⚠ Index information check was skipped{RESET}")?;
            }
        }
        writeln!(f)?;

        // Check 7: Metadata version sync
        match &self.version_drift {
            Some(Ok((stored, live))) => {
                if stored == live {
                    writeln!(f, "{GREEN}✓ Metadata version is in sync (v{live}){RESET}")?;
                } else {
                    writeln!(
                        f,
                        "{YELLOW}⚠ Metadata version drift: metadata last synced at v{stored}, but the data table is at v{live}{RESET}"
                    )?;
                    writeln!(
                        f,
                        "{YELLOW}  → A write may have bypassed metadata sync, or the database was modified out of band{RESET}"
                    )?;
                }
            }
            Some(Err(e)) => {
                writeln!(f, "{RED}✗ Failed to check metadata version: {e}{RESET}")?;
            }
            None => {
                writeln!(f, "{YELLOW}⚠ Metadata version check was skipped{RESET}")?;
            }
        }

        Ok(())
    }
}
