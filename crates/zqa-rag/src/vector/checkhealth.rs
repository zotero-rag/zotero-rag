//! Backend-agnostic utilities for checking the health of a vector store. This module defines
//! the neutral [`HealthCheckResult`] type and the [`HealthCheckable`] trait; each backend is
//! responsible for implementing the checks that make sense for it.

use std::{fmt, io};

use async_trait::async_trait;

use crate::vector::backends::backend::VectorBackend;

/// Errors from running health checks or diagnostics through [`HealthCheckable`] implementations.
/// Individual check failures are captured inside [`HealthCheckResult`] rather than returned, so
/// this error is only used when a fast path fails before the result object exists.
#[derive(Debug, thiserror::Error)]
pub enum HealthCheckError {
    /// The store is in a state the diagnostics did not expect (likely a bug).
    #[error("The store is in an invalid state: {0}")]
    InvalidState(String),
}

/// Backend-agnostic health check result. Not every check applies to every backend: file-based
/// stores can report storage size, while remote stores cannot, so checks are reported as
/// `Option`s where `None` means "not run" or "not applicable" and `Some(Err(_))` means the check
/// ran and failed.
///
/// The type is generic over the backend's error type `E`.
#[derive(Debug)]
#[must_use = "You should probably use this; functions exposing this generally do not have side effects."]
pub struct HealthCheckResult<E> {
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
    /// Index information: (index_name, index_type). `None` when the check hasn't run or is not
    /// applicable, `Some(Ok(index_info))` on success, and `Some(Err(...))` on failure.
    pub index_info: Option<Result<Vec<(String, String)>, E>>,
    /// Metadata version drift, as `(stored, live)`, for backends that version writes.
    /// `None` when the check hasn't run or is not applicable, `Some(Ok((stored, live)))` with the
    /// two versions (equal means in sync), and `Some(Err(...))` when the versions could not be
    /// read.
    pub version_drift: Option<Result<(u64, u64), E>>,
}

/// How a line of a health check report went, so frontends can color or mark it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HealthCheckStatus {
    /// The check passed.
    Pass,
    /// A detail about a check, such as the storage size. It carries no verdict.
    Info,
    /// The check passed with a caveat, or was skipped.
    Warn,
    /// The check failed.
    Fail,
}

/// One line of a health check report, as built by [`HealthCheckResult::report`].
#[derive(Clone, Debug)]
pub struct HealthCheckReportLine {
    /// How the check went.
    pub status: HealthCheckStatus,
    /// The line to show, without a trailing newline. Pass, warn and fail lines start with a
    /// marker such as `✓`, and detail lines are indented with two spaces.
    pub text: String,
}

/// Build a report line.
///
/// # Arguments
///
/// * `status` - How the check went.
/// * `text` - The line to show, without a trailing newline.
///
/// # Returns
///
/// The report line.
fn line(status: HealthCheckStatus, text: impl Into<String>) -> HealthCheckReportLine {
    HealthCheckReportLine {
        status,
        text: text.into(),
    }
}

impl<E: fmt::Display> HealthCheckResult<E> {
    /// Describe the checks as report lines, in the order they ran.
    ///
    /// The report stops after a check that later checks depend on: when storage does not exist,
    /// or when the table is not accessible or that check was skipped. A check that did not run
    /// shows as a [`HealthCheckStatus::Warn`] line. The [`fmt::Display`] impl prints these lines
    /// under a title, without color; frontends that can style text should use this instead.
    ///
    /// # Returns
    ///
    /// One line per check result or detail. The lines have no trailing newlines.
    #[must_use]
    pub fn report(&self) -> Vec<HealthCheckReportLine> {
        use HealthCheckStatus::{Fail, Info, Pass, Warn};

        if !self.storage_exists {
            return vec![
                line(Fail, "✗ Database storage does not exist"),
                line(Warn, "  → Subsequent checks will be skipped"),
            ];
        }

        let mut lines = vec![line(Pass, "✓ Database storage exists")];

        match &self.storage_size {
            Some(Ok(size)) => {
                lines.push(line(Info, format!("  Size: {}", format_file_size(*size))));
            }
            Some(Err(e)) => lines.push(line(
                Fail,
                format!("  Error: Failed to calculate size: {e}"),
            )),
            None => {}
        }

        match &self.table_accessible {
            Some(Ok(())) => lines.push(line(Pass, "✓ Table is accessible")),
            Some(Err(e)) => {
                lines.push(line(Fail, format!("✗ Table is not accessible: {e}")));
                lines.push(line(Warn, "  → Subsequent checks will be skipped"));
                return lines;
            }
            None => {
                lines.push(line(Warn, "⚠ Table accessibility check was skipped"));
                return lines;
            }
        }

        match &self.num_rows {
            Some(Ok(0)) => lines.push(line(Warn, "⚠ Table has no rows")),
            Some(Ok(count)) => lines.push(line(Info, format!("  Table has {count} rows"))),
            Some(Err(e)) => lines.push(line(Fail, format!("✗ Failed to get row count: {e}"))),
            None => lines.push(line(Warn, "⚠ Row count check was skipped")),
        }

        match &self.index_info {
            Some(Ok(indices)) if indices.is_empty() => match self.num_rows {
                Some(Ok(count)) if count > 10000 => {
                    lines.push(line(
                        Warn,
                        "⚠ No indices found (may impact query performance)",
                    ));
                }
                Some(Ok(_)) => lines.push(line(
                    Pass,
                    "✓ No indices found. This should not affect performance for your library size.",
                )),
                _ => {}
            },
            Some(Ok(indices)) => {
                lines.push(line(Pass, format!("✓ Found {} index(es):", indices.len())));
                lines.extend(
                    indices
                        .iter()
                        .map(|(name, index_type)| line(Info, format!("  - {name} ({index_type})"))),
                );
            }
            Some(Err(e)) => lines.push(line(
                Fail,
                format!("✗ Failed to get index information: {e}"),
            )),
            None => lines.push(line(Warn, "⚠ Index information check was skipped")),
        }

        match &self.version_drift {
            Some(Ok((stored, live))) if stored == live => {
                lines.push(line(
                    Pass,
                    format!("✓ Metadata version is in sync (v{live})"),
                ));
            }
            Some(Ok((stored, live))) => {
                lines.push(line(
                    Warn,
                    format!(
                        "⚠ Metadata version drift: metadata last synced at v{stored}, but the data table is at v{live}"
                    ),
                ));
                lines.push(line(
                    Warn,
                    "  → A write may have bypassed metadata sync, or the database was modified out of band",
                ));
            }
            Some(Err(e)) => lines.push(line(
                Fail,
                format!("✗ Failed to check metadata version: {e}"),
            )),
            None => lines.push(line(Warn, "⚠ Metadata version check was skipped")),
        }

        lines
    }
}

/// A vector store backend that can report on its own health. Implementors should run every
/// check that is meaningful for them, capture per-check failures in the corresponding
/// [`HealthCheckResult`] field, and always produce a result object.
#[async_trait]
pub trait HealthCheckable: VectorBackend {
    /// Run health checks on the store and return the collected results. Per-check failures are
    /// captured in the corresponding [`HealthCheckResult`] field rather than returned, so this
    /// is infallible by contract: implementations should always produce a result object.
    ///
    /// # Returns
    ///
    /// A [`HealthCheckResult`] describing each check that ran.
    async fn health_check(&self) -> HealthCheckResult<Self::Error>;
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

impl<E: fmt::Display> fmt::Display for HealthCheckResult<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Vector Store Health Check Results")?;
        writeln!(f, "=================================")?;

        for line in self.report() {
            writeln!(f, "{}", line.text)?;
        }

        Ok(())
    }
}
