//! Similar to `rag::vector::checkhealth`, this provides utilities to aid end-users in
//! troubleshooting issues, providing suggestions where possible.

use std::io::Write;

use crate::vector::checkhealth::{HealthCheckError, HealthCheckable, RowCount};

const HELP: &str = "\x1b[32;1m";
const SYMPTOM: &str = "\x1b[33;1m";
const RESET: &str = "\x1b[0m";

/// Print a `cargo`-style "help" message.
///
/// # Arguments:
///
/// * `out`: A writer object, such as a file pointer or `stdout`.
/// * `msg`: The message to write.
///
/// # Errors
///
/// Returns an error if writing to the output stream fails.
fn help(out: &mut impl Write, msg: &str) -> Result<(), HealthCheckError> {
    writeln!(out, "{HELP}help:{RESET} {msg}")?;

    Ok(())
}

/// Print a helpful message showing the symptom observed from the healthcheck.
///
/// # Arguments:
///
/// * `out`: A writer object, such as a file pointer or `stdout`.
/// * `msg`: The message to write.
///
/// # Errors
///
/// Returns an error if writing to the output stream fails.
fn symptom(out: &mut impl Write, msg: &str) -> Result<(), HealthCheckError> {
    writeln!(out, "{SYMPTOM}symptom:{RESET} {msg}")?;

    Ok(())
}

/// Run health checks on a vector store, and provide helpful suggestions to the user to fix
/// errors they may have gotten from a health check. Note that this does not actually run those
/// fixes--this is so the user of this function has autonomy over that (e.g., the user may want to
/// first print some message or ask for confirmation before proceeding). There are a few
/// assumptions made here, mainly that the end-user understands what "/embed" and "/index" mean.
/// These parts of the messages may later change, but for now, when this crate is somewhat tailored
/// to `zqa`, this is a very low priority.
///
/// # Arguments:
///
/// * `backend`: The vector store backend to diagnose.
/// * `stdout`: A writer object. This does not *have* to be `stdout`, but it is unlikely you would
///   want these messages going to an error stream, considering the messages printed here are meant
///   for end-users.
///
/// # Returns
///
/// Nothing; errors if writing fails or if the health check is in an invalid state for some reason
/// (an invalid state being one that is not expected, and is likely a bug).
///
/// # Errors
///
/// Returns an error if writing to the output stream fails or if the health check is in an
/// invalid state.
pub async fn doctor<T: HealthCheckable>(
    backend: &T,
    stdout: &mut impl Write,
) -> Result<(), HealthCheckError>
where
    T::Record: RowCount,
{
    let healthcheck_results = backend.health_check().await;

    if !healthcheck_results.storage_exists {
        symptom(stdout, "database storage does not exist.")?;
        help(stdout, "maybe you are not in the right directory?")?;

        return Ok(());
    }

    let tbl_accessible =
        healthcheck_results
            .table_accessible
            .ok_or(HealthCheckError::InvalidState(
            "Invalid healthcheck result: if storage exists, `table_accessible` cannot be `None`."
                .into(),
        ))?;

    if tbl_accessible.is_err() {
        // Usually, there isn't much we can do here
        symptom(stdout, "the database table is not accessible.")?;
        help(
            stdout,
            "check that the `data/` directory actually contains the DB and is not corrupted.",
        )?;

        return Ok(());
    }

    let Some(row_count) = healthcheck_results.num_rows else {
        symptom(stdout, "there are no rows in the database.")?;
        return Ok(());
    };

    if row_count.is_err() {
        symptom(stdout, "row count cannot be obtained.")?;
        help(
            stdout,
            "this is usually transient; if this persists, your database may be corrupted.",
        )?;

        writeln!(stdout)?;
    }

    // `None` means the backend doesn't support it or the check has not run
    if let Some(zero_embedding_items) = healthcheck_results.zero_embedding_items
        && let Ok(zero_records) = zero_embedding_items
        && !zero_records.is_empty()
    {
        symptom(stdout, "some items have zero embedding vectors.")?;
        help(stdout, "run `/embed fix` to fix this.")?;

        writeln!(stdout)?;
    }

    let Some(index_info) = healthcheck_results.index_info else {
        writeln!(stdout, "Analysis completed.")?;
        return Ok(());
    };

    if let Ok(indices) = index_info {
        if indices.is_empty()
            && let Ok(row_count) = row_count
            && row_count > 10000
        {
            symptom(stdout, "there were no indices with > 10k rows.")?;
            help(stdout, "run /index to create indices.")?;
        }
    } else {
        symptom(stdout, "index information could not be obtained")?;
        help(
            stdout,
            "this is usually transient; if this persists, your database may be corrupted.",
        )?;
    }
    writeln!(stdout, "Analysis completed.")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::sync::Arc;

    use arrow_array::{RecordBatch, StringArray};
    use dotenv::dotenv;
    use zqa_macros::test_ok;

    use super::doctor;
    use crate::config::VoyageAIConfig;
    use crate::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use crate::embedding::common::EmbeddingProviderConfig;
    use crate::vector::backends::backend::VectorBackend;
    use crate::vector::backends::lance::LanceBackend;

    /// Builds a [`LanceBackend`] pointing at an isolated temp directory, along with the schema it
    /// was constructed with. The returned [`tempfile::TempDir`] guard must be kept alive for the
    /// test's duration. Because each backend points at its own URI, these tests need no
    /// `#[serial]`.
    fn temp_backend() -> (tempfile::TempDir, LanceBackend, Arc<arrow_schema::Schema>) {
        dotenv().ok();

        let dir = tempfile::tempdir().unwrap();
        let uri = dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        ]));

        let backend = LanceBackend::new(
            EmbeddingProviderConfig::VoyageAI(VoyageAIConfig {
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                api_key: env::var("VOYAGE_AI_API_KEY").unwrap_or_default(),
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            }),
            Arc::clone(&schema),
            "pdf_text".into(),
        )
        .with_uri(&uri);

        (dir, backend, schema)
    }

    /// When the database does not exist, `doctor` should report that symptom and short-circuit
    /// without running the remaining checks (and without any network access).
    #[tokio::test]
    async fn test_doctor_reports_missing_database() {
        let (_dir, backend, _) = temp_backend();

        let mut out: Vec<u8> = Vec::new();
        let result = doctor(&backend, &mut out).await;
        test_ok!(result);

        let output = String::from_utf8(out).unwrap();
        assert!(output.contains("database storage does not exist"));
        assert!(output.contains("help:"));
        // The early return means the later checks (and their success marker) are never reached.
        assert!(!output.contains("Analysis completed."));
    }

    /// Builds a small, healthy database in an isolated temp directory and checks that `doctor` runs
    /// every check through to completion without flagging a missing database.
    #[tokio::test]
    async fn test_doctor_healthy_database_completes() {
        let (_dir, backend, schema) = temp_backend();

        let record_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Hello world", "Test document"])),
                Arc::new(StringArray::from(vec!["doc1.pdf", "doc2.pdf"])),
            ],
        )
        .unwrap();

        backend
            .insert_items(vec![record_batch], None)
            .await
            .unwrap();

        let mut out: Vec<u8> = Vec::new();
        let result = doctor(&backend, &mut out).await;
        test_ok!(result);

        let output = String::from_utf8(out).unwrap();
        assert!(output.contains("Analysis completed."));
        assert!(!output.contains("database storage does not exist"));
    }
}
