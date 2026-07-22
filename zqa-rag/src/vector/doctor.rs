//! Similar to `rag::vector::checkhealth`, this provides utilities to aid end-users in
//! troubleshooting issues, providing suggestions where possible.

use std::io::Write;

use crate::capabilities::EmbeddingProvider;
use crate::vector::backends::lance::LanceError;
use crate::vector::checkhealth::lancedb_health_check;

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
fn help(out: &mut impl Write, msg: &str) -> Result<(), LanceError> {
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
fn symptom(out: &mut impl Write, msg: &str) -> Result<(), LanceError> {
    writeln!(out, "{SYMPTOM}symptom:{RESET} {msg}")?;

    Ok(())
}

/// Run health checks on the LanceDB database, and provide helpful suggestions to the user to fix
/// errors they may have gotten from a health check. Note that this does not actually run those
/// fixes--this is so the user of this function has autonomy over that (e.g., the user may want to
/// first print some message or ask for confirmation before proceeding). There are a few
/// assumptions made here, mainly that the end-user understands what "/embed" and "/index" mean.
/// These parts of the messages may later change, but for now, when this crate is somewhat tailored
/// to `zqa`, this is a very low priority.
///
/// # Arguments:
///
/// * `embeddings_provider`: The embedding provider. Must be one of `EmbeddingProviders`.
/// * `db_uri`: The URI of the database to diagnose. Pass a store's own URI so diagnostics target
///   the same database the store uses rather than the process-global `LANCEDB_URI`.
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
/// Returns an error if writing to the output stream fails or if the health check is in an invalid state.
pub async fn doctor(
    embedding_provider: EmbeddingProvider,
    db_uri: &str,
    stdout: &mut impl Write,
) -> Result<(), LanceError> {
    let healthcheck_results = lancedb_health_check(embedding_provider, db_uri).await?;

    if !healthcheck_results.directory_exists {
        symptom(stdout, "database directory does not exist.")?;
        help(stdout, "maybe you are not in the right directory?")?;

        return Ok(());
    }

    let tbl_accessible =
        healthcheck_results
            .table_accessible
            .ok_or(LanceError::InvalidStateError(
            "Invalid healthcheck result: if directory exists, `table_accessible` cannot be `None`."
                .into(),
        ))?;

    if tbl_accessible.is_err() {
        // Usually, there isn't much we can do here
        symptom(stdout, "the LanceDB table is not accessible.")?;
        help(
            stdout,
            "check that the `data/` directory actually contains the DB and is not corrupted.",
        )?;

        return Ok(());
    }

    let row_count = healthcheck_results
        .num_rows
        .ok_or(LanceError::InvalidStateError(
            "Invalid healthcheck result: if the table is accessible, `num_rows` cannot be `None`."
                .into(),
        ))?;

    if row_count.is_err() {
        symptom(stdout, "row count cannot be obtained.")?;
        help(
            stdout,
            "this is usually transient; if this persists, your database may be corrupted.",
        )?;

        writeln!(stdout)?;
    }

    let zero_embedding_items = healthcheck_results
        .zero_embedding_items
        .ok_or(LanceError::InvalidStateError(
            "Invalid healthcheck result: if the table is accessible, `zero_embedding_items` cannot be `None`."
                .into(),
        ))?;

    if let Ok(zero_batches) = zero_embedding_items
        && !zero_batches.is_empty()
    {
        symptom(stdout, "some items have zero embedding vectors.")?;
        help(stdout, "run `/embed fix` to fix this.")?;

        writeln!(stdout)?;
    }

    let index_info = healthcheck_results
        .index_info
        .ok_or(LanceError::InvalidStateError(
        "Invalid healthcheck result: if the table is accessible, `index_info` cannot be `None`."
            .into(),
    ))?;

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
    use crate::capabilities::EmbeddingProvider;
    use crate::config::VoyageAIConfig;
    use crate::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use crate::embedding::common::EmbeddingProviderConfig;
    use crate::vector::backends::backend::VectorBackend;
    use crate::vector::backends::lance::LanceBackend;

    /// Returns a URI inside a fresh temp directory whose `lancedb-table` subdirectory does not yet
    /// exist. Pointing callers at an explicit URI (rather than the global `LANCEDB_URI`) keeps these
    /// tests isolated, so they need no `#[serial]`. The returned [`tempfile::TempDir`] guard must be
    /// kept alive for the test's duration.
    fn temp_db_uri() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();
        (dir, uri)
    }

    /// When the database directory does not exist, `doctor` should report that symptom and
    /// short-circuit without running the remaining checks (and without any network access).
    #[tokio::test]
    async fn test_doctor_reports_missing_database() {
        let (_dir, db_uri) = temp_db_uri();

        let mut out: Vec<u8> = Vec::new();
        let result = doctor(EmbeddingProvider::VoyageAI, &db_uri, &mut out).await;
        test_ok!(result);

        let output = String::from_utf8(out).unwrap();
        assert!(output.contains("database directory does not exist"));
        assert!(output.contains("help:"));
        // The early return means the later checks (and their success marker) are never reached.
        assert!(!output.contains("Analysis completed."));
    }

    /// Builds a small, healthy database in an isolated temp directory and checks that `doctor` runs
    /// every check through to completion without flagging a missing database.
    #[tokio::test]
    async fn test_doctor_healthy_database_completes() {
        dotenv().ok();

        let (_dir, db_uri) = temp_db_uri();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        ]);
        let record_batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(StringArray::from(vec!["Hello world", "Test document"])),
                Arc::new(StringArray::from(vec!["doc1.pdf", "doc2.pdf"])),
            ],
        )
        .unwrap();

        let backend = LanceBackend::new(
            EmbeddingProviderConfig::VoyageAI(VoyageAIConfig {
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                api_key: env::var("VOYAGE_AI_API_KEY").unwrap_or_default(),
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            }),
            Arc::new(schema),
            "pdf_text".into(),
        )
        .with_uri(&db_uri);
        backend.insert_items(vec![record_batch], None).await.unwrap();

        let mut out: Vec<u8> = Vec::new();
        let result = doctor(EmbeddingProvider::VoyageAI, &db_uri, &mut out).await;
        test_ok!(result);

        let output = String::from_utf8(out).unwrap();
        assert!(output.contains("Analysis completed."));
        assert!(!output.contains("database directory does not exist"));
    }
}
