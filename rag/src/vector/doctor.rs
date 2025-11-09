//! Similar to `rag::vector::checkhealth`, this provides utilities to aid end-users in
//! troubleshooting issues, providing suggestions where possible.

use std::io::Write;

use crate::vector::{checkhealth::lancedb_health_check, lance::LanceError};

const HELP: &str = "\x1b[32;1m";
const SYMPTOM: &str = "\x1b[33;1m";
const RESET: &str = "\x1b[0m";

/// Print a `cargo`-style "help" message.
///
/// # Arguments:
///
/// * `out`: A writer object, such as a file pointer or `stdout`.
/// * `msg`: The message to write.
fn help(out: &mut impl Write, msg: &str) -> Result<(), LanceError> {
    writeln!(out, "{}help:{} {}", HELP, RESET, msg)?;

    Ok(())
}

/// Print a helpful message showing the symptom observed from the healthcheck.
///
/// # Arguments:
///
/// * `out`: A writer object, such as a file pointer or `stdout`.
/// * `msg`: The message to write.
fn symptom(out: &mut impl Write, msg: &str) -> Result<(), LanceError> {
    writeln!(out, "{}symptom:{} {}", SYMPTOM, RESET, msg)?;

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
/// * `stdout`: A writer object. This does not *have* to be `stdout`, but it is unlikely you would
///   want these messages going to an error stream, considering the messages printed here are meant
///   for end-users.
///
/// # Returns
///
/// Nothing; errors if writing fails or if the health check is in an invalid state for some reason
/// (an invalid state being one that is not expected, and is likely a bug).
pub async fn doctor(embedding_provider: &str, stdout: &mut impl Write) -> Result<(), LanceError> {
    let healthcheck_results = lancedb_health_check(embedding_provider).await?;

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
            // TODO: /index is not an implemented command yet.
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
