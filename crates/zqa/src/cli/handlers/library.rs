//! Command handlers for library-related tasks

use std::io::Write;

use zqa_rag::vector::doctor::doctor as rag_doctor;

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::store::common::ZoteroStore;
use crate::utils::arrow::{ArrowError, library_to_arrow};
use crate::utils::library::{get_new_library_items, parse_library, parse_library_metadata};
use crate::utils::terminal::read_line;

/// Print table statistics for the current LanceDB database.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the command output was written successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if writing to an output stream fails.
pub(crate) async fn handle_stats_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    match ctx.store.get_metadata().await {
        Ok(stats) => writeln!(&mut ctx.out, "{stats}")?,
        Err(e) => writeln!(&mut ctx.err, "Could not get database statistics: {e}")?,
    }

    Ok(())
}

/// Process the user's Zotero library into the LanceDB-backed retrieval store.
///
/// This parses the library, extracts text from each file, generates embeddings, and stores the
/// records in LanceDB. Items that could not be embedded are not stored, so the next `/process`
/// picks them up again.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if processing completed or the user declined to continue.
///
/// # Errors
///
/// Returns a [`CLIError`] if input/output fails, configuration is invalid,
/// or parsing / insertion setup fails.
pub(crate) async fn handle_process_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    const WARNING_THRESHOLD: usize = 100;

    let library_path = ctx.path_options.library_path.as_deref();
    let item_metadata = if ctx.store.exists().await {
        get_new_library_items(&ctx.store, library_path).await
    } else {
        parse_library_metadata(library_path, None, None)
    };

    if let Err(parse_err) = item_metadata {
        writeln!(
            &mut ctx.err,
            "Could not parse library metadata: {parse_err}"
        )?;
        return Ok(());
    }

    let item_metadata = item_metadata.unwrap();
    let metadata_length = item_metadata.len();
    if metadata_length >= WARNING_THRESHOLD {
        writeln!(
            &mut ctx.out,
            "Your library has {metadata_length} new items. Parsing may take a while. Continue?"
        )?;
        write!(&mut ctx.out, "(/process) >>> ")?;
        ctx.out.flush()?;

        let option = read_line(&mut ctx.input);
        let option = option.trim().to_lowercase();
        if ["n", "no", "false", "0"].contains(&option.as_str()) {
            return Ok(());
        }
    }

    let items = parse_library(&ctx.store, library_path, None, None)
        .await
        .map_err(ArrowError::from)?;
    log::info!("Finished parsing library items.");

    let result = match library_to_arrow(&items, &ctx.store.get_embedding_config()) {
        Ok(batch) => {
            let num_failed = items.len() - batch.num_rows();
            ctx.store
                .upsert_batches(vec![batch])
                .await
                .map(|()| num_failed)
        }
        Err(e) => Err(e.into()),
    };

    match result {
        Ok(0) => writeln!(&mut ctx.out, "Successfully parsed library!")?,
        Ok(num_failed) => writeln!(
            &mut ctx.err,
            "{num_failed} items could not be embedded and were not saved. Run '/process' again to retry them."
        )?,
        Err(e) => writeln!(&mut ctx.err, "Parsing library failed: {e}")?,
    }

    Ok(())
}

/// Remove duplicate rows from the LanceDB table.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if deduplication completed and the result was written successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if configuration is invalid, deduplication fails,
/// or writing output fails.
pub(crate) async fn handle_dedup_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    let result = ctx.store.dedup_by_title().await;

    match result {
        Ok(count) => {
            writeln!(ctx.out, "Deduped {count} rows")?;
        }
        Err(e) => {
            // Avoid terminating CLI
            writeln!(&mut ctx.err, "Deduplication failed: {e}")?;
        }
    }

    Ok(())
}

/// Create or update the LanceDB indices used by retrieval.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if index creation completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if index creation fails or writing output fails.
pub(crate) async fn handle_index_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(
        &mut ctx.out,
        "Updating indices. This may take a while depending on how many items need to be added."
    )?;

    if let Err(e) = ctx.store.create_or_update_indices().await {
        writeln!(&mut ctx.err, "Failed to update indexes: {e}")?;
    }

    writeln!(
        &mut ctx.out,
        "Done! You should verify the indices exist with /checkhealth."
    )?;

    Ok(())
}

/// Run health checks against the LanceDB database and print the results.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the health-check output was written successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if writing output fails.
pub(crate) async fn handle_checkhealth_cmd<O: Write, E: Write>(
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    if let Err(e) = writeln!(ctx.out, "{}", ctx.store.health_check().await) {
        let _ = writeln!(ctx.err, "{e}");
    }

    Ok(())
}

/// Run database diagnostics and explain how to fix any issues found.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if diagnostics completed successfully.
///
/// # Errors
///
/// * [`CLIError`] if writing output fails.
pub(crate) async fn handle_doctor_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    if let Err(e) = rag_doctor(ctx.store.backend(), &mut ctx.out).await {
        writeln!(ctx.err, "{e}")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use zqa_macros::{test_contains, test_ok};
    use zqa_macros_proc::retry;

    use super::{handle_checkhealth_cmd, handle_process_cmd, handle_stats_cmd};
    use crate::common::test_support::TestPaths;

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_process_and_stats() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut ctx = paths.context(vec![]);

        let result = handle_process_cmd(&mut ctx).await;
        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.clone().into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let stats = handle_stats_cmd(&mut ctx).await;
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_ok!(stats);
        test_contains!(output, "LanceDB metadata:");
        test_contains!(output, "Number of rows: 8");
    }

    #[tokio::test]
    async fn test_checkhealth_no_database() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut ctx = paths.context(vec![]);
        handle_checkhealth_cmd(&mut ctx).await.unwrap();
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();

        assert!(output.contains("storage does not exist"));
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_checkhealth_with_database() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut setup_ctx = paths.context(vec![]);
        let result = handle_process_cmd(&mut setup_ctx).await;
        test_ok!(result);

        let mut ctx = paths.context(vec![]);
        handle_checkhealth_cmd(&mut ctx).await.unwrap();
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();

        test_contains!(output, "Vector Store Health Check Results");
        test_contains!(output, "storage exists");
        test_contains!(output, "Table is accessible");
        test_contains!(output, "Table has");
    }
}
