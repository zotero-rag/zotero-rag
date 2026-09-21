//! Command handlers for library-related tasks

use zqa_rag::vector::doctor::doctor as rag_doctor;

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::io::EngineEvent;
use crate::store::common::ZoteroStore;
use crate::utils::arrow::{ArrowError, library_to_arrow};
use crate::utils::library::{get_new_library_items, parse_library, parse_library_metadata};
use crate::utils::terminal::read_line;

/// Emit table statistics for the current LanceDB database.
///
/// # Arguments
///
/// * `ctx` - The application context, including the store and event sender.
///
/// # Returns
///
/// `Ok(())` if event publication completed without error.
///
/// # Errors
///
/// * `CLIError::ChannelError` - If the event receiver is closed.
pub(crate) async fn handle_stats_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    match ctx.store.get_metadata().await {
        Ok(stats) => {
            ctx.emit(EngineEvent::Text {
                message: format!("{stats}\n"),
            })
            .await?;
        }
        Err(e) => {
            ctx.emit(EngineEvent::Error {
                message: format!("Could not get database statistics: {e}\n"),
            })
            .await?;
        }
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
/// * `ctx` - The application context, including the store, configuration, and event sender.
///
/// # Returns
///
/// `Ok(())` if processing completed or the user declined to continue.
///
/// # Errors
///
/// Returns a [`CLIError`] if configuration is invalid, parsing / insertion setup fails,
/// or the event receiver is closed.
pub(crate) async fn handle_process_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    const WARNING_THRESHOLD: usize = 100;

    let library_path = ctx.path_options.library_path.as_deref();
    let item_metadata = if ctx.store.exists().await {
        get_new_library_items(&ctx.store, library_path).await
    } else {
        parse_library_metadata(library_path, None, None)
    };

    if let Err(parse_err) = item_metadata {
        ctx.emit(EngineEvent::Error {
            message: format!("Could not parse library metadata: {parse_err}\n"),
        })
        .await?;
        return Ok(());
    }

    let item_metadata = item_metadata.unwrap();
    let metadata_length = item_metadata.len();
    if metadata_length >= WARNING_THRESHOLD {
        ctx.emit(EngineEvent::Text {
            message: format!(
                "Your library has {metadata_length} new items. Parsing may take a while. Continue?\n"
            ),
        })
        .await?;
        ctx.emit(EngineEvent::Text {
            message: "(/process) >>> ".into(),
        })
        .await?;

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
        Ok(0) => {
            ctx.emit(EngineEvent::StatusUpdate {
                message: "Successfully parsed library!\n".into(),
            })
            .await?;
        }
        Ok(num_failed) => {
            ctx.emit(EngineEvent::Warning {
                message: format!(
                    "{num_failed} items could not be embedded and were not saved. Run '/process' again to retry them.\n"
                ),
            })
            .await?;
        }
        Err(e) => {
            ctx.emit(EngineEvent::Error {
                message: format!("Parsing library failed: {e}\n"),
            })
            .await?;
        }
    }

    Ok(())
}

/// Remove duplicate rows from the LanceDB table.
///
/// # Arguments
///
/// * `ctx` - The application context, including the store and event sender.
///
/// # Returns
///
/// `Ok(())` if the deduplication result or error was passed to the event sender.
///
/// # Errors
///
/// * `CLIError::ChannelError` - If the event receiver is closed.
pub(crate) async fn handle_dedup_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    let result = ctx.store.dedup_by_title().await;

    match result {
        Ok(count) => {
            ctx.emit(EngineEvent::Text {
                message: format!("Deduped {count} rows\n"),
            })
            .await?;
        }
        Err(e) => {
            // Avoid terminating CLI
            ctx.emit(EngineEvent::Error {
                message: format!("Deduplication failed: {e}\n"),
            })
            .await?;
        }
    }

    Ok(())
}

/// Create or update the LanceDB indices used by retrieval.
///
/// # Arguments
///
/// * `ctx` - The application context, including the store and event sender.
///
/// # Returns
///
/// `Ok(())` if the index update and event publication completed without a channel error.
/// Index-creation failures are reported as error events.
///
/// # Errors
///
/// * `CLIError::ChannelError` - If the event receiver is closed.
pub(crate) async fn handle_index_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    ctx.emit(EngineEvent::StatusUpdate {
        message: "Updating indices. This may take a while depending on how many items need to be added.\n"
            .into(),
    })
    .await?;

    if let Err(e) = ctx.store.create_or_update_indices().await {
        ctx.emit(EngineEvent::Error {
            message: format!("Failed to update indexes: {e}\n"),
        })
        .await?;
    }

    ctx.emit(EngineEvent::StatusUpdate {
        message: "Done! You should verify the indices exist with /checkhealth.\n".into(),
    })
    .await?;

    Ok(())
}

/// Run health checks against the LanceDB database and emit the results.
///
/// # Arguments
///
/// * `ctx` - The application context, including the store and event sender.
///
/// # Returns
///
/// `Ok(())` if health-check event publication completed without error.
///
/// # Errors
///
/// * `CLIError::ChannelError` - If the event receiver is closed.
pub(crate) async fn handle_checkhealth_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    ctx.emit(EngineEvent::Text {
        message: format!("{}\n", ctx.store.health_check().await),
    })
    .await?;

    Ok(())
}

/// Run database diagnostics and explain how to fix any issues found.
///
/// # Arguments
///
/// * `ctx` - The application context, including the store, configuration, and event sender.
///
/// # Returns
///
/// `Ok(())` if diagnostics completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if the event receiver is closed.
/// Diagnostic failures are reported as error events.
pub(crate) async fn handle_doctor_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    let mut output = Vec::new();
    let result = rag_doctor(ctx.store.backend(), &mut output).await;

    if !output.is_empty() {
        ctx.emit(EngineEvent::Text {
            message: String::from_utf8_lossy(&output).into_owned(),
        })
        .await?;
    }

    if let Err(e) = result {
        ctx.emit(EngineEvent::Error {
            message: format!("{e}\n"),
        })
        .await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use zqa_macros::test_ok;
    use zqa_macros_proc::retry;

    use super::{handle_checkhealth_cmd, handle_process_cmd, handle_stats_cmd};
    use crate::common::test_support::{TestPaths, capture_events};
    use crate::io::EngineEvent;

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_process_and_stats() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut ctx = paths.context(vec![]);

        let (result, events) = capture_events(&mut ctx, handle_process_cmd).await;
        test_ok!(result);
        assert!(result.is_ok());

        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::StatusUpdate { message } if message.contains("Successfully parsed library!")
        )));

        let (stats, events) = capture_events(&mut ctx, handle_stats_cmd).await;
        test_ok!(stats);
        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::Text { message }
                if message.contains("LanceDB metadata:") && message.contains("Number of rows: 8")
        )));
    }

    #[tokio::test]
    async fn test_checkhealth_no_database() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut ctx = paths.context(vec![]);
        let (result, events) = capture_events(&mut ctx, handle_checkhealth_cmd).await;
        result.unwrap();

        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::Text { message } if message.contains("storage does not exist")
        )));
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
        let (result, events) = capture_events(&mut ctx, handle_checkhealth_cmd).await;
        result.unwrap();

        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::Text { message } if message.contains("Vector Store Health Check Results")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::Text { message } if message.contains("storage exists")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::Text { message } if message.contains("Table is accessible")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::Text { message } if message.contains("Table has")
        )));
    }
}
