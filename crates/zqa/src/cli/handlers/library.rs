//! Command handlers for library-related tasks

use std::fs::File;

use arrow_array::RecordBatch;
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use zqa_rag::vector::doctor::doctor as rag_doctor;

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::full_library_to_arrow;
use crate::io::EngineEvent;
use crate::store::common::ZoteroStore;
use crate::utils::arrow::library_to_arrow;
use crate::utils::library::{
    ZoteroItem, ZoteroItemSet, get_new_library_items, parse_library_metadata,
};
use crate::utils::terminal::{DIM_TEXT, RESET, read_line};

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
/// This parses the library, extracts text from each file, stores the records in LanceDB,
/// and generates embeddings. If the embedding step fails, parsed records are kept in
/// [`BATCH_ITER_FILE`](crate::cli::app::BATCH_ITER_FILE) so embedding can be retried later.
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
/// Returns a [`CLIError`] if file operations fail, configuration is invalid,
/// parsing / insertion setup fails, or the event receiver is closed.
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

    let record_batch = full_library_to_arrow(&ctx.store, library_path, None, None).await?;
    let schema = record_batch.schema();
    let batches = vec![record_batch.clone()];

    // Write to binary file using Arrow IPC format
    let batch_iter_path = &ctx.path_options.batch_iter_path;
    let file = File::create(batch_iter_path)?;
    let mut writer = FileWriter::try_new(file, &schema)?;

    writer.write(&record_batch)?;
    writer.finish()?;
    log::debug!(
        "Saved Arrow recovery file: path={}, rows={}",
        batch_iter_path.display(),
        record_batch.num_rows()
    );

    let result = ctx.store.upsert_batches(batches).await;

    match result {
        Ok(()) => {
            ctx.emit(EngineEvent::StatusUpdate {
                message: "Successfully parsed library!\n".into(),
            })
            .await?;
            std::fs::remove_file(batch_iter_path)?;
            log::debug!(
                "Removed Arrow recovery file after successful write: {}",
                batch_iter_path.display()
            );
        }
        Err(e) => {
            log::debug!(
                "Database write failed; retaining Arrow recovery file {}: {}",
                batch_iter_path.display(),
                zqa_rag::logging::preview(&e)
            );
            ctx.emit(EngineEvent::Error {
                message: format!("Parsing library failed: {e}\n"),
            })
            .await?;
            ctx.emit(EngineEvent::Warning {
                message: format!(
                    "The parsed PDFs have been saved in '{}'. Run '/embed' to retry embedding.\n",
                    batch_iter_path.display()
                ),
            })
            .await?;
        }
    }

    Ok(())
}

/// Retry embedding from saved batch data or repair zero-vector rows.
///
/// When `fix` is `false`, this reads [`BATCH_ITER_FILE`](crate::cli::app::BATCH_ITER_FILE) and inserts the saved batches into
/// LanceDB. When `fix` is `true`, it repairs rows whose stored embeddings are all zero.
///
/// # Arguments
///
/// * `fix` - Whether to repair zero-vector rows instead of replaying saved batch data.
/// * `ctx` - The application context, including the store, configuration, and event sender.
///
/// # Returns
///
/// `Ok(())` if the command completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if reading batch data, accessing configuration,
/// or database operations fail, or the event receiver is closed.
pub(crate) async fn handle_embed_cmd(fix: bool, ctx: &mut Context) -> Result<(), CLIError> {
    if fix {
        return fix_zero_embeddings(ctx).await;
    }

    let batch_iter_path = &ctx.path_options.batch_iter_path;
    let batch_iter_display = batch_iter_path.display();
    log::debug!("Replaying Arrow recovery file: {batch_iter_display}");

    let file = File::open(batch_iter_path)?;
    let reader = FileReader::try_new(file, None)?;

    let mut batches = Vec::<RecordBatch>::new();
    for batch in reader {
        batches.push(batch?);
    }

    if batches.is_empty() {
        ctx.emit(EngineEvent::Warning {
            message: format!(
                "(/embed) It seems {batch_iter_display} contains no batches. Exiting early.\n"
            ),
        })
        .await?;
        return Ok(());
    }

    let n_batches = batches.len();
    log::debug!(
        "Loaded Arrow recovery data: batches={n_batches}, rows={}",
        batches.iter().map(RecordBatch::num_rows).sum::<usize>()
    );

    let suffix = if n_batches > 1 { "es" } else { "" };
    ctx.emit(EngineEvent::StatusUpdate {
        message: format!("Successfully loaded {n_batches} batch{suffix}.\n"),
    })
    .await?;

    let db = ctx.store.upsert_batches(batches).await;

    if db.is_ok() {
        ctx.emit(EngineEvent::StatusUpdate {
            message: "Successfully parsed library!\n".into(),
        })
        .await?;
        std::fs::remove_file(batch_iter_path)?;
        log::debug!("Removed Arrow recovery file after successful replay: {batch_iter_display}");
    } else if let Err(e) = db {
        log::debug!(
            "Replay failed; retaining Arrow recovery file {batch_iter_display}: {}",
            zqa_rag::logging::preview(&e)
        );
        ctx.emit(EngineEvent::Error {
            message: format!("Parsing library failed: {e}\n"),
        })
        .await?;
        ctx.emit(EngineEvent::Warning {
            message: format!("Your {batch_iter_display} file has been left untouched.\n"),
        })
        .await?;
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

/// Run database diagnostics and attempt automatic repairs where supported.
///
/// Currently, only zero-vector embeddings are automatically repaired; other failures are
/// reported to the user.
///
/// # Arguments
///
/// * `ctx` - The application context, including the store, configuration, and event sender.
///
/// # Returns
///
/// `Ok(())` if diagnostics and any attempted repair completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if repair fails or the event receiver is closed.
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

    // Currently, we can really only fix the zero-embeddings issue
    fix_zero_embeddings(ctx).await
}

/// Repair rows in LanceDB whose stored embedding vectors are all zeros.
///
/// Some zero vectors indicate failed embedding generation, while others correspond to rows with
/// empty extracted text. Empty-text rows are deleted; non-empty rows are re-embedded.
///
/// # Arguments
///
/// * `ctx` - The application context, including the store, configuration, and event sender.
///
/// # Returns
///
/// `Ok(())` if zero-vector handling completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if configuration is invalid, database operations fail,
/// embedding regeneration fails, or the event receiver is closed.
async fn fix_zero_embeddings(ctx: &mut Context) -> Result<(), CLIError> {
    let healthcheck = ctx.store.health_check().await;

    let zero_batches = match healthcheck.zero_embedding_items {
        Some(Ok(zero_items)) => {
            let num_zeros: usize = zero_items
                .iter()
                .map(arrow_array::RecordBatch::num_rows)
                .sum();

            if num_zeros > 0 {
                ctx.emit(EngineEvent::StatusUpdate {
                    message: format!("{DIM_TEXT}Fixing {num_zeros} zero-embedding items.{RESET}\n"),
                })
                .await?;
            }

            zero_items
        }
        Some(Err(e)) => return Err(e.into()),
        None => Vec::new(),
    };

    let embedding_config = ctx
        .config
        .get_embedding_config()
        .ok_or(CLIError::ConfigError(
            "Could not get embedding config".into(),
        ))?;

    if zero_batches.is_empty() {
        ctx.emit(EngineEvent::StatusUpdate {
            message: format!("{DIM_TEXT}Done!{RESET}\n"),
        })
        .await?;
        return Ok(());
    }

    let zero_subset: Vec<ZoteroItem> = ZoteroItemSet::from(zero_batches).into();
    let nonempty_zero_subset = zero_subset
        .iter()
        .filter(|&item| !item.text.is_empty())
        .cloned()
        .collect::<Vec<_>>();

    let num_empty_texts = zero_subset.len() - nonempty_zero_subset.len();

    let zero_subset_keys: Vec<_> = zero_subset
        .iter()
        .map(|item| item.metadata.library_key.clone())
        .collect();

    ctx.store.delete_by_library_keys(&zero_subset_keys).await?;

    ctx.emit(EngineEvent::StatusUpdate {
        message: format!("{num_empty_texts} items had empty texts, and will be deleted.\n\n"),
    })
    .await?;

    if nonempty_zero_subset.is_empty() {
        return Ok(());
    }

    let include_embeddings = ctx.store.exists().await;
    let nonempty_zero_subset_batch =
        library_to_arrow(&nonempty_zero_subset, &embedding_config, include_embeddings)?;

    let batches = vec![nonempty_zero_subset_batch.clone()];

    ctx.store.upsert_batches(batches).await?;

    ctx.emit(EngineEvent::StatusUpdate {
        message: "Successfully fixed zero embeddings!\n\n".into(),
    })
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::{
        FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_ipc::writer::FileWriter;
    use lancedb::connect;
    use zqa_macros::test_ok;
    use zqa_macros_proc::retry;
    use zqa_rag::constants::DEFAULT_VOYAGE_EMBEDDING_DIM;

    use super::{handle_checkhealth_cmd, handle_embed_cmd, handle_process_cmd, handle_stats_cmd};
    use crate::common::test_support::{TestPaths, capture_events};
    use crate::io::EngineEvent;

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_embed() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut ctx = paths.context(vec![]);

        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "pdf_text",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(data)]).unwrap();

        let file = File::create(&ctx.path_options.batch_iter_path).unwrap();
        let mut writer = FileWriter::try_new(file, &schema).unwrap();
        writer.write(&record_batch).unwrap();
        writer.finish().unwrap();

        let (result, events) =
            capture_events(&mut ctx, async |ctx| handle_embed_cmd(false, ctx).await).await;
        test_ok!(result);
        assert!(result.is_ok());

        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::StatusUpdate { message } if message.contains("Successfully parsed library!")
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            EngineEvent::Warning { .. }
                | EngineEvent::Error { .. }
                | EngineEvent::RecoverableWarning { .. }
        )));
    }

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
    #[tokio::test]
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

    async fn insert_zero_embedding_row(db_uri: &str) {
        let dims = DEFAULT_VOYAGE_EMBEDDING_DIM as i32;
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new(
                "embeddings",
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::Float32,
                        true,
                    )),
                    dims,
                ),
                false,
            ),
        ]));

        #[allow(clippy::cast_sign_loss)]
        let zeros = Float32Array::from(vec![0.0f32; dims as usize]);
        let embedding_col = FixedSizeListArray::try_new(
            Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                true,
            )),
            dims,
            Arc::new(zeros),
            None,
        )
        .unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["ZEROTEST001"])),
                Arc::new(StringArray::from(vec!["Zero Test Item"])),
                Arc::new(StringArray::from(vec!["/dev/null"])),
                Arc::new(StringArray::from(vec![""])),
                Arc::new(embedding_col),
            ],
        )
        .unwrap();

        let db = connect(db_uri).execute().await.unwrap();
        let tbl = db.open_table("data").execute().await.unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        tbl.merge_insert(&["library_key"])
            .when_not_matched_insert_all()
            .clone()
            .execute(Box::new(reader))
            .await
            .unwrap();
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_fix_zero_embeddings_no_zeros() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut setup_ctx = paths.context(vec![]);
        let setup_result = handle_process_cmd(&mut setup_ctx).await;
        test_ok!(setup_result);

        let mut first_ctx = paths.context(vec![]);
        let first_result = handle_embed_cmd(true, &mut first_ctx).await;
        test_ok!(first_result);
        assert!(first_result.is_ok());

        let mut ctx = paths.context(vec![]);
        let (result, events) =
            capture_events(&mut ctx, async |ctx| handle_embed_cmd(true, ctx).await).await;
        test_ok!(result);
        assert!(result.is_ok());

        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::StatusUpdate { message } if message.contains("Done!")
        )));
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_fix_zero_embeddings_with_zero_rows() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut setup_ctx = paths.context(vec![]);
        let setup_result = handle_process_cmd(&mut setup_ctx).await;
        test_ok!(setup_result);

        insert_zero_embedding_row(&paths.db_uri).await;

        let mut ctx = paths.context(vec![]);
        let (result, events) =
            capture_events(&mut ctx, async |ctx| handle_embed_cmd(true, ctx).await).await;
        test_ok!(result);
        assert!(result.is_ok());

        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::StatusUpdate { message }
                if message.contains("items had empty texts, and will be deleted.")
        )));
    }
}
