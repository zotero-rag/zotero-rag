//! Command handlers for library-related tasks

use std::io;
use std::{fs::File, io::Write};

use arrow_array::RecordBatch;
use arrow_ipc::{reader::FileReader, writer::FileWriter};
use zqa_rag::vector::checkhealth::lancedb_health_check;
use zqa_rag::vector::doctor::doctor as rag_doctor;

use crate::utils::terminal::{DIM_TEXT, RESET};
use crate::{
    cli::{app::BATCH_ITER_FILE, errors::CLIError},
    common::Context,
    full_library_to_arrow,
    store::common::ZoteroStore,
    utils::{
        arrow::library_to_arrow,
        library::{ZoteroItem, ZoteroItemSet, get_new_library_items, parse_library_metadata},
    },
};

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
/// This parses the library, extracts text from each file, stores the records in LanceDB,
/// and generates embeddings. If the embedding step fails, parsed records are kept in
/// [`BATCH_ITER_FILE`] so embedding can be retried later.
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

    let item_metadata =
        if ctx.store.exists().await {
            get_new_library_items(&ctx.config.get_embedding_config().ok_or(
                CLIError::ConfigError("Could not get embedding config".into()),
            )?)
            .await
        } else {
            parse_library_metadata(None, None)
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

        let mut option = String::new();
        io::stdin().read_line(&mut option)?;

        let option = option.trim().to_lowercase();
        if ["n", "no", "false", "0"].contains(&option.as_str()) {
            return Ok(());
        }
    }

    let record_batch = full_library_to_arrow(&ctx.config, None, None).await?;
    let schema = record_batch.schema();
    let batches = vec![record_batch.clone()];

    // Write to binary file using Arrow IPC format
    let file = File::create(BATCH_ITER_FILE)?;
    let mut writer = FileWriter::try_new(file, &schema)?;

    writer.write(&record_batch)?;
    writer.finish()?;

    let result = ctx.store.upsert_batches(batches).await;

    match result {
        Ok(()) => {
            writeln!(&mut ctx.out, "Successfully parsed library!")?;
            std::fs::remove_file(BATCH_ITER_FILE)?;
        }
        Err(e) => {
            writeln!(&mut ctx.err, "Parsing library failed: {e}")?;
            writeln!(
                &mut ctx.err,
                "The parsed PDFs have been saved in 'batch_iter.bin'. Run '/embed' to retry embedding."
            )?;
        }
    }

    Ok(())
}

/// Retry embedding from saved batch data or repair zero-vector rows.
///
/// When `fix` is `false`, this reads [`BATCH_ITER_FILE`] and inserts the saved batches into
/// LanceDB. When `fix` is `true`, it repairs rows whose stored embeddings are all zero.
///
/// # Arguments
///
/// * `fix` - Whether to repair zero-vector rows instead of replaying saved batch data.
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the command completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if reading batch data, accessing configuration,
/// database operations, or writing output fails.
pub(crate) async fn handle_embed_cmd<O, E>(
    fix: bool,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    if fix {
        return fix_zero_embeddings(ctx).await;
    }

    let file = File::open(BATCH_ITER_FILE)?;
    let reader = FileReader::try_new(file, None)?;

    let mut batches = Vec::<RecordBatch>::new();
    for batch in reader {
        batches.push(batch?);
    }

    if batches.is_empty() {
        writeln!(
            &mut ctx.err,
            "(/embed) It seems {BATCH_ITER_FILE} contains no batches. Exiting early."
        )?;
        return Ok(());
    }

    let n_batches = batches.len();

    write!(ctx.out, "Successfully loaded {n_batches} batch")?;

    if n_batches > 1 {
        write!(&mut ctx.out, "es")?;
    }
    writeln!(ctx.out, ".")?;

    let db = ctx.store.upsert_batches(batches).await;

    if db.is_ok() {
        writeln!(ctx.out, "Successfully parsed library!")?;
        std::fs::remove_file(BATCH_ITER_FILE)?;
    } else if let Err(e) = db {
        writeln!(ctx.err, "Parsing library failed: {e}")?;
        writeln!(
            ctx.err,
            "Your {BATCH_ITER_FILE} file has been left untouched."
        )?;
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
    let _ = match lancedb_health_check(ctx.config.embedding_provider).await {
        Ok(result) => writeln!(ctx.out, "{result}"),
        Err(e) => writeln!(ctx.err, "{e}"),
    };

    Ok(())
}

/// Run database diagnostics and attempt automatic repairs where supported.
///
/// Currently, only zero-vector embeddings are automatically repaired; other failures are
/// reported to the user.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if diagnostics and any attempted repair completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if diagnostics, repair, or writing output fails.
pub(crate) async fn handle_doctor_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    if let Err(e) = rag_doctor(ctx.config.embedding_provider, &mut ctx.out).await {
        writeln!(ctx.err, "{e}")?;
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
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if zero-vector handling completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if configuration is invalid, database operations fail,
/// embedding regeneration fails, or writing output fails.
async fn fix_zero_embeddings<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    let healthcheck = lancedb_health_check(ctx.config.embedding_provider).await?;

    let zero_batches = match healthcheck.zero_embedding_items {
        Some(Ok(zero_items)) => {
            let num_zeros: usize = zero_items
                .iter()
                .map(arrow_array::RecordBatch::num_rows)
                .sum();

            if num_zeros > 0 {
                writeln!(
                    ctx.out,
                    "{DIM_TEXT}Fixing {num_zeros} zero-embedding items.{RESET}"
                )?;
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
        writeln!(ctx.out, "{DIM_TEXT}Done!{RESET}")?;
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

    writeln!(
        ctx.out,
        "{num_empty_texts} items had empty texts, and will be deleted.\n"
    )?;

    if nonempty_zero_subset.is_empty() {
        return Ok(());
    }

    let include_embeddings = ctx.store.exists().await;
    let nonempty_zero_subset_batch = library_to_arrow(
        nonempty_zero_subset,
        embedding_config.clone(),
        include_embeddings,
    )
    .await?;

    let batches = vec![nonempty_zero_subset_batch.clone()];

    ctx.store.upsert_batches(batches).await?;

    writeln!(ctx.out, "Successfully fixed zero embeddings!\n")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        sync::Arc,
    };

    use arrow_array::{
        FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_ipc::writer::FileWriter;
    use lancedb::connect;
    use serial_test::serial;
    use temp_env;
    use zqa_macros::{test_contains, test_ok};
    use zqa_macros_proc::retry;
    use zqa_rag::constants::DEFAULT_VOYAGE_EMBEDDING_DIM;

    use super::{handle_checkhealth_cmd, handle_embed_cmd, handle_process_cmd, handle_stats_cmd};
    use crate::cli::app::{BATCH_ITER_FILE, tests::create_test_context};

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_embed() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "pdf_text",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(data)]).unwrap();

        let file = File::create(BATCH_ITER_FILE).unwrap();
        let mut writer = FileWriter::try_new(file, &schema).unwrap();
        writer.write(&record_batch).unwrap();
        writer.finish().unwrap();

        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_embed_cmd(false, &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let err = String::from_utf8(ctx.err.into_inner()).unwrap();
        assert!(err.is_empty());

        if fs::metadata(BATCH_ITER_FILE).is_ok() {
            fs::remove_file(BATCH_ITER_FILE).expect("Failed to clean up BATCH_ITER_FILE");
        }
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_process_and_stats() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_process_cmd(&mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.clone().into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let stats =
            temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], handle_stats_cmd(&mut ctx))
                .await;
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_ok!(stats);
        test_contains!(output, "LanceDB Statistics:");
        test_contains!(output, "Number of rows: 8");

        if fs::metadata(BATCH_ITER_FILE).is_ok() {
            fs::remove_file(BATCH_ITER_FILE).expect("Failed to clean up BATCH_ITER_FILE");
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_checkhealth_no_database() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();
        let output = temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], async move {
            handle_checkhealth_cmd(&mut ctx).await.unwrap();
            String::from_utf8(ctx.out.into_inner()).unwrap()
        })
        .await;

        assert!(output.contains("directory does not exist"));
    }

    #[retry(3)]
    #[tokio::test]
    #[serial]
    async fn test_checkhealth_with_database() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_process_cmd(&mut setup_ctx),
        )
        .await;
        test_ok!(result);

        let mut ctx = create_test_context();
        let output = temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], async move {
            handle_checkhealth_cmd(&mut ctx).await.unwrap();
            String::from_utf8(ctx.out.into_inner()).unwrap()
        })
        .await;

        test_contains!(output, "LanceDB Health Check Results");
        test_contains!(output, "directory exists");
        test_contains!(output, "Table is accessible");
        test_contains!(output, "Table has");
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
    #[serial]
    async fn test_fix_zero_embeddings_no_zeros() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_process_cmd(&mut setup_ctx),
        )
        .await;
        test_ok!(setup_result);

        let mut first_ctx = create_test_context();
        let first_result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_embed_cmd(true, &mut first_ctx),
        )
        .await;
        test_ok!(first_result);
        assert!(first_result.is_ok());

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_embed_cmd(true, &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "Done!");
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_fix_zero_embeddings_with_zero_rows() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_process_cmd(&mut setup_ctx),
        )
        .await;
        test_ok!(setup_result);

        insert_zero_embedding_row(&db_uri).await;

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_embed_cmd(true, &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "items had empty texts, and will be deleted.");
    }
}
