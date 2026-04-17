//! Command handlers for library-related tasks

use std::{fs::File, io::Write};

use crate::utils::terminal::{DIM_TEXT, RESET};
use arrow_array::RecordBatch;
use arrow_ipc::{reader::FileReader, writer::FileWriter};
use std::io;
use zqa_rag::vector::doctor::doctor as rag_doctor;
use zqa_rag::vector::lance::db_statistics;
use zqa_rag::vector::{
    checkhealth::lancedb_health_check,
    lance::{
        create_or_update_indexes, dedup_rows, delete_rows, get_zero_vector_records, insert_records,
        lancedb_exists,
    },
};

use crate::{
    cli::{app::BATCH_ITER_FILE, errors::CLIError},
    common::Context,
    full_library_to_arrow,
    utils::{
        arrow::{DbFields, get_schema, library_to_arrow},
        library::{ZoteroItem, ZoteroItemSet, get_new_library_items, parse_library_metadata},
    },
};

/// Prints out table statistics from the created DB. Fails if the database does not exist, could
/// not be read, or the statistics could not be computed.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
pub(crate) async fn handle_stats_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    match db_statistics().await {
        Ok(stats) => writeln!(&mut ctx.out, "{stats}")?,
        Err(e) => writeln!(&mut ctx.err, "Could not get database statistics: {e}")?,
    }

    Ok(())
}

/// Process a user's Zotero library. This acts as one of the main functions provided by the CLI.
/// This parses the library, extracts the text from each file, stores them in a `LanceDB`
/// table, and adds their embeddings. If the last step fails, the parsed texts are stored
/// in `BATCH_ITER_FILE`.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
pub(crate) async fn handle_process_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    const WARNING_THRESHOLD: usize = 100;

    let item_metadata =
        if lancedb_exists().await {
            get_new_library_items(&ctx.config.get_embedding_config().ok_or(
                CLIError::ConfigError("Could not get embedding config".into()),
            )?)
            .await
        } else {
            parse_library_metadata(None, None)
        };

    if let Err(parse_err) = item_metadata {
        writeln!(
            &mut ctx.out,
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

    let result = insert_records(
        batches,
        Some(&[DbFields::LibraryKey.as_ref()]),
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        DbFields::PdfText.as_ref(),
    )
    .await;

    match result {
        Ok(_) => {
            writeln!(&mut ctx.out, "Successfully parsed library!")?;
            std::fs::remove_file(BATCH_ITER_FILE)?;
        }
        Err(e) => {
            writeln!(&mut ctx.out, "Parsing library failed: {e}")?;
            writeln!(
                &mut ctx.out,
                "The parsed PDFs have been saved in 'batch_iter.bin'. Run '/embed' to retry embedding."
            )?;
        }
    }

    Ok(())
}

/// Embed text from PDFs parsed, in case this step previously failed. This function reads
/// the `BATCH_ITER_FILE` and uses the data in there to compute embeddings and write out
/// the `LanceDB` table.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
/// * `fix_zeros` - If `true`, fixes zero-embedding vectors, but does not handle PDFs
///   parsed but not embedded.
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

    let db = insert_records(
        batches,
        Some(&[DbFields::LibraryKey.as_ref()]),
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        DbFields::PdfText.as_ref(),
    )
    .await;

    if db.is_ok() {
        writeln!(ctx.out, "Successfully parsed library!")?;
        std::fs::remove_file(BATCH_ITER_FILE)?;
    } else if let Err(e) = db {
        writeln!(ctx.out, "Parsing library failed: {e}")?;
        writeln!(
            ctx.out,
            "Your {BATCH_ITER_FILE} file has been left untouched."
        )?;
    }

    Ok(())
}

pub(crate) async fn handle_dedup_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    let result = dedup_rows(
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        get_schema(ctx.config.embedding_provider).await,
        DbFields::Title.as_ref(),
        DbFields::LibraryKey.as_ref(),
    )
    .await;

    match result {
        Ok(count) => {
            writeln!(ctx.out, "Deduped {count} rows")?;
        }
        Err(e) => {
            return Err(CLIError::LanceError(e.to_string()));
        }
    }

    Ok(())
}

pub(crate) async fn handle_index_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(
        &mut ctx.out,
        "Updating indices. This may take a while depending on how many items need to be added."
    )?;

    create_or_update_indexes(DbFields::PdfText.as_ref(), DbFields::Embeddings.as_ref()).await?;

    writeln!(
        &mut ctx.out,
        "Done! You should verify the indices exist with /checkhealth."
    )?;

    Ok(())
}

/// Performs comprehensive health checks on the `LanceDB` database and reports status
/// with colored output using ASCII escape codes.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
pub(crate) async fn handle_checkhealth_cmd<O: Write, E: Write>(
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    let _ = match lancedb_health_check(ctx.config.embedding_provider).await {
        Ok(result) => writeln!(ctx.out, "{result}"),
        Err(e) => writeln!(ctx.err, "{e}"),
    };

    Ok(())
}

/// Runs health checks on the `LanceDB` database and provides helpful suggestions to the user on how
/// to fix any issues, if that is possible. Automatically attempt to fix issues found. Currently,
/// only zero-embedding vectors can be fixed, since a lot of the other issues are possibly just DB
/// corruption. Maybe we can diagnose that in the future.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
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

/// Fix the zero-embedding problem. In some error cases, we store zero-vectors as embeddings in
/// `LanceDB`. This function fixes those errors by replacing them with "real" embeddings. Note that
/// there are cases where the embeddings are zeros not because there was an error, but because the
/// extracted text was empty. This could be the result of a failed attempt to parse, or some other
/// similar error. APIs like Voyage do accept empty strings, and simply return a zero vector.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
async fn fix_zero_embeddings<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    let healthcheck = lancedb_health_check(ctx.config.embedding_provider).await?;

    if let Some(Ok(zero_items)) = healthcheck.zero_embedding_items {
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
    }

    let zero_batches: Vec<RecordBatch> =
        get_zero_vector_records(&ctx.config.get_embedding_config().ok_or(
            CLIError::ConfigError("Could not get embedding config".into()),
        )?)
        .await?;

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
        .map(|item| item.metadata.library_key.as_str())
        .collect();

    delete_rows(
        DbFields::LibraryKey.as_ref(),
        &zero_subset_keys,
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
    )
    .await?;

    writeln!(
        ctx.out,
        "{num_empty_texts} items had empty texts, and will be deleted.\n"
    )?;

    if nonempty_zero_subset.is_empty() {
        return Ok(());
    }

    let nonempty_zero_subset_batch = library_to_arrow(
        nonempty_zero_subset,
        ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
    )
    .await?;

    let batches = vec![nonempty_zero_subset_batch.clone()];

    insert_records(
        batches,
        Some(&[DbFields::LibraryKey.as_ref()]),
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        DbFields::PdfText.as_ref(),
    )
    .await?;

    writeln!(ctx.out, "Successfully fixed zero embeddings!\n")?;

    Ok(())
}
