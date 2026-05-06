//! Command handlers for batch embedding API operations.

use std::{fs::File, io::Write, sync::Arc};

use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
use arrow_ipc::reader::FileReader;
use arrow_schema::{DataType, Field, Schema};
use zqa_rag::{
    capabilities::BatchJobState,
    embedding::batch_adapter::{BatchJobMetadata, BatchProviderAdapter},
};

use crate::{
    cli::{app::BATCH_ITER_FILE, errors::CLIError},
    common::Context,
};

/// Path to the persisted batch job metadata file.
pub(crate) const BATCH_JOB_FILE: &str = "batch_job.json";

/// Handle `/batch submit` — read texts from `batch_iter.bin` and submit them to the batch API.
///
/// After submission, writes [`BatchJobMetadata`] to [`BATCH_JOB_FILE`] so the user can check
/// status and collect results in a later session.
///
/// # Errors
///
/// Returns a [`CLIError`] if the batch file cannot be read, the provider does not support batch
/// embeddings, the API call fails, or writing output fails.
pub(crate) async fn handle_batch_submit_cmd<O: Write, E: Write>(
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    let file = File::open(BATCH_ITER_FILE).map_err(|_| {
        CLIError::ConfigError(format!(
            "Could not open {BATCH_ITER_FILE}. Run /process first to parse your library."
        ))
    })?;
    let reader = FileReader::try_new(file, None)?;

    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }

    if batches.is_empty() {
        writeln!(ctx.err, "No batches found in {BATCH_ITER_FILE}.")?;
        return Ok(());
    }

    let pdf_text_idx = batches[0]
        .schema()
        .column_with_name("pdf_text")
        .map(|(i, _)| i)
        .ok_or_else(|| CLIError::ConfigError("No pdf_text column found in batch data".into()))?;

    let texts: Vec<String> = batches
        .iter()
        .flat_map(|batch| {
            arrow_array::cast::as_string_array(batch.column(pdf_text_idx))
                .iter()
                .filter_map(|s| s.map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect();

    let total = texts.len();
    writeln!(ctx.out, "Submitting {total} texts to the batch embedding API...")?;

    let embedding_config =
        ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?;

    let adapter = BatchProviderAdapter::from_embedding_config(&embedding_config)?;
    let metadata = adapter.submit_texts(texts).await?;

    writeln!(ctx.out, "Batch submitted successfully!")?;
    writeln!(ctx.out, "  Batch ID : {}", metadata.batch_id)?;
    writeln!(ctx.out, "  Provider : {}", metadata.provider)?;
    writeln!(ctx.out, "  Texts    : {}", metadata.total_texts)?;
    writeln!(ctx.out)?;
    writeln!(ctx.out, "Check progress with '/batch status'.")?;
    writeln!(
        ctx.out,
        "Once complete, run '/batch collect' to retrieve and store the embeddings."
    )?;

    let json = serde_json::to_string(&metadata)
        .map_err(|e| CLIError::ConfigError(format!("Failed to serialize batch metadata: {e}")))?;
    std::fs::write(BATCH_JOB_FILE, json)?;

    Ok(())
}

/// Handle `/batch status [batch_id]` — report the current state of a batch job.
///
/// If `batch_id` is `None`, the ID is read from [`BATCH_JOB_FILE`].
///
/// # Errors
///
/// Returns a [`CLIError`] if the metadata file cannot be read, the API call fails, or writing
/// output fails.
pub(crate) async fn handle_batch_status_cmd<O: Write, E: Write>(
    batch_id: Option<String>,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    let (bid, provider) = if let Some(id) = batch_id {
        (id, None)
    } else {
        let meta = read_batch_metadata()?;
        (meta.batch_id, Some(meta.provider))
    };

    let embedding_config =
        ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?;

    let adapter = BatchProviderAdapter::from_embedding_config(&embedding_config)?;
    let state = adapter.get_status(&bid).await?;

    let status_str = match &state {
        BatchJobState::Created => "Created (queued, not yet started)",
        BatchJobState::InProgress => "In progress",
        BatchJobState::Completed => "Completed",
        BatchJobState::Failed => "Failed",
        BatchJobState::Canceling => "Canceling",
        BatchJobState::Canceled => "Canceled",
        _ => "Unknown",
    };

    writeln!(ctx.out, "Batch ID : {bid}")?;
    if let Some(prov) = provider {
        writeln!(ctx.out, "Provider : {prov}")?;
    }
    writeln!(ctx.out, "Status   : {status_str}")?;

    if matches!(state, BatchJobState::Completed) {
        writeln!(ctx.out)?;
        writeln!(
            ctx.out,
            "Batch complete! Run '/batch collect' to write the embeddings to the database."
        )?;
    }

    Ok(())
}

/// Handle `/batch collect [batch_id]` — download completed embeddings and write them to LanceDB.
///
/// If `batch_id` is `None`, the ID is read from [`BATCH_JOB_FILE`]. The command also requires
/// `batch_iter.bin` to be present so it can attach the embeddings to the original text records.
///
/// On success, both [`BATCH_JOB_FILE`] and `batch_iter.bin` are deleted.
///
/// # Errors
///
/// Returns a [`CLIError`] if the batch is not yet complete, either state file is missing,
/// the API call fails, building Arrow arrays fails, or the database write fails.
pub(crate) async fn handle_batch_collect_cmd<O: Write, E: Write>(
    batch_id: Option<String>,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    let metadata = {
        let mut m = read_batch_metadata()?;
        if let Some(id) = batch_id {
            m.batch_id = id;
        }
        m
    };

    let embedding_config =
        ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?;

    let adapter = BatchProviderAdapter::from_embedding_config(&embedding_config)?;

    let state = adapter.get_status(&metadata.batch_id).await?;
    if !matches!(state, BatchJobState::Completed) {
        let status_str = match state {
            BatchJobState::Created => "Created",
            BatchJobState::InProgress => "InProgress",
            BatchJobState::Failed => "Failed",
            BatchJobState::Canceling => "Canceling",
            BatchJobState::Canceled => "Canceled",
            _ => "Unknown",
        };
        writeln!(
            ctx.err,
            "Batch '{}' is not yet complete (status: {status_str}). Try again later.",
            metadata.batch_id
        )?;
        return Ok(());
    }

    writeln!(
        ctx.out,
        "Downloading embeddings for batch '{}'...",
        metadata.batch_id
    )?;
    let embeddings = adapter.collect_embeddings(&metadata).await?;
    writeln!(ctx.out, "Retrieved {} embeddings.", embeddings.len())?;

    let file = File::open(BATCH_ITER_FILE).map_err(|_| {
        CLIError::ConfigError(format!(
            "Could not open {BATCH_ITER_FILE}. \
             The file must still be present to attach the downloaded embeddings."
        ))
    })?;
    let reader = FileReader::try_new(file, None)?;

    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }

    let embedding_dim = embedding_config.embedding_dims();

    let mut offset = 0usize;
    let mut new_batches = Vec::with_capacity(batches.len());
    for batch in batches {
        let n = batch.num_rows();
        let batch_embeddings = embeddings.get(offset..offset + n).ok_or_else(|| {
            CLIError::ConfigError("Embedding count does not match the number of batch rows".into())
        })?;
        offset += n;
        new_batches.push(attach_embeddings(&batch, batch_embeddings, embedding_dim)?);
    }

    writeln!(ctx.out, "Writing embeddings to the database...")?;
    ctx.store.upsert_precomputed_batches(new_batches).await?;

    std::fs::remove_file(BATCH_ITER_FILE).ok();
    std::fs::remove_file(BATCH_JOB_FILE).ok();

    writeln!(ctx.out, "Successfully stored batch embeddings!")?;
    writeln!(ctx.out, "Run '/index' to update the search indices.")?;

    Ok(())
}

/// Read [`BatchJobMetadata`] from [`BATCH_JOB_FILE`].
fn read_batch_metadata() -> Result<BatchJobMetadata, CLIError> {
    let content = std::fs::read_to_string(BATCH_JOB_FILE).map_err(|_| {
        CLIError::ConfigError(format!(
            "Could not read {BATCH_JOB_FILE}. Run '/batch submit' first."
        ))
    })?;
    serde_json::from_str(&content)
        .map_err(|e| CLIError::ConfigError(format!("Failed to parse batch metadata: {e}")))
}

/// Append a pre-computed `embeddings` column to an Arrow [`RecordBatch`].
fn attach_embeddings(
    batch: &RecordBatch,
    embeddings: &[Vec<f32>],
    dim: usize,
) -> Result<RecordBatch, CLIError> {
    let dim_i32 = i32::try_from(dim)
        .map_err(|_| CLIError::ConfigError(format!("Embedding dimension {dim} overflows i32")))?;
    let flattened: Vec<f32> = embeddings.iter().flatten().copied().collect();
    let values = Float32Array::from(flattened);
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let embedding_col = FixedSizeListArray::try_new(
        Arc::clone(&item_field),
        dim_i32,
        Arc::new(values),
        None,
    )?;

    let embedding_field = Arc::new(Field::new(
        "embeddings",
        DataType::FixedSizeList(item_field, dim_i32),
        false,
    ));

    let new_schema = Arc::new(Schema::new(
        batch
            .schema()
            .fields()
            .iter()
            .cloned()
            .chain(std::iter::once(embedding_field))
            .collect::<Vec<_>>(),
    ));

    let mut columns: Vec<Arc<dyn arrow_array::Array>> = batch.columns().to_vec();
    columns.push(Arc::new(embedding_col));

    RecordBatch::try_new(new_schema, columns).map_err(Into::into)
}
