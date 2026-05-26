//! Command handlers for `/batch` commands.
//!
//! The philosophy here is that a user shouldn't have to care about how this works. As far as they
//! are concerned, this should "just work" (within reason). Our protocol is as follows:
//!
//! * Users can dispatch multiple `/batch` commands at once (i.e., while others are being processed
//!   by the provider).
//!   * In such cases, we ask ONCE that they intended to do this, and report the overlap size. At
//!     this point, they can either process only the new ones in this batch, or ask to process the
//!     entire thing anyway. In the latter case, we also give them an option to *replace* the
//!     previous batch, but *only* if the former batch is a proper subset of this one. If this is
//!     affirmative, we send a cancellation request to the provider and delete the old file.
//!   * We should always, when asking for such confirmations, also flag if that batch's
//!     provider/model ID are different from the currently-used one. This is actually a bigger
//!     problem since the DB shouldn't contain embeddings from different providers in the first place.
//! * The entire abstraction for a batch submission is a file placed in the user's state dir (see
//!   [`crate::state`] for an implementation), in a `batches` subdirectory. This means:
//!   * A file existing, and being valid (in its contents) implies that that batch exists. Whether
//!     it *actually* exists or is invalid data (e.g., it contains a batch ID that doesn't exist) is
//!     not guaranteed.
//!   * A file *not* existing means that as far as we are concerned, that batch wasn't dispatched by
//!     us; not our circus, not our monkeys.
//!   * The origin of the file is irrelevant. Users are free to create a file in that directory, and
//!     we will treat it as valid if the structure is right.
//!   * Files are named `batch_<id>.log`, where `id` are (1-based) sequence numbers to avoid dealing with
//!     clock drift ([`std::time::SystemTime`] is a fun read for the uninitiated).
//! * We check the status of batches on startup and when the user explicitly requests it.
//! * When we check the status of a batch:
//!   * If on startup, checking the status reveals a state that is not "submitted" (or that
//!     provider's lifecycle equivalent), we notify the user, and they decide how to proceed. This
//!     includes a response indicating that the batch doesn't exist.
//!   * If the batch has completely failed and there are no successes, we prompt the user to try again.
//!   * If the batch has partially succeeded, we notify the user and let them decide. The default
//!     option here should be to add the processed results to our backend and retry the remaining.
//!     The user should also have the option to just ignore and retry the whole thing, or pick the
//!     successes and drop the rest.
//!   * If the batch has completely succeeded, we notify the user only if this check was initiated
//!     at startup. We add these to our backend and remove this batch.
//! * Broadly, our semantics resemble a write-ahead log. The files are sorted by creation time, and
//!   we scan a cache when inserting a batch into the backend to check conflicts (and the newest
//!   one wins). Moreover, we maintain a cache of hashes (try saying that thrice quickly) with the
//!   following semantics:
//!   * The file is named `.hash_cache`, and is placed in `~/.local/state/zqa/batches`.
//!   * This file is shared across batches. Shared/exclusive locks to this file preclude other
//!     file handles from accessing it (this is unlikely, but it is a stronger guarantee than doing
//!     nothing).
//!   * The file contains a map from text hashes to corresponding provider ID, provider model, and
//!     the sequence number of the (newest) batch it belonged to.
//!   * Before writing to the DB, we filter out items that match the hash, provider, and provider
//!     model.
//!   * After writing to the DB, we add the hashes of the texts added to this cache along with that
//!     same auxiliary information.
//!
//! For the structure of this file, see [`BatchEmbeddingMetadata`].

use std::collections::HashSet;
use std::io::Seek;
use std::{
    collections::HashMap,
    fmt::Display,
    fs::{self, OpenOptions},
    hash::{DefaultHasher, Hasher},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use humantime::format_duration;
use serde::{Deserialize, Serialize};
use zqa_rag::{
    capabilities::BatchJobState,
    embedding::common::{
        BatchEmbeddingInput, BatchEmbeddingRequest, BatchEmbeddingResult, BatchSubmission,
    },
    llm::factory::BatchEmbeddingClient,
    providers::{ProviderId, registry::provider_registry},
};

use crate::{
    cli::{commands::BatchCommand, errors::CLIError},
    common::Context,
    utils::{
        arrow::library_to_arrow_with_embeddings,
        library::ZoteroItem,
        terminal::{read_char, read_number},
    },
};
use crate::{state::get_state_dir, utils::library::parse_library};

#[derive(Debug, Serialize, Deserialize)]
struct BatchItem {
    file_path: PathBuf,
    library_key: String,
    title: String,
    text: String,
    hash: u64,
}

impl From<ZoteroItem> for BatchItem {
    fn from(value: ZoteroItem) -> Self {
        // TODO: Replace DefaultHasher with a stable algorithm (e.g. xxhash or sha2)
        let mut hasher = DefaultHasher::new();
        hasher.write(value.text.as_bytes());

        Self {
            file_path: value.metadata.file_path,
            library_key: value.metadata.library_key,
            title: value.metadata.title,
            text: value.text,
            hash: hasher.finish(),
        }
    }
}

/// Metadata about a batch embedding request. This is the file structure used to represent a batch
/// in progress.
#[derive(Debug, Serialize, Deserialize)]
struct BatchEmbeddingMetadata {
    /// The sequence number of this batch
    seq_id: usize,
    /// The batch ID
    batch_id: String,
    /// The file ID from the provider's Files API
    file_id: String,
    /// The batch embedding provider
    provider: ProviderId,
    /// The embedding model name
    model: String,
    /// Date of creation (local time, in UTC). Used to give users context when referring to a batch.
    created_at: DateTime<Utc>,
    /// Items in this batch
    items: Vec<BatchItem>,
    /// Optional indices to the items above if we know this information (by checking).
    succeeded: Option<Vec<usize>>,
    /// Optional indices to the items above if we know this information (by checking).
    failed: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    /// Embedding provider ID
    provider_id: ProviderId,
    /// Embedding model string
    model: String,
    /// Sequence number of the batch that added this hash.
    seq_id: usize,
}

/// The structure stored in the hash cache.
type HashCache = HashMap<u64, CacheEntry>;

impl Display for BatchEmbeddingMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elapsed = (Utc::now() - self.created_at)
            .to_std()
            .map_or("Unknown time ago".into(), |d| {
                format_duration(d).to_string()
            });

        f.write_fmt(format_args!(
            "({}) {} ({}) - {} items",
            elapsed,
            self.provider.as_str(),
            self.model,
            self.items.len()
        ))
    }
}

/// Gets the next available sequence number among the batch files in the state directory.
///
/// # Returns
///
/// The next available sequence number
///
/// # Errors
///
/// * `CLIError::StateDirError` if the user's base directory could not be obtained. See
///   `directories::BaseDirError` for when this can occur.
/// * `CLIError::IOError` if writing to the state directory failed. This is typically caused
///   by permission issues.
fn get_seq_id() -> Result<usize, CLIError> {
    let batch_dir = get_state_dir()?.join("batches");
    if !batch_dir.exists() {
        // If it doesn't exist, we start at 1. This also eliminates this cause of an `Err` result
        // from [`std::fs::read_dir`].
        return Ok(1);
    }

    let last_seq = fs::read_dir(batch_dir)?
        .filter_map(|entry| {
            let file_name = entry.ok()?.file_name().into_string().ok()?;
            file_name
                .strip_prefix("batch_")?
                .strip_suffix(".log")?
                .parse::<usize>()
                .ok()
        })
        .max();

    last_seq.map_or_else(|| Ok(1), |val| Ok(val + 1))
}

/// Write out metadata for a *new* batch.
///
/// This creates a new 1-indexed batch file in the state dir and writes out a
/// [`BatchEmbeddingMetadata`] object.
///
/// # Arguments
///
/// * `provider` - The provider of the embedding model
/// * `model` - The embedding model string
/// * `items` - The items in this batch
/// * `submission` - The result of a `submit_batch` on a batch embedding provider
///
/// # Errors
///
/// * `CLIError::StateDirError` if the user's base directory could not be obtained. See
///   `directories::BaseDirError` for when this can occur.
/// * `CLIError::IOError` in the following cases:
///     * writing to the state directory failed. This is typically caused by permission issues.
///     * writing out the serialized data failed
/// * `CLIError::SerializationError` if JSON serialization failed.
fn write_batch_metadata(
    provider: ProviderId,
    model: String,
    items: Vec<BatchItem>,
    submission: BatchSubmission,
) -> Result<(), CLIError> {
    let batch_dir = get_state_dir()?.join("batches");
    if !batch_dir.exists() {
        fs::create_dir_all(&batch_dir)?;
    }

    let seq = get_seq_id()?;
    let metadata = BatchEmbeddingMetadata {
        seq_id: seq,
        batch_id: submission.batch_id,
        file_id: submission.file_id,
        provider,
        model,
        created_at: Utc::now(),
        items,
        succeeded: None,
        failed: None,
    };

    let filename = format!("batch_{seq}.log");
    let contents = serde_json::to_string_pretty(&metadata)?;
    fs::write(batch_dir.join(filename), contents)?;

    Ok(())
}

async fn handle_successful_batch_results<O, E>(
    ctx: &mut Context<O, E>,
    batch: &BatchEmbeddingMetadata,
    successes: &[BatchEmbeddingResult],
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    let ids_to_items = batch
        .items
        .iter()
        .map(|i| (i.library_key.as_str(), i)) // we use `library_key` as the id in the batch
        .collect::<HashMap<&str, &BatchItem>>();

    let items_to_embeddings: Vec<(&BatchItem, Vec<f32>)> = results.succeeded.into_iter().filter_map(|res| {
        if let Some(item) = ids_to_items.get(res.id.as_str()) {
            Some((*item, res.embedding))
        } else {
            writeln!(&mut ctx.err,
                "Item {} from batch response not in library. If you removed items from your library, this is fine.",
                res.id
            ).ok()?;

            None
        }
    })
    .collect();

    let batch_dir = get_state_dir()?.join("batches");

    let mut buf = Vec::<u8>::new(); // allocate before obtaining lock
    let mut cache_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(batch_dir.join(".hash_cache"))?;

    cache_file.lock()?;
    cache_file.read_to_end(&mut buf)?;

    // First batch ever: the cache will be empty but the file exists
    let cache = if buf.is_empty() {
        HashCache::new()
    } else {
        serde_json::from_slice::<HashCache>(buf.as_ref())?
    };

    // Filter the items in `batch` using `items_to_embeddings`: if the cache doesn't contain it,
    // it's new so we keep it; otherwise, look for a mismatch
    let to_insert = items_to_embeddings
        .into_iter()
        .filter(|(item, _)| {
            cache.get(&item.hash).is_none_or(|entry| {
                entry.provider_id != batch.provider || entry.model != batch.model
            })
        })
        .collect::<Vec<_>>();

    if to_insert.is_empty() {
        // TODO: this can result in a "zombie" batch, need to handle
        writeln!(
            &mut ctx.out,
            "All items in the batch were duplicates; no action taken.",
        )?;

        return Ok(());
    }

    writeln!(
        &mut ctx.out,
        "{} items to insert, {} items dropped as duplicates.",
        to_insert.len(),
        batch.items.len() - to_insert.len()
    )?;

    let item_hashes: HashSet<u64> = batch.items.iter().map(|i| i.hash).collect();

    // Partition cache items to separate those whose sequence ID needs to be updated
    let (mut overwriteable, rest): (HashMap<_, _>, HashMap<_, _>) =
        cache.into_iter().partition(|(k, v)| {
            item_hashes.contains(k)
                && v.provider_id == batch.provider
                && v.model == batch.model
                && v.seq_id < batch.seq_id
        });

    // For the matches, update the batch seq number
    for v in overwriteable.values_mut() {
        v.seq_id = batch.seq_id;
    }

    overwriteable.extend(rest);

    // Add the batch's new items to the set of cache entries to update
    for (item, _) in &to_insert {
        overwriteable.insert(
            item.hash,
            CacheEntry {
                provider_id: batch.provider,
                model: batch.model.clone(),
                seq_id: batch.seq_id,
            },
        );
    }

    // Build batch to insert into the LanceDB store
    let library_keys: Vec<&str> = to_insert
        .iter()
        .map(|(i, _)| i.library_key.as_str())
        .collect();
    let titles: Vec<&str> = to_insert.iter().map(|(i, _)| i.title.as_str()).collect();
    let file_paths: Vec<&str> = to_insert
        .iter()
        .map(|(i, _)| {
            i.file_path.to_str().ok_or(CLIError::CommandError(format!(
                "Invalid file path for item: {}",
                i.title.clone()
            )))
        })
        .collect::<Result<_, _>>()?;
    let pdf_texts: Vec<&str> = to_insert.iter().map(|(i, _)| i.text.as_str()).collect();
    let embeddings: Vec<Vec<f32>> = to_insert.into_iter().map(|(_, e)| e).collect();

    let embedding_config = ctx
        .config
        .get_embedding_config()
        .ok_or(CLIError::ConfigError("Embedding config not set up.".into()))?;

    ctx.store
        .upsert_batches(vec![library_to_arrow_with_embeddings(
            &library_keys,
            &titles,
            &file_paths,
            &pdf_texts,
            embeddings,
            &embedding_config,
        )?])
        .await?;
    ctx.store.create_or_update_indices().await?;

    cache_file.seek(io::SeekFrom::Start(0))?;
    cache_file.set_len(0)?;

    cache_file.write_all(serde_json::to_string_pretty(&overwriteable)?.as_bytes())?;
    cache_file.unlock()?;

    Ok(())
}

/// Interactively fetch batch results, and update the LanceDB store and the hash cache following the
/// protocol above.
#[allow(clippy::too_many_lines)]
async fn prompt_and_fetch_batch_results<O, E>(
    ctx: &mut Context<O, E>,
    client: BatchEmbeddingClient,
    batch: &BatchEmbeddingMetadata,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(&mut ctx.out, "Fetch results now? ([y]/n)")?;
    if read_char(&mut io::stdin().lock(), 'y', &['y', 'n']) != 'y' {
        return Ok(());
    }

    let batch_id = batch.batch_id.as_str();
    let results = client.fetch_results(batch_id).await?;

    // TODO: attempt a best-effort match anyway; maybe use a boolean flag or something and handle it
    // specially later.
    if results.succeeded.len() + results.failed.len() != batch.items.len() {
        writeln!(
            &mut ctx.err,
            "Length mismatch between batch results and saved batch. The file may have been corrupted."
        )?;
    }

    match (results.succeeded.is_empty(), results.failed.is_empty()) {
        (true, true) => {
            writeln!(
                &mut ctx.out,
                "The batch API did not send any results despite the batch being completed."
            )?;
            // TODO: prompt for retry
        }
        (true, false) => {
            // complete failure
            writeln!(
                &mut ctx.err,
                "All {} batch items failed.",
                results.failed.len()
            )?;

            for err in results.failed.iter().take(3) {
                writeln!(&mut ctx.err, "  {} - {}", err.id, err.error)?;
            }

            writeln!(&mut ctx.out, "Retry the entire batch? ([y]/n)")?;
            // TODO: implement retry here
        }
        (false, true) => {
            // complete success
            handle_successful_batch_results(ctx, batch, &results.succeeded).await?;

            let batch_dir = get_state_dir()?.join("batches");
            if results.failed.is_empty() {
                fs::remove_file(batch_dir.join(format!("batch_{}.log", batch.seq_id)))?;
            }
        }
        (false, false) => {
            // partial success
            writeln!(
                &mut ctx.out,
                "{} of {} items succeeded.",
                results.succeeded.len(),
                results.succeeded.len() + results.failed.len()
            )?;

            // TODO: prompt for retry
        }
    }

    Ok(())
}

async fn handle_batch_check_status_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    let config = &ctx
        .config
        .get_embedding_config()
        .ok_or(CLIError::CommandError(
        concat!(
            "Embedding config not set up. Verify that ~/.config/zqa/config.toml exists ",
            "and is set up correctly. See https://github.com/zotero-rag/zotero-rag#configuration ",
            "for configuration instructions."
        )
        .into(),
    ))?;

    // Get a list of batches
    let batch_dir = get_state_dir()?.join("batches");
    if !batch_dir.exists() {
        writeln!(
            &mut ctx.err,
            "No batches have been submitted. Try `/batch create` to submit one."
        )?;
        return Ok(());
    }

    let mut files = fs::read_dir(&batch_dir)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|f| {
            f.starts_with("batch_")
                && Path::new(f)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("log"))
        })
        .filter_map(|f| {
            f.strip_prefix("batch_")?
                .strip_suffix(".log")?
                .parse::<usize>()
                .ok()
        })
        .collect::<Vec<_>>();
    files.sort_unstable();
    files.reverse();

    let batches = files
        .into_iter()
        .filter_map(|id| fs::read_to_string(batch_dir.join(format!("batch_{id}.log"))).ok())
        .filter_map(|c| serde_json::from_str::<BatchEmbeddingMetadata>(&c).ok())
        .collect::<Vec<_>>();

    let selected_batch = match batches.len() {
        0 => {
            writeln!(
                &mut ctx.err,
                "No valid batches were found. Your state directory may have been corrupted."
            )?;
            return Ok(());
        }
        1 => batches.first().unwrap(),
        _ => {
            writeln!(
                &mut ctx.out,
                "You have multiple submitted batches. Please choose one from the below options:"
            )?;

            // We enforce a u8 (max 255) for `choice` below regardless, so we should also limit the
            // iterator. In any case, a list of 200+ items is overwhelming for users anyway.
            batches
                .iter()
                .take((u8::MAX - 1) as usize)
                .enumerate()
                .for_each(|(i, batch)| {
                    _ = writeln!(&mut ctx.out, "{}. {}", i + 1, batch);
                });

            let choice = read_number(
                &mut io::stdin().lock(),
                1,
                (
                    1,
                    u8::try_from(batches.len().min((u8::MAX - 1) as usize)).unwrap() + 1, // we don't expect > 255 batches
                ),
            );

            batches.get((choice as usize).saturating_sub(1)).unwrap()
        }
    };

    let registry = provider_registry();
    let embedder = match registry.create_batch_embedding(config) {
        Err(e) => {
            return Err(CLIError::CommandError(e.to_string()));
        }
        Ok(embedder) => embedder,
    };

    let status = embedder
        .check_batch_status(&selected_batch.batch_id)
        .await?;
    writeln!(&mut ctx.out, "Current status: {}", status.as_str())?;

    // TODO: For better UX, if this is completed, prompt to fetch, when that subcmd is implemented
    if status == BatchJobState::Completed {
        prompt_and_fetch_batch_results(ctx, embedder, selected_batch).await?;
    }

    Ok(())
}

async fn handle_batch_create_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    let new_items = match parse_library(&ctx.store, None, None).await {
        Ok(items) => items,
        Err(parse_err) => {
            writeln!(
                &mut ctx.err,
                "Could not parse library metadata: {parse_err}"
            )?;
            return Ok(());
        }
    };
    let new_items: Vec<BatchItem> = new_items.into_iter().map(Into::into).collect();

    let cfg = ctx.config.get_embedding_config();
    let Some(cfg) = cfg else {
        // `CommandError` variants don't exit the CLI
        return Err(CLIError::CommandError(
            concat!(
                "Could not get a config for batch embeddings. Ensure your config has an ",
                "embedding provider configured. See https://github.com/zotero-rag/zotero-rag#configuration ",
                "for configuration instructions."
            )
            .into(),
        ))?;
    };

    let registry = provider_registry();
    let embedder = match registry.create_batch_embedding(&cfg) {
        Err(e) => {
            return Err(CLIError::CommandError(e.to_string()));
        }
        Ok(embedder) => embedder,
    };

    let request = BatchEmbeddingRequest {
        model: cfg.model_name().into(),
        dims: cfg.dims(),
        inputs: new_items
            .iter()
            .map(|item| BatchEmbeddingInput {
                id: item.library_key.clone(),
                text: item.text.clone(),
            })
            .collect(),
    };

    let submission = embedder.submit_batch(request).await?;
    let batch_id = submission.batch_id.clone();
    write_batch_metadata(
        cfg.provider_id(),
        cfg.model_name().into(),
        new_items,
        submission,
    )?;

    writeln!(ctx.out, "Batch {batch_id} successfully created.")?;
    Ok(())
}

/// Handle the `/batch` commands.
pub(crate) async fn handle_batch_cmd<O, E>(
    subcmd: BatchCommand,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    match subcmd {
        BatchCommand::Create => handle_batch_create_cmd(ctx).await,
        BatchCommand::CheckStatus => handle_batch_check_status_cmd(ctx).await,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serial_test::serial;
    use tempfile::tempdir;
    use zqa_rag::{embedding::common::BatchSubmission, providers::ProviderId};

    use super::{
        BatchEmbeddingMetadata, BatchItem, get_seq_id, handle_batch_check_status_cmd,
        write_batch_metadata,
    };
    use crate::cli::app::tests::create_test_context;

    fn make_submission(batch_id: &str) -> BatchSubmission {
        BatchSubmission {
            batch_id: batch_id.into(),
            file_id: "file-id".into(),
        }
    }

    #[test]
    fn get_seq_id_returns_one_when_batch_dir_absent() {
        let tmp = tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            assert_eq!(get_seq_id().unwrap(), 1);
        });
    }

    #[test]
    fn get_seq_id_returns_one_for_empty_dir() {
        let tmp = tempdir().unwrap();
        let batch_dir = tmp.path().join("batches");
        fs::create_dir_all(&batch_dir).unwrap();

        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            assert_eq!(get_seq_id().unwrap(), 1);
        });
    }

    #[test]
    fn get_seq_id_returns_next_after_existing_files() {
        let tmp = tempdir().unwrap();
        let batch_dir = tmp.path().join("batches");
        fs::create_dir_all(&batch_dir).unwrap();
        fs::write(batch_dir.join("batch_1.log"), "").unwrap();
        fs::write(batch_dir.join("batch_2.log"), "").unwrap();
        fs::write(batch_dir.join("batch_3.log"), "").unwrap();

        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            assert_eq!(get_seq_id().unwrap(), 4);
        });
    }

    #[test]
    fn get_seq_id_ignores_non_batch_files() {
        let tmp = tempdir().unwrap();
        let batch_dir = tmp.path().join("batches");
        fs::create_dir_all(&batch_dir).unwrap();
        fs::write(batch_dir.join("other_file"), "").unwrap();
        fs::write(batch_dir.join("cache.bin"), "").unwrap();

        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            assert_eq!(get_seq_id().unwrap(), 1);
        });
    }

    #[test]
    fn get_seq_id_ignores_non_numeric_batch_suffixes() {
        let tmp = tempdir().unwrap();
        let batch_dir = tmp.path().join("batches");
        fs::create_dir_all(&batch_dir).unwrap();
        fs::write(batch_dir.join("batch_1.log"), "").unwrap();
        // "batch_abc.log" has a non-numeric suffix and should be silently skipped
        fs::write(batch_dir.join("batch_abc.log"), "").unwrap();

        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            assert_eq!(get_seq_id().unwrap(), 2);
        });
    }

    #[test]
    fn write_batch_metadata_creates_batches_dir() {
        let tmp = tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            write_batch_metadata(
                ProviderId::VoyageAI,
                "voyage-3".into(),
                vec![],
                make_submission("batch-abc"),
            )
            .unwrap();

            assert!(tmp.path().join("batches").exists());
        });
    }

    #[test]
    fn write_batch_metadata_stores_correct_fields() {
        let tmp = tempdir().unwrap();
        let items = vec![
            BatchItem {
                file_path: std::path::PathBuf::from("/tmp/a.pdf"),
                library_key: "KEY-A".into(),
                title: "Title A".into(),
                text: "text a".into(),
                hash: 1,
            },
            BatchItem {
                file_path: std::path::PathBuf::from("/tmp/b.pdf"),
                library_key: "KEY-B".into(),
                title: "Title B".into(),
                text: "text b".into(),
                hash: 2,
            },
        ];

        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            write_batch_metadata(
                ProviderId::VoyageAI,
                "voyage-3".into(),
                items,
                make_submission("my-batch-id"),
            )
            .unwrap();

            let batch_dir = tmp.path().join("batches");
            let files: Vec<_> = fs::read_dir(&batch_dir)
                .unwrap()
                .filter_map(std::result::Result::ok)
                .collect();
            assert_eq!(files.len(), 1);

            let content = fs::read_to_string(files[0].path()).unwrap();
            let meta: BatchEmbeddingMetadata = serde_json::from_str(&content).unwrap();

            assert_eq!(meta.batch_id, "my-batch-id");
            assert_eq!(meta.provider, ProviderId::VoyageAI);
            assert_eq!(meta.model, "voyage-3");
            assert_eq!(meta.items.len(), 2);
            assert_eq!(meta.items[0].library_key, "KEY-A");
            assert_eq!(meta.items[0].hash, 1);
            assert_eq!(meta.items[1].library_key, "KEY-B");
            assert_eq!(meta.items[1].hash, 2);
            assert!(meta.succeeded.is_none());
            assert!(meta.failed.is_none());
        });
    }

    #[test]
    fn write_batch_metadata_increments_sequence_number() {
        let tmp = tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            write_batch_metadata(
                ProviderId::VoyageAI,
                "voyage-3".into(),
                vec![],
                make_submission("batch-1"),
            )
            .unwrap();

            write_batch_metadata(
                ProviderId::VoyageAI,
                "voyage-3".into(),
                vec![],
                make_submission("batch-2"),
            )
            .unwrap();

            let batch_dir = tmp.path().join("batches");
            assert!(batch_dir.join("batch_1.log").exists());
            assert!(batch_dir.join("batch_2.log").exists());
        });
    }

    #[tokio::test]
    #[serial]
    async fn check_status_errors_when_embedding_config_missing() {
        let tmp = tempdir().unwrap();
        temp_env::async_with_vars(
            [("ZQA_STATE_DIR", Some(tmp.path().to_str().unwrap()))],
            async {
                let mut ctx = create_test_context();
                // Strip the (only) embedding provider config so `get_embedding_config` returns
                // `None`, which should short-circuit before any provider/network interaction.
                ctx.config.voyageai = None;

                let result = handle_batch_check_status_cmd(&mut ctx).await;
                assert!(matches!(
                    result,
                    Err(crate::cli::errors::CLIError::CommandError(_))
                ));
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn check_status_reports_when_no_batch_dir() {
        let tmp = tempdir().unwrap();
        temp_env::async_with_vars(
            [("ZQA_STATE_DIR", Some(tmp.path().to_str().unwrap()))],
            async {
                let mut ctx = create_test_context();

                let result = handle_batch_check_status_cmd(&mut ctx).await;
                assert!(result.is_ok());

                let err = String::from_utf8(ctx.err.into_inner()).unwrap();
                assert!(err.contains("No batches have been submitted"));
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn check_status_reports_when_no_valid_batches() {
        let tmp = tempdir().unwrap();
        let batch_dir = tmp.path().join("batches");
        fs::create_dir_all(&batch_dir).unwrap();
        // A correctly-named file whose contents don't deserialize into `BatchEmbeddingMetadata`
        // should be skipped, leaving no valid batches to choose from.
        fs::write(batch_dir.join("batch_1.log"), "not valid json").unwrap();

        temp_env::async_with_vars(
            [("ZQA_STATE_DIR", Some(tmp.path().to_str().unwrap()))],
            async {
                let mut ctx = create_test_context();

                let result = handle_batch_check_status_cmd(&mut ctx).await;
                assert!(result.is_ok());

                let err = String::from_utf8(ctx.err.into_inner()).unwrap();
                assert!(err.contains("No valid batches were found"));
            },
        )
        .await;
    }
}
