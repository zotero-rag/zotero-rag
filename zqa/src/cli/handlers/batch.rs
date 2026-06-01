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
//! * When we check the status of a batch and fetch its results, exactly one of the following applies:
//!   * If on startup, checking the status reveals a state that is not "submitted" (or that
//!     provider's lifecycle equivalent), we notify the user, and they decide how to proceed. This
//!     includes a response indicating that the batch doesn't exist.
//!   * If the provider returns a number of results that does not match the items in our batch file,
//!     we treat the file as corrupted and surface a hard error. We don't try to partially reconcile.
//!   * If the batch has completely failed and there are no successes, we prompt the user to try
//!     again. A retry creates a new batch with the same items (see "retries" below).
//!   * If the batch has partially succeeded, we notify the user and let them decide. The default
//!     option here should be to add the processed results to our backend and retry the remaining.
//!     The user should also have the option to just ignore and retry the whole thing, or pick the
//!     successes and drop the rest. Regardless of choice, successes that are applied flow through
//!     the hash cache so they aren't re-embedded later.
//!   * If the batch has completely succeeded, we notify the user only if this check was initiated
//!     at startup. We add these to our backend and remove the batch file.
//! * Retries are *new* batches. Batch IDs are immutable at the provider, and we mirror that on disk:
//!   a retry submits a fresh request (containing only the items that need re-embedding) and writes
//!   a new `batch_<id>.log` with its own sequence number. The original file is removed once its
//!   successes have been applied and the retry has been submitted. Each file is a self-contained
//!   record of one batch: `items`, `batch_id`, `provider`, `model`, and `seq_id` are immutable
//!   once written. The `succeeded` / `failed` index fields are bookkeeping — they may be updated
//!   in place to record what we've observed about that batch, but the classification is always a
//!   snapshot of a single fetch, never a merge across runs.
//! * Broadly, our semantics resemble a write-ahead log. The files are processed in sequence-number
//!   order, and we scan a cache when inserting a batch into the backend to check conflicts (and
//!   the newest one wins). Moreover, we maintain a cache of hashes (try saying that thrice
//!   quickly) with the following semantics:
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
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use humantime::format_duration;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;
use zqa_rag::capabilities::BatchAPIProvider;
use zqa_rag::{
    capabilities::BatchJobState,
    embedding::common::{
        BatchEmbeddingInput, BatchEmbeddingRequest, BatchEmbeddingResult, BatchSubmission,
    },
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchItem {
    file_path: PathBuf,
    library_key: String,
    title: String,
    text: String,
    hash: u64,
}

impl From<ZoteroItem> for BatchItem {
    fn from(value: ZoteroItem) -> Self {
        Self {
            file_path: value.metadata.file_path,
            library_key: value.metadata.library_key,
            title: value.metadata.title,
            hash: xxh3::xxh3_64(value.text.as_bytes()),
            text: value.text,
        }
    }
}

/// Metadata about a batch embedding request. This is the file structure used to represent a batch
/// in progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Clone, Serialize, Deserialize)]
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

/// Given an existing hash cache, metadata about a batch, and the hashes of the items this batch
/// added to the backend, return an updated cache. Existing entries owned by this batch (matching
/// provider/model with an older sequence number) have their sequence number bumped; `new_hashes`
/// are inserted with this batch's provider/model/sequence number.
///
/// # Arguments
///
/// * `cache` - The existing hash cache.
/// * `batch` - Details about the batch whose results are being processed.
/// * `new_hashes` - Hashes of the items this batch inserted into the backend.
///
/// # Returns
///
/// An updated hash cache.
fn update_hash_cache(
    cache: HashCache,
    batch: &BatchEmbeddingMetadata,
    new_hashes: impl IntoIterator<Item = u64>,
) -> HashCache {
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
    for hash in new_hashes {
        overwriteable.insert(
            hash,
            CacheEntry {
                provider_id: batch.provider,
                model: batch.model.clone(),
                seq_id: batch.seq_id,
            },
        );
    }

    overwriteable
}

/// Given a `batch` that has nonzero successes, insert those items into the vector store and update the
/// hash cache. As necessary, this function updates the sequence IDs of elements in the hash cache
/// (since the newest one wins) and adds elements from the batch that are not already present in the
/// cache.
///
/// # Arguments
///
/// * `ctx` - The app's context object
/// * `batch` - Metadata about the batch embedding request
/// * `successes` - Result objects corresponding to the batch API's successful results
///
/// # Errors
///
/// * `CLIError::StateDirError` if the user's base directory could not be obtained. See
///   `directories::BaseDirError` for when this can occur.
/// * `CLIError::IOError` in the following cases:
///     * writing to the state directory failed. This is typically caused by permission issues.
///     * writing out the serialized data failed
/// * `CLIError::SerializationError` if JSON serialization failed.
async fn handle_successful_batch_results<O, E>(
    ctx: &mut Context<O, E>,
    batch: &BatchEmbeddingMetadata,
    successes: Vec<BatchEmbeddingResult>,
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

    let items_to_embeddings: Vec<(&BatchItem, Vec<f32>)> = successes.into_iter().filter_map(|res| {
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

    let updated_cache = update_hash_cache(cache, batch, to_insert.iter().map(|(i, _)| i.hash));

    if to_insert.is_empty() {
        cache_file.seek(io::SeekFrom::Start(0))?;
        cache_file.set_len(0)?;
        cache_file.write_all(serde_json::to_string_pretty(&updated_cache)?.as_bytes())?;
        cache_file.unlock()?;

        // The user likely doesn't really care about the WAL semantics.
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

    cache_file.write_all(serde_json::to_string_pretty(&updated_cache)?.as_bytes())?;
    cache_file.unlock()?;

    Ok(())
}

/// Given a set of items, submit a batch request based on the configuration in `ctx`, and write the
/// batch metadata file in the state dir.
///
/// # Arguments
///
/// * `ctx` - The app's context object
/// * `items` - The items to send as a batch request
///
/// # Errors
///
/// * `CLIError::StateDirError` if the user's base directory could not be obtained. See
///   `directories::BaseDirError` for when this can occur.
/// * `CLIError::IOError` in the following cases:
///     * writing to the state directory failed. This is typically caused by permission issues.
///     * writing out the serialized data failed
async fn retry_items<O, E>(ctx: &mut Context<O, E>, items: Vec<BatchItem>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
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
        ));
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
        inputs: items
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
        items,
        submission,
    )?;

    writeln!(ctx.out, "Batch {batch_id} successfully created.")?;
    Ok(())
}

/// Interactively fetch batch results, and update the LanceDB store and the hash cache following the
/// protocol above.
///
/// # Arguments
///
/// * `ctx` - The app's context object
/// * `client` - The client to use to interact with the batch API
/// * `batch` - Metadata about the batch to fetch results for
///
/// # Errors
///
/// * `CLIError::StateDirError` if the user's base directory could not be obtained. See
///   `directories::BaseDirError` for when this can occur.
/// * `CLIError::IOError` in the following cases:
///     * writing to the state directory failed. This is typically caused by permission issues.
///     * writing out the serialized data failed
/// * `CLIError::SerializationError` if JSON serialization failed.
#[allow(clippy::too_many_lines)]
async fn prompt_and_fetch_batch_results<O, E>(
    ctx: &mut Context<O, E>,
    client: impl BatchAPIProvider,
    batch: &BatchEmbeddingMetadata,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(&mut ctx.out, "Fetch results now? ([y]/n)")?;
    if read_char(&mut ctx.input, 'y', &['y', 'n']) != 'y' {
        return Ok(());
    }

    let batch_dir = get_state_dir()?.join("batches");

    if batch.succeeded.is_some()
        && let Some(failed_idx) = &batch.failed
    {
        let failed_items: Vec<BatchItem> = failed_idx
            .iter()
            .filter_map(|i| batch.items.get(*i).cloned())
            .collect();

        retry_items(ctx, failed_items).await?;
        fs::remove_file(batch_dir.join(format!("batch_{}.log", batch.seq_id)))?;

        return Ok(());
    }

    let batch_id = batch.batch_id.as_str();
    let results = client.get_batch_results(batch_id).await?;

    if results.succeeded.len() + results.failed.len() != batch.items.len() {
        return Err(CLIError::CommandError("Length mismatch between batch results and saved batch. The file may have been corrupted.".into()));
    }

    if batch.items.is_empty() {
        return Err(CLIError::CommandError(
            "Cannot fetch the results of an empty batch. This may be due to a corrupted file."
                .into(),
        ));
    }

    match (results.succeeded.is_empty(), results.failed.is_empty()) {
        (true, true) => {
            // The checks above early-return with an error
            unreachable!("Both succeeded and failed batches cannot be empty.");
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
            if read_char(&mut ctx.input, 'y', &['y', 'n']) == 'y' {
                retry_items(ctx, batch.items.clone()).await?;
                fs::remove_file(batch_dir.join(format!("batch_{}.log", batch.seq_id)))?;
            }
        }
        (false, true) => {
            // complete success
            handle_successful_batch_results(ctx, batch, results.succeeded).await?;

            fs::remove_file(batch_dir.join(format!("batch_{}.log", batch.seq_id)))?;
        }
        (false, false) => {
            // partial success
            writeln!(
                &mut ctx.out,
                "{} of {} items succeeded, and will be imported.",
                results.succeeded.len(),
                results.succeeded.len() + results.failed.len()
            )?;

            let ids_to_idx: HashMap<&str, usize> = batch
                .items
                .iter()
                .enumerate()
                .map(|(i, item)| (item.library_key.as_str(), i))
                .collect();

            // Collect indices first so we can move later
            let succ_idx: Vec<usize> = results
                .succeeded
                .iter()
                .filter_map(|r| ids_to_idx.get(r.id.as_str()).copied())
                .collect();
            debug_assert_eq!(succ_idx.len(), results.succeeded.len()); // implicitly assumed invariants

            let failed_idx: Vec<usize> = results
                .failed
                .iter()
                .filter_map(|r| ids_to_idx.get(r.id.as_str()).copied())
                .collect();
            debug_assert_eq!(failed_idx.len(), results.failed.len());

            handle_successful_batch_results(ctx, batch, results.succeeded).await?;

            writeln!(
                &mut ctx.err,
                "\nBelow are a few of the errors sent by the API:"
            )?;
            for err in results.failed.iter().take(3) {
                writeln!(&mut ctx.err, "  {} - {}", err.id, err.error)?;
            }

            // Persist the classification so (a) a crash between here and the retry submission
            // below doesn't lose what we've already applied to the DB, and (b) a future
            // check-status pass can short-circuit re-fetching once that's implemented.
            let mut batch_metadata = batch.clone();
            batch_metadata.succeeded = Some(succ_idx);
            batch_metadata.failed = Some(failed_idx.clone());
            fs::write(
                batch_dir.join(format!("batch_{}.log", batch.seq_id)),
                serde_json::to_string_pretty(&batch_metadata)?,
            )?;

            // Retry failed items
            writeln!(
                &mut ctx.out,
                "Do you want to retry the failed items? ([y]/n) "
            )?;
            if read_char(&mut ctx.input, 'y', &['y', 'n']) == 'y' {
                let failed_items: Vec<BatchItem> = failed_idx
                    .iter()
                    .filter_map(|i| batch.items.get(*i).cloned())
                    .collect();

                retry_items(ctx, failed_items).await?;
                fs::remove_file(batch_dir.join(format!("batch_{}.log", batch.seq_id)))?;
            }
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
                &mut ctx.input,
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

    let status = embedder.get_batch_status(&selected_batch.batch_id).await?;
    writeln!(&mut ctx.out, "Current status: {}", status.as_str())?;

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
            return Err(CLIError::CommandError(format!(
                "Could not parse library metadata: {parse_err}",
            )));
        }
    };
    let new_items: Vec<BatchItem> = new_items.into_iter().map(Into::into).collect();

    // Creating a new batch is equivalent to "retrying" with all items
    retry_items(ctx, new_items).await
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
    use std::{fs, io::Cursor};

    use chrono::{Duration, Utc};
    use serial_test::serial;
    use tempfile::tempdir;
    use zqa_macros::{test_contains, test_eq};
    use zqa_rag::capabilities::{BatchAPIProvider, BatchJobState};
    use zqa_rag::embedding::common::{
        BatchEmbeddingError, BatchEmbeddingRequest, BatchEmbeddingResult, BatchEmbeddingResults,
        BatchSubmission,
    };
    use zqa_rag::llm::errors::LLMError;
    use zqa_rag::providers::ProviderId;

    use super::{
        BatchEmbeddingMetadata, BatchItem, CacheEntry, HashCache, get_seq_id,
        handle_batch_check_status_cmd, prompt_and_fetch_batch_results, update_hash_cache,
        write_batch_metadata,
    };
    use crate::cli::app::tests::create_test_context;
    use crate::utils::library::{ZoteroItem, ZoteroItemMetadata};

    /// A [`CacheEntry`] for a VoyageAI batch with the given model and sequence number.
    fn entry(model: &str, seq: usize) -> CacheEntry {
        CacheEntry {
            provider_id: ProviderId::VoyageAI,
            model: model.into(),
            seq_id: seq,
        }
    }

    #[test]
    fn update_hash_cache_inserts_new_items() {
        // `make_metadata_at` gives items with hashes 0..count and model "voyage-3".
        let mut batch = make_metadata_at(Utc::now(), 2);
        batch.seq_id = 5;

        let updated =
            update_hash_cache(HashCache::new(), &batch, batch.items.iter().map(|i| i.hash));

        for item in &batch.items {
            let e = updated.get(&item.hash).expect("new item should be cached");
            test_eq!(e.seq_id, 5);
            test_eq!(e.provider_id, ProviderId::VoyageAI);
            test_eq!(e.model, "voyage-3");
        }
    }

    #[test]
    fn update_hash_cache_bumps_seq_for_duplicate() {
        let mut batch = make_metadata_at(Utc::now(), 1);
        batch.seq_id = 5;
        let h = batch.items[0].hash;

        let mut cache = HashCache::new();
        cache.insert(h, entry("voyage-3", 2)); // older, same provider/model

        // Nothing to insert (it's a duplicate), but the seq must advance to the newer batch so
        // a later overlapping batch is correctly judged older.
        let updated = update_hash_cache(cache, &batch, std::iter::empty());

        test_eq!(updated.get(&h).unwrap().seq_id, 5);
    }

    #[test]
    fn update_hash_cache_does_not_lower_seq() {
        let mut batch = make_metadata_at(Utc::now(), 1);
        batch.seq_id = 5;
        let h = batch.items[0].hash;

        let mut cache = HashCache::new();
        cache.insert(h, entry("voyage-3", 9)); // newer than this batch

        let updated = update_hash_cache(cache, &batch, std::iter::empty());

        // A newer batch already owns this hash; an older one must not claw it back.
        test_eq!(updated.get(&h).unwrap().seq_id, 9);
    }

    #[test]
    fn update_hash_cache_overwrites_on_model_change() {
        let mut batch = make_metadata_at(Utc::now(), 1);
        batch.seq_id = 5;
        let h = batch.items[0].hash;

        let mut cache = HashCache::new();
        cache.insert(h, entry("old-model", 9)); // model mismatch

        // On a model change the item is re-embedded (so its hash is in `new_hashes`); the new
        // batch wins regardless of the old, higher seq.
        let updated = update_hash_cache(cache, &batch, [h]);

        let e = updated.get(&h).unwrap();
        test_eq!(e.model, "voyage-3");
        test_eq!(e.seq_id, 5);
    }

    #[test]
    fn update_hash_cache_leaves_unrelated_entries() {
        let mut batch = make_metadata_at(Utc::now(), 1);
        batch.seq_id = 5;

        // Hashes for this batch are 0..1, so this entry belongs to some other batch.
        let unrelated_hash = 999_999_u64;
        let mut cache = HashCache::new();
        cache.insert(unrelated_hash, entry("voyage-3", 2));

        let updated = update_hash_cache(cache, &batch, std::iter::empty());

        let e = updated.get(&unrelated_hash).unwrap();
        test_eq!(e.seq_id, 2); // untouched
        test_eq!(e.model, "voyage-3");
    }

    /// A stand-in [`BatchAPIProvider`] that returns canned results, so the fetch and branch logic
    /// in [`prompt_and_fetch_batch_results`] can be exercised without a real provider or network.
    /// Injecting this is exactly what the `BatchAPIProvider`-trait refactor enabled.
    struct MockBatchProvider {
        results: BatchEmbeddingResults,
    }

    impl BatchAPIProvider for MockBatchProvider {
        async fn submit_batch(
            &self,
            _request: BatchEmbeddingRequest,
        ) -> Result<BatchSubmission, LLMError> {
            Ok(make_submission("mock-batch"))
        }

        async fn get_batch_status(&self, _batch_id: &str) -> Result<BatchJobState, LLMError> {
            Ok(BatchJobState::Completed)
        }

        async fn get_batch_results(
            &self,
            _batch_id: &str,
        ) -> Result<BatchEmbeddingResults, LLMError> {
            Ok(self.results.clone())
        }
    }

    /// Build a [`BatchEmbeddingResults`] from the ids that should land in each bucket. Embeddings
    /// are zero vectors — the branch logic under test never inspects their contents.
    fn make_results(succeeded_ids: &[&str], failed_ids: &[&str]) -> BatchEmbeddingResults {
        BatchEmbeddingResults {
            succeeded: succeeded_ids
                .iter()
                .map(|id| BatchEmbeddingResult {
                    id: (*id).into(),
                    embedding: vec![0.0_f32; 8],
                })
                .collect(),
            failed: failed_ids
                .iter()
                .map(|id| BatchEmbeddingError {
                    id: (*id).into(),
                    error: "mock failure".into(),
                })
                .collect(),
        }
    }

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
            test_eq!(get_seq_id().unwrap(), 1);
        });
    }

    #[test]
    fn get_seq_id_returns_one_for_empty_dir() {
        let tmp = tempdir().unwrap();
        let batch_dir = tmp.path().join("batches");
        fs::create_dir_all(&batch_dir).unwrap();

        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            test_eq!(get_seq_id().unwrap(), 1);
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
            test_eq!(get_seq_id().unwrap(), 4);
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
            test_eq!(get_seq_id().unwrap(), 1);
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
            test_eq!(get_seq_id().unwrap(), 2);
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
            test_eq!(files.len(), 1);

            let content = fs::read_to_string(files[0].path()).unwrap();
            let meta: BatchEmbeddingMetadata = serde_json::from_str(&content).unwrap();

            test_eq!(meta.batch_id, "my-batch-id");
            test_eq!(meta.provider, ProviderId::VoyageAI);
            test_eq!(meta.model, "voyage-3");
            test_eq!(meta.items.len(), 2);
            test_eq!(meta.items[0].library_key, "KEY-A");
            test_eq!(meta.items[0].hash, 1);
            test_eq!(meta.items[1].library_key, "KEY-B");
            test_eq!(meta.items[1].hash, 2);
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
                test_contains!(err, "No batches have been submitted");
            },
        )
        .await;
    }

    #[test]
    fn from_zotero_item_maps_all_fields() {
        let zi = ZoteroItem {
            metadata: ZoteroItemMetadata {
                library_key: "KEY-1".into(),
                title: "Some Title".into(),
                file_path: std::path::PathBuf::from("/tmp/x.pdf"),
                authors: Some(vec!["Author A".into()]),
            },
            text: "body text".into(),
        };

        let bi: BatchItem = zi.into();
        test_eq!(bi.library_key, "KEY-1");
        test_eq!(bi.title, "Some Title");
        test_eq!(bi.file_path, std::path::PathBuf::from("/tmp/x.pdf"));
        test_eq!(bi.text, "body text");
    }

    fn make_zotero_item(text: &str) -> ZoteroItem {
        ZoteroItem {
            metadata: ZoteroItemMetadata {
                library_key: "K".into(),
                title: "T".into(),
                file_path: std::path::PathBuf::from("/tmp/p.pdf"),
                authors: None,
            },
            text: text.into(),
        }
    }

    #[test]
    fn from_zotero_item_hash_is_deterministic() {
        let a: BatchItem = make_zotero_item("identical text").into();
        let b: BatchItem = make_zotero_item("identical text").into();
        test_eq!(a.hash, b.hash);
    }

    #[test]
    fn from_zotero_item_hash_distinguishes_text() {
        let a: BatchItem = make_zotero_item("one").into();
        let b: BatchItem = make_zotero_item("two").into();
        assert_ne!(a.hash, b.hash);
    }

    fn make_metadata_at(when: chrono::DateTime<Utc>, items_count: usize) -> BatchEmbeddingMetadata {
        let items = (0..items_count)
            .map(|i| BatchItem {
                file_path: std::path::PathBuf::from(format!("/tmp/{i}.pdf")),
                library_key: format!("K{i}"),
                title: format!("T{i}"),
                text: format!("text {i}"),
                hash: i as u64,
            })
            .collect();
        BatchEmbeddingMetadata {
            seq_id: 1,
            batch_id: "bid".into(),
            file_id: "fid".into(),
            provider: ProviderId::VoyageAI,
            model: "voyage-3".into(),
            created_at: when,
            items,
            succeeded: None,
            failed: None,
        }
    }

    #[test]
    fn display_includes_provider_model_and_count() {
        let meta = make_metadata_at(Utc::now() - Duration::seconds(30), 3);
        let s = format!("{meta}");

        test_contains!(s, "voyageai");
        test_contains!(s, "voyage-3");
        test_contains!(s, "3 items");
    }

    #[test]
    fn display_handles_future_created_at() {
        // A negative duration can't be converted to `std::time::Duration`, so the formatter
        // falls back to the "Unknown" branch.
        let meta = make_metadata_at(Utc::now() + Duration::days(365), 0);
        let s = format!("{meta}");
        assert!(s.contains("Unknown"), "expected unknown-time fallback: {s}");
    }

    #[test]
    fn get_seq_id_handles_gaps_in_sequence() {
        let tmp = tempdir().unwrap();
        let batch_dir = tmp.path().join("batches");
        fs::create_dir_all(&batch_dir).unwrap();
        // Gaps from deleted batches: next seq should be (max + 1), not (count + 1).
        fs::write(batch_dir.join("batch_1.log"), "").unwrap();
        fs::write(batch_dir.join("batch_3.log"), "").unwrap();
        fs::write(batch_dir.join("batch_7.log"), "").unwrap();

        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            test_eq!(get_seq_id().unwrap(), 8);
        });
    }

    #[test]
    fn metadata_round_trips_with_indices_populated() {
        let mut meta = make_metadata_at(Utc::now() - Duration::seconds(10), 4);
        meta.succeeded = Some(vec![0, 2]);
        meta.failed = Some(vec![1, 3]);

        let json = serde_json::to_string_pretty(&meta).unwrap();
        let round_tripped: BatchEmbeddingMetadata = serde_json::from_str(&json).unwrap();

        test_eq!(round_tripped.succeeded.as_deref(), Some(&[0usize, 2][..]));
        test_eq!(round_tripped.failed.as_deref(), Some(&[1usize, 3][..]));
        test_eq!(round_tripped.items.len(), 4);
        test_eq!(round_tripped.batch_id, "bid");
        test_eq!(round_tripped.provider, ProviderId::VoyageAI);
    }

    #[tokio::test]
    #[serial]
    async fn fetch_results_declined_returns_early() {
        let tmp = tempdir().unwrap();
        temp_env::async_with_vars(
            [("ZQA_STATE_DIR", Some(tmp.path().to_str().unwrap()))],
            async {
                let mut ctx = create_test_context();
                // Decline the very first "Fetch results now?" prompt.
                ctx.input = Box::new(Cursor::new(b"n\n".to_vec()));

                let batch = make_metadata_at(Utc::now() - Duration::seconds(5), 2);
                // The provider would return a mismatched (empty) payload, but declining must
                // short-circuit before it is ever consulted.
                let mock = MockBatchProvider {
                    results: make_results(&[], &[]),
                };

                let result = prompt_and_fetch_batch_results(&mut ctx, mock, &batch).await;
                assert!(result.is_ok());

                let out = String::from_utf8(ctx.out.into_inner()).unwrap();
                test_contains!(out, "Fetch results now?");
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn fetch_results_errors_on_length_mismatch() {
        let tmp = tempdir().unwrap();
        temp_env::async_with_vars(
            [("ZQA_STATE_DIR", Some(tmp.path().to_str().unwrap()))],
            async {
                let mut ctx = create_test_context();
                ctx.input = Box::new(Cursor::new(b"y\n".to_vec()));

                // The batch has 3 items, but the provider returns results for only 2. The
                // length-mismatch guard must reject before any DB write or retry.
                let batch = make_metadata_at(Utc::now() - Duration::seconds(5), 3);
                let mock = MockBatchProvider {
                    results: make_results(&["K0"], &["K1"]),
                };

                let result = prompt_and_fetch_batch_results(&mut ctx, mock, &batch).await;
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
    async fn fetch_results_errors_on_empty_batch() {
        let tmp = tempdir().unwrap();
        temp_env::async_with_vars(
            [("ZQA_STATE_DIR", Some(tmp.path().to_str().unwrap()))],
            async {
                let mut ctx = create_test_context();
                ctx.input = Box::new(Cursor::new(b"y\n".to_vec()));

                // 0 items + 0 results passes the length check (0 == 0) but trips the empty-batch
                // guard right after it.
                let batch = make_metadata_at(Utc::now() - Duration::seconds(5), 0);
                let mock = MockBatchProvider {
                    results: make_results(&[], &[]),
                };

                let result = prompt_and_fetch_batch_results(&mut ctx, mock, &batch).await;
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
    async fn fetch_results_all_failed_declining_retry_keeps_log() {
        let tmp = tempdir().unwrap();
        let batch_dir = tmp.path().join("batches");
        fs::create_dir_all(&batch_dir).unwrap();
        // `make_metadata_at` uses seq_id 1, so this is the batch's own WAL entry.
        fs::write(batch_dir.join("batch_1.log"), "placeholder").unwrap();

        temp_env::async_with_vars(
            [("ZQA_STATE_DIR", Some(tmp.path().to_str().unwrap()))],
            async {
                let mut ctx = create_test_context();
                // 'y' to fetch, then 'n' to decline retrying the whole batch.
                ctx.input = Box::new(Cursor::new(b"y\nn\n".to_vec()));

                let batch = make_metadata_at(Utc::now() - Duration::seconds(5), 2);
                let mock = MockBatchProvider {
                    results: make_results(&[], &["K0", "K1"]),
                };

                let result = prompt_and_fetch_batch_results(&mut ctx, mock, &batch).await;
                assert!(result.is_ok());

                let err = String::from_utf8(ctx.err.into_inner()).unwrap();
                test_contains!(err, "All 2 batch items failed");
                // Declining the retry must leave the WAL entry untouched — nothing was applied,
                // so there is nothing to clean up.
                assert!(batch_dir.join("batch_1.log").exists());
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
                test_contains!(err, "No valid batches were found");
            },
        )
        .await;
    }
}
