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
//!   * This file is shared across batches.
//!   * The file contains a map from text hashes to corresponding provider ID, provider model, and
//!     the sequence number of the (newest) batch it belonged to.
//!   * Before writing to the DB, we filter out items that match the hash, provider, and provider
//!     model.
//!   * After writing to the DB, we add the hashes of the texts added to this cache along with that
//!     same auxiliary information.
//!
//! For the structure of this file, see [`BatchEmbeddingMetadata`].

use std::{
    fmt::Display,
    fs,
    hash::{DefaultHasher, Hash, Hasher},
    io::{self, Write},
    path::Path,
};

use chrono::{DateTime, Utc};
use humantime::format_duration;
use serde::{Deserialize, Serialize};
use zqa_rag::{
    embedding::common::{BatchEmbeddingInput, BatchEmbeddingRequest, BatchSubmission},
    providers::{ProviderId, registry::provider_registry},
};

use crate::{
    cli::{commands::BatchCommand, errors::CLIError},
    common::Context,
    state::read_number,
};
use crate::{state::get_state_dir, utils::library::parse_library};

/// Metadata about a batch embedding request. This is the file structure used to represent a batch
/// in progress.
#[derive(Debug, Serialize, Deserialize)]
struct BatchEmbeddingMetadata {
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
    /// Hashes of each text in this batch. Since our texts can be quite large, and we only need
    /// equality tests, a hash is a simpler solution.
    hashes: Vec<String>,
    /// Optional indices to the hashes above if we know this information (by checking).
    succeeded: Option<Vec<usize>>,
    /// Optional indices to the hashes above if we know this information (by checking).
    failed: Option<Vec<usize>>,
}

impl Display for BatchEmbeddingMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elapsed = (self.created_at - Utc::now())
            .to_std()
            .map_or("Unknown time ago".into(), |d| {
                format_duration(d).to_string()
            });

        f.write_fmt(format_args!(
            "({}) {} ({}) - {} items\n",
            elapsed,
            self.provider.as_str(),
            self.model,
            self.hashes.len()
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
/// * `hashes` - Hashes for the texts in this batch
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
    hashes: Vec<String>,
    submission: BatchSubmission,
) -> Result<(), CLIError> {
    let batch_dir = get_state_dir()?.join("batches");
    if !batch_dir.exists() {
        fs::create_dir_all(&batch_dir)?;
    }

    let metadata = BatchEmbeddingMetadata {
        batch_id: submission.batch_id,
        file_id: submission.file_id,
        provider,
        model,
        created_at: Utc::now(),
        hashes,
        succeeded: None,
        failed: None,
    };

    let seq = get_seq_id()?;
    let filename = format!("batch_{seq}.log");
    let contents = serde_json::to_string_pretty(&metadata)?;
    fs::write(batch_dir.join(filename), contents)?;

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

            batches.iter().enumerate().for_each(|(i, batch)| {
                _ = writeln!(&mut ctx.out, "{}. {}", i + 1, batch);
            });
            let choice = read_number(
                &mut io::stdin().lock(),
                1,
                (1, u8::try_from(batches.len()).unwrap() + 1), // we don't expect > 255 batches
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

    let hashes = new_items
        .iter()
        .map(|item| {
            // TODO: Consider using `sha2`, `fxhash`, or similar
            let mut hasher = DefaultHasher::new();
            item.text.hash(&mut hasher);

            hasher.finish().to_string()
        })
        .collect::<Vec<_>>();

    let request = BatchEmbeddingRequest {
        model: cfg.model_name().into(),
        dims: cfg.dims(),
        inputs: new_items
            .into_iter()
            .map(|item| BatchEmbeddingInput {
                id: item.metadata.library_key,
                text: item.text,
            })
            .collect(),
    };

    let submission = embedder.submit_batch(request).await?;
    let batch_id = submission.batch_id.clone();
    write_batch_metadata(
        cfg.provider_id(),
        cfg.model_name().into(),
        hashes,
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
        BatchCommand::FetchResults => todo!(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serial_test::serial;
    use tempfile::tempdir;
    use zqa_rag::{embedding::common::BatchSubmission, providers::ProviderId};

    use super::{
        BatchEmbeddingMetadata, get_seq_id, handle_batch_check_status_cmd, write_batch_metadata,
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
        let hashes = vec!["hash-a".to_string(), "hash-b".to_string()];

        temp_env::with_var("ZQA_STATE_DIR", Some(tmp.path()), || {
            write_batch_metadata(
                ProviderId::VoyageAI,
                "voyage-3".into(),
                hashes.clone(),
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
            assert_eq!(meta.hashes, hashes);
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
