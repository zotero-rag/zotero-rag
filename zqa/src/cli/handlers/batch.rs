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
//!   [`crate::state`] for an implementation). This means:
//!   * A file existing, and being valid (in its contents) implies that that batch exists. Whether
//!     it *actually* exists or is invalid data (e.g., it contains a batch ID that doesn't exist) is
//!     not guaranteed.
//!   * A file *not* existing means that as far as we are concerned, that batch wasn't dispatched by
//!     us; not our circus, not our monkeys.
//!   * The origin of the file is irrelevant. Users are free to create a file in that directory, and
//!     we will treat it as valid if the structure is right.
//!   * Files are named `batch_<id>.log`, where `id` are sequence numbers to avoid dealing with
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
//!     at startup (as opposed to by user request). We add these to our backend and remove this batch.
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

use std::io::Write;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use zqa_rag::providers::ProviderId;

use crate::{
    cli::{commands::BatchCommand, errors::CLIError},
    common::Context,
};

/// Metadata about a batch embedding request. This is the file structure used to represent a batch
/// in progress.
#[derive(Debug, Serialize, Deserialize)]
struct BatchEmbeddingMetadata {
    /// The batch ID
    batch_id: String,
    /// The batch embedding provider
    provider: ProviderId,
    /// Date of creation, from the API (as opposed to local time). Used to give users context when
    /// referring to a batch.
    created_at: NaiveDate,
    /// Hashes of each text in this batch. Since our texts can be quite large, and we only need
    /// equality tests, a hash is a simpler solution.
    hashes: Vec<String>,
    /// Optional indices to the hashes above if we know this information (by checking).
    succeeded: Option<Vec<usize>>,
    /// Optional indices to the hashes above if we know this information (by checking).
    failed: Option<Vec<usize>>,
}

/// Handle the `/batch` commands.
pub(crate) async fn handle_batch_cmd<O, E>(
    _subcmd: BatchCommand,
    _ctx: &mut Context<O, E>,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    Ok(())
}
