use std::collections::HashSet;
use std::fmt::Write;
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant, SystemTime};
use std::{env, thread};

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use directories::UserDirs;
use rusqlite::{Connection, ErrorCode, OpenFlags};
use serde::Serialize;
use thiserror::Error;
use tokio::sync::mpsc;
use zqa_pdftools::parse::extract_text;

use crate::io::EngineEvent;
use crate::izip;
use crate::store::common::ZoteroStore;
use crate::utils::arrow::DbFields;

/// Gets the Zotero library path. Works on Linux, macOS, and Windows systems.
/// On CI environments, returns a location to a toy library in assets/ instead.
///
/// Returns None if either the OS is not one of the above, or if we could not
/// get the directory.
fn get_lib_path() -> Option<PathBuf> {
    if env::var("CI").is_ok() {
        let assets_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("Zotero");

        if assets_dir.exists() {
            return Some(assets_dir);
        }
        return None;
    }

    match env::consts::OS {
        "linux" | "macos" | "windows" => {
            UserDirs::new().map(|user_dirs| PathBuf::from(user_dirs.home_dir()).join("Zotero"))
        }
        _ => None,
    }
}

/// Resolves the Zotero library path, preferring an explicit `override_path` and otherwise falling
/// back to [`get_lib_path`]'s environment-based resolution.
///
/// The override is how tests point at the isolated toy library under `assets/` without setting the
/// process-global `CI` env var, which is what lets them avoid `#[serial]`.
fn resolve_lib_path(override_path: Option<&Path>) -> Option<PathBuf> {
    override_path.map(Path::to_path_buf).or_else(get_lib_path)
}

/// The magic number at the start of an active SQLite rollback journal. See
/// <https://sqlite.org/fileformat.html#the_rollback_journal>.
const JOURNAL_MAGIC: [u8; 8] = [0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7];

/// How many times to attempt copying a database that Zotero keeps writing to. The attempts span
/// 4.5 s, well past [`MTIME_RESOLUTION`], so a single write just before the first attempt does not
/// exhaust them.
const UNLOCKED_READ_ATTEMPTS: u32 = 10;

/// How long to wait between copy attempts.
const UNLOCKED_READ_RETRY_DELAY: Duration = Duration::from_millis(500);

/// The coarsest modification-time resolution we expect. HFS+ records mtimes to 1 s, and FAT and
/// exFAT to 2 s.
const MTIME_RESOLUTION: Duration = Duration::from_secs(2);

// The retries must outlast the recent-write guard in `db_file_state`.
const _: () = assert!(
    UNLOCKED_READ_RETRY_DELAY.as_millis() * (UNLOCKED_READ_ATTEMPTS as u128 - 1)
        > MTIME_RESOLUTION.as_millis()
);

/// Runs `read` against `zotero.sqlite` in the Zotero library directory `lib_path`.
///
/// A running Zotero holds an exclusive lock on its database, so reads through a normal connection
/// fail with `SQLITE_BUSY`. In that case, this copies the database, and its write-ahead log if
/// there is one, to a temporary directory and runs `read` again on the copy. Zotero 10 keeps
/// recent commits in the write-ahead log until it checkpoints, so reading the main file alone
/// would miss them.
///
/// The copy is taken without locks, so it only counts if no rollback-journal write was in
/// progress and the main file did not change while it was copied (see [`db_file_state`]).
/// Otherwise it is retried a few times. A write-ahead log copied mid-append is fine: SQLite only
/// replays frames up to the last complete commit.
///
/// # Arguments
///
/// * `lib_path` - The Zotero library directory, containing `zotero.sqlite`.
/// * `read` - The read to run. It may run twice, so it must be safe to repeat. It must return lock
///   errors (see [`is_lock_error`]) rather than skipping them.
///
/// # Returns
///
/// The result of `read`.
///
/// # Errors
///
/// * `LibraryParsingError::SqlError` if the database cannot be opened or copied, `read` fails for
///   a reason other than the lock, or Zotero kept writing to the database through every attempt.
fn with_zotero_db<T>(
    lib_path: &Path,
    mut read: impl FnMut(&Connection) -> Result<T, rusqlite::Error>,
) -> Result<T, LibraryParsingError> {
    let db_path = lib_path.join("zotero.sqlite");
    // Not `SQLITE_OPEN_READ_ONLY`: if Zotero crashed mid-write, SQLite needs write access to roll
    // back the leftover journal before anything can read the database.
    let conn = Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;

    // Zotero never releases its lock while running, so waiting on it only delays the fallback.
    conn.busy_timeout(Duration::ZERO)?;
    match read(&conn) {
        Err(e) if is_lock_error(&e) => {}
        result => return Ok(result?),
    }

    log::debug!(
        "{} is locked, likely by a running Zotero; reading a copy",
        db_path.display()
    );
    let mut wal_path = db_path.as_os_str().to_owned();
    wal_path.push("-wal");
    let copy_dir = tempfile::tempdir()?;
    let copy_path = copy_dir.path().join("zotero.sqlite");
    let copy_wal_path = copy_dir.path().join("zotero.sqlite-wal");
    for attempt in 1..=UNLOCKED_READ_ATTEMPTS {
        if let Some(before) = db_file_state(&db_path)? {
            std::fs::copy(&db_path, &copy_path)?;
            match std::fs::copy(&wal_path, &copy_wal_path) {
                Ok(_) => {}
                // Don't let a log copied by an earlier attempt replay over this copy.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    match std::fs::remove_file(&copy_wal_path) {
                        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                            return Err(e.into());
                        }
                        _ => {}
                    }
                }
                Err(e) => return Err(e.into()),
            }
            if db_file_state(&db_path)? == Some(before) {
                return Ok(read(&Connection::open(&copy_path)?)?);
            }
        }
        if attempt < UNLOCKED_READ_ATTEMPTS {
            thread::sleep(UNLOCKED_READ_RETRY_DELAY);
        }
    }

    Err(LibraryParsingError::SqlError(
        "Zotero is busy writing to its database. Try again in a moment.".into(),
    ))
}

/// Describes the database file at `db_path`, so that callers can tell whether it changed.
///
/// # Arguments
///
/// * `db_path` - Path to the SQLite database file.
///
/// # Returns
///
/// `None` if a write transaction is in progress, judged by whether the rollback journal starts
/// with [`JOURNAL_MAGIC`], or if the file was modified within [`MTIME_RESOLUTION`]. Otherwise, the
/// file's modification time and length.
///
/// The second condition makes sure that a later write changes the modification time even on a
/// filesystem with coarse mtimes. SQLite's file change counter can't replace it: in exclusive
/// locking mode, which Zotero uses, the counter is not incremented on each transaction.
///
/// # Errors
///
/// * `std::io::Error` if the journal exists but cannot be read, or the database file's metadata
///   cannot be read.
fn db_file_state(db_path: &Path) -> Result<Option<(SystemTime, u64)>, std::io::Error> {
    let mut journal_path = db_path.as_os_str().to_owned();
    journal_path.push("-journal");

    // An idle journal is missing, empty, or has a zeroed header, depending on the journal mode.
    let mut header = [0; JOURNAL_MAGIC.len()];
    match File::open(&journal_path) {
        Ok(mut journal) => {
            if journal.read_exact(&mut header).is_ok() && header == JOURNAL_MAGIC {
                return Ok(None);
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    let metadata = std::fs::metadata(db_path)?;
    let modified = metadata.modified()?;
    // An mtime in the future (clock skew) also counts as recent.
    if modified
        .elapsed()
        .map_or(true, |age| age < MTIME_RESOLUTION)
    {
        return Ok(None);
    }
    Ok(Some((modified, metadata.len())))
}

/// Whether `e` means another connection, such as a running Zotero, holds a lock on the database.
fn is_lock_error(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(e, _)
            if matches!(e.code, ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
    )
}

/// Metadata for items in the Zotero library.
#[derive(Debug, Clone, Serialize)]
pub struct ZoteroItemMetadata {
    pub library_key: String,
    pub title: String,
    pub file_path: PathBuf,
    pub authors: Option<Vec<String>>,
}

impl PartialEq for ZoteroItemMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.library_key == other.library_key && self.title == other.title
    }
}

impl Eq for ZoteroItemMetadata {}

impl Hash for ZoteroItemMetadata {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(self.library_key.as_bytes());
        state.write(self.title.as_bytes());
    }
}

/// A Zotero library item. Includes full-text from parsing PDFs when they exist.
#[derive(Clone, Serialize)]
pub struct ZoteroItem {
    pub metadata: ZoteroItemMetadata,
    pub text: String,
}

// A convenience struct that represents a set of Zotero library items; this helps get around
// coherence rules so we can implement `From<Vec<RecordBatch>>`.
#[derive(Clone)]
pub struct ZoteroItemSet {
    pub items: Vec<ZoteroItem>,
}

impl From<ZoteroItemSet> for Vec<ZoteroItem> {
    fn from(value: ZoteroItemSet) -> Self {
        value.items
    }
}

impl From<Vec<ZoteroItem>> for ZoteroItemSet {
    fn from(value: Vec<ZoteroItem>) -> Self {
        Self { items: value }
    }
}

impl From<Vec<RecordBatch>> for ZoteroItemSet {
    fn from(batches: Vec<RecordBatch>) -> Self {
        batches
            .iter()
            .flat_map(|batch| {
                let schema = batch.schema();
                let key_idx = schema.index_of(DbFields::LibraryKey.as_ref()).unwrap();
                let title_idx = schema.index_of(DbFields::Title.as_ref()).unwrap();
                let file_path_idx = schema.index_of(DbFields::FilePath.as_ref()).unwrap();
                let text_idx = schema.index_of(DbFields::PdfText.as_ref()).unwrap();

                let lib_keys = get_column_from_batch(batch, key_idx);
                let titles = get_column_from_batch(batch, title_idx);
                let file_paths = get_column_from_batch(batch, file_path_idx);
                let texts = get_column_from_batch(batch, text_idx);

                let zipped = izip!(lib_keys, titles, file_paths, texts);
                let items_batch: Vec<ZoteroItem> = zipped
                    .map(|(lib_key, title, file_path, text)| ZoteroItem {
                        metadata: ZoteroItemMetadata {
                            library_key: lib_key,
                            title,
                            file_path: PathBuf::from(file_path),
                            authors: None,
                        },
                        text,
                    })
                    .collect();

                items_batch
            })
            .collect::<Vec<_>>()
            .into()
    }
}

/// A general error struct for Zotero library parsing.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum LibraryParsingError {
    #[error("SQLite error: {0}")]
    SqlError(String),
    #[error("LanceDB error when parsing library: {0}")]
    LanceDBError(String),
    #[error("PDF parsing error: {0}")]
    PdfParsingError(String),
}

impl From<std::io::Error> for LibraryParsingError {
    fn from(e: std::io::Error) -> Self {
        LibraryParsingError::SqlError(e.to_string())
    }
}

impl From<rusqlite::Error> for LibraryParsingError {
    fn from(e: rusqlite::Error) -> Self {
        LibraryParsingError::SqlError(e.to_string())
    }
}

impl From<Box<dyn std::error::Error>> for LibraryParsingError {
    fn from(value: Box<dyn std::error::Error>) -> Self {
        LibraryParsingError::PdfParsingError(value.to_string())
    }
}

/// From a `RecordBatch`, return all values from a specified column as a `Vec<String>`.
#[must_use]
pub(crate) fn get_column_from_batch(batch: &RecordBatch, column: usize) -> Vec<String> {
    let results = batch.column(column).as_string::<i32>();

    results
        .iter()
        .filter_map(|s| Some(s?.to_string()))
        .collect()
}

/// Assuming an existing `LanceDB` database exists, returns a list of items present in the Zotero
/// library but not in the database. The primary use case for this is to update the DB with new
/// items. Note that this does not take into account removed items.
///
/// # Arguments:
///
/// * `embedding_config` - The embedding provider configuration for the configured `LanceDB` embedding.
///
/// # Returns
///
/// If successful, a list of `ZoteroItemMetadata` objects corresponding to new items.
///
/// # Errors
///
/// * `LibraryParsingError::SqliteError` if the library path was not found, the query could not be prepared, or
///   columns from the result set could not be parsed, or `query_map` fails.
/// * `LibraryParsingError::LanceDBError` if fetching the rows from LanceDB fails.
pub async fn get_new_library_items<T: ZoteroStore>(
    store: &T,
    library_path: Option<&Path>,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    let metadata_vecs = store
        .existing_item_metadata()
        .await
        .map_err(|e| LibraryParsingError::LanceDBError(e.to_string()))?;

    let library_items = parse_library_metadata(library_path, None, None)?;
    let library_count = library_items.len();

    let db_items_set: HashSet<_> = metadata_vecs.iter().collect();

    let new_items = library_items
        .into_iter()
        .filter(|item| !db_items_set.contains(item))
        .collect::<Vec<_>>();
    log::debug!(
        "Library filtering: metadata_items={library_count}, existing_items={}, new_items={}",
        metadata_vecs.len(),
        new_items.len()
    );
    Ok(new_items)
}

/// Parses the Zotero library metadata. If successful, returns a list of metadata for each item.
///
/// # Arguments
///
/// * `library_path` - An optional override for the Zotero library directory. When `None`, the path
///   is resolved from the environment (see [`get_lib_path`]).
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
///
/// # Errors
///
/// * `LibraryParsingError::SqliteError` if the library path was not found, the query could not be prepared, or
///   columns from the result set could not be parsed, or `query_map` fails.
pub fn parse_library_metadata(
    library_path: Option<&Path>,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    if let Some(path) = resolve_lib_path(library_path) {
        log::debug!(
            "Reading Zotero metadata: path={}, offset={start_from:?}, limit={limit:?}",
            path.display()
        );
        let mut query = "SELECT DISTINCT
                idv.value AS title,
                ia.path AS filePath,
                i2.key AS libraryKey
            FROM items i
            JOIN itemData id ON i.itemID = id.itemID
            JOIN fields f ON id.fieldID = f.fieldID
            JOIN itemDataValues idv ON id.valueID = idv.valueID
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            LEFT JOIN itemAttachments ia ON i.itemID = ia.parentItemID
            JOIN items i2 ON ia.itemID = i2.itemID
            WHERE f.fieldName = 'title'
            AND ia.contentType = 'application/pdf'
            AND it.typeName IN ('conferencePaper', 'journalArticle', 'preprint') "
            .to_string();

        // Useful for debugging
        if let Some(limit_val) = limit {
            let _ = write!(query, " LIMIT {limit_val}");
        }

        if let Some(offset) = start_from {
            let _ = write!(query, " OFFSET {offset}");
        }

        let item_iter = with_zotero_db(&path, |conn| {
            let mut stmt = conn.prepare(&query)?;
            let rows = stmt.query_map([], |row| {
                let res_path: String = row.get(1)?;
                let split_idx = res_path.find(':').unwrap_or(0);
                let filename = res_path.split_at(split_idx + 1).1;
                let lib_key: String = row.get(2)?;

                Ok(ZoteroItemMetadata {
                    library_key: lib_key.clone(),
                    title: row.get(0)?,
                    file_path: path.join("storage").join(lib_key).join(filename),
                    authors: None,
                })
            })?;

            let mut items = Vec::new();
            for row in rows {
                match row {
                    Ok(item) => items.push(item),
                    Err(e) if is_lock_error(&e) => return Err(e),
                    Err(e) => log::debug!("Skipping invalid Zotero metadata row: {e}"),
                }
            }
            Ok(items)
        })?;
        log::debug!("Read {} Zotero metadata items", item_iter.len());
        Ok(item_iter)
    } else {
        Err(LibraryParsingError::SqlError(
            "Library not found!".to_string(),
        ))
    }
}

/// Given a set of `items`, set the authors metadata in-place.
///
/// # Arguments
///
/// * `items` - The items whose metadata needs to be filled in.
/// * `library_path` - An optional override for the Zotero library directory. When `None`, the path
///   is resolved from the environment (see [`get_lib_path`]).
///
/// # Errors
///
/// * `LibraryParsingError::SqlError` if the operation failed for any items.
pub fn get_authors(
    items: &mut [ZoteroItem],
    library_path: Option<&Path>,
) -> Result<(), LibraryParsingError> {
    if let Some(path) = resolve_lib_path(library_path) {
        // For some reason, the `key` field in the `items` table (what we call `library_key`) seems
        // to not completely be a key. Specifically, there are separate keys per item depending on
        // whether you want the file path or the authors; the item metadata is stored with the file
        // path, but to get the authors, you need a different key. To save you some debugging
        // effort, here are some SQL queries that are useful here:
        //
        // ```sql
        // SELECT DISTINCT
        //   ia.path AS filePath,
        //   i.key AS authorKey,
        //   i2.key AS filePathKey
        // FROM items i
        // LEFT JOIN itemAttachments ia ON i.itemID = ia.parentItemID
        // JOIN items i2 ON ia.itemID = i2.itemID;
        // ```
        // This gives you the file path and the `key`s associated with that item's author and file path.
        //
        // ```sql
        //  SELECT c.firstName, c.lastName
        //    FROM items i
        //    JOIN itemData id ON i.itemID = id.itemID
        //    JOIN fields f ON id.fieldID = f.fieldID
        //    JOIN itemCreators ic ON i.itemID = ic.itemID
        //    JOIN creators c ON ic.creatorID = c.creatorID
        //    WHERE i.key = 'RQRBISX9'
        //    AND f.fieldName = 'title'
        //    ORDER BY ic.orderIndex;
        // ```
        // For a given (author-associated) key, this gives you the ordered first and last name pairs of
        // the authors for that paper.
        //
        // The query below basically just uses the ideas from the above two queries, except that it
        // does away with having to deal with one row per author per paper by using `GROUP_CONCAT`.

        let query = "
            SELECT GROUP_CONCAT(c.lastName || ', ' || c.firstName, ';') AS authors
            FROM items i
            LEFT JOIN itemAttachments ia ON i.itemID = ia.parentItemID
            JOIN items i2 ON ia.itemID = i2.itemID
            LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
            LEFT JOIN creators c ON ic.creatorID = c.creatorID
            WHERE i2.key = ?1
            GROUP BY i.key
            ORDER BY MIN(ic.orderIndex);"
            .to_string();

        // Collect first and assign after: `with_zotero_db` may discard an attempt.
        let all_authors = with_zotero_db(&path, |conn| {
            let mut stmt = conn.prepare(&query)?;
            let mut all_authors = Vec::with_capacity(items.len());
            for item in items.iter() {
                let library_key = &item.metadata.library_key;
                let authors = match stmt.query(rusqlite::params![library_key])?.next()? {
                    Some(row) => {
                        let authors: String = row.get(0)?;
                        Some(
                            authors
                                .split(';')
                                .map(str::trim)
                                .map(String::from)
                                .collect(),
                        )
                    }
                    None => None,
                };
                all_authors.push(authors);
            }
            Ok(all_authors)
        })?;

        for (item, authors) in items.iter_mut().zip(all_authors) {
            if let Some(authors) = authors {
                item.metadata.authors = Some(authors);
            }
        }
    } else {
        return Err(LibraryParsingError::SqlError(
            "Library not found when fetching authors.".into(),
        ));
    }

    Ok(())
}

/// Possible outcomes for PDF parsing, with enumerated failure conditions for more granular
/// accounting.
enum ParseOutcome {
    /// PDF extraction succeeded, and the result is attached.
    Success(ZoteroItem),
    /// The item in Zotero is not a PDF, and was not parsed.
    NotPdf,
    /// The item's metadata had an invalid path.
    InvalidPath,
    /// The PDF extraction failed, with the attached error string from the `zqa_pdftools` crate.
    FailedExtraction(String),
    /// The PDF extraction thread panicked.
    Panic,
}

/// Parses the Zotero library, also parsing PDF files if they exist on disk. If not, we currently
/// discard those items.
///
/// # Arguments
///
/// * `event_tx` - An optional channel for [`EngineEvent`]s used for progress updates and sending a
///   summary of parsing outcomes.
/// * `store` - The vector store, used to determine which items are already present.
/// * `library_path` - An optional override for the Zotero library directory. When `None`, the path
///   is resolved from the environment (see [`get_lib_path`]).
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
///
/// # Errors
///
/// * `LibraryParsingError::LanceDBError` if fetching library metadata fails.
/// * `LibraryParsingError::PdfParsingError` if an error reason statistics could not be updated.
#[allow(clippy::too_many_lines)]
pub async fn parse_library<T: ZoteroStore>(
    event_tx: Option<&mpsc::Sender<EngineEvent>>,
    store: &T,
    library_path: Option<&Path>,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItem>, LibraryParsingError> {
    // `parse_library` is `pub`, so calls can overlap. Numbering keeps each call's progress IDs unique.
    static NEXT_PARSE_ID: AtomicUsize = AtomicUsize::new(0);

    let start_time = Instant::now();

    let store_exists = store.exists().await;
    log::debug!("Library parsing: existing_store={store_exists}");
    let metadata = if store_exists {
        get_new_library_items(store, library_path).await?
    } else {
        parse_library_metadata(library_path, start_from, limit)?
    };

    if metadata.is_empty() {
        log::warn!("The library seems to be empty.");

        return Ok(Vec::new());
    }

    log::info!("Found library with {} new items.", metadata.len());

    let n_threads = thread::available_parallelism()
        .unwrap_or(std::num::NonZero::<usize>::MIN)
        .get();
    log::debug!("Using {n_threads} threads");

    // Channels for sending tasks and receiving results
    let (task_tx, task_rx) = crossbeam_channel::bounded::<ZoteroItemMetadata>(metadata.len());
    let (outcome_tx, mut outcome_rx) = mpsc::unbounded_channel::<ParseOutcome>();

    let metadata_len = metadata.len();
    for item in metadata {
        task_tx.send(item).map_err(|e| {
            log::error!("Failed to send task to worker threads: {e}");
            LibraryParsingError::PdfParsingError("Channel send error".into())
        })?;
    }
    drop(task_tx);

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let task_rx = task_rx.clone();
            let outcome_tx = outcome_tx.clone();

            thread::spawn(move || {
                // This `recv()` is instant since the items have been sent, so blocking is fine.
                while let Ok(task) = task_rx.recv() {
                    let result = catch_unwind(AssertUnwindSafe(|| -> ParseOutcome {
                        /* Handle each ZoteroItemMetadata item. This has all the info needed to
                         * actually figure out where the file is on disk and parse it--it's here
                         * that we integrate with `pdftools` to get text out of each PDF. */
                        let Some(path_str) = task.file_path.to_str() else {
                            // Best not to try printing the file path here, give the user the
                            // library key instead.
                            log::warn!(
                                "Skipping item with invalid UTF-8 in path: {:?}",
                                task.library_key
                            );

                            return ParseOutcome::InvalidPath;
                        };

                        // TODO: Handle other formats
                        if !Path::new(path_str)
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
                        {
                            log::warn!("Path {path_str} is not a PDF file.");
                            return ParseOutcome::NotPdf;
                        }

                        log::debug!("Processing {path_str}");

                        match extract_text(path_str) {
                            Ok(content) => ParseOutcome::Success(ZoteroItem {
                                metadata: task,
                                text: content.text_content,
                            }),
                            Err(e) => {
                                log::warn!(
                                    "Failed to parse PDF for item {} with path {}: {}",
                                    task.library_key,
                                    path_str,
                                    e
                                );
                                ParseOutcome::FailedExtraction(e.to_string())
                            }
                        }
                    }))
                    .unwrap_or_else(|_| {
                        log::error!("Thread panicked while processing item");
                        ParseOutcome::Panic
                    });

                    // The receiver only closes if `parse_library` was cancelled, so nobody wants the rest.
                    if outcome_tx.send(result).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(outcome_tx);

    let progress_id = format!(
        "parse-library-{}",
        NEXT_PARSE_ID.fetch_add(1, atomic::Ordering::Relaxed)
    );
    EngineEvent::ProgressStart {
        id: progress_id.clone(),
        label: "Parsing Zotero library".into(),
        total: Some(metadata_len as u64),
    }
    .maybe_send(event_tx)
    .await;

    #[allow(clippy::items_after_statements, reason = "locality of behavior")]
    const FAILED_EXTRACTION_SAMPLE_COUNT: usize = 5;
    let mut results: Vec<ZoteroItem> = Vec::new();
    let mut extraction_failure_samples = Vec::with_capacity(FAILED_EXTRACTION_SAMPLE_COUNT);
    let mut completed = 0;

    // Failure type counts
    let mut not_pdf = 0;
    let mut invalid_path = 0;
    let mut panicked_thread = 0;
    let mut failed_extraction = 0;

    while let Some(item) = outcome_rx.recv().await {
        match item {
            ParseOutcome::Success(item) => results.push(item),
            ParseOutcome::NotPdf => not_pdf += 1,
            ParseOutcome::InvalidPath => invalid_path += 1,
            ParseOutcome::Panic => panicked_thread += 1,
            ParseOutcome::FailedExtraction(e) => {
                if extraction_failure_samples.len() < FAILED_EXTRACTION_SAMPLE_COUNT {
                    extraction_failure_samples.push(format!("\t\t{e}"));
                }

                failed_extraction += 1;
            }
        }

        completed += 1;
        EngineEvent::ProgressUpdate {
            id: progress_id.clone(),
            completed,
        }
        .maybe_send(event_tx)
        .await;
    }

    EngineEvent::ProgressComplete { id: progress_id }
        .maybe_send(event_tx)
        .await;

    for handle in handles {
        if let Err(e) = handle.join() {
            log::error!("Thread panicked: {e:?}");
        }
    }

    let end_time = Instant::now();
    let elapsed_time = (end_time - start_time).as_secs();
    let minutes = elapsed_time / 60;
    let seconds = elapsed_time % 60;

    log::info!(
        "Parsed {} items from library in {}min {}s.",
        results.len(),
        minutes,
        seconds
    );

    let fail_count = metadata_len - results.len();
    if fail_count == 0 {
        log::info!("There were no errors during parsing.");
    } else {
        log::warn!("{fail_count} items could not be parsed.");

        let mut summary = format!("{fail_count} items could not be parsed:");

        for (count, reason) in [
            (not_pdf, "were not PDFs"),
            (invalid_path, "had invalid file paths"),
            (panicked_thread, "had unexpected errors"),
            (
                failed_extraction,
                "had PDFs whose content couldn't be extracted",
            ),
        ] {
            if count > 0 {
                let _ = write!(summary, "\n\t{count} failed because they {reason}.");
            }
        }

        if failed_extraction > 0 {
            let _ = write!(summary, "\n{}", extraction_failure_samples.join("\n"));
        }
        let _ = write!(summary, "\n\n");

        EngineEvent::Warning { message: summary }
            .maybe_send(event_tx)
            .await;
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dotenv::dotenv;
    use zqa_macros::{test_eq, test_ok};
    use zqa_rag::config::VoyageAIConfig;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::embedding::common::EmbeddingProviderConfig;

    use super::*;
    use crate::LanceZoteroStore;
    use crate::common::setup_logger;

    #[test]
    fn test_library_fetching_works() {
        dotenv().ok();
        // Read the toy library shipped in `assets/` rather than a real `~/Zotero`, so this does not
        // depend on the developer's library (which may be locked by a running Zotero app).
        let library_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("Zotero");
        let library_items = parse_library_metadata(Some(&library_path), None, None);

        test_ok!(library_items);
        let items = library_items.unwrap();
        assert_ne!(items, [] as [ZoteroItemMetadata; 0]);
    }

    #[test]
    fn test_library_fetching_works_while_locked() {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("Zotero");
        let unlocked = parse_library_metadata(Some(&assets), None, None).unwrap();

        // An awkward directory name, plus invalid UTF-8 on Linux. APFS rejects file names that are
        // not valid UTF-8, and Windows paths are not byte strings.
        let dir = tempfile::tempdir().unwrap();
        #[cfg(target_os = "linux")]
        let dir_name =
            <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(b"Zotero #1 %20\xff");
        #[cfg(not(target_os = "linux"))]
        let dir_name = std::ffi::OsStr::new("Zotero #1 %20");
        let library_path = dir.path().join(dir_name);
        std::fs::create_dir(&library_path).unwrap();
        std::fs::copy(
            assets.join("zotero.sqlite"),
            library_path.join("zotero.sqlite"),
        )
        .unwrap();

        // Recent writes also block unlocked reads, so age the file to test each guard on its own.
        let age_file = || {
            File::options()
                .write(true)
                .open(library_path.join("zotero.sqlite"))
                .unwrap()
                .set_modified(SystemTime::now() - Duration::from_mins(1))
                .unwrap();
        };

        // Hold the lock the way a running Zotero does, with a write in progress. The small cache
        // makes SQLite spill uncommitted pages into the database file.
        let zotero = Connection::open(library_path.join("zotero.sqlite")).unwrap();
        zotero
            .execute_batch(
                "PRAGMA locking_mode = EXCLUSIVE; PRAGMA cache_size = 1; BEGIN EXCLUSIVE;
                 UPDATE itemDataValues SET value = value || ' (uncommitted)';",
            )
            .unwrap();
        age_file();

        // Checked only on Linux. Apple's system SQLite doesn't spill these pages despite the small
        // cache, so on macOS the database file holds only committed data and the guard has nothing
        // to reject.
        #[cfg(target_os = "linux")]
        assert!(parse_library_metadata(Some(&library_path), None, None).is_err());

        zotero.execute_batch("ROLLBACK; BEGIN EXCLUSIVE;").unwrap();
        age_file();

        let start = Instant::now();
        let locked = parse_library_metadata(Some(&library_path), None, None).unwrap();
        assert!(start.elapsed() < Duration::from_secs(1));
        test_eq!(locked, unlocked);

        let mut items: Vec<ZoteroItem> = locked
            .into_iter()
            .map(|metadata| ZoteroItem {
                metadata,
                text: String::new(),
            })
            .collect();
        test_ok!(get_authors(&mut items, Some(&library_path)));
        assert!(items.iter().any(|item| item.metadata.authors.is_some()));

        // Zotero 10 uses WAL mode, where committed changes stay in the `-wal` file until a
        // checkpoint. The read must include them. Switching modes just wrote the main file, so the
        // read also has to wait out the recent-write guard rather than give up.
        zotero
            .execute_batch(
                "COMMIT; PRAGMA journal_mode = WAL; PRAGMA wal_autocheckpoint = 0;
                 UPDATE itemDataValues SET value = value || ' (in WAL)';",
            )
            .unwrap();
        let in_wal = parse_library_metadata(Some(&library_path), None, None).unwrap();
        test_eq!(in_wal.len(), unlocked.len());
        assert!(in_wal.iter().all(|item| item.title.ends_with(" (in WAL)")));
    }

    /// Test that on CI, the toy library is loaded instead of searching for a non-existent "real"
    /// library. The "toy" library is a real Zotero library copied over to the `assets/` directory,
    /// but only contains 10 papers, so that the other tests can run much faster.
    ///
    /// This is never meant to run on CI! Use this locally to ensure that the `get_lib_path`
    /// function correctly handles CI instead, by removing the `#[ignore]` and adding a `FAKE_CI`
    /// variable to your `.env`. The value of this does not matter, it just has to exist.
    #[test]
    #[ignore = "This test is meant to be run locally only"]
    fn test_toy_library_loaded_in_ci() {
        dotenv().ok();

        if env::var("FAKE_CI").is_ok() {
            let lib_path = temp_env::with_vars([("CI", Some("true"))], get_lib_path);

            assert!(lib_path.is_some());
            let lib_path = lib_path.unwrap();
            assert!(lib_path.to_str().unwrap().contains("zqa"));

            let library_items = parse_library_metadata(None, None, None);
            test_ok!(library_items);

            let items = library_items.unwrap();
            assert_ne!(items, [] as [ZoteroItemMetadata; 0]);
            assert_eq!(items.len(), 10);
        } else {
            panic!(concat!(
                "You have enabled `test_toy_library_loaded_in_ci`, but ",
                "have not set the `FAKE_CI` variable. This is not valid."
            ));
        }
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the success and failure cases share the store setup"
    )]
    async fn test_parse_library() {
        // Check the titles/authors are in the expected order and pairs.
        // The parsing might change; we only care about keywords
        const EXPECTED_TITLES: [&str; 5] = [
            "An expert system",
            "Online Learning Rate Adaptation",
            "Mono2Micro",
            "Anomaly Detection",
            "Learning Rate Curriculum",
        ];
        let expected_authors = [
            vec!["Yedida", "Krishna", "Kalia", "Menzies", "Xiao", "Vukovic"],
            vec!["Baydin", "Cornish", "Rubio", "Schmidt", "Wood"],
            vec!["Krishna", "Xiao", "Vukovic", "Kalia", "Sinha", "Banerjee"],
            vec!["Yedida", "Mehendale", "Challa", "Danda", "Sarkar", "Saha"],
            vec!["Croitoru", "Ristea", "Ionescu", "Sebe"],
        ];

        dotenv().ok();
        let _ = setup_logger(log::LevelFilter::Info);

        let tmp = tempfile::tempdir().unwrap();
        let db_uri = tmp
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        // Pin the store to an isolated temp URI and read the toy library shipped in `assets/`, so
        // this test touches no process-global `LANCEDB_URI`/`CI` state (parallel-safe) and does not
        // depend on a real `~/Zotero` that may be locked by a running Zotero app.
        let library_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("Zotero");

        let embedding_config = EmbeddingProviderConfig::VoyageAI(VoyageAIConfig {
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            api_key: env::var("VOYAGE_AI_API_KEY").expect("VOYAGE_AI_API_KEY not set"),
            reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            max_concurrent_requests: zqa_rag::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
            max_retries: zqa_rag::constants::DEFAULT_MAX_RETRIES,
        });
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]));
        let store = LanceZoteroStore::from_schema(embedding_config, schema).with_uri(&db_uri);

        // Nothing reads events until parsing returns, so the channel has to hold all of them.
        let (event_tx, mut event_rx) = mpsc::channel(256);
        let items = parse_library(
            Some(&event_tx),
            &store,
            Some(&library_path),
            Some(0),
            Some(7),
        )
        .await;
        test_ok!(items);

        let mut items = items.unwrap();
        assert!(!items.is_empty());
        test_eq!(items.len(), 7);

        let Ok(EngineEvent::ProgressStart {
            id, total: Some(7), ..
        }) = event_rx.try_recv()
        else {
            panic!("parsing should start a progress bar for 7 items");
        };

        for expected in 1..=7 {
            let Ok(EngineEvent::ProgressUpdate {
                id: update_id,
                completed,
            }) = event_rx.try_recv()
            else {
                panic!("expected progress update {expected}");
            };

            test_eq!(update_id, id);
            test_eq!(completed, expected);
        }

        assert!(matches!(
            event_rx.try_recv(),
            Ok(EngineEvent::ProgressComplete { id: complete_id }) if complete_id == id
        ));

        // Every item parsed, so no failure summary follows.
        assert!(event_rx.try_recv().is_err());

        // Now fetch authors from the Zotero DB
        let authors_result = get_authors(&mut items, Some(&library_path));
        test_ok!(authors_result);

        let mut found_bits = 0;

        for item in &items {
            assert!(item.metadata.authors.is_some());

            let authors = item.metadata.authors.as_ref().unwrap();
            let idx = EXPECTED_TITLES
                .iter()
                .enumerate()
                .find(|(_, title)| item.metadata.title.contains(**title));

            if let Some((idx, _)) = idx {
                for expected_author in &expected_authors[idx] {
                    assert!(
                        authors.iter().any(|a| a.contains(expected_author)),
                        "Author {expected_author} not found in {authors:?}"
                    );
                }

                // At this point, all checks have passed, so mark it as found
                found_bits |= 1 << idx;
            }
        }

        test_eq!(found_bits, 0b11111);

        // Without the `storage` directory every PDF is missing, so each item fails extraction. The
        // summary counts all 7 failures but lists only the first five errors.
        let no_pdfs_path = tmp.path().join("Zotero");
        std::fs::create_dir(&no_pdfs_path).unwrap();
        std::fs::copy(
            library_path.join("zotero.sqlite"),
            no_pdfs_path.join("zotero.sqlite"),
        )
        .unwrap();

        let items = parse_library(
            Some(&event_tx),
            &store,
            Some(&no_pdfs_path),
            Some(0),
            Some(7),
        )
        .await;
        test_ok!(items);
        assert!(items.unwrap().is_empty());

        let warnings: Vec<_> = std::iter::from_fn(|| event_rx.try_recv().ok())
            .filter_map(|event| match event {
                EngineEvent::Warning { message } => Some(message),
                _ => None,
            })
            .collect();

        test_eq!(warnings.len(), 1);
        assert!(warnings[0].starts_with("7 items could not be parsed:"));
        assert!(
            warnings[0]
                .contains("7 failed because they had PDFs whose content couldn't be extracted.")
        );
        test_eq!(warnings[0].matches("\n\t\t").count(), 5);
    }
}
