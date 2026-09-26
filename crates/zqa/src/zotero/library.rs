//! Zotero library metadata, author lookup, and PDF extraction.

use std::collections::HashSet;
use std::fmt::Write;
use std::hash::Hash;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, atomic};
use std::time::Instant;
use std::{env, thread};

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use directories::UserDirs;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rusqlite::{Connection, ErrorCode, OpenFlags};
use serde::Serialize;
use thiserror::Error;
use zqa_pdftools::parse::extract_text;

use super::zotero_api::{LocalApiError, ZoteroApi};
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
    SqlError(#[source] Arc<rusqlite::Error>),
    #[error("Zotero library directory could not be found")]
    LibraryNotFound,
    #[error(transparent)]
    LocalApi(#[from] LocalApiError),
    #[error("LanceDB error when parsing library: {0}")]
    LanceDBError(String),
    #[error("PDF parsing error: {0}")]
    PdfParsingError(String),
}

impl From<rusqlite::Error> for LibraryParsingError {
    fn from(e: rusqlite::Error) -> Self {
        LibraryParsingError::SqlError(Arc::new(e))
    }
}

impl LibraryParsingError {
    /// Identifies contention before converting database errors to display strings.
    fn is_locked(&self) -> bool {
        matches!(self, Self::SqlError(error) if matches!(
            error.sqlite_error_code(),
            Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
        ))
    }
}

/// Opens an existing Zotero database read-only and disables SQLite's five-second busy wait.
fn open_library_database(path: &Path) -> Result<Connection, rusqlite::Error> {
    let conn =
        Connection::open_with_flags(path.join("zotero.sqlite"), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.busy_timeout(std::time::Duration::ZERO)?;
    Ok(conn)
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
/// * `LibraryParsingError::LibraryNotFound` if the library directory cannot be resolved.
/// * `LibraryParsingError::SqlError` if the database cannot be opened or queried.
/// * `LibraryParsingError::LocalApi` if a locked database's API fallback fails or cannot verify the library.
/// * `LibraryParsingError::LanceDBError` if fetching the rows from LanceDB fails.
pub async fn get_new_library_items<T: ZoteroStore>(
    store: &T,
    library_path: Option<&Path>,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    let metadata_vecs = store
        .existing_item_metadata()
        .await
        .map_err(|e| LibraryParsingError::LanceDBError(e.to_string()))?;

    let library_items = parse_library_metadata(library_path, None, None).await?;
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

/// Parses Zotero metadata, using its local API immediately if the SQLite database is locked.
/// The API's storage directory must match the selected library before results are accepted.
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
/// * `LibraryParsingError::LibraryNotFound` if the library directory cannot be resolved.
/// * `LibraryParsingError::SqlError` if the database cannot be opened or queried.
/// * `LibraryParsingError::LocalApi` if a locked database's API fallback fails or cannot verify the library.
pub async fn parse_library_metadata(
    library_path: Option<&Path>,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    let path = resolve_lib_path(library_path).ok_or(LibraryParsingError::LibraryNotFound)?;
    log::debug!(
        "Reading Zotero metadata: path={}, offset={start_from:?}, limit={limit:?}",
        path.display()
    );

    let items = match parse_library_metadata_sqlite(&path, start_from, limit) {
        Err(error) if error.is_locked() => {
            let mut api = ZoteroApi::new()?;
            Ok(api.metadata(&path, start_from, limit).await?)
        }
        result => result,
    }?;

    log::debug!("Read {} Zotero metadata items", items.len());
    Ok(items)
}

/// Reads metadata directly, without waiting for another connection's SQLite locks.
fn parse_library_metadata_sqlite(
    path: &Path,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    let conn = open_library_database(path)?;

    // NOTE: Maintainers, keep trash inclusion aligned with `ZoteroApi::items`; both readers include trashed items.
    // NOTE: Maintainers, keep parent item types aligned with `ZoteroApi::metadata`.
    // NOTE: Maintainers, keep pagination ordering aligned with `ZoteroApi::metadata`:
    // attachment date added, key, then group ID (zero for the personal library).
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
            LEFT JOIN groups g ON i2.libraryID = g.libraryID
            WHERE f.fieldName = 'title'
            AND ia.contentType = 'application/pdf'
            AND ia.path IS NOT NULL AND ia.path != ''
            AND it.typeName IN ('conferencePaper', 'journalArticle', 'preprint')
            ORDER BY i2.dateAdded, i2.key, COALESCE(g.groupID, 0) "
        .to_string();

    // Useful for debugging
    if let Some(limit_val) = limit {
        let _ = write!(query, " LIMIT {limit_val}");
    }

    if let Some(offset) = start_from {
        if limit.is_none() {
            query.push_str(" LIMIT -1");
        }
        let _ = write!(query, " OFFSET {offset}");
    }

    let mut stmt = conn.prepare(&query)?;

    let item_iter: Vec<ZoteroItemMetadata> = stmt
        .query_map([], |row| {
            let res_path: String = row.get(1)?;
            let lib_key: String = row.get(2)?;

            // NOTE: Maintainers, keep local paths aligned with `ZoteroApi::attachment_path`.
            // Only `storage:` paths belong under Zotero's storage directory. Linked files
            // retain their absolute paths, including drive letters on Windows.
            let file_path = match res_path.strip_prefix("storage:") {
                Some(filename) => path.join("storage").join(&lib_key).join(filename),
                None => PathBuf::from(res_path),
            };

            Ok(ZoteroItemMetadata {
                library_key: lib_key,
                title: row.get(0)?,
                file_path,
                authors: None,
            })
        })?
        .inspect(|row| {
            if let Err(error) = row {
                log::debug!("Failed to read Zotero metadata row: {error}");
            }
        })
        .collect::<Result<_, _>>()?;

    Ok(item_iter)
}

/// Sets authors in-place, using the matching local Zotero API when SQLite is locked.
///
/// # Arguments
///
/// * `items` - The items whose metadata needs to be filled in.
/// * `library_path` - An optional override for the Zotero library directory. When `None`, the path
///   is resolved from the environment (see [`get_lib_path`]).
///
/// # Errors
///
/// * `LibraryParsingError::LibraryNotFound` if the library directory cannot be resolved.
/// * `LibraryParsingError::SqlError` if a database query fails.
/// * `LibraryParsingError::LocalApi` if the API fallback fails or cannot verify the library.
pub async fn get_authors(
    items: &mut [ZoteroItem],
    library_path: Option<&Path>,
) -> Result<(), LibraryParsingError> {
    if items.is_empty() {
        return Ok(());
    }
    let path = resolve_lib_path(library_path).ok_or(LibraryParsingError::LibraryNotFound)?;
    match get_authors_sqlite(items, &path) {
        Err(error) if error.is_locked() => {
            let mut api = ZoteroApi::new()?;
            Ok(api.authors(items, &path).await?)
        }
        result => result,
    }
}

/// Fills author metadata through SQLite without retrying locked reads.
fn get_authors_sqlite(items: &mut [ZoteroItem], path: &Path) -> Result<(), LibraryParsingError> {
    let conn = open_library_database(path)?;

    // Zotero represents a paper and its PDF attachment as separate rows in `items`, each with
    // its own key. Our `library_key` identifies the attachment, since that key also identifies
    // its storage folder. The paper's title and creators belong to the parent item instead.
    // To look up creators, first match the attachment's key through `ia.itemID = i.itemID`,
    // then follow `ia.parentItemID` to the parent's rows in `itemCreators`.
    //
    // Return one row per creator in `orderIndex` order so the resulting vector preserves
    // Zotero's creator order. Each row contains a complete name, including any punctuation
    // within that name. Single-field creators, such as organizations,
    // have an empty firstName and should be returned without a trailing comma.
    //
    // NOTE: Maintainers, keep creator ordering and single-field names aligned with `ItemData::authors`.
    let query = "
        SELECT CASE WHEN c.firstName = '' THEN c.lastName
                    ELSE c.lastName || ', ' || c.firstName END
        FROM itemAttachments ia
        JOIN items i ON ia.itemID = i.itemID
        JOIN itemCreators ic ON ia.parentItemID = ic.itemID
        JOIN creators c ON ic.creatorID = c.creatorID
        WHERE i.key = ?1
        ORDER BY ic.orderIndex";

    let mut stmt = conn.prepare(query)?;
    for item in items {
        let authors = stmt
            .query_map([&item.metadata.library_key], |row| row.get(0))?
            .collect::<Result<Vec<String>, _>>()?;
        item.metadata.authors = (!authors.is_empty()).then_some(authors);
    }

    Ok(())
}

/// Get the Unicode characters for each tick of the progress bar.
///
/// This uses the ⣿ pattern, which is part of the Unicode Braille Pattern block. Each of the 8 dots
/// is represented by a bit, and the block itself starts at U+2800. The 8 dots are represented by
/// an offset in a byte. The representation is as follows: the last three bits, read in reverse
/// order, describe the first three dots of the left column; the next three bits describe the first
/// three dots of the right column. The last two bits (which are the two most significant bits of
/// the byte in reverse order) describe the bottom two. I'm sure there's some historical reason why
/// the first three in each column are separate from the last two, and this is certainly a choice
/// we've made.
const fn compute_pbar_ticks() -> [char; 8] {
    const FILLED_BOX: u32 = 0x28FF;
    const DOTS: [u32; 8] = [1, 1 << 1, 1 << 2, 1 << 6, 1 << 7, 1 << 5, 1 << 4, 1 << 3];

    let mut chars = ['\0'; 8];
    let mut i = 0;
    while i < 8 {
        chars[i] = char::from_u32(FILLED_BOX - DOTS[i]).unwrap();
        i += 1;
    }
    chars
}

#[inline]
fn get_pbar_ticks() -> String {
    const PBAR_TICKS: [char; 8] = compute_pbar_ticks();
    PBAR_TICKS.iter().collect()
}

/// Parses the Zotero library, also parsing PDF files if they exist on disk. If not, we currently
/// discard those items.
///
/// # Arguments
///
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
///
/// # Panics
///
/// * If metadata could not be converted to a `u64`.
/// * If a Mutex lock could not be acquired on the progress bar.
/// * If the threads could not be joined.
#[allow(clippy::too_many_lines)]
pub async fn parse_library<T: ZoteroStore>(
    store: &T,
    library_path: Option<&Path>,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItem>, LibraryParsingError> {
    let start_time = Instant::now();

    let store_exists = store.exists().await;
    log::debug!("Library parsing: existing_store={store_exists}");
    let metadata = if store_exists {
        get_new_library_items(store, library_path).await?
    } else {
        parse_library_metadata(library_path, start_from, limit).await?
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

    let chunk_size = metadata.len().div_ceil(n_threads);
    log::debug!("Using chunk size of {chunk_size}");

    // Track failure conditions
    let not_pdf_counts = Arc::new(AtomicUsize::new(0));
    let invalid_path_counts = Arc::new(AtomicUsize::new(0));
    let failed_extraction_counts = Arc::new(AtomicUsize::new(0));
    let panic_counts = Arc::new(AtomicUsize::new(0));

    // Channels for sending tasks and receiving results and errors
    let (task_tx, task_rx) = crossbeam_channel::bounded::<ZoteroItemMetadata>(metadata.len());
    let (res_tx, res_rx) = crossbeam_channel::bounded::<ZoteroItem>(metadata.len());
    let (err_tx, err_rx) = crossbeam_channel::bounded::<String>(metadata.len());

    let metadata_len = metadata.len();
    for item in metadata {
        task_tx.send(item).map_err(|e| {
            log::error!("Failed to send task to worker threads: {e}");
            LibraryParsingError::PdfParsingError("Channel send error".into())
        })?;
    }
    drop(task_tx);

    let mbar = Arc::new(MultiProgress::new());

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let task_rx = task_rx.clone();
            let res_tx = res_tx.clone();
            let err_tx = err_tx.clone();

            let mbar = Arc::clone(&mbar);
            let not_pdf_counts = Arc::clone(&not_pdf_counts);
            let invalid_path_counts = Arc::clone(&invalid_path_counts);
            let panic_counts = Arc::clone(&panic_counts);
            let failed_extraction_counts = Arc::clone(&failed_extraction_counts);

            thread::spawn(move || {
                let pbar = mbar.add(ProgressBar::no_length());
                pbar.set_style(
                    ProgressStyle::with_template("{spinner} {wide_msg}")
                        .unwrap()
                        .tick_chars(&get_pbar_ticks()),
                );

                while let Ok(task) = task_rx.recv() {
                    pbar.set_message(task.title.clone());
                    pbar.inc(1);

                    let result = catch_unwind(AssertUnwindSafe(|| {
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

                            invalid_path_counts.fetch_add(1, atomic::Ordering::Relaxed);
                            return;
                        };

                        // TODO: Handle other formats
                        if !Path::new(path_str)
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
                        {
                            log::warn!("Path {path_str} is not a PDF file.");
                            not_pdf_counts.fetch_add(1, atomic::Ordering::Relaxed);

                            return;
                        }
                        log::debug!("Processing {path_str}");

                        match extract_text(path_str) {
                            Ok(content) => {
                                if let Err(e) = res_tx.send(ZoteroItem {
                                    metadata: task,
                                    text: content.text_content,
                                }) {
                                    log::error!("Failed to send result: {e:#?}");
                                }
                            }
                            Err(e) => {
                                log::warn!(
                                    "Failed to parse PDF for item {} with path {}: {}",
                                    task.library_key,
                                    path_str,
                                    e
                                );
                                if let Err(send_err) = err_tx.send(e.to_string()) {
                                    log::error!("Failed to send error: {send_err}");
                                }
                                failed_extraction_counts.fetch_add(1, atomic::Ordering::Relaxed);
                            }
                        }
                    }));

                    if result.is_err() {
                        log::error!("Thread panicked while processing item");
                        panic_counts.fetch_add(1, atomic::Ordering::Relaxed);
                    }
                }

                pbar.finish_with_message("done");
            })
        })
        .collect();
    drop(res_tx);
    drop(err_tx);

    let mut results: Vec<ZoteroItem> = Vec::new();
    while let Ok(item) = res_rx.recv() {
        results.push(item);
    }

    if let Err(e) = mbar.clear() {
        log::error!("Error when clearing MultiProgress: {e:#?}");
    }

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
        println!("{fail_count} items could not be parsed:");

        let not_pdf_count = not_pdf_counts.load(atomic::Ordering::Relaxed);
        if not_pdf_count > 0 {
            println!("\t{not_pdf_count} failed because they were not PDFs.");
        }

        let invalid_path_count = invalid_path_counts.load(atomic::Ordering::Relaxed);
        if invalid_path_count > 0 {
            println!("\t{invalid_path_count} failed because they had invalid file paths.");
        }

        let failed_extraction_count = failed_extraction_counts.load(atomic::Ordering::Relaxed);
        if failed_extraction_count > 0 {
            println!("\t{failed_extraction_count} failed PDF text extraction:");
        }

        while let Ok(e) = err_rx.recv() {
            println!("\t\tError: {e}");
        }

        let panic_count = panic_counts.load(atomic::Ordering::Relaxed);
        if panic_count > 0 {
            println!("\t{panic_count} failed because parsing failed.");
        }
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

    /// Both metadata and author readers must discover an exclusive lock without busy retries.
    #[test]
    fn locked_reads_fail_immediately() {
        let library = tempfile::tempdir().unwrap();
        let writer = Connection::open(library.path().join("zotero.sqlite")).unwrap();
        writer
            .execute_batch("CREATE TABLE marker (id INTEGER); BEGIN EXCLUSIVE;")
            .unwrap();
        let start = Instant::now();
        let metadata = parse_library_metadata_sqlite(library.path(), None, None);
        assert!(metadata.unwrap_err().is_locked());
        let mut items = vec![ZoteroItem {
            metadata: ZoteroItemMetadata {
                library_key: "ATTACH01".into(),
                title: "A paper".into(),
                file_path: PathBuf::new(),
                authors: None,
            },
            text: String::new(),
        }];
        assert!(
            get_authors_sqlite(&mut items, library.path())
                .unwrap_err()
                .is_locked()
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        writer.execute_batch("ROLLBACK;").unwrap();
    }

    /// Author order follows orderIndex, including institutional names and absent creators.
    #[test]
    fn sqlite_authors_preserve_creator_order_and_missing_values() {
        let library = tempfile::tempdir().unwrap();
        let database = Connection::open(library.path().join("zotero.sqlite")).unwrap();
        database
            .execute_batch(
                "CREATE TABLE items (itemID INTEGER, key TEXT);
            CREATE TABLE itemAttachments (itemID INTEGER, parentItemID INTEGER);
            CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, orderIndex INTEGER);
            CREATE TABLE creators (creatorID INTEGER, firstName TEXT, lastName TEXT);
            INSERT INTO items VALUES (2, 'ATTACH01'), (3, 'ATTACH02');
            INSERT INTO itemAttachments VALUES (2, 1), (3, 4);
            INSERT INTO creators VALUES (1, '', 'Institute'), (2, 'Ada', 'Lovelace');
            INSERT INTO itemCreators VALUES (1, 1, 1), (1, 2, 0);",
            )
            .unwrap();
        let mut items: Vec<_> = ["ATTACH01", "ATTACH02"]
            .into_iter()
            .map(|key| ZoteroItem {
                metadata: ZoteroItemMetadata {
                    library_key: key.into(),
                    title: String::new(),
                    file_path: PathBuf::new(),
                    authors: None,
                },
                text: String::new(),
            })
            .collect();
        get_authors_sqlite(&mut items, library.path()).unwrap();
        assert_eq!(
            items[0].metadata.authors.as_ref().unwrap(),
            &["Lovelace, Ada", "Institute"]
        );
        assert!(items[1].metadata.authors.is_none());
    }

    /// A missing or malformed database must not be created or mistaken for a lock.
    #[tokio::test]
    async fn database_errors_do_not_trigger_api_fallback() {
        let library = tempfile::tempdir().unwrap();
        let error = parse_library_metadata(Some(library.path()), None, None)
            .await
            .unwrap_err();
        assert!(matches!(error, LibraryParsingError::SqlError(_)));
        assert!(!error.is_locked());
        assert!(!library.path().join("zotero.sqlite").exists());
        std::fs::write(library.path().join("zotero.sqlite"), b"not a database").unwrap();
        let error = parse_library_metadata(Some(library.path()), None, None)
            .await
            .unwrap_err();
        assert!(matches!(error, LibraryParsingError::SqlError(_)));
        assert!(!error.is_locked());
    }

    /// Copy the toy database so regression tests can change attachment rows independently.
    fn copy_toy_database() -> tempfile::TempDir {
        let library = tempfile::tempdir().unwrap();
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/Zotero/zotero.sqlite");
        std::fs::copy(source, library.path().join("zotero.sqlite")).unwrap();
        library
    }

    /// Exclude attachments without local paths before pagination while retaining usable PDFs.
    #[test]
    fn sqlite_metadata_skips_null_and_empty_attachment_paths() {
        let library = copy_toy_database();
        let expected = parse_library_metadata_sqlite(library.path(), None, None).unwrap();
        let excluded = &expected[0].library_key;
        let database = Connection::open(library.path().join("zotero.sqlite")).unwrap();

        for path in [None, Some("")] {
            database.execute(
                "UPDATE itemAttachments SET path = ?1 WHERE itemID = (SELECT itemID FROM items WHERE key = ?2)",
                rusqlite::params![path, excluded],
            ).unwrap();

            let actual = parse_library_metadata_sqlite(library.path(), None, None).unwrap();
            assert_eq!(actual, expected[1..]);
            let first = parse_library_metadata_sqlite(library.path(), Some(0), Some(1)).unwrap();
            assert_eq!(first, expected[1..2]);
        }
    }

    /// Break date ties by attachment key and apply offsets to that deterministic order.
    #[test]
    fn sqlite_pagination_uses_attachment_dates_and_keys() {
        let library = copy_toy_database();
        let database = Connection::open(library.path().join("zotero.sqlite")).unwrap();
        database
            .execute("UPDATE items SET dateAdded = '2020-01-01 00:00:00'", [])
            .unwrap();
        let mut expected = parse_library_metadata_sqlite(library.path(), None, None).unwrap();
        expected.sort_by(|left, right| left.library_key.cmp(&right.library_key));

        let actual = parse_library_metadata_sqlite(library.path(), Some(1), Some(3)).unwrap();
        assert_eq!(actual, expected[1..4]);
        let remaining = parse_library_metadata_sqlite(library.path(), Some(1), None).unwrap();
        assert_eq!(remaining, expected[1..]);
    }

    /// Keep the toy-library title coverage used by the PDF integration test's first seven items.
    #[test]
    fn toy_library_pagination_retains_expected_titles_and_authors() {
        let library = copy_toy_database();
        let metadata = parse_library_metadata_sqlite(library.path(), Some(0), Some(7)).unwrap();
        assert_eq!(metadata.len(), 7);
        let mut items: Vec<_> = metadata
            .into_iter()
            .map(|metadata| ZoteroItem {
                metadata,
                text: String::new(),
            })
            .collect();
        get_authors_sqlite(&mut items, library.path()).unwrap();

        for title in [
            "An expert system",
            "Online Learning Rate Adaptation",
            "Mono2Micro",
            "Anomaly Detection",
            "Learning Rate Curriculum",
        ] {
            let item = items
                .iter()
                .find(|item| item.metadata.title.contains(title))
                .unwrap();
            assert!(
                item.metadata
                    .authors
                    .as_ref()
                    .is_some_and(|authors| !authors.is_empty())
            );
        }
    }

    #[tokio::test]
    async fn test_library_fetching_works() {
        dotenv().ok();
        // Read the toy library shipped in `assets/` rather than a real `~/Zotero`, so this does not
        // depend on the developer's library (which may be locked by a running Zotero app).
        let library_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("Zotero");
        let library_items = parse_library_metadata(Some(&library_path), None, None).await;

        test_ok!(library_items);
        let items = library_items.unwrap();
        assert!(!items.is_empty());
    }

    /// Test that on CI, the toy library is loaded instead of searching for a non-existent "real"
    /// library. The "toy" library is a real Zotero library copied over to the `assets/` directory,
    /// but only contains 10 papers, so that the other tests can run much faster.
    ///
    /// This is never meant to run on CI! Use this locally to ensure that the `get_lib_path`
    /// function correctly handles CI instead, by removing the `#[ignore]` and adding a `FAKE_CI`
    /// variable to your `.env`. The value of this does not matter, it just has to exist.
    #[tokio::test]
    #[ignore = "This test is meant to be run locally only"]
    async fn test_toy_library_loaded_in_ci() {
        dotenv().ok();

        if env::var("FAKE_CI").is_ok() {
            let lib_path = temp_env::with_vars([("CI", Some("true"))], get_lib_path);

            assert!(lib_path.is_some());
            let lib_path = lib_path.unwrap();
            assert!(lib_path.to_str().unwrap().contains("zqa"));

            let library_items = parse_library_metadata(None, None, None).await;
            test_ok!(library_items);

            let items = library_items.unwrap();
            assert!(!items.is_empty());
            assert_eq!(items.len(), 10);
        } else {
            panic!(concat!(
                "You have enabled `test_toy_library_loaded_in_ci`, but ",
                "have not set the `FAKE_CI` variable. This is not valid."
            ));
        }
    }

    #[tokio::test]
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
        });
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]));
        let store = LanceZoteroStore::from_schema(embedding_config, schema).with_uri(&db_uri);
        let items = parse_library(&store, Some(&library_path), Some(0), Some(7)).await;
        test_ok!(items);

        let mut items = items.unwrap();
        assert!(!items.is_empty());
        test_eq!(items.len(), 7);

        // Now fetch authors from the Zotero DB
        let authors_result = get_authors(&mut items, Some(&library_path)).await;
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
    }

    #[test]
    fn test_get_pbar_ticks() {
        let ticks = get_pbar_ticks();

        assert_eq!(ticks, "⣾⣽⣻⢿⡿⣟⣯⣷");
    }
}
