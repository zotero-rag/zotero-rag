use arrow_array::RecordBatch;
use directories::UserDirs;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rusqlite::Connection;
use std::collections::HashSet;
use std::env;
use std::fmt::Write;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use thiserror::Error;
use zqa_rag::embedding::common::EmbeddingProviderConfig;
use zqa_rag::vector::lance::{LanceError, get_lancedb_items, lancedb_exists};

use zqa_pdftools::parse::extract_text;

use crate::izip;
use crate::utils::arrow::{DbFields, get_column_from_batch};

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

/// Metadata for items in the Zotero library.
#[derive(Debug, Clone)]
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
#[derive(Clone)]
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

impl AsRef<str> for ZoteroItem {
    fn as_ref(&self) -> &str {
        &self.text
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
pub enum LibraryParsingError {
    #[error("SQLite error: {0}")]
    SqlError(String),
    #[error("LanceDB error when parsing library: {0}")]
    LanceDBError(String),
    #[error("PDF parsing error: {0}")]
    PdfParsingError(String),
}

impl From<rusqlite::Error> for LibraryParsingError {
    fn from(e: rusqlite::Error) -> Self {
        LibraryParsingError::SqlError(e.to_string())
    }
}

impl From<LanceError> for LibraryParsingError {
    fn from(value: LanceError) -> Self {
        LibraryParsingError::LanceDBError(value.to_string())
    }
}

impl From<Box<dyn std::error::Error>> for LibraryParsingError {
    fn from(value: Box<dyn std::error::Error>) -> Self {
        LibraryParsingError::PdfParsingError(value.to_string())
    }
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
pub async fn get_new_library_items(
    embedding_config: &EmbeddingProviderConfig,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    let db_items = get_lancedb_items(
        embedding_config,
        vec![
            DbFields::LibraryKey.into(),
            DbFields::Title.into(),
            DbFields::FilePath.into(),
        ],
    )
    .await?;

    let metadata_vecs = db_items
        .iter()
        .flat_map(|batch| {
            let library_keys = get_column_from_batch(batch, 0);
            let titles = get_column_from_batch(batch, 1);
            let file_paths = get_column_from_batch(batch, 2);

            let zipped = izip!(library_keys, titles, file_paths).collect::<Vec<_>>();
            zipped
                .iter()
                .map(|(key, title, path)| ZoteroItemMetadata {
                    library_key: key.clone(),
                    title: title.clone(),
                    file_path: PathBuf::from(path.clone()),
                    authors: None,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let library_items = parse_library_metadata(None, None)?;

    let db_items_set: HashSet<_> = metadata_vecs.iter().collect();

    Ok(library_items
        .into_iter()
        .filter(|item| !db_items_set.contains(item))
        .collect::<Vec<_>>())
}

/// Parses the Zotero library metadata. If successful, returns a list of metadata for each item.
///
/// # Arguments
///
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
///
/// # Errors
///
/// * `LibraryParsingError::SqliteError` if the library path was not found, the query could not be prepared, or
///   columns from the result set could not be parsed, or `query_map` fails.
pub fn parse_library_metadata(
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    if let Some(path) = get_lib_path() {
        let conn = Connection::open(path.join("zotero.sqlite"))?;

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
          AND it.typeName IN ('conferencePaper', 'journalArticle', 'preprint') "
            .to_string();

        // Useful for debugging
        if let Some(limit_val) = limit {
            let _ = write!(query, " LIMIT {limit_val}");
        }

        if let Some(offset) = start_from {
            let _ = write!(query, " OFFSET {offset}");
        }

        let mut stmt = conn.prepare(&query)?;

        let item_iter: Vec<ZoteroItemMetadata> = stmt
            .query_map([], |row| {
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
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(item_iter)
    } else {
        Err(LibraryParsingError::SqlError(
            "Library not found!".to_string(),
        ))
    }
}

/// Given a Zotero item, fetch the authors and fill them in-place.
///
/// # Arguments
///
/// * `item` - The item whose metadata needs to be filled in
///
/// # Returns
///
/// A unit type wrapped in a `Result`.
fn get_authors_for_item(item: &mut ZoteroItem) -> Result<(), LibraryParsingError> {
    if let Some(path) = get_lib_path() {
        let conn = Connection::open(path.join("zotero.sqlite"))?;
        let library_key = &item.metadata.library_key;

        let query = "SELECT c.firstName, c.lastName
            FROM items i
            JOIN itemData id ON i.itemID = id.itemID
            JOIN fields f ON id.fieldID = f.fieldID
            JOIN itemDataValues idv ON id.valueID = idv.valueID
            JOIN itemCreators ic ON i.itemID = ic.itemID
            JOIN creators c ON ic.creatorID = c.creatorID
            WHERE i.key = ?1
            AND f.fieldName = 'title'
            ORDER BY ic.orderIndex
        "
        .to_string();

        let mut stmt = conn.prepare(&query).unwrap();
        let item_iter: Vec<String> = stmt
            .query_map(rusqlite::params![library_key], |row| {
                let first_name: String = row.get(0)?;
                let last_name: String = row.get(1)?;

                Ok(format!("{last_name}, {first_name}"))
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        item.metadata.authors = Some(item_iter);

        Ok(())
    } else {
        Err(LibraryParsingError::SqlError(
            "Library not found when fetching authors.".into(),
        ))
    }
}

/// Given a set of `items`, set the authors metadata in-place.
///
/// # Arguments
///
/// * `items` - The items whose metadata needs to be filled in.
///
/// # Returns
///
/// A `Result` describing if the operation was successful.
///
/// # Errors
///
/// * `LibraryParsingError::SqlError` if the operation failed for any items.
pub fn get_authors(items: &mut [ZoteroItem]) -> Result<(), LibraryParsingError> {
    for item in items {
        get_authors_for_item(item)?;
    }

    Ok(())
}

fn create_progress_bar(item_count: usize) -> ProgressBar {
    let pb = ProgressBar::new(item_count as u64);
    pb.set_style(
        ProgressStyle::with_template("{prefix:>12.cyan.bold} [{bar:57}] {pos}/{len} {wide_msg}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_prefix("Processing");

    pb
}

/// Parses the Zotero library, also parsing PDF files if they exist on disk. If not, we currently
/// discard those items.
///
/// # Arguments
///
/// * `embedding_config` - The embedding provider configuration for the configured `LanceDB` embedding.
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
pub async fn parse_library(
    embedding_config: &EmbeddingProviderConfig,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItem>, LibraryParsingError> {
    let start_time = Instant::now();

    let metadata = if lancedb_exists().await {
        get_new_library_items(embedding_config).await?
    } else {
        parse_library_metadata(start_from, limit)?
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

    let mbar = MultiProgress::new();

    // Keep track of how many were skipped, and for what reasons. Currently, we won't track the
    // actual items themselves.
    #[allow(clippy::items_after_statements)]
    enum FailReason {
        NotPdf = 0,
        InvalidPath = 1,
        Panic = 2,
    }

    let not_pdf_counts = AtomicUsize::new(0);
    let invalid_path_counts = AtomicUsize::new(0);
    let panic_counts = AtomicUsize::new(0);

    let (task_tx, task_rx) = crossbeam_channel::bounded::<ZoteroItemMetadata>(metadata.len());
    let (res_tx, res_rx) = crossbeam_channel::bounded::<ZoteroItem>(metadata.len());

    let handles = (0..n_threads).map(|_| {
        thread::spawn(|| {
            while let Ok(task) = task_rx.recv() {
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
                    Ok(text) => {
                        res_tx.send(ZoteroItem {
                            metadata: task.clone(),
                            text,
                        });
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to parse PDF for item {} with path {}: {}",
                            task.library_key,
                            path_str,
                            e
                        );
                    }
                }
            }
        });
    });
    drop(res_tx);

    let pbar = create_progress_bar(metadata.len());
    for result in res_rx {
        pbar.inc(1);
    }
    pbar.finish_and_clear();

    let end_time = Instant::now();
    let elapsed_time = (end_time - start_time).as_secs();
    let minutes = elapsed_time / 60;
    let seconds = elapsed_time % 60;

    let results = handles
        .into_iter()
        .flat_map(|handle| handle.join().unwrap_or_else(|_| Vec::new()))
        .collect::<Vec<_>>();
    log::info!(
        "Parsed {} items from library in {}min {}s.",
        results.len(),
        minutes,
        seconds
    );

    let fail_count = metadata.len() - results.len();
    if fail_count == 0 {
        log::info!("There were no errors during parsing.");
    } else {
        log::warn!("{fail_count} items could not be parsed.");
        println!("{fail_count} items could not be parsed:");

        let fail_counts = Arc::clone(&skip_counts);
        if let Ok(fail_counts) = fail_counts.lock() {
            // There's probably a more elegant way to do this, but let's not prematurely optimize.
            if fail_counts[FailReason::NotPdf as usize] > 0 {
                println!(
                    "\t{} failed because they were not PDFs.",
                    fail_counts[FailReason::NotPdf as usize]
                );
            }

            if fail_counts[FailReason::InvalidPath as usize] > 0 {
                println!(
                    "\t{} failed because they had invalid file paths.",
                    fail_counts[FailReason::InvalidPath as usize]
                );
            }
        } else {
            log::warn!(
                "We couldn't get a lock on `fail_counts`, so detailed stats are not printed."
            );
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::common::setup_logger;

    use super::*;
    use dotenv::dotenv;
    use zqa_rag::{
        config::VoyageAIConfig,
        constants::{
            DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL,
            DEFAULT_VOYAGE_RERANK_MODEL,
        },
        vector::lance::TABLE_NAME,
    };

    #[test]
    fn test_library_fetching_works() {
        dotenv().ok();
        let library_items = parse_library_metadata(None, None);

        assert!(library_items.is_ok());
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
    #[test]
    #[ignore = "This test is meant to be run locally only"]
    fn test_toy_library_loaded_in_ci() {
        dotenv().ok();

        if env::var("FAKE_CI").is_ok() {
            let lib_path = temp_env::with_vars([("CI", Some("true"))], get_lib_path);

            assert!(lib_path.is_some());
            let lib_path = lib_path.unwrap();
            assert!(lib_path.to_str().unwrap().contains("zqa"));

            let library_items = parse_library_metadata(None, None);
            assert!(library_items.is_ok());

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
        dotenv().ok();
        let _ = setup_logger(log::LevelFilter::Info);
        let _ = fs::remove_dir_all(TABLE_NAME);
        let _ = fs::remove_dir_all(format!("zqa/{TABLE_NAME}"));
        let _ = fs::remove_dir_all(format!("rag/{TABLE_NAME}"));

        let items = parse_library(
            &EmbeddingProviderConfig::VoyageAI(VoyageAIConfig {
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                api_key: env::var("VOYAGE_AI_API_KEY").expect("VOYAGE_AI_API_KEY not set"),
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            }),
            Some(0),
            Some(5),
        )
        .await;
        assert!(items.is_ok());

        // Two of the items in the toy library are HTML files, so we actually
        // expect those to fail.
        let mut items = items.unwrap();
        assert!(!items.is_empty());
        assert!((0..=5).contains(&items.len()));

        // Now fetch authors from the Zotero DB
        let authors_result = get_authors(&mut items);
        assert!(authors_result.is_ok());
        for item in items {
            assert!(item.metadata.authors.is_some());
        }
    }
}
