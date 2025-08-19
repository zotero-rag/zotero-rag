use core::fmt;
use directories::UserDirs;
use indicatif::ProgressBar;
use rag::vector::lance::{DB_URI, LanceError, get_lancedb_items};
use rusqlite::Connection;
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use pdftools::parse::extract_text;

use crate::izip;
use crate::utils::arrow::get_column_from_batch;

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
        } else {
            return None;
        }
    }

    match env::consts::OS {
        "linux" | "macos" | "windows" => {
            UserDirs::new().map(|user_dirs| PathBuf::from(user_dirs.home_dir()).join("Zotero"))
        }
        _ => None,
    }
}

/// Metadata for items in the Zotero library.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ZoteroItemMetadata {
    pub library_key: String,
    pub title: String,
    pub file_path: PathBuf,
}

/// A Zotero library item. Includes full-text from parsing PDFs when they exist.
pub struct ZoteroItem {
    pub metadata: ZoteroItemMetadata,
    pub text: String,
}

/// A general error struct for Zotero library parsing.
#[derive(Clone, Debug)]
pub enum LibraryParsingError {
    LibNotFoundError,
    LanceDBError(String),
    PdfParsingError(String),
}

impl From<rusqlite::Error> for LibraryParsingError {
    fn from(_: rusqlite::Error) -> Self {
        LibraryParsingError::LibNotFoundError
    }
}

impl From<LanceError> for LibraryParsingError {
    fn from(value: LanceError) -> Self {
        LibraryParsingError::LanceDBError(value.to_string())
    }
}

impl From<Box<dyn Error>> for LibraryParsingError {
    fn from(value: Box<dyn Error>) -> Self {
        LibraryParsingError::PdfParsingError(value.to_string())
    }
}

impl fmt::Display for LibraryParsingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LibraryParsingError::LibNotFoundError => write!(f, "Library not found!"),
            LibraryParsingError::LanceDBError(msg) => {
                write!(f, "LanceDB error when parsing library: {msg}")
            }
            LibraryParsingError::PdfParsingError(m) => write!(f, "PDF parsing error: {m}"),
        }
    }
}

impl Error for LibraryParsingError {}

/// Assuming an existing LanceDB database exists, returns a list of items present in the Zotero
/// library but not in the database. The primary use case for this is to update the DB with new
/// items. Note that this does not take into account removed items.
///
/// # Arguments:
///
/// * `embedding_name` - The embedding used by the current DB.
///
/// # Returns
///
/// If successful, a list of `ZoteroItemMetadata` objects corresponding to new items.
pub async fn get_new_library_items(
    embedding_name: &str,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    // If the DB does not exist, every item is new.
    if !PathBuf::from(DB_URI).exists() {
        return parse_library_metadata(None, None);
    }

    let db_items = get_lancedb_items(
        embedding_name,
        vec!["library_key".into(), "title".into(), "file_path".into()],
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
/// # Arguments:
///
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
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
            query.push_str(&format!(" LIMIT {limit_val}"));
        }

        if let Some(offset) = start_from {
            query.push_str(&format!(" OFFSET {offset}"));
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
                })
            })?
            .filter_map(|x| x.ok())
            .collect();

        Ok(item_iter)
    } else {
        Err(LibraryParsingError::LibNotFoundError)
    }
}

/// Parses the Zotero library, also parsing PDF files if they exist on disk. If not, we currently
/// discard those items.
///
/// # Arguments:
///
/// * `embedding_name` - The embedding used by the current DB.
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
pub async fn parse_library(
    embedding_name: &str,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItem>, LibraryParsingError> {
    let start_time = Instant::now();

    let metadata = match PathBuf::from(DB_URI).exists() {
        true => get_new_library_items(embedding_name).await?,
        false => parse_library_metadata(start_from, limit)?,
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

    let bar = Arc::new(Mutex::new(ProgressBar::new(
        metadata.len().try_into().unwrap(),
    )));
    {
        let pbar = bar.lock().unwrap();
        pbar.inc(1);
    } // Drop the lock

    let handles = metadata
        .chunks(chunk_size)
        .map(|chunk| {
            // Parse each chunked subset of items
            let bar = Arc::clone(&bar);
            let chunk = chunk.to_vec();
            let cur_chunk_size = chunk.len();

            thread::spawn(move || {
                let result = chunk
                    .iter()
                    .filter_map(|m| {
                        /* Handle each ZoteroItemMetadata item. This has all the info needed to
                         * actually figure out where the file is on disk and parse it--it's here
                         * that we integrate with `pdftools` to get text out of each PDF. */
                        let path_str = match m.file_path.to_str() {
                            Some(p) => p,
                            None => {
                                // Best not to try printing the file path here, give the user the
                                // library key instead.
                                log::warn!(
                                    "Skipping item with invalid UTF-8 in path: {:?}",
                                    m.library_key
                                );
                                return None;
                            }
                        };

                        // TODO: Handle other formats
                        if !path_str.ends_with(".pdf") {
                            log::warn!("Path {path_str} is not a PDF file.");
                            return None;
                        }
                        log::debug!("Processing {path_str}");

                        match extract_text(path_str) {
                            Ok(text) => Some(ZoteroItem {
                                metadata: m.clone(),
                                text,
                            }),
                            Err(e) => {
                                log::warn!(
                                    "Failed to parse PDF for item {} with path {}: {}",
                                    m.library_key,
                                    path_str,
                                    e
                                );
                                None
                            }
                        }
                    })
                    .collect::<Vec<_>>();

                // Batch-update the progress bar
                let pbar = bar.lock().unwrap();
                pbar.inc(cur_chunk_size as u64);

                result
            })
        })
        .collect::<Vec<_>>();

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

    let pbar = bar.lock().unwrap();
    pbar.finish();

    let fail_count = metadata.len() - results.len();
    if fail_count == 0 {
        log::info!("There were no errors during parsing.");
    } else {
        log::warn!("{fail_count} items could not be parsed.");
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dotenv::dotenv;
    use ftail::Ftail;

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
    #[ignore]
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
        let _ = Ftail::new().console(log::LevelFilter::Info).init();
        let items = parse_library("voyageai", Some(0), Some(5)).await;

        assert!(items.is_ok());

        // Two of the items in the toy library are HTML files, so we actually
        // expect those to fail.
        let items = items.unwrap();
        assert!(!items.is_empty());
        assert!(items.len() <= 5);
    }
}
