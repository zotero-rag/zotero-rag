use core::fmt;
use directories::UserDirs;
use indicatif::ProgressBar;
use rusqlite::Connection;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use pdftools::parse::extract_text;

/// Gets the Zotero library path. Works on Linux, macOS, and Windows systems.
///
/// Returns None if either the OS is not one of the above, or if we could not
/// get the directory.
fn get_lib_path() -> Option<PathBuf> {
    match env::consts::OS {
        "linux" | "macos" | "windows" => {
            UserDirs::new().map(|user_dirs| PathBuf::from(user_dirs.home_dir()).join("Zotero"))
        }
        _ => None,
    }
}

/// Metadata for items in the Zotero library.
#[derive(Clone)]
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
    PdfParsingError(String),
}

impl From<rusqlite::Error> for LibraryParsingError {
    fn from(_: rusqlite::Error) -> Self {
        LibraryParsingError::LibNotFoundError
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
            LibraryParsingError::PdfParsingError(m) => write!(f, "PDF parsing error: {m}"),
        }
    }
}

impl Error for LibraryParsingError {}

/// Parses the Zotero library metadata. If successful, returns a list of metadata for each item.
///
/// Arguments:
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
pub fn parse_library_metadata(
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    if let Some(path) = get_lib_path() {
        let conn = Connection::open(path.join("zotero.sqlite"))?;

        let mut query = "WITH itemTitles AS (
                SELECT DISTINCT itemDataValues.value AS title,
                    items.itemID AS itemID
                FROM items
                INNER JOIN itemData ON items.itemID = itemData.itemID
                INNER JOIN itemTypes ON items.itemTypeID = itemTypes.itemTypeID
                INNER JOIN fields ON itemData.fieldID = fields.fieldID
                INNER JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
                WHERE fields.fieldName = 'title'
                AND itemTypes.typeName IN ('conferencePaper', 'journalArticle', 'preprint')
            ),
            itemPaths AS (
                SELECT itemAttachments.path AS filePath,
                       itemAttachments.parentItemID as itemID,
                       itemAttachments.itemID AS childItemID
                FROM items
                LEFT JOIN itemAttachments ON items.itemID = itemAttachments.parentItemID
            )
            SELECT itemTitles.title AS title,
                   itemPaths.filePath AS filePath,
                   items.key AS libraryKey
            FROM itemTitles
            NATURAL JOIN itemPaths
            JOIN items ON itemPaths.childItemID = items.itemID"
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
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
pub fn parse_library(
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<ZoteroItem>, LibraryParsingError> {
    let start_time = Instant::now();
    let metadata = parse_library_metadata(start_from, limit)?;

    if metadata.is_empty() {
        log::warn!("The library seems to be empty.");

        return Ok(Vec::new());
    }

    log::info!("Found library with {} items.", metadata.len());

    let n_threads = thread::available_parallelism()
        .unwrap_or(std::num::NonZero::<usize>::MIN)
        .get();

    let chunk_size = metadata.len().div_ceil(n_threads);

    let bar = Arc::new(Mutex::new(ProgressBar::new(
        metadata.len().try_into().unwrap(),
    )));
    let pbar = bar.lock().unwrap();
    pbar.inc(1);

    let handles = metadata
        .chunks(chunk_size)
        .map(|chunk| {
            // Parse each chunked subset of items
            let bar = Arc::clone(&bar);
            let chunk = chunk.to_vec();
            let cur_chunk_size = chunk.len();

            let handle = thread::spawn(move || {
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
                            return None;
                        }

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
            });

            handle
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

    #[test]
    fn test_library_fetching_works() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        let library_items = parse_library_metadata(None, None);

        assert!(library_items.is_ok());
        let items = library_items.unwrap();
        assert!(!items.is_empty());
    }

    #[test]
    fn test_parse_library() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        let items = parse_library(Some(0), Some(5));

        assert!(items.is_ok());

        let items = items.unwrap();
        assert_eq!(items.len(), 5);
    }
}
