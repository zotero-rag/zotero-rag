use core::fmt;
use directories::UserDirs;
use indicatif::ProgressBar;
use rusqlite::Connection;
use std::env;
use std::error::Error;
use std::path::PathBuf;

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
/// TODO: For now, we assume the PDF text always exists. This should be generalized later.
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
            LibraryParsingError::PdfParsingError(m) => write!(f, "PDF parsing error: {}", m),
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
        if limit.is_some() {
            query.push_str(&format!(" LIMIT {}", limit.unwrap()));
        }

        if start_from.is_some() {
            query.push_str(&format!(" OFFSET {}", start_from.unwrap()));
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
/// discard those items. TODO: Update this logic later on.
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
    let metadata = parse_library_metadata(start_from, limit)?;

    log::info!("Found library with {} items.", metadata.len());
    let bar = ProgressBar::new(metadata.len().try_into().unwrap());

    let mut count = 0;
    let mut failed_count = 0;
    let items = metadata
        .iter()
        .filter_map(|m| {
            let path_str = match m.file_path.to_str() {
                Some(p) => p,
                None => {
                    failed_count += 1;
                    log::warn!(
                        "Skipping item with invalid UTF-8 in path: {:?}",
                        m.library_key
                    );
                    return None;
                }
            };

            if !path_str.ends_with(".pdf") {
                return None;
            }

            log::debug!(
                "({} / {}) Parsing file {}",
                count + 1,
                metadata.len(),
                path_str
            );
            count += 1;

            let returned = match extract_text(path_str) {
                Ok(text) => Some(ZoteroItem {
                    metadata: m.clone(),
                    text,
                }),
                Err(e) => {
                    failed_count += 1;
                    log::warn!(
                        "Failed to parse PDF for item {} with path {}: {}",
                        m.library_key,
                        path_str,
                        e
                    );
                    None
                }
            };

            bar.inc(1);

            returned
        })
        .collect::<Vec<_>>();

    bar.finish();

    if failed_count > 0 {
        log::warn!("Failed to parse {} PDF files", failed_count);
    }

    log::info!("Parsed {} items from library.", items.len());

    Ok(items)
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

        dbg!(items.len());
    }
}
