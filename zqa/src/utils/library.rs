use core::fmt;
use directories::UserDirs;
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
    pub paper_abstract: Option<String>,
    pub notes: Option<String>,
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
pub fn parse_library_metadata() -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
    if let Some(path) = get_lib_path() {
        let conn = Connection::open(path.join("zotero.sqlite"))?;

        // The SQL query essentially combines the titles and abstracts for each paper into a
        // single row. This is done using GROUP BY on the keys, and MAX to get the field itself.
        let mut stmt = conn.prepare(
            "SELECT items.key AS libraryKey,
                MAX(CASE WHEN fieldsCombined.fieldName = 'title' THEN itemDataValues.value END) AS title,
                MAX(CASE WHEN fieldsCombined.fieldName = 'abstract' THEN itemDataValues.value END) AS abstract,
                itemNotes.note AS notes,
                itemAttachments.path AS filePath
            FROM items
            INNER JOIN itemData ON items.itemID = itemData.itemID
            INNER JOIN fieldsCombined ON itemData.fieldID = fieldsCombined.fieldID
            INNER JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            INNER JOIN itemAttachments ON items.itemID = itemAttachments.itemID
            LEFT JOIN itemNotes ON items.itemID = itemNotes.itemID
            WHERE fieldsCombined.fieldName IN ('title', 'abstract')
            GROUP BY items.key;
        ")?;

        let item_iter: Vec<ZoteroItemMetadata> = stmt
            .query_map([], |row| {
                let res_path: String = row.get(4)?;
                let split_idx = res_path.find(':').unwrap_or(0);
                let filename = res_path.split_at(split_idx + 1).1;
                let lib_key: String = row.get(0)?;

                Ok(ZoteroItemMetadata {
                    library_key: lib_key.clone(),
                    title: row.get(1)?,
                    paper_abstract: row.get(2).unwrap_or_default(),
                    notes: row.get(3)?,
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
pub fn parse_library() -> Result<Vec<ZoteroItem>, LibraryParsingError> {
    let metadata = parse_library_metadata()?;

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

            match extract_text(path_str) {
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
            }
        })
        .collect::<Vec<_>>();

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
    fn library_fetching_works() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        let library_items = parse_library_metadata();

        assert!(library_items.is_ok());
        let items = library_items.unwrap();
        assert!(!items.is_empty());
    }
}
