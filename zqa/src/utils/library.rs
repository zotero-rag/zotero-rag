use core::fmt;
use directories::UserDirs;
use rusqlite::Connection;
use std::env;
use std::error::Error;
use std::path::PathBuf;

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
pub struct ZoteroItemMetadata {
    pub library_key: String,
    pub title: String,
    pub paper_abstract: Option<String>,
    pub notes: Option<String>,
    pub file_path: PathBuf,
}

/// A general error struct for Zotero library parsing.
#[derive(Clone, Debug)]
pub enum LibraryParsingError {
    LibNotFoundError,
}

impl From<rusqlite::Error> for LibraryParsingError {
    fn from(_: rusqlite::Error) -> Self {
        LibraryParsingError::LibNotFoundError
    }
}

impl fmt::Display for LibraryParsingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LibraryParsingError::LibNotFoundError => write!(f, "Library not found!"),
        }
    }
}

impl Error for LibraryParsingError {}

/// Parses the Zotero library. If successful, returns a list of metadata for each item.
pub fn parse_library() -> Result<Vec<ZoteroItemMetadata>, LibraryParsingError> {
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
                let filename = res_path.split_at(split_idx).1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_fetching_works() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        let library_items = parse_library();

        assert!(library_items.is_ok());
        let items = library_items.unwrap();
        assert!(!items.is_empty());
    }
}
