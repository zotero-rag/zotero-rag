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
    library_key: String,
    title: String,
    paper_abstract: String,
}

/// A general error struct for Zotero library parsing.
#[derive(Debug)]
pub struct LibraryParsingError {
    message: String,
}

impl fmt::Display for LibraryParsingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.message)
    }
}

impl Error for LibraryParsingError {}

/// Parses the Zotero library. If successful, returns a list of metadata for each item.
/// The SQL query essentially combines the titles and abstracts for each paper into a
/// single row. This is done using GROUP BY on the keys, and MAX to get the field itself.
pub fn parse_library() -> Result<Vec<ZoteroItemMetadata>, Box<dyn Error>> {
    if let Some(path) = get_lib_path() {
        let conn = Connection::open(&path)?;

        let mut stmt = conn.prepare(
            "SELECT items.key AS libraryKey,
               MAX(CASE WHEN fieldsCombined.fieldName = 'title' THEN itemDataValues.value END) AS title,
               MAX(CASE WHEN fieldsCombined.fieldName = 'abstract' THEN itemDataValues.value END) AS abstract
            FROM items
            INNER JOIN itemData ON items.itemID = itemData.itemID
            INNER JOIN fieldsCombined ON itemData.fieldID = fieldsCombined.fieldID
            INNER JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            WHERE fieldsCombined.fieldName IN ('title', 'abstract')
            GROUP BY items.key
            LIMIT 5;
        ")?;

        let item_iter: Vec<ZoteroItemMetadata> = stmt
            .query_map([], |row| {
                Ok(ZoteroItemMetadata {
                    library_key: row.get(0)?,
                    title: row.get(1)?,
                    paper_abstract: row.get(2)?,
                })
            })?
            .filter_map(|x| x.ok())
            .collect();

        Ok(item_iter)
    } else {
        Err(Box::new(LibraryParsingError {
            message: "Failed to get library path".to_string(),
        }))
    }
}
