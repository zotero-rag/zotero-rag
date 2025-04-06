use arrow_array::{self, ArrayRef, RecordBatch, StringArray};
use core::fmt;
use lancedb::arrow::arrow_schema;
use std::error::Error;
use std::sync::Arc;

use super::library::{parse_library, LibraryParsingError};

#[derive(Debug, Clone)]
pub enum ArrowError {
    LibNotFoundError,
    ArrowSchemaError(String),
}

impl fmt::Display for ArrowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibNotFoundError => write!(f, "Library not found"),
            Self::ArrowSchemaError(msg) => write!(f, "Arrow schema error: {}", msg),
        }
    }
}

impl From<LibraryParsingError> for ArrowError {
    fn from(value: LibraryParsingError) -> Self {
        match value {
            LibraryParsingError::LibNotFoundError => Self::LibNotFoundError,
        }
    }
}

impl From<arrow_schema::ArrowError> for ArrowError {
    fn from(value: arrow_schema::ArrowError) -> Self {
        Self::ArrowSchemaError(value.to_string())
    }
}

impl Error for ArrowError {}

fn library_to_arrow() -> Result<RecordBatch, ArrowError> {
    let lib_items = parse_library()?;

    // Convert ZoteroItemMetadata to something that can be converted to Arrow
    // Need to extract fields and create appropriate Arrow arrays
    let schema = arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("abstract", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("notes", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
    ]);

    // Convert ZoteroItemMetadata to Arrow arrays
    let library_keys = StringArray::from(
        lib_items
            .iter()
            .map(|item| item.library_key.as_str())
            .collect::<Vec<&str>>(),
    );

    let titles = StringArray::from(
        lib_items
            .iter()
            .map(|item| item.title.as_str())
            .collect::<Vec<&str>>(),
    );

    let abstracts: StringArray = lib_items
        .iter()
        .map(|item| item.paper_abstract.as_deref())
        .collect();

    let notes: StringArray = lib_items.iter().map(|item| item.notes.as_deref()).collect();

    let file_paths = StringArray::from(
        lib_items
            .iter()
            .map(|item| item.file_path.to_str().unwrap())
            .collect::<Vec<&str>>(),
    );

    let record_batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(library_keys) as ArrayRef,
            Arc::new(titles) as ArrayRef,
            Arc::new(abstracts) as ArrayRef,
            Arc::new(notes) as ArrayRef,
            Arc::new(file_paths) as ArrayRef,
        ],
    )?;

    Ok(record_batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_fetching_works() {
        let batch = library_to_arrow();

        assert!(batch.is_ok());
        assert!(batch.as_ref().unwrap().num_columns() > 0);
        assert!(batch.as_ref().unwrap().num_rows() > 0);
    }
}
