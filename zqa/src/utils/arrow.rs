use arrow_array::{
    ArrayRef, FixedSizeListArray, RecordBatch, StringArray, cast::AsArray, types::Float32Type,
};
use arrow_schema;
use core::fmt;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{
    izip,
    utils::library::{ZoteroItem, ZoteroItemMetadata},
};
use rag::{
    llm::embeddings::get_embedding_dims_by_provider,
    vector::lance::{LanceError, lancedb_exists, vector_search as rag_vector_search},
};

use super::library::{LibraryParsingError, parse_library};

#[derive(Debug, Clone)]
pub enum ArrowError {
    ArrowSchemaError(String),
    LanceError(String),
    LibNotFoundError,
    PathEncodingError,
    PdfParsingError(String),
}

impl fmt::Display for ArrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArrowSchemaError(msg) => write!(f, "Arrow schema error: {msg}"),
            Self::LanceError(msg) => write!(f, "LanceDB error: {msg}"),
            Self::LibNotFoundError => write!(f, "Library not found"),
            Self::PathEncodingError => write!(f, "Path contains invalid UTF-8 characters"),
            Self::PdfParsingError(msg) => write!(f, "{msg}"),
        }
    }
}

impl From<LibraryParsingError> for ArrowError {
    fn from(value: LibraryParsingError) -> Self {
        match value {
            LibraryParsingError::LibNotFoundError => Self::LibNotFoundError,
            LibraryParsingError::LanceDBError(msg) => Self::LanceError(msg),
            LibraryParsingError::PdfParsingError(msg) => Self::PdfParsingError(msg),
        }
    }
}

impl From<arrow_schema::ArrowError> for ArrowError {
    fn from(value: arrow_schema::ArrowError) -> Self {
        Self::ArrowSchemaError(value.to_string())
    }
}

impl From<LanceError> for ArrowError {
    fn from(value: LanceError) -> Self {
        Self::LanceError(value.to_string())
    }
}

impl Error for ArrowError {}

/// Get the schema for our LanceDB table. This is required for both getting library items and
/// checkhealth.
///
/// # Arguments:
/// * `embedding_name` - The embedding used by the current DB.
///
/// # Returns:
/// The schema in Arrow format.
pub async fn get_schema(embedding_name: &str) -> arrow_schema::Schema {
    // Convert ZoteroItemMetadata to something that can be converted to Arrow
    // Need to extract fields and create appropriate Arrow arrays
    let mut schema_fields = vec![
        arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
    ];

    if lancedb_exists().await {
        schema_fields.push(arrow_schema::Field::new(
            "embeddings",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                get_embedding_dims_by_provider(embedding_name) as i32,
            ),
            true,
        ));
    }
    arrow_schema::Schema::new(schema_fields)
}

/// Converts Zotero library items to an Arrow RecordBatch.
///
/// This function parses the Zotero library using `parse_library()` and converts
/// the resulting `ZoteroItemMetadata` entries into a structured Arrow RecordBatch.
/// The RecordBatch contains the following columns:
/// - library_key: The unique key for each item in the Zotero library
/// - title: The title of the paper/document
/// - abstract: The abstract of the paper (optional)
/// - notes: Any notes associated with the item (optional)
/// - file_path: Path to the document file
///
/// # Returns
///
/// A `Result` containing either the Arrow `RecordBatch` with all library items
/// or an `ArrowError` if parsing fails or schema conversion fails.
///
/// # Errors
///
/// This function returns an error if:
/// - The Zotero library can't be found or parsed
/// - There's an error creating the Arrow schema
/// - There's an error converting the data to Arrow format
/// - Any file paths contain invalid UTF-8 characters
///
/// # Arguments
///
/// * `embedding_name` - The embedding used by the current DB.
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
pub async fn library_to_arrow(
    embedding_name: &str,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<RecordBatch, ArrowError> {
    let lib_items = parse_library(embedding_name, start_from, limit).await?;
    log::info!("Finished parsing library items.");

    let schema = Arc::new(get_schema(embedding_name).await);

    // Convert ZoteroItemMetadata to Arrow arrays
    let library_keys = StringArray::from(
        lib_items
            .iter()
            .map(|item| item.metadata.library_key.as_str())
            .collect::<Vec<&str>>(),
    );

    let titles = StringArray::from(
        lib_items
            .iter()
            .map(|item| item.metadata.title.as_str())
            .collect::<Vec<&str>>(),
    );

    let pdf_texts = StringArray::from(
        lib_items
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<&str>>(),
    );

    // Convert file paths to strings, returning an error if any path has invalid UTF-8
    let file_paths_vec: Result<Vec<&str>, ArrowError> = lib_items
        .iter()
        .map(|item| {
            item.metadata
                .file_path
                .to_str()
                .ok_or(ArrowError::PathEncodingError)
        })
        .collect();
    let file_paths = StringArray::from(file_paths_vec?);

    let mut record_batch_cols = vec![
        Arc::new(library_keys) as ArrayRef,
        Arc::new(titles) as ArrayRef,
        Arc::new(file_paths) as ArrayRef,
        Arc::new(pdf_texts) as ArrayRef,
    ];
    let embedding_dims = get_embedding_dims_by_provider(embedding_name);

    if lancedb_exists().await {
        record_batch_cols.push(Arc::new(FixedSizeListArray::from_iter_primitive::<
            Float32Type,
            _,
            _,
        >(
            (0..lib_items.len())
                .map(|_| Some(vec![Some(0.0); embedding_dims as usize]))
                .collect::<Vec<Option<Vec<Option<f32>>>>>(),
            embedding_dims as i32,
        )));
    }
    let record_batch = RecordBatch::try_new(schema.clone(), record_batch_cols)?;

    Ok(record_batch)
}

/// From a `RecordBatch`, return all values from a specified column as a `Vec<String>`.
///
/// # Arguments
///
/// * `batch`: A reference to a `RecordBatch`.
/// * `column`: The index of the column to use.
///
/// # Returns
///
/// A `Vec<String>` containing all the items in the specified column of the `RecordBatch`.
pub fn get_column_from_batch(batch: &RecordBatch, column: usize) -> Vec<String> {
    let results = batch.column(column).as_string::<i32>();

    results
        .iter()
        .filter_map(|s| Some(s?.to_string()))
        .collect()
}

/// Perform vector search using a query and a specified embedding method.
///
/// This function is a Zotero-specific wrapper for the `vector_search` function in the `rag` crate.
/// It is implemented here since the knowledge of which column is which in the `RecordBatch`es that
/// we create is in this file, so there's better locality-of-behaviour; this also makes the
/// underlying implementation of `vector_search` simpler and potentially allows other RAG
/// applications to be built on top of it.
///
/// In some sense, this function is the reverse of the `library_to_arrow` function, which creates a
/// `RecordBatch` from vectors after calling `parse_library`.
///
/// # Arguments
///
/// * `query` - The query to search the LanceDB table for.
/// * `embedding_name` - The embedding method to use. Must be one of `EmbeddingProviders`. Note
///   that this must be the same embedding provider used when initially creating the database.
///
/// # Returns
///
/// A `Vec<ZoteroItem>` containing the resulting items from the Zotero library. Returns an
/// `ArrowError` that wraps the underlying `LanceError` if the `rag` crate's `vector_search` is
/// unsuccessful for any reason.
pub async fn vector_search(
    query: String,
    embedding_name: String,
) -> Result<Vec<ZoteroItem>, ArrowError> {
    let batches = rag_vector_search(query, embedding_name).await?;

    let items: Vec<ZoteroItem> = batches
        .iter()
        .flat_map(|batch| {
            let schema = batch.schema();
            let key_idx = schema.index_of("library_key").unwrap();
            let title_idx = schema.index_of("title").unwrap();
            let file_path_idx = schema.index_of("file_path").unwrap();
            let text_idx = schema.index_of("pdf_text").unwrap();

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
                    },
                    text,
                })
                .collect();

            items_batch
        })
        .collect();

    Ok(items)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::common::setup_logger;

    use super::*;
    use arrow_array::RecordBatchIterator;
    use dotenv::dotenv;
    use rag::vector::lance::TABLE_NAME;

    #[tokio::test]
    async fn test_library_to_arrow_works() {
        dotenv().ok();
        let _ = setup_logger(log::LevelFilter::Info);
        let _ = fs::remove_dir_all(TABLE_NAME);
        let _ = fs::remove_dir_all(format!("zqa/{}", TABLE_NAME));
        let _ = fs::remove_dir_all(format!("rag/{}", TABLE_NAME));

        let record_batch = library_to_arrow("voyageai", Some(0), Some(5)).await;
        assert!(
            record_batch.is_ok(),
            "Failed to fetch library: {:?}",
            record_batch.err()
        );

        let record_batch = record_batch.unwrap();
        let schema = record_batch.schema();
        let batches = vec![Ok(record_batch)];
        let mut batch_iter = RecordBatchIterator::new(batches.into_iter(), schema);

        // Get the first batch
        let batch = batch_iter
            .next()
            .expect("No batches in iterator")
            .expect("Error in batch");

        // Whether it's 4 or 5 depends on whether the DB exists; this isn't technically guaranteed,
        // but both are valid states.
        assert!(
            [4, 5].contains(&batch.num_columns()),
            "Expected 4 or 5 columns in record batch"
        );
        assert!(
            batch.num_rows() > 0,
            "Expected non-zero rows in record batch"
        );
        assert!(
            batch.num_rows() <= 5,
            "Expected fewer than five rows in record batch"
        );
    }
}
