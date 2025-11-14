use arrow_array::{ArrayRef, RecordBatch, StringArray, cast::AsArray};
use arrow_schema;
use std::sync::Arc;

use crate::{
    config::Config,
    utils::library::{ZoteroItem, ZoteroItemSet},
};
use rag::{
    embedding::common::{
        EmbeddingProviderConfig, get_embedding_dims_by_provider, get_embedding_provider,
        get_embedding_provider_with_config, get_reranking_provider,
    },
    llm::errors::LLMError,
    vector::lance::{LanceError, lancedb_exists, vector_search as rag_vector_search},
};

use super::library::{LibraryParsingError, parse_library};

use thiserror::Error;

/// An enum containing the fields stored by our application in LanceDB, in order. Implementations
/// `as_ref()` and `into()` are provided to convert this to `&str` and `String` respectively.
pub(crate) enum DbFields {
    LibraryKey,
    Title,
    PdfText,
    FilePath,
    Embeddings,
}

impl AsRef<str> for DbFields {
    fn as_ref(&self) -> &str {
        match self {
            Self::LibraryKey => "library_key",
            Self::Title => "title",
            Self::FilePath => "file_path",
            Self::PdfText => "pdf_text",
            Self::Embeddings => "embeddings",
        }
    }
}

impl From<DbFields> for String {
    fn from(value: DbFields) -> Self {
        value.as_ref().into()
    }
}

/// This name is a bit of a misnomer, in that this does not only represent errors from Arrow.
/// However, the rationale behind naming it as such is that `arrow.rs` is the high-level interface
/// for the application to LanceDB, PDF parsing, and other lower-level operations. As errors from
/// those functions propagate, they are captured here. In general, this enum should not be used
/// outside this file, except perhaps to `impl From<ArrowError>`.
#[derive(Debug, Error)]
pub enum ArrowError {
    #[error("Arrow schema error: {0}")]
    ArrowSchemaError(#[from] arrow_schema::ArrowError),
    #[error("LanceDB error: {0}")]
    LanceError(String),
    #[error("Library not found")]
    LibNotFoundError,
    #[error(transparent)]
    LLMError(#[from] LLMError),
    #[error("Path contains invalid UTF-8 characters")]
    PathEncodingError,
    #[error("{0}")]
    PdfParsingError(String),
    #[error("{0}")]
    Other(String),
}

impl From<lancedb::Error> for ArrowError {
    fn from(value: lancedb::Error) -> Self {
        Self::LanceError(value.to_string())
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

impl From<LanceError> for ArrowError {
    fn from(value: LanceError) -> Self {
        Self::LanceError(value.to_string())
    }
}

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
        arrow_schema::Field::new(DbFields::LibraryKey, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::Title, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::FilePath, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::PdfText, arrow_schema::DataType::Utf8, false),
    ];

    if lancedb_exists().await {
        schema_fields.push(arrow_schema::Field::new(
            DbFields::Embeddings,
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                get_embedding_dims_by_provider(embedding_name) as i32,
            ),
            false,
        ));
    }
    arrow_schema::Schema::new(schema_fields)
}

/// A helper that converts an arbitrary `Vec<ZoteroItem>` into a `RecordBatch`. Note that because
/// we need the embedding provider as well, this can't be done with just a `From<..>`
/// implementation.
///
/// # Arguments:
///
/// * `items` - The items to convert to a `RecordBatch`
/// * `embedding_config` - Configuration for the embedding provider to use when computing embeddings.
///
/// # Returns
///
/// A `RecordBatch` that can be used to interact with LanceDB.
pub async fn library_to_arrow(
    items: Vec<ZoteroItem>,
    embedding_config: EmbeddingProviderConfig,
) -> Result<RecordBatch, ArrowError> {
    let schema = Arc::new(get_schema(embedding_config.provider_name()).await);

    // Convert ZoteroItemMetadata to Arrow arrays
    let library_keys = StringArray::from(
        items
            .iter()
            .map(|item| item.metadata.library_key.as_str())
            .collect::<Vec<&str>>(),
    );

    let titles = StringArray::from(
        items
            .iter()
            .map(|item| item.metadata.title.as_str())
            .collect::<Vec<&str>>(),
    );

    let pdf_texts = StringArray::from(
        items
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<&str>>(),
    );

    // Convert file paths to strings, returning an error if any path has invalid UTF-8
    let file_paths_vec: Result<Vec<&str>, ArrowError> = items
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
        Arc::new(pdf_texts.clone()) as ArrayRef,
    ];

    if lancedb_exists().await {
        let embedding_provider = get_embedding_provider_with_config(embedding_config)?;
        let query_vec = embedding_provider.compute_source_embeddings(Arc::new(pdf_texts))?;
        let query_vec = query_vec.as_fixed_size_list();

        record_batch_cols.push(Arc::new(query_vec.clone()));
    }
    let record_batch = RecordBatch::try_new(schema.clone(), record_batch_cols)?;

    Ok(record_batch)
}

/// Converts new Zotero library items to an Arrow RecordBatch.
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
/// * `config` - Configuration containing embedding provider information.
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
pub async fn full_library_to_arrow(
    config: &Config,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<RecordBatch, ArrowError> {
    let lib_items = parse_library(&config.embedding_provider, start_from, limit).await?;
    log::info!("Finished parsing library items.");

    library_to_arrow(
        lib_items,
        config.get_embedding_config().ok_or(ArrowError::Other(
            "Failed to get embedding config from application config".to_string(),
        ))?,
    )
    .await
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
/// This function also uses a reranking provider to perform reranking of the vector search results.
///
/// # Arguments
///
/// * `query` - The query to search the LanceDB table for.
/// * `embedding_name` - The embedding method to use. Must be one of `EmbeddingProviders`. Note
///   that this must be the same embedding provider used when initially creating the database.
/// * `reranker` - The reranker provider to use.
///
/// # Returns
///
/// A `Vec<ZoteroItem>` containing the resulting items from the Zotero library. Returns an
/// `ArrowError` that wraps the underlying `LanceError` if the `rag` crate's `vector_search` is
/// unsuccessful for any reason.
pub async fn vector_search(
    query: String,
    embedding_name: &str,
    reranker: String,
) -> Result<Vec<ZoteroItem>, ArrowError> {
    let batches = rag_vector_search(query.clone(), embedding_name).await?;

    let items: ZoteroItemSet = batches.into();
    let items: Vec<ZoteroItem> = items.into();

    let rerank_provider = get_reranking_provider::<ZoteroItem>(&reranker)?;
    let items = rerank_provider.rerank(items, &query).await?;

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

        let record_batch = full_library_to_arrow("voyageai", Some(0), Some(5)).await;
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
