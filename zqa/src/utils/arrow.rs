use std::{path::PathBuf, sync::Arc};

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray, cast::AsArray,
};
use arrow_schema;
use thiserror::Error;
use zqa_rag::{
    capabilities::EmbeddingProvider,
    embedding::common::{
        EmbeddingProviderConfig, get_embedding_dims_by_provider, get_embedding_provider_with_config,
    },
    llm::errors::LLMError,
    vector::backends::lance::{LANCE_DATA_TABLE_NAME, LanceError, get_db_uri},
};

use super::library::{LibraryParsingError, parse_library};
use crate::{store::lance::LanceZoteroStore, utils::library::ZoteroItem};

/// An enum containing the fields stored by our application in `LanceDB`, in order. Implementations
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
/// for the application to `LanceDB`, PDF parsing, and other lower-level operations. As errors from
/// those functions propagate, they are captured here. In general, this enum should not be used
/// outside this file, except perhaps to `impl From<ArrowError>`.
#[derive(Debug, Error)]
pub enum ArrowError {
    #[error("Arrow schema error: {0}")]
    ArrowSchemaError(#[from] arrow_schema::ArrowError),
    #[error("LanceDB error: {0}")]
    LanceError(String),
    #[error("SQLite error: {0}")]
    SqliteError(String),
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
            LibraryParsingError::SqlError(msg) => Self::SqliteError(msg),
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

/// Checks whether the configured LanceDB database exists and contains the expected table.
pub(crate) async fn lancedb_exists() -> bool {
    let uri = get_db_uri();
    if !PathBuf::from(&uri).exists() {
        return false;
    }

    if let Ok(db) = lancedb::connect(&uri).execute().await {
        db.open_table(LANCE_DATA_TABLE_NAME).execute().await.is_ok()
    } else {
        false
    }
}

/// Get the schema for our `LanceDB` table. This is required for both getting library items and
/// checkhealth.
///
/// # Arguments
///
/// * `embedding_provider` - The embedding used by the current DB.
/// * `include_embeddings` - Whether to include the embeddings field in the schema.
///
/// # Returns
///
/// The schema in Arrow format.
#[must_use]
pub fn get_schema(
    embedding_provider: EmbeddingProvider,
    include_embeddings: bool,
) -> arrow_schema::Schema {
    // Convert ZoteroItemMetadata to something that can be converted to Arrow
    // Need to extract fields and create appropriate Arrow arrays
    let mut schema_fields = vec![
        arrow_schema::Field::new(DbFields::LibraryKey, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::Title, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::FilePath, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::PdfText, arrow_schema::DataType::Utf8, false),
    ];

    if include_embeddings {
        schema_fields.push(arrow_schema::Field::new(
            DbFields::Embeddings,
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                get_embedding_dims_by_provider(embedding_provider) as i32,
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
/// * `include_embeddings` - Whether to include the embeddings field in the schema.
///
/// # Errors
///
/// * `ArrowError::PathEncodingError` if a Zotero item's path is not valid Unicode.
/// * `ArrowError::LLMError` if the embedding provider could not be obtained or embedding fails.
/// * `ArrowError::ArrowSchemaError` if creating the final `RecordBatch` fails.
///
/// # Returns
///
/// A `RecordBatch` that can be used to interact with `LanceDB`.
pub fn library_to_arrow(
    items: &[ZoteroItem],
    embedding_config: &EmbeddingProviderConfig,
    include_embeddings: bool,
) -> Result<RecordBatch, ArrowError> {
    let schema = Arc::new(get_schema(embedding_config.provider(), include_embeddings));

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

    if include_embeddings {
        let embedding_provider = get_embedding_provider_with_config(embedding_config)?;
        let query_vec = embedding_provider.compute_source_embeddings(Arc::new(pdf_texts))?;
        let query_vec = query_vec.as_fixed_size_list();

        record_batch_cols.push(Arc::new(query_vec.clone()));
    }
    let record_batch = RecordBatch::try_new(schema.clone(), record_batch_cols)?;

    Ok(record_batch)
}

/// Converts new Zotero library items to an Arrow `RecordBatch`.
///
/// This function parses the Zotero library using `parse_library()` and converts
/// the resulting `ZoteroItemMetadata` entries into a structured Arrow `RecordBatch`.
/// The `RecordBatch` contains the following columns:
/// - `library_key`: The unique key for each item in the Zotero library
/// - title: The title of the paper/document
/// - abstract: The abstract of the paper (optional)
/// - notes: Any notes associated with the item (optional)
/// - `file_path`: Path to the document file
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
/// * `store` - [`LanceZoteroStore`] with configuration
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
pub async fn full_library_to_arrow(
    store: &LanceZoteroStore,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<RecordBatch, ArrowError> {
    let lib_items = parse_library(store, start_from, limit).await?;
    log::info!("Finished parsing library items.");

    let include_embeddings = lancedb_exists().await;
    library_to_arrow(
        &lib_items,
        &store.get_embedding_config(),
        include_embeddings,
    )
}

/// Given metadata about Zotero items, *including embeddings*, inserts them into the LanceDB store.
///
/// This function does not check that the metadata provided correspond to items in Zotero; nor does
/// it query Zotero at all. This function is a "manual" alternative to [`library_to_arrow`] when you
/// already have embeddings. This function also assumes that the array slices passed all have the
/// same length, and that for any $i$, the $i$th element of each array corresponds to the same item.
///
/// # Arguments
///
/// * `library_keys` - The library keys of the items
/// * `titles` - The titles of the items
/// * `file_paths` - The UTF-8 file paths to the items
/// * `pdf_texts` - The full texts for each item
/// * `embeddings` - The embeddings for each element
/// * `embedding_config` - The embedding config
///
/// # Returns
///
/// An Arrow [`RecordBatch`] with the Zotero schema and passed items.
///
/// # Errors
///
/// * `ArrowError::Other` - if the lengths of the arrays do not match, or any of the embedding
///   vectors do not have the dimensions configured in `embedding_config`.
/// * `ArrowError::ArrowSchemaError` - if the schema does not match the items.
pub fn library_to_arrow_with_embeddings(
    library_keys: &[&str],
    titles: &[&str],
    file_paths: &[&str],
    pdf_texts: &[&str],
    embeddings: Vec<Vec<f32>>,
    embedding_config: &EmbeddingProviderConfig,
) -> Result<RecordBatch, ArrowError> {
    if library_keys.len() != titles.len()
        || library_keys.len() != file_paths.len()
        || library_keys.len() != pdf_texts.len()
        || library_keys.len() != embeddings.len()
    {
        return Err(ArrowError::Other(
            "Passed slices do not have equal lengths.".into(),
        ));
    }

    let expected_dim = embedding_config.dims();
    if embeddings.iter().any(|e| e.len() != expected_dim) {
        return Err(ArrowError::Other(format!(
            "All embeddings must have dimension {expected_dim}."
        )));
    }

    let schema = Arc::new(get_schema(embedding_config.provider(), true));
    let library_keys = StringArray::from(Vec::from(library_keys));
    let titles = StringArray::from(Vec::from(titles));
    let pdf_texts = StringArray::from(Vec::from(pdf_texts));
    let file_paths = StringArray::from(Vec::from(file_paths));

    let flattened_embeddings = embeddings.into_iter().flatten().collect::<Vec<_>>();
    let embeddings_array = Float32Array::from(flattened_embeddings);
    let field = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        true,
    ));

    #[allow(clippy::cast_possible_truncation)]
    let embeddings_array =
        FixedSizeListArray::new(field, expected_dim as i32, Arc::new(embeddings_array), None);

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(library_keys),
            Arc::new(titles),
            Arc::new(file_paths),
            Arc::new(pdf_texts),
            Arc::new(embeddings_array),
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use arrow_array::RecordBatchIterator;
    use dotenv::dotenv;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };

    use super::*;
    use crate::{
        common::setup_logger,
        config::{Config, VoyageAIConfig},
    };

    fn get_config() -> Config {
        let mut config = Config {
            voyageai: Some(VoyageAIConfig {
                reranker: Some(DEFAULT_VOYAGE_RERANK_MODEL.into()),
                embedding_model: Some(DEFAULT_VOYAGE_EMBEDDING_MODEL.into()),
                embedding_dims: Some(DEFAULT_VOYAGE_EMBEDDING_DIM as usize),
                api_key: Some(String::new()),
            }),
            ..Default::default()
        };

        config.read_env().unwrap();
        config
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_library_to_arrow_works() {
        dotenv().ok();
        let _ = setup_logger(log::LevelFilter::Info);

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let config = get_config();

        let record_batch = temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], async {
            let embedding_config = config.get_embedding_config().unwrap();
            let schema = Arc::new(get_schema(embedding_config.provider(), true));
            let store = LanceZoteroStore::from_schema(embedding_config, schema);
            full_library_to_arrow(&store, Some(0), Some(5)).await
        })
        .await;

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
