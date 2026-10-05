use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema;
use thiserror::Error;
use zqa_rag::embedding::common::{EmbeddingProviderConfig, get_embedding_provider_with_config};
use zqa_rag::llm::errors::LLMError;
use zqa_rag::vector::backends::lance::LanceError;

use super::library::{LibraryParsingError, parse_library};
use crate::store::lance::LanceZoteroStore;
use crate::utils::library::ZoteroItem;

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

/// Get the schema for our `LanceDB` table using the configured embedding dimensions.
///
/// This is required for both getting library items and checkhealth.
///
/// # Arguments
///
/// * `embedding_config` - The embedding provider and its configuration.
///
/// # Returns
///
/// The schema in Arrow format.
///
/// # Panics
///
/// * If the embedding dimensions exceed Arrow's `i32` limit.
#[must_use]
pub fn get_schema(embedding_config: &EmbeddingProviderConfig) -> arrow_schema::Schema {
    // Convert ZoteroItemMetadata to something that can be converted to Arrow
    // Need to extract fields and create appropriate Arrow arrays
    arrow_schema::Schema::new(vec![
        arrow_schema::Field::new(DbFields::LibraryKey, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::Title, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::FilePath, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(DbFields::PdfText, arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(
            DbFields::Embeddings,
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                i32::try_from(embedding_config.dims())
                    .expect("Embedding dimensions exceed Arrow's i32 limit"),
            ),
            false,
        ),
    ])
}

/// A helper that converts an arbitrary `Vec<ZoteroItem>` into a `RecordBatch`, computing their
/// embeddings. Note that because we need the embedding provider as well, this can't be done with
/// just a `From<..>` implementation.
///
/// Items the embedding provider could not embed are left out of the batch rather than stored with
/// a placeholder vector. Since they are not in the store, the next `/process` picks them up again.
///
/// # Arguments:
///
/// * `items` - The items to convert to a `RecordBatch`
/// * `embedding_config` - Configuration for the embedding provider to use when computing embeddings.
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
) -> Result<RecordBatch, ArrowError> {
    let embedding_provider = get_embedding_provider_with_config(embedding_config)?;
    let pdf_texts = StringArray::from(
        items
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<&str>>(),
    );
    let embeddings = embedding_provider.compute_source_embeddings(Arc::new(pdf_texts))?;

    embedded_items_to_arrow(items, embeddings.as_fixed_size_list(), embedding_config)
}

/// Build a `RecordBatch` from items and their computed embeddings, leaving out the items whose
/// embedding is null because the provider failed to embed them.
///
/// # Arguments
///
/// * `items` - The items to convert to a `RecordBatch`
/// * `embeddings` - The embeddings for `items`, in the same order, with a null entry for each item
///   that could not be embedded.
/// * `embedding_config` - The embedding config, which determines the schema.
///
/// # Errors
///
/// * `ArrowError::PathEncodingError` if a Zotero item's path is not valid Unicode.
/// * `ArrowError::Other` if the number or dimension of the embeddings does not match the items
///   and the config.
/// * `ArrowError::ArrowSchemaError` if creating the final `RecordBatch` fails.
///
/// # Returns
///
/// A `RecordBatch` containing only the items that were embedded.
fn embedded_items_to_arrow(
    items: &[ZoteroItem],
    embeddings: &FixedSizeListArray,
    embedding_config: &EmbeddingProviderConfig,
) -> Result<RecordBatch, ArrowError> {
    if embeddings.len() != items.len() {
        return Err(ArrowError::Other(format!(
            "Got {} embeddings for {} items.",
            embeddings.len(),
            items.len()
        )));
    }
    let dim = embedding_config.dims();
    if usize::try_from(embeddings.value_length()).ok() != Some(dim) {
        return Err(ArrowError::Other(format!(
            "All embeddings must have dimension {dim}."
        )));
    }

    let (embedded, failed): (Vec<_>, Vec<_>) = items
        .iter()
        .enumerate()
        .partition(|&(i, _)| embeddings.is_valid(i));
    let values = embeddings.values().as_primitive::<Float32Type>();
    let values = if failed.is_empty() {
        // The common case: reuse the provider's buffer instead of copying it.
        values.clone()
    } else {
        log::warn!(
            "{} items could not be embedded and were not saved; they will be retried on the next run: {}",
            failed.len(),
            failed
                .iter()
                .map(|(_, item)| item.metadata.title.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        embedded
            .iter()
            .flat_map(|&(i, _)| &values.values()[i * dim..(i + 1) * dim])
            .copied()
            .collect::<Float32Array>()
    };

    let embedded: Vec<&ZoteroItem> = embedded.into_iter().map(|(_, item)| item).collect();
    let file_paths = embedded
        .iter()
        .map(|item| {
            item.metadata
                .file_path
                .to_str()
                .ok_or(ArrowError::PathEncodingError)
        })
        .collect::<Result<Vec<&str>, ArrowError>>()?;
    let library_keys: StringArray = embedded
        .iter()
        .map(|item| Some(item.metadata.library_key.as_str()))
        .collect();
    let titles: StringArray = embedded
        .iter()
        .map(|item| Some(item.metadata.title.as_str()))
        .collect();
    let pdf_texts: StringArray = embedded
        .iter()
        .map(|item| Some(item.text.as_str()))
        .collect();
    let embeddings = FixedSizeListArray::new(
        Arc::new(arrow_schema::Field::new(
            "item",
            arrow_schema::DataType::Float32,
            true,
        )),
        embeddings.value_length(),
        Arc::new(values),
        None,
    );

    Ok(RecordBatch::try_new(
        Arc::new(get_schema(embedding_config)),
        vec![
            Arc::new(library_keys),
            Arc::new(titles),
            Arc::new(StringArray::from(file_paths)),
            Arc::new(pdf_texts),
            Arc::new(embeddings),
        ],
    )?)
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
/// * `library_path` - An optional override for the Zotero library directory. When `None`, the path
///   is resolved from the environment.
/// * `start_from` - An optional offset for the SQL query. Useful for debugging, pagination,
///   multi-threading, etc.
/// * `limit` - Optional limit, meant to be used in conjunction with `start_from`.
pub async fn full_library_to_arrow(
    store: &LanceZoteroStore,
    library_path: Option<&std::path::Path>,
    start_from: Option<usize>,
    limit: Option<usize>,
) -> Result<RecordBatch, ArrowError> {
    let lib_items = parse_library(store, library_path, start_from, limit).await?;
    log::info!("Finished parsing library items.");

    library_to_arrow(&lib_items, &store.get_embedding_config())
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

    let schema = Arc::new(get_schema(embedding_config));
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
    use zqa_macros::test_eq;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };

    use super::*;
    use crate::common::setup_logger;
    use crate::config::{Config, VoyageAIConfig};
    use crate::store::common::ZoteroStore;
    use crate::utils::library::ZoteroItemSet;

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
        let db_uri = temp_dir.path().join("lancedb-table");

        let config = get_config();

        // Isolate this test: pin the store to a temp DB URI and read the toy library shipped in
        // `assets/`, so it touches no process-global `LANCEDB_URI`/`CI` state and is parallel-safe.
        let library_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("Zotero");
        let embedding_config = config.get_embedding_config().unwrap();
        let schema = Arc::new(get_schema(&embedding_config));
        let store = LanceZoteroStore::from_schema(embedding_config, schema)
            .with_uri(db_uri.to_str().unwrap());
        let record_batch = full_library_to_arrow(&store, Some(&library_path), Some(0), Some(5))
            .await
            .expect("Failed to fetch library");
        let schema = record_batch.schema();
        let batches = vec![Ok(record_batch)];
        let mut batch_iter = RecordBatchIterator::new(batches.into_iter(), schema);

        // Get the first batch
        let batch = batch_iter
            .next()
            .expect("No batches in iterator")
            .expect("Error in batch");

        test_eq!(batch.num_columns(), 5);
        assert!(
            (1..=5).contains(&batch.num_rows()),
            "Expected between one and five rows in record batch"
        );

        let mut item = ZoteroItemSet::from(vec![batch]).items.remove(0);
        // Empty text exercises configured zero embeddings without calling the live API.
        item.text.clear();

        for (embedding_config, expected_dims) in [
            (
                EmbeddingProviderConfig::Cohere(zqa_rag::config::CohereConfig {
                    api_key: "test-key".into(),
                    embedding_model: "embed-v4.0".into(),
                    embedding_dims: 256,
                    reranker: String::new(),
                    max_concurrent_requests: zqa_rag::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
                    max_retries: zqa_rag::constants::DEFAULT_MAX_RETRIES,
                }),
                256,
            ),
            (
                EmbeddingProviderConfig::VoyageAI(zqa_rag::config::VoyageAIConfig {
                    api_key: "test-key".into(),
                    embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
                    embedding_dims: 256,
                    reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
                    max_concurrent_requests: zqa_rag::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
                    max_retries: zqa_rag::constants::DEFAULT_MAX_RETRIES,
                }),
                256,
            ),
            (
                EmbeddingProviderConfig::Gemini(zqa_rag::config::GeminiConfig {
                    embedding_dims: 768,
                    ..Default::default()
                }),
                768,
            ),
        ] {
            let batch = library_to_arrow(&[], &embedding_config).unwrap();
            test_eq!(
                batch.column(4).as_fixed_size_list().value_length(),
                expected_dims
            );

            let uri = temp_dir.path().join(embedding_config.provider_name());
            let store = LanceZoteroStore::from_embedding_config(embedding_config.clone())
                .with_uri(uri.to_str().unwrap());

            if matches!(embedding_config, EmbeddingProviderConfig::Cohere(_)) {
                // Cover both initial table creation and ingestion into an existing table.
                store.upsert_items(vec![item.clone()]).await.unwrap();
                store.upsert_items(vec![item.clone()]).await.unwrap();
            } else {
                let source_batch = library_to_arrow(&[], &embedding_config).unwrap();
                store.upsert_batches(vec![source_batch]).await.unwrap();
                test_eq!(store.existing_item_metadata().await.unwrap().len(), 0);
            }

            let batch = library_to_arrow_with_embeddings(
                &["configured"],
                &["Configured embeddings"],
                &["paper.pdf"],
                &["Text with precomputed embeddings"],
                vec![vec![0.5; embedding_config.dims()]],
                &embedding_config,
            )
            .unwrap();
            test_eq!(
                batch.column(4).as_fixed_size_list().value_length(),
                expected_dims
            );
            store.upsert_batches(vec![batch]).await.unwrap();

            // The empty-text item is left out, so that `/process` parses it again.
            test_eq!(store.existing_item_metadata().await.unwrap().len(), 1);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_existing_item_metadata_leaves_out_rows_to_parse_again() {
        let temp_dir = tempfile::tempdir().unwrap();
        let embedding_config = EmbeddingProviderConfig::Cohere(zqa_rag::config::CohereConfig {
            api_key: "test-key".into(),
            embedding_model: "embed-v4.0".into(),
            embedding_dims: 8,
            reranker: String::new(),
            max_concurrent_requests: zqa_rag::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
            max_retries: zqa_rag::constants::DEFAULT_MAX_RETRIES,
        });
        let store = LanceZoteroStore::from_embedding_config(embedding_config.clone())
            .with_uri(temp_dir.path().to_str().unwrap());
        let dims = embedding_config.dims();

        // A legacy row with text but a zero vector (older versions stored failed embeddings this
        // way), an empty-text row, and a blank-text row with a nonzero vector are all left out.
        let batch = library_to_arrow_with_embeddings(
            &["ok", "legacy", "empty", "blank"],
            &["Embedded", "Legacy", "Empty", "Blank"],
            &["ok.pdf", "legacy.pdf", "empty.pdf", "blank.pdf"],
            &["Some text", "Text whose embedding failed", "", " \n "],
            vec![
                vec![0.5; dims],
                vec![0.0; dims],
                vec![0.0; dims],
                vec![0.5; dims],
            ],
            &embedding_config,
        )
        .unwrap();
        store.upsert_batches(vec![batch]).await.unwrap();
        let keys = |metadata: Vec<crate::utils::library::ZoteroItemMetadata>| {
            metadata
                .into_iter()
                .map(|m| m.library_key)
                .collect::<Vec<_>>()
        };
        test_eq!(
            keys(store.existing_item_metadata().await.unwrap()),
            vec!["ok"]
        );

        // Parsing the legacy item again replaces its stored row.
        let batch = library_to_arrow_with_embeddings(
            &["legacy"],
            &["Legacy"],
            &["legacy.pdf"],
            &["Text whose embedding failed"],
            vec![vec![0.5; dims]],
            &embedding_config,
        )
        .unwrap();
        store.upsert_batches(vec![batch]).await.unwrap();
        let mut existing = keys(store.existing_item_metadata().await.unwrap());
        existing.sort_unstable();
        test_eq!(existing, vec!["legacy", "ok"]);
    }

    #[test]
    fn test_embedded_items_to_arrow_skips_failed_items() {
        let embedding_config = EmbeddingProviderConfig::Gemini(zqa_rag::config::GeminiConfig {
            embedding_dims: 4,
            ..Default::default()
        });
        let item = |key: &str| ZoteroItem {
            metadata: crate::utils::library::ZoteroItemMetadata {
                library_key: key.into(),
                title: format!("Paper {key}"),
                file_path: "paper.pdf".into(),
                authors: None,
            },
            text: "Some text".into(),
        };

        // The second item's embedding is null because the provider failed to embed it.
        let embeddings = FixedSizeListArray::new(
            Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                true,
            )),
            4,
            Arc::new(Float32Array::from(vec![0.5; 12])),
            Some(vec![true, false, true].into()),
        );
        let batch = embedded_items_to_arrow(
            &[item("a"), item("b"), item("c")],
            &embeddings,
            &embedding_config,
        )
        .unwrap();

        let keys = batch.column(0).as_string::<i32>();
        test_eq!(keys.iter().flatten().collect::<Vec<_>>(), vec!["a", "c"]);
        test_eq!(batch.column(4).null_count(), 0);

        // Mismatched lengths are rejected rather than misaligning items and embeddings.
        assert!(embedded_items_to_arrow(&[item("a")], &embeddings, &embedding_config).is_err());
    }
}
