use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use zqa_rag::{
    embedding::common::EmbeddingProviderConfig,
    reranking::common::{RerankProviderConfig, get_reranking_provider_with_config},
    vector::backends::{
        backend::VectorBackend,
        lance::{LanceBackend, LanceMetadata},
    },
};

use crate::store::common::VectorSearchStats;
use crate::{
    cli::errors::CLIError,
    config::Config,
    store::common::ZoteroStore,
    utils::{
        arrow::{DbFields, get_schema, library_to_arrow},
        library::{ZoteroItem, ZoteroItemSet},
    },
};

/// Zotero-specific store backed by LanceDB.
#[derive(Clone)]
pub struct LanceZoteroStore {
    backend: LanceBackend,
    embedding_config: EmbeddingProviderConfig,
}

impl LanceZoteroStore {
    /// Create a new Lance-backed Zotero store from an existing backend and embedding config.
    #[must_use]
    fn new(backend: LanceBackend, embedding_config: EmbeddingProviderConfig) -> Self {
        Self {
            backend,
            embedding_config,
        }
    }

    /// Create a Lance-backed Zotero store from an embedding config and Arrow schema.
    #[must_use]
    pub fn from_schema(embedding_config: EmbeddingProviderConfig, schema: Arc<Schema>) -> Self {
        let backend = LanceBackend::new(
            embedding_config.clone(),
            schema,
            DbFields::PdfText.as_ref().to_string(),
        );

        Self::new(backend, embedding_config)
    }

    /// Get a read-only embedding config
    #[must_use]
    pub fn get_embedding_config(&self) -> EmbeddingProviderConfig {
        self.embedding_config.clone()
    }

    /// Create a Lance-backed Zotero store from an embedding configuration.
    pub async fn from_embedding_config(embedding_config: EmbeddingProviderConfig) -> Self {
        let schema = Arc::new(get_schema(embedding_config.provider(), true).await);
        Self::from_schema(embedding_config, schema)
    }

    /// Create a Lance-backed Zotero store from the application config.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if no embedding configuration is available.
    pub(crate) async fn from_config(config: &Config) -> Result<Self, CLIError> {
        let embedding_config = config.get_embedding_config().ok_or(CLIError::ConfigError(
            "Could not get embedding config".into(),
        ))?;

        Ok(Self::from_embedding_config(embedding_config).await)
    }

    /// Upsert Arrow record batches into the LanceDB table by Zotero library key.
    ///
    /// TODO: We should probably deprecate this at some point in favor of the `upsert_items` from
    /// the trait. I'm keeping this around for now to keep refactor scopes relatively manageable.
    /// Ideally, we would not have any Lance-specific architecture, but currently, commands such as
    /// `/process` rely on this.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if LanceDB insertion fails.
    pub(crate) async fn upsert_batches(&self, batches: Vec<RecordBatch>) -> Result<(), CLIError> {
        self.backend
            .insert_items(batches, Some(&[DbFields::LibraryKey.as_ref()]))
            .await
            .map_err(Into::into)
    }

    /// Create or update retrieval indices for the LanceDB table.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if index creation or update fails.
    pub(crate) async fn create_or_update_indices(&self) -> Result<(), CLIError> {
        self.backend
            .create_or_update_indices(DbFields::PdfText.as_ref(), DbFields::Embeddings.as_ref())
            .await
            .map_err(Into::into)
    }
}

#[async_trait]
impl ZoteroStore for LanceZoteroStore {
    type StoreError = CLIError;
    type Metadata = LanceMetadata;

    async fn exists(&self) -> bool {
        self.backend.db_exists().await
    }

    /// Perform vector search and optional reranking.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if search or reranking fails.
    async fn vector_search(
        &self,
        query: String,
        limit: usize,
        reranker_config: Option<&RerankProviderConfig>,
    ) -> Result<(Vec<ZoteroItem>, VectorSearchStats), CLIError> {
        let embedding_tokens = query.len();
        let items = <Self as ZoteroStore>::vector_search_raw(self, &query, limit).await?;

        let filtered_items: Vec<ZoteroItem> = items
            .into_iter()
            .filter(|item| !item.text.trim().is_empty())
            .collect();

        if filtered_items.is_empty() {
            return Ok((
                Vec::new(),
                VectorSearchStats {
                    embedding_tokens,
                    rerank_tokens: 0,
                },
            ));
        }

        let Some(reranker) = reranker_config else {
            return Ok((
                filtered_items,
                VectorSearchStats {
                    embedding_tokens,
                    rerank_tokens: 0,
                },
            ));
        };

        let rerank_provider = get_reranking_provider_with_config(reranker)?;
        let item_strings = filtered_items
            .iter()
            .map(|f| f.text.as_str())
            .collect::<Vec<_>>();

        let rerank_tokens = item_strings.iter().map(|s| s.len()).sum::<usize>() + query.len();
        let indices = rerank_provider.rerank(&item_strings, &query).await?;

        let reranked_items = indices
            .into_iter()
            .filter_map(|idx| filtered_items.get(idx).cloned())
            .collect();

        Ok((
            reranked_items,
            VectorSearchStats {
                embedding_tokens,
                rerank_tokens,
            },
        ))
    }

    /// Return metadata for Zotero items that already exist in the store.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if the existing rows cannot be fetched.
    async fn existing_item_metadata(
        &self,
    ) -> Result<Vec<crate::utils::library::ZoteroItemMetadata>, CLIError> {
        let db_items = self
            .backend
            .get_items(&[
                DbFields::LibraryKey.into(),
                DbFields::Title.into(),
                DbFields::FilePath.into(),
            ])
            .await?;

        Ok(db_items
            .iter()
            .flat_map(|batch| {
                let library_keys = crate::utils::library::get_column_from_batch(batch, 0);
                let titles = crate::utils::library::get_column_from_batch(batch, 1);
                let file_paths = crate::utils::library::get_column_from_batch(batch, 2);

                crate::izip!(library_keys, titles, file_paths)
                    .map(
                        |(key, title, path)| crate::utils::library::ZoteroItemMetadata {
                            library_key: key,
                            title,
                            file_path: std::path::PathBuf::from(path),
                            authors: None,
                        },
                    )
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    /// Return metadata for the underlying LanceDB table.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if LanceDB metadata could not be read.
    async fn get_metadata(&self) -> Result<LanceMetadata, CLIError> {
        self.backend.get_metadata().await.map_err(Into::into)
    }

    /// Upserts the given items into the store.
    ///
    /// # Arguments
    ///
    /// * `items` - The items to upsert.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if the upsert fails.
    async fn upsert_items(&self, items: Vec<ZoteroItem>) -> Result<(), Self::StoreError> {
        let include_embeddings = self.exists().await;
        let batch =
            library_to_arrow(items, self.embedding_config.clone(), include_embeddings).await?;
        self.upsert_batches(vec![batch]).await
    }

    /// Performs a raw vector search on the store, returning the top `limit` results.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string.
    /// * `limit` - The maximum number of results to return.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if the search fails.
    async fn vector_search_raw(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ZoteroItem>, Self::StoreError> {
        let batches = self.backend.vector_search(query.to_string(), limit).await?;
        Ok(ZoteroItemSet::from(batches).into())
    }

    /// Returns the items with the given keys from the store.
    ///
    /// # Arguments
    ///
    /// * `keys` - The keys of the items to return.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if the search fails.
    async fn get_items_by_keys(
        &self,
        keys: &[String],
    ) -> Result<Vec<ZoteroItem>, Self::StoreError> {
        let batches = self
            .backend
            .search_by_column(DbFields::LibraryKey.as_ref(), keys)
            .await?;
        Ok(ZoteroItemSet::from(batches).into())
    }

    /// Deletes the items with the given keys from the store.
    ///
    /// # Arguments
    ///
    /// * `keys` - The keys of the items to delete.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if the deletion fails.
    async fn delete_by_library_keys(&self, keys: &[String]) -> Result<(), Self::StoreError> {
        self.backend
            .delete_rows(DbFields::LibraryKey.as_ref(), keys)
            .await
            .map_err(Into::into)
    }

    /// Deduplicates items in the store by title, keeping the first occurrence.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if the deduplication fails.
    async fn dedup_by_title(&self) -> Result<usize, Self::StoreError> {
        self.backend
            .dedup_rows(DbFields::Title.as_ref(), DbFields::LibraryKey.as_ref())
            .await
            .map_err(Into::into)
    }
}
