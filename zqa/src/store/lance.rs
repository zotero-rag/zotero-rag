use std::sync::Arc;

use arrow_array::RecordBatch;
use async_trait::async_trait;
use zqa_rag::vector::backends::{
    backend::VectorBackend,
    lance::{LanceBackend, LanceMetadata},
};

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
pub(crate) struct LanceZoteroStore {
    backend: LanceBackend,
    embedding_config: zqa_rag::embedding::common::EmbeddingProviderConfig,
}

impl LanceZoteroStore {
    /// Create a new Lance-backed Zotero store from an existing backend and embedding config.
    #[must_use]
    pub(crate) fn new(
        backend: LanceBackend,
        embedding_config: zqa_rag::embedding::common::EmbeddingProviderConfig,
    ) -> Self {
        Self {
            backend,
            embedding_config,
        }
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
        let schema = get_schema(embedding_config.provider()).await;
        let backend = LanceBackend::new(
            embedding_config.clone(),
            Arc::new(schema),
            DbFields::PdfText.as_ref().to_string(),
        );

        Ok(Self::new(backend, embedding_config))
    }

    /// Return metadata for the underlying LanceDB table.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if LanceDB metadata could not be read.
    pub(crate) async fn metadata(&self) -> Result<LanceMetadata, CLIError> {
        self.backend.get_metadata().await.map_err(Into::into)
    }

    /// Upsert Arrow record batches into the LanceDB table by Zotero library key.
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

    async fn exists(&self) -> bool {
        self.backend.db_exists().await
    }

    async fn upsert_items(&self, items: Vec<ZoteroItem>) -> Result<(), Self::StoreError> {
        let batch = library_to_arrow(items, self.embedding_config.clone()).await?;
        self.upsert_batches(vec![batch]).await
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<ZoteroItem>, Self::StoreError> {
        let batches = self.backend.vector_search(query.to_string(), limit).await?;
        Ok(ZoteroItemSet::from(batches).into())
    }

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

    async fn delete_by_library_keys(&self, keys: &[String]) -> Result<(), Self::StoreError> {
        self.backend
            .delete_rows(DbFields::LibraryKey.as_ref(), keys)
            .await
            .map_err(Into::into)
    }

    async fn dedup_by_title(&self) -> Result<usize, Self::StoreError> {
        self.backend
            .dedup_rows(DbFields::Title.as_ref(), DbFields::LibraryKey.as_ref())
            .await
            .map_err(Into::into)
    }
}
