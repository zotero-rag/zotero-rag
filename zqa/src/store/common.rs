use async_trait::async_trait;
use zqa_rag::reranking::common::RerankProviderConfig;

use crate::{
    store::lance::VectorSearchStats,
    utils::library::{ZoteroItem, ZoteroItemMetadata},
};

/// An application-level trait for Zotero store implementations.
#[async_trait]
pub trait ZoteroStore: Send + Sync {
    /// The error type returned by store operations.
    type StoreError: std::error::Error + Send + Sync;
    /// The metadata type associated with the store.
    type Metadata;

    /// Returns `true` if the store exists, `false` otherwise. Useful to check that the store is
    /// configured correctly.
    async fn exists(&self) -> bool;
    /// Returns the metadata associated with the store.
    async fn get_metadata(&self) -> Result<Self::Metadata, Self::StoreError>;
    /// Returns the metadata for all existing items in the store. This is useful for operations
    /// such as set differences (e.g., finding newly-added items).
    async fn existing_item_metadata(&self) -> Result<Vec<ZoteroItemMetadata>, Self::StoreError>;
    /// Performs a vector search on the store, returning the top `limit` results.
    async fn vector_search(
        &self,
        query: String,
        limit: usize,
        reranker_config: Option<&RerankProviderConfig>,
    ) -> Result<(Vec<ZoteroItem>, VectorSearchStats), Self::StoreError>;
    /// Upserts the given items into the store.
    async fn upsert_items(&self, items: Vec<ZoteroItem>) -> Result<(), Self::StoreError>;
    /// Searches the store for items matching the given query, returning the top `limit` results.
    /// This variant does not perform reranking.
    async fn vector_search_raw(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ZoteroItem>, Self::StoreError>;
    /// Returns the items with the given keys from the store. This is useful for retrieving
    /// items by their library keys without performing a full text search.
    async fn get_items_by_keys(&self, keys: &[String])
    -> Result<Vec<ZoteroItem>, Self::StoreError>;
    /// Deletes the items with the given keys from the store.
    async fn delete_by_library_keys(&self, keys: &[String]) -> Result<(), Self::StoreError>;
    /// Deletes duplicate items from the store based on their title. Returns the number of items
    /// deleted.
    async fn dedup_by_title(&self) -> Result<usize, Self::StoreError>;
}
