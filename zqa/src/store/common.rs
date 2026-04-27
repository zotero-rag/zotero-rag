use async_trait::async_trait;
use zqa_rag::reranking::common::RerankProviderConfig;

use crate::{
    store::lance::VectorSearchStats,
    utils::library::{ZoteroItem, ZoteroItemMetadata},
};

#[async_trait]
pub trait ZoteroStore {
    type StoreError: std::error::Error + Send + Sync;
    type Metadata;

    async fn exists(&self) -> bool;
    async fn get_metadata(&self) -> Result<Self::Metadata, Self::StoreError>;
    async fn existing_item_metadata(&self) -> Result<Vec<ZoteroItemMetadata>, Self::StoreError>;
    async fn vector_search(
        &self,
        query: String,
        limit: usize,
        reranker_config: Option<&RerankProviderConfig>,
    ) -> Result<(Vec<ZoteroItem>, VectorSearchStats), Self::StoreError>;
    async fn upsert_items(&self, items: Vec<ZoteroItem>) -> Result<(), Self::StoreError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<ZoteroItem>, Self::StoreError>;
    async fn get_items_by_keys(&self, keys: &[String])
    -> Result<Vec<ZoteroItem>, Self::StoreError>;
    async fn delete_by_library_keys(&self, keys: &[String]) -> Result<(), Self::StoreError>;
    async fn dedup_by_title(&self) -> Result<usize, Self::StoreError>;
}
