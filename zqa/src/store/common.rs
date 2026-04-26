use async_trait::async_trait;

use crate::utils::library::ZoteroItem;

#[async_trait]
pub trait ZoteroStore {
    type StoreError: std::error::Error + Send + Sync;

    async fn exists(&self) -> bool;
    async fn upsert_items(&self, items: Vec<ZoteroItem>) -> Result<(), Self::StoreError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<ZoteroItem>, Self::StoreError>;
    async fn get_items_by_keys(&self, keys: &[String])
    -> Result<Vec<ZoteroItem>, Self::StoreError>;
    async fn delete_by_library_keys(&self, keys: &[String]) -> Result<(), Self::StoreError>;
    async fn dedup_by_title(&self) -> Result<usize, Self::StoreError>;
}
