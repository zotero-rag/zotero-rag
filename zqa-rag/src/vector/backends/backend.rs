//! Traits for vector database backends.

use async_trait::async_trait;

use crate::providers::ProviderId;

/// Registration methods for embedding providers to interface with the backend.
pub trait VectorBackendRegistrar: VectorBackend + Send + Sync {
    /// Get the provider ID for this object
    fn provider_id(&self) -> ProviderId;

    /// Register this provider with the vector backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the registration fails.
    fn register(&self, db: &Self::Connection, config: &Self::Config) -> Result<(), Self::Error>;
}

/// A vector database backend.
#[async_trait]
pub trait VectorBackend: Send + Sync {
    /// The record type for the backend.
    type Record: Send;
    /// The error type for the backend.
    type Error: std::error::Error + Send;
    /// The configuration type for the backend. A backend may not have a config at all, in which
    /// case the [`Config`] type should be `()`.
    type Config: Send + Sync;
    /// The connection type for the backend.
    type Connection: Send + Sync;

    /// Returns the *base* path for the DB. This can be path on the local disk for LanceDB's
    /// database, a connection URL, etc.
    #[must_use]
    fn get_db_path(&self) -> String;

    /// Whether the path specified by [`get_db_path`] exists.
    #[must_use]
    async fn db_exists(&self) -> bool;

    /// Create indices for the database. The details of the type of index are left to the trait
    /// implementer. If the index exists, then it is updated instead.
    ///
    /// # Arguments
    ///
    /// * `text_col` - The name of the text column to index.
    /// * `embedding_col` - The name of the embedding column to index.
    async fn create_or_update_indices(
        &self,
        text_col: &str,
        embedding_col: &str,
    ) -> Result<(), Self::Error>;

    /// Connect to the database.
    async fn connect(&self) -> Result<Self::Connection, Self::Error>;

    /// Delete rows from the database based on the given column and keys.
    ///
    /// # Arguments
    ///
    /// * `col` - The name of the column to delete rows from.
    /// * `keys` - The keys of the rows to delete. The values must correspond to values in the
    ///   `col` column.
    async fn delete_rows(&self, col: &str, keys: &[String]) -> Result<(), Self::Error>;

    /// Deduplicate rows in the database based on the given column and key.
    ///
    /// # Arguments
    ///
    /// * `by` - The column to deduplicate rows by.
    /// * `key` - The column to use as the key for deletion.
    ///
    /// # Returns
    ///
    /// The number of rows deleted.
    async fn dedup_rows(&self, by: &str, key: &str) -> Result<usize, Self::Error>;

    /// Get all items from the database.
    ///
    /// # Arguments
    ///
    /// * `columns` - The columns to return.
    ///
    /// # Returns
    ///
    /// A vector of all items in the database.
    async fn get_items(&self, columns: &[String]) -> Result<Vec<Self::Record>, Self::Error>;

    /// Search for items in the database using a vector query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search for.
    /// * `limit` - The maximum number of results to return.
    ///
    /// # Returns
    ///
    /// A vector of items that match the query.
    async fn vector_search(
        &self,
        query: String,
        limit: usize,
    ) -> Result<Vec<Self::Record>, Self::Error>;

    /// Search for an item in the database by its key.
    ///
    /// # Arguments
    ///
    /// * `key_col` - The column to use as the key.
    /// * `key` - The key to search for.
    ///
    /// # Returns
    ///
    /// The item that matches the key, if one exists.
    async fn search_by_key(
        &self,
        key_col: &str,
        key: &str,
    ) -> Result<Option<Self::Record>, Self::Error>;

    /// Search for items in the database by a column value.
    ///
    /// # Arguments
    ///
    /// * `col` - The column to search by.
    /// * `values` - The values to search for.
    ///
    /// # Returns
    ///
    /// A vector of items that match the column value.
    async fn search_by_column(
        &self,
        col: &str,
        values: &[String],
    ) -> Result<Vec<Self::Record>, Self::Error>;

    /// Insert items into the database.
    ///
    /// # Arguments
    ///
    /// * `items` - The items to insert.
    /// * `merge_on` - `None` if you want to create or overwrite the current database; otherwise, a
    ///   reference to an array of keys to merge on.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the items were inserted successfully, or an error otherwise.
    async fn insert_items(
        &self,
        items: Vec<Self::Record>,
        merge_on: Option<&[&str]>,
    ) -> Result<(), Self::Error>;
}
