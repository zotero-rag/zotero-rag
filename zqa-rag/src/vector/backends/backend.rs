//! Traits for vector database backends.

use std::fmt::Display;

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
pub trait VectorBackend {
    /// The record type for the backend.
    type Record;
    /// The error type for the backend.
    type Error;
    /// The configuration type for the backend. A backend may not have a config at all, in which
    /// case the [`Config`] type should be `()`.
    type Config;
    /// The connection type for the backend.
    type Connection;

    /// Returns the *base* path for the DB. This can be path on the local disk for LanceDB's
    /// database, a connection URL, etc.
    #[must_use]
    fn get_db_path(&self) -> String;

    /// Whether the path specified by [`get_db_path`] exists.
    #[must_use]
    fn db_exists(&self) -> impl Future<Output = bool> + Send + '_;

    /// Create indices for the database. The details of the type of index are left to the trait
    /// implementer. If the index exists, then it is updated instead.
    ///
    /// # Arguments
    ///
    /// * `text_col` - The name of the text column to index.
    /// * `embedding_col` - The name of the embedding column to index.
    fn create_or_update_indices<'a>(
        &'a self,
        text_col: &'a str,
        embedding_col: &'a str,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a;

    /// Connect to the database using the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to use for connecting to the database.
    fn connect(&self) -> impl Future<Output = Result<Self::Connection, Self::Error>> + Send + '_;

    /// Delete rows from the database based on the given column and keys.
    ///
    /// # Arguments
    ///
    /// * `col` - The name of the column to delete rows from.
    /// * `keys` - The keys of the rows to delete. The values must correspond to values in the
    ///   `col` column.
    /// * `config` - The configuration to use for connecting to the database.
    fn delete_rows<'a>(
        &'a self,
        col: &'a str,
        keys: &'a [impl AsRef<str> + Send + Sync],
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a;

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
    fn dedup_rows<'a>(
        &'a self,
        by: &'a str,
        key: &'a str,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send + 'a;

    /// Get all items from the database.
    ///
    /// # Arguments
    ///
    /// * `columns` - The columns to return.
    ///
    /// # Returns
    ///
    /// A vector of all items in the database.
    fn get_items<'a>(
        &'a self,
        columns: &'a [impl AsRef<str> + Send + Sync + Display],
    ) -> impl Future<Output = Result<Vec<Self::Record>, Self::Error>> + Send + 'a;

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
    fn vector_search(
        &self,
        query: String,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<Self::Record>, Self::Error>> + Send + '_;

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
    fn search_by_key<'a>(
        &'a self,
        key_col: &'a str,
        key: &'a str,
    ) -> impl Future<Output = Result<Option<Self::Record>, Self::Error>> + Send + 'a;

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
    fn search_by_column<'a>(
        &'a self,
        col: &'a str,
        values: &'a [impl AsRef<str> + Send + Sync + Display],
    ) -> impl Future<Output = Result<Vec<Self::Record>, Self::Error>> + Send + 'a;

    /// Insert items into the database.
    ///
    /// # Arguments
    ///
    /// * `items` - The items to insert.
    /// * `merge_on` - `None` if you want to create or overwrite the current database; otherwise, a
    ///   reference to an array of keys to merge on.
    /// * `source_col` - The name of the column in `items` that contains the source document text.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the items were inserted successfully, or an error otherwise.
    fn insert_items<'a>(
        &'a self,
        items: Vec<Self::Record>,
        merge_on: Option<&'a [&str]>,
        source_col: &'a str,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a;
}
