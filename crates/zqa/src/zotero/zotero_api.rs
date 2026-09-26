//! Read-only access to a running Zotero instance when its SQLite database is locked.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use reqwest::header::HeaderValue;
use reqwest::{Client, Response, StatusCode, Url};
use serde::{Deserialize, de::DeserializeOwned};
use thiserror::Error;

use super::library::{ZoteroItem, ZoteroItemMetadata};

/// Failures while reading the local API after SQLite reports a lock.
#[derive(Clone, Debug, Error)]
pub enum LocalApiError {
    #[error(
        "Zotero database is locked and its local API is disabled. Enable Settings > Advanced > Allow other applications on this computer to communicate with Zotero, or close Zotero."
    )]
    Disabled,
    #[error(
        "Zotero database is locked and its local API request failed: {0}. Check that Zotero's local API is available on port 23119, or close Zotero."
    )]
    Request(#[source] Arc<reqwest::Error>),
    #[error("Could not verify the Zotero library directory: {0}")]
    Io(#[source] Arc<std::io::Error>),
    #[error("Zotero's local API returned invalid or inconsistent data: {0}")]
    InvalidData(String),
    #[error(
        "Zotero database is locked, but the running instance could not be matched to {0}. Close Zotero to read the selected library directly."
    )]
    UnverifiedLibrary(PathBuf),
}

impl LocalApiError {
    /// Identify group-specific HTTP failures while preserving API-wide failures.
    fn is_unavailable_group(&self, library: &str) -> bool {
        library.starts_with("groups/")
            && matches!(self, Self::Request(error) if error.status().is_some_and(|status| {
                status == StatusCode::NOT_FOUND || status.is_server_error()
            }))
    }
}

impl From<reqwest::Error> for LocalApiError {
    fn from(error: reqwest::Error) -> Self {
        Self::Request(Arc::new(error))
    }
}

impl From<std::io::Error> for LocalApiError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(Arc::new(error))
    }
}

/// A group library returned by `/api/users/0/groups`.
/// Only its ID is needed to read items through `/api/groups/{id}/items`.
#[derive(Deserialize)]
struct Group {
    id: u64,
}

/// An item response from the local API, containing its library-scoped key and item data.
/// Bibliographic items and their attachments are separate responses with distinct keys.
#[derive(Deserialize)]
struct ApiItem {
    key: String,
    data: ItemData,
}

/// The fields read from an item response's `data` object.
/// Bibliographic items supply titles and creators; attachments supply parent keys and file metadata.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ItemData {
    /// Zotero item type, such as `journalArticle`, `preprint`, or `attachment`.
    item_type: String,
    /// Creation timestamp used to order attachments before applying pagination.
    #[serde(default)]
    date_added: String,
    /// Item title, defaulting to empty when omitted. PDF metadata uses the parent item's title.
    #[serde(default)]
    title: String,
    /// Parent item's key for a child item, linking a PDF attachment to its bibliographic metadata.
    parent_item: Option<String>,
    /// Attachment MIME type, used to select `application/pdf` files.
    content_type: Option<String>,
    /// Attachment storage mode, such as `imported_file`, `imported_url`, or `linked_file`.
    /// Imported files belong to Zotero's storage directory; linked files need path resolution.
    link_mode: Option<String>,
    /// Attachment filename. Stored files are located under `storage/<attachment-key>/`.
    filename: Option<String>,
    /// Raw linked-file path, including the `attachments:` prefix for base-relative files.
    path: Option<String>,
    /// Creators in Zotero's stored order, including all returned creator roles.
    /// Defaults to an empty list when the item has no `creators` field.
    #[serde(default)]
    creators: Vec<Creator>,
}

/// A creator's name from a bibliographic item's ordered `creators` array.
/// Single-field names, such as organizations, use `name`; other names use first and last names.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Creator {
    name: Option<String>,
    #[serde(default)]
    first_name: String,
    #[serde(default)]
    last_name: String,
}

impl ItemData {
    /// Identify files managed under the active Zotero data directory's storage folder.
    fn is_stored_attachment(&self) -> bool {
        self.item_type == "attachment"
            && matches!(
                self.link_mode.as_deref(),
                Some("imported_file" | "imported_url")
            )
    }

    /// Format creators in Zotero's order, including single-field organization names.
    fn authors(&self) -> Option<Vec<String>> {
        // NOTE: Maintainers, keep creator ordering and single-field names aligned with `get_authors_sqlite`.
        let authors: Vec<_> = self
            .creators
            .iter()
            .map(|creator| {
                creator.name.clone().unwrap_or_else(|| {
                    if creator.first_name.is_empty() {
                        creator.last_name.clone()
                    } else {
                        format!("{}, {}", creator.last_name, creator.first_name)
                    }
                })
            })
            .collect();

        (!authors.is_empty()).then_some(authors)
    }
}

/// A read-only client for Zotero's local API on port 23119.
///
/// Cache the server ID returned by the API and send it on subsequent requests so an
/// operation fails if a different Zotero instance takes over the port.
pub(crate) struct ZoteroApi {
    client: Client,
    base: String,
    /// Learned from a response's `Zotero-Server-ID` header; absent until one is received.
    server_id: Option<String>,
    /// Collection versions retained across related reads within one fallback operation.
    versions: HashMap<String, Option<HeaderValue>>,
}

impl ZoteroApi {
    /// Build a loopback-only client with no proxy, redirects, or automatic retries.
    pub(crate) fn new() -> Result<Self, LocalApiError> {
        Ok(Self {
            client: Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                .connect_timeout(Duration::from_millis(250))
                .timeout(Duration::from_secs(10))
                .build()?,
            base: "http://127.0.0.1:23119/api".into(),
            server_id: None,
            versions: HashMap::new(),
        })
    }

    /// Send a bounded GET and reject instance changes before accepting response data.
    async fn get(
        &mut self,
        path: &str,
        query: &[(String, String)],
    ) -> Result<Response, LocalApiError> {
        let mut request = self
            .client
            .get(format!("{}/{path}", self.base))
            .query(query);

        if let Some(id) = &self.server_id {
            request = request.header("Zotero-Server-ID", id);
        }

        let response = request.send().await?;

        // Zotero returns 403 when the local API is disabled; provide the setting to enable it.
        if response.status() == StatusCode::FORBIDDEN {
            return Err(LocalApiError::Disabled);
        }

        let response = response.error_for_status()?;

        // `error_for_status` rejects only 4xx/5xx. Reject other non-2xx responses too,
        // particularly 3xx since this client deliberately does not follow redirects.
        if !response.status().is_success() {
            return Err(LocalApiError::InvalidData(format!(
                "unexpected HTTP status {}",
                response.status()
            )));
        }

        if let Some(id) = response.headers().get("Zotero-Server-ID") {
            let id = id
                .to_str()
                .map_err(|error| LocalApiError::InvalidData(error.to_string()))?;

            if self
                .server_id
                .as_deref()
                .is_some_and(|expected| expected != id)
            {
                return Err(LocalApiError::InvalidData(
                    "the running Zotero instance changed during the request".into(),
                ));
            }

            self.server_id = Some(id.into());
        }

        // Single-item responses report the item's version, not the library's version.
        // Compare only collection headers, keeping each library's version independent.
        if path.ends_with("/items") || path == "users/0/groups" {
            let current = response.headers().get("Last-Modified-Version").cloned();
            let expected = self
                .versions
                .entry(path.into())
                .or_insert_with(|| current.clone());

            if *expected != current {
                return Err(LocalApiError::InvalidData(
                    "the library changed during the operation; retry the operation".into(),
                ));
            }
        }

        Ok(response)
    }

    /// Collect pages in a stable order until the endpoint is exhausted or `stop_when` matches.
    /// Reject library version changes across pages and related requests.
    async fn list<T: DeserializeOwned>(
        &mut self,
        path: &str,
        mut query: Vec<(String, String)>,
        stop_when: impl Fn(&[T]) -> bool,
    ) -> Result<Vec<T>, LocalApiError> {
        const PAGE_SIZE: usize = 100;

        query.extend([
            ("format".into(), "json".into()),
            ("limit".into(), PAGE_SIZE.to_string()),
        ]);

        let mut result = Vec::new();

        loop {
            let mut page_query = query.clone();
            page_query.push(("start".into(), result.len().to_string()));
            let response = self.get(path, &page_query).await?;

            let page: Vec<T> = response.json().await?;

            let finished = page.len() < PAGE_SIZE || stop_when(&page);
            result.extend(page);

            if finished {
                return Ok(result);
            }
        }
    }

    /// Enumerate the personal library and all locally available group libraries.
    async fn libraries(&mut self) -> Result<Vec<String>, LocalApiError> {
        let mut groups: Vec<Group> = self.list("users/0/groups", Vec::new(), |_| false).await?;
        groups.sort_unstable_by_key(|group| group.id);

        let mut libraries = vec!["users/0".into()];
        libraries.extend(
            groups
                .into_iter()
                .map(|group| format!("groups/{}", group.id)),
        );

        Ok(libraries)
    }

    /// Fetch library metadata, or exact keys including trashed items for author lookup.
    ///
    /// # Arguments
    ///
    /// * `library` - API prefix identifying a personal or group library.
    /// * `keys` - Item keys to retrieve, or `None` to read the whole library.
    ///
    /// # Returns
    ///
    /// Items from every response page, with absent exact keys resolved individually.
    /// Return an empty list with a warning for group-specific HTTP 404/5xx responses.
    /// Transport, identity, and data consistency failures remain errors for every library.
    async fn items(
        &mut self,
        library: &str,
        keys: Option<&[String]>,
    ) -> Result<Vec<ApiItem>, LocalApiError> {
        // NOTE: Maintainers, keep trash inclusion aligned with the SQLite queries in `library.rs`.
        // Both readers include trashed items so opening Zotero does not change the indexed set.
        let mut query = vec![
            ("includeTrashed".into(), "1".into()),
            ("sort".into(), "dateAdded".into()),
            ("direction".into(), "asc".into()),
        ];

        if let Some(keys) = keys {
            query.push(("itemKey".into(), keys.join(",")));
        }

        // Zotero's additional search scopes can exclude trashed items even with `includeTrashed`.
        // Fetch metadata without `itemType` filters, and resolve absent exact keys individually.
        let mut items: Vec<ApiItem> = match self
            .list(&format!("{library}/items"), query, |_| false)
            .await
        {
            Ok(items) => items,
            Err(error) if error.is_unavailable_group(library) => {
                log::warn!("Skipping unavailable Zotero library {library}: {error}");
                return Ok(Vec::new());
            }
            Err(error) => return Err(error),
        };

        let Some(keys) = keys else {
            return Ok(items);
        };

        let mut read_individually = false;

        for key in keys {
            if items.iter().any(|item| &item.key == key) {
                continue;
            }

            read_individually = true;
            match self.get(&format!("{library}/items/{key}"), &[]).await {
                Ok(response) => items.push(response.json().await?),
                Err(LocalApiError::Request(error))
                    if error.status() == Some(StatusCode::NOT_FOUND) => {}
                Err(error) => return Err(error),
            }
        }

        // Individual item versions cannot establish library consistency. Recheck the
        // collection after exact-key reads, including keys that disappeared with a 404.
        if read_individually {
            self.get(&format!("{library}/items"), &[("limit".into(), "1".into())])
                .await?;
        }

        Ok(items)
    }

    /// Resolve an attachment URL without following a redirect or downloading its content.
    async fn file_path(&mut self, library: &str, key: &str) -> Result<PathBuf, LocalApiError> {
        let text = self
            .get(&format!("{library}/items/{key}/file/view/url"), &[])
            .await?
            .text()
            .await?;

        Url::parse(text.trim())
            .ok()
            .and_then(|url| url.to_file_path().ok())
            .ok_or_else(|| {
                LocalApiError::InvalidData(
                    "attachment endpoint did not return a local file URL".into(),
                )
            })
    }

    /// Resolve local attachments, excluding URL-only links and other nonlocal modes.
    async fn attachment_path(
        &mut self,
        library: &str,
        item: &ApiItem,
        path: &Path,
    ) -> Result<Option<PathBuf>, LocalApiError> {
        if item.data.item_type != "attachment" {
            return Ok(None);
        }

        // NOTE: Maintainers, keep local paths aligned with `parse_library_metadata_sqlite`.
        if item.data.is_stored_attachment() {
            let filename = item.data.filename.as_deref().ok_or_else(|| {
                LocalApiError::InvalidData("stored attachment has no filename".into())
            })?;

            if Path::new(filename)
                .file_name()
                .and_then(|name| name.to_str())
                != Some(filename)
            {
                return Err(LocalApiError::InvalidData(
                    "stored attachment filename is not a single path component".into(),
                ));
            }

            return Ok(Some(path.join("storage").join(&item.key).join(filename)));
        }

        if item.data.link_mode.as_deref() == Some("linked_file") {
            // NOTE: Maintainers, keep base-relative path exclusion aligned with `parse_library_metadata_sqlite`.
            // The Linked Attachment Base Directory lives in profile preferences, not SQLite.
            if item
                .data
                .path
                .as_deref()
                .is_some_and(|path| path.starts_with("attachments:"))
            {
                log::debug!(
                    "Skipping base-relative Zotero attachment {} in {library}",
                    item.key
                );
                return Ok(None);
            }

            return self.file_path(library, &item.key).await.map(Some);
        }

        Ok(None)
    }

    /// Verify the API uses the requested data directory using a stored attachment's location.
    ///
    /// The API serves the running profile, which may differ from the selected data directory.
    /// A server ID detects instance changes but does not identify that directory.
    /// Linked files cannot establish identity: multiple profiles may link to the same file.
    async fn verify_library(
        &mut self,
        library: &str,
        items: &[ApiItem],
        path: &Path,
    ) -> Result<bool, LocalApiError> {
        let Some(item) = items.iter().find(|item| item.data.is_stored_attachment()) else {
            return Ok(false);
        };

        let file = self.file_path(library, &item.key).await?;

        let Some(root) = file.ancestors().nth(3) else {
            return Err(LocalApiError::UnverifiedLibrary(path.into()));
        };

        let is_storage_path = file
            .parent()
            .is_some_and(|parent| parent.ends_with(Path::new("storage").join(&item.key)));
        if !is_storage_path || root.canonicalize()? != path.canonicalize()? {
            return Err(LocalApiError::UnverifiedLibrary(path.into()));
        }

        Ok(true)
    }

    /// Load PDF metadata and verify the selected library before returning any results.
    pub(crate) async fn metadata(
        &mut self,
        path: &Path,
        start: Option<usize>,
        limit: Option<usize>,
    ) -> Result<Vec<ZoteroItemMetadata>, LocalApiError> {
        let mut result = Vec::new();
        let mut verified = false;

        for library in self.libraries().await? {
            let items = self.items(&library, None).await?;

            if !verified {
                verified = self.verify_library(&library, &items, path).await?;
            }

            let parents: HashMap<_, _> = items
                .iter()
                .map(|item| (item.key.as_str(), &item.data))
                .collect();

            let group_id = library
                .strip_prefix("groups/")
                .and_then(|id| id.parse::<u64>().ok())
                .unwrap_or(0);

            for item in &items {
                if item.data.content_type.as_deref() != Some("application/pdf") {
                    continue;
                }

                let Some(parent) = item
                    .data
                    .parent_item
                    .as_deref()
                    .and_then(|key| parents.get(key))
                else {
                    continue;
                };

                // NOTE: Maintainers, keep parent item types aligned with `parse_library_metadata_sqlite`.
                if !matches!(
                    parent.item_type.as_str(),
                    "conferencePaper" | "journalArticle" | "preprint"
                ) {
                    continue;
                }

                let Some(file_path) = self.attachment_path(&library, item, path).await? else {
                    continue;
                };

                let date_added = item.data.date_added.trim_end_matches('Z').replace('T', " ");
                result.push((
                    date_added,
                    group_id,
                    ZoteroItemMetadata {
                        library_key: item.key.clone(),
                        title: parent.title.clone(),
                        file_path,
                        authors: parent.authors(),
                    },
                ));
            }
        }

        if !verified {
            return Err(LocalApiError::UnverifiedLibrary(path.into()));
        }

        // NOTE: Maintainers, keep pagination ordering aligned with `parse_library_metadata_sqlite`:
        // attachment date added, key, then group ID (zero for the personal library).
        result.sort_by(|left, right| {
            (&left.0, &left.2.library_key, left.1).cmp(&(&right.0, &right.2.library_key, right.1))
        });

        Ok(result
            .into_iter()
            .map(|(_, _, metadata)| metadata)
            .skip(start.unwrap_or(0))
            .take(limit.unwrap_or(usize::MAX))
            .collect())
    }

    /// Look up attachment parents in batches and apply authors only after identity validation.
    ///
    /// Match indexed keys and attachment paths, retaining the first matching library.
    /// Exclude resolved items from subsequent libraries' requests.
    pub(crate) async fn authors(
        &mut self,
        items: &mut [ZoteroItem],
        path: &Path,
    ) -> Result<(), LocalApiError> {
        let mut authors = HashMap::new();
        let mut verified = false;
        let libraries = self.libraries().await?;

        for library in &libraries {
            let mut keys: Vec<_> = items
                .iter()
                .enumerate()
                .filter(|(index, _)| !authors.contains_key(index))
                .map(|(_, item)| item.metadata.library_key.clone())
                .collect();
            keys.sort_unstable();
            keys.dedup();

            if keys.is_empty() {
                break;
            }

            for keys in keys.chunks(50) {
                let attachments: Vec<_> = self
                    .items(library, Some(keys))
                    .await?
                    .into_iter()
                    .filter(|attachment| keys.contains(&attachment.key))
                    .collect();

                if !verified {
                    verified = self.verify_library(library, &attachments, path).await?;
                }

                let parent_keys: Vec<_> = attachments
                    .iter()
                    .filter_map(|item| item.data.parent_item.clone())
                    .collect();

                if parent_keys.is_empty() {
                    continue;
                }

                let parents = self.items(library, Some(&parent_keys)).await?;

                let parents: HashMap<_, _> = parents
                    .iter()
                    .map(|item| (item.key.as_str(), &item.data))
                    .collect();

                for attachment in &attachments {
                    let Some(parent) = attachment
                        .data
                        .parent_item
                        .as_deref()
                        .and_then(|key| parents.get(key))
                    else {
                        continue;
                    };

                    let Some(file_path) = self.attachment_path(library, attachment, path).await?
                    else {
                        continue;
                    };

                    // Keys are library-scoped. Match the indexed attachment's path before
                    // accepting a result; parent titles may have changed since indexing.
                    for (index, item) in items.iter().enumerate() {
                        if item.metadata.library_key == attachment.key
                            && item.metadata.file_path == file_path
                        {
                            authors.entry(index).or_insert_with(|| parent.authors());
                        }
                    }
                }
            }
        }

        // Look for active attachments first. If none prove identity, include trash without
        // search scopes, which can hide the only stored attachment in the library.
        if !verified {
            'libraries: for library in libraries {
                for item_type in [Some("attachment"), None] {
                    let mut query = vec![
                        ("includeTrashed".into(), "1".into()),
                        ("sort".into(), "dateAdded".into()),
                        ("direction".into(), "asc".into()),
                    ];

                    if let Some(item_type) = item_type {
                        query.push(("itemType".into(), item_type.into()));
                    }

                    let candidates = match self
                        .list(&format!("{library}/items"), query, |page: &[ApiItem]| {
                            page.iter().any(|item| item.data.is_stored_attachment())
                        })
                        .await
                    {
                        Ok(items) => items,
                        Err(error) if error.is_unavailable_group(&library) => {
                            log::warn!("Skipping unavailable Zotero library {library}: {error}");
                            continue 'libraries;
                        }
                        Err(error) => return Err(error),
                    };

                    if self.verify_library(&library, &candidates, path).await? {
                        verified = true;
                        break 'libraries;
                    }
                }
            }
        }

        if !verified {
            return Err(LocalApiError::UnverifiedLibrary(path.into()));
        }

        for (index, value) in authors {
            items[index].metadata.authors = value;
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "zotero_api_tests.rs"]
mod tests;
