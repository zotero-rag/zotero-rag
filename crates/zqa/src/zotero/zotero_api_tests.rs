//! HTTP contract tests for the local Zotero reader.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::{self, JoinHandle};

use serde_json::{Value, json};

use super::*;

/// An expected HTTP request and the response the test server should return for it.
struct Reply {
    path: &'static str,
    query: Vec<(&'static str, String)>,
    absent_query: Vec<&'static str>,
    status: u16,
    body: String,
    version: &'static str,
    server_id: &'static str,
}

impl Reply {
    /// Describe one expected API GET and its response body.
    fn json(path: &'static str, body: &Value) -> Self {
        Self {
            path,
            query: Vec::new(),
            absent_query: Vec::new(),
            status: 200,
            body: body.to_string(),
            version: "1",
            server_id: "test-instance",
        }
    }
}

/// A scripted loopback server whose worker thread is joined when the test releases it.
struct Server {
    address: std::net::SocketAddr,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        // Wake accept if a failed client assertion skipped the next request.
        let _ = TcpStream::connect(self.address);
        if let Some(thread) = self.thread.take() {
            let result = thread.join();
            if !std::thread::panicking() {
                result.unwrap();
            }
        }
    }
}

/// Serve expected endpoints and filters, allowing independent requests in either order.
fn server(replies: Vec<Reply>) -> (ZoteroApi, Server) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let thread = thread::spawn(move || {
        let mut expected_server_id = None;

        let mut listener = Some(listener);
        let mut replies = replies;

        while !replies.is_empty() {
            let (mut stream, _) = listener.as_ref().unwrap().accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            assert!(
                !request.is_empty(),
                "expected {} more API requests",
                replies.len()
            );
            let target = request.split_whitespace().nth(1).unwrap();
            let url = Url::parse(&format!("http://localhost{target}")).unwrap();
            let index = replies
                .iter()
                .position(|reply| {
                    url.path() == reply.path
                        && reply.query.iter().all(|(key, value)| {
                            url.query_pairs()
                                .any(|(k, v)| k == *key && v == value.as_str())
                        })
                })
                .unwrap_or_else(|| panic!("unexpected request: {target}"));
            let reply = replies.remove(index);

            for key in reply.absent_query {
                assert!(
                    !url.query_pairs().any(|(k, _)| k == key),
                    "unexpected query parameter {key}"
                );
            }

            let mut request_server_id = None;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }

                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("Zotero-Server-ID")
                {
                    request_server_id = Some(value.trim().to_owned());
                }
            }

            assert_eq!(request_server_id.as_deref(), expected_server_id);
            expected_server_id = Some(reply.server_id);

            // Close the listener before the final reply so a subsequent request is refused.
            if replies.is_empty() {
                listener.take();
            }

            write!(stream, "HTTP/1.1 {} Test\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\nLast-Modified-Version: {}\r\nZotero-Server-ID: {}\r\n\r\n{}", reply.status, reply.body.len(), reply.version, reply.server_id, reply.body).unwrap();
        }
    });
    let mut api = ZoteroApi::new().unwrap();
    api.base = format!("http://{address}/api");
    (
        api,
        Server {
            address,
            thread: Some(thread),
        },
    )
}

/// Build a parent with both personal and institutional creators.
fn parent() -> Value {
    json!({"key": "PARENT01", "data": {"itemType": "journalArticle", "title": "A paper", "creators": [
        {"firstName": "Ada", "lastName": "Lovelace"}, {"name": "Research Institute"}
    ]}})
}

/// Build a stored PDF attachment that identifies its parent item.
fn attachment() -> Value {
    json!({"key": "ATTACH01", "data": {"itemType": "attachment", "parentItem": "PARENT01", "contentType": "application/pdf", "linkMode": "imported_file", "filename": "A paper.pdf"}})
}

/// Build an indexed search result whose author metadata still needs to be populated.
fn indexed_item(key: &str, file_path: PathBuf) -> ZoteroItem {
    ZoteroItem {
        metadata: ZoteroItemMetadata {
            library_key: key.into(),
            title: "A paper".into(),
            file_path,
            authors: None,
        },
        text: String::new(),
    }
}

/// Build a linked PDF with the same bibliographic parent as the stored fixture.
fn linked_attachment(key: &str) -> Value {
    json!({"key": key, "data": {"itemType": "attachment", "parentItem": "PARENT01",
        "contentType": "application/pdf", "linkMode": "linked_file"}})
}

/// Require the discovery scope and page offset, excluding exact-key lookup filters.
fn identity_page(items: &Value, start: usize, item_type: Option<&str>) -> Reply {
    let mut reply = Reply::json("/api/users/0/items", items);
    reply.query = vec![("includeTrashed", "1".into()), ("start", start.to_string())];
    reply.absent_query = vec!["itemKey"];
    if let Some(item_type) = item_type {
        reply.query.push(("itemType", item_type.into()));
    } else {
        reply.absent_query.push("itemType");
    }
    reply
}

/// Return a file URL response anchored in the specified library, even if the PDF is not downloaded.
fn file_reply(path: &Path, endpoint: &'static str) -> Reply {
    let mut reply = Reply::json(endpoint, &Value::Null);
    reply.body = Url::from_file_path(path.join("storage/ATTACH01/A paper.pdf"))
        .unwrap()
        .to_string();
    reply
}

/// Personal and group PDFs retain parent metadata and resolve stored and linked paths.
#[tokio::test]
async fn metadata_maps_personal_and_group_libraries_and_linked_paths() {
    let library = tempfile::tempdir().unwrap();
    let linked = library.path().join("linked file.pdf");
    let mut linked_reply = Reply::json("/api/groups/42/items/LINKED01/file/view/url", &Value::Null);
    linked_reply.body = Url::from_file_path(&linked).unwrap().to_string();
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([{"id": 42}])),
        Reply::json(
            "/api/users/0/items",
            &json!([parent(), attachment(), {"key": "NOTE0001", "data": {"itemType": "note"}}]),
        ),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        Reply::json(
            "/api/groups/42/items",
            &json!([parent(), linked_attachment("LINKED01")]),
        ),
        linked_reply,
    ]);
    let items = api.metadata(library.path(), None, None).await.unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].library_key, "ATTACH01");
    assert_eq!(items[0].title, "A paper");
    assert_eq!(
        items[0].file_path,
        library.path().join("storage/ATTACH01/A paper.pdf")
    );
    assert_eq!(
        items[0].authors.as_ref().unwrap(),
        &["Lovelace, Ada", "Research Institute"]
    );
    assert_eq!(items[1].file_path, linked);
}

/// A running profile cannot replace the explicitly selected library.
#[tokio::test]
async fn rejects_a_different_running_profile() {
    let requested = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([parent(), attachment()])),
        file_reply(other.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);
    assert!(matches!(
        api.metadata(requested.path(), None, None).await,
        Err(LocalApiError::UnverifiedLibrary(_))
    ));
}

/// An empty API result provides no evidence of the selected database identity.
#[tokio::test]
async fn does_not_report_an_unverified_empty_library_as_success() {
    let requested = tempfile::tempdir().unwrap();
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([])),
    ]);
    assert!(matches!(
        api.metadata(requested.path(), None, None).await,
        Err(LocalApiError::UnverifiedLibrary(_))
    ));
}

/// Disabled API access produces an actionable error.
#[tokio::test]
async fn disabled_api_reports_the_setting_to_enable() {
    let mut forbidden = Reply::json("/api/users/0/groups", &Value::Null);
    forbidden.status = 403;
    let (mut api, _server) = server(vec![forbidden]);
    let error = api.libraries().await.unwrap_err();
    assert!(matches!(error, LocalApiError::Disabled));
    assert!(error.to_string().contains("Allow other applications"));
}

/// Reject remote URLs instead of accepting them as local attachment paths.
#[tokio::test]
async fn file_path_rejects_non_file_urls() {
    let mut remote_file = Reply::json("/api/users/0/items/ATTACH01/file/view/url", &Value::Null);
    remote_file.body = "https://example.com/file.pdf".into();
    let (mut api, _server) = server(vec![remote_file]);
    assert!(matches!(
        api.file_path("users/0", "ATTACH01").await,
        Err(LocalApiError::InvalidData(_))
    ));
}

/// Pagination collects all results and rejects version changes.
#[tokio::test]
async fn reads_all_pages_and_rejects_changes_between_pages() {
    let first: Vec<_> = (0..100).map(|id| json!({"id": id})).collect();
    let mut second = Reply::json("/api/users/0/groups", &json!([{"id": 100}]));
    second.query.push(("start", "100".into()));
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!(first)),
        second,
    ]);
    assert_eq!(api.libraries().await.unwrap().len(), 102);
    let mut changed = Reply::json("/api/users/0/groups", &json!([{"id": 100}]));
    changed.version = "2";
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!(first)),
        changed,
    ]);
    assert!(matches!(
        api.libraries().await,
        Err(LocalApiError::InvalidData(_))
    ));
}

/// Recover missing exact keys without comparing item versions to library versions.
#[tokio::test]
async fn exact_key_lookup_rechecks_the_library_version() {
    for version in ["1", "2"] {
        let mut item = Reply::json("/api/users/0/items/ATTACH01", &attachment());
        item.version = "0";
        let mut recheck = Reply::json("/api/users/0/items", &json!([]));
        recheck.query.push(("limit", "1".into()));
        recheck.version = version;
        let (mut api, _server) = server(vec![
            Reply::json("/api/users/0/items", &json!([])),
            item,
            recheck,
        ]);

        let result = api.items("users/0", Some(&["ATTACH01".into()])).await;
        if version == "1" {
            let items = result.unwrap();
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].key, "ATTACH01");
        } else {
            assert!(
                matches!(result, Err(LocalApiError::InvalidData(message)) if message.contains("library changed"))
            );
        }
    }
}

/// Learn the server ID from a response, reuse it, and reject a different instance.
#[tokio::test]
async fn learns_server_id_and_rejects_instance_changes() {
    let mut changed = Reply::json("/api/users/0/groups", &json!([]));
    changed.server_id = "different-instance";
    changed.version = "2";
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        changed,
    ]);

    api.libraries().await.unwrap();

    assert!(matches!(
        api.libraries().await,
        Err(LocalApiError::InvalidData(message)) if message.contains("instance changed")
    ));
}

/// Reject redirects separately from client and server errors.
#[tokio::test]
async fn rejects_redirects_and_preserves_http_error_statuses() {
    for status in [302, 500] {
        let mut reply = Reply::json("/api/users/0/groups", &json!([]));
        reply.status = status;
        let (mut api, _server) = server(vec![reply]);

        let error = api.libraries().await.unwrap_err();
        if status < 400 {
            assert!(matches!(error, LocalApiError::InvalidData(_)));
        } else {
            assert!(matches!(
                error,
                LocalApiError::Request(source) if source.status().unwrap().as_u16() == status
            ));
        }
    }
}

/// Exclude nonlocal and base-relative PDFs before pagination without requesting their file URLs.
#[tokio::test]
async fn metadata_skips_nonlocal_attachment_modes() {
    let library = tempfile::tempdir().unwrap();
    let mut links: Vec<_> = ["linked_url", "embedded_image", "unknown"]
        .into_iter()
        .map(|mode| {
            let mut item = attachment();
            item["key"] = json!(mode);
            item["data"]["linkMode"] = json!(mode);
            item
        })
        .collect();
    let mut relative = linked_attachment("RELATIVE");
    relative["data"]["path"] = json!("attachments:Papers/linked.pdf");
    let mut stored = attachment();
    stored["data"]["dateAdded"] = json!("2020-01-01T00:00:00Z");
    links.extend([relative, parent(), stored]);
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!(links)),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);

    let items = api
        .metadata(library.path(), Some(0), Some(1))
        .await
        .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].library_key, "ATTACH01");
}

/// Retain verified PDFs when an unrelated group cannot be read.
#[tokio::test]
async fn metadata_retains_personal_results_when_a_group_is_unavailable() {
    for status in [404, 500] {
        let library = tempfile::tempdir().unwrap();
        let mut unavailable = Reply::json("/api/groups/42/items", &json!([]));
        unavailable.status = status;
        let mut available = attachment();
        available["key"] = json!("SECOND01");
        let (mut api, _server) = server(vec![
            Reply::json("/api/users/0/groups", &json!([{"id": 42}, {"id": 43}])),
            Reply::json("/api/users/0/items", &json!([parent(), attachment()])),
            file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
            unavailable,
            Reply::json("/api/groups/43/items", &json!([parent(), available])),
        ]);

        let items = api.metadata(library.path(), None, None).await.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].library_key, "ATTACH01");
        assert_eq!(items[1].library_key, "SECOND01");
    }
}

/// Reject instance changes even when they occur in an unrelated group.
#[tokio::test]
async fn group_failures_do_not_hide_identity_errors() {
    let library = tempfile::tempdir().unwrap();
    let mut changed = Reply::json("/api/groups/42/items", &json!([]));
    changed.status = 412;
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([{"id": 42}])),
        Reply::json("/api/users/0/items", &json!([parent(), attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        changed,
    ]);

    assert!(matches!(api.metadata(library.path(), None, None).await,
        Err(LocalApiError::Request(error)) if error.status() == Some(StatusCode::PRECONDITION_FAILED)));
}

/// Stop after resolving all requested items, without querying unrelated group libraries.
#[tokio::test]
async fn authors_do_not_query_resolved_keys_in_other_libraries() {
    let library = tempfile::tempdir().unwrap();
    let mut attachments = Reply::json("/api/users/0/items", &json!([attachment()]));
    attachments.query.push(("itemKey", "ATTACH01".into()));
    let mut parents = Reply::json("/api/users/0/items", &json!([parent()]));
    parents.query.push(("itemKey", "PARENT01".into()));
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([{"id": 42}, {"id": 43}])),
        attachments,
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        parents,
    ]);
    let mut items = vec![indexed_item(
        "ATTACH01",
        library.path().join("storage/ATTACH01/A paper.pdf"),
    )];

    api.authors(&mut items, library.path()).await.unwrap();
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap(),
        &["Lovelace, Ada", "Research Institute"]
    );
}

/// Match same-key attachments to their indexed files without overwriting an earlier result.
#[tokio::test]
async fn authors_distinguish_same_keys_in_personal_and_group_libraries() {
    let library = tempfile::tempdir().unwrap();
    let mut group_attachment = attachment();
    group_attachment["data"]["filename"] = json!("group.pdf");
    let mut group_parent = parent();
    group_parent["data"]["creators"] = json!([{"name": "Group Institute"}]);
    let mut group_items = Reply::json("/api/groups/42/items", &json!([group_attachment]));
    group_items.version = "9";
    let mut group_parents = Reply::json("/api/groups/42/items", &json!([group_parent]));
    group_parents.version = "9";
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([{"id": 42}])),
        Reply::json("/api/users/0/items", &json!([attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        Reply::json("/api/users/0/items", &json!([parent()])),
        group_items,
        group_parents,
    ]);
    let mut items: Vec<_> = ["A paper.pdf", "group.pdf"]
        .into_iter()
        .map(|filename| {
            indexed_item(
                "ATTACH01",
                library.path().join("storage/ATTACH01").join(filename),
            )
        })
        .collect();

    api.authors(&mut items, library.path()).await.unwrap();
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap()[0],
        "Lovelace, Ada"
    );
    assert_eq!(
        items[1].metadata.authors.as_ref().unwrap(),
        &["Group Institute"]
    );
}

/// Apply global date/key ordering before offset and limit, including across library boundaries.
#[tokio::test]
async fn metadata_orders_dates_and_key_ties_across_libraries() {
    let library = tempfile::tempdir().unwrap();
    let mut first = attachment();
    first["data"]["dateAdded"] = json!("2020-01-01T00:00:00Z");
    let mut group_first = first.clone();
    group_first["key"] = json!("AAA00001");
    let mut older = first.clone();
    older["key"] = json!("ZZZ00001");
    older["data"]["dateAdded"] = json!("2019-01-01T00:00:00Z");
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([{"id": 42}])),
        Reply::json("/api/users/0/items", &json!([parent(), first])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        Reply::json(
            "/api/groups/42/items",
            &json!([parent(), group_first, older]),
        ),
    ]);

    let items = api
        .metadata(library.path(), Some(1), Some(1))
        .await
        .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].library_key, "AAA00001");
}

/// Keep linked-file paths usable when SQLite indexing is followed by API author lookup.
#[tokio::test]
async fn authors_resolve_sqlite_linked_files_after_parent_title_changes() {
    let library = tempfile::tempdir().unwrap();
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/Zotero/zotero.sqlite");
    let database_path = library.path().join("zotero.sqlite");
    std::fs::copy(source, &database_path).unwrap();
    let linked_path = library.path().join("linked paper.pdf");
    let database = rusqlite::Connection::open(database_path).unwrap();
    database
        .execute(
            "UPDATE itemAttachments SET path = ?1, linkMode = 2
         WHERE itemID = (SELECT itemID FROM items WHERE key = '7R5XZ5PX')",
            [linked_path.to_str().unwrap()],
        )
        .unwrap();
    drop(database);

    let metadata = super::super::library::parse_library_metadata(Some(library.path()), None, None)
        .await
        .unwrap()
        .into_iter()
        .find(|item| item.library_key == "7R5XZ5PX")
        .unwrap();
    assert_eq!(metadata.file_path, linked_path);
    let mut items = vec![ZoteroItem {
        metadata,
        text: String::new(),
    }];
    let mut edited_parent = parent();
    edited_parent["data"]["title"] = json!("Edited after indexing");
    let linked = linked_attachment("7R5XZ5PX");
    let mut linked_reply = Reply::json("/api/users/0/items/7R5XZ5PX/file/view/url", &Value::Null);
    linked_reply.body = Url::from_file_path(&linked_path).unwrap().to_string();
    let mut candidates = vec![linked.clone(); 100];
    candidates[99] = attachment();
    candidates[99]["data"]["deleted"] = json!(1);
    let active_page = identity_page(&json!([linked.clone()]), 0, Some("attachment"));
    let identity_page = identity_page(&json!(candidates), 0, None);

    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([linked])),
        Reply::json("/api/users/0/items", &json!([edited_parent])),
        linked_reply,
        active_page,
        identity_page,
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);

    api.authors(&mut items, library.path()).await.unwrap();
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap()[0],
        "Lovelace, Ada"
    );
}

/// Fail if the shared API exits before the last group request, even after personal verification.
#[tokio::test]
async fn metadata_propagates_connection_failure_after_personal_verification() {
    let library = tempfile::tempdir().unwrap();
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([{"id": 42}])),
        Reply::json("/api/users/0/items", &json!([parent(), attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);

    assert!(matches!(api.metadata(library.path(), None, None).await,
        Err(LocalApiError::Request(error)) if error.is_connect()));
}

/// Page through active attachments and stop at the first stored file, without scanning all items.
#[tokio::test]
async fn identity_discovery_stops_at_the_first_stored_attachment_page() {
    let library = tempfile::tempdir().unwrap();
    let linked = linked_attachment("LINKED01");
    let linked_path = library.path().join("linked.pdf");
    let mut linked_reply = Reply::json("/api/users/0/items/LINKED01/file/view/url", &Value::Null);
    linked_reply.body = Url::from_file_path(&linked_path).unwrap().to_string();
    let first_page = identity_page(&json!(vec![linked.clone(); 100]), 0, Some("attachment"));
    let mut candidates = vec![linked.clone(); 100];
    candidates[99] = attachment();
    let second_page = identity_page(&json!(candidates), 100, Some("attachment"));
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([linked])),
        Reply::json("/api/users/0/items", &json!([parent()])),
        linked_reply,
        first_page,
        second_page,
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);
    let mut items = vec![indexed_item("LINKED01", linked_path)];

    api.authors(&mut items, library.path()).await.unwrap();
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap()[0],
        "Lovelace, Ada"
    );
}

/// Reject a reparent or metadata edit between attachment and parent reads without changing results.
#[tokio::test]
async fn authors_reject_changes_between_related_reads() {
    let library = tempfile::tempdir().unwrap();
    let mut parents = Reply::json("/api/users/0/items", &json!([parent()]));
    parents.version = "2";
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        parents,
    ]);
    let mut items = vec![indexed_item(
        "ATTACH01",
        library.path().join("storage/ATTACH01/A paper.pdf"),
    )];
    items[0].metadata.authors = Some(vec!["Previously indexed author".into()]);

    assert!(matches!(api.authors(&mut items, library.path()).await,
        Err(LocalApiError::InvalidData(message)) if message.contains("library changed")));
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap(),
        &["Previously indexed author"]
    );
}

/// Exercise the shared fallback dispatch with a real exclusive SQLite lock and HTTP responses.
#[tokio::test]
async fn locked_sqlite_reads_fall_back_to_api_without_waiting() {
    use super::super::library::{
        get_authors_sqlite, parse_library_metadata_sqlite, with_api_fallback,
    };

    let library = tempfile::tempdir().unwrap();
    let writer = rusqlite::Connection::open(library.path().join("zotero.sqlite")).unwrap();
    writer
        .execute_batch("CREATE TABLE marker (id INTEGER); BEGIN EXCLUSIVE;")
        .unwrap();
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([parent(), attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        Reply::json("/api/users/0/items", &json!([parent()])),
    ]);
    let start = std::time::Instant::now();
    let metadata = with_api_fallback(
        parse_library_metadata_sqlite(library.path(), None, None),
        api.metadata(library.path(), None, None),
    )
    .await
    .unwrap();
    assert_eq!(metadata.len(), 1);
    assert_eq!(metadata[0].library_key, "ATTACH01");
    let mut items = vec![indexed_item(
        &metadata[0].library_key,
        metadata[0].file_path.clone(),
    )];

    with_api_fallback(
        get_authors_sqlite(&mut items, library.path()),
        api.authors(&mut items, library.path()),
    )
    .await
    .unwrap();
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap(),
        &["Lovelace, Ada", "Research Institute"]
    );
    assert!(start.elapsed() < Duration::from_secs(1));
}
