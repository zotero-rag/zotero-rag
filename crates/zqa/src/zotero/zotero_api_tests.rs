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
    status: u16,
    body: String,
    version: &'static str,
    server_id: &'static str,
}

impl Reply {
    /// Describes one expected API GET and its response body.
    fn json(path: &'static str, body: &Value) -> Self {
        Self {
            path,
            query: Vec::new(),
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

/// Runs a scripted loopback server, asserting the client uses the expected endpoints and filters.
fn server(replies: Vec<Reply>) -> (ZoteroApi, Server) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let thread = thread::spawn(move || {
        let mut expected_server_id = None;

        let mut listener = Some(listener);
        let reply_count = replies.len();

        for (index, reply) in replies.into_iter().enumerate() {
            let (mut stream, _) = listener.as_ref().unwrap().accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            assert!(
                !request.is_empty(),
                "expected another API request to {}",
                reply.path
            );
            let target = request.split_whitespace().nth(1).unwrap();
            let url = Url::parse(&format!("http://localhost{target}")).unwrap();
            assert_eq!(url.path(), reply.path);
            for (key, value) in reply.query {
                assert!(
                    url.query_pairs().any(|(k, v)| k == key && v == value),
                    "missing query parameter {key}={value}"
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
            if index + 1 == reply_count {
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

/// Builds a parent with both personal and institutional creators.
fn parent() -> Value {
    json!({"key": "PARENT01", "data": {"itemType": "journalArticle", "title": "A paper", "creators": [
        {"firstName": "Ada", "lastName": "Lovelace"}, {"name": "Research Institute"}
    ]}})
}

/// Builds a stored PDF attachment that identifies its parent item.
fn attachment() -> Value {
    json!({"key": "ATTACH01", "data": {"itemType": "attachment", "parentItem": "PARENT01", "contentType": "application/pdf", "linkMode": "imported_file", "filename": "A paper.pdf"}})
}

/// Returns a file URL response anchored in the specified library, even if the PDF is not downloaded.
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
            &json!([parent(), {"key": "LINKED01", "data": {"itemType": "attachment", "parentItem": "PARENT01", "contentType": "application/pdf", "linkMode": "linked_file"}}]),
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

/// Author lookup queries attachment and parent keys rather than the full library.
#[tokio::test]
async fn authors_fetch_only_matching_attachments_and_parents() {
    let library = tempfile::tempdir().unwrap();
    let mut attachments = Reply::json("/api/users/0/items", &json!([attachment()]));
    attachments.query.push(("itemKey", "ATTACH01".into()));
    let mut parents = Reply::json("/api/users/0/items", &json!([parent()]));
    parents.query.push(("itemKey", "PARENT01".into()));
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        attachments,
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        parents,
    ]);
    let mut items = vec![ZoteroItem {
        metadata: ZoteroItemMetadata {
            library_key: "ATTACH01".into(),
            title: "A paper".into(),
            file_path: library.path().join("storage/ATTACH01/A paper.pdf"),
            authors: None,
        },
        text: String::new(),
    }];
    api.authors(&mut items, library.path()).await.unwrap();
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap(),
        &["Lovelace, Ada", "Research Institute"]
    );
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

/// Malformed responses and nonlocal attachment URLs fail explicitly.
#[tokio::test]
async fn rejects_malformed_responses_and_non_file_urls() {
    let mut malformed = Reply::json("/api/users/0/groups", &Value::Null);
    malformed.body = "not json".into();
    let (mut api, _server) = server(vec![malformed]);
    assert!(matches!(
        api.libraries().await,
        Err(LocalApiError::Request(_))
    ));
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

/// A stopped API does not introduce the five-second SQLite retry delay.
#[tokio::test]
async fn refused_connection_fails_without_the_sqlite_busy_wait() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut api = ZoteroApi::new().unwrap();
    api.base = format!("http://{address}/api");
    let start = std::time::Instant::now();
    assert!(matches!(
        api.libraries().await,
        Err(LocalApiError::Request(_))
    ));
    assert!(start.elapsed() < Duration::from_secs(1));
}

/// Exact lookup recovers trashed items omitted by Zotero's filtered collection endpoint.
#[tokio::test]
async fn resolves_trashed_items_missing_from_batch_results() {
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/items", &json!([])),
        Reply::json("/api/users/0/items/ATTACH01", &attachment()),
    ]);
    let items = api
        .items("users/0", Some(&["ATTACH01".into()]), "dateAdded", "asc")
        .await
        .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].key, "ATTACH01");
}

/// A limit and offset apply to PDFs across libraries, not to the parent metadata response.
#[tokio::test]
async fn pagination_applies_after_mapping_attachments() {
    let library = tempfile::tempdir().unwrap();
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([parent(), attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);
    assert!(
        api.metadata(library.path(), Some(1), Some(1))
            .await
            .unwrap()
            .is_empty()
    );
}

/// Learn the server ID from a response, reuse it, and reject a different instance.
#[tokio::test]
async fn learns_server_id_and_rejects_instance_changes() {
    let mut changed = Reply::json("/api/users/0/groups", &json!([]));
    changed.server_id = "different-instance";
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/groups", &json!([])),
        changed,
    ]);

    assert!(api.server_id.is_none());
    api.libraries().await.unwrap();
    assert_eq!(api.server_id.as_deref(), Some("test-instance"));
    api.libraries().await.unwrap();

    assert!(matches!(
        api.libraries().await,
        Err(LocalApiError::InvalidData(message)) if message.contains("instance changed")
    ));
}

/// Reject redirects separately from client and server errors.
#[tokio::test]
async fn rejects_redirects_and_preserves_http_error_statuses() {
    for status in [301, 302, 304, 307, 308, 404, 412, 500] {
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

/// Apply the caller's sort field and direction to each item page.
#[tokio::test]
async fn items_use_requested_sort_order_on_every_page() {
    let first: Vec<_> = (0..100)
        .map(|id| json!({"key": id.to_string(), "data": {"itemType": "note"}}))
        .collect();
    let mut first_reply = Reply::json("/api/users/0/items", &json!(first));
    first_reply.query = vec![
        ("sort", "dateModified".into()),
        ("direction", "desc".into()),
    ];
    let mut second_reply = Reply::json("/api/users/0/items", &json!([]));
    second_reply.query = vec![
        ("sort", "dateModified".into()),
        ("direction", "desc".into()),
        ("start", "100".into()),
    ];
    let (mut api, _server) = server(vec![first_reply, second_reply]);

    let items = api
        .items("users/0", None, "dateModified", "desc")
        .await
        .unwrap();
    assert_eq!(items.len(), 100);
}

/// Ignore URL-only PDF links without requesting a file URL or losing stored PDFs.
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
    links.extend([parent(), attachment()]);
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!(links)),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);

    let items = api.metadata(library.path(), None, None).await.unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].library_key, "ATTACH01");
}

/// Retain verified PDFs when an unrelated group cannot be read.
#[tokio::test]
async fn metadata_retains_personal_results_when_a_group_is_unavailable() {
    for status in [404, 500, 503] {
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
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([{"id": 42}, {"id": 43}])),
        Reply::json("/api/users/0/items", &json!([attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        Reply::json("/api/users/0/items", &json!([parent()])),
    ]);
    let mut items = vec![ZoteroItem {
        metadata: ZoteroItemMetadata {
            library_key: "ATTACH01".into(),
            title: "A paper".into(),
            file_path: library.path().join("storage/ATTACH01/A paper.pdf"),
            authors: None,
        },
        text: String::new(),
    }];

    api.authors(&mut items, library.path()).await.unwrap();
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap()[0],
        "Lovelace, Ada"
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
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([{"id": 42}])),
        Reply::json("/api/users/0/items", &json!([attachment()])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        Reply::json("/api/users/0/items", &json!([parent()])),
        Reply::json("/api/groups/42/items", &json!([group_attachment])),
        Reply::json("/api/groups/42/items", &json!([group_parent])),
    ]);
    let mut items: Vec<_> = ["A paper.pdf", "group.pdf"]
        .into_iter()
        .map(|filename| ZoteroItem {
            metadata: ZoteroItemMetadata {
                library_key: "ATTACH01".into(),
                title: "A paper".into(),
                file_path: library.path().join("storage/ATTACH01").join(filename),
                authors: None,
            },
            text: String::new(),
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
    let linked = json!({"key": "7R5XZ5PX", "data": {
        "itemType": "attachment", "parentItem": "PARENT01", "linkMode": "linked_file"
    }});
    let mut linked_reply = Reply::json("/api/users/0/items/7R5XZ5PX/file/view/url", &Value::Null);
    linked_reply.body = Url::from_file_path(&linked_path).unwrap().to_string();
    let mut candidates = vec![linked.clone(); 100];
    candidates[99] = attachment();
    let mut identity_page = Reply::json("/api/users/0/items", &json!(candidates));
    identity_page.query = vec![("itemType", "attachment".into()), ("start", "0".into())];

    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([linked])),
        Reply::json("/api/users/0/items", &json!([edited_parent])),
        linked_reply,
        identity_page,
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);

    api.authors(&mut items, library.path()).await.unwrap();
    assert_eq!(
        items[0].metadata.authors.as_ref().unwrap()[0],
        "Lovelace, Ada"
    );
}

/// Ignore extra attachment keys before resolving file paths or requesting their parents.
#[tokio::test]
async fn authors_do_not_resolve_unrequested_linked_files() {
    let library = tempfile::tempdir().unwrap();
    let unrelated = json!({"key": "OTHER001", "data": {
        "itemType": "attachment", "parentItem": "OTHER002", "linkMode": "linked_file"
    }});
    let mut parents = Reply::json("/api/users/0/items", &json!([parent()]));
    parents.query.push(("itemKey", "PARENT01".into()));
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([attachment(), unrelated])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
        parents,
    ]);
    let mut items = vec![ZoteroItem {
        metadata: ZoteroItemMetadata {
            library_key: "ATTACH01".into(),
            title: "An outdated title".into(),
            file_path: library.path().join("storage/ATTACH01/A paper.pdf"),
            authors: None,
        },
        text: String::new(),
    }];

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

/// Skip base-relative linked files before pagination without resolving their file URLs.
#[tokio::test]
async fn metadata_skips_base_relative_linked_files() {
    let library = tempfile::tempdir().unwrap();
    let mut relative = attachment();
    relative["key"] = json!("RELATIVE");
    relative["data"]["linkMode"] = json!("linked_file");
    relative["data"]["path"] = json!("attachments:Papers/linked.pdf");
    relative["data"]["dateAdded"] = json!("2000-01-01T00:00:00Z");
    let mut stored = attachment();
    stored["data"]["dateAdded"] = json!("2020-01-01T00:00:00Z");
    let (mut api, _server) = server(vec![
        Reply::json("/api/users/0/groups", &json!([])),
        Reply::json("/api/users/0/items", &json!([parent(), relative, stored])),
        file_reply(library.path(), "/api/users/0/items/ATTACH01/file/view/url"),
    ]);

    let items = api
        .metadata(library.path(), Some(0), Some(1))
        .await
        .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].library_key, "ATTACH01");
}

/// Continue identity discovery past linked-only pages while rejecting concurrent library changes.
#[tokio::test]
async fn identity_discovery_paginates_attachments_and_checks_versions() {
    for version in ["1", "2"] {
        let library = tempfile::tempdir().unwrap();
        let linked = json!({"key": "LINKED01", "data": {
            "itemType": "attachment", "parentItem": "PARENT01", "linkMode": "linked_file"
        }});
        let linked_path = library.path().join("linked.pdf");
        let mut linked_reply =
            Reply::json("/api/users/0/items/LINKED01/file/view/url", &Value::Null);
        linked_reply.body = Url::from_file_path(&linked_path).unwrap().to_string();
        let mut first_page = Reply::json("/api/users/0/items", &json!(vec![linked.clone(); 100]));
        first_page.query = vec![("itemType", "attachment".into()), ("start", "0".into())];
        let mut second_page = Reply::json("/api/users/0/items", &json!([attachment()]));
        second_page.query = vec![("itemType", "attachment".into()), ("start", "100".into())];
        second_page.version = version;
        let mut replies = vec![
            Reply::json("/api/users/0/groups", &json!([])),
            Reply::json("/api/users/0/items", &json!([linked])),
            Reply::json("/api/users/0/items", &json!([parent()])),
            linked_reply,
            first_page,
            second_page,
        ];

        if version == "1" {
            replies.push(file_reply(
                library.path(),
                "/api/users/0/items/ATTACH01/file/view/url",
            ));
        }

        let (mut api, _server) = server(replies);
        let mut items = vec![ZoteroItem {
            metadata: ZoteroItemMetadata {
                library_key: "LINKED01".into(),
                title: "A paper".into(),
                file_path: linked_path,
                authors: None,
            },
            text: String::new(),
        }];
        let result = api.authors(&mut items, library.path()).await;

        if version == "1" {
            result.unwrap();
            assert_eq!(
                items[0].metadata.authors.as_ref().unwrap()[0],
                "Lovelace, Ada"
            );
        } else {
            assert!(
                matches!(result, Err(LocalApiError::InvalidData(message)) if message.contains("library changed"))
            );
            assert!(items[0].metadata.authors.is_none());
        }
    }
}
