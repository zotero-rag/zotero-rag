use std::io::Write;
use std::sync::Arc;

use rustyline::error::ReadlineError;

use crate::cli::commands::{Command, parse_command};
use crate::cli::errors::CLIError;
use crate::cli::handlers::cli::{
    handle_config_cmd, handle_help_cmd, handle_new_conversation_cmd, handle_quit_cmd,
};
use crate::cli::handlers::conversation::handle_resume_cmd;
use crate::cli::handlers::documents::handle_docs_cmd;
use crate::cli::handlers::library::{
    handle_checkhealth_cmd, handle_dedup_cmd, handle_doctor_cmd, handle_embed_cmd,
    handle_index_cmd, handle_process_cmd, handle_stats_cmd,
};
use crate::cli::handlers::query::{handle_query_cmd, handle_search_cmd};
use crate::cli::placeholder::PlaceholderText;
use crate::cli::readline::get_readline_config;
use crate::common::Context;

/// A file that contains parsed PDF texts from the user's Zotero library. In case the
/// embedding generation fails, the user does not need to rerun the full PDF parsing,
/// and can simply retry the embedding. Note that this is *not* supposed to be user-facing
/// and all interaction with it is meant for use by the CLI.
pub(crate) const BATCH_ITER_FILE: &str = "batch_iter.bin";

/// Handle a single command or query from the user.
///
/// # Arguments
///
/// * `command` - The command string entered by the user.
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(true)` if the CLI should continue running, `Ok(false)` if it should exit.
#[allow(clippy::too_many_lines)]
pub(crate) async fn dispatch_command<O: Write, E: Write>(
    command: &str,
    ctx: &mut Context<O, E>,
) -> Result<bool, CLIError> {
    let command =
        parse_command(command.trim()).map_err(|e| CLIError::CommandError(e.to_string()))?;

    match command {
        Command::DoNothing => {
            return Ok(true);
        }
        Command::Process => handle_process_cmd(ctx).await,
        Command::Embed { fix } => handle_embed_cmd(fix, ctx).await,
        Command::Dedup => handle_dedup_cmd(ctx).await,
        Command::Index => handle_index_cmd(ctx).await,
        Command::Resume => handle_resume_cmd(ctx).await,
        Command::Help => handle_help_cmd(ctx).await,
        Command::Quit => {
            return handle_quit_cmd(ctx).await.and(Ok(false));
        }
        Command::CheckHealth => handle_checkhealth_cmd(ctx).await,
        Command::Query { text } => handle_query_cmd(text, ctx).await,
        Command::Doctor => handle_doctor_cmd(ctx).await,
        Command::NewConversation => handle_new_conversation_cmd(ctx).await,
        Command::Config => handle_config_cmd(ctx).await,
        Command::Stats => handle_stats_cmd(ctx).await,
        Command::Search { query } => handle_search_cmd(query, ctx).await,
        Command::Docs(subcmd) => handle_docs_cmd(subcmd, ctx).await,
    }
    .and(Ok(true))
}

/// The core CLI implementation that implements a REPL for user commands.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Errors
///
/// * `CLIError::ReadlineError` - If we could not get the history file path, a `readline` editor could not be created,
///   or the history could not be saved.
/// * `CLIError::IOError` - If `writeln!` fails.
///
/// # Panics
///
/// Cannot happen; `unwrap()` is called on `strip_prefix` result after checking that the prefix exists.
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_continue)]
pub(crate) async fn cli<O: Write, E: Write>(mut ctx: Context<O, E>) -> Result<(), CLIError> {
    // First, get the path to the history file used by the `readline` implementation.
    let user_dirs = directories::UserDirs::new().ok_or(CLIError::ReadlineError(
        "Could not get user directories".into(),
    ))?;
    let home_dir = user_dirs.home_dir();
    let history_path = home_dir.join(".zqa_history");

    // Create the `readline` "editor" with our `PlaceholderText` helper
    let mut rl =
        rustyline::Editor::<PlaceholderText, rustyline::history::DefaultHistory>::with_config(
            get_readline_config(),
        )
        .map_err(|e| CLIError::ReadlineError(e.to_string()))?;

    if rl.load_history(&history_path).is_err() {
        log::debug!("No previous history.");
    }

    rl.set_helper(Some(PlaceholderText::new(
        "Type in a question or /help for options".to_string(),
        Arc::clone(&ctx.state.imports),
    )));

    loop {
        let readline = rl.readline(">>> ");

        match readline {
            Ok(command) => {
                if !command.trim().is_empty()
                    && let Err(e) = rl.add_history_entry(command.trim())
                {
                    log::debug!("Failed to write history entry: {e}");
                }

                if !dispatch_command(&command, &mut ctx).await? {
                    break;
                }
            }
            Err(ReadlineError::Signal(rustyline::error::Signal::Resize)) => {
                // Handle SIGWINCH; we should just rewrite the prompt and continue
                continue;
            }
            Err(ReadlineError::Interrupted) => {
                // Handle SIGINT by saving the conversation if needed
                // save_current_conversation(&mut ctx)?;
                break;
            }
            _ => break,
        }
    }

    rl.save_history(&history_path)
        .map_err(|e| CLIError::ReadlineError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs::{self, File};
    use std::io::Cursor;
    use std::sync::Arc;

    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatchIterator};
    use arrow_array::{RecordBatch, StringArray};
    use arrow_ipc::writer::FileWriter;
    use chrono::Local;
    use lancedb::connect;
    use serde_json::json;
    use serial_test::serial;
    use temp_env;
    use zqa_macros::{test_contains, test_eq, test_ok};
    use zqa_macros_proc::retry;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::embedding::common::EmbeddingProviderConfig;
    use zqa_rag::llm::base::{ASSISTANT_ROLE, ChatHistoryContent, ChatHistoryItem, USER_ROLE};
    use zqa_rag::llm::tools::Tool;
    use zqa_rag::reranking::common::RerankProviderConfig;

    use crate::common::Context;
    use crate::common::State;
    use crate::config::{Config, VoyageAIConfig};
    use crate::state::{SavedChatHistory, save_conversation};
    use crate::tools::retrieval::RetrievalTool;

    pub(crate) fn get_config() -> Config {
        let mut config = Config {
            voyageai: Some(VoyageAIConfig {
                reranker: Some(DEFAULT_VOYAGE_RERANK_MODEL.into()),
                embedding_model: Some(DEFAULT_VOYAGE_EMBEDDING_MODEL.into()),
                embedding_dims: Some(DEFAULT_VOYAGE_EMBEDDING_DIM as usize),
                api_key: Some(String::new()),
            }),
            ..Default::default()
        };

        config.read_env().unwrap();
        config
    }

    /// Create a default `Context` object where the output and error streams are buffers that can
    /// be written into. This allows for the output to be easily inspected in tests.
    pub(crate) fn create_test_context() -> Context<Cursor<Vec<u8>>, Cursor<Vec<u8>>> {
        let out_buf: Vec<u8> = Vec::new();
        let out = Cursor::new(out_buf);

        let err_buf: Vec<u8> = Vec::new();
        let err = Cursor::new(err_buf);

        let config = get_config();

        Context {
            state: State::default(),
            config,
            out,
            err,
        }
    }

    fn make_retrieval_tool(_schema_key: &str) -> RetrievalTool {
        let api_key = std::env::var("VOYAGE_AI_API_KEY").unwrap_or_default();
        let config = zqa_rag::config::VoyageAIConfig {
            api_key,
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
        };

        // Build a minimal tool; the embedding config is only used in `call`, not in the metadata
        // methods, so we use a dummy VoyageAI config here.
        RetrievalTool::new(
            EmbeddingProviderConfig::VoyageAI(config.clone()),
            Some(RerankProviderConfig::VoyageAI(config)),
        )
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_embed() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();

        // Create `RecordBatch` object to write out
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "pdf_text",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(data)]).unwrap();

        // Write out the object to `BATCH_ITER_FILE`
        let file = File::create(BATCH_ITER_FILE).unwrap();
        let mut writer = FileWriter::try_new(file, &schema).unwrap();

        writer.write(&record_batch).unwrap();
        writer.finish().unwrap();

        // Actually call `dispatch_command`
        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/embed", &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.unwrap()); // Should continue loop

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let err = String::from_utf8(ctx.err.into_inner()).unwrap();
        assert!(err.is_empty());

        // Clean up
        if fs::metadata(BATCH_ITER_FILE).is_ok() {
            fs::remove_file(BATCH_ITER_FILE).expect("Failed to clean up BATCH_ITER_FILE");
        }
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_process() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();

        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/process", &mut ctx),
        )
        .await;

        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.clone().into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let stats = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/stats", &mut ctx),
        )
        .await;
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_ok!(stats);
        assert!(stats.unwrap());
        assert!(output.contains("Table statistics:"));
        assert!(output.contains("Number of rows: 8"));

        // Cleanup
        if fs::metadata(BATCH_ITER_FILE).is_ok() {
            fs::remove_file(BATCH_ITER_FILE).expect("Failed to clean up BATCH_ITER_FILE");
        }
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_search_only() {
        dotenv::dotenv().ok();
        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();

        // `process` needs to be run before `search_for_papers`
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(result);

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command(
                "/search How should I oversample in defect prediction?",
                &mut ctx,
            ),
        )
        .await;

        if let Err(e) = &result {
            dbg!(e);
        }

        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.len() > 20);
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_run_query() {
        dotenv::dotenv().ok();
        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();

        // `process` needs to be run before `run_query`
        let _ = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/process", &mut setup_ctx),
        )
        .await;

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("How should I oversample in defect prediction?", &mut ctx),
        )
        .await;

        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Total token usage:"));
    }

    #[tokio::test]
    #[serial]
    async fn test_checkhealth_no_database() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();
        let output = temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], async move {
            dispatch_command("/checkhealth", &mut ctx).await.unwrap();
            String::from_utf8(ctx.out.into_inner()).unwrap()
        })
        .await;

        assert!(output.contains("directory does not exist"));
    }

    #[retry(3)]
    #[tokio::test]
    #[serial]
    async fn test_checkhealth_with_database() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        // First create a database by running process
        let mut setup_ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(result);

        // Now run health check
        let mut ctx = create_test_context();
        let output = temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], async move {
            dispatch_command("/checkhealth", &mut ctx).await.unwrap();
            String::from_utf8(ctx.out.into_inner()).unwrap()
        })
        .await;

        test_contains!(output, "LanceDB Health Check Results");
        test_contains!(output, "directory exists");
        test_contains!(output, "Table is accessible");
        test_contains!(output, "Table has");
    }

    #[test]
    fn test_resume_no_conversations() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let mut ctx = create_test_context();
            let mut reader = Cursor::new("");
            resume(&mut ctx, &mut reader).unwrap();

            let output = String::from_utf8(ctx.out.into_inner()).unwrap();
            test_contains!(output, "No saved conversations found.");
        });
    }

    #[test]
    fn test_resume_loads_selected_conversation() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let history_a = vec![
                ChatHistoryItem {
                    role: USER_ROLE.into(),
                    content: vec![ChatHistoryContent::Text("What is attention?".into())],
                },
                ChatHistoryItem {
                    role: ASSISTANT_ROLE.into(),
                    content: vec![ChatHistoryContent::Text(
                        "Attention is a mechanism...".into(),
                    )],
                },
            ];
            let history_b = vec![ChatHistoryItem {
                role: USER_ROLE.into(),
                content: vec![ChatHistoryContent::Text(
                    "Tell me about transformers.".into(),
                )],
            }];

            save_conversation(&SavedChatHistory {
                history: history_a.clone(),
                date: Local::now(),
                title: "Conversation A".into(),
            })
            .unwrap();

            save_conversation(&SavedChatHistory {
                history: history_b.clone(),
                // Avoid same filename
                date: Local::now() + chrono::Duration::seconds(1),
                title: "Conversation B".into(),
            })
            .unwrap();

            let mut ctx = create_test_context();
            // The list is sorted reverse-chronologically; B was saved last so it's [1].
            let mut reader = Cursor::new("1\n");
            resume(&mut ctx, &mut reader).unwrap();

            let out = String::from_utf8(ctx.out.into_inner()).unwrap();
            test_contains!(out, "Resumed:");

            let loaded = ctx.state.chat_history.lock().unwrap();
            test_eq!(loaded.len(), history_b.len());
            assert!(!ctx.state.dirty.load(std::sync::atomic::Ordering::Relaxed));
        });
    }

    #[test]
    fn test_resume_invalid_selection() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            save_conversation(&SavedChatHistory {
                history: vec![ChatHistoryItem {
                    role: USER_ROLE.into(),
                    content: vec![ChatHistoryContent::Text("Hello".into())],
                }],
                date: Local::now(),
                title: "Only Conversation".into(),
            })
            .unwrap();

            let mut ctx = create_test_context();
            let mut reader = Cursor::new("99\n");
            resume(&mut ctx, &mut reader).unwrap();

            let err = String::from_utf8(ctx.err.into_inner()).unwrap();
            test_contains!(err, "Invalid selection.");
        });
    }

    #[test]
    fn test_get_document_mentions_unquoted() {
        let mentions = get_document_mentions("Compare @symbols.pdf with @subtables.pdf");
        test_eq!(mentions, vec!["symbols.pdf", "subtables.pdf"]);
    }

    #[test]
    fn test_get_document_mentions_quoted_spaces() {
        let mentions = get_document_mentions("Summarize @\"image 1.pdf\" for me");
        test_eq!(mentions, vec!["image 1.pdf"]);
    }

    #[test]
    fn test_get_document_mentions_ignores_email_like_text() {
        let mentions = get_document_mentions("email me at test@example.com about @symbols.pdf");
        test_eq!(mentions, vec!["symbols.pdf"]);
    }

    #[test]
    fn test_get_document_session_key_preserves_relative_input() {
        let original = std::path::Path::new("papers/image 1.pdf");
        let key = get_document_session_key(original);

        test_ok!(key);
        let key = key.unwrap();

        test_eq!(key, "papers/image 1.pdf");
    }

    #[tokio::test]
    async fn test_dispatch_command_help() {
        let mut ctx = create_test_context();
        let result = dispatch_command("/help", &mut ctx).await.unwrap();
        assert!(result);
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "Available commands:");

        // Check that all the `match` arms are contained
        test_contains!(output, "/help");
        test_contains!(output, "/checkhealth");
        test_contains!(output, "/doctor");
        test_contains!(output, "/embed");
        test_contains!(output, "/process");
        test_contains!(output, "/index");
        test_contains!(output, "/stats");
        test_contains!(output, "/dedup");
        test_contains!(output, "/resume");
        test_contains!(output, "/config");
        test_contains!(output, "/new");
        test_contains!(output, "/quit"); // one of the variants suffices to show the user
    }

    /// Inserts a single row with a zero-valued embedding vector and empty `pdf_text` directly into
    /// the LanceDB table at `db_uri`. This bypasses the embedding function so we can deterministically
    /// create zero-embedding rows for testing `fix_zero_embeddings`.
    async fn insert_zero_embedding_row(db_uri: &str) {
        let dims = DEFAULT_VOYAGE_EMBEDDING_DIM as i32;
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new(
                "embeddings",
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::Float32,
                        true,
                    )),
                    dims,
                ),
                false,
            ),
        ]));

        #[allow(clippy::cast_sign_loss)]
        let zeros = Float32Array::from(vec![0.0f32; dims as usize]);
        let embedding_col = FixedSizeListArray::try_new(
            Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                true,
            )),
            dims,
            Arc::new(zeros),
            None,
        )
        .unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["ZEROTEST001"])),
                Arc::new(StringArray::from(vec!["Zero Test Item"])),
                Arc::new(StringArray::from(vec!["/dev/null"])),
                Arc::new(StringArray::from(vec![""])), // will be deleted by fix
                Arc::new(embedding_col),
            ],
        )
        .unwrap();

        let db = connect(db_uri).execute().await.unwrap();
        let tbl = db.open_table("data").execute().await.unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        tbl.merge_insert(&["library_key"])
            .when_not_matched_insert_all()
            .clone()
            .execute(Box::new(reader))
            .await
            .unwrap();
    }

    /// When the database has no zero-embedding records, `fix_zero_embeddings` (via `/embed fix`)
    /// should complete immediately and report "Done!". Running `/embed fix` twice after `/process`
    /// guarantees the second run sees a clean database (the first clears any zeros produced by
    /// unparseable CI fixtures).
    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_fix_zero_embeddings_no_zeros() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(setup_result);

        // First run clears any zero-embedding rows produced by unparseable CI fixtures.
        let mut first_ctx = create_test_context();
        let first_result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/embed fix", &mut first_ctx),
        )
        .await;
        test_ok!(first_result);
        assert!(first_result.unwrap());

        // Second run: the database is clean, so the function should exit early with "Done!".
        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/embed fix", &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "Done!");
    }

    /// When zero-embedding rows whose `pdf_text` is empty are present, `fix_zero_embeddings` (via
    /// `/embed fix`) should delete them and report how many were removed.
    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_fix_zero_embeddings_with_zero_rows() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(setup_result);

        // Directly inject a row with a zero embedding vector and empty text into the DB.
        insert_zero_embedding_row(&db_uri).await;

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/embed fix", &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "items had empty texts, and will be deleted.");
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_call_returns_results_key() {
        dotenv::dotenv().ok();
        let _ = crate::common::setup_logger(log::LevelFilter::Info);

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();

        // Create test database with assets data
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            dispatch_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(setup_result);

        // Now test the retrieval tool
        let tool = make_retrieval_tool("input_schema");

        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            tool.call(json!({"query": "machine learning"})),
        )
        .await;
        test_ok!(result);

        let value = result.unwrap();
        assert!(
            value.get("results").is_some(),
            "Expected 'results' key in output, got: {value}"
        );
    }
}
