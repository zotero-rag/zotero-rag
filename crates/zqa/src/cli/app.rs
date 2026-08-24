use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use rustyline::error::ReadlineError;

use crate::cli::commands::{Command, parse_command};
use crate::cli::errors::CLIError;
use crate::cli::handlers::batch::handle_batch_cmd;
use crate::cli::handlers::cli::{
    handle_config_cmd, handle_help_cmd, handle_new_conversation_cmd, handle_quit_cmd,
};
use crate::cli::handlers::conversation::{handle_resume_cmd, save_current_conversation};
use crate::cli::handlers::documents::handle_docs_cmd;
use crate::cli::handlers::library::{
    handle_checkhealth_cmd, handle_dedup_cmd, handle_doctor_cmd, handle_embed_cmd,
    handle_index_cmd, handle_process_cmd, handle_stats_cmd,
};
use crate::cli::handlers::query::{handle_query_cmd, handle_search_cmd};
use crate::cli::placeholder::PlaceholderText;
use crate::cli::readline::get_readline_config;
use crate::common::Context;
use crate::state::get_state_dir;

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
pub(crate) async fn dispatch_command<O: Write, E: Write>(
    command: &str,
    ctx: &mut Context<O, E>,
) -> Result<bool, CLIError> {
    let command =
        parse_command(command.trim()).map_err(|e| CLIError::CommandError(e.to_string()))?;

    match command {
        Command::Batch(subcmd) => handle_batch_cmd(subcmd, ctx).await,
        Command::CheckHealth => handle_checkhealth_cmd(ctx).await,
        Command::Config => handle_config_cmd(ctx),
        Command::Dedup => handle_dedup_cmd(ctx).await,
        Command::Docs(subcmd) => handle_docs_cmd(subcmd, ctx),
        Command::Doctor => handle_doctor_cmd(ctx).await,
        Command::DoNothing => {
            return Ok(true);
        }
        Command::Embed { fix } => handle_embed_cmd(fix, ctx).await,
        Command::Help => handle_help_cmd(ctx),
        Command::Index => handle_index_cmd(ctx).await,
        Command::NewConversation => handle_new_conversation_cmd(ctx),
        Command::Process => handle_process_cmd(ctx).await,
        Command::Quit => {
            return handle_quit_cmd(ctx).and(Ok(false));
        }
        Command::Query { text } => handle_query_cmd(text, ctx).await,
        Command::Resume => handle_resume_cmd(ctx),
        Command::Search { query } => handle_search_cmd(query, ctx).await,
        Command::Stats => handle_stats_cmd(ctx).await,
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
/// * `CLIError::StateDirError` - If the state dir could not be obtained.
#[allow(clippy::needless_continue)]
pub(crate) async fn cli<O: Write, E: Write>(mut ctx: Context<O, E>) -> Result<(), CLIError> {
    // At startup, we should check if there are pending embedding batches and notify the user if so.
    // [`crate::cli::handlers::batch`] has more details about the semantics of interacting with
    // batch APIs. We don't do the check itself since we have two bad options:
    //
    // * If we `await` the check, we risk a longer startup time
    // * If we use a task, we need to be careful about not messing with the CLI output/readline.
    let batch_dir = get_state_dir()?.join("batches");

    // `batch_dir` is created lazily on the first batch
    if batch_dir.exists() {
        let batch_files: Vec<String> = fs::read_dir(&batch_dir)?
            .filter_map(|x| x.ok()?.file_name().into_string().ok())
            .filter(|f| {
                f.starts_with("batch_")
                    && Path::new(f)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("log"))
            })
            .collect();

        if !batch_files.is_empty() {
            writeln!(
                &mut ctx.out,
                "You have {} embedding batch{} pending. Use /batch check to check their status.\n",
                batch_files.len(),
                if batch_files.len() > 1 { "es" } else { "" }
            )?;
        }
    }

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

                match dispatch_command(&command, &mut ctx).await {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(CLIError::CommandError(ref e)) => {
                        if let Err(write_err) = writeln!(&mut ctx.err, "Error: {e}") {
                            log::error!("Failed to write to stderr: {write_err}");
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(ReadlineError::Signal(rustyline::error::Signal::Resize)) => {
                // Handle SIGWINCH; we should just rewrite the prompt and continue
                continue;
            }
            Err(ReadlineError::Interrupted) => {
                // Handle SIGINT by saving the conversation if needed
                save_current_conversation(&mut ctx)?;
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
mod tests {

    use std::sync::{Arc, atomic};

    use serde_json::json;
    use serial_test::serial;
    use temp_env;
    use zqa_macros::{test_contains, test_eq, test_ok};
    use zqa_macros_proc::retry;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::embedding::common::EmbeddingProviderConfig;
    use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, MessageRole};
    use zqa_rag::llm::tools::Tool;
    use zqa_rag::reranking::common::RerankProviderConfig;

    use super::dispatch_command;
    use crate::common::test_support::{TestPaths, create_test_context};
    use crate::store::lance::LanceZoteroStore;
    use crate::tools::retrieval::RetrievalTool;

    fn make_retrieval_tool(paths: &TestPaths) -> RetrievalTool<LanceZoteroStore> {
        let api_key = std::env::var("VOYAGE_AI_API_KEY").unwrap_or_default();
        let config = zqa_rag::config::VoyageAIConfig {
            api_key,
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
        };
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]));
        let store = LanceZoteroStore::from_schema(
            EmbeddingProviderConfig::VoyageAI(config.clone()),
            schema,
        )
        .with_uri(&paths.db_uri);
        RetrievalTool::new(
            Arc::new(store),
            Some(RerankProviderConfig::VoyageAI(config)),
            paths.path_options.library_path.clone(),
        )
    }

    #[tokio::test]
    #[serial]
    async fn test_dispatch_new_command_resets_conversation_state() {
        let temp_dir = tempfile::tempdir().unwrap();
        let state_dir = temp_dir.path().to_str().unwrap().to_string();

        temp_env::async_with_vars([("ZQA_STATE_DIR", Some(state_dir.as_str()))], async {
            let mut ctx = create_test_context(vec![]);
            *ctx.state.title.lock().unwrap() = Some("Existing conversation".to_string());
            ctx.state
                .chat_history
                .lock()
                .unwrap()
                .push(ChatHistoryItem {
                    role: MessageRole::User,
                    content: vec![ChatHistoryContent::Text("Summarize this paper".into())],
                });
            ctx.state.dirty.store(true, atomic::Ordering::Relaxed);

            let result = dispatch_command("/new", &mut ctx).await;
            test_ok!(result);
            assert!(result.unwrap());

            assert!(!ctx.state.dirty.load(atomic::Ordering::Relaxed));
            assert!(ctx.state.chat_history.lock().unwrap().is_empty());
            test_eq!(*ctx.state.title.lock().unwrap(), None);
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn test_dispatch_resume_command_handles_no_saved_conversations() {
        let temp_dir = tempfile::tempdir().unwrap();
        let state_dir = temp_dir.path().to_str().unwrap().to_string();

        temp_env::async_with_vars([("ZQA_STATE_DIR", Some(state_dir.as_str()))], async {
            let mut ctx = create_test_context(vec![]);

            let result = dispatch_command("/resume", &mut ctx).await;
            test_ok!(result);
            assert!(result.unwrap());

            let output = String::from_utf8(ctx.out.into_inner()).unwrap();
            test_contains!(output, "No saved conversations found.");
        })
        .await;
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_dispatch_dedup_command() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut ctx = paths.context(vec![]);

        let process_result = dispatch_command("/process", &mut ctx).await;
        test_ok!(process_result);
        assert!(process_result.unwrap());

        let dedup_result = dispatch_command("/dedup", &mut ctx).await;
        test_ok!(dedup_result);
        assert!(dedup_result.unwrap());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "Deduped ");
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_call_returns_results_key() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut setup_ctx = paths.context(vec![]);

        // Create test database with assets data
        let setup_result = dispatch_command("/process", &mut setup_ctx).await;
        test_ok!(setup_result);

        // Now test the retrieval tool
        let tool = make_retrieval_tool(&paths);

        let result = tool.call(json!({"query": "machine learning"})).await;
        test_ok!(result);

        let value = result.unwrap();
        assert!(
            value.get("results").is_some(),
            "Expected 'results' key in output, got: {value}"
        );
    }
}
