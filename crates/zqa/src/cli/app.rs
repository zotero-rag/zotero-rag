use std::fs;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Arc;

use rpassword::read_password;
use rustyline::error::ReadlineError;
use tokio::sync::mpsc;

use crate::cli::commands::{Command, parse_command};
use crate::cli::errors::CLIError;
use crate::cli::handlers::batch::handle_batch_cmd;
use crate::cli::handlers::cli::{
    handle_config_cmd, handle_help_cmd, handle_new_conversation_cmd, handle_quit_cmd,
};
use crate::cli::handlers::conversation::{handle_resume_cmd, save_current_conversation};
use crate::cli::handlers::documents::handle_docs_cmd;
use crate::cli::handlers::library::{
    handle_checkhealth_cmd, handle_dedup_cmd, handle_doctor_cmd, handle_index_cmd,
    handle_process_cmd, handle_stats_cmd,
};
use crate::cli::handlers::query::{handle_query_cmd, handle_search_cmd};
use crate::cli::placeholder::PlaceholderText;
use crate::cli::readline::get_readline_config;
use crate::common::Context;
use crate::io::EngineEvent;
use crate::state::get_state_dir;

/// Handle a single command or query from the user.
///
/// # Arguments
///
/// * `command` - The command string entered by the user.
/// * `ctx` - A `Context` object that contains CLI args
///
/// # Returns
///
/// `Ok(true)` if the CLI should continue running, `Ok(false)` if it should exit.
pub(crate) async fn dispatch_command(command: &str, ctx: &mut Context) -> Result<bool, CLIError> {
    let command =
        parse_command(command.trim()).map_err(|e| CLIError::CommandError(e.to_string()))?;

    match command {
        Command::Batch(subcmd) => handle_batch_cmd(subcmd, ctx).await,
        Command::CheckHealth => handle_checkhealth_cmd(ctx).await,
        Command::Config => handle_config_cmd(ctx).await,
        Command::Dedup => handle_dedup_cmd(ctx).await,
        Command::Docs(subcmd) => handle_docs_cmd(subcmd, ctx).await,
        Command::Doctor => handle_doctor_cmd(ctx).await,
        Command::DoNothing => {
            return Ok(true);
        }
        Command::Help => handle_help_cmd(ctx).await,
        Command::Index => handle_index_cmd(ctx).await,
        Command::NewConversation => handle_new_conversation_cmd(ctx).await,
        Command::Process => handle_process_cmd(ctx).await,
        Command::Quit => {
            return handle_quit_cmd(ctx).await.and(Ok(false));
        }
        Command::Query { text } => handle_query_cmd(text, ctx).await,
        Command::Resume => handle_resume_cmd(ctx).await,
        Command::Search { query } => handle_search_cmd(query, ctx).await,
        Command::Stats => handle_stats_cmd(ctx).await,
    }
    .and(Ok(true))
}

/// Run one REPL command, rendering its events while it runs.
///
/// Events are rendered as they arrive, so a prompt is answered from `input` while the command
/// waits on it. Events still queued when the command finishes are rendered before this returns.
/// If the command fails, its error is written to `err` and the REPL keeps running.
///
/// # Arguments
///
/// * `command` - The command string entered by the user.
/// * `ctx` - Application state, configuration, storage, and the event sender.
/// * `rx` - The receiver for events sent through `ctx`.
/// * `input` - Input for event-prompt replies.
/// * `output` - Output for model text, status updates, and prompts.
/// * `err` - Output for warnings and errors.
///
/// # Returns
///
/// `Ok(true)` if the REPL should keep running, `Ok(false)` if the command asked it to exit.
///
/// # Errors
///
/// * `CLIError::IOError` - If rendering an event, writing the command's error, or reading a prompt
///   reply fails.
async fn run_command<R: BufRead, O: Write, E: Write>(
    command: &str,
    ctx: &mut Context,
    rx: &mut mpsc::Receiver<EngineEvent>,
    input: &mut R,
    output: &mut O,
    err: &mut E,
) -> Result<bool, CLIError> {
    let result = {
        let dispatch = dispatch_command(command, ctx);
        tokio::pin!(dispatch);

        loop {
            tokio::select! {
                result = &mut dispatch => break result,
                Some(event) = rx.recv() => {
                    event.handle_event(input, output, err, read_password)?;
                }
            }
        }
    };

    while let Ok(event) = rx.try_recv() {
        event.handle_event(input, output, err, read_password)?;
    }

    match result {
        Ok(keep_running) => Ok(keep_running),
        Err(e) => {
            write!(err, "{e}")?;
            Ok(true)
        }
    }
}

/// The core CLI implementation that implements a REPL for user commands.
///
/// # Arguments
///
/// * `ctx` - Application state, configuration, storage, and the event sender.
/// * `rx` - A channel that receives `EngineEvent`s.
/// * `input` - Input for event-prompt replies.
/// * `output` - Output for model text, status updates, and prompts.
/// * `err` - Output for warnings and errors.
///
/// # Errors
///
/// * `CLIError::ReadlineError` - If we could not get the history file path, a `readline` editor could not be created,
///   or the history could not be saved.
/// * `CLIError::IOError` - If rendering an event, flushing output, or reading a prompt reply fails.
/// * `CLIError::StateDirError` - If the state dir could not be obtained.
#[allow(clippy::needless_continue)]
pub(crate) async fn cli<R: BufRead, O: Write, E: Write>(
    mut ctx: Context,
    mut rx: mpsc::Receiver<EngineEvent>,
    mut input: R,
    mut output: O,
    mut err: E,
) -> Result<(), CLIError> {
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
                &mut output,
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

                if !run_command(
                    &command,
                    &mut ctx,
                    &mut rx,
                    &mut input,
                    &mut output,
                    &mut err,
                )
                .await?
                {
                    break;
                }
            }
            Err(ReadlineError::Signal(rustyline::error::Signal::Resize)) => {
                // Handle SIGWINCH; we should just rewrite the prompt and continue
                continue;
            }
            Err(ReadlineError::Interrupted) => {
                // Handle SIGINT by saving the conversation if needed
                save_current_conversation(&mut ctx).await?;

                while let Ok(event) = rx.try_recv() {
                    event.handle_event(&mut input, &mut output, &mut err, read_password)?;
                }

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

    use std::io;
    use std::sync::{Arc, atomic};

    use serde_json::json;
    use serial_test::serial;
    use temp_env;
    use tokio::sync::mpsc;
    use zqa_macros::{test_contains, test_eq, test_ok};
    use zqa_macros_proc::retry;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::embedding::common::EmbeddingProviderConfig;
    use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, MessageRole};
    use zqa_rag::llm::tools::Tool;
    use zqa_rag::reranking::common::RerankProviderConfig;

    use super::{dispatch_command, run_command};
    use crate::common::test_support::{TestPaths, capture_events, create_test_context};
    use crate::io::EngineEvent;
    use crate::state::UsageMetadata;
    use crate::store::lance::LanceZoteroStore;
    use crate::tools::retrieval::RetrievalTool;

    fn make_retrieval_tool(paths: &TestPaths) -> RetrievalTool<LanceZoteroStore> {
        let api_key = std::env::var("VOYAGE_AI_API_KEY").unwrap_or_default();
        let config = zqa_rag::config::VoyageAIConfig {
            api_key,
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            max_concurrent_requests: zqa_rag::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
            max_retries: zqa_rag::constants::DEFAULT_MAX_RETRIES,
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

            let (result, events) =
                capture_events(&mut ctx, async |ctx| dispatch_command("/resume", ctx).await).await;
            test_ok!(result);
            assert!(result.unwrap());
            assert!(events.iter().any(|event| matches!(
                event,
                EngineEvent::Text { message } if message.contains("No saved conversations found.")
            )));
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn test_run_command_quit_then_resume_reloads_conversation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let state_dir = temp_dir.path().to_str().unwrap().to_string();

        temp_env::async_with_vars([("ZQA_STATE_DIR", Some(state_dir.as_str()))], async {
            let history = vec![ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::Text("Summarize this paper".into())],
            }];
            let usage = UsageMetadata {
                input_tokens: 1000,
                output_tokens: 500,
                estimated_cost: 12,
                ..UsageMetadata::default()
            };
            let mut output = Vec::new();
            let mut err = Vec::new();

            let mut ctx = create_test_context(vec![]);
            let (tx, mut rx) = mpsc::channel(256);
            ctx.event_tx = Some(tx);
            *ctx.state.title.lock().unwrap() = Some("Existing conversation".to_string());
            *ctx.state.chat_history.lock().unwrap() = history.clone();
            ctx.state.usage = usage;
            ctx.state.dirty.store(true, atomic::Ordering::Relaxed);

            let keep_running = run_command(
                "/quit",
                &mut ctx,
                &mut rx,
                &mut io::empty(),
                &mut output,
                &mut err,
            )
            .await;
            test_ok!(keep_running);
            assert!(!keep_running.unwrap());

            // A fresh context stands in for the next process: nothing is loaded until `/resume`.
            let mut ctx = create_test_context(vec![]);
            let (tx, mut rx) = mpsc::channel(256);
            ctx.event_tx = Some(tx);

            let keep_running = run_command(
                "/resume",
                &mut ctx,
                &mut rx,
                &mut "1\n".as_bytes(),
                &mut output,
                &mut err,
            )
            .await;
            test_ok!(keep_running);
            assert!(keep_running.unwrap());

            let output = String::from_utf8(output).unwrap();
            test_contains!(output, "Existing conversation (1 message)");
            test_contains!(output, "Resumed: Existing conversation");
            test_eq!(String::from_utf8(err).unwrap(), "");
            test_eq!(*ctx.state.chat_history.lock().unwrap(), history);
            test_eq!(
                *ctx.state.title.lock().unwrap(),
                Some("Existing conversation".to_string())
            );
            test_eq!(ctx.state.usage, usage);
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

        let (dedup_result, events) =
            capture_events(&mut ctx, async |ctx| dispatch_command("/dedup", ctx).await).await;
        test_ok!(dedup_result);
        assert!(dedup_result.unwrap());
        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::Text { message } if message.contains("Deduped ")
        )));
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
