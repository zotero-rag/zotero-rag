use std::io::Write;
use std::sync::Arc;

use rustyline::error::ReadlineError;

use crate::cli::commands::{Command, parse_command};
use crate::cli::errors::CLIError;
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
        Command::DoNothing => {
            return Ok(true);
        }
        Command::Process => handle_process_cmd(ctx).await,
        Command::Embed { fix } => handle_embed_cmd(fix, ctx).await,
        Command::Dedup => handle_dedup_cmd(ctx).await,
        Command::Index => handle_index_cmd(ctx).await,
        Command::Resume => handle_resume_cmd(ctx),
        Command::Help => handle_help_cmd(ctx),
        Command::Quit => {
            return handle_quit_cmd(ctx).and(Ok(false));
        }
        Command::CheckHealth => handle_checkhealth_cmd(ctx).await,
        Command::Query { text } => handle_query_cmd(text, ctx).await,
        Command::Doctor => handle_doctor_cmd(ctx).await,
        Command::NewConversation => handle_new_conversation_cmd(ctx),
        Command::Config => handle_config_cmd(ctx),
        Command::Stats => handle_stats_cmd(ctx).await,
        Command::Search { query } => handle_search_cmd(query, ctx).await,
        Command::Docs(subcmd) => handle_docs_cmd(subcmd, ctx),
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
pub(crate) mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use serde_json::json;
    use serial_test::serial;
    use temp_env;
    use zqa_macros::test_ok;
    use zqa_macros_proc::retry;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::embedding::common::EmbeddingProviderConfig;
    use zqa_rag::llm::tools::Tool;
    use zqa_rag::reranking::common::RerankProviderConfig;
    use zqa_rag::vector::backends::lance::LanceBackend;

    use super::dispatch_command;
    use crate::common::Context;
    use crate::common::State;
    use crate::config::{Config, VoyageAIConfig};
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

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]);

        let config = get_config();

        Context {
            state: State::default(),
            backend: LanceBackend::new(
                config.get_embedding_config().unwrap(),
                Arc::new(schema),
                "pdf_text".into(),
            ),
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
