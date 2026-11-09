#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

use std::fs;
use std::io::{IsTerminal, stderr, stdout};
use std::path::Path;

use clap::Parser;

pub mod cli;
pub mod common;
pub mod config;
pub mod io;
pub mod session;
pub mod state;
pub mod store;
pub mod tools;
pub mod utils;

// Re-export commonly used items
use cli::app::{cli, render_while};
use common::{Args, Context, setup_logger};
use config::Config;
use state::{check_or_create_first_run_file, oobe};
pub use store::lance::LanceZoteroStore;
use tokio::sync::{mpsc, oneshot};
pub use utils::arrow::full_library_to_arrow;
use zqa_rag::config::{LLMClientConfig, OllamaConfig};
use zqa_rag::embedding::common::EmbeddingProviderConfig;
use zqa_rag::reranking::common::RerankProviderConfig;

use crate::cli::errors::CLIError;
use crate::common::{PathOptions, State};
use crate::io::{EngineEvent, TerminalProgress};
use crate::state::StateError;
use crate::utils::terminal::{RESET, YELLOW, YELLOW_BOLD};

/// Run first-run setup, then load the config it wrote into `ctx`.
///
/// Setup asks its questions as prompt events on `ctx`, so the caller must render events and reply
/// while this runs. If the user declines setup, nothing is written and the config is reloaded
/// unchanged. The store is left as it is; [`start`] builds it from the new config afterwards.
///
/// # Arguments
///
/// * `ctx` - The context whose event sender carries the prompts, and that receives the new config.
///
/// # Errors
///
/// * Returns a [`CLIError`] if setup fails or the new config can't be loaded. `ctx` keeps its
///   previous config in that case.
pub(crate) async fn run_setup(ctx: &mut Context) -> Result<(), CLIError> {
    oobe(ctx).await?;

    ctx.config = load_config()?;

    Ok(())
}

/// Load the application configuration from the TOML config file (if present) and the
/// environment, in priority order: TOML < env.
///
/// # Errors
///
/// Returns a [`CLIError`] if the config file exists but cannot be parsed.
///
/// # Panics
///
/// Panics if reading configuration from the environment fails.
pub fn load_config() -> Result<Config, CLIError> {
    // Load the configs in priority order: TOML < env < CLI args
    let mut config = Config::default();
    if let Some(user_dirs) = directories::UserDirs::new() {
        let config_path = user_dirs
            .home_dir()
            .join(".config")
            .join("zqa")
            .join("config.toml");

        if config_path.exists() {
            config = Config::from_file(config_path)?;
        }
    }

    // Overwrite with env
    config.read_env().unwrap();

    Ok(config)
}

/// Point LanceDB at the XDG state directory unless the caller has already set `LANCEDB_URI`.
///
/// The vector store resolves its location from the `LANCEDB_URI` environment variable, falling
/// back to a path relative to the current working directory (see
/// [`zqa_rag::vector::backends::lance::get_db_uri`]). Every front-end (the CLI and the GUI)
/// must call this at startup so they all open the same database; otherwise a front-end launched
/// from a different working directory would open an empty relative path and report a missing
/// table.
///
/// This sets a process-global environment variable, so it must be called while the process is
/// still single-threaded, i.e. before spawning any threads or async tasks.
pub fn set_default_lancedb_uri() {
    if std::env::var("LANCEDB_URI").is_err()
        && let Ok(state_dir) = state::get_state_dir()
    {
        let db_path = state_dir.join(zqa_rag::vector::backends::lance::LANCEDB_URI);

        // Safety: callers invoke this at startup, before any threads/async tasks spawn.
        unsafe { std::env::set_var("LANCEDB_URI", db_path) };
    }
}

/// List a notice for each role in `config` whose provider needs an API key and has none.
///
/// Generation gets a warning, since commands such as `/search` still work without it. Embedding
/// and reranking get errors, since most commands need them. Reranking is checked only when a
/// reranker provider is set: omitting `reranker_provider` opts out of reranking.
///
/// # Arguments
///
/// * `config` - The config to check.
///
/// # Returns
///
/// The `Warning` and `Error` events to show, in role order. Empty if every key is set.
fn missing_api_key_events(config: &Config) -> Vec<EngineEvent> {
    let mut events = Vec::new();

    if config.get_generation_config().is_none_or(|c| {
        (match c {
            LLMClientConfig::Anthropic(cfg) => cfg.api_key,
            LLMClientConfig::OpenAI(cfg) => cfg.api_key,
            LLMClientConfig::Gemini(cfg) => cfg.api_key,
            LLMClientConfig::OpenRouter(cfg) => cfg.api_key,
            LLMClientConfig::Ollama(_) => String::from("foo"), // Doesn't need one
            _ => unimplemented!("unsupported generation provider: {}", c.provider_id()),
        })
        .is_empty()
    }) {
        events.push(EngineEvent::Warning {
            message: "No API key is set for generation models. Some commands will not work. Set up a config at ~/.config/zqa/config.toml or in a `.env` file.\n".into(),
        });
    }

    if config.get_embedding_config().is_none_or(|c| {
        (match c {
            EmbeddingProviderConfig::OpenAI(cfg) => cfg.api_key,
            EmbeddingProviderConfig::Cohere(cfg) => cfg.api_key,
            EmbeddingProviderConfig::VoyageAI(cfg) => cfg.api_key,
            EmbeddingProviderConfig::Gemini(cfg) => cfg.api_key,
            // Ollama doesn't need an API key so we force a false condition
            EmbeddingProviderConfig::Ollama(_) => String::from("foo"),
            _ => unimplemented!("unsupported embedding provider: {}", c.provider_id()),
        })
        .is_empty()
    }) {
        events.push(EngineEvent::Error {
            message: "No API key is set for embedding models. Most commands will not work. Set up a config at ~/.config/zqa/config.toml or in a `.env` file.\n".into(),
        });
    }

    if config.reranker_provider.is_some()
        && config.get_reranker_config().is_none_or(|c| {
            (match c {
                RerankProviderConfig::VoyageAI(cfg) => cfg.api_key,
                RerankProviderConfig::Cohere(cfg) => cfg.api_key,
                _ => unimplemented!("unsupported reranking provider: {}", c.provider_id()),
            })
            .is_empty()
        })
    {
        events.push(EngineEvent::Error {
            message: "No API key is set for reranking. Most commands will not work. Set up a config at ~/.config/zqa/config.toml; if you want to opt out of reranking, omit `reranker_provider` instead.\n".into(),
        });
    }

    events
}

/// Count the embedding batches submitted to a batch API whose results haven't been collected.
///
/// Each pending batch has a `batch_*.log` file in `<state dir>/batches`, which is created lazily on
/// the first batch. [`crate::cli::handlers::batch`] has more details about the batch lifecycle.
///
/// This counts the log files without asking the batch APIs about each batch, since waiting on those
/// requests would slow startup. `/batch check` does that.
///
/// # Returns
///
/// The number of pending batches, or 0 if no batch was ever submitted.
///
/// # Errors
///
/// * Returns a [`CLIError`] if the state directory can't be found or the batch directory can't be
///   read.
fn pending_batch_count() -> Result<usize, CLIError> {
    let batch_dir = state::get_state_dir()?.join("batches");
    if !batch_dir.exists() {
        return Ok(0);
    }

    let count = fs::read_dir(&batch_dir)?
        .filter_map(|x| x.ok()?.file_name().into_string().ok())
        .filter(|f| {
            f.starts_with("batch_")
                && Path::new(f)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("log"))
        })
        .count();

    Ok(count)
}

/// Build the store for `config`, or a placeholder if `config` can't build one yet.
///
/// First-run setup can complete a config that has no section for its embedding provider, so a store
/// is only required once [`start`] has finished. Startup itself doesn't use the store, and until
/// then the context holds an Ollama one, which needs no API key to build. Frontends build the real
/// store from the final config after `start` returns, even if it returned an error. Admittedly, not
/// the slickest way to do this, but oh well.
///
/// # Arguments
///
/// * `config` - The config loaded before startup.
///
/// # Returns
///
/// The store for `config`, or the Ollama placeholder if `config` has no section for its embedding
/// provider.
pub(crate) fn store_or_placeholder(config: &Config) -> LanceZoteroStore {
    LanceZoteroStore::from_config(config).unwrap_or_else(|_| {
        LanceZoteroStore::from_embedding_config(EmbeddingProviderConfig::Ollama(
            OllamaConfig::default(),
        ))
    })
}

/// Run the startup steps every frontend needs before its first command.
///
/// In order, this:
///
/// 1. Checks for the first-run marker. If that fails, the user picks whether to retry, continue as
///    if this weren't a first run (optionally seeing the full error), or quit.
/// 2. Runs first-run setup if this is the first run. A setup failure is reported as an `Error`
///    event and startup continues with the config `ctx` already has.
/// 3. Reports each missing API key as a `Warning` or `Error` event. This comes after setup so a
///    first run isn't warned about keys it is about to ask for.
/// 4. Reports pending embedding batches as a `StatusUpdate`.
///
/// When `interactive` is `false`, steps 1 and 2 are skipped. Nothing marks the first run as done
/// then, so setup is still offered the next time an interactive frontend starts.
///
/// Every message and prompt is an event on `ctx`, so the caller must render events and reply while
/// this runs.
///
/// # Arguments
///
/// * `ctx` - The context whose event sender carries the notices and prompts. Setup replaces its
///   config and store.
/// * `interactive` - Whether someone can answer prompts, such as the CLI reading from a terminal.
///
/// # Returns
///
/// `Ok(true)` to continue to the first command, or `Ok(false)` if the user chose to quit.
///
/// # Errors
///
/// * Returns a [`CLIError`] if an event can't be sent, a prompt is dropped without a reply, or the
///   batch directory can't be read.
pub(crate) async fn start(ctx: &mut Context, interactive: bool) -> Result<bool, CLIError> {
    let is_first_run = interactive
        && loop {
            let error = match check_or_create_first_run_file() {
                Ok(is_first_run) => break is_first_run,
                Err(error) => error,
            };

            let (reply, answer) = oneshot::channel();
            ctx.emit(EngineEvent::Choose {
                message: format!("Error setting up: {error}."),
                options: vec![
                    "Retry".into(),
                    "Ignore".into(),
                    "Show error and ignore".into(),
                    "Quit".into(),
                ],
                reply,
                default: 0,
            })
            .await?;

            // Every choice except retrying continues as if this isn't a first run.
            match answer.await.map_err(StateError::from)? {
                0 => {}
                1 => break false,
                2 => {
                    ctx.emit(EngineEvent::Error {
                        message: format!("{error:?}\n"),
                    })
                    .await?;
                    break false;
                }
                _ => return Ok(false),
            }
        };

    if is_first_run && let Err(e) = run_setup(ctx).await {
        ctx.emit(EngineEvent::Error {
            message: format!("Error during setup: {e}\n"),
        })
        .await?;
    }

    ctx.emit_multiple(missing_api_key_events(&ctx.config))
        .await?;

    let pending = pending_batch_count()?;
    if pending > 0 {
        ctx.emit(EngineEvent::StatusUpdate {
            message: format!(
                "You have {pending} embedding batch{} pending. Use /batch check to check their status.\n\n",
                if pending > 1 { "es" } else { "" }
            ),
        })
        .await?;
    }

    Ok(true)
}

/// Main entry point for the zqa application.
///
/// This function handles configuration loading, first-run setup, and launches the CLI.
///
/// # Errors
///
/// Returns an error if the CLI encounters an unrecoverable error during execution.
///
/// # Panics
///
/// * If the logger could not be set up
/// * If the input could not be read from `stdin`.
pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Point LanceDB at the XDG state dir if the caller hasn't already overridden it.
    set_default_lancedb_uri();

    let config = load_config()?;
    let args = Args::parse();

    // Avoid RUST_LOG from interfering by not instantiating the logger if it's disabled.
    setup_logger(args.log_level).expect("Failed to set up logger.");

    log::debug!(
        "You are running {} version {}. Log level: {}",
        env!("CARGO_CRATE_NAME"),
        env!("CARGO_PKG_VERSION"),
        args.log_level
    );

    if args.log_level < log::LevelFilter::Info && args.print_summaries {
        let warning = "`--print-summaries` requires `--log-level=info`. This will have no effect.";

        if args.log_level <= log::LevelFilter::Warn {
            log::warn!("{warning}");
        } else {
            eprintln!("{YELLOW_BOLD}warn: {RESET}{YELLOW}{warning}{RESET}");
        }
    }

    log::debug!(
        "Loaded configuration: generation_provider={:?}, embedding_provider={:?}, reranker_provider={:?}, tool_iteration_limit={}",
        config.model_provider,
        config.embedding_provider,
        config.reranker_provider,
        config.tool_iteration_limit
    );

    let store = store_or_placeholder(&config);

    let (event_tx, mut event_rx) = mpsc::channel(256);
    let mut context = Context {
        state: State::default(),
        event_tx: Some(event_tx),
        config,
        store,
        path_options: PathOptions::default(),
    };

    let (keep_running, _) = render_while(
        start(&mut context, std::io::stdin().is_terminal()),
        &mut event_rx,
        &mut std::io::stdin().lock(),
        &mut stdout(),
        &mut stderr(),
        &mut TerminalProgress::new(),
    )
    .await?;

    if !keep_running? {
        std::process::exit(1);
    }

    // Replace the placeholder from `store_or_placeholder` with the store for the final config.
    context.store = LanceZoteroStore::from_config(&context.config)?;

    cli(
        context,
        event_rx,
        std::io::stdin().lock(),
        stdout(),
        stderr(),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use zqa_rag::capabilities::ModelProvider;

    use super::start;
    use crate::common::test_support::{capture_events, create_test_context};
    use crate::io::EngineEvent;

    #[tokio::test]
    #[serial]
    async fn start_skips_setup_without_a_terminal() {
        let temp_dir = tempfile::tempdir().unwrap();
        let state_dir = temp_dir.path().to_str().unwrap().to_string();

        temp_env::async_with_vars([("ZQA_STATE_DIR", Some(state_dir.as_str()))], async {
            // The startup key check doesn't know the mock provider, so use a real one.
            let mut ctx = create_test_context(vec![]);
            ctx.config.model_provider = ModelProvider::Anthropic;

            let (result, events) =
                capture_events(&mut ctx, async |ctx| start(ctx, false).await).await;

            assert!(result.unwrap());
            assert!(!events.iter().any(|event| matches!(
                event,
                EngineEvent::Confirm { .. }
                    | EngineEvent::Choose { .. }
                    | EngineEvent::Line { .. }
                    | EngineEvent::Secret { .. }
            )));

            // The first run stays pending, so the next interactive start still offers setup.
            assert!(!temp_dir.path().join("first_run").exists());
        })
        .await;
    }
}
