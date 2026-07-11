use std::{
    collections::HashMap,
    io::{BufRead, Write},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64},
    },
};

use clap::Parser;
use fern;
use humantime;
use log::LevelFilter;
use zqa_pdftools::parse::ExtractedContent;
use zqa_rag::llm::{base::ChatHistoryItem, tools::CallbackFn};

use crate::{config::Config, store::lance::LanceZoteroStore};

#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Use the full-screen terminal UI instead of the line-based CLI. The TUI shows the
    /// conversation, a query box, and a sidebar with session statistics.
    #[arg(long, default_value_t = false)]
    pub tui: bool,

    /// Log level. Options: debug, info, warn, error, off (default)
    #[arg(long, default_value = "off")]
    pub log_level: log::LevelFilter,

    /// Whether to also print out the extracted snippets of the retrieved papers. Note that these
    /// are printed out at the INFO log level, so if the log level is unset or lower than INFO, this
    /// effectively does nothing.
    #[arg(short, long, default_value_t = false)]
    pub print_summaries: bool,
}

/// A user-imported document that is not from their Zotero library.
#[derive(Clone)]
pub(crate) struct UserDocument {
    /// The filename as on disk
    pub(crate) filename: String,
    /// The result of `extract_text`
    pub(crate) contents: ExtractedContent,
    /// The text before "Introduction"
    pub(crate) summary: String,
}

/// The application state. This is embedded in the context, and all state variables are
/// encapsulated in this struct to avoid polluting the `Context`.
#[derive(Default)]
pub(crate) struct State {
    /// The current conversation's chat history
    pub(crate) chat_history: Arc<Mutex<Vec<ChatHistoryItem>>>,
    /// Has the chat history been modified?
    pub(crate) dirty: AtomicBool,
    /// A title generated for the current conversation
    pub(crate) title: Arc<Mutex<Option<String>>>,
    /// Accumulated session cost, in US cents (Money pattern)
    /// TODO: How do providers handle other currencies?
    pub(crate) session_cost: AtomicU64,
    /// Accumulated input tokens across all queries in this session
    pub(crate) session_input_tokens: AtomicU64,
    /// Accumulated output tokens across all queries in this session
    pub(crate) session_output_tokens: AtomicU64,
    /// Extracted content for imported documents
    pub(crate) imports: Arc<RwLock<HashMap<String, Arc<UserDocument>>>>,
}

/// A structure that holds the application context, including CLI arguments and writers
/// for `stdout` and `stderr`.
pub(crate) struct Context<OutStream: Write, ErrStream: Write> {
    /// Application state
    pub(crate) state: State,
    /// Config from TOML and env
    pub(crate) config: Config,
    /// The store to use for storage and retrieval
    pub(crate) store: LanceZoteroStore,
    /// Abstraction for `stdin()`. Boxed rather than generic because, unlike `out`/`err` (which
    /// tests substitute with an inspectable `Cursor` to assert on), input is only ever *supplied*:
    /// no caller needs the concrete reader type. Boxing also keeps the input concern off every
    /// handler signature.
    pub(crate) input: Box<dyn BufRead>,
    /// Abstraction for `stdout()`
    pub(crate) out: OutStream,
    /// Abstraction for `stderr()`
    pub(crate) err: ErrStream,
    /// Callback for text streamed from the model as it arrives. Handlers fall back to printing
    /// on `stdout` when this is `None`; the TUI routes streamed text into its transcript.
    /// A callback (rather than a `Write` handle) because it is shared with `'static` tasks
    /// that outlive any borrow of `out`.
    pub(crate) on_stream_text: Option<Arc<CallbackFn<str>>>,
    /// Callback for tool trace lines (tool invocations and timings). Tool wrappers fall back to
    /// printing dimmed text on `stderr` when this is `None`.
    pub(crate) on_tool_trace: Option<Arc<CallbackFn<str>>>,
}

/// Initialize the `fern` logger.
///
/// # Arguments
///
/// * `log_level` - The log level to use.
/// * `output` - Where log lines are written. The CLI logs to `stdout`; the TUI logs to a file
///   since it owns the terminal.
///
/// # Errors
///
/// * `log::SetLoggerError` if the logger could not be initialized.
pub fn setup_logger(
    log_level: LevelFilter,
    output: fern::Output,
) -> Result<(), log::SetLoggerError> {
    // Set up logging via fern
    fern::Dispatch::new()
        // Perform allocation-free log formatting
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                humantime::format_rfc3339_millis(std::time::SystemTime::now()),
                record.level(),
                record.target(),
                message
            ));
        })
        .level(log_level)
        .level_for("rustyline", log::LevelFilter::Off)
        .chain(output)
        .apply()
}
