use clap::Parser;
use fern;
use humantime;
use log::LevelFilter;
use std::{
    io::Write,
    sync::{Arc, Mutex, atomic::AtomicBool},
};
use zqa_rag::llm::base::ChatHistoryItem;

use crate::config::Config;

#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Whether to use the tui interface. This is very unstable and is not feature-complete, and it
    /// should not be used for regular tasks.
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

/// The application state. This is embedded in the context, and all state variables are
/// encapsulated in this struct to avoid polluting the `Context`.
#[derive(Default)]
pub(crate) struct State {
    /// The current conversation's chat history
    pub(crate) chat_history: Arc<Mutex<Vec<ChatHistoryItem>>>,
    /// Has the chat history been modified?
    pub(crate) dirty: AtomicBool,
}

/// A structure that holds the application context, including CLI arguments and an writers
/// for `stdout` and `stderr`.
pub(crate) struct Context<OutStream: Write, ErrStream: Write> {
    /// Application state
    pub(crate) state: State,
    /// Config from TOML and env
    pub(crate) config: Config,
    /// CLI arguments passed
    pub(crate) args: Args,
    /// Abstraction for `stdout()`
    pub(crate) out: OutStream,
    /// Abstraction for `stderr()`
    pub(crate) err: ErrStream,
}

/// Initialize the `fern` logger.
///
/// # Arguments
///
/// * `log_level` - The log level to use.
///
/// # Errors
///
/// `log::SetLoggerError` if the logger could not be initialized.
pub fn setup_logger(log_level: LevelFilter) -> Result<(), log::SetLoggerError> {
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
        .chain(std::io::stdout())
        .apply()
}
