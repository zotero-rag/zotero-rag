use clap::Parser;
use fern;
use humantime;
use log::LevelFilter;
use std::io::Write;

#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Whether to use the tui interface. This is very unstable and is not feature-complete, and it
    /// should not be used for regular tasks.
    #[arg(long, default_value_t = false)]
    pub tui: bool,

    /// Log level
    #[arg(long, default_value_t = String::from("off"))]
    pub log_level: String,

    /// Choice of embedding provider. Valid options are 'anthropic', 'openai', 'voyageai',
    /// 'gemini', and 'cohere'. Note that 'anthropic' uses the OpenAI embedding model, so will require
    /// `OPENAI_API_KEY` to be set. 'voyageai' will require `VOYAGE_AI_API_KEY` to be set. 'gemini'
    /// requires either `GEMINI_API_KEY` or `GOOGLE_API_KEY` to be set. 'cohere' requires
    /// `COHERE_API_KEY` to be set.
    #[arg(short, long, default_value_t = String::from("voyageai"))]
    pub embedding_provider: String,

    /// Choice of embedding model. Must be a valid model provided by `embedding_provider`.
    #[arg(short, long, default_value_t = String::from("voyage-3-large"))]
    pub embedding_model: String,

    /// Choice of reranking provider. Valid options are 'voyageai' and 'cohere'. These require
    /// `VOYAGE_AI_API_KEY` and `COHERE_API_KEY` to be set respectively.
    #[arg(short, long, default_value_t = String::from("voyageai"))]
    pub reranker: String,

    /// Choice of model provider, must be one of "anthropic", "openai", "gemini", or "openrouter".
    /// Requires `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`
    /// to be set accordingly.
    #[arg(short, long, default_value_t = String::from("anthropic"))]
    pub model_provider: String,

    /// Choice of model to use. This must be a model that is available from the `model_provider`,
    /// and the appropriate key variable should be set accordingly.
    #[arg(short, long, default_value_t = String::from("claude-sonnet-4-5"))]
    pub model: String,

    /// The maximum number of concurrent embedding requests.
    #[arg(long, default_value_t = 5)]
    pub max_concurrent_requests: usize,
}

/// A structure that holds the application context, including CLI arguments and an writers
/// for `stdout` and `stderr`.
pub struct Context<OutStream: Write, ErrStream: Write> {
    // CLI arguments passed
    pub args: Args,
    // Abstraction for stdout()
    pub out: OutStream,
    // Abstraction for stderr()
    pub err: ErrStream,
}

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
            ))
        })
        .level(log_level)
        .level_for("rustyline", log::LevelFilter::Off)
        .chain(std::io::stdout())
        .apply()
}
