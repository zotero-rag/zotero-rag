use clap::Parser;
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

    /// Choice of embedding provider. Valid options are 'anthropic', 'openai', and 'voyageai'. Note
    /// that 'anthropic' uses the OpenAI embedding model, so will require `OPENAI_API_KEY` to be
    /// set. 'voyageai' will require `VOYAGE_AI_API_KEY` to be set.
    #[arg(short, long, default_value_t = String::from("voyageai"))]
    pub embedding: String,

    /// Choice of model provider, must be one of "anthropic" or "openai". Requires `ANTHROPIC_KEY`
    /// or `OPENAI_API_KEY` to be set accordingly.
    #[arg(short, long, default_value_t = String::from("anthropic"))]
    pub model_provider: String,
}

/// A structure that holds the application context, including CLI arguments and an output writer.
pub struct Context<W: Write> {
    // CLI arguments passed
    pub args: Args,
    // Abstraction that can write stuff
    pub out: W,
}
