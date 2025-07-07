use clap::Parser;

#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Whether to use the tui interface
    #[arg(long, default_value_t = false)]
    pub tui: bool,

    /// Choice of embedding provider. Valid options are 'anthropic', 'openai', and 'voyageai'. Note
    /// that 'anthropic' uses the OpenAI embedding model, so will require `OPENAI_API_KEY` to be
    /// set. 'voyageai' will require `VOYAGE_AI_API_KEY` to be set.
    #[arg(short, long, default_value_t = String::from("voyageai"))]
    pub embedding: String,
}
