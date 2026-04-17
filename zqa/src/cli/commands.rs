use thiserror::Error;

pub(crate) enum Command {
    Help,
    CheckHealth,
    DoNothing,
    Doctor,
    Embed { fix: bool },
    Process,
    Index,
    Stats,
    Dedup,
    Resume,
    Config,
    NewConversation,
    Quit,
    Docs(DocsCommand),
    Search { query: String },
    Query { text: String },
}

pub(crate) enum DocsCommand {
    Clear,
    List,
    Remove { key: String },
}

#[derive(Debug, Error)]
pub(crate) enum CommandParseError {
    #[error("invalid command: {0}")]
    InvalidCommand(String),
}

pub(crate) fn parse_command(command: &str) -> Result<Command, CommandParseError> {
    match command {
        "" => Ok(Command::DoNothing),
        "/help" | "help" | "?" => Ok(Command::Help),
        "/checkhealth" => Ok(Command::CheckHealth),
        "/doctor" => Ok(Command::Doctor),
        "/embed" => Ok(Command::Embed { fix: false }),
        "/process" => Ok(Command::Process),
        "/index" => Ok(Command::Index),
        "/stats" => Ok(Command::Stats),
        "/dedup" => Ok(Command::Dedup),
        "/resume" => Ok(Command::Resume),
        "/config" => Ok(Command::Config),
        "/quit" | "/exit" | "quit" | "exit" | "/new" => Ok(Command::Quit),
        query => {
            // Check for a threshold to ensure this isn't an accidental Enter-hit.
            #[allow(clippy::items_after_statements)]
            const MIN_QUERY_LENGTH: usize = 10;

            if query.len() < MIN_QUERY_LENGTH {
                return Err(CommandParseError::InvalidCommand(query.to_string()));
            }

            // Search queries have priority
            if query.starts_with("/search") {
                let search_term = query.strip_prefix("/search").unwrap().trim();
                if search_term.is_empty() {
                    return Err(CommandParseError::InvalidCommand(
                        "Please provide a search term after /search.".to_string(),
                    ));
                }

                return Ok(Command::Search {
                    query: search_term.to_string(),
                });
            } else if query.starts_with("/embed") {
                // Handle `/embed fix`
                let subcmd = query.strip_prefix("/embed").unwrap().trim();
                if subcmd != "fix" {
                    return Err(CommandParseError::InvalidCommand(format!(
                        "Invalid subcommand to /embed: {subcmd}"
                    )));
                }

                return Ok(Command::Embed { fix: true });
            } else if query.starts_with("/docs") {
                let subcmd = query.strip_prefix("/docs").unwrap().trim();
                let parts: Vec<&str> = subcmd.splitn(2, ' ').collect();

                return match parts[0] {
                    "clear" => Ok(Command::Docs(DocsCommand::Clear)),
                    "list" => Ok(Command::Docs(DocsCommand::List)),
                    "remove" => Ok(Command::Docs(DocsCommand::Remove {
                        key: parts[1].to_string(),
                    })),
                    _ => Err(CommandParseError::InvalidCommand(format!(
                        "Invalid subcommand to /docs: {subcmd}"
                    ))),
                };
            }

            Ok(Command::Query { text: query.into() })
        }
    }
}
