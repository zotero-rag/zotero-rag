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
    Batch(BatchSubcommand),
    Docs(DocsCommand),
    Search { query: String },
    Query { text: String },
}

pub(crate) enum BatchSubcommand {
    Submit,
    Status { batch_id: Option<String> },
    Collect { batch_id: Option<String> },
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
        "/new" => Ok(Command::NewConversation),
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
        "/quit" | "/exit" | "quit" | "exit" => Ok(Command::Quit),
        query if query.starts_with('/') => {
            if let Some(search_term) = query.strip_prefix("/search") {
                let search_term = search_term.trim();
                if search_term.is_empty() {
                    return Err(CommandParseError::InvalidCommand(
                        "Please provide a search term after /search.".to_string(),
                    ));
                }

                return Ok(Command::Search {
                    query: search_term.to_string(),
                });
            }

            if let Some(subcmd) = query.strip_prefix("/embed") {
                let subcmd = subcmd.trim();
                if subcmd != "fix" {
                    return Err(CommandParseError::InvalidCommand(format!(
                        "Invalid subcommand to /embed: {subcmd}"
                    )));
                }

                return Ok(Command::Embed { fix: true });
            }

            if let Some(rest) = query.strip_prefix("/batch") {
                let rest = rest.trim();
                let mut parts = rest.splitn(2, ' ');
                let sub = parts.next().unwrap_or("");
                let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());

                return match sub {
                    "submit" => Ok(Command::Batch(BatchSubcommand::Submit)),
                    "status" => Ok(Command::Batch(BatchSubcommand::Status {
                        batch_id: arg.map(str::to_string),
                    })),
                    "collect" => Ok(Command::Batch(BatchSubcommand::Collect {
                        batch_id: arg.map(str::to_string),
                    })),
                    _ => Err(CommandParseError::InvalidCommand(format!(
                        "Invalid subcommand to /batch: '{rest}'. Valid subcommands: submit, status, collect."
                    ))),
                };
            }

            if let Some(subcmd) = query.strip_prefix("/docs") {
                let subcmd = subcmd.trim();
                let parts: Vec<&str> = subcmd.splitn(2, ' ').collect();

                return match parts[0] {
                    "clear" => Ok(Command::Docs(DocsCommand::Clear)),
                    "list" => Ok(Command::Docs(DocsCommand::List)),
                    "remove" => {
                        let key = parts
                            .get(1)
                            .map(|key| key.trim())
                            .filter(|key| !key.is_empty());
                        match key {
                            Some(key) => Ok(Command::Docs(DocsCommand::Remove {
                                key: key.to_string(),
                            })),
                            None => Err(CommandParseError::InvalidCommand(
                                "Please provide a key after /docs remove.".to_string(),
                            )),
                        }
                    }
                    _ => Err(CommandParseError::InvalidCommand(format!(
                        "Invalid subcommand to /docs: {subcmd}"
                    ))),
                };
            }

            Err(CommandParseError::InvalidCommand(query.to_string()))
        }
        query => {
            // Check for a threshold to ensure this isn't an accidental Enter-hit.
            #[allow(clippy::items_after_statements)]
            const MIN_QUERY_LENGTH: usize = 10;

            if query.len() < MIN_QUERY_LENGTH {
                return Err(CommandParseError::InvalidCommand(query.to_string()));
            }

            Ok(Command::Query { text: query.into() })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, CommandParseError, DocsCommand, parse_command};

    #[test]
    fn test_parse_search_command() {
        match parse_command("/search a") {
            Ok(Command::Search { query }) => assert_eq!(query, "a"),
            Ok(_) => panic!("unexpected command variant"),
            Err(err) => panic!("unexpected parse error: {err}"),
        }
    }

    #[test]
    fn test_parse_docs_remove_requires_key() {
        assert!(matches!(
            parse_command("/docs remove"),
            Err(CommandParseError::InvalidCommand(message))
                if message == "Please provide a key after /docs remove."
        ));
    }

    #[test]
    fn test_parse_docs_remove_command() {
        match parse_command("/docs remove paper.pdf") {
            Ok(Command::Docs(DocsCommand::Remove { key })) => assert_eq!(key, "paper.pdf"),
            Ok(_) => panic!("unexpected command variant"),
            Err(err) => panic!("unexpected parse error: {err}"),
        }
    }

    #[test]
    fn test_parse_short_query_is_invalid() {
        assert!(matches!(
            parse_command("short"),
            Err(CommandParseError::InvalidCommand(message)) if message == "short"
        ));
    }
}
