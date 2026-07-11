use thiserror::Error;

/// A slash command completion, along with a short description for suggestion lists.
pub(crate) struct SlashCommandSpec {
    /// The text inserted when the command is completed.
    pub(crate) name: &'static str,
    /// A one-line summary of what the command does.
    pub(crate) description: &'static str,
}

/// All slash-command completions, used by the CLI's readline completer and the TUI's
/// suggestion list.
pub(crate) const SLASH_COMMANDS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        name: "/batch check",
        description: "Check on the status of a submitted batch",
    },
    SlashCommandSpec {
        name: "/batch create",
        description: "Pre-process Zotero library with a batch embedding API",
    },
    SlashCommandSpec {
        name: "/checkhealth",
        description: "Run health checks on your LanceDB",
    },
    SlashCommandSpec {
        name: "/config",
        description: "Show the currently used configuration",
    },
    SlashCommandSpec {
        name: "/docs clear",
        description: "Clear all documents in this session",
    },
    SlashCommandSpec {
        name: "/docs list",
        description: "List all documents in this session",
    },
    SlashCommandSpec {
        name: "/docs remove",
        description: "Remove a document from the session",
    },
    SlashCommandSpec {
        name: "/dedup",
        description: "Remove duplicate items",
    },
    SlashCommandSpec {
        name: "/doctor",
        description: "Attempt to fix issues spotted by /checkhealth",
    },
    SlashCommandSpec {
        name: "/embed fix",
        description: "Repair rows with zero embeddings",
    },
    SlashCommandSpec {
        name: "/embed",
        description: "Repair failed DB creation by re-adding embeddings",
    },
    SlashCommandSpec {
        name: "/exit",
        description: "Exit the program",
    },
    SlashCommandSpec {
        name: "/help",
        description: "Show the help message",
    },
    SlashCommandSpec {
        name: "/index",
        description: "Create or update indices",
    },
    SlashCommandSpec {
        name: "/new",
        description: "Save the current conversation and start a new one",
    },
    SlashCommandSpec {
        name: "/process",
        description: "Pre-process Zotero library to update the database",
    },
    SlashCommandSpec {
        name: "/quit",
        description: "Exit the program",
    },
    SlashCommandSpec {
        name: "/resume",
        description: "Resume a previous conversation",
    },
    SlashCommandSpec {
        name: "/search",
        description: "Search for papers without summarizing them",
    },
    SlashCommandSpec {
        name: "/stats",
        description: "Show table statistics",
    },
];

pub(crate) enum BatchCommand {
    Cancel(usize),
    CheckStatus,
    Create,
}

pub(crate) enum Command {
    Batch(BatchCommand),
    CheckHealth,
    Config,
    Dedup,
    Docs(DocsCommand),
    Doctor,
    DoNothing,
    Embed { fix: bool },
    Help,
    Index,
    NewConversation,
    Process,
    Query { text: String },
    Quit,
    Resume,
    Search { query: String },
    Stats,
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
        "/checkhealth" => Ok(Command::CheckHealth),
        "/config" => Ok(Command::Config),
        "/dedup" => Ok(Command::Dedup),
        "/doctor" => Ok(Command::Doctor),
        "/embed" => Ok(Command::Embed { fix: false }),
        "/help" | "help" | "?" => Ok(Command::Help),
        "/index" => Ok(Command::Index),
        "/new" => Ok(Command::NewConversation),
        "/process" => Ok(Command::Process),
        "/resume" => Ok(Command::Resume),
        "/stats" => Ok(Command::Stats),
        "/quit" | "/exit" | "quit" | "exit" => Ok(Command::Quit),
        query if query.starts_with('/') => {
            // Handle commands that take subcommands
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

            if let Some(subcmd) = query.strip_prefix("/batch") {
                let subcmd = subcmd.trim();
                let parts: Vec<&str> = subcmd.splitn(2, ' ').collect();

                return match parts[0] {
                    "cancel" => {
                        let id = parts
                            .get(1)
                            .and_then(|key| key.trim().parse::<usize>().ok());

                        match id {
                            None => Err(CommandParseError::InvalidCommand(format!(
                                "Invalid parameter to /batch cancel: '{}', expected a number.",
                                parts.get(1).copied().unwrap_or("")
                            ))),
                            Some(id) => Ok(Command::Batch(BatchCommand::Cancel(id))),
                        }
                    }
                    "check" => Ok(Command::Batch(BatchCommand::CheckStatus)),
                    "create" => Ok(Command::Batch(BatchCommand::Create)),
                    _ => Err(CommandParseError::InvalidCommand(format!(
                        "Invalid subcommand to /batch: {subcmd}"
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
