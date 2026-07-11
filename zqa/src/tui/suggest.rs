//! Input suggestions for the TUI query box.
//!
//! Suggestions appear below the query box, like in most modern coding TUIs:
//!
//! * A line starting with `/` suggests matching slash commands (with descriptions).
//! * `/docs remove ` suggests keys of documents imported into the session.
//! * An unfinished `@` file mention suggests PDF files in the current directory, reusing the
//!   CLI completer's mention parsing.

use crate::cli::commands::SLASH_COMMANDS;
use crate::cli::placeholder::{
    format_mention_completion, get_active_mention, get_file_completion_candidates,
};

/// A completion suggestion shown below the query box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Suggestion {
    /// The full input line after accepting this suggestion.
    pub(crate) replacement: String,
    /// The completion text shown in the list.
    pub(crate) label: String,
    /// A short description shown dimmed next to the label; may be empty.
    pub(crate) description: &'static str,
}

/// Compute suggestions for the current input line.
///
/// # Arguments
///
/// * `line` - The current input line.
/// * `cursor` - The cursor's byte offset in `line`, used to locate `@` mentions.
/// * `documents` - Keys of documents imported into the session, for `/docs remove`.
///
/// # Returns
///
/// Matching suggestions, or an empty list when nothing applies.
pub(crate) fn suggestions_for(line: &str, cursor: usize, documents: &[String]) -> Vec<Suggestion> {
    if line.is_empty() {
        return Vec::new();
    }

    if let Some(prefix) = line.strip_prefix("/docs remove ") {
        return documents
            .iter()
            .filter(|key| key.starts_with(prefix) && key.as_str() != prefix)
            .map(|key| Suggestion {
                replacement: format!("/docs remove {key}"),
                label: key.clone(),
                description: "",
            })
            .collect();
    }

    if line.starts_with('/') {
        return SLASH_COMMANDS
            .iter()
            .filter(|cmd| cmd.name.starts_with(line) && cmd.name != line)
            .map(|cmd| Suggestion {
                // The trailing space makes chained completion (e.g. `/docs remove <key>`)
                // flow naturally, and is harmless for argument-less commands.
                replacement: format!("{} ", cmd.name),
                label: cmd.name.to_string(),
                description: cmd.description,
            })
            .collect();
    }

    if let Some(mention) = get_active_mention(line, cursor) {
        let candidates = get_file_completion_candidates(mention.query).unwrap_or_default();

        return candidates
            .into_iter()
            .map(|path| {
                let completed = format_mention_completion(&path, mention.force_quotes);
                let replacement =
                    format!("{}{completed}{}", &line[..mention.start], &line[cursor..]);

                Suggestion {
                    replacement,
                    label: path,
                    description: "",
                }
            })
            .collect();
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use serial_test::serial;
    use tempfile::TempDir;
    use zqa_macros::test_eq;

    use super::suggestions_for;

    #[test]
    fn test_slash_command_suggestions_match_prefix() {
        let suggestions = suggestions_for("/che", 4, &[]);

        test_eq!(suggestions.len(), 1);
        test_eq!(suggestions[0].replacement, "/checkhealth ");
        test_eq!(suggestions[0].label, "/checkhealth");
        assert!(!suggestions[0].description.is_empty());
    }

    #[test]
    fn test_exact_command_is_not_suggested() {
        let suggestions = suggestions_for("/help", 5, &[]);
        assert!(suggestions.is_empty());

        // A shorter exact match still suggests longer commands.
        let suggestions = suggestions_for("/embed", 6, &[]);
        test_eq!(suggestions.len(), 1);
        test_eq!(suggestions[0].label, "/embed fix");
    }

    #[test]
    fn test_docs_remove_suggests_session_documents() {
        let documents = vec!["alpha.pdf".to_string(), "beta.pdf".to_string()];
        let suggestions = suggestions_for("/docs remove a", 14, &documents);

        test_eq!(suggestions.len(), 1);
        test_eq!(suggestions[0].replacement, "/docs remove alpha.pdf");
    }

    #[test]
    fn test_plain_text_has_no_suggestions() {
        assert!(suggestions_for("what papers discuss attention?", 5, &[]).is_empty());
        assert!(suggestions_for("", 0, &[]).is_empty());
    }

    #[test]
    #[serial]
    fn test_mention_suggests_pdfs_in_cwd() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let original_cwd = std::env::current_dir().expect("Failed to get cwd");
        std::env::set_current_dir(temp_dir.path()).expect("Failed to change dir");

        let mut file = File::create(temp_dir.path().join("paper.pdf")).unwrap();
        file.write_all(b"%PDF-1.4\n").unwrap();

        let line = "Summarize @pa";
        let suggestions = suggestions_for(line, line.len(), &[]);

        std::env::set_current_dir(&original_cwd).ok();

        test_eq!(suggestions.len(), 1);
        test_eq!(suggestions[0].replacement, "Summarize @paper.pdf");
    }
}
