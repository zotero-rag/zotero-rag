use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use nucleo_matcher::pattern;
use rustyline::Helper;
use rustyline::completion::Completer;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationResult, Validator};

use crate::common::UserDocument;

/// A struct that will implement placeholder text using readline. The placeholder text itself is
/// configurable, and the various `impl`s necessary to interface with `rustyline` (the readline
/// implementation) handle the rest.
pub struct PlaceholderText {
    pub(crate) text: String,

    /// Documents in the [`crate::common::State`].
    pub(crate) documents: Arc<RwLock<HashMap<String, Arc<UserDocument>>>>,

    /// The `nucleo` matcher object. Instantiating this is fairly expensive since it pre-allocates
    /// memory; and the docs recommend persisting it for multiple queries.
    matcher: RefCell<nucleo_matcher::Matcher>,

    /// The currently shown filename hint text.
    shown_hint: RefCell<Option<String>>,
}

impl PlaceholderText {
    #[must_use]
    pub(crate) fn new(
        placeholder_text: String,
        documents: Arc<RwLock<HashMap<String, Arc<UserDocument>>>>,
    ) -> Self {
        Self {
            text: placeholder_text,
            documents,
            matcher: RefCell::new(nucleo_matcher::Matcher::new(
                nucleo_matcher::Config::DEFAULT.match_paths(),
            )),
            shown_hint: RefCell::new(None),
        }
    }
}

#[derive(Clone, Copy)]
struct MentionSpan<'a> {
    start: usize,
    query: &'a str,
    force_quotes: bool,
}

const SLASH_COMMANDS: &[&str] = &[
    "/checkhealth",
    "/config",
    "/dedup",
    "/doctor",
    "/embed fix",
    "/embed",
    "/exit",
    "/help",
    "/index",
    "/new",
    "/process",
    "/quit",
    "/resume",
    "/search",
    "/stats",
    "/sync",
];

/// Returns the name of the best-match PDF file in the current directory, if any exist.
/// Matching is performed between `query` and files in the current directory using a fuzzy matcher.
///
/// # Arguments
///
/// * `query` - The string to fuzzy match with.
///
/// # Returns
///
/// If matches are found, returns the best match; otherwise, returns `None`.
fn get_best_file_match(query: &str, matcher: &mut nucleo_matcher::Matcher) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let files = std::fs::read_dir(&cwd).ok()?;

    let filenames = files
        .filter_map(std::result::Result::ok)
        .filter(|f| {
            f.path()
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
        })
        .filter_map(|f| Some(f.path().to_str()?.to_string()))
        .collect::<Vec<_>>();

    let file_matches = pattern::Pattern::parse(
        query,
        pattern::CaseMatching::Smart,
        pattern::Normalization::Smart,
    )
    .match_list(filenames, matcher);

    let best = file_matches.first()?.0.clone();
    let prefix_len = cwd.to_str()?.len() + 1;

    if best.starts_with(cwd.to_str()?) && best.len() >= prefix_len {
        Some(best[prefix_len..].to_string())
    } else {
        Some(best)
    }
}

fn format_mention_completion(path: &str, force_quotes: bool) -> String {
    if force_quotes || path.contains(char::is_whitespace) {
        format!("\"{path}\"")
    } else {
        path.to_string()
    }
}

fn get_active_mention(line: &str, pos: usize) -> Option<MentionSpan<'_>> {
    let prefix = &line[..pos];
    let at_pos = prefix.rfind('@')?;

    if at_pos > 0
        && prefix[..at_pos]
            .chars()
            .last()
            .is_some_and(|c| !c.is_whitespace())
    {
        return None;
    }

    let after_at = &prefix[at_pos + 1..];
    if let Some(query) = after_at.strip_prefix('"') {
        if query.contains('"') {
            return None;
        }

        Some(MentionSpan {
            start: at_pos + 1,
            query,
            force_quotes: true,
        })
    } else if after_at.contains(char::is_whitespace) {
        None
    } else {
        Some(MentionSpan {
            start: at_pos + 1,
            query: after_at,
            force_quotes: false,
        })
    }
}

fn get_file_completion_candidates(prefix: &str) -> Option<Vec<String>> {
    let cwd = std::env::current_dir().ok()?;
    let mut candidates = std::fs::read_dir(cwd)
        .ok()?
        .filter_map(std::result::Result::ok)
        .filter(|f| {
            f.path()
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
        })
        .filter_map(|f| {
            let name = f.file_name().into_string().ok()?;
            if prefix.is_empty() || name.starts_with(prefix) {
                Some(name)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    candidates.sort();
    Some(candidates)
}

/// Handles completions for the `PlaceholderText` Helper. Note that for showing dimmed
/// placeholders, we don't need to support completions at all.
impl Completer for PlaceholderText {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        if line.starts_with("/docs remove ") {
            let candidates: Vec<String> = if let Ok(lock) = self.documents.read() {
                lock.keys().cloned().collect()
            } else {
                Vec::new()
            };

            return Ok(("/docs remove ".len(), candidates));
        }

        if line.starts_with('/') {
            let candidates: Vec<String> = SLASH_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(line))
                .map(std::string::ToString::to_string)
                .collect();
            return Ok((0, candidates));
        }

        if let Some(mention) = get_active_mention(line, pos) {
            if let Ok(path_ref) = self.shown_hint.try_borrow()
                && let Some(path) = path_ref.as_deref()
            {
                return Ok((
                    mention.start,
                    vec![format_mention_completion(path, mention.force_quotes)],
                ));
            }

            let candidates = get_file_completion_candidates(mention.query)
                .unwrap_or_default()
                .into_iter()
                .map(|path| format_mention_completion(&path, mention.force_quotes))
                .collect();
            return Ok((mention.start, candidates));
        }

        Ok((0, Vec::new()))
    }
}

/// Handles displaying hints for the `PlaceholderText` Helper. For our use case, we really only
/// need to display the hint when there is no text entered by the user.
impl Hinter for PlaceholderText {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        if pos == 0 {
            self.shown_hint.replace(None);
            return Some(self.text.clone());
        }

        if line.starts_with('/') {
            self.shown_hint.replace(None);
            return SLASH_COMMANDS
                .iter()
                .find(|cmd| cmd.starts_with(line) && **cmd != line)
                .map(|cmd| cmd[pos..].to_string());
        }

        if let Some(mention) = get_active_mention(line, pos) {
            let best_match = self
                .matcher
                .try_borrow_mut()
                .ok()
                .and_then(|mut matcher| get_best_file_match(mention.query, &mut matcher));
            self.shown_hint.replace(best_match.clone());

            return best_match.map(|path| {
                format!(
                    "{} (Tab to accept)",
                    format_mention_completion(&path, mention.force_quotes)
                )
            });
        }

        self.shown_hint.replace(None);
        None
    }
}

/// Handles highlighting for the placeholder text. This relies on ANSI escape codes. For a
/// reference, see
/// [Wikipedia](https://en.wikipedia.org/wiki/ANSI_escape_code#Control_Sequence_Introducer_commands).
/// Alternatively, see `man 4 console_codes`, and search for "ECMA-48 CSI Sequences".
impl Highlighter for PlaceholderText {
    fn highlight_hint<'h>(&self, hint: &'h str) -> std::borrow::Cow<'h, str> {
        const DIM_TEXT: &str = "\x1b[2m";
        const RESET: &str = "\x1b[0m";

        format!("{DIM_TEXT}{hint}{RESET}").into()
    }
}

/// We do not have invalid scenarios for the simple placeholder text.
impl Validator for PlaceholderText {
    fn validate(
        &self,
        _ctx: &mut rustyline::validate::ValidationContext,
    ) -> rustyline::Result<rustyline::validate::ValidationResult> {
        Ok(ValidationResult::Valid(None))
    }
}

impl Helper for PlaceholderText {}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_temp_pdf(dir: &TempDir, name: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let mut file = File::create(&path).expect("Failed to create temp file");
        file.write_all(b"%PDF-1.4\n")
            .expect("Failed to write PDF content");
        path
    }

    #[test]
    #[serial]
    fn test_get_best_file_match_exact_match() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let original_cwd = std::env::current_dir().expect("Failed to get cwd");

        std::env::set_current_dir(temp_dir.path()).expect("Failed to change dir");

        create_temp_pdf(&temp_dir, "document.pdf");

        let mut matcher =
            nucleo_matcher::Matcher::new(nucleo_matcher::Config::DEFAULT.match_paths());

        let result = get_best_file_match("document", &mut matcher);

        std::env::set_current_dir(&original_cwd).ok();

        assert_eq!(result, Some("document.pdf".to_string()));
    }

    #[test]
    #[serial]
    fn test_get_best_file_match_fuzzy_match() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let original_cwd = std::env::current_dir().expect("Failed to get cwd");

        std::env::set_current_dir(temp_dir.path()).expect("Failed to change dir");

        create_temp_pdf(&temp_dir, "my_awesome_paper.pdf");

        let mut matcher =
            nucleo_matcher::Matcher::new(nucleo_matcher::Config::DEFAULT.match_paths());

        let result = get_best_file_match("paper", &mut matcher);

        std::env::set_current_dir(&original_cwd).ok();

        assert_eq!(result, Some("my_awesome_paper.pdf".to_string()));
    }

    #[test]
    #[serial]
    fn test_get_best_file_match_no_pdfs() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let original_cwd = std::env::current_dir().expect("Failed to get cwd");

        std::env::set_current_dir(temp_dir.path()).expect("Failed to change dir");

        let mut matcher =
            nucleo_matcher::Matcher::new(nucleo_matcher::Config::DEFAULT.match_paths());

        let result = get_best_file_match("anything", &mut matcher);

        std::env::set_current_dir(&original_cwd).ok();

        assert_eq!(result, None);
    }

    #[test]
    #[serial]
    fn test_get_best_file_match_multiple_pdfs() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let original_cwd = std::env::current_dir().expect("Failed to get cwd");

        std::env::set_current_dir(temp_dir.path()).expect("Failed to change dir");

        create_temp_pdf(&temp_dir, "paper_alpha.pdf");
        create_temp_pdf(&temp_dir, "paper_beta.pdf");
        create_temp_pdf(&temp_dir, "thesis.pdf");

        let mut matcher =
            nucleo_matcher::Matcher::new(nucleo_matcher::Config::DEFAULT.match_paths());

        let result = get_best_file_match("paper", &mut matcher);

        std::env::set_current_dir(&original_cwd).ok();

        assert!(result.is_some());
        let filename = result.unwrap();
        assert!(filename == "paper_alpha.pdf" || filename == "paper_beta.pdf");
    }

    #[test]
    #[serial]
    fn test_get_best_file_match_ignores_non_pdfs() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let original_cwd = std::env::current_dir().expect("Failed to get cwd");

        std::env::set_current_dir(temp_dir.path()).expect("Failed to change dir");

        let txt_path = temp_dir.path().join("document.txt");
        File::create(&txt_path).expect("Failed to create txt file");

        let mut matcher =
            nucleo_matcher::Matcher::new(nucleo_matcher::Config::DEFAULT.match_paths());

        let result = get_best_file_match("document", &mut matcher);

        std::env::set_current_dir(&original_cwd).ok();

        assert_eq!(result, None);
    }

    #[test]
    #[serial]
    fn test_get_best_file_match_case_insensitive_extension() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let original_cwd = std::env::current_dir().expect("Failed to get cwd");

        std::env::set_current_dir(temp_dir.path()).expect("Failed to change dir");

        create_temp_pdf(&temp_dir, "document.PDF");

        let mut matcher =
            nucleo_matcher::Matcher::new(nucleo_matcher::Config::DEFAULT.match_paths());

        let result = get_best_file_match("document", &mut matcher);

        std::env::set_current_dir(&original_cwd).ok();

        assert_eq!(result, Some("document.PDF".to_string()));
    }

    #[test]
    #[serial]
    fn test_get_best_file_match_empty_query() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let original_cwd = std::env::current_dir().expect("Failed to get cwd");

        std::env::set_current_dir(temp_dir.path()).expect("Failed to change dir");

        create_temp_pdf(&temp_dir, "document.pdf");

        let mut matcher =
            nucleo_matcher::Matcher::new(nucleo_matcher::Config::DEFAULT.match_paths());

        let result = get_best_file_match("", &mut matcher);

        std::env::set_current_dir(&original_cwd).ok();

        assert!(result.is_some());
    }

    #[test]
    fn test_format_mention_completion_quotes_spaces() {
        assert_eq!(
            format_mention_completion("image 1.pdf", false),
            "\"image 1.pdf\""
        );
    }

    #[test]
    fn test_get_active_mention_unquoted() {
        let mention = get_active_mention("Compare @symbols.pdf", "Compare @symbols.pdf".len())
            .expect("Expected active mention");
        assert_eq!(mention.start, 9);
        assert_eq!(mention.query, "symbols.pdf");
        assert!(!mention.force_quotes);
    }

    #[test]
    fn test_get_active_mention_quoted() {
        let line = "Compare @\"image 1.pdf";
        let mention = get_active_mention(line, line.len()).expect("Expected active mention");
        assert_eq!(mention.start, 9);
        assert_eq!(mention.query, "image 1.pdf");
        assert!(mention.force_quotes);
    }
}
