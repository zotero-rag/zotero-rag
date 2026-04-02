use std::cell::RefCell;

use nucleo_matcher::pattern;
use rustyline::Helper;
use rustyline::completion::{Completer, FilenameCompleter};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationResult, Validator};

/// A struct that will implement placeholder text using readline. The placeholder text itself is
/// configurable, and the various `impl`s necessary to interface with `rustyline` (the readline
/// implementation) handle the rest.
pub struct PlaceholderText {
    pub placeholder_text: String,

    /// The `nucleo` matcher object. Instantiating this is fairly expensive since it pre-allocates
    /// memory; and the docs recommend persisting it for multiple queries.
    matcher: RefCell<nucleo_matcher::Matcher>,

    /// The currently shown filename hint text.
    shown_hint: RefCell<Option<String>>,
}

impl PlaceholderText {
    #[must_use]
    pub fn new(placeholder_text: String) -> Self {
        Self {
            placeholder_text,
            matcher: RefCell::new(nucleo_matcher::Matcher::new(
                nucleo_matcher::Config::DEFAULT.match_paths(),
            )),
            shown_hint: RefCell::new(None),
        }
    }
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
        if line.starts_with('/') {
            let candidates: Vec<String> = SLASH_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(line))
                .map(std::string::ToString::to_string)
                .collect();
            return Ok((0, candidates));
        }

        if let Ok(path_ref) = self.shown_hint.try_borrow()
            && let Some(at_pos) = line[..pos].rfind('@')
            && let Some(path) = path_ref.as_deref()
        {
            return Ok((at_pos + 1, vec![path.to_string()]));
        }

        if line[..pos].ends_with('@') {
            return FilenameCompleter::new()
                .complete_path(line, pos)
                .map(|(_, pairs)| pairs.into_iter().map(|p| p.replacement).collect::<Vec<_>>())
                .map(|v| (pos, v));
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
            return Some(self.placeholder_text.clone());
        }

        if line.starts_with('/') {
            self.shown_hint.replace(None);
            return SLASH_COMMANDS
                .iter()
                .find(|cmd| cmd.starts_with(line) && **cmd != line)
                .map(|cmd| cmd[pos..].to_string());
        }

        // I was very proud of this `if` chain, don't you dare remove it
        if let Some(at_pos) = line[..pos].rfind('@')
            && let Some(space_pos) = line[..pos].rfind(' ').map_or_else(|| Some(0), Some)
            && at_pos >= space_pos  // Equality case handles first char '@', no space yet
            && at_pos < pos
        {
            let query = &line[at_pos + 1..pos];
            let best_match = self
                .matcher
                .try_borrow_mut()
                .ok()
                .and_then(|mut matcher| get_best_file_match(query, &mut matcher));
            self.shown_hint.replace(best_match.clone());

            return best_match;
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
}
