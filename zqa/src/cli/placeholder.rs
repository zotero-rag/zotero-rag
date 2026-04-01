use std::cell::RefCell;
use std::rc::Rc;

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

    shown_hint: Rc<RefCell<Option<String>>>,
}

impl PlaceholderText {
    #[must_use]
    pub fn new(placeholder_text: String) -> Self {
        Self {
            placeholder_text,
            shown_hint: Rc::new(RefCell::new(None)),
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
fn get_best_file_match(query: &str) -> Option<String> {
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

    let mut matcher = nucleo_matcher::Matcher::new(nucleo_matcher::Config::DEFAULT.match_paths());
    let file_matches = pattern::Pattern::parse(
        query,
        pattern::CaseMatching::Smart,
        pattern::Normalization::Smart,
    )
    .match_list(filenames, &mut matcher);

    let best = file_matches.first()?.0.clone();
    let prefix_len = cwd.to_str()?.len() + query.len() + 1;

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
            && let Some(path) = path_ref.as_deref()
        {
            return Ok((pos, vec![path.to_string()]));
        }

        if line.chars().nth(pos - 1) == Some('@') {
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
            return Some(self.placeholder_text.clone());
        }

        if line.starts_with('/') {
            return SLASH_COMMANDS
                .iter()
                .find(|cmd| cmd.starts_with(line) && **cmd != line)
                .map(|cmd| cmd[pos..].to_string());
        }

        if let Some(space_pos) = line[..pos].rfind(' ')
            && line.len() > space_pos + 2
            && &line[space_pos + 1..space_pos + 2] == "@"
        {
            let query = &line[space_pos + 2..pos];

            let best_match = get_best_file_match(query);
            self.shown_hint.replace(best_match);

            return get_best_file_match(query);
        }

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
