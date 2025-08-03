use rustyline::Helper;
use rustyline::completion::Completer;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationResult, Validator};

/// A struct that will implement placeholder text using readline. The placeholder text itself is
/// configurable, and the various `impl`s necessary to interface with `rustyline` (the readline
/// implementation) handle the rest.
pub struct PlaceholderText {
    pub placeholder_text: String,
}

/// Handles completions for the `PlaceholderText` Helper. Note that for showing dimmed
/// placeholders, we don't need to support completions at all.
impl Completer for PlaceholderText {
    type Candidate = String;

    fn complete(
        &self,
        _line: &str,
        _pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        Ok((0, Vec::new()))
    }
}

/// Handles displaying hints for the `PlaceholderText` Helper. For our use case, we really only
/// need to display the hint when there is no text entered by the user.
impl Hinter for PlaceholderText {
    type Hint = String;

    fn hint(&self, _line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        match pos {
            0 => Some(self.placeholder_text.clone()),
            _ => None,
        }
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
