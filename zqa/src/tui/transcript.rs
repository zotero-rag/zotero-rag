//! The TUI's scrollable conversation transcript.
//!
//! Handler output arrives as plain text lines (some containing ANSI escapes from the CLI's
//! styling, which are stripped) and is stored as [`Entry`]s. Rendering wraps entries to the
//! viewport width on demand; the scroll position is tracked as a number of wrapped lines above
//! the bottom, so new output keeps the view pinned to the bottom unless the user has scrolled
//! up.

/// The kind of a transcript entry, which controls how it is styled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntryKind {
    /// A line the user submitted.
    User,
    /// Primary output: command results and model text.
    Output,
    /// Secondary output: timings, notices, and handler diagnostics.
    Info,
    /// A command error.
    Error,
    /// A tool invocation or timing trace.
    Tool,
}

/// A single logical line in the transcript.
#[derive(Debug)]
pub(crate) struct Entry {
    /// The kind of the entry, controlling its styling.
    pub(crate) kind: EntryKind,
    /// The line's text, with ANSI escapes stripped and tabs expanded.
    pub(crate) text: String,
}

/// The conversation transcript and its scroll state.
pub(crate) struct Transcript {
    entries: Vec<Entry>,
    /// Wrapped lines scrolled up from the bottom; 0 keeps the view pinned to new output.
    scroll_from_bottom: usize,
}

impl Transcript {
    /// Create an empty transcript.
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            scroll_from_bottom: 0,
        }
    }

    /// Append `text` as one or more entries of the given kind.
    ///
    /// ANSI escapes are stripped, tabs are expanded, and embedded newlines split the text into
    /// separate entries. User lines get a `> ` marker and tool traces a small indent, so that
    /// wrapped continuation text stays aligned with the styling.
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of the pushed entry.
    /// * `text` - The text to append.
    pub(crate) fn push(&mut self, kind: EntryKind, text: &str) {
        for line in strip_ansi(text).split('\n') {
            let line = expand_tabs(line);
            let text = match kind {
                EntryKind::User => format!("> {line}"),
                EntryKind::Tool => format!("  {line}"),
                _ => line,
            };

            self.entries.push(Entry { kind, text });
        }
    }

    /// Whether the most recent entry is a blank line (used to avoid doubled spacing).
    pub(crate) fn last_is_blank(&self) -> bool {
        self.entries.last().is_some_and(|e| e.text.is_empty())
    }

    /// Scroll up (towards older output) by `lines` wrapped lines.
    pub(crate) fn scroll_up(&mut self, lines: usize) {
        // Clamped against the content size at render time.
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(lines);
    }

    /// Scroll down (towards newer output) by `lines` wrapped lines.
    pub(crate) fn scroll_down(&mut self, lines: usize) {
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(lines);
    }

    /// Jump back to the newest output.
    pub(crate) fn scroll_to_bottom(&mut self) {
        self.scroll_from_bottom = 0;
    }

    /// Whether the view is scrolled away from the newest output.
    pub(crate) fn is_scrolled(&self) -> bool {
        self.scroll_from_bottom > 0
    }

    /// Wrap all entries to `width` and return the `height` lines visible at the current scroll
    /// position, clamping the scroll offset to the content size.
    ///
    /// Re-wrapping the whole transcript per frame is deliberate: it keeps resize and scroll
    /// logic trivial, and session transcripts are small enough that this is not a bottleneck.
    ///
    /// # Arguments
    ///
    /// * `width` - The viewport width in columns.
    /// * `height` - The viewport height in rows.
    ///
    /// # Returns
    ///
    /// The visible wrapped lines, oldest first, each tagged with its entry kind.
    pub(crate) fn visible_window(
        &mut self,
        width: usize,
        height: usize,
    ) -> Vec<(EntryKind, String)> {
        let wrapped: Vec<(EntryKind, String)> = self
            .entries
            .iter()
            .flat_map(|entry| {
                wrap_line(&entry.text, width)
                    .into_iter()
                    .map(|line| (entry.kind, line))
            })
            .collect();

        let max_scroll = wrapped.len().saturating_sub(height);
        self.scroll_from_bottom = self.scroll_from_bottom.min(max_scroll);

        let end = wrapped.len() - self.scroll_from_bottom;
        let start = end.saturating_sub(height);

        wrapped[start..end].to_vec()
    }
}

/// Remove ANSI CSI escape sequences (e.g. the CLI's color codes) from `text`.
pub(crate) fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }

        if chars.peek() == Some(&'[') {
            chars.next();
            // Skip parameter and intermediate bytes through the sequence's final byte.
            for c in chars.by_ref() {
                if matches!(c, '@'..='~') {
                    break;
                }
            }
        }
        // A bare escape that starts no CSI sequence is dropped.
    }

    out
}

/// Expand tabs to spaces using 8-column tab stops, like a terminal would.
fn expand_tabs(line: &str) -> String {
    const TAB_STOP: usize = 8;

    if !line.contains('\t') {
        return line.to_string();
    }

    let mut out = String::with_capacity(line.len());
    let mut col = 0;
    for c in line.chars() {
        if c == '\t' {
            let spaces = TAB_STOP - (col % TAB_STOP);
            out.extend(std::iter::repeat_n(' ', spaces));
            col += spaces;
        } else {
            out.push(c);
            col += 1;
        }
    }

    out
}

/// Greedily word-wrap `text` to `width` columns, hard-splitting words longer than a line.
///
/// # Arguments
///
/// * `text` - The line to wrap; must not contain newlines.
/// * `width` - The maximum line width in characters.
///
/// # Returns
///
/// The wrapped lines; always at least one (possibly empty) line.
fn wrap_line(text: &str, width: usize) -> Vec<String> {
    if width == 0 || text.chars().count() <= width {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_len = 0;
    let mut first_word = true;

    // `split(' ')` keeps empty items for runs of spaces, preserving indentation on rejoin.
    for word in text.split(' ') {
        let word_len = word.chars().count();
        let sep = usize::from(!std::mem::take(&mut first_word));

        if current_len + sep + word_len <= width {
            if sep == 1 {
                current.push(' ');
            }
            current.push_str(word);
            current_len += sep + word_len;
            continue;
        }

        if !current.is_empty() {
            lines.push(std::mem::take(&mut current));
        }

        // Hard-split a word that can never fit on one line.
        let mut chunk = String::new();
        let mut chunk_len = 0;
        for c in word.chars() {
            if chunk_len == width {
                lines.push(std::mem::take(&mut chunk));
                chunk_len = 0;
            }
            chunk.push(c);
            chunk_len += 1;
        }
        current = chunk;
        current_len = chunk_len;
    }

    lines.push(current);
    lines
}

#[cfg(test)]
mod tests {
    use zqa_macros::test_eq;

    use super::{EntryKind, Transcript, expand_tabs, strip_ansi, wrap_line};

    #[test]
    fn test_strip_ansi_removes_color_codes() {
        test_eq!(strip_ansi("\x1b[2mdim text\x1b[0m"), "dim text");
        test_eq!(strip_ansi("\x1b[33;1mwarn: \x1b[0mrest"), "warn: rest");
        test_eq!(strip_ansi("plain"), "plain");
    }

    #[test]
    fn test_expand_tabs_aligns_to_stops() {
        test_eq!(expand_tabs("/help\t\tShow"), "/help           Show");
        test_eq!(expand_tabs("no tabs"), "no tabs");
    }

    #[test]
    fn test_wrap_line_word_boundaries() {
        test_eq!(
            wrap_line("the quick brown fox", 10),
            vec!["the quick", "brown fox"]
        );
    }

    #[test]
    fn test_wrap_line_hard_splits_long_words() {
        test_eq!(wrap_line("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
    }

    #[test]
    fn test_wrap_line_preserves_short_and_empty_lines() {
        test_eq!(wrap_line("short", 80), vec!["short"]);
        test_eq!(wrap_line("", 80), vec![""]);
    }

    #[test]
    fn test_push_splits_lines_and_prefixes_user_entries() {
        let mut transcript = Transcript::new();
        transcript.push(EntryKind::User, "what is attention?");
        transcript.push(EntryKind::Output, "line one\nline two");

        let window = transcript.visible_window(80, 10);
        test_eq!(window.len(), 3);
        test_eq!(window[0], (EntryKind::User, "> what is attention?".into()));
        test_eq!(window[1], (EntryKind::Output, "line one".into()));
        test_eq!(window[2], (EntryKind::Output, "line two".into()));
    }

    #[test]
    fn test_scrolling_moves_and_clamps_window() {
        let mut transcript = Transcript::new();
        for i in 0..20 {
            transcript.push(EntryKind::Output, &format!("line {i}"));
        }

        // Pinned to the bottom by default.
        let window = transcript.visible_window(80, 5);
        test_eq!(window.last().unwrap().1, "line 19");

        transcript.scroll_up(5);
        let window = transcript.visible_window(80, 5);
        test_eq!(window.last().unwrap().1, "line 14");
        assert!(transcript.is_scrolled());

        // Scrolling past the top clamps to the oldest line.
        transcript.scroll_up(1000);
        let window = transcript.visible_window(80, 5);
        test_eq!(window.first().unwrap().1, "line 0");

        transcript.scroll_to_bottom();
        let window = transcript.visible_window(80, 5);
        test_eq!(window.last().unwrap().1, "line 19");
        assert!(!transcript.is_scrolled());
    }

    #[test]
    fn test_window_smaller_than_content() {
        let mut transcript = Transcript::new();
        transcript.push(EntryKind::Output, "only line");

        let window = transcript.visible_window(80, 5);
        test_eq!(window.len(), 1);
    }

    #[test]
    fn test_last_is_blank() {
        let mut transcript = Transcript::new();
        assert!(!transcript.last_is_blank());

        transcript.push(EntryKind::Output, "text");
        assert!(!transcript.last_is_blank());

        transcript.push(EntryKind::Output, "");
        assert!(transcript.last_is_blank());
    }
}
