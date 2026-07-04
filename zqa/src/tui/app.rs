//! TUI application state and rendering.

use std::collections::VecDeque;

use ratatui::Frame;
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph};

use crate::cli::placeholder::SLASH_COMMANDS;
use crate::tui::editor::{EditorAction, EditorMode, LineEditor};
use crate::tui::writer::{SidebarSnapshot, WorkerEvent};

/// Maximum number of transcript lines kept in memory; the oldest lines are dropped past this.
const MAX_TRANSCRIPT_LINES: usize = 5000;

/// Maximum number of command suggestions shown below the input box.
const MAX_SUGGESTIONS: usize = 5;

/// Number of lines scrolled per mouse wheel notch.
const WHEEL_SCROLL_LINES: usize = 3;

/// Minimum terminal width at which the sidebar is shown.
const MIN_WIDTH_FOR_SIDEBAR: u16 = 70;

/// Width of the sidebar, including its left border.
const SIDEBAR_WIDTH: u16 = 28;

const SPINNER_FRAMES: &[&str] = &["|", "/", "-", "\\"];

const PLACEHOLDER: &str = "Ask a question, or type / for commands";

/// The provenance of a transcript line, which determines its styling.
#[derive(Clone, Copy)]
enum LineKind {
    /// A line the user submitted.
    User,
    /// A model answer, synced from the chat history.
    Answer,
    /// Handler output written to stdout.
    Output,
    /// Handler output written to stderr (timings, warnings).
    Stderr,
    /// A message from the TUI itself.
    Notice,
    /// A command error.
    Error,
}

struct TranscriptLine {
    kind: LineKind,
    text: String,
}

/// The full state of the TUI: transcript, input editor, suggestions, and sidebar data.
pub(crate) struct App {
    transcript: VecDeque<TranscriptLine>,
    /// Incomplete (not yet newline-terminated) stdout output from the current command.
    partial_out: String,
    /// Incomplete stderr output from the current command.
    partial_err: String,
    editor: LineEditor,
    suggestions: Vec<&'static str>,
    suggestion_idx: usize,
    sidebar: SidebarSnapshot,
    /// How many lines the transcript is scrolled up from the bottom; 0 sticks to the bottom.
    scroll: usize,
    /// Scroll ceiling, computed during the last render.
    max_scroll: usize,
    /// Transcript viewport height from the last render, used for half-page scrolling.
    view_height: usize,
    /// Whether a command is currently being executed by the worker.
    running: bool,
    spinner_frame: usize,
    /// Set when the worker reports that the session should end (e.g., after `/quit`).
    pub(crate) should_quit: bool,
    /// Set when the user force-quits (Ctrl-C while a command is running).
    pub(crate) force_quit: bool,
}

impl App {
    /// Create the application state.
    ///
    /// # Arguments
    ///
    /// * `vi` - Whether the input editor should use vi keybindings.
    pub(crate) fn new(vi: bool) -> Self {
        let mut app = Self {
            transcript: VecDeque::new(),
            partial_out: String::new(),
            partial_err: String::new(),
            editor: LineEditor::new(vi),
            suggestions: Vec::new(),
            suggestion_idx: 0,
            sidebar: SidebarSnapshot::default(),
            scroll: 0,
            max_scroll: 0,
            view_height: 0,
            running: false,
            spinner_frame: 0,
            should_quit: false,
            force_quit: false,
        };

        app.push_lines(
            LineKind::Notice,
            "Welcome to zqa. Type a question, or /help for options. Press Ctrl-C to exit.",
        );
        app.push_lines(LineKind::Notice, "");
        app
    }

    /// Handle a key event.
    ///
    /// # Arguments
    ///
    /// * `key` - The key event to process.
    ///
    /// # Returns
    ///
    /// A command line to dispatch to the worker, if the key submitted one.
    pub(crate) fn on_key(&mut self, key: KeyEvent) -> Option<String> {
        if key.kind == KeyEventKind::Release {
            return None;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    if self.running {
                        self.force_quit = true;
                    } else {
                        self.push_lines(LineKind::Notice, "Saving conversation and exiting...");
                        self.running = true;
                        return Some("/quit".to_string());
                    }
                    return None;
                }
                KeyCode::Char('u') => {
                    self.scroll_up(self.half_page());
                    return None;
                }
                KeyCode::Char('d') => {
                    self.scroll_down(self.half_page());
                    return None;
                }
                _ => {}
            }
        }

        // While the suggestion list is visible, Tab accepts and Up/Down navigate it; otherwise
        // those keys go to the editor (history navigation).
        if self.suggestions.is_empty() {
            if key.code == KeyCode::Tab {
                return None;
            }
        } else {
            match key.code {
                KeyCode::Tab => {
                    self.accept_suggestion();
                    return None;
                }
                KeyCode::Down => {
                    self.suggestion_idx = (self.suggestion_idx + 1) % self.suggestions.len();
                    return None;
                }
                KeyCode::Up => {
                    self.suggestion_idx =
                        (self.suggestion_idx + self.suggestions.len() - 1) % self.suggestions.len();
                    return None;
                }
                _ => {}
            }
        }

        match self.editor.handle_key(key) {
            EditorAction::Submit(line) => {
                self.refresh_suggestions();
                let line = line.trim().to_string();
                if line.is_empty() {
                    return None;
                }

                if self.running {
                    self.push_lines(
                        LineKind::Notice,
                        "A command is already running; wait for it to finish.",
                    );
                    self.editor.set_line(&line);
                    return None;
                }

                self.push_lines(LineKind::User, &line);
                self.running = true;
                self.scroll = 0;
                Some(line)
            }
            EditorAction::Continue => {
                self.refresh_suggestions();
                None
            }
        }
    }

    /// Handle a mouse event (wheel scrolling in the transcript).
    pub(crate) fn on_mouse(&mut self, event: MouseEvent) {
        match event.kind {
            MouseEventKind::ScrollUp => self.scroll_up(WHEEL_SCROLL_LINES),
            MouseEventKind::ScrollDown => self.scroll_down(WHEEL_SCROLL_LINES),
            _ => {}
        }
    }

    /// Handle an event from the command worker.
    pub(crate) fn on_worker_event(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::Stdout(bytes) => self.append_stream(false, &bytes),
            WorkerEvent::Stderr(bytes) => self.append_stream(true, &bytes),
            WorkerEvent::HistoryItem { is_user, text } => {
                let kind = if is_user {
                    LineKind::User
                } else {
                    LineKind::Answer
                };
                self.push_lines(kind, &text);
                self.push_lines(LineKind::Answer, "");
            }
            WorkerEvent::Sidebar(snapshot) => self.sidebar = snapshot,
            WorkerEvent::Done {
                should_continue,
                error,
            } => {
                self.flush_streams();
                if let Some(e) = error {
                    self.push_lines(LineKind::Error, &format!("Error: {e}"));
                }
                self.running = false;
                if !should_continue {
                    self.should_quit = true;
                }
            }
        }
    }

    /// Advance the spinner; called on every UI tick.
    pub(crate) fn tick(&mut self) {
        self.spinner_frame = self.spinner_frame.wrapping_add(1);
    }

    /// Append handler output bytes, splitting completed lines into the transcript.
    fn append_stream(&mut self, is_stderr: bool, bytes: &[u8]) {
        let kind = if is_stderr {
            LineKind::Stderr
        } else {
            LineKind::Output
        };

        // Take the buffer out so completed lines can be pushed while draining it.
        let mut buffer = std::mem::take(if is_stderr {
            &mut self.partial_err
        } else {
            &mut self.partial_out
        });
        buffer.push_str(&String::from_utf8_lossy(bytes));

        while let Some(newline_pos) = buffer.find('\n') {
            let line: String = buffer.drain(..=newline_pos).collect();
            self.push_lines(kind, line.trim_end_matches('\n'));
        }

        *(if is_stderr {
            &mut self.partial_err
        } else {
            &mut self.partial_out
        }) = buffer;
    }

    /// Push any incomplete stream output into the transcript; called when a command finishes.
    fn flush_streams(&mut self) {
        let partial_out = std::mem::take(&mut self.partial_out);
        if !partial_out.is_empty() {
            self.push_lines(LineKind::Output, &partial_out);
        }

        let partial_err = std::mem::take(&mut self.partial_err);
        if !partial_err.is_empty() {
            self.push_lines(LineKind::Stderr, &partial_err);
        }
    }

    /// Push text into the transcript, splitting on newlines and sanitizing each line.
    fn push_lines(&mut self, kind: LineKind, text: &str) {
        for line in text.split('\n') {
            self.transcript.push_back(TranscriptLine {
                kind,
                text: sanitize(line),
            });
        }

        while self.transcript.len() > MAX_TRANSCRIPT_LINES {
            self.transcript.pop_front();
        }
    }

    /// Recompute the suggestion list from the current input line.
    fn refresh_suggestions(&mut self) {
        let line = self.editor.line();
        if line.starts_with('/') {
            self.suggestions = SLASH_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(&line) && **cmd != line)
                .copied()
                .collect();
        } else {
            self.suggestions.clear();
        }

        self.suggestion_idx = self
            .suggestion_idx
            .min(self.suggestions.len().saturating_sub(1));
    }

    /// Replace the input line with the selected suggestion.
    fn accept_suggestion(&mut self) {
        if let Some(suggestion) = self.suggestions.get(self.suggestion_idx) {
            self.editor.set_line(&format!("{suggestion} "));
            self.refresh_suggestions();
        }
    }

    /// Half the height of the transcript viewport, for Ctrl-U/Ctrl-D scrolling.
    fn half_page(&self) -> usize {
        (self.view_height / 2).max(1)
    }

    fn scroll_up(&mut self, lines: usize) {
        self.scroll = (self.scroll + lines).min(self.max_scroll);
    }

    fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    /// Render the whole UI into the given frame.
    pub(crate) fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let (main, side) = if area.width >= MIN_WIDTH_FOR_SIDEBAR {
            let chunks =
                Layout::horizontal([Constraint::Min(30), Constraint::Length(SIDEBAR_WIDTH)])
                    .split(area);
            (chunks[0], Some(chunks[1]))
        } else {
            (area, None)
        };

        let suggestion_count = self.suggestions.len().min(MAX_SUGGESTIONS) as u16;
        let chunks = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(3),
            Constraint::Length(suggestion_count),
            Constraint::Length(1),
        ])
        .split(main);

        self.render_transcript(frame, chunks[0]);
        self.render_input(frame, chunks[1]);
        self.render_suggestions(frame, chunks[2]);
        self.render_status(frame, chunks[3]);
        if let Some(side) = side {
            self.render_sidebar(frame, side);
        }
    }

    /// Render the borderless, scrollable chat transcript.
    fn render_transcript(&mut self, frame: &mut Frame, area: Rect) {
        if area.width < 3 || area.height == 0 {
            return;
        }

        // Leave a one-column gutter on each side so text does not touch the edges.
        let width = (area.width - 2) as usize;
        let mut lines: Vec<Line> = Vec::new();
        for entry in &self.transcript {
            let (style, text) = match entry.kind {
                LineKind::User => (
                    Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    format!("> {}", entry.text),
                ),
                LineKind::Answer | LineKind::Output => (Style::new(), entry.text.clone()),
                LineKind::Stderr => (Style::new().add_modifier(Modifier::DIM), entry.text.clone()),
                LineKind::Notice => (
                    Style::new().add_modifier(Modifier::DIM | Modifier::ITALIC),
                    entry.text.clone(),
                ),
                LineKind::Error => (Style::new().fg(Color::Red), entry.text.clone()),
            };

            for wrapped in wrap_line(&text, width) {
                lines.push(Line::styled(wrapped, style));
            }
        }

        let view_height = area.height as usize;
        self.view_height = view_height;
        self.max_scroll = lines.len().saturating_sub(view_height);
        self.scroll = self.scroll.min(self.max_scroll);

        let start = self.max_scroll - self.scroll;
        let visible: Vec<Line> = lines.into_iter().skip(start).take(view_height).collect();

        let inner = Rect {
            x: area.x + 1,
            y: area.y,
            width: area.width - 2,
            height: area.height,
        };
        frame.render_widget(Paragraph::new(visible), inner);
    }

    /// Render the bordered query input box, including the cursor.
    fn render_input(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .padding(Padding::horizontal(1));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.width == 0 || inner.height == 0 {
            return;
        }

        let width = inner.width as usize;
        let line = self.editor.line();

        if line.is_empty() {
            frame.render_widget(
                Paragraph::new(PLACEHOLDER).style(Style::new().add_modifier(Modifier::DIM)),
                inner,
            );
            frame.set_cursor_position((inner.x, inner.y));
            return;
        }

        // Scroll the line horizontally so the cursor stays visible.
        let cursor = self.editor.cursor();
        let offset = cursor.saturating_sub(width.saturating_sub(1));
        let visible: String = line.chars().skip(offset).take(width).collect();

        frame.render_widget(Paragraph::new(visible), inner);
        frame.set_cursor_position((inner.x + (cursor - offset) as u16, inner.y));
    }

    /// Render the command suggestion list below the input box.
    fn render_suggestions(&self, frame: &mut Frame, area: Rect) {
        if area.height == 0 {
            return;
        }

        let lines: Vec<Line> = self
            .suggestions
            .iter()
            .take(MAX_SUGGESTIONS)
            .enumerate()
            .map(|(i, suggestion)| {
                let style = if i == self.suggestion_idx {
                    Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                } else {
                    Style::new().add_modifier(Modifier::DIM)
                };
                Line::styled(format!("  {suggestion}"), style)
            })
            .collect();

        frame.render_widget(Paragraph::new(lines), area);
    }

    /// Render the one-line status bar: editor mode, activity, and key hints.
    fn render_status(&self, frame: &mut Frame, area: Rect) {
        if area.height == 0 {
            return;
        }

        let mut spans: Vec<Span> = Vec::new();

        let mode_label = match self.editor.mode() {
            EditorMode::Emacs => None,
            EditorMode::ViInsert => Some("INSERT"),
            EditorMode::ViNormal => Some("NORMAL"),
        };
        if let Some(label) = mode_label {
            spans.push(Span::styled(
                format!(" {label} "),
                Style::new().add_modifier(Modifier::REVERSED),
            ));
            spans.push(Span::raw(" "));
        }

        if self.running {
            let spinner = SPINNER_FRAMES[self.spinner_frame % SPINNER_FRAMES.len()];
            spans.push(Span::styled(
                format!("{spinner} working... "),
                Style::new().fg(Color::Yellow),
            ));
        } else if self.scroll > 0 {
            spans.push(Span::styled(
                format!("[scrolled up {} lines] ", self.scroll),
                Style::new().add_modifier(Modifier::DIM),
            ));
        }

        spans.push(Span::styled(
            "Enter send | Tab complete | ^U/^D or wheel scroll | ^C quit",
            Style::new().add_modifier(Modifier::DIM),
        ));

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Render the session sidebar: model, cost, message count, title, and imported documents.
    fn render_sidebar(&self, frame: &mut Frame, area: Rect) {
        let block = Block::new()
            .borders(Borders::LEFT)
            .border_style(Style::new().add_modifier(Modifier::DIM))
            .padding(Padding::horizontal(1));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.width == 0 {
            return;
        }

        let width = inner.width as usize;
        let dim = Style::new().add_modifier(Modifier::DIM);
        let bold = Style::new().add_modifier(Modifier::BOLD);

        let mut lines: Vec<Line> = vec![Line::styled("Session", bold)];
        if let Some(model) = &self.sidebar.model {
            for wrapped in wrap_line(model, width) {
                lines.push(Line::styled(wrapped, dim));
            }
        }
        lines.push(Line::raw(format!(
            "Cost     ${:.4}",
            self.sidebar.session_cost_cents as f64 / 100.0
        )));
        lines.push(Line::raw(format!(
            "Messages {}",
            self.sidebar.message_count
        )));

        if let Some(title) = &self.sidebar.title {
            lines.push(Line::raw(""));
            for wrapped in wrap_line(title, width) {
                lines.push(Line::styled(wrapped, dim));
            }
        }

        lines.push(Line::raw(""));
        lines.push(Line::styled(
            format!("Documents ({})", self.sidebar.documents.len()),
            bold,
        ));
        if self.sidebar.documents.is_empty() {
            lines.push(Line::styled("none imported", dim));
        } else {
            for doc in &self.sidebar.documents {
                lines.push(Line::styled(format!("- {}", truncate(doc, width)), dim));
            }
        }

        frame.render_widget(Paragraph::new(lines), inner);
    }
}

/// Strip ANSI CSI escape sequences, expand tabs, and drop carriage returns.
///
/// Handlers emit color codes meant for a real terminal (see [`crate::utils::terminal`]); the
/// transcript applies its own styling instead.
///
/// # Arguments
///
/// * `text` - The raw line, without a trailing newline.
///
/// # Returns
///
/// The cleaned line.
fn sanitize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();

    while let Some(c) = chars.next() {
        match c {
            '\x1b' => {
                if chars.next() == Some('[') {
                    // Consume the CSI sequence through its final byte (0x40..=0x7e).
                    for csi in chars.by_ref() {
                        if matches!(csi, '\x40'..='\x7e') {
                            break;
                        }
                    }
                }
            }
            '\t' => out.push_str("    "),
            '\r' => {}
            _ => out.push(c),
        }
    }

    out
}

/// Wrap a single line of text to the given display width, breaking at spaces when possible.
///
/// # Arguments
///
/// * `text` - The line to wrap; must not contain newlines.
/// * `width` - The maximum number of characters per output line.
///
/// # Returns
///
/// The wrapped lines. Always contains at least one element.
fn wrap_line(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= width {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + width).min(chars.len());
        let split = if end < chars.len() {
            // Break after the last space that fits, if there is one.
            match chars[start..end].iter().rposition(|c| *c == ' ') {
                Some(pos) if pos > 0 => start + pos + 1,
                _ => end,
            }
        } else {
            end
        };

        lines.push(chars[start..split].iter().collect());
        start = split;
    }

    lines
}

/// Truncate a string to `width` characters, appending "..." if it was shortened.
fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }

    let keep = width.saturating_sub(3);
    let mut out: String = text.chars().take(keep).collect();
    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::{sanitize, truncate, wrap_line};

    #[test]
    fn test_sanitize_strips_ansi_codes() {
        assert_eq!(sanitize("\x1b[2mdim text\x1b[0m"), "dim text");
        assert_eq!(sanitize("\x1b[1;31mbold red\x1b[0m rest"), "bold red rest");
    }

    #[test]
    fn test_sanitize_expands_tabs_and_drops_cr() {
        assert_eq!(sanitize("\tIndented\r"), "    Indented");
    }

    #[test]
    fn test_wrap_line_short_text_is_unchanged() {
        assert_eq!(wrap_line("hello", 10), vec!["hello"]);
    }

    #[test]
    fn test_wrap_line_breaks_at_spaces() {
        let wrapped = wrap_line("one two three four", 9);
        assert_eq!(wrapped, vec!["one two ", "three ", "four"]);
    }

    #[test]
    fn test_wrap_line_hard_splits_long_words() {
        let wrapped = wrap_line("abcdefghij", 4);
        assert_eq!(wrapped, vec!["abcd", "efgh", "ij"]);
    }

    #[test]
    fn test_truncate_appends_ellipsis() {
        assert_eq!(truncate("a_very_long_filename.pdf", 10), "a_very_...");
        assert_eq!(truncate("short.pdf", 10), "short.pdf");
    }
}
