//! TUI state, suggestions, scrolling, and rendering.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::cli::commands::SLASH_COMMANDS;

use super::editor::Editor;

const SIDEBAR_MIN_WIDTH: u16 = 96;

/// Session values displayed in the sidebar.
#[derive(Default)]
pub(super) struct SessionStats {
    pub(super) input_tokens: u64,
    pub(super) output_tokens: u64,
    pub(super) session_cost: u64,
    pub(super) message_count: usize,
    pub(super) title: Option<String>,
    pub(super) documents: Vec<String>,
}

/// State of command submission and synchronous prompts.
#[derive(Default)]
pub(super) struct CommandFlow {
    pub(super) busy: bool,
    pub(super) prompt_pending: bool,
    pub(super) quit_after_completion: bool,
}

/// Mutable state rendered by the terminal loop.
pub(super) struct App {
    pub(super) editor: Editor,
    pub(super) transcript: String,
    pub(super) status: String,
    pub(super) flow: CommandFlow,
    pub(super) stats: SessionStats,
    pub(super) submitted_history: Vec<String>,
    pub(super) scroll: usize,
    pub(super) follow_bottom: bool,
    suggestion_index: usize,
    transcript_height: usize,
    transcript_width: usize,
}

impl App {
    pub(super) fn new(editor: Editor) -> Self {
        Self {
            editor,
            transcript: "zqa TUI ready. Type /help for commands.\n".into(),
            status: "Enter: submit | Tab: complete | Ctrl-C: quit".into(),
            flow: CommandFlow::default(),
            stats: SessionStats::default(),
            submitted_history: Vec::new(),
            scroll: 0,
            follow_bottom: true,
            suggestion_index: 0,
            transcript_height: 1,
            transcript_width: 1,
        }
    }

    pub(super) fn suggestions(&self) -> Vec<&'static str> {
        let text = self.editor.text();
        if !text.starts_with('/') {
            return Vec::new();
        }
        SLASH_COMMANDS
            .iter()
            .copied()
            .filter(|command| command.starts_with(&text))
            .collect()
    }

    pub(super) fn move_suggestion(&mut self, down: bool) {
        let length = self.suggestions().len();
        if length == 0 {
            return;
        }
        self.suggestion_index = if down {
            (self.suggestion_index + 1) % length
        } else {
            (self.suggestion_index + length - 1) % length
        };
    }

    pub(super) fn accept_suggestion(&mut self) {
        let suggestions = self.suggestions();
        if let Some(suggestion) = suggestions.get(
            self.suggestion_index
                .min(suggestions.len().saturating_sub(1)),
        ) {
            self.editor.set_text(suggestion);
        }
    }

    pub(super) fn reset_suggestion(&mut self) {
        self.suggestion_index = 0;
    }

    pub(super) fn append(&mut self, text: &str) {
        self.transcript.push_str(&strip_ansi(text));
        if self.follow_bottom {
            self.scroll = self.max_scroll();
        }
    }

    pub(super) fn replace_transcript(&mut self, text: &str) {
        self.transcript = strip_ansi(text);
        self.follow_bottom = true;
        self.scroll = self.max_scroll();
    }

    pub(super) fn scroll_half_page(&mut self, down: bool) {
        let amount = (self.transcript_height / 2).max(1);
        if down {
            self.scroll = (self.scroll + amount).min(self.max_scroll());
        } else {
            self.scroll = self.scroll.saturating_sub(amount);
        }
        self.follow_bottom = self.scroll == self.max_scroll();
    }

    pub(super) fn max_scroll(&self) -> usize {
        wrapped_line_count(&self.transcript, self.transcript_width)
            .saturating_sub(self.transcript_height)
    }

    pub(super) fn set_transcript_area(&mut self, area: Rect) {
        self.transcript_height = usize::from(area.height.max(1));
        self.transcript_width = usize::from(area.width.max(1));
        self.scroll = if self.follow_bottom {
            self.max_scroll()
        } else {
            self.scroll.min(self.max_scroll())
        };
    }

    pub(super) fn transcript_contains(column: u16, row: u16, area: Rect) -> bool {
        column >= area.x && column < area.right() && row >= area.y && row < area.bottom()
    }
}

pub(super) fn render(frame: &mut Frame<'_>, app: &mut App, model: &str) -> Rect {
    let area = frame.area();
    let suggestions = app.suggestions();
    let suggestion_height = u16::try_from(suggestions.len().min(3)).unwrap_or(3);
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(2),
            Constraint::Length(suggestion_height),
            Constraint::Length(1),
        ])
        .split(area);
    let body = if area.width >= SIDEBAR_MIN_WIDTH {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(40), Constraint::Length(28)])
            .split(vertical[0])
    } else {
        Layout::default()
            .constraints([Constraint::Percentage(100)])
            .split(vertical[0])
    };
    let transcript_area = body[0];
    app.set_transcript_area(transcript_area);
    frame.render_widget(
        Paragraph::new(app.transcript.as_str())
            .wrap(Wrap { trim: false })
            .scroll((u16::try_from(app.scroll).unwrap_or(u16::MAX), 0)),
        transcript_area,
    );

    if body.len() > 1 {
        let title = app.stats.title.as_deref().unwrap_or("New conversation");
        let mut documents = app.stats.documents.clone();
        documents.sort_unstable();
        let document_text = if documents.is_empty() {
            "  (none)".into()
        } else {
            documents
                .iter()
                .map(|name| format!("  {name}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let sidebar = format!(
            "Session\n\nInput tokens:  {}\nOutput tokens: {}\nCost:          ${:.2}\nMessages:      {}\n\nModel\n  {model}\n\nTitle\n  {title}\n\nDocuments\n{document_text}",
            app.stats.input_tokens,
            app.stats.output_tokens,
            app.stats.session_cost as f64 / 100.0,
            app.stats.message_count,
        );
        frame.render_widget(
            Paragraph::new(sidebar)
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::LEFT)),
            body[1],
        );
    }

    render_input(
        frame,
        app,
        &suggestions,
        vertical[1],
        vertical[2],
        vertical[3],
    );
    transcript_area
}

fn render_input(
    frame: &mut Frame<'_>,
    app: &App,
    suggestions: &[&'static str],
    query_area: Rect,
    suggestions_area: Rect,
    status_area: Rect,
) {
    let query_style = if app.flow.prompt_pending {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };
    let query_text = app.editor.text();
    let query_prefix = query_text
        .chars()
        .take(app.editor.cursor())
        .collect::<String>();
    let cursor_width = Line::from(query_prefix.as_str()).width();
    let input_width = usize::from(query_area.width.max(1));
    let horizontal_offset = cursor_width.saturating_sub(input_width.saturating_sub(1));
    frame.render_widget(
        Paragraph::new(query_text)
            .style(query_style)
            .scroll((0, u16::try_from(horizontal_offset).unwrap_or(u16::MAX)))
            .block(Block::default().borders(Borders::TOP)),
        query_area,
    );
    let cursor_column = u16::try_from(cursor_width.saturating_sub(horizontal_offset))
        .unwrap_or(u16::MAX)
        .min(query_area.width.saturating_sub(1));
    frame.set_cursor_position((query_area.x + cursor_column, query_area.y + 1));

    let suggestion_start = app.suggestion_index.saturating_sub(2);
    for (index, suggestion) in suggestions
        .iter()
        .enumerate()
        .skip(suggestion_start)
        .take(3)
    {
        let style = if index == app.suggestion_index {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(*suggestion, style))),
            Rect::new(
                suggestions_area.x,
                suggestions_area.y + u16::try_from(index - suggestion_start).unwrap_or(0),
                suggestions_area.width,
                1,
            ),
        );
    }
    frame.render_widget(
        Paragraph::new(app.status.as_str()).style(Style::default().fg(Color::DarkGray)),
        status_area,
    );
}

fn wrapped_line_count(text: &str, width: usize) -> usize {
    text.lines()
        .map(|line| line.chars().count().max(1).div_ceil(width.max(1)))
        .sum::<usize>()
        .max(1)
}

fn strip_ansi(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut escape = false;
    let mut control_sequence = false;
    for character in text.chars() {
        if !escape && character == '\x1b' {
            escape = true;
        } else if escape && !control_sequence && character == '[' {
            control_sequence = true;
        } else if escape && (!control_sequence || ('@'..='~').contains(&character)) {
            escape = false;
            control_sequence = false;
        } else if !escape {
            output.push(character);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use ratatui::{Terminal, backend::TestBackend, layout::Rect};
    use rustyline::EditMode;

    use super::{App, strip_ansi, wrapped_line_count};
    use crate::tui::editor::Editor;

    #[test]
    fn suggestions_share_slash_command_prefixes() {
        let mut app = App::new(Editor::new(EditMode::Emacs, Vec::new()));
        app.editor.set_text("/doc");
        assert_eq!(
            app.suggestions(),
            vec!["/docs clear", "/docs list", "/docs remove ", "/doctor"]
        );
        app.move_suggestion(true);
        app.accept_suggestion();
        assert_eq!(app.editor.text(), "/docs list");
    }

    #[test]
    fn scrolling_clamps_to_wrapped_content() {
        let mut app = App::new(Editor::new(EditMode::Emacs, Vec::new()));
        app.transcript = "1234567890\nabcdefghij\n".into();
        app.set_transcript_area(Rect::new(0, 0, 5, 2));
        app.scroll_half_page(true);
        assert!(app.scroll <= app.max_scroll());
        assert_eq!(wrapped_line_count(&app.transcript, 5), 4);
    }

    #[test]
    fn wide_render_includes_sidebar() {
        let backend = TestBackend::new(110, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Editor::new(EditMode::Emacs, Vec::new()));
        terminal
            .draw(|frame| {
                super::render(frame, &mut app, "test-model");
            })
            .unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(rendered.contains("Input tokens:"));
        assert!(rendered.contains("test-model"));
    }

    #[test]
    fn ansi_sequences_are_removed() {
        assert_eq!(strip_ansi("a\x1b[31mred\x1b[0mz"), "aredz");
    }
}
