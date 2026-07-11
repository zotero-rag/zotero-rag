//! Rendering for the TUI.
//!
//! The layout, top to bottom: the transcript (with a sidebar to its right on wide terminals),
//! the bordered query box, the suggestion list, and a one-line footer with mode/status hints.
//! Styling is deliberately restrained: primary content uses the default foreground, secondary
//! content (tool traces, timings, diagnostics) is dimmed, and a single accent color marks
//! interactive elements.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Margin, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use crate::tui::app::App;
use crate::tui::input::{InputMode, Keybindings};
use crate::tui::transcript::EntryKind;
use crate::utils::terminal::format_number;

/// Accent color for interactive elements (prompt, selection, user lines).
const ACCENT: Color = Color::Cyan;

/// Style for secondary content. The terminal's dim attribute (like the CLI's `DIM_TEXT`)
/// rather than a fixed gray, so it stays legible on both light and dark themes.
const SECONDARY: Style = Style::new().add_modifier(Modifier::DIM);

/// Maximum number of suggestion rows shown below the query box.
const MAX_SUGGESTION_ROWS: usize = 6;

/// Sidebar width in columns, including its border.
const SIDEBAR_WIDTH: u16 = 30;

/// Hide the sidebar on terminals narrower than this.
const MIN_WIDTH_FOR_SIDEBAR: u16 = 80;

/// Draw one frame of the TUI.
///
/// # Arguments
///
/// * `frame` - The ratatui frame to draw into.
/// * `app` - The application state; mutable so the transcript can clamp its scroll offset and
///   the app can remember the chat viewport height for keyboard paging.
pub(crate) fn draw(frame: &mut Frame<'_>, app: &mut App) {
    let suggestion_rows = app.suggestions.len().min(MAX_SUGGESTION_ROWS);
    let [main_area, input_area, suggest_area, footer_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(3),
        Constraint::Length(u16::try_from(suggestion_rows).unwrap_or(0)),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    let (chat_area, sidebar_area) = if frame.area().width >= MIN_WIDTH_FOR_SIDEBAR {
        let [chat, sidebar] =
            Layout::horizontal([Constraint::Min(1), Constraint::Length(SIDEBAR_WIDTH)])
                .areas(main_area);
        (chat, Some(sidebar))
    } else {
        (main_area, None)
    };

    draw_chat(frame, app, chat_area);
    if let Some(area) = sidebar_area {
        draw_sidebar(frame, app, area);
    }
    draw_input(frame, app, input_area);
    draw_suggestions(frame, app, suggest_area);
    draw_footer(frame, app, footer_area);
}

fn entry_style(kind: EntryKind) -> Style {
    match kind {
        EntryKind::User => Style::new().fg(ACCENT).bold(),
        EntryKind::Output => Style::new(),
        EntryKind::Info | EntryKind::Tool => SECONDARY,
        EntryKind::Error => Style::new().fg(Color::Red),
    }
}

fn draw_chat(frame: &mut Frame<'_>, app: &mut App, area: Rect) {
    let inner = area.inner(Margin::new(1, 0));
    app.chat_height = inner.height;

    let lines: Vec<Line<'_>> = app
        .transcript
        .visible_window(usize::from(inner.width), usize::from(inner.height))
        .into_iter()
        .map(|(kind, text)| Line::styled(text, entry_style(kind)))
        .collect();

    frame.render_widget(Paragraph::new(lines), inner);
}

/// Truncate `text` to at most `max` characters, ending with `...` when shortened.
fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }

    let kept: String = text.chars().take(max.saturating_sub(3)).collect();
    format!("{kept}...")
}

fn draw_sidebar(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let block = Block::new().borders(Borders::LEFT).border_style(SECONDARY);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let width = usize::from(inner.width).saturating_sub(1);
    let header = |text: &str| Line::styled(format!(" {text}"), Style::new().fg(ACCENT).bold());
    let item = |label: &str, value: String| {
        Line::from(vec![
            Span::styled(format!(" {label:<9}"), SECONDARY),
            Span::raw(truncate_chars(&value, width.saturating_sub(10))),
        ])
    };

    let mut lines: Vec<Line<'_>> = vec![header("session")];
    if let Some(model) = &app.session.generation_model {
        lines.push(item("model", model.clone()));
    }
    if let Some(model) = &app.session.embedding_model {
        lines.push(item("embed", model.clone()));
    }

    lines.push(Line::raw(""));
    lines.push(header("usage"));
    lines.push(item("tok in", format_number(app.sidebar.input_tokens)));
    lines.push(item("tok out", format_number(app.sidebar.output_tokens)));
    lines.push(item(
        "cost",
        format!("${:.2}", app.sidebar.session_cost_cents as f64 / 100.0),
    ));

    lines.push(Line::raw(""));
    lines.push(header("conversation"));
    let title = app
        .sidebar
        .conversation_title
        .clone()
        .unwrap_or_else(|| "(untitled)".to_string());
    lines.push(Line::styled(
        format!(" {}", truncate_chars(&title, width)),
        Style::new(),
    ));
    lines.push(item("messages", app.sidebar.message_count.to_string()));

    lines.push(Line::raw(""));
    lines.push(header(&format!(
        "documents ({})",
        app.sidebar.documents.len()
    )));
    if app.sidebar.documents.is_empty() {
        lines.push(Line::styled(" (none)", SECONDARY));
    }
    for doc in &app.sidebar.documents {
        lines.push(Line::raw(format!(
            " - {}",
            truncate_chars(doc, width.saturating_sub(3))
        )));
    }

    frame.render_widget(Paragraph::new(lines), inner);
}

fn draw_input(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let mut block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(SECONDARY);
    if app.awaiting_reply {
        block = block
            .border_style(Style::new().fg(Color::Yellow))
            .title(Span::styled(" reply ", Style::new().fg(Color::Yellow)));
    }

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let prompt = "> ";
    let available = usize::from(inner.width).saturating_sub(prompt.len()).max(1);
    let chars: Vec<char> = app.editor.text().chars().collect();
    let cursor = app.editor.cursor_chars();

    // Scroll the line horizontally so the cursor stays visible.
    let scroll = cursor.saturating_sub(available.saturating_sub(1));
    let visible: String = chars.iter().skip(scroll).take(available).collect();

    let content = if chars.is_empty() {
        let placeholder = if app.awaiting_reply {
            "answer the prompt above (Enter accepts the default)"
        } else {
            "Ask your library, or / for commands"
        };
        Line::from(vec![
            Span::styled(prompt, Style::new().fg(ACCENT).bold()),
            Span::styled(placeholder, SECONDARY),
        ])
    } else {
        Line::from(vec![
            Span::styled(prompt, Style::new().fg(ACCENT).bold()),
            Span::raw(visible),
        ])
    };
    frame.render_widget(Paragraph::new(content), inner);

    let cursor_x = inner.x
        + u16::try_from(prompt.len() + (cursor - scroll)).unwrap_or(inner.width.saturating_sub(1));
    frame.set_cursor_position(Position::new(
        cursor_x.min(inner.right().saturating_sub(1)),
        inner.y,
    ));
}

fn draw_suggestions(frame: &mut Frame<'_>, app: &App, area: Rect) {
    if area.height == 0 || app.suggestions.is_empty() {
        return;
    }

    let rows = usize::from(area.height);
    // Keep the selection inside the visible window.
    let start = app
        .selected_suggestion
        .saturating_sub(rows.saturating_sub(1));

    let lines: Vec<Line<'_>> = app
        .suggestions
        .iter()
        .enumerate()
        .skip(start)
        .take(rows)
        .map(|(i, suggestion)| {
            let selected = i == app.selected_suggestion;
            let marker = if selected { " > " } else { "   " };
            let label_style = if selected {
                Style::new().fg(ACCENT).bold()
            } else {
                Style::new()
            };

            let mut spans = vec![
                Span::styled(marker, Style::new().fg(ACCENT)),
                Span::styled(format!("{:<20}", suggestion.label), label_style),
            ];
            if !suggestion.description.is_empty() {
                spans.push(Span::styled(
                    format!("  {}", suggestion.description),
                    SECONDARY,
                ));
            }

            Line::from(spans)
        })
        .collect();

    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_footer(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let mut spans: Vec<Span<'_>> = vec![Span::raw(" ")];

    if let Some(notice) = &app.notice {
        spans.push(Span::styled(notice.clone(), Style::new().fg(Color::Yellow)));
    } else if app.awaiting_reply {
        spans.push(Span::styled(
            "a command is waiting for your reply",
            Style::new().fg(Color::Yellow),
        ));
    } else if app.busy > 0 {
        spans.push(Span::styled(
            format!("{} working...", app.spinner_char()),
            Style::new().fg(ACCENT),
        ));
    } else if app.editor.bindings() == Keybindings::Vi {
        let mode = match app.editor.mode() {
            InputMode::Insert => "-- INSERT --",
            InputMode::Normal => "-- NORMAL --",
        };
        spans.push(Span::styled(mode, SECONDARY));
    }

    if app.transcript.is_scrolled() {
        spans.push(Span::styled("  (scrolled; Ctrl-D for newest)", SECONDARY));
    }

    frame.render_widget(Paragraph::new(Line::from(spans)), area);

    let hints = Line::styled(
        "Tab complete | Ctrl-U/D scroll | Ctrl-C Ctrl-C quit ",
        SECONDARY,
    )
    .right_aligned();
    frame.render_widget(Paragraph::new(hints), area);
}

#[cfg(test)]
mod tests {
    use zqa_macros::test_eq;

    use super::truncate_chars;

    #[test]
    fn test_truncate_chars() {
        test_eq!(truncate_chars("short", 10), "short");
        test_eq!(truncate_chars("exactly-10", 10), "exactly-10");
        test_eq!(truncate_chars("this is too long", 10), "this is...");
    }
}
