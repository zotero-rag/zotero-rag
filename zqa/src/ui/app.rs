use crossterm::event::{self, Event, KeyCode};
use ratatui::{
    DefaultTerminal, Frame,
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::Line,
    widgets::{Block, Paragraph, Widget},
};
use std::io;

use crate::ui::commands;

const OUTPUT_LINES: usize = 10;
const MARGIN: usize = 3;

#[derive(Debug)]
pub struct App {
    // User inputs
    pub user_query: String,

    // Output display
    pub output_lines: Vec<String>,
    pub line_styles: Vec<Style>,
    pub scroll_idx: usize, // How many lines from the bottom have we scrolled?

    // Internal state
    pub exit: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            user_query: String::new(),
            output_lines: vec![String::from("Welcome to Zotero QA!"), String::new()],
            line_styles: vec![Style::default(), Style::default()],
            scroll_idx: 0,
            exit: false,
        }
    }
}

impl App {
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn draw(&self, frame: &mut Frame) {
        let frame_area = frame.area();

        // 2 for the output box borders
        // 1 for the gap
        // 3 for the input box
        // 2 for the input box borders
        let height = (OUTPUT_LINES + MARGIN + 8) as u16;

        let x = frame_area.x + (MARGIN as u16);
        let y = frame_area.height - height + (MARGIN as u16);

        let area = Rect::new(x, y, frame_area.width - (2 * MARGIN as u16), height);
        frame.render_widget(self, area);
    }

    fn handle_events(&mut self) -> io::Result<()> {
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Enter => self.process_commands(),
                KeyCode::Backspace => {
                    self.user_query.pop();
                }
                _ => {
                    if key.code.as_char().is_some() {
                        self.user_query += &key.code.to_string();
                    }
                }
            }
        }

        Ok(())
    }

    fn process_commands(&mut self) {
        match self.user_query.as_str() {
            "/process" => commands::handle_process_command(self),
            "/quit" => commands::exit(self),
            "/exit" => commands::exit(self),
            _ => (),
        }
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let output_block = Block::bordered();
        let input_block = Block::bordered();

        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length((OUTPUT_LINES + 2) as u16), // Output box
                Constraint::Length(1),                         // Gap
                Constraint::Length(3),                         // Input box
            ])
            .split(area);

        let cur_input = self.user_query.clone();
        let (input_display, style) = match cur_input.len() {
            0 => ("Ask a question", Style::default().fg(Color::Gray).dim()),
            _ => (cur_input.as_str(), Style::default()),
        };

        let input_text = Paragraph::new(input_display)
            .style(style)
            .block(input_block);

        // Grab the 10 lines before `self.scroll_idx`
        let first_line_idx_raw: i16 =
            self.output_lines.len() as i16 - self.scroll_idx as i16 - OUTPUT_LINES as i16;
        let first_line_idx = std::cmp::max(0, first_line_idx_raw) as usize;
        let last_line_idx = std::cmp::min(self.output_lines.len(), first_line_idx + OUTPUT_LINES);

        let output_lines = &self.output_lines[first_line_idx..last_line_idx];
        let output_styles = &self.line_styles[first_line_idx..last_line_idx];

        let output_line_structs = std::iter::zip(output_lines, output_styles)
            .map(|(line, style)| Line::from(vec![line.as_str().into()]).style(*style))
            .collect::<Vec<_>>();

        let output_paragraph = Paragraph::new(output_line_structs).block(output_block);

        input_text.render(layout[2], buf);
        output_paragraph.render(layout[0], buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use temp_env::with_vars;

    #[test]
    fn default_app_state_is_sane() {
        let app = App::default();
        assert_eq!(app.user_query, "");
        assert_eq!(app.output_lines.len(), 2);
        assert_eq!(app.line_styles.len(), 2);
        assert!(!app.exit);
        assert_eq!(app.scroll_idx, 0);
    }

    #[test]
    fn process_commands_sets_exit_on_quit() {
        let mut app = App {
            user_query: "/quit".into(),
            ..Default::default()
        };
        app.process_commands();
        assert!(app.exit);
    }

    #[test]
    #[serial]
    fn process_command_clears_input_and_maybe_warns() {
        // Ensure we use the toy Zotero library under assets on CI
        with_vars([("CI", Some("true"))], || {
            let mut app = App {
                user_query: "/process".into(),
                ..Default::default()
            };
            let before_lines = app.output_lines.len();
            let before_styles = app.line_styles.len();

            app.process_commands();

            // Input should always be cleared
            assert_eq!(app.user_query, "");
            // Styles vector should always match lines vector length
            assert_eq!(app.output_lines.len(), app.line_styles.len());
            // Either we warned (grew by 1) or nothing was added
            assert!(
                app.output_lines.len() == before_lines
                    || app.output_lines.len() == before_lines + 1
            );
            assert!(
                app.line_styles.len() == before_styles
                    || app.line_styles.len() == before_styles + 1
            );
        });
    }

    #[test]
    fn render_does_not_panic() {
        // Render into a small buffer to ensure the widget path works
        let app = App::default();
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);
        // Should not panic
        Widget::render(&app, area, &mut buf);
        // Buffer area should remain unchanged
        assert_eq!(buf.area(), &area);
    }
}
