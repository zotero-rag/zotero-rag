use crossterm::event::{self, Event, KeyCode};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style, Stylize},
    widgets::{Block, Paragraph, Widget},
    DefaultTerminal, Frame,
};
use std::io;

#[derive(Debug, Default)]
pub struct App {
    user_query: String,
    exit: bool,
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
        const MARGIN: u16 = 4;
        let frame_area = frame.area();
        let x = frame_area.x + MARGIN;
        let y = frame_area.height - MARGIN;

        let area = Rect::new(x, y, frame_area.width - 2 * MARGIN, 3);
        frame.render_widget(self, area);
    }

    fn handle_events(&mut self) -> io::Result<()> {
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Enter => {
                    self.exit = true;
                }
                _ => {
                    self.user_query += &key.code.to_string();
                }
            }
        }

        Ok(())
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let input_block = Block::bordered();

        let cur_input = self.user_query.clone();
        let (input_display, style) = match cur_input.len() {
            0 => ("Ask a question", Style::default().fg(Color::Gray).dim()),
            _ => (cur_input.as_str(), Style::default()),
        };

        let input_text = Paragraph::new(input_display)
            .style(style)
            .block(input_block);

        input_text.render(area, buf);
    }
}
