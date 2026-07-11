//! Character-oriented line editing for the terminal interface.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rustyline::EditMode;

/// Modal state used by vi editing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ViMode {
    Insert,
    Normal,
}

/// A focused, single-line editor.
pub(super) struct Editor {
    chars: Vec<char>,
    cursor: usize,
    edit_mode: EditMode,
    vi_mode: ViMode,
    history: Vec<String>,
    history_index: Option<usize>,
}

impl Editor {
    pub(super) fn new(edit_mode: EditMode, history: Vec<String>) -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
            edit_mode,
            vi_mode: ViMode::Insert,
            history,
            history_index: None,
        }
    }

    pub(super) fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub(super) fn cursor(&self) -> usize {
        self.cursor
    }

    pub(super) fn set_text(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }

    pub(super) fn take(&mut self) -> String {
        self.cursor = 0;
        self.history_index = None;
        self.vi_mode = ViMode::Insert;
        self.chars.drain(..).collect()
    }

    pub(super) fn remember(&mut self, text: &str) {
        if !text.trim().is_empty() {
            self.history.push(text.to_string());
        }
    }

    fn history(&mut self, older: bool) {
        if self.history.is_empty() {
            return;
        }
        let index = match (self.history_index, older) {
            (None, true) => self.history.len() - 1,
            (Some(0), true) => 0,
            (Some(index), true) => index - 1,
            (Some(index), false) if index + 1 < self.history.len() => index + 1,
            (_, false) => {
                self.history_index = None;
                self.set_text("");
                return;
            }
        };
        self.history_index = Some(index);
        let text = self.history[index].clone();
        self.set_text(&text);
    }

    fn previous_word(&mut self) {
        while self.cursor > 0 && self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
        while self.cursor > 0 && !self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
    }

    fn next_word(&mut self) {
        while self.cursor < self.chars.len() && !self.chars[self.cursor].is_whitespace() {
            self.cursor += 1;
        }
        while self.cursor < self.chars.len() && self.chars[self.cursor].is_whitespace() {
            self.cursor += 1;
        }
    }

    fn delete_previous_word(&mut self) {
        let end = self.cursor;
        self.previous_word();
        self.chars.drain(self.cursor..end);
    }

    fn insert(&mut self, character: char) {
        self.chars.insert(self.cursor, character);
        self.cursor += 1;
    }

    pub(super) fn handle_key(&mut self, key: KeyEvent) {
        if self.edit_mode == EditMode::Vi && self.vi_mode == ViMode::Normal {
            self.handle_vi_normal(key);
            return;
        }
        if self.edit_mode == EditMode::Vi && key.code == KeyCode::Esc {
            self.vi_mode = ViMode::Normal;
            self.cursor = self.cursor.saturating_sub(1);
            return;
        }
        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('a')) | (_, KeyCode::Home) => self.cursor = 0,
            (KeyModifiers::CONTROL, KeyCode::Char('e')) | (_, KeyCode::End) => {
                self.cursor = self.chars.len();
            }
            (KeyModifiers::CONTROL, KeyCode::Char('b')) | (_, KeyCode::Left) => {
                self.cursor = self.cursor.saturating_sub(1);
            }
            (KeyModifiers::CONTROL, KeyCode::Char('f')) | (_, KeyCode::Right) => {
                self.cursor = (self.cursor + 1).min(self.chars.len());
            }
            (KeyModifiers::ALT, KeyCode::Char('b')) => self.previous_word(),
            (KeyModifiers::ALT, KeyCode::Char('f')) => self.next_word(),
            (KeyModifiers::CONTROL, KeyCode::Char('k')) => self.chars.truncate(self.cursor),
            (KeyModifiers::CONTROL, KeyCode::Char('w')) => self.delete_previous_word(),
            (_, KeyCode::Backspace) if self.cursor > 0 => {
                self.cursor -= 1;
                self.chars.remove(self.cursor);
            }
            (_, KeyCode::Delete) if self.cursor < self.chars.len() => {
                self.chars.remove(self.cursor);
            }
            (_, KeyCode::Up) => self.history(true),
            (_, KeyCode::Down) => self.history(false),
            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(character)) => {
                self.insert(character);
            }
            _ => {}
        }
    }

    fn handle_vi_normal(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('i') => self.vi_mode = ViMode::Insert,
            KeyCode::Char('I') => {
                self.cursor = 0;
                self.vi_mode = ViMode::Insert;
            }
            KeyCode::Char('a') => {
                self.cursor = (self.cursor + 1).min(self.chars.len());
                self.vi_mode = ViMode::Insert;
            }
            KeyCode::Char('A') => {
                self.cursor = self.chars.len();
                self.vi_mode = ViMode::Insert;
            }
            KeyCode::Char('h') | KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Char('l') | KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.chars.len().saturating_sub(1));
            }
            KeyCode::Char('0') | KeyCode::Home => self.cursor = 0,
            KeyCode::Char('$') | KeyCode::End => self.cursor = self.chars.len().saturating_sub(1),
            KeyCode::Char('w') => self.next_word(),
            KeyCode::Char('b') => self.previous_word(),
            KeyCode::Char('x') | KeyCode::Delete if self.cursor < self.chars.len() => {
                self.chars.remove(self.cursor);
                self.cursor = self.cursor.min(self.chars.len().saturating_sub(1));
            }
            KeyCode::Char('X') | KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                self.chars.remove(self.cursor);
            }
            KeyCode::Char('D') => self.chars.truncate(self.cursor),
            KeyCode::Char('C') => {
                self.chars.truncate(self.cursor);
                self.vi_mode = ViMode::Insert;
            }
            KeyCode::Char('k') | KeyCode::Up => self.history(true),
            KeyCode::Char('j') | KeyCode::Down => self.history(false),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use rustyline::EditMode;

    use super::Editor;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn emacs_word_motions_and_deletion_use_characters() {
        let mut editor = Editor::new(EditMode::Emacs, Vec::new());
        editor.set_text("alpha beta");
        editor.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::ALT));
        assert_eq!(editor.cursor(), 6);
        editor.handle_key(KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL));
        assert_eq!(editor.text(), "beta");
    }

    #[test]
    fn vi_normal_mode_edits_line() {
        let mut editor = Editor::new(EditMode::Vi, Vec::new());
        editor.set_text("abc");
        editor.handle_key(key(KeyCode::Esc));
        editor.handle_key(key(KeyCode::Char('0')));
        editor.handle_key(key(KeyCode::Char('x')));
        assert_eq!(editor.text(), "bc");
    }

    #[test]
    fn history_moves_both_directions() {
        let mut editor = Editor::new(EditMode::Emacs, vec!["one".into(), "two".into()]);
        editor.handle_key(key(KeyCode::Up));
        assert_eq!(editor.text(), "two");
        editor.handle_key(key(KeyCode::Up));
        assert_eq!(editor.text(), "one");
        editor.handle_key(key(KeyCode::Down));
        assert_eq!(editor.text(), "two");
    }
}
