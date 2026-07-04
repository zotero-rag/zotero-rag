//! A single-line editor for the TUI query box.
//!
//! Supports emacs-style bindings by default, plus a basic subset of vi motions when the user's
//! readline configuration selects vi mode. The mode is detected the same way as in the readline
//! CLI (see [`crate::cli::readline::get_edit_mode`]), so `.inputrc`/`.editrc` settings carry
//! over. Word motions treat any non-whitespace run as a word (vi "WORD" semantics).

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

/// The result of feeding a key event into the editor.
pub(crate) enum EditorAction {
    /// The key was handled (or ignored) without submitting the line.
    Continue,
    /// The user pressed Enter; the contained string is the submitted line.
    Submit(String),
}

/// The current input mode of the editor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EditorMode {
    Emacs,
    ViInsert,
    ViNormal,
}

/// A single-line editor holding the in-progress query, its cursor, and input history.
pub(crate) struct LineEditor {
    /// The current line. Stored as `char`s to keep cursor arithmetic simple.
    chars: Vec<char>,
    /// Cursor position as an index into `chars`; may equal `chars.len()`.
    cursor: usize,
    mode: EditorMode,
    /// Whether vi keybindings are enabled.
    vi: bool,
    /// A pending vi operator (`d` or `c`) awaiting a motion.
    pending_op: Option<char>,
    /// Session-local input history, oldest first.
    history: Vec<String>,
    /// Current position while navigating history; `None` when editing a fresh line.
    history_idx: Option<usize>,
    /// The in-progress line stashed away while navigating history.
    stash: Vec<char>,
}

impl LineEditor {
    /// Create an editor in emacs mode, or vi insert mode if `vi` is set.
    pub(crate) fn new(vi: bool) -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
            mode: if vi {
                EditorMode::ViInsert
            } else {
                EditorMode::Emacs
            },
            vi,
            pending_op: None,
            history: Vec::new(),
            history_idx: None,
            stash: Vec::new(),
        }
    }

    /// The current line as a string.
    pub(crate) fn line(&self) -> String {
        self.chars.iter().collect()
    }

    /// The cursor position, in characters.
    pub(crate) fn cursor(&self) -> usize {
        self.cursor
    }

    /// The current input mode.
    pub(crate) fn mode(&self) -> EditorMode {
        self.mode
    }

    /// Replace the line contents and move the cursor to the end.
    pub(crate) fn set_line(&mut self, line: &str) {
        self.chars = line.chars().collect();
        self.cursor = self.chars.len();
    }

    /// Feed a key event into the editor.
    ///
    /// # Arguments
    ///
    /// * `key` - The key event to process.
    ///
    /// # Returns
    ///
    /// An [`EditorAction`] indicating whether the line was submitted.
    pub(crate) fn handle_key(&mut self, key: KeyEvent) -> EditorAction {
        if key.kind == KeyEventKind::Release {
            return EditorAction::Continue;
        }

        match self.mode {
            EditorMode::Emacs | EditorMode::ViInsert => self.handle_insert_key(key),
            EditorMode::ViNormal => self.handle_normal_key(key),
        }
    }

    /// Handle a key in emacs or vi insert mode.
    fn handle_insert_key(&mut self, key: KeyEvent) -> EditorAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        match key.code {
            KeyCode::Enter => return self.submit(),
            KeyCode::Esc if self.vi => {
                self.mode = EditorMode::ViNormal;
                self.cursor = self.cursor.saturating_sub(1);
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.chars.len()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.chars.len(),
            KeyCode::Backspace => self.delete_before_cursor(),
            KeyCode::Delete => self.delete_at_cursor(),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            KeyCode::Char(c) if ctrl => match c {
                'a' => self.cursor = 0,
                'e' => self.cursor = self.chars.len(),
                'b' => self.cursor = self.cursor.saturating_sub(1),
                'f' => self.cursor = (self.cursor + 1).min(self.chars.len()),
                'h' => self.delete_before_cursor(),
                'k' => self.delete_range(self.cursor, self.chars.len()),
                'w' => self.delete_range(self.prev_word_start(), self.cursor),
                'p' => self.history_prev(),
                'n' => self.history_next(),
                _ => {}
            },
            KeyCode::Char(c) if alt => match c {
                'b' => self.cursor = self.prev_word_start(),
                'f' => self.cursor = self.next_word_start(),
                'd' => self.delete_range(self.cursor, self.next_word_start()),
                _ => {}
            },
            KeyCode::Char(c) => {
                self.chars.insert(self.cursor, c);
                self.cursor += 1;
            }
            _ => {}
        }

        EditorAction::Continue
    }

    /// Handle a key in vi normal mode.
    fn handle_normal_key(&mut self, key: KeyEvent) -> EditorAction {
        if let Some(op) = self.pending_op {
            self.pending_op = None;
            self.apply_operator(op, key.code);
            return EditorAction::Continue;
        }

        match key.code {
            KeyCode::Enter => return self.submit(),
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.max_normal_cursor()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.max_normal_cursor(),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            KeyCode::Char(c) => match c {
                'i' => self.mode = EditorMode::ViInsert,
                'I' => {
                    self.cursor = 0;
                    self.mode = EditorMode::ViInsert;
                }
                'a' => {
                    self.cursor = (self.cursor + 1).min(self.chars.len());
                    self.mode = EditorMode::ViInsert;
                }
                'A' => {
                    self.cursor = self.chars.len();
                    self.mode = EditorMode::ViInsert;
                }
                'h' => self.cursor = self.cursor.saturating_sub(1),
                'l' => self.cursor = (self.cursor + 1).min(self.max_normal_cursor()),
                '0' => self.cursor = 0,
                '^' => self.cursor = self.first_non_blank(),
                '$' => self.cursor = self.max_normal_cursor(),
                'w' => self.cursor = self.next_word_start().min(self.max_normal_cursor()),
                'b' => self.cursor = self.prev_word_start(),
                'e' => self.cursor = self.next_word_end(),
                'x' => self.delete_at_cursor(),
                'X' => self.delete_before_cursor(),
                'D' => self.delete_range(self.cursor, self.chars.len()),
                'C' => {
                    self.delete_range(self.cursor, self.chars.len());
                    self.mode = EditorMode::ViInsert;
                }
                's' => {
                    self.delete_at_cursor();
                    self.mode = EditorMode::ViInsert;
                }
                'S' => {
                    self.chars.clear();
                    self.cursor = 0;
                    self.mode = EditorMode::ViInsert;
                }
                'd' => self.pending_op = Some('d'),
                'c' => self.pending_op = Some('c'),
                'j' => self.history_next(),
                'k' => self.history_prev(),
                _ => {}
            },
            _ => {}
        }

        EditorAction::Continue
    }

    /// Apply a pending vi operator (`d` or `c`) to the motion given by `motion`.
    fn apply_operator(&mut self, op: char, motion: KeyCode) {
        let range = match motion {
            // `dd`/`cc` operate on the whole line.
            KeyCode::Char(m) if m == op => Some((0, self.chars.len())),
            KeyCode::Char('w') => Some((self.cursor, self.next_word_start())),
            KeyCode::Char('e') => Some((
                self.cursor,
                (self.next_word_end() + 1).min(self.chars.len()),
            )),
            KeyCode::Char('b') => Some((self.prev_word_start(), self.cursor)),
            KeyCode::Char('$') => Some((self.cursor, self.chars.len())),
            KeyCode::Char('0' | '^') => Some((0, self.cursor)),
            KeyCode::Char('h') | KeyCode::Left => {
                Some((self.cursor.saturating_sub(1), self.cursor))
            }
            KeyCode::Char('l') | KeyCode::Right => {
                Some((self.cursor, (self.cursor + 1).min(self.chars.len())))
            }
            _ => None,
        };

        if let Some((start, end)) = range {
            self.delete_range(start, end);
        }

        if op == 'c' {
            self.mode = EditorMode::ViInsert;
        }
    }

    /// Submit the current line, pushing it into history and clearing the editor.
    fn submit(&mut self) -> EditorAction {
        let line = self.line();
        let trimmed = line.trim();

        if !trimmed.is_empty() && self.history.last().map(String::as_str) != Some(trimmed) {
            self.history.push(trimmed.to_string());
        }

        self.chars.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.stash.clear();
        self.pending_op = None;
        if self.vi {
            self.mode = EditorMode::ViInsert;
        }

        EditorAction::Submit(line)
    }

    /// The largest cursor position allowed in vi normal mode (on the last character).
    fn max_normal_cursor(&self) -> usize {
        self.chars.len().saturating_sub(1)
    }

    /// Index of the first non-whitespace character, for the vi `^` motion.
    fn first_non_blank(&self) -> usize {
        self.chars
            .iter()
            .position(|c| !c.is_whitespace())
            .unwrap_or(0)
    }

    /// Start of the previous word (whitespace-delimited), scanning left from the cursor.
    fn prev_word_start(&self) -> usize {
        let mut i = self.cursor.min(self.chars.len());
        while i > 0 && self.chars[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !self.chars[i - 1].is_whitespace() {
            i -= 1;
        }
        i
    }

    /// Start of the next word (whitespace-delimited), scanning right from the cursor.
    fn next_word_start(&self) -> usize {
        let mut i = self.cursor;
        while i < self.chars.len() && !self.chars[i].is_whitespace() {
            i += 1;
        }
        while i < self.chars.len() && self.chars[i].is_whitespace() {
            i += 1;
        }
        i
    }

    /// End (last character) of the next word, for the vi `e` motion.
    fn next_word_end(&self) -> usize {
        let mut i = self.cursor + 1;
        while i < self.chars.len() && self.chars[i].is_whitespace() {
            i += 1;
        }
        while i < self.chars.len() && !self.chars[i].is_whitespace() {
            i += 1;
        }
        i.saturating_sub(1).min(self.max_normal_cursor())
    }

    /// Delete the character before the cursor, if any.
    fn delete_before_cursor(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    /// Delete the character under the cursor, if any.
    fn delete_at_cursor(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    /// Delete the character range `[start, end)` and place the cursor at `start`.
    fn delete_range(&mut self, start: usize, end: usize) {
        let end = end.min(self.chars.len());
        if start < end {
            self.chars.drain(start..end);
            self.cursor = start;
        }
    }

    /// Move to the previous history entry, stashing the in-progress line first.
    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }

        match self.history_idx {
            None => {
                self.stash = std::mem::take(&mut self.chars);
                self.history_idx = Some(self.history.len() - 1);
            }
            Some(0) => return,
            Some(i) => self.history_idx = Some(i - 1),
        }

        // Invariant: `history_idx` was just set to a valid index.
        self.set_line(&self.history[self.history_idx.unwrap()].clone());
    }

    /// Move to the next history entry, or restore the stashed line past the newest entry.
    fn history_next(&mut self) {
        match self.history_idx {
            None => {}
            Some(i) if i + 1 < self.history.len() => {
                self.history_idx = Some(i + 1);
                self.set_line(&self.history[i + 1].clone());
            }
            Some(_) => {
                self.history_idx = None;
                self.chars = std::mem::take(&mut self.stash);
                self.cursor = self.chars.len();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use super::{EditorAction, EditorMode, LineEditor};

    fn press(editor: &mut LineEditor, code: KeyCode) -> EditorAction {
        editor.handle_key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn press_ctrl(editor: &mut LineEditor, c: char) -> EditorAction {
        editor.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }

    fn press_alt(editor: &mut LineEditor, c: char) -> EditorAction {
        editor.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::ALT))
    }

    fn type_str(editor: &mut LineEditor, text: &str) {
        for c in text.chars() {
            press(editor, KeyCode::Char(c));
        }
    }

    #[test]
    fn test_insert_and_submit() {
        let mut editor = LineEditor::new(false);
        type_str(&mut editor, "hello world");

        match press(&mut editor, KeyCode::Enter) {
            EditorAction::Submit(line) => assert_eq!(line, "hello world"),
            EditorAction::Continue => panic!("expected a submit"),
        }
        assert!(editor.line().is_empty());
    }

    #[test]
    fn test_emacs_kill_to_end() {
        let mut editor = LineEditor::new(false);
        type_str(&mut editor, "hello world");
        press_ctrl(&mut editor, 'a');
        for _ in 0..5 {
            press_ctrl(&mut editor, 'f');
        }
        press_ctrl(&mut editor, 'k');

        assert_eq!(editor.line(), "hello");
    }

    #[test]
    fn test_emacs_kill_prev_word() {
        let mut editor = LineEditor::new(false);
        type_str(&mut editor, "hello world");
        press_ctrl(&mut editor, 'w');

        assert_eq!(editor.line(), "hello ");
        assert_eq!(editor.cursor(), 6);
    }

    #[test]
    fn test_emacs_word_motions() {
        let mut editor = LineEditor::new(false);
        type_str(&mut editor, "one two three");
        press_ctrl(&mut editor, 'a');
        press_alt(&mut editor, 'f');
        assert_eq!(editor.cursor(), 4);

        press_alt(&mut editor, 'd');
        assert_eq!(editor.line(), "one three");
    }

    #[test]
    fn test_vi_dd_clears_line() {
        let mut editor = LineEditor::new(true);
        type_str(&mut editor, "delete me");
        press(&mut editor, KeyCode::Esc);
        assert_eq!(editor.mode(), EditorMode::ViNormal);

        press(&mut editor, KeyCode::Char('d'));
        press(&mut editor, KeyCode::Char('d'));
        assert!(editor.line().is_empty());
    }

    #[test]
    fn test_vi_word_motion_and_dw() {
        let mut editor = LineEditor::new(true);
        type_str(&mut editor, "one two three");
        press(&mut editor, KeyCode::Esc);
        press(&mut editor, KeyCode::Char('0'));
        press(&mut editor, KeyCode::Char('w'));
        assert_eq!(editor.cursor(), 4);

        press(&mut editor, KeyCode::Char('d'));
        press(&mut editor, KeyCode::Char('w'));
        assert_eq!(editor.line(), "one three");
    }

    #[test]
    fn test_vi_change_to_end_enters_insert() {
        let mut editor = LineEditor::new(true);
        type_str(&mut editor, "keep drop");
        press(&mut editor, KeyCode::Esc);
        press(&mut editor, KeyCode::Char('0'));
        for _ in 0..5 {
            press(&mut editor, KeyCode::Char('l'));
        }
        press(&mut editor, KeyCode::Char('C'));

        assert_eq!(editor.line(), "keep ");
        assert_eq!(editor.mode(), EditorMode::ViInsert);
    }

    #[test]
    fn test_vi_append_at_end() {
        let mut editor = LineEditor::new(true);
        type_str(&mut editor, "abc");
        press(&mut editor, KeyCode::Esc);
        press(&mut editor, KeyCode::Char('A'));
        type_str(&mut editor, "d");

        assert_eq!(editor.line(), "abcd");
    }

    #[test]
    fn test_history_navigation() {
        let mut editor = LineEditor::new(false);
        type_str(&mut editor, "first");
        press(&mut editor, KeyCode::Enter);
        type_str(&mut editor, "second");
        press(&mut editor, KeyCode::Enter);

        type_str(&mut editor, "in progress");
        press(&mut editor, KeyCode::Up);
        assert_eq!(editor.line(), "second");
        press(&mut editor, KeyCode::Up);
        assert_eq!(editor.line(), "first");
        press(&mut editor, KeyCode::Down);
        assert_eq!(editor.line(), "second");
        press(&mut editor, KeyCode::Down);
        assert_eq!(editor.line(), "in progress");
    }
}
