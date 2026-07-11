//! A single-line editor for the TUI query box, with emacs and vi keybindings.
//!
//! This mirrors the editing behavior the CLI gets from readline: the active flavor is chosen
//! the same way (see [`crate::cli::readline::get_edit_mode`]). Emacs bindings cover the common
//! readline set (`C-a`, `C-e`, `C-b`, `C-f`, `C-k`, `C-u`, `C-w`, `C-d`, `C-t`, `M-b`, `M-f`,
//! `M-d`). Vi bindings support a practical subset: motions `h l 0 ^ $ w b e`, insert entries
//! `i a I A`, edits `x X s S D C`, and the `d`/`c` operators combined with `w b e 0 $` (plus
//! `dd`/`cc`).
//!
//! Keys the editor does not handle are reported as [`KeyOutcome::Ignored`] so the caller can
//! use them for app-level actions such as scrolling; notably, `C-u`/`C-d` on an empty line and
//! in vi normal mode fall through, matching vim's scroll bindings.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Which keybinding flavor the editor uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Keybindings {
    /// Readline-style emacs bindings; the editor is always in insert mode.
    Emacs,
    /// Vi bindings with insert and normal modes.
    Vi,
}

/// The editor's current input mode. Emacs bindings only ever use [`InputMode::Insert`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InputMode {
    /// Typed characters are inserted at the cursor.
    Insert,
    /// Vi command mode: characters are interpreted as motions and operators.
    Normal,
}

/// What the editor did with a key event.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KeyOutcome {
    /// The key was handled; the line or cursor may have changed.
    Edited,
    /// Enter was pressed; contains the submitted line. The editor is reset.
    Submitted(String),
    /// The key is not an editing key; the caller may act on it (e.g. scrolling).
    Ignored,
}

/// Character classes used for vi-style word motions.
fn vi_char_class(c: char) -> u8 {
    if c.is_whitespace() {
        0
    } else if c.is_alphanumeric() || c == '_' {
        1
    } else {
        2
    }
}

/// A single-line editor holding the query box's buffer, cursor, and mode.
pub(crate) struct LineEditor {
    bindings: Keybindings,
    mode: InputMode,
    buffer: String,
    /// Byte offset of the cursor; always on a `char` boundary.
    cursor: usize,
    /// A vi operator (`d` or `c`) waiting for its motion.
    pending_operator: Option<char>,
}

impl LineEditor {
    /// Create an empty editor using the given keybinding flavor.
    pub(crate) fn new(bindings: Keybindings) -> Self {
        Self {
            bindings,
            mode: InputMode::Insert,
            buffer: String::new(),
            cursor: 0,
            pending_operator: None,
        }
    }

    /// The current line contents.
    pub(crate) fn text(&self) -> &str {
        &self.buffer
    }

    /// Whether the line is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// The keybinding flavor in use.
    pub(crate) fn bindings(&self) -> Keybindings {
        self.bindings
    }

    /// The current input mode.
    pub(crate) fn mode(&self) -> InputMode {
        self.mode
    }

    /// The number of characters before the cursor (i.e. the cursor column).
    pub(crate) fn cursor_chars(&self) -> usize {
        self.buffer[..self.cursor].chars().count()
    }

    /// The cursor's byte offset into [`Self::text`].
    pub(crate) fn cursor_bytes(&self) -> usize {
        self.cursor
    }

    /// Replace the line contents, placing the cursor at the end in insert mode.
    pub(crate) fn set_text(&mut self, text: String) {
        self.cursor = text.len();
        self.buffer = text;
        self.mode = InputMode::Insert;
        self.pending_operator = None;
    }

    /// Clear the line, returning to insert mode.
    pub(crate) fn clear(&mut self) {
        self.set_text(String::new());
    }

    /// Handle one key event.
    ///
    /// # Arguments
    ///
    /// * `key` - The key event to process.
    ///
    /// # Returns
    ///
    /// A [`KeyOutcome`] describing what happened; see the variants for caller obligations.
    pub(crate) fn handle_key(&mut self, key: &KeyEvent) -> KeyOutcome {
        match self.mode {
            InputMode::Insert => self.handle_insert_key(key),
            InputMode::Normal => self.handle_normal_key(key),
        }
    }

    fn submit(&mut self) -> KeyOutcome {
        let line = std::mem::take(&mut self.buffer);
        self.cursor = 0;
        self.mode = InputMode::Insert;
        self.pending_operator = None;

        KeyOutcome::Submitted(line)
    }

    fn handle_insert_key(&mut self, key: &KeyEvent) -> KeyOutcome {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        if key.code == KeyCode::Enter {
            return self.submit();
        }

        if key.code == KeyCode::Esc && self.bindings == Keybindings::Vi {
            self.mode = InputMode::Normal;
            // Vim places the cursor on the previous character when leaving insert mode.
            self.move_left();
            self.clamp_normal_cursor();
            return KeyOutcome::Edited;
        }

        if ctrl {
            return self.handle_insert_ctrl_key(key.code);
        }

        if alt {
            // Word-wise bindings are emacs-only; vi insert mode has no meta bindings.
            if self.bindings != Keybindings::Emacs {
                return KeyOutcome::Ignored;
            }

            return match key.code {
                KeyCode::Char('b') => {
                    self.cursor = self.emacs_word_left();
                    KeyOutcome::Edited
                }
                KeyCode::Char('f') => {
                    self.cursor = self.emacs_word_right();
                    KeyOutcome::Edited
                }
                KeyCode::Char('d') => {
                    let end = self.emacs_word_right();
                    self.buffer.drain(self.cursor..end);
                    KeyOutcome::Edited
                }
                KeyCode::Backspace => {
                    let start = self.emacs_word_left();
                    self.buffer.drain(start..self.cursor);
                    self.cursor = start;
                    KeyOutcome::Edited
                }
                _ => KeyOutcome::Ignored,
            };
        }

        match key.code {
            KeyCode::Char(c) => {
                self.buffer.insert(self.cursor, c);
                self.cursor += c.len_utf8();
                KeyOutcome::Edited
            }
            KeyCode::Backspace => {
                self.delete_back();
                KeyOutcome::Edited
            }
            KeyCode::Delete => {
                self.delete_forward();
                KeyOutcome::Edited
            }
            KeyCode::Left => {
                self.move_left();
                KeyOutcome::Edited
            }
            KeyCode::Right => {
                self.move_right();
                KeyOutcome::Edited
            }
            KeyCode::Home => {
                self.cursor = 0;
                KeyOutcome::Edited
            }
            KeyCode::End => {
                self.cursor = self.buffer.len();
                KeyOutcome::Edited
            }
            _ => KeyOutcome::Ignored,
        }
    }

    fn handle_insert_ctrl_key(&mut self, code: KeyCode) -> KeyOutcome {
        // Bindings vim also supports in insert mode.
        match code {
            KeyCode::Char('h') => {
                self.delete_back();
                return KeyOutcome::Edited;
            }
            KeyCode::Char('w') => {
                let start = self.whitespace_word_left();
                self.buffer.drain(start..self.cursor);
                self.cursor = start;
                return KeyOutcome::Edited;
            }
            KeyCode::Char('u') => {
                // On an empty line, fall through so the caller can scroll the transcript.
                if self.buffer.is_empty() {
                    return KeyOutcome::Ignored;
                }

                self.buffer.drain(..self.cursor);
                self.cursor = 0;
                return KeyOutcome::Edited;
            }
            _ => {}
        }

        if self.bindings != Keybindings::Emacs {
            return KeyOutcome::Ignored;
        }

        match code {
            KeyCode::Char('a') => {
                self.cursor = 0;
                KeyOutcome::Edited
            }
            KeyCode::Char('e') => {
                self.cursor = self.buffer.len();
                KeyOutcome::Edited
            }
            KeyCode::Char('b') => {
                self.move_left();
                KeyOutcome::Edited
            }
            KeyCode::Char('f') => {
                self.move_right();
                KeyOutcome::Edited
            }
            KeyCode::Char('d') => {
                // On an empty line, fall through so the caller can scroll the transcript.
                if self.buffer.is_empty() {
                    return KeyOutcome::Ignored;
                }

                self.delete_forward();
                KeyOutcome::Edited
            }
            KeyCode::Char('k') => {
                self.buffer.truncate(self.cursor);
                KeyOutcome::Edited
            }
            KeyCode::Char('t') => {
                self.transpose_chars();
                KeyOutcome::Edited
            }
            _ => KeyOutcome::Ignored,
        }
    }

    fn handle_normal_key(&mut self, key: &KeyEvent) -> KeyOutcome {
        // Control chords (including C-u/C-d scrolling) are the caller's to handle.
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return KeyOutcome::Ignored;
        }

        match key.code {
            KeyCode::Enter => self.submit(),
            KeyCode::Esc => {
                self.pending_operator = None;
                KeyOutcome::Edited
            }
            KeyCode::Left | KeyCode::Backspace => {
                self.move_left();
                KeyOutcome::Edited
            }
            KeyCode::Right => {
                self.move_right();
                self.clamp_normal_cursor();
                KeyOutcome::Edited
            }
            KeyCode::Home => {
                self.cursor = 0;
                KeyOutcome::Edited
            }
            KeyCode::End => {
                self.cursor = self.buffer.len();
                self.clamp_normal_cursor();
                KeyOutcome::Edited
            }
            KeyCode::Delete => {
                self.delete_forward();
                self.clamp_normal_cursor();
                KeyOutcome::Edited
            }
            KeyCode::Char(c) => self.handle_normal_char(c),
            _ => KeyOutcome::Ignored,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn handle_normal_char(&mut self, c: char) -> KeyOutcome {
        if let Some(op) = self.pending_operator.take() {
            return self.apply_operator(op, c);
        }

        match c {
            'h' => {
                self.move_left();
                KeyOutcome::Edited
            }
            'l' | ' ' => {
                self.move_right();
                self.clamp_normal_cursor();
                KeyOutcome::Edited
            }
            '0' | '^' => {
                self.cursor = 0;
                KeyOutcome::Edited
            }
            '$' => {
                self.cursor = self.buffer.len();
                self.clamp_normal_cursor();
                KeyOutcome::Edited
            }
            'w' => {
                self.cursor = self.vi_next_word_start();
                self.clamp_normal_cursor();
                KeyOutcome::Edited
            }
            'b' => {
                self.cursor = self.vi_prev_word_start();
                KeyOutcome::Edited
            }
            'e' => {
                let end = self.vi_word_end_exclusive();
                self.cursor = self.prev_char_start(end);
                KeyOutcome::Edited
            }
            'i' => {
                self.mode = InputMode::Insert;
                KeyOutcome::Edited
            }
            'a' => {
                self.move_right();
                self.mode = InputMode::Insert;
                KeyOutcome::Edited
            }
            'I' => {
                self.cursor = 0;
                self.mode = InputMode::Insert;
                KeyOutcome::Edited
            }
            'A' => {
                self.cursor = self.buffer.len();
                self.mode = InputMode::Insert;
                KeyOutcome::Edited
            }
            'x' => {
                self.delete_forward();
                self.clamp_normal_cursor();
                KeyOutcome::Edited
            }
            'X' => {
                self.delete_back();
                KeyOutcome::Edited
            }
            's' => {
                self.delete_forward();
                self.mode = InputMode::Insert;
                KeyOutcome::Edited
            }
            'S' => {
                self.buffer.clear();
                self.cursor = 0;
                self.mode = InputMode::Insert;
                KeyOutcome::Edited
            }
            'D' => {
                self.buffer.truncate(self.cursor);
                self.clamp_normal_cursor();
                KeyOutcome::Edited
            }
            'C' => {
                self.buffer.truncate(self.cursor);
                self.mode = InputMode::Insert;
                KeyOutcome::Edited
            }
            'd' | 'c' => {
                self.pending_operator = Some(c);
                KeyOutcome::Edited
            }
            _ => KeyOutcome::Ignored,
        }
    }

    fn apply_operator(&mut self, op: char, motion: char) -> KeyOutcome {
        let range = if motion == op {
            // dd / cc operate on the whole line.
            Some((0, self.buffer.len()))
        } else {
            match motion {
                // `cw` behaves like `ce` in vim, so both operators use the word end.
                'w' if op == 'd' => Some((self.cursor, self.vi_next_word_start())),
                'w' | 'e' => Some((self.cursor, self.vi_word_end_exclusive())),
                'b' => Some((self.vi_prev_word_start(), self.cursor)),
                '0' | '^' => Some((0, self.cursor)),
                '$' => Some((self.cursor, self.buffer.len())),
                _ => None,
            }
        };

        // Invalid motions swallow the operator, like vim beeping.
        if let Some((start, end)) = range
            && start < end
        {
            self.buffer.drain(start..end);
            self.cursor = start;
        }

        if op == 'c' {
            self.mode = InputMode::Insert;
        } else {
            self.clamp_normal_cursor();
        }

        KeyOutcome::Edited
    }

    /// In normal mode the cursor sits *on* a character, so it may not rest past the last one.
    fn clamp_normal_cursor(&mut self) {
        if self.mode == InputMode::Normal && self.cursor >= self.buffer.len() {
            self.cursor = self.prev_char_start(self.buffer.len());
        }
    }

    /// Byte offset of the character before `from`, or 0 at the start.
    fn prev_char_start(&self, from: usize) -> usize {
        self.buffer[..from]
            .char_indices()
            .next_back()
            .map_or(0, |(idx, _)| idx)
    }

    /// Byte offset just past the character at `from`, or the buffer length at the end.
    fn next_char_end(&self, from: usize) -> usize {
        self.buffer[from..]
            .chars()
            .next()
            .map_or(self.buffer.len(), |c| from + c.len_utf8())
    }

    fn move_left(&mut self) {
        self.cursor = self.prev_char_start(self.cursor);
    }

    fn move_right(&mut self) {
        self.cursor = self.next_char_end(self.cursor);
    }

    fn delete_back(&mut self) {
        if self.cursor > 0 {
            let start = self.prev_char_start(self.cursor);
            self.buffer.drain(start..self.cursor);
            self.cursor = start;
        }
    }

    fn delete_forward(&mut self) {
        if self.cursor < self.buffer.len() {
            let end = self.next_char_end(self.cursor);
            self.buffer.drain(self.cursor..end);
        }
    }

    /// Emacs `C-t`: swap the characters around the cursor, advancing it.
    fn transpose_chars(&mut self) {
        if self.cursor == 0 || self.buffer.chars().count() < 2 {
            return;
        }

        // At the end of the line, transpose the last two characters instead.
        if self.cursor == self.buffer.len() {
            self.cursor = self.prev_char_start(self.cursor);
        }

        let prev = self.prev_char_start(self.cursor);
        let current: char = self.buffer[self.cursor..]
            .chars()
            .next()
            .unwrap_or_default();
        let end = self.next_char_end(self.cursor);

        let moved: String = self.buffer.drain(self.cursor..end).collect();
        self.buffer.insert_str(prev, &moved);
        self.cursor = prev + current.len_utf8();
        self.move_right();
    }

    /// Start of the previous whitespace-delimited word (readline `C-w`).
    fn whitespace_word_left(&self) -> usize {
        let mut pos = self.cursor;
        while pos > 0 && self.char_at(self.prev_char_start(pos)).is_whitespace() {
            pos = self.prev_char_start(pos);
        }
        while pos > 0 && !self.char_at(self.prev_char_start(pos)).is_whitespace() {
            pos = self.prev_char_start(pos);
        }

        pos
    }

    /// Start of the previous alphanumeric word (emacs `M-b`).
    fn emacs_word_left(&self) -> usize {
        let mut pos = self.cursor;
        while pos > 0 && !self.char_at(self.prev_char_start(pos)).is_alphanumeric() {
            pos = self.prev_char_start(pos);
        }
        while pos > 0 && self.char_at(self.prev_char_start(pos)).is_alphanumeric() {
            pos = self.prev_char_start(pos);
        }

        pos
    }

    /// End of the next alphanumeric word (emacs `M-f`).
    fn emacs_word_right(&self) -> usize {
        let mut pos = self.cursor;
        while pos < self.buffer.len() && !self.char_at(pos).is_alphanumeric() {
            pos = self.next_char_end(pos);
        }
        while pos < self.buffer.len() && self.char_at(pos).is_alphanumeric() {
            pos = self.next_char_end(pos);
        }

        pos
    }

    /// Start of the next vi word (`w`).
    fn vi_next_word_start(&self) -> usize {
        let mut pos = self.cursor;
        if pos >= self.buffer.len() {
            return pos;
        }

        let class = vi_char_class(self.char_at(pos));
        if class != 0 {
            while pos < self.buffer.len() && vi_char_class(self.char_at(pos)) == class {
                pos = self.next_char_end(pos);
            }
        }
        while pos < self.buffer.len() && self.char_at(pos).is_whitespace() {
            pos = self.next_char_end(pos);
        }

        pos
    }

    /// Start of the previous vi word (`b`).
    fn vi_prev_word_start(&self) -> usize {
        let mut pos = self.cursor;
        while pos > 0 && self.char_at(self.prev_char_start(pos)).is_whitespace() {
            pos = self.prev_char_start(pos);
        }
        if pos == 0 {
            return 0;
        }

        let class = vi_char_class(self.char_at(self.prev_char_start(pos)));
        while pos > 0 && vi_char_class(self.char_at(self.prev_char_start(pos))) == class {
            pos = self.prev_char_start(pos);
        }

        pos
    }

    /// One past the end of the current (or next) vi word (`e`).
    fn vi_word_end_exclusive(&self) -> usize {
        let mut pos = self.next_char_end(self.cursor);
        while pos < self.buffer.len() && self.char_at(pos).is_whitespace() {
            pos = self.next_char_end(pos);
        }
        if pos >= self.buffer.len() {
            return self.buffer.len();
        }

        let class = vi_char_class(self.char_at(pos));
        while pos < self.buffer.len() && vi_char_class(self.char_at(pos)) == class {
            pos = self.next_char_end(pos);
        }

        pos
    }

    /// The character starting at byte offset `pos`.
    ///
    /// # Panics
    ///
    /// * If `pos` is not a character boundary inside the buffer.
    fn char_at(&self, pos: usize) -> char {
        self.buffer[pos..].chars().next().expect("pos out of range")
    }
}

#[cfg(test)]
mod tests {
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use zqa_macros::test_eq;

    use super::{InputMode, KeyOutcome, Keybindings, LineEditor};

    fn press(editor: &mut LineEditor, code: KeyCode) -> KeyOutcome {
        editor.handle_key(&KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn press_ctrl(editor: &mut LineEditor, c: char) -> KeyOutcome {
        editor.handle_key(&KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }

    fn press_alt(editor: &mut LineEditor, c: char) -> KeyOutcome {
        editor.handle_key(&KeyEvent::new(KeyCode::Char(c), KeyModifiers::ALT))
    }

    fn type_str(editor: &mut LineEditor, text: &str) {
        for c in text.chars() {
            press(editor, KeyCode::Char(c));
        }
    }

    #[test]
    fn test_emacs_insert_and_submit() {
        let mut editor = LineEditor::new(Keybindings::Emacs);
        type_str(&mut editor, "hello world");

        test_eq!(editor.text(), "hello world");
        test_eq!(
            press(&mut editor, KeyCode::Enter),
            KeyOutcome::Submitted("hello world".to_string())
        );
        assert!(editor.is_empty());
    }

    #[test]
    fn test_emacs_line_motions_and_kills() {
        let mut editor = LineEditor::new(Keybindings::Emacs);
        type_str(&mut editor, "hello world");

        press_ctrl(&mut editor, 'a');
        test_eq!(editor.cursor_chars(), 0);

        press_ctrl(&mut editor, 'k');
        test_eq!(editor.text(), "");

        type_str(&mut editor, "hello world");
        press_ctrl(&mut editor, 'w');
        test_eq!(editor.text(), "hello ");

        press_ctrl(&mut editor, 'u');
        test_eq!(editor.text(), "");
    }

    #[test]
    fn test_emacs_word_motions() {
        let mut editor = LineEditor::new(Keybindings::Emacs);
        type_str(&mut editor, "alpha beta gamma");

        press_alt(&mut editor, 'b');
        test_eq!(editor.cursor_chars(), 11);

        press_alt(&mut editor, 'b');
        test_eq!(editor.cursor_chars(), 6);

        press_alt(&mut editor, 'f');
        test_eq!(editor.cursor_chars(), 10);

        press_alt(&mut editor, 'd');
        test_eq!(editor.text(), "alpha beta");
    }

    #[test]
    fn test_ctrl_u_and_ctrl_d_fall_through_on_empty_line() {
        let mut editor = LineEditor::new(Keybindings::Emacs);

        test_eq!(press_ctrl(&mut editor, 'u'), KeyOutcome::Ignored);
        test_eq!(press_ctrl(&mut editor, 'd'), KeyOutcome::Ignored);

        type_str(&mut editor, "hi");
        test_eq!(press_ctrl(&mut editor, 'u'), KeyOutcome::Edited);
        test_eq!(editor.text(), "");
    }

    #[test]
    fn test_emacs_transpose() {
        let mut editor = LineEditor::new(Keybindings::Emacs);
        type_str(&mut editor, "ab");

        press_ctrl(&mut editor, 't');
        test_eq!(editor.text(), "ba");
    }

    #[test]
    fn test_vi_mode_switching() {
        let mut editor = LineEditor::new(Keybindings::Vi);
        type_str(&mut editor, "hello");
        test_eq!(editor.mode(), InputMode::Insert);

        press(&mut editor, KeyCode::Esc);
        test_eq!(editor.mode(), InputMode::Normal);
        // The cursor moves onto the last character, vim-style.
        test_eq!(editor.cursor_chars(), 4);

        press(&mut editor, KeyCode::Char('i'));
        test_eq!(editor.mode(), InputMode::Insert);
    }

    #[test]
    fn test_vi_motions() {
        let mut editor = LineEditor::new(Keybindings::Vi);
        type_str(&mut editor, "alpha beta gamma");
        press(&mut editor, KeyCode::Esc);

        press(&mut editor, KeyCode::Char('0'));
        test_eq!(editor.cursor_chars(), 0);

        press(&mut editor, KeyCode::Char('w'));
        test_eq!(editor.cursor_chars(), 6);

        press(&mut editor, KeyCode::Char('e'));
        test_eq!(editor.cursor_chars(), 9);

        press(&mut editor, KeyCode::Char('$'));
        test_eq!(editor.cursor_chars(), 15);

        press(&mut editor, KeyCode::Char('b'));
        test_eq!(editor.cursor_chars(), 11);
    }

    #[test]
    fn test_vi_delete_word() {
        let mut editor = LineEditor::new(Keybindings::Vi);
        type_str(&mut editor, "alpha beta gamma");
        press(&mut editor, KeyCode::Esc);
        press(&mut editor, KeyCode::Char('0'));

        press(&mut editor, KeyCode::Char('d'));
        press(&mut editor, KeyCode::Char('w'));
        test_eq!(editor.text(), "beta gamma");
        test_eq!(editor.mode(), InputMode::Normal);
    }

    #[test]
    fn test_vi_change_word_enters_insert() {
        let mut editor = LineEditor::new(Keybindings::Vi);
        type_str(&mut editor, "alpha beta");
        press(&mut editor, KeyCode::Esc);
        press(&mut editor, KeyCode::Char('0'));

        press(&mut editor, KeyCode::Char('c'));
        press(&mut editor, KeyCode::Char('w'));
        // `cw` acts like `ce`: the trailing space survives.
        test_eq!(editor.text(), " beta");
        test_eq!(editor.mode(), InputMode::Insert);
    }

    #[test]
    fn test_vi_dd_clears_line() {
        let mut editor = LineEditor::new(Keybindings::Vi);
        type_str(&mut editor, "delete me");
        press(&mut editor, KeyCode::Esc);

        press(&mut editor, KeyCode::Char('d'));
        press(&mut editor, KeyCode::Char('d'));
        test_eq!(editor.text(), "");
    }

    #[test]
    fn test_vi_x_and_append() {
        let mut editor = LineEditor::new(Keybindings::Vi);
        type_str(&mut editor, "abc");
        press(&mut editor, KeyCode::Esc);

        press(&mut editor, KeyCode::Char('x'));
        test_eq!(editor.text(), "ab");

        press(&mut editor, KeyCode::Char('A'));
        type_str(&mut editor, "z");
        test_eq!(editor.text(), "abz");
    }

    #[test]
    fn test_vi_normal_mode_ignores_ctrl_chords() {
        let mut editor = LineEditor::new(Keybindings::Vi);
        type_str(&mut editor, "text");
        press(&mut editor, KeyCode::Esc);

        test_eq!(press_ctrl(&mut editor, 'u'), KeyOutcome::Ignored);
        test_eq!(press_ctrl(&mut editor, 'd'), KeyOutcome::Ignored);
        test_eq!(editor.text(), "text");
    }

    #[test]
    fn test_vi_enter_submits_from_normal_mode() {
        let mut editor = LineEditor::new(Keybindings::Vi);
        type_str(&mut editor, "query");
        press(&mut editor, KeyCode::Esc);

        test_eq!(
            press(&mut editor, KeyCode::Enter),
            KeyOutcome::Submitted("query".to_string())
        );
        test_eq!(editor.mode(), InputMode::Insert);
    }

    #[test]
    fn test_set_text_places_cursor_at_end() {
        let mut editor = LineEditor::new(Keybindings::Emacs);
        editor.set_text("/help".to_string());

        test_eq!(editor.text(), "/help");
        test_eq!(editor.cursor_chars(), 5);
    }

    #[test]
    fn test_unicode_editing() {
        let mut editor = LineEditor::new(Keybindings::Emacs);
        type_str(&mut editor, "caf\u{e9}");

        press(&mut editor, KeyCode::Backspace);
        test_eq!(editor.text(), "caf");
    }
}
