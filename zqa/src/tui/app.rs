//! The TUI application: state, event handling, and the command runner.
//!
//! [`run_tui`] is the entry point. It spawns the rendering/input loop on a dedicated thread and
//! runs command handlers on the calling async task, so handlers behave exactly as they do under
//! the CLI. The two sides communicate over channels (see [`crate::tui::io`]).

use std::io;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseEvent, MouseEventKind,
};
use ratatui::crossterm::execute;
use rustyline::EditMode;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};

use crate::cli::app::{dispatch_command, pending_batches_notice};
use crate::cli::errors::CLIError;
use crate::cli::readline::get_edit_mode;
use crate::common::{Context, State};
use crate::config::Config;
use crate::store::lance::LanceZoteroStore;
use crate::tui::input::{KeyOutcome, Keybindings, LineEditor};
use crate::tui::io::{ChannelReader, ChannelWriter, OutputLane, SidebarSnapshot, UiMessage};
use crate::tui::suggest::{Suggestion, suggestions_for};
use crate::tui::transcript::{EntryKind, Transcript};
use crate::tui::ui;

/// How long a first Ctrl-C keeps quitting "armed" before it expires.
const CTRL_C_QUIT_WINDOW: Duration = Duration::from_secs(2);

/// How many wrapped lines a mouse wheel tick scrolls.
const MOUSE_SCROLL_LINES: usize = 3;

/// Actions the UI thread sends to the command runner.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum UserAction {
    /// The user submitted a command or query.
    Submit(String),
    /// The user asked to quit; the runner saves the conversation and exits.
    Quit,
}

/// Static session facts shown in the sidebar.
pub(crate) struct SessionInfo {
    /// The configured generation model's name.
    pub(crate) generation_model: Option<String>,
    /// The configured embedding model's name.
    pub(crate) embedding_model: Option<String>,
}

/// The full state of the TUI, owned by the UI thread.
pub(crate) struct App {
    /// The conversation transcript.
    pub(crate) transcript: Transcript,
    /// The query box's line editor.
    pub(crate) editor: LineEditor,
    /// Suggestions for the current input line.
    pub(crate) suggestions: Vec<Suggestion>,
    /// Index of the highlighted suggestion.
    pub(crate) selected_suggestion: usize,
    /// The latest sidebar snapshot from the runner.
    pub(crate) sidebar: SidebarSnapshot,
    /// Static session facts shown in the sidebar.
    pub(crate) session: SessionInfo,
    /// Number of submitted commands that have not finished yet.
    pub(crate) busy: usize,
    /// Whether a handler is blocked waiting for a reply.
    pub(crate) awaiting_reply: bool,
    /// A one-line footer notice (e.g. the Ctrl-C quit hint).
    pub(crate) notice: Option<String>,
    /// Height of the chat viewport at the last draw, for Ctrl-U/Ctrl-D paging.
    pub(crate) chat_height: u16,
    /// When the first Ctrl-C armed quitting, if it did.
    ctrl_c_armed_at: Option<Instant>,
    /// Suggestions stay hidden after Esc until the input line changes.
    suggestions_suppressed: bool,
    /// The input line the suggestions were last computed for.
    last_suggest_line: String,
    /// In-session input history, oldest first.
    history: Vec<String>,
    /// Position while navigating history; `None` when not navigating.
    history_index: Option<usize>,
    /// The line being typed before history navigation started.
    draft: String,
    /// Reference point for the busy spinner's animation.
    spinner_epoch: Instant,
    should_quit: bool,
    action_tx: UnboundedSender<UserAction>,
    reply_tx: Sender<String>,
}

impl App {
    /// Create the initial application state.
    ///
    /// # Arguments
    ///
    /// * `session` - Static session facts for the sidebar.
    /// * `keybindings` - The editor keybinding flavor.
    /// * `action_tx` - Channel for sending [`UserAction`]s to the command runner.
    /// * `reply_tx` - Channel for answering handler input requests.
    pub(crate) fn new(
        session: SessionInfo,
        keybindings: Keybindings,
        action_tx: UnboundedSender<UserAction>,
        reply_tx: Sender<String>,
    ) -> Self {
        Self {
            transcript: Transcript::new(),
            editor: LineEditor::new(keybindings),
            suggestions: Vec::new(),
            selected_suggestion: 0,
            sidebar: SidebarSnapshot::default(),
            session,
            busy: 0,
            awaiting_reply: false,
            notice: None,
            chat_height: 0,
            ctrl_c_armed_at: None,
            suggestions_suppressed: false,
            last_suggest_line: String::new(),
            history: Vec::new(),
            history_index: None,
            draft: String::new(),
            spinner_epoch: Instant::now(),
            should_quit: false,
            action_tx,
            reply_tx,
        }
    }

    /// Whether the UI loop should exit.
    pub(crate) fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// The current busy-spinner frame.
    pub(crate) fn spinner_char(&self) -> char {
        const FRAMES: [char; 4] = ['|', '/', '-', '\\'];
        let elapsed = self.spinner_epoch.elapsed().as_millis() / 120;

        #[allow(clippy::cast_possible_truncation)]
        FRAMES[(elapsed % 4) as usize]
    }

    /// Process a message from the command runner.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to process.
    ///
    /// # Returns
    ///
    /// `true` if the runner asked the UI to shut down.
    pub(crate) fn on_message(&mut self, message: UiMessage) -> bool {
        match message {
            UiMessage::Out(text) | UiMessage::Stream(text) => {
                self.transcript.push(EntryKind::Output, &text);
            }
            UiMessage::Info(text) => self.transcript.push(EntryKind::Info, &text),
            UiMessage::Error(text) => self.transcript.push(EntryKind::Error, &text),
            UiMessage::ToolTrace(text) => self.transcript.push(EntryKind::Tool, &text),
            UiMessage::InputRequested => self.awaiting_reply = true,
            UiMessage::CommandDone(snapshot) => {
                self.busy = self.busy.saturating_sub(1);
                self.sidebar = snapshot;
                if !self.transcript.last_is_blank() {
                    self.transcript.push(EntryKind::Output, "");
                }
            }
            UiMessage::Shutdown => return true,
        }

        false
    }

    /// Process a key event.
    pub(crate) fn on_key(&mut self, key: &KeyEvent) {
        // Ctrl-C clears the input; pressed again on an empty line, it quits.
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.on_ctrl_c();
            return;
        }

        // Any other key disarms a pending Ctrl-C quit.
        self.ctrl_c_armed_at = None;
        self.notice = None;

        if !self.suggestions.is_empty() {
            match key.code {
                KeyCode::Tab => {
                    self.accept_selected_suggestion();
                    return;
                }
                KeyCode::Down => {
                    self.selected_suggestion =
                        (self.selected_suggestion + 1) % self.suggestions.len();
                    return;
                }
                KeyCode::Up => {
                    self.selected_suggestion = self
                        .selected_suggestion
                        .checked_sub(1)
                        .unwrap_or(self.suggestions.len() - 1);
                    return;
                }
                KeyCode::Esc => {
                    // Dismiss the menu, then still let the editor see Esc (vi mode switch).
                    self.suggestions.clear();
                    self.suggestions_suppressed = true;
                }
                _ => {}
            }
        }

        match self.editor.handle_key(key) {
            KeyOutcome::Edited => {
                self.history_index = None;
                self.refresh_suggestions();
            }
            KeyOutcome::Submitted(line) => self.submit_line(line),
            KeyOutcome::Ignored => self.on_app_key(key),
        }
    }

    /// Process a mouse event (wheel scrolling in the transcript).
    pub(crate) fn on_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => self.transcript.scroll_up(MOUSE_SCROLL_LINES),
            MouseEventKind::ScrollDown => self.transcript.scroll_down(MOUSE_SCROLL_LINES),
            _ => {}
        }
    }

    /// Periodic maintenance: expire a stale Ctrl-C quit arm.
    pub(crate) fn tick(&mut self) {
        if self
            .ctrl_c_armed_at
            .is_some_and(|at| at.elapsed() > CTRL_C_QUIT_WINDOW)
        {
            self.ctrl_c_armed_at = None;
            self.notice = None;
        }
    }

    fn on_ctrl_c(&mut self) {
        if !self.editor.is_empty() {
            self.editor.clear();
            self.suggestions.clear();
            self.ctrl_c_armed_at = None;
            return;
        }

        if self.ctrl_c_armed_at.is_some() {
            self.should_quit = true;
            let _ = self.action_tx.send(UserAction::Quit);
            return;
        }

        self.ctrl_c_armed_at = Some(Instant::now());
        self.notice = Some("Press Ctrl-C again to quit".to_string());
    }

    /// Keys the editor ignored: transcript scrolling and input history.
    fn on_app_key(&mut self, key: &KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let half_page = usize::from(self.chat_height / 2).max(1);
        let full_page = usize::from(self.chat_height).max(1);

        match key.code {
            KeyCode::Char('u') if ctrl => self.transcript.scroll_up(half_page),
            KeyCode::Char('d') if ctrl => self.transcript.scroll_down(half_page),
            KeyCode::PageUp => self.transcript.scroll_up(full_page),
            KeyCode::PageDown => self.transcript.scroll_down(full_page),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            _ => {}
        }
    }

    fn refresh_suggestions(&mut self) {
        let line = self.editor.text().to_string();
        if line != self.last_suggest_line {
            self.suggestions_suppressed = false;
        }

        if self.suggestions_suppressed {
            self.suggestions.clear();
        } else {
            self.suggestions =
                suggestions_for(&line, self.editor.cursor_bytes(), &self.sidebar.documents);
            self.selected_suggestion = 0;
        }

        self.last_suggest_line = line;
    }

    fn accept_selected_suggestion(&mut self) {
        if let Some(suggestion) = self.suggestions.get(self.selected_suggestion) {
            self.editor.set_text(suggestion.replacement.clone());
            self.refresh_suggestions();
        }
    }

    fn submit_line(&mut self, line: String) {
        self.suggestions.clear();
        self.suggestions_suppressed = false;
        self.last_suggest_line.clear();
        self.history_index = None;
        self.transcript.scroll_to_bottom();

        // A handler is waiting: the line answers its prompt instead of being a command. Empty
        // lines are meaningful here (they pick the prompt's default).
        if self.awaiting_reply {
            self.transcript.push(EntryKind::User, &line);
            self.awaiting_reply = false;
            let _ = self.reply_tx.send(line);
            return;
        }

        let line = line.trim();
        if line.is_empty() {
            return;
        }

        self.transcript.push(EntryKind::User, line);
        self.history.push(line.to_string());
        self.busy += 1;
        let _ = self.action_tx.send(UserAction::Submit(line.to_string()));
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }

        let next_index = match self.history_index {
            None => {
                self.draft = self.editor.text().to_string();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };

        self.history_index = Some(next_index);
        self.editor.set_text(self.history[next_index].clone());
        self.suggestions.clear();
        self.last_suggest_line = self.editor.text().to_string();
    }

    fn history_next(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };

        if index + 1 < self.history.len() {
            self.history_index = Some(index + 1);
            self.editor.set_text(self.history[index + 1].clone());
        } else {
            self.history_index = None;
            self.editor.set_text(std::mem::take(&mut self.draft));
        }

        self.suggestions.clear();
        self.last_suggest_line = self.editor.text().to_string();
    }
}

/// Launch the TUI frontend.
///
/// Spawns the rendering/input loop on a dedicated thread and runs command handlers on the
/// current task, bridged by channels: handler output streams into the transcript, and prompts
/// (such as `/resume`'s selection) turn the query box into a reply prompt. Progress bars are
/// disabled globally since the TUI owns the terminal.
///
/// # Arguments
///
/// * `config` - The loaded application configuration.
/// * `store` - The vector store to run commands against.
///
/// # Errors
///
/// * `CLIError::IOError` - If the terminal could not be set up or the UI thread failed.
/// * Any error a command handler returns, except recoverable [`CLIError::CommandError`]s,
///   which are shown in the transcript instead.
pub(crate) async fn run_tui(config: Config, store: LanceZoteroStore) -> Result<(), CLIError> {
    zqa_rag::progress::set_progress_bars_enabled(false);

    let (ui_tx, ui_rx) = crossbeam_channel::unbounded::<UiMessage>();
    let (reply_tx, reply_rx) = crossbeam_channel::unbounded::<String>();
    let (action_tx, mut action_rx) = unbounded_channel::<UserAction>();

    let stream_tx = ui_tx.clone();
    let trace_tx = ui_tx.clone();
    let mut ctx = Context {
        state: State::default(),
        config,
        store,
        input: Box::new(ChannelReader::new(ui_tx.clone(), reply_rx)),
        out: ChannelWriter::new(ui_tx.clone(), OutputLane::Out),
        err: ChannelWriter::new(ui_tx.clone(), OutputLane::Info),
        on_stream_text: Some(Arc::new(move |text: &str| {
            let _ = stream_tx.send(UiMessage::Stream(text.to_string()));
        })),
        on_tool_trace: Some(Arc::new(move |line: &str| {
            let _ = trace_tx.send(UiMessage::ToolTrace(line.to_string()));
        })),
    };

    let session = SessionInfo {
        generation_model: ctx.config.get_generation_model_name(),
        embedding_model: ctx
            .config
            .get_embedding_config()
            .map(|c| c.model_name().to_string()),
    };
    let keybindings = match get_edit_mode() {
        EditMode::Vi => Keybindings::Vi,
        _ => Keybindings::Emacs,
    };

    let _ = ui_tx.send(UiMessage::Info(format!(
        "zqa {} - ask a question about your library, or type /help for commands.",
        env!("CARGO_PKG_VERSION")
    )));
    if let Some(notice) = pending_batches_notice()? {
        let _ = ui_tx.send(UiMessage::Out(notice));
    }
    let _ = ui_tx.send(UiMessage::CommandDone(SidebarSnapshot::from_context(&ctx)));

    let ui_thread =
        thread::spawn(move || ui_thread_main(&ui_rx, action_tx, reply_tx, session, keybindings));

    let mut fatal: Option<CLIError> = None;
    while let Some(action) = action_rx.recv().await {
        match action {
            UserAction::Submit(line) => match dispatch_command(&line, &mut ctx).await {
                Ok(should_continue) => {
                    flush_context(&mut ctx);
                    let _ = ui_tx.send(UiMessage::CommandDone(SidebarSnapshot::from_context(&ctx)));
                    if !should_continue {
                        break;
                    }
                }
                Err(CLIError::CommandError(e)) => {
                    flush_context(&mut ctx);
                    let _ = ui_tx.send(UiMessage::Error(format!("Error: {e}")));
                    let _ = ui_tx.send(UiMessage::CommandDone(SidebarSnapshot::from_context(&ctx)));
                }
                Err(e) => {
                    fatal = Some(e);
                    break;
                }
            },
            UserAction::Quit => {
                // Reuse the /quit handler so the conversation is saved on the way out.
                if let Err(e) = dispatch_command("/quit", &mut ctx).await {
                    fatal = Some(e);
                }
                break;
            }
        }
    }

    let _ = ui_tx.send(UiMessage::Shutdown);
    match ui_thread.join() {
        Ok(result) => result.map_err(CLIError::IOError)?,
        Err(_) => {
            // ratatui's panic hook already restored the terminal.
            return Err(CLIError::IOError(io::Error::other(
                "the TUI thread panicked",
            )));
        }
    }

    match fatal {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Flush any partial handler output (e.g. prompts without trailing newlines) to the UI.
fn flush_context(ctx: &mut Context<ChannelWriter, ChannelWriter>) {
    use std::io::Write;

    let _ = ctx.out.flush();
    let _ = ctx.err.flush();
}

/// The UI thread: set up the terminal, run the event loop, and restore the terminal.
fn ui_thread_main(
    ui_rx: &Receiver<UiMessage>,
    action_tx: UnboundedSender<UserAction>,
    reply_tx: Sender<String>,
    session: SessionInfo,
    keybindings: Keybindings,
) -> io::Result<()> {
    let mut terminal = ratatui::try_init()?;
    let _ = execute!(io::stdout(), EnableMouseCapture);

    let mut app = App::new(session, keybindings, action_tx, reply_tx);
    let result = ui_loop(&mut terminal, &mut app, ui_rx);

    let _ = execute!(io::stdout(), DisableMouseCapture);
    ratatui::restore();

    result
}

/// The UI event loop: drain runner messages, draw, and handle input until shutdown.
fn ui_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    ui_rx: &Receiver<UiMessage>,
) -> io::Result<()> {
    loop {
        loop {
            match ui_rx.try_recv() {
                Ok(message) => {
                    if app.on_message(message) {
                        return Ok(());
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Ok(()),
            }
        }

        app.tick();
        terminal.draw(|frame| ui::draw(frame, app))?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => app.on_key(&key),
                Event::Mouse(mouse) => app.on_mouse(mouse),
                _ => {}
            }
        }

        if app.should_quit() {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::Receiver;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
    use tokio::sync::mpsc::UnboundedReceiver;
    use zqa_macros::test_eq;

    use super::{App, SessionInfo, UserAction};
    use crate::tui::input::Keybindings;
    use crate::tui::io::{SidebarSnapshot, UiMessage};

    fn create_test_app() -> (App, UnboundedReceiver<UserAction>, Receiver<String>) {
        let (action_tx, action_rx) = tokio::sync::mpsc::unbounded_channel();
        let (reply_tx, reply_rx) = crossbeam_channel::unbounded();
        let session = SessionInfo {
            generation_model: Some("claude-sonnet-4-5".to_string()),
            embedding_model: Some("voyage-3".to_string()),
        };

        let app = App::new(session, Keybindings::Emacs, action_tx, reply_tx);
        (app, action_rx, reply_rx)
    }

    fn press(app: &mut App, code: KeyCode) {
        app.on_key(&KeyEvent::new(code, KeyModifiers::NONE));
    }

    fn press_ctrl(app: &mut App, c: char) {
        app.on_key(&KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL));
    }

    fn type_str(app: &mut App, text: &str) {
        for c in text.chars() {
            press(app, KeyCode::Char(c));
        }
    }

    #[test]
    fn test_submit_sends_action_and_echoes() {
        let (mut app, mut action_rx, _reply_rx) = create_test_app();
        type_str(&mut app, "what papers discuss dropout?");
        press(&mut app, KeyCode::Enter);

        test_eq!(
            action_rx.try_recv(),
            Ok(UserAction::Submit(
                "what papers discuss dropout?".to_string()
            ))
        );
        test_eq!(app.busy, 1);

        let window = app.transcript.visible_window(80, 5);
        test_eq!(window[0].1, "> what papers discuss dropout?");
    }

    #[test]
    fn test_empty_submit_is_ignored() {
        let (mut app, mut action_rx, _reply_rx) = create_test_app();
        press(&mut app, KeyCode::Enter);

        assert!(action_rx.try_recv().is_err());
        test_eq!(app.busy, 0);
    }

    #[test]
    fn test_slash_input_shows_suggestions_and_tab_accepts() {
        let (mut app, _action_rx, _reply_rx) = create_test_app();
        type_str(&mut app, "/che");

        test_eq!(app.suggestions.len(), 1);
        test_eq!(app.suggestions[0].label, "/checkhealth");

        press(&mut app, KeyCode::Tab);
        test_eq!(app.editor.text(), "/checkhealth ");
    }

    #[test]
    fn test_suggestion_navigation_wraps() {
        let (mut app, _action_rx, _reply_rx) = create_test_app();
        type_str(&mut app, "/d");

        let count = app.suggestions.len();
        assert!(count > 1);

        press(&mut app, KeyCode::Down);
        test_eq!(app.selected_suggestion, 1);

        press(&mut app, KeyCode::Up);
        test_eq!(app.selected_suggestion, 0);

        press(&mut app, KeyCode::Up);
        test_eq!(app.selected_suggestion, count - 1);
    }

    #[test]
    fn test_esc_dismisses_suggestions_until_input_changes() {
        let (mut app, _action_rx, _reply_rx) = create_test_app();
        type_str(&mut app, "/che");
        assert!(!app.suggestions.is_empty());

        press(&mut app, KeyCode::Esc);
        assert!(app.suggestions.is_empty());

        // Cursor movement alone does not resurface the menu...
        press(&mut app, KeyCode::Left);
        assert!(app.suggestions.is_empty());

        // ...but changing the text does.
        press(&mut app, KeyCode::End);
        type_str(&mut app, "c");
        assert!(!app.suggestions.is_empty());
    }

    #[test]
    fn test_reply_mode_routes_line_to_handler() {
        let (mut app, mut action_rx, reply_rx) = create_test_app();
        assert!(!app.on_message(UiMessage::InputRequested));
        assert!(app.awaiting_reply);

        type_str(&mut app, "2");
        press(&mut app, KeyCode::Enter);

        test_eq!(reply_rx.try_recv(), Ok("2".to_string()));
        assert!(action_rx.try_recv().is_err());
        assert!(!app.awaiting_reply);
    }

    #[test]
    fn test_ctrl_c_clears_input_then_arms_quit() {
        let (mut app, mut action_rx, _reply_rx) = create_test_app();
        type_str(&mut app, "some text");

        press_ctrl(&mut app, 'c');
        assert!(app.editor.is_empty());
        assert!(!app.should_quit());

        press_ctrl(&mut app, 'c');
        assert!(app.notice.is_some());
        assert!(!app.should_quit());

        press_ctrl(&mut app, 'c');
        assert!(app.should_quit());
        test_eq!(action_rx.try_recv(), Ok(UserAction::Quit));
    }

    #[test]
    fn test_other_keys_disarm_ctrl_c() {
        let (mut app, _action_rx, _reply_rx) = create_test_app();
        press_ctrl(&mut app, 'c');
        assert!(app.notice.is_some());

        press(&mut app, KeyCode::Char('x'));
        assert!(app.notice.is_none());

        press_ctrl(&mut app, 'c'); // clears the "x"
        press_ctrl(&mut app, 'c'); // arms again
        assert!(!app.should_quit());
    }

    #[test]
    fn test_ctrl_u_and_d_scroll_when_line_empty() {
        let (mut app, _action_rx, _reply_rx) = create_test_app();
        app.chat_height = 10;
        for i in 0..40 {
            app.transcript.push(
                crate::tui::transcript::EntryKind::Output,
                &format!("line {i}"),
            );
        }
        // Establish the scroll clamp with a first render.
        let _ = app.transcript.visible_window(80, 10);

        press_ctrl(&mut app, 'u');
        assert!(app.transcript.is_scrolled());

        press_ctrl(&mut app, 'd');
        assert!(!app.transcript.is_scrolled());
    }

    #[test]
    fn test_mouse_wheel_scrolls_transcript() {
        let (mut app, _action_rx, _reply_rx) = create_test_app();
        for i in 0..40 {
            app.transcript.push(
                crate::tui::transcript::EntryKind::Output,
                &format!("line {i}"),
            );
        }
        let _ = app.transcript.visible_window(80, 10);

        app.on_mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        });
        assert!(app.transcript.is_scrolled());
    }

    #[test]
    fn test_history_navigation() {
        let (mut app, mut _action_rx, _reply_rx) = create_test_app();
        type_str(&mut app, "first query here");
        press(&mut app, KeyCode::Enter);
        type_str(&mut app, "second query here");
        press(&mut app, KeyCode::Enter);

        type_str(&mut app, "draft");
        press(&mut app, KeyCode::Up);
        test_eq!(app.editor.text(), "second query here");

        press(&mut app, KeyCode::Up);
        test_eq!(app.editor.text(), "first query here");

        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Down);
        test_eq!(app.editor.text(), "draft");
    }

    #[test]
    fn test_command_done_updates_sidebar_and_busy() {
        let (mut app, _action_rx, _reply_rx) = create_test_app();
        type_str(&mut app, "/stats");
        press(&mut app, KeyCode::Enter);
        test_eq!(app.busy, 1);

        let snapshot = SidebarSnapshot {
            input_tokens: 10,
            ..Default::default()
        };
        app.on_message(UiMessage::CommandDone(snapshot));

        test_eq!(app.busy, 0);
        test_eq!(app.sidebar.input_tokens, 10);
    }

    #[test]
    fn test_shutdown_message_requests_exit() {
        let (mut app, _action_rx, _reply_rx) = create_test_app();
        assert!(app.on_message(UiMessage::Shutdown));
    }
}
