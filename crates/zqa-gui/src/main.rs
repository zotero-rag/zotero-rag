//! Native GUI front-end for zqa, built on GPUI Kit.
//!
//! The window is a chat-style harness over the same engine the CLI drives: [`bridge`] runs the tokio-based
//! pipeline on a dedicated thread and streams output back as [`UiEvent`]s, which this view folds into a
//! transcript of turns.
//!
//! Layout follows the usual desktop AI-harness pattern: a translucent sidebar (window vibrancy shows through
//! on macOS) with session and library commands, and an opaque main pane holding the conversation. There is
//! no system title bar; the top strip of each pane drags the window instead.
mod bridge;
#[cfg(target_os = "macos")]
mod macos;
mod prompt_card;
mod prompts;

use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use bridge::{EngineCommand, UiEvent, spawn_engine};
use futures::StreamExt;
use futures::channel::mpsc::UnboundedReceiver;
use gpui_kit::assets::Assets;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{InputEvent, Textarea, TextareaState};
use gpui_kit::component::progress::Progress;
use gpui_kit::component::spinner::Spinner;
use gpui_kit::component::{
    ActiveTheme as _, Icon, IconName, InteractiveElementExt as _, Root, Sizable as _,
    TITLE_BAR_HEIGHT, Theme, ThemeMode, TitleBar, h_flex, v_flex,
};
use gpui_kit::prelude::*;
use gpui_kit::{
    AnyElement, App, Context, Decorations, Div, Entity, Focusable as _, FontWeight, HighlightStyle,
    IntoElement, KeyBinding, MouseButton, Pixels, PromptButton, PromptLevel, Rems, Render,
    ScrollHandle, SharedString, Stateful, StyledText, Subscription, Window,
    WindowBackgroundAppearance, WindowBounds, actions, div, px, rems, size, transparent_black,
};
use prompt_card::{PromptCard, PromptEvent};
use prompts::PromptRequest;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use zqa::io::{EngineEvent, StyledSpan, TextStyle};
use zqa::state::{SavedChatHistory, UsageMetadata};

/// Width of the left sidebar.
const SIDEBAR_WIDTH: Pixels = px(232.);
/// Left inset of the sidebar brand row; macOS traffic lights sit over it.
const TRAFFIC_LIGHT_INSET: Pixels = if cfg!(target_os = "macos") {
    px(80.)
} else {
    px(3.)
};
/// Width of the centered conversation column.
const CONTENT_WIDTH: Rems = rems(46.);

// Window actions
actions!(window, [CloseWindow, QuitApp]);

/// What a sidebar item does when clicked.
#[derive(Clone)]
enum SidebarAction {
    /// Send a command string to the engine.
    Run(&'static str),
    /// Resume a conversation from a saved chat history. This also saves the current history
    /// to disk.
    ResumeConversation(Arc<SavedChatHistory>),
    /// Switch the main pane between the conversation and the settings placeholder.
    ToggleSettings,
}

/// One rendered block in the transcript.
#[derive(Debug)]
enum ChatRow {
    /// A command or question sent by the user.
    User(SharedString),
    /// Reasoning output from the engine.
    Reasoning(SharedString),
    /// A tool call, shown as a collapsible panel.
    ToolCall(ToolCallRow),
    /// Answer-style stdout. Grows in place while a command streams.
    Answer(String),
    /// A block of text whose parts are styled differently, such as `/help` with its headings. The
    /// ranges are byte ranges into the text for each span that isn't plain; see
    /// [`ZqaApp::layout_spans`].
    Styled(String, Vec<(Range<usize>, TextStyle)>),
    /// Engine status lines (timings, warnings) written to stderr.
    Status(String),
    /// A command that failed.
    Failed(SharedString),
}

/// A tool call in the transcript. The panel shows the tool's name and status, and expands to show
/// its arguments and response when clicked.
///
/// TODO: The expanded panel shows raw JSON, which can be hard to read. We should consider a better
/// formatting solution here, maybe with a trait.
#[derive(Debug)]
struct ToolCallRow {
    /// The engine's ID for the call, which pairs a `ToolResponse` with its `ToolCall`.
    id: SharedString,
    /// The name of the tool that was called.
    name: SharedString,
    /// The arguments the model passed to the tool.
    args: Value,
    /// `None` while the tool is running, then the tool's result or its error message.
    response: Option<Result<Value, String>>,
    /// Whether the panel is expanded to show `args` and `response`.
    expanded: bool,
}

/// A task the engine reported with `ProgressStart`, shown above the "Working..." line until it
/// completes or the command ends.
#[derive(Debug, PartialEq)]
struct ProgressTask {
    /// The engine's ID for the task, which its updates and completion refer to.
    id: String,
    /// What the task is doing, such as "Parsing library".
    label: SharedString,
    /// The amount of work in the task. `None` means the amount is unknown, so the task shows a
    /// spinner instead of a bar.
    total: Option<u64>,
    /// The work completed so far, in the same units as `total`.
    completed: u64,
}

struct ChatRows(Vec<ChatRow>);

impl ChatRows {
    /// Convert complete chat history into renderable transcript rows.
    fn from_history(history: &[zqa_rag::llm::base::ChatHistoryItem]) -> Self {
        let mut rows = Vec::new();
        let mut tool_calls: HashMap<SharedString, (SharedString, Value)> = HashMap::new();

        for item in history {
            for content in &item.content {
                match content {
                    zqa_rag::llm::base::ChatHistoryContent::Text(text) => {
                        rows.push(match item.role {
                            zqa_rag::llm::base::MessageRole::User => ChatRow::User(text.into()),
                            zqa_rag::llm::base::MessageRole::Assistant => {
                                ChatRow::Answer(text.to_string())
                            }
                            zqa_rag::llm::base::MessageRole::Tool => {
                                ChatRow::Status(text.to_string())
                            }
                        });
                    }
                    zqa_rag::llm::base::ChatHistoryContent::Reasoning(text) => {
                        rows.push(ChatRow::Reasoning(text.into()));
                    }
                    zqa_rag::llm::base::ChatHistoryContent::ToolCallRequest(request) => {
                        tool_calls.insert(
                            request.id.as_str().into(),
                            (request.tool_name.as_str().into(), request.args.clone()),
                        );
                    }
                    zqa_rag::llm::base::ChatHistoryContent::ToolCallResponse(response) => {
                        if let Some((name, args)) = tool_calls.remove(response.id.as_str()) {
                            rows.push(ChatRow::ToolCall(ToolCallRow {
                                id: response.id.as_str().into(),
                                name,
                                args,
                                response: Some(Ok(response.result.clone())),
                                expanded: false,
                            }));
                        }
                    }
                }
            }
        }

        Self(rows)
    }
}

impl From<&SavedChatHistory> for ChatRows {
    fn from(value: &SavedChatHistory) -> Self {
        Self::from_history(&value.history)
    }
}

/// What the engine is doing right now, mirrored into the header.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Ready,
    Running,
    Resuming,
    Stopping,
    ShuttingDown,
    Ended,
}

impl Phase {
    /// Whether the engine can accept a new command in this state.
    fn accepts_commands(self) -> bool {
        matches!(self, Phase::Ready)
    }
}

/// Which pane the main area currently shows.
#[derive(Clone, Copy, PartialEq)]
enum Pane {
    Chat,
    Settings,
}

/// The single-window application view.
struct ZqaApp {
    /// The query/command input box state.
    /// TODO: A nicety for users would be to go back to a conversation and have their
    /// input remembered.
    input_state: Entity<TextareaState>,
    /// Requests waiting behind the visible question card.
    pending_prompts: VecDeque<PromptRequest>,
    active_prompt: Option<Entity<PromptCard>>,
    prompt_subscription: Option<Subscription>,
    /// Transcript blocks, oldest first.
    rows: Vec<ChatRow>,
    /// Saved conversations loaded from the zqa state directory.
    conversation_history: Result<Vec<Arc<SavedChatHistory>>, zqa::state::StateError>,
    /// What the engine thread is doing right now.
    phase: Phase,
    exit_confirmation_open: bool,
    /// Whether a `/new` reset is in flight; the transcript clears only when the engine confirms it, so a failed
    /// save cannot lose the visible copy.
    pending_reset: bool,
    /// The current conversation's total token usage, shown under the input box. `TokenUsage` events
    /// carry one query's usage each and are added to this; it resets on `/new` and is replaced by the
    /// saved total when a conversation is resumed from the sidebar.
    usage: UsageMetadata,
    /// Tasks that have started and not completed, in the order they started. Cleared when a
    /// command ends, since the engine may not complete a task that a cancelled command started.
    progress: Vec<ProgressTask>,
    /// Which pane the main area shows.
    pane: Pane,
    /// Whether the dark theme is active; toggled from the header.
    dark_theme: bool,
    /// Whether a left press on a drag surface is pending; consumed by the next mouse move to start a window drag.
    drag_armed: bool,
    /// Scrolls the transcript; pinned to the bottom while content streams in.
    scroll_handle: ScrollHandle,
    /// Channel carrying commands to the engine thread.
    cmd_tx: UnboundedSender<EngineCommand>,
    /// Channel signalling the engine to cancel the in-flight command.
    cancel_tx: UnboundedSender<()>,
    /// Kept alive so the input subscription is not dropped.
    _subscriptions: Vec<Subscription>,
}

impl ZqaApp {
    fn new(
        cmd_tx: UnboundedSender<EngineCommand>,
        cancel_tx: UnboundedSender<()>,
        event_rx: UnboundedReceiver<UiEvent>,
        dark_theme: bool,
        conversation_history: Result<Vec<Arc<SavedChatHistory>>, zqa::state::StateError>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        // A chat input: grows from one line up to six; Enter submits and Shift+Enter
        // inserts a newline (`submit_on_enter`).
        let input_state = cx.new(|cx| {
            TextareaState::new(window, cx)
                .auto_grow(1, 6)
                .submit_on_enter(true)
                .placeholder("Ask a question, or type /help")
        });

        Self::focus_input(&input_state, window, cx);
        let app = cx.entity().downgrade();
        window.on_window_should_close(cx, move |window, cx| {
            let _ = app.update(cx, |app, cx| app.request_exit(window, cx));
            false
        });

        // Submit on Enter as well as via the button.
        let subscription = cx.subscribe_in(
            &input_state,
            window,
            |this, _, ev: &InputEvent, window, cx| {
                if matches!(ev, InputEvent::PressEnter { .. }) {
                    this.submit(window, cx);
                }
            },
        );

        // Drain engine output on GPUI's executor and fold it into the transcript.
        cx.spawn_in(window, async move |this, cx| {
            let mut event_rx = event_rx;

            while let Some(event) = event_rx.next().await {
                let update = this.update_in(cx, |app, window, cx| {
                    match event {
                        UiEvent::Engine(event) => app.handle_engine_event(event, window, cx),
                        UiEvent::ShutdownComplete(result) => match result {
                            Ok(()) => cx.quit(),
                            Err(message) => {
                                app.phase = Phase::Ready;
                                app.pane = Pane::Chat;
                                app.rows.push(ChatRow::Failed(message.into()));
                                Self::focus_input(&app.input_state, window, cx);
                            }
                        },
                        UiEvent::Done(result) => {
                            app.clear_prompts();
                            app.progress.clear();
                            Self::focus_input(&app.input_state, window, cx);

                            if app.phase != Phase::ShuttingDown {
                                app.phase = Phase::Ready;
                            }

                            let refresh_history = app.pending_reset && result.is_ok();
                            Self::finish_command(&mut app.rows, &mut app.pending_reset, result);
                            if refresh_history {
                                app.usage = UsageMetadata::default();
                                app.conversation_history = Self::load_conversation_history();
                            }
                        }
                        UiEvent::ConversationResumed(result) => {
                            app.clear_prompts();
                            app.progress.clear();
                            Self::focus_input(&app.input_state, window, cx);

                            if app.phase != Phase::ShuttingDown {
                                app.phase = Phase::Ready;
                            }

                            match result {
                                Ok(conversation) => {
                                    app.rows = ChatRows::from(conversation.as_ref()).0;
                                    app.usage = conversation.usage;
                                    app.pane = Pane::Chat;
                                    app.conversation_history = Self::load_conversation_history();
                                }
                                Err(message) => app.rows.push(ChatRow::Failed(message.into())),
                            }
                        }
                        UiEvent::Cancelled => {
                            app.clear_prompts();
                            app.progress.clear();
                            Self::focus_input(&app.input_state, window, cx);

                            if app.phase != Phase::ShuttingDown {
                                app.phase = Phase::Ready;
                            }

                            // A cancelled reset may or may not have taken effect engine-side; keeping the transcript
                            // is the conservative reading.
                            app.pending_reset = false;
                            Self::fold_stderr(&mut app.rows, "(cancelled)");
                        }
                    }
                    app.scroll_handle.scroll_to_bottom();
                    cx.notify();
                });
                if update.is_err() {
                    break;
                }
            }

            // The loop ends when every event sender is dropped, i.e. the engine thread has
            // exited. Reflect that in the UI so input is disabled rather than silently
            // accepted into a dead channel.
            let _ = this.update(cx, |app, cx| {
                app.clear_prompts();
                app.progress.clear();
                app.phase = Phase::Ended;
                cx.notify();
            });
        })
        .detach();

        Self {
            input_state,
            pending_prompts: VecDeque::new(),
            active_prompt: None,
            prompt_subscription: None,
            rows: Vec::new(),
            conversation_history,
            phase: Phase::Ready,
            exit_confirmation_open: false,
            pending_reset: false,
            usage: UsageMetadata::default(),
            progress: Vec::new(),
            pane: Pane::Chat,
            dark_theme,
            drag_armed: false,
            scroll_handle: ScrollHandle::new(),
            cmd_tx,
            cancel_tx,
            _subscriptions: vec![subscription],
        }
    }

    /// Add a collapsed panel for a tool call that has started running.
    fn start_tool_call(rows: &mut Vec<ChatRow>, id: String, name: String, args: Value) {
        rows.push(ChatRow::ToolCall(ToolCallRow {
            id: id.into(),
            name: name.into(),
            args,
            response: None,
            expanded: false,
        }));
    }

    /// Record a tool's response on the panel its `ToolCall` opened.
    ///
    /// The engine sends the response with the call's ID, so the panel is found by ID; parallel tool
    /// calls can finish in any order. A response with no matching running call still gets its own
    /// panel, with no arguments, so the result is not lost.
    fn finish_tool_call(
        rows: &mut Vec<ChatRow>,
        id: String,
        name: String,
        response: Result<Value, String>,
    ) {
        let running_call = rows.iter_mut().rev().find_map(|row| match row {
            ChatRow::ToolCall(call) if call.id == id && call.response.is_none() => Some(call),
            _ => None,
        });

        match running_call {
            Some(call) => call.response = Some(response),
            None => rows.push(ChatRow::ToolCall(ToolCallRow {
                id: id.into(),
                name: name.into(),
                args: Value::Null,
                response: Some(response),
                expanded: false,
            })),
        }
    }

    /// Keep input requests in arrival order without blocking the GUI event consumer.
    fn queue_prompt(&mut self, event: EngineEvent, window: &mut Window, cx: &mut Context<Self>) {
        if self.phase != Phase::Running {
            return;
        }

        match PromptRequest::from_event(event) {
            Ok(request) => self.pending_prompts.push_back(request),
            Err(message) => {
                self.rows.push(ChatRow::Failed(message.into()));
                self.request_stop(cx);
                return;
            }
        }

        self.pane = Pane::Chat;
        self.show_next_prompt(window, cx);
    }

    /// Focus the next live request. Each card owns a fresh editor so secret input and undo
    /// history cannot leak into a subsequent prompt or the chat composer.
    fn show_next_prompt(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.active_prompt.is_some() {
            return;
        }

        while let Some(request) = self.pending_prompts.pop_front() {
            if request.is_closed() {
                continue;
            }

            let card = cx.new(|cx| PromptCard::new(request, window, cx));
            self.prompt_subscription = Some(cx.subscribe_in(
                &card,
                window,
                |this, card, event: &PromptEvent, window, cx| {
                    if this.active_prompt.as_ref() != Some(card) {
                        return;
                    }

                    match event {
                        PromptEvent::Submitted => {
                            this.active_prompt = None;
                            this.prompt_subscription = None;
                            this.show_next_prompt(window, cx);

                            if this.active_prompt.is_none() {
                                Self::focus_input(&this.input_state, window, cx);
                            }
                        }
                        PromptEvent::Cancelled => this.request_stop(cx),
                    }

                    cx.notify();
                },
            ));
            window.focus(&card.read(cx).focus_handle(cx), cx);
            self.active_prompt = Some(card);
            break;
        }
    }

    /// Discard unanswered requests and their editors when the command ends or is cancelled.
    fn clear_prompts(&mut self) {
        self.prompt_subscription = None;
        self.active_prompt = None;
        self.pending_prompts.clear();
    }

    fn handle_engine_event(
        &mut self,
        event: EngineEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            EngineEvent::Text { message } => {
                Self::fold_stdout(&mut self.rows, &message);
            }
            EngineEvent::Reasoning { message } => {
                self.rows.push(ChatRow::Reasoning(message.into()));
            }
            EngineEvent::StyledText { spans } => {
                Self::fold_styled(&mut self.rows, &spans);
            }
            EngineEvent::StatusUpdate { message } | EngineEvent::Warning { message } => {
                self.rows.push(ChatRow::Status(message));
            }
            EngineEvent::Error { message } => {
                self.rows.push(ChatRow::Failed(message.into()));
            }
            EngineEvent::ToolCall { id, name, args } => {
                Self::start_tool_call(&mut self.rows, id, name, args);
            }
            EngineEvent::ToolResponse { id, name, response } => {
                Self::finish_tool_call(&mut self.rows, id, name, response);
            }
            EngineEvent::TokenUsage { usage } => {
                self.usage += usage;
            }
            EngineEvent::ProgressStart { id, label, total } => {
                Self::start_progress(&mut self.progress, id, label, total);
            }
            EngineEvent::ProgressUpdate { id, completed } => {
                if let Some(task) = self.progress.iter_mut().find(|task| task.id == id) {
                    task.completed = completed;
                }
            }
            EngineEvent::ProgressComplete { id } => {
                self.progress.retain(|task| task.id != id);
            }
            request @ (EngineEvent::Confirm { .. }
            | EngineEvent::Choose { .. }
            | EngineEvent::Line { .. }
            | EngineEvent::Secret { .. }) => {
                self.queue_prompt(request, window, cx);
            }
            _ => {
                // EngineEvent is non_exhaustive across crate boundaries.
            }
        }
    }

    /// Add a task that has started. A task that reuses a running task's ID replaces it, as in the
    /// CLI, so a restarted task doesn't show twice.
    fn start_progress(
        tasks: &mut Vec<ProgressTask>,
        id: String,
        label: String,
        total: Option<u64>,
    ) {
        tasks.retain(|task| task.id != id);
        tasks.push(ProgressTask {
            id,
            label: label.into(),
            total,
            completed: 0,
        });
    }

    /// Load saved conversations for the sidebar.
    ///
    /// # Returns
    ///
    /// Saved conversations in reverse chronological order, or a displayable error.
    fn load_conversation_history() -> Result<Vec<Arc<SavedChatHistory>>, zqa::state::StateError> {
        zqa::state::get_conversation_history().map(|history| {
            history
                .unwrap_or_default()
                .into_iter()
                .map(Arc::new)
                .collect()
        })
    }

    /// Fold a chunk of command stdout into the transcript.
    ///
    /// Chunks append to the trailing answer row so one command renders as a single bubble; a whitespace-only
    /// chunk before any answer text is dropped rather than opening an empty row.
    fn fold_stdout(rows: &mut Vec<ChatRow>, text: &str) {
        if let Some(ChatRow::Answer(existing)) = rows.last_mut() {
            existing.push_str(text);
            return;
        }
        if !text.trim().is_empty() {
            rows.push(ChatRow::Answer(text.to_string()));
        }
    }

    /// Fold a styled block of text into the transcript.
    ///
    /// A block that is all `Secondary` text, such as a timing, joins the muted status lines like
    /// other engine status does. Anything else becomes its own [`ChatRow::Styled`] row.
    fn fold_styled(rows: &mut Vec<ChatRow>, spans: &[StyledSpan]) {
        if spans
            .iter()
            .all(|span| matches!(span.style, TextStyle::Secondary))
        {
            let text: String = spans.iter().map(|span| span.text.as_str()).collect();
            Self::fold_stderr(rows, &text);
            return;
        }

        let (text, styles) = Self::layout_spans(spans);
        if !text.is_empty() {
            rows.push(ChatRow::Styled(text, styles));
        }
    }

    /// Join `spans` into one string and record where each non-plain span lies in it.
    ///
    /// Trailing whitespace is trimmed, as for answer rows, and ranges are cut to the trimmed text.
    /// Plain spans get no range, since they use the row's default style.
    ///
    /// # Returns
    ///
    /// The joined text, and the byte range and style of each non-plain span, in order.
    fn layout_spans(spans: &[StyledSpan]) -> (String, Vec<(Range<usize>, TextStyle)>) {
        let mut text = String::new();
        let mut styles = Vec::new();

        for span in spans {
            let start = text.len();
            text.push_str(&span.text);

            if !matches!(span.style, TextStyle::Plain) {
                styles.push((start..text.len(), span.style));
            }
        }

        text.truncate(text.trim_end().len());
        let len = text.len();
        let styles = styles
            .into_iter()
            .map(|(range, style)| (range.start..range.end.min(len), style))
            .filter(|(range, _)| !range.is_empty())
            .collect();

        (text, styles)
    }

    /// Fold stderr lines (timings, warnings) into the transcript.
    ///
    /// Consecutive lines merge into one muted block; stdout arriving between them splits them again, which
    /// preserves chronological order.
    fn fold_stderr(rows: &mut Vec<ChatRow>, text: &str) {
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            match rows.last_mut() {
                Some(ChatRow::Status(existing)) => {
                    existing.push('\n');
                    existing.push_str(line);
                }
                _ => rows.push(ChatRow::Status(line.to_string())),
            }
        }
    }

    /// Record a dispatched command in the transcript state.
    ///
    /// `/new` is a reset rather than a conversational turn: it is not echoed as a user bubble, and it only
    /// arms the deferred reset that [`Self::finish_command`] commits or drops.
    fn record_command(rows: &mut Vec<ChatRow>, pending_reset: &mut bool, command: &str) {
        if command == "/new" {
            *pending_reset = true;
        } else {
            rows.push(ChatRow::User(command.into()));
        }
    }

    /// Apply a finished command's result to the transcript and any pending reset.
    ///
    /// A pending `/new` reset is committed only on success: if the engine could not save the old conversation,
    /// the failure surfaces as a [`ChatRow::Failed`] next to the retained transcript instead of silently
    /// discarding it.
    fn finish_command(
        rows: &mut Vec<ChatRow>,
        pending_reset: &mut bool,
        result: Result<bool, String>,
    ) {
        if result.is_ok() && *pending_reset {
            rows.clear();
        }
        *pending_reset = false;
        if let Err(msg) = result {
            rows.push(ChatRow::Failed(msg.into()));
        }
    }

    /// Send a command to the engine thread.
    ///
    /// # Arguments
    ///
    /// * `command` - The raw input text; trimmed before dispatch.
    ///
    /// # Returns
    ///
    /// Whether the command was dispatched; false when it is empty or the engine is busy or ended.
    fn dispatch(&mut self, command: String, cx: &mut Context<Self>) -> bool {
        let command = command.trim().to_string();
        if command.is_empty() || !self.phase.accepts_commands() {
            return false;
        }

        // `/new` starts a fresh engine conversation, so the visible transcript restarts with it. The reset
        // is deferred until the engine confirms it, and the command itself is not echoed; see [`record_command`]
        // and [`finish_command`].
        Self::record_command(&mut self.rows, &mut self.pending_reset, &command);

        self.phase = Phase::Running;
        let _ = self.cmd_tx.send(EngineCommand::Dispatch(command));
        self.scroll_handle.scroll_to_bottom();
        cx.notify();
        true
    }

    /// Read the input box, dispatch its contents, and reset the input.
    fn submit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let command = self.input_state.read(cx).value().to_string();
        if !self.dispatch(command, cx) {
            return;
        }

        self.input_state
            .update(cx, |state, cx| state.set_value("", window, cx));
        Self::focus_input(&self.input_state, window, cx);
    }

    /// Focus the chat input.
    fn focus_input(state: &Entity<TextareaState>, window: &mut Window, cx: &mut Context<Self>) {
        let handle = state.read(cx).focus_handle(cx);
        window.focus(&handle, cx);
    }

    /// Ask the engine to cancel the in-flight command.
    ///
    /// `phase` moves to `Stopping` immediately so further clicks and submissions are rejected, and returns
    /// to `Ready` only when the engine confirms with [`UiEvent::Cancelled`] (or [`UiEvent::Done`] if the
    /// command finished first).
    fn request_stop(&mut self, cx: &mut Context<Self>) {
        if self.phase != Phase::Running {
            return;
        }
        let _ = self.cancel_tx.send(());
        self.phase = Phase::Stopping;
        self.clear_prompts();
        cx.notify();
    }

    /// Confirm interruption of active work, then request a save before exiting.
    ///
    /// Repeated close requests share one dialog. Dismissing it leaves the engine untouched;
    /// accepting it starts shutdown, but the app stays open until the engine acknowledges saving.
    fn request_exit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.exit_confirmation_open || self.phase == Phase::ShuttingDown {
            return;
        }

        if matches!(self.phase, Phase::Ready | Phase::Ended) {
            self.begin_shutdown(cx);
            return;
        }

        self.exit_confirmation_open = true;
        let answer = window.prompt(
            PromptLevel::Warning,
            "Quit while work is in progress?",
            Some("The unfinished operation may be interrupted and its answer lost. Completed conversation history will be saved before quitting."),
            &[PromptButton::cancel("Keep working"), PromptButton::new("Quit")],
            cx,
        );
        cx.spawn(async move |this, cx| {
            let answer = answer.await;
            let _ = this.update(cx, |this, cx| {
                this.exit_confirmation_open = false;

                if answer == Ok(1) {
                    this.begin_shutdown(cx);
                }

                cx.notify();
            });
        })
        .detach();
    }

    /// Stop active dispatch and queue shutdown. Save failures leave the engine available for retry.
    fn begin_shutdown(&mut self, cx: &mut Context<Self>) {
        if self.phase == Phase::Ended {
            cx.quit();
            return;
        }

        if self.cmd_tx.send(EngineCommand::Shutdown).is_err() {
            self.rows.push(ChatRow::Failed(
                "The engine is unavailable; could not save before exit.".into(),
            ));
            self.pane = Pane::Chat;
            cx.notify();
            return;
        }

        self.phase = Phase::ShuttingDown;
        self.clear_prompts();
        cx.notify();
    }

    /// Make a pane's top strip drag the window, with double-click to zoom.
    ///
    /// The press is armed on mouse down and consumed by the next mouse move, mirroring gpui-component's
    /// `TitleBar`: starting the drag directly from mouse down makes macOS delay clicks while it disambiguates
    /// double clicks.
    fn make_draggable(&mut self, surface: Stateful<Div>, cx: &mut Context<Self>) -> Stateful<Div> {
        surface
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.drag_armed = true;
                    cx.stop_propagation();
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, _| {
                    this.drag_armed = false;
                }),
            )
            .on_mouse_move(cx.listener(|this, _, window, _cx| {
                if this.drag_armed {
                    this.drag_armed = false;
                    window.start_window_move();
                }
            }))
            .when(cfg!(target_os = "macos"), |this| {
                this.on_double_click(|_, window, _| window.titlebar_double_click())
            })
            .when(cfg!(target_os = "linux"), |this| {
                this.on_double_click(|_, window, _| window.zoom_window())
            })
    }

    /// Render the translucent sidebar: brand row, command items, settings.
    fn render_sidebar(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let conversation_history = self.render_conversation_history(cx);
        let window_controls = Self::render_window_controls(window);
        let brand = self.make_draggable(
            div()
                .id("brand-drag")
                .flex()
                .items_center()
                .justify_between()
                .h(TITLE_BAR_HEIGHT)
                .pl(TRAFFIC_LIGHT_INSET)
                .flex_shrink_0()
                .child(
                    div()
                        .text_size(px(13.))
                        .font_weight(FontWeight::SEMIBOLD)
                        .child("zqa"),
                )
                .child(window_controls),
            cx,
        );

        v_flex()
            .w(SIDEBAR_WIDTH)
            .h_full()
            .flex_shrink_0()
            .bg(cx.theme().background.opacity(0.72))
            .border_r_1()
            .border_color(cx.theme().border)
            .child(brand)
            .child(
                v_flex()
                    .flex_1()
                    .min_h_0()
                    .p_2()
                    .gap_1()
                    .child(self.sidebar_item(
                        "sb-new",
                        Some(IconName::Plus),
                        "New chat",
                        SidebarAction::Run("/new"),
                        false,
                        cx,
                    ))
                    .child(self.sidebar_item(
                        "sb-help",
                        Some(IconName::Info),
                        "Help",
                        SidebarAction::Run("/help"),
                        false,
                        cx,
                    ))
                    .child(
                        div()
                            .px_2()
                            .pt_2()
                            .pb_1()
                            .text_size(px(11.))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(cx.theme().muted_foreground)
                            .child("History"),
                    )
                    .child(conversation_history),
            )
            .child(div().p_2().child(self.sidebar_item(
                "sb-settings",
                Some(IconName::Settings),
                "Settings",
                SidebarAction::ToggleSettings,
                self.pane == Pane::Settings,
                cx,
            )))
            .into_any_element()
    }

    /// Render the min/zoom/close cluster on Linux, where the window manager draws no
    /// controls for client-decorated windows. Empty on other platforms.
    ///
    /// `use<>` keeps the returned element `'static`: edition 2024 return-position
    /// `impl Trait` otherwise captures the `window` borrow, which would pin it for
    /// the rest of the caller.
    fn render_window_controls(window: &mut Window) -> impl IntoElement + use<> {
        let client_decorated = cfg!(target_os = "linux")
            && matches!(window.window_decorations(), Decorations::Client { .. });

        h_flex()
            .gap_1()
            .pr_2()
            // Clicks on the controls must not start a window drag.
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .when(client_decorated, |this| {
                this.child(
                    Button::new("win-minimize")
                        .ghost()
                        .small()
                        .compact()
                        .icon(IconName::Minus)
                        .on_click(|_, window, _| window.minimize_window()),
                )
                .child(
                    Button::new("win-zoom")
                        .ghost()
                        .small()
                        .compact()
                        .icon(IconName::Maximize)
                        .on_click(|_, window, _| window.zoom_window()),
                )
                .child(
                    Button::new("win-close")
                        .ghost()
                        .small()
                        .compact()
                        .icon(IconName::Close)
                        .on_click(|_, window, _| window.remove_window()),
                )
            })
    }

    /// Render one sidebar command row.
    fn sidebar_item(
        &mut self,
        id: &'static str,
        icon: Option<IconName>,
        label: &'static str,
        action: SidebarAction,
        selected: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        // The settings toggle is pure UI and works in any engine state; commands need
        // a live engine.
        let enabled =
            matches!(&action, SidebarAction::ToggleSettings) || self.phase.accepts_commands();

        let mut el = div()
            .id(id)
            .flex()
            .items_center()
            .gap_2()
            .rounded_md()
            .px_2()
            .py_1()
            .text_size(px(13.))
            .when(selected, |this| this.bg(cx.theme().accent))
            .when(enabled, |this| {
                this.text_color(cx.theme().foreground)
                    .hover(|style| style.bg(cx.theme().accent))
            })
            .when(!enabled, |this| {
                this.text_color(cx.theme().muted_foreground).opacity(0.6)
            });

        if let Some(icon) = icon {
            el = el
                .child(
                    Icon::new(icon)
                        .small()
                        .text_color(cx.theme().muted_foreground),
                )
                .child(label)
                .on_click(cx.listener(move |this, _, _, cx| {
                    if !enabled {
                        return;
                    }
                    this.execute_sidebar_action(&action, cx);
                }))
        }

        el
    }

    /// Apply a sidebar action after its enabled state has been checked.
    fn execute_sidebar_action(&mut self, action: &SidebarAction, cx: &mut Context<Self>) {
        match action {
            SidebarAction::Run(command) => {
                self.pane = Pane::Chat;
                self.dispatch((*command).to_string(), cx);
            }
            SidebarAction::ResumeConversation(conversation) => {
                self.pane = Pane::Chat;
                self.phase = Phase::Resuming;
                let _ = self
                    .cmd_tx
                    .send(EngineCommand::ResumeConversation(Arc::clone(conversation)));
                cx.notify();
            }
            SidebarAction::ToggleSettings => {
                self.pane = match self.pane {
                    Pane::Settings => Pane::Chat,
                    Pane::Chat => Pane::Settings,
                };
                cx.notify();
            }
        }
    }

    /// Render the saved conversation list or its empty/error state.
    fn render_conversation_history(&self, cx: &mut Context<Self>) -> AnyElement {
        match &self.conversation_history {
            Ok(history) if history.is_empty() => div()
                .px_2()
                .py_1()
                .text_size(px(11.))
                .text_color(cx.theme().muted_foreground)
                .child("No saved conversations yet.")
                .into_any_element(),
            Ok(history) => div()
                .id("conversation-history")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .child(v_flex().gap_1().children(history.iter().enumerate().map(
                    |(index, conversation)| {
                        Self::conversation_history_item(index, conversation, cx)
                    },
                )))
                .into_any_element(),
            Err(error) => div()
                .px_2()
                .py_1()
                .text_size(px(11.))
                .text_color(cx.theme().danger)
                .child(SharedString::from(format!(
                    "Failed to load history: {error}"
                )))
                .into_any_element(),
        }
    }

    /// Render one saved conversation's title and latest text preview.
    fn conversation_history_item(
        index: usize,
        conversation: &Arc<SavedChatHistory>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let title: SharedString = conversation.title.clone().into();
        let preview: SharedString = conversation
            .preview()
            .unwrap_or("No text messages")
            .to_owned()
            .into();
        let action = SidebarAction::ResumeConversation(Arc::clone(conversation));

        div()
            .id(("history-item", index))
            .flex()
            .items_center()
            .rounded_md()
            .px_2()
            .py_1()
            .text_color(cx.theme().foreground)
            .hover(|style| style.bg(cx.theme().accent))
            .child(
                v_flex()
                    .w_full()
                    .min_w_0()
                    .child(
                        div()
                            .text_size(px(13.))
                            .font_weight(FontWeight::MEDIUM)
                            .truncate()
                            .child(title),
                    )
                    .child(
                        div()
                            .text_size(px(11.))
                            .text_color(cx.theme().muted_foreground)
                            .truncate()
                            .child(preview),
                    ),
            )
            .on_click(cx.listener(move |this, _, _, cx| {
                if this.phase.accepts_commands() {
                    this.execute_sidebar_action(&action, cx);
                }
            }))
    }

    /// Render the main pane: header plus the active pane's body.
    fn render_main(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let body = match self.pane {
            Pane::Chat => v_flex()
                .flex_1()
                .min_h_0()
                .child(self.render_transcript(cx))
                .when_some(self.active_prompt.clone(), |body, prompt| {
                    body.child(
                        div()
                            .flex_shrink_0()
                            .px_4()
                            .pb_2()
                            .flex()
                            .justify_center()
                            .child(div().w_full().max_w(CONTENT_WIDTH).child(prompt)),
                    )
                })
                .child(self.render_dock(cx))
                .into_any_element(),
            Pane::Settings => self.render_settings(cx),
        };

        v_flex()
            .flex_1()
            .min_w_0()
            .h_full()
            // Opaque over the window vibrancy; only the sidebar is translucent.
            .bg(cx.theme().background)
            .child(self.render_header(cx))
            .child(body)
            .into_any_element()
    }

    /// Render the settings pane: a placeholder until real settings exist.
    fn render_settings(&self, cx: &Context<Self>) -> AnyElement {
        div()
            .flex_1()
            .flex()
            .items_center()
            .justify_center()
            .child(
                v_flex()
                    .gap_2()
                    .items_center()
                    .child(
                        Icon::new(IconName::Settings)
                            .large()
                            .text_color(cx.theme().muted_foreground),
                    )
                    .child(
                        div()
                            .text_size(px(17.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("Settings"),
                    )
                    .child(
                        div()
                            .text_size(px(13.))
                            .text_color(cx.theme().muted_foreground)
                            .child("Nothing here yet."),
                    ),
            )
            .into_any_element()
    }

    /// Render the main pane's draggable header: engine phase on the left, theme
    /// toggle on the right.
    fn render_header(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let lead = if self.pane == Pane::Settings {
            // A pane title instead of engine state while settings are up.
            div()
                .text_size(px(13.))
                .font_weight(FontWeight::SEMIBOLD)
                .child("Settings")
                .into_any_element()
        } else {
            let (dot, label) = match self.phase {
                Phase::Ready => (cx.theme().success, "Ready"),
                Phase::Running => (cx.theme().info, "Working"),
                Phase::Resuming => (cx.theme().info, "Resuming"),
                Phase::Stopping => (cx.theme().warning, "Stopping"),
                Phase::ShuttingDown => (cx.theme().warning, "Saving before exit"),
                Phase::Ended => (cx.theme().muted_foreground, "Session ended"),
            };
            h_flex()
                .gap_2()
                .items_center()
                .child(div().size_2().rounded_full().bg(dot))
                .child(
                    div()
                        .text_size(px(12.))
                        .text_color(cx.theme().muted_foreground)
                        .child(label),
                )
                .into_any_element()
        };

        let toggle = Button::new("toggle-theme")
            .accessibility_label("Toggle color theme")
            .ghost()
            .small()
            .compact()
            .icon(if self.dark_theme {
                IconName::Sun
            } else {
                IconName::Moon
            })
            .on_click(cx.listener(|this, _, window, cx| {
                let mode = if this.dark_theme {
                    ThemeMode::Light
                } else {
                    ThemeMode::Dark
                };
                this.dark_theme = !this.dark_theme;
                Theme::change(mode, Some(window), cx);
            }));

        let header = div()
            .id("header-drag")
            .flex()
            .items_center()
            .justify_between()
            .h(TITLE_BAR_HEIGHT)
            .px_3()
            .flex_shrink_0()
            .child(lead)
            // Clicks on the toggle must not start a window drag.
            .child(
                div()
                    .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    .child(toggle),
            );

        self.make_draggable(header, cx).into_any_element()
    }

    /// Render the scrolling transcript of conversation turns.
    fn render_transcript(&self, cx: &Context<Self>) -> AnyElement {
        let busy = matches!(
            self.phase,
            Phase::Running | Phase::Resuming | Phase::Stopping | Phase::ShuttingDown
        );

        let scroll = || {
            div()
                .id("transcript")
                .flex_1()
                .overflow_y_scroll()
                .track_scroll(&self.scroll_handle)
        };

        if self.rows.is_empty() && !busy {
            return scroll()
                .flex()
                .items_center()
                .justify_center()
                .child(self.render_welcome(cx))
                .into_any_element();
        }

        // TODO(ZOT-220): the full row list is re-cloned and re-laid-out on every repaint,
        // which is O(n) per frame and quadratic over a session. Virtualize the transcript
        // so per-frame cost is O(visible) and scrollback stays unbounded without a cap.
        let column = v_flex()
            .w_full()
            .max_w(CONTENT_WIDTH)
            .px_2()
            .py_4()
            .gap_3()
            .children(
                self.rows
                    .iter()
                    .enumerate()
                    .filter_map(|(ix, row)| self.render_row(ix, row, cx)),
            )
            .when(busy, |this| {
                this.children(
                    self.progress
                        .iter()
                        .enumerate()
                        .map(|(ix, task)| Self::render_progress(ix, task, cx)),
                )
                .child(
                    h_flex()
                        .w_full()
                        .items_center()
                        .gap_2()
                        .child(Spinner::new().xsmall())
                        .child(
                            div()
                                .text_size(px(13.))
                                .text_color(cx.theme().muted_foreground)
                                .child(match self.phase {
                                    Phase::Resuming => "Resuming...",
                                    Phase::Stopping => "Stopping...",
                                    Phase::ShuttingDown => "Saving before exit...",
                                    _ => "Working...",
                                }),
                        ),
                )
            });

        scroll()
            .child(div().w_full().flex().justify_center().child(column))
            .into_any_element()
    }

    /// Render a running task: a spinner and its label when the amount of work is unknown, or its
    /// label, a `completed / total` count and a progress bar when it is known.
    fn render_progress(ix: usize, task: &ProgressTask, cx: &Context<Self>) -> AnyElement {
        let label = div()
            .flex_1()
            .text_size(px(13.))
            .text_color(cx.theme().muted_foreground)
            .child(task.label.clone());

        let Some(total) = task.total else {
            return h_flex()
                .w_full()
                .items_center()
                .gap_2()
                .child(Spinner::new().xsmall())
                .child(label)
                .into_any_element();
        };

        // An empty task has nothing left to do, so it shows as full.
        let percent = if total == 0 {
            100.0
        } else {
            task.completed as f32 / total as f32 * 100.0
        };

        v_flex()
            .w_full()
            .gap_1()
            .child(
                h_flex().w_full().items_center().gap_2().child(label).child(
                    div()
                        .text_size(px(12.))
                        .text_color(cx.theme().muted_foreground)
                        .child(format!(
                            "{} / {}",
                            format_count(task.completed),
                            format_count(total)
                        )),
                ),
            )
            .child(Progress::new(("progress", ix)).value(percent))
            .into_any_element()
    }

    /// Render one transcript block. `ix` is the row's index in `self.rows`, which interactive rows
    /// use for their element IDs and to find themselves again when clicked.
    fn render_row(&self, ix: usize, row: &ChatRow, cx: &Context<Self>) -> Option<AnyElement> {
        let element = match row {
            ChatRow::User(text) => h_flex()
                .w_full()
                .justify_end()
                .child(
                    div()
                        .max_w(rems(36.))
                        .rounded_xl()
                        .bg(cx.theme().secondary)
                        .text_color(cx.theme().secondary_foreground)
                        .px_3()
                        .py_2()
                        .text_size(px(13.))
                        .whitespace_normal()
                        .child(text.trim_end().to_string()),
                )
                .into_any_element(),
            ChatRow::Answer(text) => {
                let body = text.trim_end();
                if body.is_empty() {
                    return None;
                }
                div()
                    .w_full()
                    .text_size(px(14.))
                    .line_height(rems(1.45))
                    .whitespace_normal()
                    .child(body.to_string())
                    .into_any_element()
            }
            ChatRow::Styled(text, styles) => {
                let highlights = styles.iter().map(|(range, style)| {
                    let highlight = match style {
                        TextStyle::Plain => HighlightStyle::default(),
                        TextStyle::Heading => HighlightStyle {
                            font_weight: Some(FontWeight::SEMIBOLD),
                            ..HighlightStyle::default()
                        },
                        TextStyle::Secondary => HighlightStyle {
                            color: Some(cx.theme().muted_foreground),
                            ..HighlightStyle::default()
                        },
                    };
                    (range.clone(), highlight)
                });

                div()
                    .w_full()
                    .text_size(px(14.))
                    .line_height(rems(1.45))
                    .whitespace_normal()
                    .child(StyledText::new(text.clone()).with_highlights(highlights))
                    .into_any_element()
            }
            ChatRow::Reasoning(text) => {
                let body = text.trim();
                if body.is_empty() {
                    return None;
                }
                div()
                    .w_full()
                    .border_l_2()
                    .border_color(cx.theme().border)
                    .pl_3()
                    .text_size(px(12.))
                    .line_height(rems(1.35))
                    .text_color(cx.theme().muted_foreground)
                    .whitespace_normal()
                    .child(body.to_string())
                    .into_any_element()
            }
            ChatRow::ToolCall(call) => Self::render_tool_call(ix, call, cx),
            ChatRow::Status(lines) => h_flex()
                .w_full()
                .justify_center()
                .child(
                    div()
                        .max_w_full()
                        .px_2()
                        .text_size(px(11.))
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_color(cx.theme().muted_foreground)
                        .whitespace_normal()
                        .child(lines.clone()),
                )
                .into_any_element(),
            ChatRow::Failed(message) => h_flex()
                .w_full()
                .justify_start()
                .child(
                    div()
                        .max_w(rems(42.))
                        .rounded_xl()
                        .border_1()
                        .border_color(cx.theme().danger.opacity(0.4))
                        .bg(cx.theme().danger.opacity(0.08))
                        .text_color(cx.theme().danger)
                        .px_3()
                        .py_2()
                        .text_size(px(13.))
                        .whitespace_normal()
                        .child(message.to_string()),
                )
                .into_any_element(),
        };
        Some(element)
    }

    /// Render a tool call as a collapsible panel.
    ///
    /// The header shows a chevron, the tool's name, and its status: a spinner while it runs, then a
    /// check or a cross. Clicking the header toggles the body, which shows the arguments and the
    /// response or error as pretty-printed JSON.
    fn render_tool_call(ix: usize, call: &ToolCallRow, cx: &Context<Self>) -> AnyElement {
        let status = match &call.response {
            None => Spinner::new().xsmall().into_any_element(),
            Some(Ok(_)) => Icon::new(IconName::CircleCheck)
                .xsmall()
                .text_color(cx.theme().muted_foreground)
                .into_any_element(),
            Some(Err(_)) => Icon::new(IconName::CircleX)
                .xsmall()
                .text_color(cx.theme().danger)
                .into_any_element(),
        };

        let header = h_flex()
            .id(("tool-call", ix))
            .w_full()
            .items_center()
            .gap_2()
            .px_3()
            .py_2()
            .cursor_pointer()
            .text_size(px(12.))
            .child(
                Icon::new(if call.expanded {
                    IconName::ChevronDown
                } else {
                    IconName::ChevronRight
                })
                .xsmall()
                .text_color(cx.theme().muted_foreground),
            )
            .child(
                div()
                    .flex_1()
                    .font_weight(FontWeight::MEDIUM)
                    .child(call.name.clone()),
            )
            .child(status)
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(ChatRow::ToolCall(call)) = this.rows.get_mut(ix) {
                    call.expanded = !call.expanded;
                    cx.notify();
                }
            }));

        let pretty = |value: &Value| {
            serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        };
        let body = call.expanded.then(|| {
            let response = match &call.response {
                None => "Response:\n(running)".to_string(),
                Some(Ok(value)) => format!("Response:\n{}", pretty(value)),
                Some(Err(message)) => format!("Error:\n{message}"),
            };

            div()
                .px_3()
                .pb_3()
                .text_size(px(11.))
                .font_family(cx.theme().mono_font_family.clone())
                .text_color(cx.theme().muted_foreground)
                .whitespace_normal()
                .child(format!("Arguments:\n{}\n\n{response}", pretty(&call.args)))
        });

        v_flex()
            .w_full()
            .rounded_lg()
            .border_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().secondary.opacity(0.35))
            .child(header)
            .children(body)
            .into_any_element()
    }

    /// Render the empty-state welcome panel.
    fn render_welcome(&self, cx: &Context<Self>) -> AnyElement {
        v_flex()
            .gap_2()
            .items_center()
            .child(
                div()
                    .size(px(56.))
                    .rounded_full()
                    .bg(cx.theme().primary.opacity(0.12))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(
                        Icon::new(IconName::Bot)
                            .large()
                            .text_color(cx.theme().primary),
                    ),
            )
            .child(
                div()
                    .text_size(px(17.))
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("Ask your library"),
            )
            .child(
                div()
                    .text_size(px(13.))
                    .text_color(cx.theme().muted_foreground)
                    .child("Query your Zotero collection, or pick a command from the sidebar"),
            )
            .into_any_element()
    }

    /// Render the floating input card with its circular action button.
    fn render_dock(&mut self, cx: &mut Context<Self>) -> AnyElement {
        if self.phase == Phase::Ended {
            return h_flex()
                .justify_center()
                .px_4()
                .py_3()
                .flex_shrink_0()
                .text_size(px(12.))
                .text_color(cx.theme().muted_foreground)
                .child("The session has ended. Restart zqa-gui to start a new one.")
                .into_any_element();
        }

        let stoppable = matches!(self.phase, Phase::Running | Phase::Stopping);
        let resuming = matches!(self.phase, Phase::Resuming | Phase::ShuttingDown);
        let busy = stoppable || resuming;

        let action = div()
            .id(if stoppable {
                "stop-button"
            } else if resuming {
                "resume-progress"
            } else {
                "send-button"
            })
            .size_8()
            .rounded_full()
            .flex()
            .items_center()
            .justify_center()
            .when(!busy, |this| {
                this.bg(cx.theme().primary)
                    .text_color(cx.theme().primary_foreground)
                    .hover(|style| style.bg(cx.theme().primary_hover))
                    .active(|style| style.bg(cx.theme().primary_active))
            })
            .when(stoppable, |this| {
                this.bg(cx.theme().secondary)
                    .text_color(cx.theme().secondary_foreground)
                    .hover(|style| style.bg(cx.theme().secondary_hover))
            })
            .when(resuming, |this| {
                this.bg(cx.theme().secondary).child(Spinner::new().xsmall())
            })
            .when(!resuming, |this| {
                this.child(Icon::new(if stoppable {
                    IconName::Pause
                } else {
                    IconName::ArrowUp
                }))
            })
            .on_click(cx.listener(move |this, _, window, cx| {
                if stoppable {
                    this.request_stop(cx);
                } else if !resuming {
                    this.submit(window, cx);
                }
            }));

        v_flex()
            .flex_shrink_0()
            .px_4()
            .pb_3()
            .pt_1()
            .child(
                div().w_full().flex().justify_center().child(
                    div()
                        .w_full()
                        .max_w(CONTENT_WIDTH)
                        .rounded_2xl()
                        .border_1()
                        .border_color(cx.theme().border)
                        .bg(cx.theme().popover)
                        .p_2()
                        .child(
                            Textarea::new(&self.input_state)
                                .appearance(false)
                                .disabled(busy),
                        )
                        .child(h_flex().justify_end().child(action)),
                ),
            )
            .when(self.usage != UsageMetadata::default(), |this| {
                this.child(self.render_usage_footer(cx))
            })
            .into_any_element()
    }

    /// Render the conversation's token usage and estimated cost as one muted line under the input
    /// box, aligned with its left edge. The cost is in U.S. dollars and is left out when no model's
    /// price is known.
    fn render_usage_footer(&self, cx: &Context<Self>) -> AnyElement {
        let mut parts = vec![
            format!(
                "{} input tokens",
                format_count(self.usage.input_tokens.into())
            ),
            format!(
                "{} output tokens",
                format_count(self.usage.output_tokens.into())
            ),
        ];

        if self.usage.estimated_cost > 0 {
            let dollars = f64::from(self.usage.estimated_cost) / 100.0;
            parts.push(format!("${dollars:.4}"));
        }

        // The same centered, width-capped column as the input box, so the line starts at its edge.
        div()
            .w_full()
            .flex()
            .justify_center()
            .child(
                div()
                    .w_full()
                    .max_w(CONTENT_WIDTH)
                    .px_3()
                    .pt_2()
                    .text_size(px(11.))
                    .text_color(cx.theme().muted_foreground)
                    .child(parts.join(" | ")),
            )
            .into_any_element()
    }
}

/// Format a count with comma thousands separators, such as `12,345`.
fn format_count(count: u64) -> String {
    let digits = count.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);

    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(digit);
    }

    formatted
}

impl Render for ZqaApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        h_flex()
            .on_action(cx.listener(|this, _: &CloseWindow, window, cx| {
                this.request_exit(window, cx);
            }))
            .on_action(cx.listener(|this, _: &QuitApp, window, cx| {
                this.request_exit(window, cx);
            }))
            .size_full()
            .text_color(cx.theme().foreground)
            .child(self.render_sidebar(window, cx))
            .child(self.render_main(cx))
    }
}

fn main() {
    // Resolve the LanceDB location the same way the CLI does, before spawning any threads.
    // Otherwise the engine thread opens the default relative path instead of the state-dir
    // database and every query reports a missing table.
    zqa::set_default_lancedb_uri();

    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel::<EngineCommand>();
    let (cancel_tx, cancel_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let (event_tx, event_rx) = futures::channel::mpsc::unbounded::<UiEvent>();
    spawn_engine(cmd_rx, cancel_rx, event_tx);

    let app = gpui_kit::application().with_assets(Assets);
    app.run(move |cx: &mut App| {
        gpui_kit::init(cx);

        cx.bind_keys([
            KeyBinding::new("cmd-w", CloseWindow, None),
            KeyBinding::new("cmd-q", QuitApp, None),
        ]);
        cx.on_window_closed(|cx, _window_id| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        // `init` always loads the light theme first; honor the OS appearance until the
        // user toggles it in-app. Must run after `init`, which overwrites the global.
        let dark_theme = matches!(
            cx.window_appearance(),
            gpui_kit::WindowAppearance::Dark | gpui_kit::WindowAppearance::VibrantDark
        );
        let mode = if dark_theme {
            ThemeMode::Dark
        } else {
            ThemeMode::Light
        };
        Theme::change(mode, None, cx);

        // `appears_transparent` + owned titlebar drag come from `TitleBar::window_options`;
        // we render the drag surfaces ourselves instead of a `TitleBar` widget. The window
        // background is blurred on macOS so the translucent sidebar picks up vibrancy.
        let mut window_options = TitleBar::window_options();
        window_options.window_bounds = Some(WindowBounds::centered(size(px(1080.), px(760.)), cx));
        window_options.window_min_size = Some(size(px(760.), px(520.)));
        window_options.window_background = if cfg!(target_os = "macos") {
            WindowBackgroundAppearance::Blurred
        } else {
            WindowBackgroundAppearance::Opaque
        };

        cx.spawn(async move |cx| {
            cx.open_window(window_options, |window, cx| {
                #[cfg(target_os = "macos")]
                if !macos::use_sidebar_material(window) {
                    eprintln!("Could not configure the native sidebar blur material.");
                }

                let history = ZqaApp::load_conversation_history();
                let view = cx.new(|cx| {
                    ZqaApp::new(cmd_tx, cancel_tx, event_rx, dark_theme, history, window, cx)
                });

                // The root stays transparent so the sidebar can show the blur.
                cx.new(|cx| Root::new(view, window, cx).bg(transparent_black()))
            })
            .expect("failed to open window");
        })
        .detach();
    });
}

#[cfg(test)]
mod tests {
    use gpui_kit::test::TestWindowExt;
    use gpui_kit::{Entity, TestAppContext, VisualTestContext, px, size};
    use tokio::sync::{mpsc, oneshot};
    use zqa::io::{EngineEvent, StyledSpan, TextStyle};
    use zqa_macros::test_eq;
    use zqa_rag::llm::base::{
        ChatHistoryContent, ChatHistoryItem, MessageRole, ToolCallRequest, ToolCallResponse,
    };

    use super::{ChatRow, ChatRows, EngineCommand, Phase, ProgressTask, UiEvent, ZqaApp};

    /// A whole-app fixture with no engine thread, provider calls, or initial filesystem reads.
    /// Tests can seed private view state through `app.update`, inject engine events through
    /// `events`, and inspect outgoing commands and cancellation signals.
    struct GuiHarness {
        app: Entity<ZqaApp>,
        events: futures::channel::mpsc::UnboundedSender<UiEvent>,
        commands: mpsc::UnboundedReceiver<EngineCommand>,
        cancellations: mpsc::UnboundedReceiver<()>,
    }

    impl GuiHarness {
        fn new(cx: &mut TestAppContext) -> (Self, &mut VisualTestContext) {
            cx.update(gpui_kit::init);
            let (cmd_tx, commands) = mpsc::unbounded_channel();
            let (cancel_tx, cancellations) = mpsc::unbounded_channel();
            let (events, event_rx) = futures::channel::mpsc::unbounded();
            let (app, cx) = cx.add_window_view(|window, cx| {
                ZqaApp::new(
                    cmd_tx,
                    cancel_tx,
                    event_rx,
                    false,
                    Ok(Vec::new()),
                    window,
                    cx,
                )
            });

            (
                Self {
                    app,
                    events,
                    commands,
                    cancellations,
                },
                cx,
            )
        }
    }

    #[gpui_kit::test]
    fn gui_composer_dispatches_and_consumes_engine_output(cx: &mut TestAppContext) {
        let (mut harness, cx) = GuiHarness::new(cx);
        cx.update(|window, cx| {
            window.input("/help", cx);
            window.press("enter", cx);
        });
        cx.run_until_parked();

        let EngineCommand::Dispatch(command) = harness.commands.try_recv().unwrap() else {
            panic!("expected a dispatch command");
        };
        test_eq!(command, "/help");
        harness.app.read_with(cx, |app, _| {
            test_eq!(app.phase, Phase::Running);
        });

        harness
            .events
            .unbounded_send(UiEvent::Engine(EngineEvent::Text {
                message: "Synthetic help output".into(),
            }))
            .unwrap();

        // A restarted task replaces the old one, and a completed task disappears.
        let progress_events = [
            EngineEvent::ProgressStart {
                id: "parse".into(),
                label: "Parsing".into(),
                total: Some(10),
            },
            EngineEvent::ProgressStart {
                id: "embed".into(),
                label: "Embedding".into(),
                total: None,
            },
            EngineEvent::ProgressStart {
                id: "parse".into(),
                label: "Parsing again".into(),
                total: Some(20),
            },
            EngineEvent::ProgressUpdate {
                id: "parse".into(),
                completed: 5,
            },
            EngineEvent::ProgressComplete { id: "embed".into() },
        ];
        for event in progress_events {
            harness
                .events
                .unbounded_send(UiEvent::Engine(event))
                .unwrap();
        }
        cx.run_until_parked();
        harness.app.read_with(cx, |app, _| {
            test_eq!(
                app.progress,
                [ProgressTask {
                    id: "parse".into(),
                    label: "Parsing again".into(),
                    total: Some(20),
                    completed: 5,
                }]
            );
        });

        harness
            .events
            .unbounded_send(UiEvent::Done(Ok(true)))
            .unwrap();
        cx.run_until_parked();
        harness.app.read_with(cx, |app, _| {
            test_eq!(app.phase, Phase::Ready);
            test_eq!(trailing_answer(&app.rows), "Synthetic help output");
            assert!(
                app.progress.is_empty(),
                "progress should clear when the command ends"
            );
        });
    }

    #[gpui_kit::test]
    fn gui_answers_queued_prompts_without_finishing_command(cx: &mut TestAppContext) {
        let (harness, cx) = GuiHarness::new(cx);
        harness.app.update(cx, |app, cx| {
            app.phase = Phase::Running;
            app.rows.push(ChatRow::Answer("Seeded conversation".into()));
            cx.notify();
        });
        let (reply, mut confirmation) = oneshot::channel();
        harness
            .events
            .unbounded_send(UiEvent::Engine(EngineEvent::Confirm {
                message: "Continue?".into(),
                default: true,
                reply,
            }))
            .unwrap();
        let (reply, mut line) = oneshot::channel();
        harness
            .events
            .unbounded_send(UiEvent::Engine(EngineEvent::Line {
                message: Some("What next?".into()),
                reply,
            }))
            .unwrap();
        cx.run_until_parked();

        cx.update(|window, cx| {
            window.click(("prompt-option", 1_usize), cx);
            window.click("send-prompt", cx);
        });
        cx.run_until_parked();
        test_eq!(confirmation.try_recv().unwrap(), false);

        cx.update(|window, cx| {
            window.input("A second response", cx);
            window.press("enter", cx);
        });
        cx.run_until_parked();
        test_eq!(line.try_recv().unwrap(), "A second response");
        harness.app.read_with(cx, |app, _| {
            test_eq!(app.phase, Phase::Running);
            assert!(app.active_prompt.is_none());
            assert!(app.pending_prompts.is_empty());
            test_eq!(trailing_answer(&app.rows), "Seeded conversation");
        });
    }

    #[gpui_kit::test]
    fn gui_long_options_wrap_and_controls_stay_inside_resized_window(cx: &mut TestAppContext) {
        let (harness, cx) = GuiHarness::new(cx);
        harness.app.update(cx, |app, cx| {
            app.phase = Phase::Running;
            cx.notify();
        });
        let (reply, _answer) = oneshot::channel();
        harness.events.unbounded_send(UiEvent::Engine(EngineEvent::Choose {
            message: "Select an approach".into(),
            options: vec![
                "Compare the evidence across these papers, explain where their conclusions disagree, and give me a detailed summary of the assumptions, limitations, and practical implications for a follow-up study. Include enough context that I can understand the trade-offs without opening every paper, and clearly distinguish established results from suggestions that still need experimental validation.\n".into(),
                "A short alternative".into(),
            ],
            default: 0,
            reply,
        })).unwrap();
        cx.run_until_parked();
        let mut heights = Vec::new();

        for (width, height) in [(1080., 760.), (760., 760.), (760., 520.), (1080., 760.)] {
            cx.simulate_resize(size(px(width), px(height)));
            cx.run_until_parked();
            cx.update(|window, cx| {
                window.render_frame(cx);
                let option = window.find(("prompt-option", 0_usize)).bounds();
                heights.push(option.size.height);

                for bounds in [
                    option,
                    window.find("send-prompt").bounds(),
                    window.find("skip-prompt").bounds(),
                ] {
                    assert!(bounds.left() >= super::SIDEBAR_WIDTH);
                    assert!(bounds.right() <= px(width));
                    assert!(bounds.top() >= px(0.));
                    assert!(bounds.bottom() <= px(height));
                }
            });
        }

        assert!(
            heights[0] > px(40.),
            "the long label must span multiple lines"
        );
        assert!(
            heights[1] > heights[0],
            "narrowing the window must reflow the label"
        );
        test_eq!(heights[2], heights[1]);
        test_eq!(heights[3], heights[0]);
    }

    #[gpui_kit::test]
    fn gui_exit_confirms_busy_work_and_waits_for_save(cx: &mut TestAppContext) {
        let (mut harness, cx) = GuiHarness::new(cx);

        for phase in [Phase::Running, Phase::Resuming, Phase::Stopping] {
            harness.app.update(cx, |app, _| app.phase = phase);
            cx.update(|window, cx| {
                harness.app.update(cx, |app, cx| {
                    app.request_exit(window, cx);
                    app.request_exit(window, cx);
                });
            });
            assert!(cx.has_pending_prompt());
            cx.simulate_prompt_answer("Keep working");
            cx.run_until_parked();
            assert!(harness.commands.try_recv().is_err());
            assert!(harness.cancellations.try_recv().is_err());
            harness.app.read_with(cx, |app, _| {
                test_eq!(app.phase, phase);
                assert!(!app.exit_confirmation_open);
            });
        }

        harness.app.update(cx, |app, _| app.phase = Phase::Running);
        cx.update(|window, cx| {
            harness
                .app
                .update(cx, |app, cx| app.request_exit(window, cx));
        });
        cx.simulate_prompt_answer("Quit");
        cx.run_until_parked();
        assert!(matches!(
            harness.commands.try_recv(),
            Ok(EngineCommand::Shutdown)
        ));
        assert!(harness.cancellations.try_recv().is_err());
        harness.events.unbounded_send(UiEvent::Cancelled).unwrap();
        cx.run_until_parked();
        harness.app.read_with(cx, |app, _| {
            test_eq!(app.phase, Phase::ShuttingDown);
        });
        cx.update(|window, cx| {
            harness
                .app
                .update(cx, |app, cx| app.request_exit(window, cx));
        });
        assert!(!cx.has_pending_prompt());
        assert!(harness.commands.try_recv().is_err());

        harness
            .events
            .unbounded_send(UiEvent::ShutdownComplete(Err("Save failed".into())))
            .unwrap();
        cx.run_until_parked();
        harness.app.read_with(cx, |app, _| {
            test_eq!(app.phase, Phase::Ready);
            assert!(matches!(app.rows.last(), Some(ChatRow::Failed(message)) if message.as_ref() == "Save failed"));
        });
        cx.update(|window, cx| {
            harness
                .app
                .update(cx, |app, cx| app.request_exit(window, cx));
        });
        assert!(!cx.has_pending_prompt());
        assert!(matches!(
            harness.commands.try_recv(),
            Ok(EngineCommand::Shutdown)
        ));
    }

    #[gpui_kit::test]
    fn gui_skip_cancels_and_clears_queued_prompts(cx: &mut TestAppContext) {
        let (mut harness, cx) = GuiHarness::new(cx);
        harness.app.update(cx, |app, cx| {
            app.phase = Phase::Running;
            cx.notify();
        });
        let (reply, mut first) = oneshot::channel();
        harness
            .events
            .unbounded_send(UiEvent::Engine(EngineEvent::Confirm {
                message: "Continue?".into(),
                default: true,
                reply,
            }))
            .unwrap();
        let (reply, mut second) = oneshot::channel();
        harness
            .events
            .unbounded_send(UiEvent::Engine(EngineEvent::Secret {
                message: "Secret".into(),
                reply,
            }))
            .unwrap();
        cx.run_until_parked();

        cx.update(|window, cx| window.click("skip-prompt", cx));
        cx.run_until_parked();
        harness.cancellations.try_recv().unwrap();
        harness.app.read_with(cx, |app, _| {
            test_eq!(app.phase, Phase::Stopping);
            assert!(app.active_prompt.is_none());
            assert!(app.pending_prompts.is_empty());
        });
        test_eq!(first.try_recv(), Err(oneshot::error::TryRecvError::Closed));
        test_eq!(second.try_recv(), Err(oneshot::error::TryRecvError::Closed));

        harness.events.unbounded_send(UiEvent::Cancelled).unwrap();
        cx.run_until_parked();
        harness.app.read_with(cx, |app, _| {
            test_eq!(app.phase, Phase::Ready);
        });
    }

    /// The text of the trailing answer row, for assertions.
    fn trailing_answer(rows: &[ChatRow]) -> String {
        match rows.last() {
            Some(ChatRow::Answer(text)) => text.clone(),
            other => panic!("expected trailing answer, got {other:?}"),
        }
    }

    /// The lines of the trailing status row, for assertions.
    fn trailing_status_lines(rows: &[ChatRow]) -> Vec<&str> {
        match rows.last() {
            Some(ChatRow::Status(lines)) => lines.lines().collect(),
            other => panic!("expected trailing status row, got {other:?}"),
        }
    }

    #[test]
    fn stdout_chunks_merge_into_one_answer_row() {
        let mut rows = Vec::new();
        ZqaApp::fold_stdout(&mut rows, "Hello, ");
        ZqaApp::fold_stdout(&mut rows, "world!\n");

        assert_eq!(rows.len(), 1);
        assert_eq!(trailing_answer(&rows), "Hello, world!\n");
    }

    #[test]
    fn assistant_history_text_becomes_an_answer_row() {
        let item = ChatHistoryItem {
            role: MessageRole::Assistant,
            content: vec![ChatHistoryContent::Text("Assistant response".into())],
        };

        let rows = ChatRows::from_history(std::slice::from_ref(&item)).0;

        assert_eq!(trailing_answer(&rows), "Assistant response");
    }

    #[test]
    fn tool_request_and_response_in_separate_items_become_one_row() {
        let history = [
            ChatHistoryItem {
                role: MessageRole::Assistant,
                content: vec![ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                    id: "call-1".into(),
                    tool_name: "search".into(),
                    args: serde_json::json!({"query": "attention"}),
                })],
            },
            ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                    id: "call-1".into(),
                    tool_name: "search".into(),
                    result: serde_json::json!({"matches": 3}),
                })],
            },
        ];

        let rows = ChatRows::from_history(&history).0;

        assert_eq!(rows.len(), 1);
        let ChatRow::ToolCall(call) = &rows[0] else {
            panic!("expected a tool call row, got {:?}", rows[0]);
        };
        assert_eq!(call.name.as_ref(), "search");
        assert_eq!(call.args, serde_json::json!({"query": "attention"}));
        assert_eq!(call.response, Some(Ok(serde_json::json!({"matches": 3}))));
        assert!(!call.expanded);
    }

    #[test]
    fn tool_responses_complete_their_own_calls() {
        let mut rows = Vec::new();
        ZqaApp::start_tool_call(
            &mut rows,
            "a".into(),
            "search".into(),
            serde_json::json!({}),
        );
        ZqaApp::start_tool_call(
            &mut rows,
            "b".into(),
            "summarize".into(),
            serde_json::json!({}),
        );

        // Parallel calls can finish out of order, and a response can arrive without its call.
        ZqaApp::finish_tool_call(
            &mut rows,
            "b".into(),
            "summarize".into(),
            Err("failed".into()),
        );
        ZqaApp::finish_tool_call(
            &mut rows,
            "a".into(),
            "search".into(),
            Ok(serde_json::json!(1)),
        );
        ZqaApp::finish_tool_call(
            &mut rows,
            "c".into(),
            "search".into(),
            Ok(serde_json::json!(2)),
        );

        let responses: Vec<_> = rows
            .iter()
            .map(|row| match row {
                ChatRow::ToolCall(call) => (call.id.as_ref(), call.response.clone()),
                other => panic!("expected a tool call row, got {other:?}"),
            })
            .collect();
        assert_eq!(
            responses,
            [
                ("a", Some(Ok(serde_json::json!(1)))),
                ("b", Some(Err("failed".to_string()))),
                ("c", Some(Ok(serde_json::json!(2)))),
            ]
        );
    }

    #[test]
    fn blank_stdout_before_any_answer_is_dropped() {
        let mut rows = Vec::new();
        ZqaApp::fold_stdout(&mut rows, "\n");

        assert!(rows.is_empty());
    }

    #[test]
    fn styled_text_becomes_a_styled_row_unless_it_is_all_secondary() {
        let span = |text: &str, style| StyledSpan {
            text: text.into(),
            style,
        };
        let mut rows = Vec::new();

        ZqaApp::fold_styled(
            &mut rows,
            &[
                span("Usage:\n", TextStyle::Heading),
                span("/help\n\n", TextStyle::Plain),
                span("Commands:\n", TextStyle::Heading),
            ],
        );
        ZqaApp::fold_styled(&mut rows, &[span("Search took 1s\n", TextStyle::Secondary)]);

        let [ChatRow::Styled(text, styles), ChatRow::Status(status)] = rows.as_slice() else {
            panic!("expected a styled row and a status row, got {rows:?}");
        };
        test_eq!(text, "Usage:\n/help\n\nCommands:");
        let ranges: Vec<_> = styles.iter().map(|(range, _)| range.clone()).collect();
        test_eq!(ranges, [0..7, 14..23]);
        assert!(
            styles
                .iter()
                .all(|(_, style)| matches!(style, TextStyle::Heading))
        );
        test_eq!(status, "Search took 1s");
    }

    #[test]
    fn consecutive_stderr_lines_merge_into_one_status_row() {
        let mut rows = Vec::new();
        ZqaApp::fold_stderr(&mut rows, "Vector search completed in 1.20s\n");
        ZqaApp::fold_stderr(&mut rows, "Final draft completed in 3.40s\n");

        assert_eq!(rows.len(), 1);
        assert_eq!(
            trailing_status_lines(&rows),
            [
                "Vector search completed in 1.20s",
                "Final draft completed in 3.40s"
            ]
        );
    }

    #[test]
    fn stdout_between_stderr_lines_preserves_order() {
        let mut rows = Vec::new();
        ZqaApp::fold_stderr(&mut rows, "Vector search completed in 1.20s\n");
        ZqaApp::fold_stdout(&mut rows, "an answer\n");
        ZqaApp::fold_stderr(&mut rows, "Final draft completed in 3.40s\n");

        assert_eq!(rows.len(), 3);
        assert_eq!(
            trailing_status_lines(&rows),
            ["Final draft completed in 3.40s"]
        );
    }

    #[test]
    fn blank_stderr_lines_are_dropped() {
        let mut rows = Vec::new();
        ZqaApp::fold_stderr(&mut rows, "\n\n");

        assert!(rows.is_empty());
    }

    #[test]
    fn only_the_ready_phase_accepts_commands() {
        assert!(Phase::Ready.accepts_commands());
        assert!(!Phase::Running.accepts_commands());
        assert!(!Phase::Resuming.accepts_commands());
        assert!(!Phase::Stopping.accepts_commands());
        assert!(!Phase::ShuttingDown.accepts_commands());
        assert!(!Phase::Ended.accepts_commands());
    }

    #[test]
    fn new_is_recorded_as_a_reset_not_a_user_turn() {
        let mut rows = Vec::new();
        let mut pending = false;
        ZqaApp::record_command(&mut rows, &mut pending, "/new");

        assert!(rows.is_empty());
        assert!(pending);
    }

    #[test]
    fn ordinary_commands_are_echoed_as_user_turns() {
        let mut rows = Vec::new();
        let mut pending = false;
        ZqaApp::record_command(&mut rows, &mut pending, "/stats");

        assert_eq!(rows.len(), 1);
        assert!(matches!(rows[0], ChatRow::User(_)));
        assert!(!pending);
    }

    #[test]
    fn deferred_reset_clears_the_transcript_on_success() {
        let mut rows = vec![ChatRow::User("hi".into()), ChatRow::Answer("hello".into())];
        let mut pending = true;
        ZqaApp::finish_command(&mut rows, &mut pending, Ok(true));

        assert!(rows.is_empty());
        assert!(!pending);
    }

    #[test]
    fn failed_reset_keeps_the_transcript_and_reports_the_error() {
        let mut rows = vec![ChatRow::User("hi".into()), ChatRow::Answer("hello".into())];
        let mut pending = true;
        ZqaApp::finish_command(&mut rows, &mut pending, Err("save failed".into()));

        assert_eq!(rows.len(), 3);
        assert!(matches!(rows.last(), Some(ChatRow::Failed(_))));
        assert!(!pending);
    }
}
