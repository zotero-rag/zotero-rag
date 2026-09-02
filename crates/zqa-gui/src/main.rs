//! Native GUI front-end for zqa, built on GPUI / gpui-component.
//!
//! The window is a chat-style harness over the same engine the CLI drives:
//! [`bridge`] runs the tokio-based pipeline on a dedicated thread and streams output
//! back as [`UiEvent`]s, which this view folds into a transcript of turns.
//!
//! Layout follows the usual desktop AI-harness pattern: a translucent sidebar
//! (window vibrancy shows through on macOS) with session and library commands, and an
//! opaque main pane holding the conversation. There is no system title bar; the top
//! strip of each pane drags the window instead.

mod bridge;

use std::collections::HashMap;
use std::sync::Arc;

use bridge::{EngineCommand, UiEvent, spawn_engine};
use futures::StreamExt;
use futures::channel::mpsc::UnboundedReceiver;
use gpui::prelude::*;
use gpui::{
    AnyElement, App, Context, Decorations, Div, Entity, Focusable as _, FontWeight, IntoElement,
    MouseButton, Pixels, Rems, Render, ScrollHandle, SharedString, Stateful, Subscription, Window,
    WindowBackgroundAppearance, WindowBounds, div, px, rems, size, transparent_black,
};
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::input::{InputEvent, Textarea, TextareaState};
use gpui_component::spinner::Spinner;
use gpui_component::{
    ActiveTheme as _, Icon, IconName, InteractiveElementExt as _, Root, Sizable as _,
    TITLE_BAR_HEIGHT, Theme, ThemeMode, TitleBar, h_flex, v_flex,
};
use gpui_component_assets::Assets;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use zqa::state::SavedChatHistory;

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
    /// A tool call name and request/response pair from the engine.
    /// TODO: This can look a little ugly since it's typically JSON request/responses. We should
    /// consider a better formatting solution right here, maybe with a trait.
    ToolCall((SharedString, Value, Value)),
    /// Answer-style stdout. Grows in place while a command streams.
    Answer(String),
    /// Engine status lines (timings, warnings) written to stderr.
    Status(String),
    /// A command that failed.
    Failed(SharedString),
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
                        if let Some((tool_name, args)) = tool_calls.remove(response.id.as_str()) {
                            rows.push(ChatRow::ToolCall((
                                tool_name,
                                args,
                                response.result.clone(),
                            )));
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
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Ready,
    Running,
    Resuming,
    Stopping,
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
    /// Transcript blocks, oldest first.
    rows: Vec<ChatRow>,
    /// Saved conversations loaded from the zqa state directory.
    conversation_history: Result<Vec<Arc<SavedChatHistory>>, zqa::state::StateError>,
    /// What the engine thread is doing right now.
    phase: Phase,
    /// Whether a `/new` reset is in flight; the transcript clears only when the
    /// engine confirms it, so a failed save cannot lose the visible copy.
    pending_reset: bool,
    /// Which pane the main area shows.
    pane: Pane,
    /// Whether the dark theme is active; toggled from the header.
    dark_theme: bool,
    /// Whether a left press on a drag surface is pending; consumed by the next
    /// mouse move to start a window drag.
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
        cx.spawn(async move |this, cx| {
            let mut event_rx = event_rx;
            while let Some(event) = event_rx.next().await {
                let update = this.update(cx, |app, cx| {
                    match event {
                        UiEvent::Stdout(text) => Self::fold_stdout(&mut app.rows, &text),
                        UiEvent::Stderr(text) => Self::fold_stderr(&mut app.rows, &text),
                        UiEvent::Done(result) => {
                            app.phase = Phase::Ready;
                            let refresh_history = app.pending_reset && result.is_ok();
                            Self::finish_command(&mut app.rows, &mut app.pending_reset, result);
                            if refresh_history {
                                app.conversation_history = Self::load_conversation_history();
                            }
                        }
                        UiEvent::ConversationResumed(result) => {
                            app.phase = Phase::Ready;
                            match result {
                                Ok(conversation) => {
                                    app.rows = ChatRows::from(conversation.as_ref()).0;
                                    app.pane = Pane::Chat;
                                    app.conversation_history = Self::load_conversation_history();
                                }
                                Err(message) => app.rows.push(ChatRow::Failed(message.into())),
                            }
                        }
                        UiEvent::Cancelled => {
                            app.phase = Phase::Ready;
                            // A cancelled reset may or may not have taken effect
                            // engine-side; keeping the transcript is the conservative
                            // reading.
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
                app.phase = Phase::Ended;
                cx.notify();
            });
        })
        .detach();

        Self {
            input_state,
            rows: Vec::new(),
            conversation_history: Self::load_conversation_history(),
            phase: Phase::Ready,
            pending_reset: false,
            pane: Pane::Chat,
            dark_theme,
            drag_armed: false,
            scroll_handle: ScrollHandle::new(),
            cmd_tx,
            cancel_tx,
            _subscriptions: vec![subscription],
        }
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
    /// Chunks append to the trailing answer row so one command renders as a single
    /// bubble; a whitespace-only chunk before any answer text is dropped rather than
    /// opening an empty row.
    fn fold_stdout(rows: &mut Vec<ChatRow>, text: &str) {
        if let Some(ChatRow::Answer(existing)) = rows.last_mut() {
            existing.push_str(text);
            return;
        }
        if !text.trim().is_empty() {
            rows.push(ChatRow::Answer(text.to_string()));
        }
    }

    /// Fold stderr lines (timings, warnings) into the transcript.
    ///
    /// Consecutive lines merge into one muted block; stdout arriving between them
    /// splits them again, which preserves chronological order.
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
    /// `/new` is a reset rather than a conversational turn: it is not echoed as a
    /// user bubble, and it only arms the deferred reset that [`Self::finish_command`]
    /// commits or drops.
    fn record_command(rows: &mut Vec<ChatRow>, pending_reset: &mut bool, command: &str) {
        if command == "/new" {
            *pending_reset = true;
        } else {
            rows.push(ChatRow::User(command.into()));
        }
    }

    /// Apply a finished command's result to the transcript and any pending reset.
    ///
    /// A pending `/new` reset is committed only on success: if the engine could not
    /// save the old conversation, the failure surfaces as a [`ChatRow::Failed`] next
    /// to the retained transcript instead of silently discarding it.
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
    /// Whether the command was dispatched; false when it is empty or the engine is
    /// busy or ended.
    fn dispatch(&mut self, command: String, cx: &mut Context<Self>) -> bool {
        let command = command.trim().to_string();
        if command.is_empty() || !self.phase.accepts_commands() {
            return false;
        }

        // `/new` starts a fresh engine conversation, so the visible transcript restarts
        // with it. The reset is deferred until the engine confirms it, and the command
        // itself is not echoed; see [`record_command`] and [`finish_command`].
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
    /// `phase` moves to `Stopping` immediately so further clicks and submissions are
    /// rejected, and returns to `Ready` only when the engine confirms with
    /// [`UiEvent::Cancelled`] (or [`UiEvent::Done`] if the command finished first).
    fn request_stop(&mut self, cx: &mut Context<Self>) {
        if self.phase != Phase::Running {
            return;
        }
        let _ = self.cancel_tx.send(());
        self.phase = Phase::Stopping;
        cx.notify();
    }

    /// Make a pane's top strip drag the window, with double-click to zoom.
    ///
    /// The press is armed on mouse down and consumed by the next mouse move, mirroring
    /// gpui-component's `TitleBar`: starting the drag directly from mouse down makes
    /// macOS delay clicks while it disambiguates double clicks.
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
                        IconName::Plus,
                        "New chat",
                        SidebarAction::Run("/new"),
                        false,
                        cx,
                    ))
                    .child(self.sidebar_item(
                        "sb-help",
                        IconName::Info,
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
                IconName::Settings,
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
        icon: IconName,
        label: &'static str,
        action: SidebarAction,
        selected: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        // The settings toggle is pure UI and works in any engine state; commands need
        // a live engine.
        let enabled =
            matches!(&action, SidebarAction::ToggleSettings) || self.phase.accepts_commands();

        div()
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
            })
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
                .child(self.render_dock(cx))
                .into_any_element(),
            Pane::Settings => self.render_settings(cx),
        };

        v_flex()
            .flex_1()
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
            Phase::Running | Phase::Resuming | Phase::Stopping
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
            .children(self.rows.iter().filter_map(|row| self.render_row(row, cx)))
            .when(busy, |this| {
                this.child(
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
                                    _ => "Working...",
                                }),
                        ),
                )
            });

        scroll()
            .child(div().w_full().flex().justify_center().child(column))
            .into_any_element()
    }

    /// Render one transcript block.
    fn render_row(&self, row: &ChatRow, cx: &Context<Self>) -> Option<AnyElement> {
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
            ChatRow::ToolCall((tool_name, request, response)) => {
                let request =
                    serde_json::to_string_pretty(request).unwrap_or_else(|_| request.to_string());
                let response =
                    serde_json::to_string_pretty(response).unwrap_or_else(|_| response.to_string());

                v_flex()
                    .w_full()
                    .rounded_lg()
                    .border_1()
                    .border_color(cx.theme().border)
                    .bg(cx.theme().secondary.opacity(0.35))
                    .p_3()
                    .gap_2()
                    .child(
                        div()
                            .text_size(px(12.))
                            .font_weight(FontWeight::MEDIUM)
                            .child(format!("Tool: {tool_name}")),
                    )
                    .child(
                        div()
                            .text_size(px(11.))
                            .font_family(cx.theme().mono_font_family.clone())
                            .text_color(cx.theme().muted_foreground)
                            .whitespace_normal()
                            .child(format!("Request:\n{request}\n\nResponse:\n{response}")),
                    )
                    .into_any_element()
            }
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
        let resuming = self.phase == Phase::Resuming;
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
                        .child(Textarea::new(&self.input_state).appearance(false))
                        .child(h_flex().justify_end().child(action)),
                ),
            )
            .into_any_element()
    }
}

impl Render for ZqaApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        h_flex()
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

    let app = gpui_platform::application().with_assets(Assets);
    app.run(move |cx: &mut App| {
        gpui_component::init(cx);

        // `init` always loads the light theme first; honor the OS appearance until the
        // user toggles it in-app. Must run after `init`, which overwrites the global.
        let dark_theme = matches!(
            cx.window_appearance(),
            gpui::WindowAppearance::Dark | gpui::WindowAppearance::VibrantDark
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
                let view =
                    cx.new(|cx| ZqaApp::new(cmd_tx, cancel_tx, event_rx, dark_theme, window, cx));
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
    use zqa_rag::llm::base::{
        ChatHistoryContent, ChatHistoryItem, MessageRole, ToolCallRequest, ToolCallResponse,
    };

    use super::{ChatRow, ChatRows, Phase, ZqaApp};

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
        let ChatRow::ToolCall((tool_name, args, result)) = &rows[0] else {
            panic!("expected a tool call row, got {:?}", rows[0]);
        };
        assert_eq!(tool_name.as_ref(), "search");
        assert_eq!(args, &serde_json::json!({"query": "attention"}));
        assert_eq!(result, &serde_json::json!({"matches": 3}));
    }

    #[test]
    fn blank_stdout_before_any_answer_is_dropped() {
        let mut rows = Vec::new();
        ZqaApp::fold_stdout(&mut rows, "\n");

        assert!(rows.is_empty());
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
