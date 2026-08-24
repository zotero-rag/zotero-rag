//! Native GUI front-end for zqa, built on GPUI / gpui-component.
//!
//! This is currently a bridge spike: it proves that the tokio-based zqa pipeline can be
//! driven from GPUI's event loop (see [`bridge`]) and renders streamed output in a window.
//! The UI is intentionally minimal.

mod bridge;

use bridge::{UiEvent, spawn_engine};
use futures::StreamExt;
use futures::channel::mpsc::UnboundedReceiver;
use gpui::prelude::*;
use gpui::{
    App, Entity, SharedString, Subscription, Window, WindowBounds, WindowOptions, px, size,
};
use gpui_component::{
    Root,
    button::{Button, ButtonVariants},
    input::{Input, InputEvent, InputState},
    v_flex,
};
use gpui_component_assets::Assets;
use tokio::sync::mpsc::UnboundedSender;

/// The single-window application view.
struct ZqaApp {
    /// The query/command input box state.
    input_state: Entity<InputState>,
    /// Accumulated stdout from dispatched commands.
    output: String,
    /// The most recent stderr line (status/timing/warning).
    status: SharedString,
    /// Whether a command is currently in flight.
    running: bool,
    /// Whether the engine thread is still accepting commands. Cleared when the engine exits
    /// (e.g. after `/quit`) so the UI stops accepting input instead of hanging on a dead channel.
    engine_alive: bool,
    /// Channel carrying commands to the engine thread.
    cmd_tx: UnboundedSender<String>,
    /// Channel signalling the engine to cancel the in-flight command.
    cancel_tx: UnboundedSender<()>,
    /// Kept alive so the input subscription is not dropped.
    _subscriptions: Vec<Subscription>,
}

impl ZqaApp {
    fn new(
        cmd_tx: UnboundedSender<String>,
        cancel_tx: UnboundedSender<()>,
        event_rx: UnboundedReceiver<UiEvent>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let input_state =
            cx.new(|cx| InputState::new(window, cx).placeholder("Ask a question, or type /help"));

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

        // Drain engine output on GPUI's executor and fold it into the view.
        cx.spawn(async move |this, cx| {
            let mut event_rx = event_rx;
            while let Some(event) = event_rx.next().await {
                let update = this.update(cx, |app, cx| {
                    match event {
                        UiEvent::Stdout(text) => app.append_output(&text),
                        UiEvent::Stderr(text) => app.status = text.into(),
                        UiEvent::Done(result) => {
                            app.running = false;
                            if let Err(msg) = result {
                                app.status = format!("error: {msg}").into();
                            }
                        }
                        UiEvent::Cancelled => {
                            app.running = false;
                            app.append_output("\n(cancelled)\n");
                            app.status = "Cancelled".into();
                        }
                    }
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
                app.engine_alive = false;
                app.running = false;
                app.status = "Session ended".into();
                cx.notify();
            });
        })
        .detach();

        Self {
            input_state,
            output: String::new(),
            status: SharedString::default(),
            running: false,
            engine_alive: true,
            cmd_tx,
            cancel_tx,
            _subscriptions: vec![subscription],
        }
    }

    /// Append text to the output pane.
    ///
    /// TODO(ZOT-220): output is a single flat `String` re-cloned and re-laid-out on every
    /// repaint, which is O(n) per repaint and quadratic over a session. Replace with a
    /// virtualized list of turns plus a live streaming block so per-frame cost is O(visible)
    /// and scrollback is unbounded without a cap.
    fn append_output(&mut self, text: &str) {
        self.output.push_str(text);
    }

    /// Read the input box, send its contents to the engine thread, and reset the input.
    fn submit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let command = self.input_state.read(cx).value().trim().to_string();
        if command.is_empty() || self.running || !self.engine_alive {
            return;
        }

        self.append_output(&format!("\n>>> {command}\n"));
        self.running = true;
        let _ = self.cmd_tx.send(command);
        self.input_state
            .update(cx, |state, cx| state.set_value("", window, cx));
        cx.notify();
    }

    /// Ask the engine to cancel the in-flight command.
    ///
    /// `running` is left set; it is cleared when the engine confirms with
    /// [`UiEvent::Cancelled`] (or [`UiEvent::Done`] if the command finished first).
    fn cancel(&mut self, cx: &mut Context<Self>) {
        if !self.running {
            return;
        }
        let _ = self.cancel_tx.send(());
        self.status = "Stopping...".into();
        cx.notify();
    }
}

impl Render for ZqaApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .p_4()
            .gap_3()
            .size_full()
            .child(
                gpui::div()
                    .id("output")
                    .flex_1()
                    .overflow_y_scroll()
                    .whitespace_normal()
                    .child(self.output.clone()),
            )
            .child(self.status.clone())
            .child(if self.engine_alive {
                v_flex()
                    .gap_2()
                    .child(Input::new(&self.input_state))
                    .child(if self.running {
                        Button::new("stop")
                            .danger()
                            .label("Stop")
                            .on_click(cx.listener(|this, _, _window, cx| this.cancel(cx)))
                    } else {
                        Button::new("submit")
                            .primary()
                            .label("Send")
                            .on_click(cx.listener(|this, _, window, cx| this.submit(window, cx)))
                    })
                    .into_any_element()
            } else {
                gpui::div().child("Session ended.").into_any_element()
            })
    }
}

fn main() {
    // Resolve the LanceDB location the same way the CLI does, before spawning any threads.
    // Otherwise the engine thread opens the default relative path instead of the state-dir
    // database and every query reports a missing table.
    zqa::set_default_lancedb_uri();

    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (cancel_tx, cancel_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let (event_tx, event_rx) = futures::channel::mpsc::unbounded::<UiEvent>();
    spawn_engine(cmd_rx, cancel_rx, event_tx);

    let app = gpui_platform::application().with_assets(Assets);
    app.run(move |cx: &mut App| {
        gpui_component::init(cx);

        let window_options = WindowOptions {
            window_bounds: Some(WindowBounds::centered(size(px(900.), px(680.)), cx)),
            ..Default::default()
        };

        cx.spawn(async move |cx| {
            cx.open_window(window_options, |window, cx| {
                let view = cx.new(|cx| ZqaApp::new(cmd_tx, cancel_tx, event_rx, window, cx));
                cx.new(|cx| Root::new(view, window, cx))
            })
            .expect("failed to open window");
        })
        .detach();
    });
}
