//! The runtime bridge between GPUI (which owns the main thread and its own executor)
//! and the tokio-based zqa pipeline.
//!
//! [`Session`](zqa::session::Session) is not `Send`, so it cannot be moved into GPUI's
//! executor or a tokio task. Instead [`spawn_engine`] dedicates one OS thread that owns
//! a tokio runtime and the session: it receives commands over a `tokio::sync::mpsc`
//! channel and runs each one on that runtime. Each dispatch is raced against a separate
//! cancel channel with `tokio::select!`, so a user can stop an in-flight command: the
//! cancel branch drops the dispatch future, which aborts the request in progress
//! (async Rust is cancel-on-drop). Output produced by the handlers is streamed back to
//! the GPUI side as [`UiEvent`]s through a `futures::mpsc` channel, whose sender is
//! wrapped in a [`ChannelWriter`] that plays the role of the session's stdout/stderr.

use std::sync::Arc;
use std::thread;

use futures::channel::mpsc::UnboundedSender;
use tokio::sync::mpsc::{self, UnboundedReceiver};
use zqa::io::EngineEvent;
use zqa::session::Session;
use zqa::state::SavedChatHistory;

/// A request sent from the UI to the engine thread.
#[derive(Debug)]
pub enum EngineCommand {
    /// Dispatch a CLI-style command or query.
    Dispatch(String),
    /// Resume a saved conversation without an interactive prompt.
    ResumeConversation(Arc<SavedChatHistory>),
}

/// A single piece of output streamed from the engine thread to the UI.
#[derive(Debug)]
pub enum UiEvent {
    /// An event from the agent engine.
    Engine(EngineEvent),
    /// A command finished. Carries the dispatch result: `Ok(keep_running)` or an
    /// error message.
    Done(Result<bool, String>),
    /// A saved conversation resume attempt finished.
    ConversationResumed(Result<Arc<SavedChatHistory>, String>),
    /// The in-flight command was cancelled by the user before it finished.
    Cancelled,
}

impl From<EngineEvent> for UiEvent {
    fn from(event: EngineEvent) -> Self {
        UiEvent::Engine(event)
    }
}

/// Spawn the engine thread.
///
/// The thread builds a [`Session`] from the loaded config and then loops, dispatching each
/// command received on `cmd_rx` and streaming output to `event_tx`. Each dispatch is raced
/// against `cancel_rx`; a value on `cancel_rx` aborts the in-flight command. It exits when
/// the command channel is closed (all senders dropped) or a command returns `Ok(false)`.
///
/// # Arguments
///
/// * `cmd_rx` - Receiver for requests sent by the UI.
/// * `cancel_rx` - Receiver signalled by the UI to cancel the in-flight command.
/// * `event_tx` - Sender used to stream [`UiEvent`]s back to the UI.
pub fn spawn_engine(
    mut cmd_rx: UnboundedReceiver<EngineCommand>,
    mut cancel_rx: UnboundedReceiver<()>,
    event_tx: UnboundedSender<UiEvent>,
) {
    thread::Builder::new()
        .name("zqa-engine".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ =
                        event_tx.unbounded_send(UiEvent::Done(Err(format!("runtime error: {e}"))));
                    return;
                }
            };

            let config = match zqa::load_config() {
                Ok(config) => config,
                Err(e) => {
                    let _ =
                        event_tx.unbounded_send(UiEvent::Done(Err(format!("config error: {e}"))));
                    return;
                }
            };

            let (engine_tx, mut engine_rx) = mpsc::channel(256);
            let mut session = match Session::new(config, Some(engine_tx)) {
                Ok(session) => session,
                Err(e) => {
                    let _ =
                        event_tx.unbounded_send(UiEvent::Done(Err(format!("session error: {e}"))));
                    return;
                }
            };

            runtime.block_on(async move {
                while let Some(command) = cmd_rx.recv().await {
                    // Drop any cancel signals that arrived while idle, so a late click on a
                    // previous command can't abort this fresh one.
                    while cancel_rx.try_recv().is_ok() {}

                    match command {
                        EngineCommand::Dispatch(command) => {
                            // `None` means the dispatch was cancelled; dropping the future here
                            // aborts the request in flight.
                            //
                            // TODO(ZOT-219): cancellation only drops this future. Detached tasks
                            // the handlers spawn (e.g. background title generation in
                            // `handle_query_cmd`) are not cancelled and run to completion. Fully
                            // cancelling them needs the core loop's cancellation support tracked
                            // in ZOT-219.
                            let result: Option<Result<bool, String>> = {
                                let dispatch = session.dispatch(&command);
                                tokio::pin!(dispatch);

                                loop {
                                    tokio::select! {
                                        // Dismissing a prompt also drops its reply sender. Prefer the
                                        // cancel signal to the resulting receive error from dispatch.
                                        biased;
                                        Some(()) = cancel_rx.recv() => {
                                            break None;
                                        }
                                        result = &mut dispatch => {
                                            break Some(result.map_err(|e| e.to_string()));
                                        }
                                        event = engine_rx.recv() => {
                                            match event {
                                                Some(event) => {
                                                    if event_tx.unbounded_send(UiEvent::Engine(event)).is_err() {
                                                        // The UI has gone away; stop the driver
                                                        return;
                                                    }
                                                }
                                                None => {
                                                    break Some(Err("The engine event channel closed".into()));
                                                }
                                            }
                                        },
                                    }
                                }
                            };

                            // Forward queued output before announcing completion or cancellation.
                            while let Ok(event) = engine_rx.try_recv() {
                                if event_tx.unbounded_send(UiEvent::Engine(event)).is_err() {
                                    return;
                                }
                            }

                            match result {
                                Some(result) => {
                                    // Only a deliberate exit (`/quit` returns `Ok(false)`) stops
                                    // the engine. A command error keeps the engine alive so the UI
                                    // stays usable; exiting on errors would strand the window with
                                    // a dead channel and a permanently "running" state.
                                    let should_exit = matches!(result, Ok(false));
                                    let _ = event_tx.unbounded_send(UiEvent::Done(result));
                                    if should_exit {
                                        break;
                                    }
                                }
                                None => {
                                    let _ = event_tx.unbounded_send(UiEvent::Cancelled);
                                }
                            }
                        }
                        EngineCommand::ResumeConversation(conversation) => {
                            let result = session
                                .resume_conversation(&conversation)
                                .await
                                .map(|()| conversation)
                                .map_err(|error| error.to_string());
                            let _ = event_tx.unbounded_send(UiEvent::ConversationResumed(result));
                        }
                    }
                }
            });
        })
        .expect("failed to spawn zqa engine thread");
}
