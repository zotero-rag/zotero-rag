//! The runtime bridge between GPUI (which owns the main thread and its own executor)
//! and the tokio-based zqa pipeline.
//!
//! [`spawn_engine`] dedicates one OS thread to a tokio runtime and [`Session`]. It runs the startup
//! steps, then receives commands sequentially and forwards typed engine events to GPUI. Dispatch can be cancelled
//! by dropping its future; shutdown itself cannot be cancelled and acknowledges a successful
//! save before exiting. A failed save leaves the engine available for retry.

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
    /// Save the conversation and stop the engine only if saving succeeds.
    Shutdown,
}

/// A single piece of output streamed from the engine thread to the UI.
#[derive(Debug)]
pub enum UiEvent {
    /// An event from the agent engine.
    Engine(EngineEvent),
    /// A command finished. Carries the dispatch result: `Ok(keep_running)` or an
    /// error message.
    Done(Result<bool, String>),
    /// A saved conversation resume attempt finished, either from the sidebar or from a typed
    /// `/resume` that switched conversations. For a typed `/resume`, this is sent in place of
    /// [`UiEvent::Done`].
    ConversationResumed(Result<Arc<SavedChatHistory>, String>),
    /// The in-flight command was cancelled by the user before it finished.
    Cancelled,
    /// Shutdown finished saving, or failed with the engine still available for retry.
    ShutdownComplete(Result<(), String>),
}

impl From<EngineEvent> for UiEvent {
    fn from(event: EngineEvent) -> Self {
        UiEvent::Engine(event)
    }
}

/// Run `future` while forwarding the engine events it sends to the UI.
///
/// Events still queued when `future` finishes are forwarded before this returns, so the UI sees
/// all of its output before any completion event the caller sends. Prompts are answered by the UI
/// like any other forwarded event; nothing here cancels `future`.
///
/// # Arguments
///
/// * `future` - The engine work to run, such as resuming a conversation.
/// * `engine_rx` - The receiver for events sent by `future`.
/// * `event_tx` - Where the events are forwarded.
///
/// # Returns
///
/// The output of `future`, or `None` if the UI has gone away. `future` is dropped unfinished in
/// that case, and the engine should stop.
async fn forward_events<T>(
    future: impl Future<Output = T>,
    engine_rx: &mut mpsc::Receiver<EngineEvent>,
    event_tx: &UnboundedSender<UiEvent>,
) -> Option<T> {
    tokio::pin!(future);

    let result = loop {
        tokio::select! {
            result = &mut future => break result,
            Some(event) = engine_rx.recv() => {
                event_tx.unbounded_send(UiEvent::Engine(event)).ok()?;
            }
        }
    };

    while let Ok(event) = engine_rx.try_recv() {
        event_tx.unbounded_send(UiEvent::Engine(event)).ok()?;
    }

    Some(result)
}

/// Spawn the engine thread.
///
/// The thread builds a [`Session`] from the loaded config, runs the startup steps (see
/// [`Session::start`]), and then loops, dispatching each
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
                // Startup isn't a command the UI sent, so it ends without a `Done`. A failure shows
                // as an error row and the engine continues with the config it has.
                let start = session.start();
                match forward_events(start, &mut engine_rx, &event_tx).await {
                    None => return,
                    Some(Ok(true)) => {}
                    Some(Ok(false)) => {
                        // The user chose to quit, and there is nothing to save yet.
                        let _ = event_tx.unbounded_send(UiEvent::ShutdownComplete(Ok(())));
                        return;
                    }
                    Some(Err(e)) => {
                        let message = format!("Startup failed: {e}\n");
                        let _ = event_tx.unbounded_send(UiEvent::Engine(EngineEvent::Error { message }));
                    }
                }

                let mut pending_command = None;

                loop {
                    let command = match pending_command.take() {
                        Some(command) => command,
                        None => match cmd_rx.recv().await {
                            Some(command) => command,
                            None => break,
                        },
                    };

                    // A late stop click must not cancel the next command.
                    while cancel_rx.try_recv().is_ok() {}

                    let (command, shutting_down) = match command {
                        EngineCommand::Dispatch(command) => (command, false),
                        EngineCommand::Shutdown => ("/quit".to_string(), true),
                        EngineCommand::ResumeConversation(conversation) => {
                            let resume = session.resume_conversation(&conversation);
                            let Some(result) = forward_events(resume, &mut engine_rx, &event_tx).await else {
                                return;
                            };

                            let result = result.map(|()| conversation).map_err(|error| error.to_string());
                            let _ = event_tx.unbounded_send(UiEvent::ConversationResumed(result));

                            continue;
                        }
                    };

                    // A typed `/resume` swaps the conversation inside the engine, so the
                    // UI only learns of it by comparing conversation IDs after it finishes.
                    // It can also finish without switching, for example when there are no
                    // saved conversations, and then the UI keeps its transcript.
                    let before_resume = (command == "/resume").then(|| session.conversation_id());
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
                                Some(()) = cancel_rx.recv(), if !shutting_down => {
                                    break None;
                                }
                                Some(command) = cmd_rx.recv(), if !shutting_down && pending_command.is_none() => {
                                    let shutdown_requested = matches!(command, EngineCommand::Shutdown);
                                    pending_command = Some(command);

                                    if shutdown_requested {
                                        break None;
                                    }
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

                    if shutting_down {
                        let result = result.expect("shutdown cannot be cancelled").map(|_| ());
                        let saved = result.is_ok();
                        let _ = event_tx.unbounded_send(UiEvent::ShutdownComplete(result));

                        if saved {
                            break;
                        }

                        continue;
                    }

                    match result {
                        Some(result) => {
                            // Only a deliberate exit (`/quit` returns `Ok(false)`) stops
                            // the engine. A command error keeps the engine alive so the UI
                            // stays usable; exiting on errors would strand the window with
                            // a dead channel and a permanently "running" state.
                            let should_exit = matches!(result, Ok(false));
                            let resumed = before_resume
                                .filter(|before| result.is_ok() && *before != session.conversation_id())
                                .and_then(|_| session.conversation().ok());
                            let _ = event_tx.unbounded_send(match resumed {
                                Some(conversation) => {
                                    UiEvent::ConversationResumed(Ok(Arc::new(conversation)))
                                }
                                None => UiEvent::Done(result),
                            });
                            if should_exit {
                                break;
                            }
                        }
                        None => {
                            let _ = event_tx.unbounded_send(UiEvent::Cancelled);
                        }
                    }
                }
            });
        })
        .expect("failed to spawn zqa engine thread");
}
