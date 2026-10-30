//! The runtime bridge between GPUI (which owns the main thread and its own executor)
//! and the tokio-based zqa pipeline.
//!
//! [`spawn_engine`] dedicates one OS thread to a tokio runtime and [`Session`]. It receives
//! commands sequentially and forwards typed engine events to GPUI. Dispatch can be cancelled
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

/// Decide whether a command switched conversations, given snapshots from before and after it.
///
/// A `/resume` can finish without switching: there may be no saved conversations, or the picked
/// one may be the conversation that is already open. Only a different history or title counts, so
/// the UI keeps its transcript, including rows the engine doesn't store such as status lines, when
/// nothing changed.
///
/// # Returns
///
/// `after` if the conversation changed, otherwise `None`.
fn switched_conversation(
    before: &SavedChatHistory,
    after: SavedChatHistory,
) -> Option<SavedChatHistory> {
    (before.history != after.history || before.title != after.title).then_some(after)
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
                let mut pending_command = None;

                loop {
                    let command = match pending_command.take() {
                        Some(command) => command,
                        None => match cmd_rx.recv().await {
                            Some(command) => command,
                            None => break,
                        },
                    };
                    let shutting_down = matches!(command, EngineCommand::Shutdown);

                    // A late stop click must not cancel the next command.
                    while cancel_rx.try_recv().is_ok() {}

                    match command {
                        command @ (EngineCommand::Dispatch(_) | EngineCommand::Shutdown) => {
                            let command = match command {
                                EngineCommand::Dispatch(command) => command,
                                _ => "/quit".into(),
                            };
                            // A typed `/resume` swaps the conversation inside the engine, so the
                            // UI only learns of it by comparing snapshots after it finishes.
                            let before_resume = (command == "/resume")
                                .then(|| session.conversation().ok())
                                .flatten();
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
                                        .filter(|_| result.is_ok())
                                        .zip(session.conversation().ok())
                                        .and_then(|(before, after)| switched_conversation(&before, after));
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
                        EngineCommand::ResumeConversation(conversation) => {
                            let result = {
                                let resume = session.resume_conversation(&conversation);
                                tokio::pin!(resume);

                                loop {
                                    tokio::select! {
                                        result = &mut resume => break result,
                                        Some(event) = engine_rx.recv() => {
                                            if event_tx.unbounded_send(UiEvent::Engine(event)).is_err() {
                                                return;
                                            }
                                        }
                                    }
                                }
                            };

                            while let Ok(event) = engine_rx.try_recv() {
                                if event_tx.unbounded_send(UiEvent::Engine(event)).is_err() {
                                    return;
                                }
                            }

                            let result = result.map(|()| conversation).map_err(|error| error.to_string());
                            let _ = event_tx.unbounded_send(UiEvent::ConversationResumed(result));
                        }
                    }
                }
            });
        })
        .expect("failed to spawn zqa engine thread");
}

#[cfg(test)]
mod tests {
    use zqa::state::SavedChatHistory;

    use super::switched_conversation;

    fn snapshot(title: &str) -> SavedChatHistory {
        serde_json::from_value(serde_json::json!({
            "history": [],
            "date": "2026-10-29T12:00:00+00:00",
            "title": title,
        }))
        .unwrap()
    }

    #[test]
    fn only_a_changed_conversation_counts_as_a_switch() {
        assert!(switched_conversation(&snapshot("Attention"), snapshot("Attention")).is_none());

        let switched = switched_conversation(&snapshot("Attention"), snapshot("Transformers"));
        assert_eq!(switched.map(|c| c.title), Some("Transformers".into()));
    }
}
