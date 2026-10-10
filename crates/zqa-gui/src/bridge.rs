//! The runtime bridge between GPUI (which owns the main thread and its own executor)
//! and the tokio-based zqa pipeline.
//!
//! [`spawn_engine`] owns a tokio runtime and [`Session`] on a dedicated thread. It drives
//! commands while translating engine events into the GUI's text streams. Cancellation drops
//! the in-flight dispatch future. Interactive requests fail without reading terminal input
//! or submitting a default answer.

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
#[derive(Debug, Clone)]
pub enum UiEvent {
    /// A chunk of command stdout (typically answer text).
    Stdout(String),
    /// A chunk of command stderr (status lines, warnings, timings).
    Stderr(String),
    /// A command finished. Carries the dispatch result: `Ok(keep_running)` or an
    /// error message.
    Done(Result<bool, String>),
    /// A saved conversation resume attempt finished.
    ConversationResumed(Result<Arc<SavedChatHistory>, String>),
    /// The in-flight command was cancelled by the user before it finished.
    Cancelled,
}

/// Forward displayable events without terminal formatting.
///
/// Input requests are rejected and their reply senders dropped without answering. The caller
/// must stop the operation on error so a handler cannot continue after a rejected request.
/// A disconnected GUI also returns an error; ignored usage and low-severity events produce no output.
fn forward_event(event: EngineEvent, tx: &UnboundedSender<UiEvent>) -> Result<(), String> {
    let (message, is_error) = match event {
        EngineEvent::Text { message } | EngineEvent::Reasoning { message } => (message, false),
        EngineEvent::StatusUpdate { message }
        | EngineEvent::Warning { message }
        | EngineEvent::Error { message } => (message, true),
        EngineEvent::ToolCall { .. } | EngineEvent::ToolResponse { .. } => {
            (format!("{event}\n"), true)
        }
        EngineEvent::RecoverableWarning { .. } | EngineEvent::TokenUsage { .. } => return Ok(()),
        EngineEvent::Confirm { .. }
        | EngineEvent::Choose { .. }
        | EngineEvent::Line { .. }
        | EngineEvent::Secret { .. } => {
            return Err(
                "Interactive prompts are not supported in this GUI. Run this command in the CLI."
                    .into(),
            );
        }
        _ => return Err("The GUI cannot handle this engine event.".into()),
    };

    let message = strip_ansi(&message);

    if message.is_empty() {
        return Ok(());
    }

    let event = if is_error {
        UiEvent::Stderr(message)
    } else {
        UiEvent::Stdout(message)
    };

    tx.unbounded_send(event).map_err(|error| error.to_string())
}

/// Remove ANSI SGR escape sequences (`ESC [ ... m` and friends) from `input`.
///
/// This is a deliberately small stripper: it drops `ESC [` up to and including the first
/// alphabetic terminator, which covers every sequence the handlers emit (see
/// `zqa::utils::terminal`). It is not a general-purpose ANSI parser.
fn strip_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.next() == Some('[') {
                while let Some(next) = chars.next()
                    && !next.is_ascii_alphabetic()
                {}
            }
            continue;
        }
        out.push(c);
    }

    out
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
                            let mut result = {
                                let dispatch = session.dispatch(&command);
                                tokio::pin!(dispatch);

                                loop {
                                    tokio::select! {
                                        biased;
                                        Some(()) = cancel_rx.recv() => break None,
                                        result = &mut dispatch => break Some(result.map_err(|e| e.to_string())),
                                        event = engine_rx.recv() => {
                                            let result = event
                                                .ok_or_else(|| "The engine event channel closed".to_string())
                                                .and_then(|event| forward_event(event, &event_tx));

                                            if let Err(error) = result {
                                                break Some(Err(error));
                                            }
                                        }
                                    }
                                }
                            };

                            // Dispatch is dropped before draining, so rejected prompts cannot resume it.
                            while let Ok(event) = engine_rx.try_recv() {
                                if let Err(error) = forward_event(event, &event_tx)
                                    && matches!(result, Some(Ok(_)))
                                {
                                    result = Some(Err(error));
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
                            let mut result = {
                                let resume = session.resume_conversation(&conversation);
                                tokio::pin!(resume);

                                loop {
                                    tokio::select! {
                                        result = &mut resume => break result.map_err(|error| error.to_string()),
                                        event = engine_rx.recv() => {
                                            let result = event
                                                .ok_or_else(|| "The engine event channel closed".to_string())
                                                .and_then(|event| forward_event(event, &event_tx));

                                            if let Err(error) = result {
                                                break Err(error);
                                            }
                                        }
                                    }
                                }
                            };

                            while let Ok(event) = engine_rx.try_recv() {
                                if let Err(error) = forward_event(event, &event_tx)
                                    && result.is_ok()
                                {
                                    result = Err(error);
                                }
                            }

                            let result = result.map(|()| conversation);
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
    use tokio::sync::oneshot;
    use zqa::io::EngineEvent;

    use super::{UiEvent, forward_event, strip_ansi};

    #[test]
    fn forwards_plain_text_and_status_in_order() {
        let (tx, mut rx) = futures::channel::mpsc::unbounded();

        for event in [
            EngineEvent::Text {
                message: "\x1b[1manswer\x1b[0m".into(),
            },
            EngineEvent::Reasoning {
                message: "reasoning".into(),
            },
            EngineEvent::StatusUpdate {
                message: "\x1b[2mworking\x1b[0m".into(),
            },
            EngineEvent::Warning {
                message: "warning".into(),
            },
            EngineEvent::Error {
                message: "failure".into(),
            },
            EngineEvent::TokenUsage {
                usage: Default::default(),
            },
            EngineEvent::RecoverableWarning {
                message: "hidden".into(),
            },
        ] {
            forward_event(event, &tx).unwrap();
        }

        assert!(matches!(rx.try_recv().unwrap(), UiEvent::Stdout(message) if message == "answer"));
        assert!(
            matches!(rx.try_recv().unwrap(), UiEvent::Stdout(message) if message == "reasoning")
        );

        for expected in ["working", "warning", "failure"] {
            assert!(
                matches!(rx.try_recv().unwrap(), UiEvent::Stderr(message) if message == expected)
            );
        }

        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn rejects_prompts_without_answering_or_forwarding_them() {
        let (tx, mut rx) = futures::channel::mpsc::unbounded();
        let (confirm, mut confirmation) = oneshot::channel();
        let (choose, mut choice) = oneshot::channel();
        let (line, mut text) = oneshot::channel();
        let (secret, mut password) = oneshot::channel();

        for event in [
            EngineEvent::Confirm {
                message: "Continue?".into(),
                default: true,
                reply: confirm,
            },
            EngineEvent::Choose {
                message: "Choose".into(),
                options: vec!["First".into()],
                default: 0,
                reply: choose,
            },
            EngineEvent::Line {
                message: None,
                reply: line,
            },
            EngineEvent::Secret {
                message: "Secret".into(),
                reply: secret,
            },
        ] {
            assert!(
                forward_event(event, &tx)
                    .unwrap_err()
                    .contains("Run this command in the CLI")
            );
        }

        assert!(matches!(
            confirmation.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert!(matches!(
            choice.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert!(matches!(
            text.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert!(matches!(
            password.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn leaves_plain_text_untouched() {
        assert_eq!(strip_ansi("hello world"), "hello world");
    }

    #[test]
    fn strips_sgr_sequences() {
        // Dim + reset (the handlers' most common colorization).
        assert_eq!(strip_ansi("\x1b[2mdim\x1b[0m"), "dim");
        // Multi-parameter sequence with surrounding text preserved.
        assert_eq!(
            strip_ansi("\x1b[31;1mred bold\x1b[0m done"),
            "red bold done"
        );
    }

    #[test]
    fn handles_trailing_escape() {
        // A lone trailing ESC with no sequence body is dropped without panicking.
        assert_eq!(strip_ansi("text\x1b"), "text");
    }
}
