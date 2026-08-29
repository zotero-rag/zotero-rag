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

use std::io::{self, Write};
use std::sync::Arc;
use std::thread;

use futures::channel::mpsc::UnboundedSender;
use tokio::sync::mpsc::UnboundedReceiver;
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

/// A [`Write`] implementation that forwards written bytes to the UI as [`UiEvent`]s,
/// stripping ANSI SGR escape sequences (the handlers colorize output for a terminal).
struct ChannelWriter {
    tx: UnboundedSender<UiEvent>,
    is_err: bool,
}

impl Write for ChannelWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // The handlers write each colored fragment as its own `&str`, so a given `write`
        // call holds either a complete escape sequence or plain text, never a split one.
        // UTF-8 boundaries can in principle be split across writes; lossy decoding is
        // acceptable here since this is display-only output.
        let text = strip_ansi(&String::from_utf8_lossy(buf));
        if !text.is_empty() {
            let event = if self.is_err {
                UiEvent::Stderr(text)
            } else {
                UiEvent::Stdout(text)
            };
            // A closed receiver means the UI is gone; drop the output rather than error.
            let _ = self.tx.unbounded_send(event);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
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

            let out = ChannelWriter {
                tx: event_tx.clone(),
                is_err: false,
            };
            let err = ChannelWriter {
                tx: event_tx.clone(),
                is_err: true,
            };

            let mut session = match Session::new(config, out, err) {
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
                            let result: Option<Result<bool, String>> = tokio::select! {
                                result = session.dispatch(&command) => {
                                    Some(result.map_err(|e| e.to_string()))
                                }
                                _ = cancel_rx.recv() => None,
                            };

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

#[cfg(test)]
mod tests {
    use super::strip_ansi;

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
