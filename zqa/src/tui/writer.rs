//! Plumbing between the command worker and the UI event loop.
//!
//! The TUI reuses the CLI command handlers unchanged. Handlers write to the
//! [`crate::common::Context`]'s output streams; in the TUI those streams are [`ChannelWriter`]s,
//! which forward the bytes to the UI event loop as [`WorkerEvent`]s so they can be rendered in
//! the transcript instead of being printed to the (alternate-screen) terminal.

use std::io;

use tokio::sync::mpsc::UnboundedSender;

/// Events sent from the command worker to the UI event loop.
pub(crate) enum WorkerEvent {
    /// Bytes a command handler wrote to the context's stdout stream.
    Stdout(Vec<u8>),
    /// Bytes a command handler wrote to the context's stderr stream.
    Stderr(Vec<u8>),
    /// A chat history item replayed from a conversation that was loaded without being
    /// printed (e.g., via `/resume`).
    HistoryItem { is_user: bool, text: String },
    /// Fresh sidebar data, sent once at startup and after every command.
    Sidebar(SidebarSnapshot),
    /// The last command finished. `should_continue` is `false` when the user quit.
    Done {
        should_continue: bool,
        error: Option<String>,
    },
}

/// A point-in-time copy of the session information shown in the sidebar.
///
/// The worker task owns the [`crate::common::Context`], so the UI cannot read the state
/// directly; instead the worker sends a snapshot after each command.
#[derive(Default)]
pub(crate) struct SidebarSnapshot {
    /// The generated title of the current conversation, if any.
    pub(crate) title: Option<String>,
    /// The generation model name, if configured.
    pub(crate) model: Option<String>,
    /// Accumulated session cost, in US cents (see [`crate::common::State`]).
    pub(crate) session_cost_cents: u64,
    /// Number of items in the current chat history.
    pub(crate) message_count: usize,
    /// Filenames of user-imported documents.
    pub(crate) documents: Vec<String>,
}

/// A [`io::Write`] implementation that forwards everything written to it to the UI event loop.
pub(crate) struct ChannelWriter {
    tx: UnboundedSender<WorkerEvent>,
    is_stderr: bool,
}

impl ChannelWriter {
    /// Create a writer that stands in for stdout.
    pub(crate) fn stdout(tx: UnboundedSender<WorkerEvent>) -> Self {
        Self {
            tx,
            is_stderr: false,
        }
    }

    /// Create a writer that stands in for stderr.
    pub(crate) fn stderr(tx: UnboundedSender<WorkerEvent>) -> Self {
        Self {
            tx,
            is_stderr: true,
        }
    }
}

impl io::Write for ChannelWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let event = if self.is_stderr {
            WorkerEvent::Stderr(buf.to_vec())
        } else {
            WorkerEvent::Stdout(buf.to_vec())
        };

        // The receiver only closes during shutdown; dropping output at that point is fine.
        let _ = self.tx.send(event);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
