//! Channel-backed I/O between command handlers and the UI thread.
//!
//! Handlers write to `Context::out` / `Context::err` and read prompt replies from
//! `Context::input`. In the TUI those streams are backed by channels: [`ChannelWriter`]
//! forwards complete lines of handler output to the UI thread as [`UiMessage`]s, and
//! [`ChannelReader`] asks the UI thread for a line of input whenever a handler prompts for one
//! (such as `/resume` asking for a selection).

use std::io::{self, BufRead, Read, Write};
use std::sync::atomic;

use crossbeam_channel::{Receiver, Sender};

use crate::common::Context;

/// A snapshot of session state shown in the TUI sidebar.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SidebarSnapshot {
    /// Total input tokens used across the session.
    pub(crate) input_tokens: u64,
    /// Total output tokens used across the session.
    pub(crate) output_tokens: u64,
    /// Accumulated session cost, in US cents.
    pub(crate) session_cost_cents: u64,
    /// The generated title of the current conversation, if one exists yet.
    pub(crate) conversation_title: Option<String>,
    /// Number of messages in the current conversation.
    pub(crate) message_count: usize,
    /// Keys of documents imported into the session, sorted.
    pub(crate) documents: Vec<String>,
}

impl SidebarSnapshot {
    /// Capture the current session state from `ctx`.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The context whose state is snapshotted.
    ///
    /// # Returns
    ///
    /// A snapshot of the values shown in the sidebar.
    pub(crate) fn from_context<O: Write, E: Write>(ctx: &Context<O, E>) -> Self {
        let documents = ctx.state.imports.read().map_or_else(
            |_| Vec::new(),
            |imports| {
                let mut keys: Vec<String> = imports.keys().cloned().collect();
                keys.sort();
                keys
            },
        );

        Self {
            input_tokens: ctx
                .state
                .session_input_tokens
                .load(atomic::Ordering::Relaxed),
            output_tokens: ctx
                .state
                .session_output_tokens
                .load(atomic::Ordering::Relaxed),
            session_cost_cents: ctx.state.session_cost.load(atomic::Ordering::Relaxed),
            conversation_title: ctx.state.title.lock().ok().and_then(|t| t.clone()),
            message_count: ctx.state.chat_history.lock().map_or(0, |h| h.len()),
            documents,
        }
    }
}

/// Messages sent from the command runner to the UI thread.
#[derive(Debug, PartialEq)]
pub(crate) enum UiMessage {
    /// A line of primary handler output (the `stdout` lane).
    Out(String),
    /// A line of secondary handler output (the `stderr` lane), rendered dimmed.
    Info(String),
    /// A command error that should stand out in the transcript.
    Error(String),
    /// A segment of text streamed from the model.
    Stream(String),
    /// A tool invocation or timing trace, rendered as secondary text.
    ToolTrace(String),
    /// A handler is blocked waiting for a line of input; the query box becomes a reply prompt.
    InputRequested,
    /// A submitted command finished; carries a fresh sidebar snapshot.
    CommandDone(SidebarSnapshot),
    /// The runner is exiting; the UI thread should tear down.
    Shutdown,
}

/// Which transcript lane a [`ChannelWriter`] feeds.
#[derive(Clone, Copy)]
pub(crate) enum OutputLane {
    /// Primary output, rendered as normal text.
    Out,
    /// Secondary output, rendered dimmed.
    Info,
}

/// An [`io::Write`] implementation that forwards complete lines to the UI thread.
///
/// Partial lines are buffered until a newline arrives or [`Write::flush`] is called, so
/// prompts written without a trailing newline (e.g. `/resume`'s selection prompt) still reach
/// the UI.
pub(crate) struct ChannelWriter {
    tx: Sender<UiMessage>,
    lane: OutputLane,
    buf: Vec<u8>,
}

impl ChannelWriter {
    /// Create a writer that sends lines to `tx` on the given `lane`.
    pub(crate) fn new(tx: Sender<UiMessage>, lane: OutputLane) -> Self {
        Self {
            tx,
            lane,
            buf: Vec::new(),
        }
    }

    fn send_line(&self, line: &[u8]) {
        let text = String::from_utf8_lossy(line).into_owned();
        let message = match self.lane {
            OutputLane::Out => UiMessage::Out(text),
            OutputLane::Info => UiMessage::Info(text),
        };

        // The UI thread may already have exited during shutdown; dropping output is fine then.
        let _ = self.tx.send(message);
    }
}

impl Write for ChannelWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(data);

        while let Some(newline_idx) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=newline_idx).collect();
            self.send_line(&line[..line.len() - 1]);
        }

        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.buf.is_empty() {
            let line = std::mem::take(&mut self.buf);
            self.send_line(&line);
        }

        Ok(())
    }
}

/// A [`BufRead`] implementation that requests a line from the UI thread on demand.
///
/// When a handler reads input, this sends [`UiMessage::InputRequested`] so the UI can switch
/// its query box into reply mode, then blocks until the user submits a line. If the UI thread
/// has exited, reads return EOF so blocked handlers can finish.
pub(crate) struct ChannelReader {
    tx: Sender<UiMessage>,
    reply_rx: Receiver<String>,
    buf: Vec<u8>,
    pos: usize,
}

impl ChannelReader {
    /// Create a reader that signals input requests on `tx` and receives replies on `reply_rx`.
    pub(crate) fn new(tx: Sender<UiMessage>, reply_rx: Receiver<String>) -> Self {
        Self {
            tx,
            reply_rx,
            buf: Vec::new(),
            pos: 0,
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let count = available.len().min(out.len());
        out[..count].copy_from_slice(&available[..count]);
        self.consume(count);

        Ok(count)
    }
}

impl BufRead for ChannelReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos >= self.buf.len() {
            if self.tx.send(UiMessage::InputRequested).is_err() {
                return Ok(&[]);
            }

            match self.reply_rx.recv() {
                Ok(line) => {
                    self.buf = line.into_bytes();
                    self.buf.push(b'\n');
                    self.pos = 0;
                }
                // The UI thread exited; treat it as EOF.
                Err(_) => return Ok(&[]),
            }
        }

        Ok(&self.buf[self.pos..])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.buf.len());
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, Write};

    use zqa_macros::test_eq;

    use super::{ChannelReader, ChannelWriter, OutputLane, SidebarSnapshot, UiMessage};
    use crate::cli::app::tests::create_test_context;

    #[test]
    fn test_channel_writer_sends_complete_lines() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut writer = ChannelWriter::new(tx, OutputLane::Out);

        writeln!(writer, "hello").unwrap();
        writeln!(writer, "one\ntwo").unwrap();

        test_eq!(rx.try_recv(), Ok(UiMessage::Out("hello".into())));
        test_eq!(rx.try_recv(), Ok(UiMessage::Out("one".into())));
        test_eq!(rx.try_recv(), Ok(UiMessage::Out("two".into())));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_channel_writer_flushes_partial_lines() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut writer = ChannelWriter::new(tx, OutputLane::Info);

        write!(writer, "Enter a number: ").unwrap();
        assert!(rx.try_recv().is_err());

        writer.flush().unwrap();
        test_eq!(
            rx.try_recv(),
            Ok(UiMessage::Info("Enter a number: ".into()))
        );
    }

    #[test]
    fn test_channel_writer_ignores_disconnected_ui() {
        let (tx, rx) = crossbeam_channel::unbounded();
        drop(rx);

        let mut writer = ChannelWriter::new(tx, OutputLane::Out);
        assert!(writeln!(writer, "into the void").is_ok());
    }

    #[test]
    fn test_channel_reader_requests_and_receives_reply() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let (reply_tx, reply_rx) = crossbeam_channel::unbounded();
        let mut reader = ChannelReader::new(tx, reply_rx);

        // Queue the reply up front; `fill_buf` sends the request before blocking on it.
        reply_tx.send("42".to_string()).unwrap();

        let mut line = String::new();
        reader.read_line(&mut line).unwrap();

        test_eq!(line, "42\n");
        test_eq!(rx.try_recv(), Ok(UiMessage::InputRequested));
    }

    #[test]
    fn test_channel_reader_eof_when_ui_gone() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let (reply_tx, reply_rx) = crossbeam_channel::unbounded::<String>();
        drop(rx);
        drop(reply_tx);

        let mut reader = ChannelReader::new(tx, reply_rx);
        let mut line = String::new();
        let read = reader.read_line(&mut line).unwrap();

        test_eq!(read, 0);
        assert!(line.is_empty());
    }

    #[test]
    fn test_sidebar_snapshot_from_context() {
        let ctx = create_test_context(vec![]);
        ctx.state
            .session_input_tokens
            .store(1234, std::sync::atomic::Ordering::Relaxed);
        *ctx.state.title.lock().unwrap() = Some("Attention papers".to_string());

        let snapshot = SidebarSnapshot::from_context(&ctx);

        test_eq!(snapshot.input_tokens, 1234);
        test_eq!(snapshot.output_tokens, 0);
        test_eq!(
            snapshot.conversation_title,
            Some("Attention papers".to_string())
        );
        test_eq!(snapshot.message_count, 0);
        assert!(snapshot.documents.is_empty());
    }
}
