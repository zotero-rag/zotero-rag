//! Channel-backed command input and output streams.

use std::io::{self, BufRead, Read, Write};

use crossbeam_channel::{Receiver, Sender};

/// Events emitted by the command worker.
pub(super) enum WorkerEvent {
    Output(String),
    Stderr(String),
    Error(String),
    Stream(String),
    PromptPending,
    Finished {
        should_continue: bool,
        error: Option<String>,
        input_tokens: u64,
        output_tokens: u64,
        session_cost: u64,
        message_count: usize,
        title: Option<String>,
        documents: Vec<String>,
        conversation: Option<String>,
    },
}

/// A writer that preserves split UTF-8 code points across writes.
#[derive(Clone)]
pub(super) struct ChannelWriter {
    sender: Sender<WorkerEvent>,
    is_error: bool,
    pending: Vec<u8>,
    ansi_escape: bool,
    ansi_control_sequence: bool,
}

impl ChannelWriter {
    pub(super) fn new(sender: Sender<WorkerEvent>, is_error: bool) -> Self {
        Self {
            sender,
            is_error,
            pending: Vec::new(),
            ansi_escape: false,
            ansi_control_sequence: false,
        }
    }

    fn emit(&mut self, text: &str) {
        let mut plain = String::with_capacity(text.len());
        for character in text.chars() {
            if !self.ansi_escape && character == '\x1b' {
                self.ansi_escape = true;
            } else if self.ansi_escape && !self.ansi_control_sequence && character == '[' {
                self.ansi_control_sequence = true;
            } else if self.ansi_escape
                && (!self.ansi_control_sequence || ('@'..='~').contains(&character))
            {
                self.ansi_escape = false;
                self.ansi_control_sequence = false;
            } else if !self.ansi_escape {
                plain.push(character);
            }
        }
        if plain.is_empty() {
            return;
        }
        let event = if self.is_error {
            WorkerEvent::Stderr(plain)
        } else {
            WorkerEvent::Output(plain)
        };
        let _ = self.sender.send(event);
    }
}

impl Write for ChannelWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(bytes);
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    let text = text.to_string();
                    if !text.is_empty() {
                        self.emit(&text);
                    }
                    self.pending.clear();
                    break;
                }
                Err(error) if error.error_len().is_none() => {
                    let valid = error.valid_up_to();
                    if valid > 0 {
                        let text = String::from_utf8(self.pending.drain(..valid).collect())
                            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                        self.emit(&text);
                    }
                    break;
                }
                Err(error) => {
                    let end = error.valid_up_to() + error.error_len().unwrap_or(1);
                    let text = String::from_utf8_lossy(&self.pending[..end]).into_owned();
                    self.pending.drain(..end);
                    self.emit(&text);
                }
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Blocking line input fed by UI submissions.
pub(super) struct ChannelInput {
    receiver: Receiver<Option<String>>,
    events: Sender<WorkerEvent>,
    buffer: Vec<u8>,
    position: usize,
}

impl ChannelInput {
    pub(super) fn new(receiver: Receiver<Option<String>>, events: Sender<WorkerEvent>) -> Self {
        Self {
            receiver,
            events,
            buffer: Vec::new(),
            position: 0,
        }
    }
}

impl BufRead for ChannelInput {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.position == self.buffer.len() {
            self.buffer.clear();
            self.position = 0;
            let _ = self.events.send(WorkerEvent::PromptPending);
            let mut line = self
                .receiver
                .recv()
                .map_err(|_| io::Error::new(io::ErrorKind::UnexpectedEof, "TUI input closed"))?
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::Interrupted, "TUI input interrupted")
                })?;
            if !line.ends_with('\n') {
                line.push('\n');
            }
            self.buffer.extend_from_slice(line.as_bytes());
        }
        Ok(&self.buffer[self.position..])
    }

    fn consume(&mut self, amount: usize) {
        self.position = (self.position + amount).min(self.buffer.len());
    }
}

impl Read for ChannelInput {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let length = available.len().min(output.len());
        output[..length].copy_from_slice(&available[..length]);
        self.consume(length);
        Ok(length)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, Write};

    use crossbeam_channel::unbounded;

    use super::{ChannelInput, ChannelWriter, WorkerEvent};

    #[test]
    fn writer_buffers_split_utf8() {
        let (sender, receiver) = unbounded();
        let mut writer = ChannelWriter::new(sender, false);
        let bytes = [b'c', b'a', b'f', 0xc3, 0xa9];
        writer.write_all(&bytes[..4]).unwrap();
        writer.write_all(&bytes[4..]).unwrap();
        let text: String = receiver
            .try_iter()
            .filter_map(|event| match event {
                WorkerEvent::Output(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(text.as_bytes(), bytes);
    }

    #[test]
    fn writer_strips_split_ansi_sequences() {
        let (sender, receiver) = unbounded();
        let mut writer = ChannelWriter::new(sender, false);
        writer.write_all(b"before\x1b[").unwrap();
        writer.write_all(b"31mred\x1b[0m after").unwrap();
        let text: String = receiver
            .try_iter()
            .filter_map(|event| match event {
                WorkerEvent::Output(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(text, "beforered after");
    }

    #[test]
    fn input_notifies_and_delivers_prompt_answer() {
        let (input_sender, input_receiver) = unbounded();
        let (event_sender, event_receiver) = unbounded();
        let mut input = ChannelInput::new(input_receiver, event_sender);
        input_sender.send(Some("2".into())).unwrap();
        let mut line = String::new();
        input.read_line(&mut line).unwrap();
        assert_eq!(line, "2\n");
        assert!(matches!(
            event_receiver.recv().unwrap(),
            WorkerEvent::PromptPending
        ));
    }

    #[test]
    fn input_interrupt_is_reported_as_io_error() {
        let (input_sender, input_receiver) = unbounded();
        let (event_sender, _event_receiver) = unbounded();
        let mut input = ChannelInput::new(input_receiver, event_sender);
        input_sender.send(None).unwrap();
        let error = input.fill_buf().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::Interrupted);
    }
}
