//! I/O primitives for working with the agent in `zqa`.

use std::fmt::Display;
use std::io::{self, BufRead, Write};

use tokio::sync::oneshot;

use crate::state::UsageMetadata;
use crate::utils::terminal::{BLUE, DIM_TEXT, ITALICS, RED, RESET, YELLOW};

/// An enum of events that can be emitted by handlers called by
/// [`crate::cli::app::dispatch_command`]. The contract here is that *every* event, including status
/// updates and error messages will be passed as an event. In general, events carry text, and
/// variants are a semantic layer on top. Structured payloads are used for data; the current
/// exceptions are `ToolCall`, `ToolResponse`, and `TokenUsage`.
///
/// You should use events to enable interactivity in your application. Events
/// are not guaranteed to be a reliable mechanism to detect operation failure: you should rely on
/// functions returning a `Result` type for this.
///
/// This is the intended contract; handlers and CLI/GUI consumers are not connected yet.
///
/// ## Display
///
/// This enum implements [`std::fmt::Display`], whose implementation is intended for CLI consumers and
/// uses ANSI SGR codes. For variants that elicit user input, it only displays the question and the options
/// (if any), but does not handle input. For multiple-choice questions, this presents the user with
/// numbered options starting from 1. In all cases with options, we follow general CLI standards of
/// presenting the default within [square brackets] and other options in (parentheses).
///
/// The enum has a [`Self::handle_event`] function that uses the `Display` impl, handles input, and sends
/// the user response via the channel.
///
/// * [`EngineEvent::Text`] is rendered as-is.
/// * [`EngineEvent::ToolCall`] is rendered in a dimmed text with the tool name followed by
///   JSON args.
/// * [`EngineEvent::ToolResponse`] is rendered in a dimmed text with two leading spaces, followed
///   by `->`, followed by the tool name in parentheses, and finally the
///   pretty-printed JSON response.
/// * [`EngineEvent::Reasoning`] renders as dimmed, italics text.
/// * [`EngineEvent::Confirm`] renders the message, followed by either "([y]/n): " or "(y/[n]): ",
///   so it assumes users are presented a yes/no question. For other binary-response questions, use
///   [`EngineEvent::Choose`] instead.
/// * [`EngineEvent::Choose`] renders the message, followed by two newlines, and numbered options in
///   order. Although the enum variant's `default` arg is 0-indexed, users see 1-based options.
///   After the options, it emits two additional newlines, followed by "(1 - max, default: ..) > ".
/// * [`EngineEvent::Line`] renders the message if it is not `None`, then prints two newlines and
///   finally prints "> ". The latter actions occur regardless of the value of `message`.
///   [`EngineEvent::Secret`] has the same behavior.
/// * [`EngineEvent::RecoverableWarning`] and [`EngineEvent::TokenUsage`] are ignored.
/// * [`EngineEvent::StatusUpdate`] is printed in light blue.
/// * [`EngineEvent::Warning`] and [`EngineEvent::Error`] are printed in yellow and red
///   respectively.
#[derive(Debug)]
#[non_exhaustive]
pub enum EngineEvent {
    /// Text from the model.
    Text { message: String },
    /// A tool call. This event is emitted before the tool is executed, but after it is parsed and
    /// known to be a valid tool call. Every emitted `ToolCall` is followed (but not necessarily
    /// immediately) by a `ToolResponse` event with the same `id`. A tool call that fails
    /// validation, such as a hallucinated tool call, emits an `Error` and no `ToolCall` event.
    /// TODO: Currently we only learn about a tool call after execution, so the hooks in `zqa-rag`
    /// should probably change.
    ToolCall {
        name: String,
        id: String,
        args: serde_json::Value,
    },
    /// A response to a tool call. This event is emitted after the tool is called. Tools are
    /// the authority on whether they failed, so an `Err` variant in the `response` field means the
    /// tool declares that it failed. This includes invalid arguments, but also semantic failures such
    /// as a `bash` tool completing but returning a non-zero exit code.
    ToolResponse {
        name: String,
        id: String,
        response: Result<serde_json::Value, String>,
    },
    /// Reasoning traces from models.
    Reasoning { message: String },
    /// A boolean input request.
    Confirm {
        message: String,
        default: bool,
        reply: oneshot::Sender<bool>,
    },
    /// A multiple-choice input request.
    /// Replies are zero-based indices into `options`. Empty options display a notice;
    /// `handle_event` rejects them with `InvalidInput`.
    Choose {
        message: String,
        options: Vec<String>,
        reply: oneshot::Sender<usize>,
        /// A 0-based index into `options`. Formatting and input handling reset out-of-range
        /// defaults to 0. The event payload itself is not validated on construction.
        default: usize,
    },
    /// An input request for sensitive data.
    Secret {
        message: String,
        reply: oneshot::Sender<String>,
    },
    /// An input request that elicits free-form input.
    Line {
        /// A question to present the user, optional. It may make more sense to yield two events
        /// where this variant is used solely to elicit an answer, and context is provided by
        /// earlier event(s).
        message: Option<String>,
        reply: oneshot::Sender<String>,
    },
    /// A general INFO level status update.
    StatusUpdate { message: String },
    /// A general ERROR level status update.
    Error { message: String },
    /// Distinct from `Warning`, this is a lower-severity message and indicates that something was
    /// not quite as expected, but that it is unlikely to be an issue. For example, if we encounter
    /// a file that is not a PDF in the user's Zotero library, we emit this. The `Display` implementation
    /// ignores these.
    RecoverableWarning { message: String },
    /// A WARN level update that signifies that something is not as expected and may cause future
    /// failures. It is possible for a session to proceed normally having emitted this, and is not
    /// fatal. Users should likely be shown these.
    Warning { message: String },
    /// An update on token usage. This is not an aggregate and consumers are responsible for
    /// accumulating these. Ignored by the `Display` implementation.
    TokenUsage { usage: UsageMetadata },
}

impl EngineEvent {
    /// Handle an event in the CLI. As described in the docs for the enum, this first uses the
    /// enum's `Display` impl, and then handles input.
    ///
    /// Prompts are flushed before reading. Line replies preserve their line ending; confirmation
    /// and choice replies ignore surrounding whitespace. A dropped reply receiver is treated as
    /// a cancelled request, so a failed reply send is ignored.
    ///
    /// # Arguments
    ///
    /// * `input` - Buffered input for confirmation, choice, and line requests.
    /// * `out` - Output for model text, status, and prompts.
    /// * `err` - Output for warnings and errors.
    /// * `read_secret` - Reads a secret without echoing it, for example `rpassword::read_password`.
    ///   Called only for `Secret`, after flushing its prompt. Tests or non-terminal consumers
    ///   can supply their own reader.
    ///
    /// # Returns
    ///
    /// `Ok(())` after rendering the event and sending any requested reply.
    ///
    /// # Errors
    ///
    /// * Returns `InvalidInput` for a choice request with no options, before writing or reading.
    /// * Returns `UnexpectedEof` when input ends before a confirmation, choice, or line reply.
    /// * Propagates read, write, flush, and secret-reader errors. The reply sender is dropped.
    ///
    /// # Example
    ///
    /// ```
    /// use zqa::io::EngineEvent;
    ///
    /// let (reply_tx, mut answer_rx) = tokio::sync::oneshot::channel();
    /// EngineEvent::Confirm { message: "Continue?".into(), default: true, reply: reply_tx }
    ///     .handle_event(&mut &b"\n"[..], &mut Vec::new(), &mut Vec::new(),
    ///         rpassword::read_password)?;
    /// assert!(answer_rx.try_recv().unwrap());
    /// ```
    pub fn handle_event<R, O, E>(
        mut self,
        input: &mut R,
        out: &mut O,
        err: &mut E,
        read_secret: impl FnOnce() -> io::Result<String>,
    ) -> io::Result<()>
    where
        R: BufRead + ?Sized,
        O: Write + ?Sized,
        E: Write + ?Sized,
    {
        if let Self::Choose { options, .. } = &mut self
            && options.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "A choice requires at least one option",
            ));
        }

        // Use `Display` impl
        match self {
            Self::RecoverableWarning { .. } | Self::TokenUsage { .. } => return Ok(()),
            Self::Warning { .. } | Self::Error { .. } => {
                write!(err, "{self}")?;
                err.flush()?;
            }
            _ => {
                write!(out, "{self}")?;
                out.flush()?;
            }
        }

        // Handle input variants
        match self {
            Self::Confirm { default, reply, .. } => {
                let answer = read_answer(input, out, "Enter y or n.\n> ", |line| {
                    match line.chars().next().map(|c| c.to_ascii_lowercase()) {
                        None => Some(default),
                        Some('y') => Some(true),
                        Some('n') => Some(false),
                        _ => None,
                    }
                })?;
                let _ = reply.send(answer);
            }
            Self::Choose {
                options,
                reply,
                default,
                ..
            } => {
                let answer = read_answer(
                    input,
                    out,
                    "Enter one of the numbered options.\n> ",
                    |line| {
                        if line.is_empty() {
                            return Some(default);
                        }
                        line.parse::<usize>()
                            .ok()
                            .filter(|choice| (1..=options.len()).contains(choice))
                            .map(|choice| choice - 1)
                    },
                )?;
                let _ = reply.send(answer);
            }
            Self::Line { reply, .. } => {
                let _ = reply.send(read_input_line(input)?);
            }
            Self::Secret { reply, .. } => {
                let _ = reply.send(read_secret()?);
            }
            _ => {}
        }
        Ok(())
    }
}

/// Read a reply without turning EOF into a default answer or panicking on I/O errors.
fn read_input_line(input: &mut (impl BufRead + ?Sized)) -> io::Result<String> {
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Input ended before a reply",
        ));
    }
    Ok(line)
}

/// Retry invalid answers using only the supplied output stream.
fn read_answer<T>(
    input: &mut (impl BufRead + ?Sized),
    out: &mut (impl Write + ?Sized),
    retry: &str,
    parse: impl Fn(&str) -> Option<T>,
) -> io::Result<T> {
    loop {
        if let Some(answer) = parse(read_input_line(input)?.trim()) {
            return Ok(answer);
        }
        write!(out, "{retry}")?;
        out.flush()?;
    }
}

impl Display for EngineEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineEvent::Text { message } => f.write_str(message),
            EngineEvent::ToolCall { name, args, .. } => write!(f, "{DIM_TEXT}{name} {args}{RESET}"),
            EngineEvent::ToolResponse { name, response, .. } => {
                write!(f, "  {DIM_TEXT}-> ({name}) ")?;
                match response {
                    Ok(response) => write!(f, "{response:#}"),
                    Err(error) => f.write_str(error),
                }?;
                f.write_str(RESET)
            }
            EngineEvent::Reasoning { message } => write!(f, "{DIM_TEXT}{ITALICS}{message}{RESET}"),
            EngineEvent::Confirm {
                message, default, ..
            } => {
                write!(
                    f,
                    "{message} ({}): ",
                    if *default { "[y]/n" } else { "y/[n]" }
                )
            }
            EngineEvent::Choose {
                message,
                options,
                default,
                ..
            } => {
                write!(f, "{message}\n\n")?;
                if options.is_empty() {
                    return writeln!(f, "(no options available)");
                }

                let default = (options.len() - 1).min(*default);
                for (i, opt) in options.iter().enumerate() {
                    if i == default {
                        writeln!(f, "[{}] {opt}", i + 1)?;
                    } else {
                        writeln!(f, "({}) {opt}", i + 1)?;
                    }
                }

                write!(f, "\n(1 - {}, default: {}) > ", options.len(), default + 1)
            }
            EngineEvent::RecoverableWarning { .. } | EngineEvent::TokenUsage { .. } => Ok(()),
            EngineEvent::Line {
                message: Some(message),
                ..
            }
            | EngineEvent::Secret { message, .. } => {
                write!(f, "{message}\n\n> ")
            }
            EngineEvent::Line { message: None, .. } => write!(f, "\n\n> "),
            EngineEvent::StatusUpdate { message } => {
                write!(f, "{BLUE}{message}{RESET}")
            }
            EngineEvent::Warning { message } => write!(f, "{YELLOW}{message}{RESET}"),
            EngineEvent::Error { message } => write!(f, "{RED}{message}{RESET}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use zqa_macros::test_eq;

    use super::*;

    /// Reject unexpected password reads without touching the terminal.
    fn no_secret() -> io::Result<String> {
        panic!("only Secret events may read a password")
    }

    /// Record flushes so a password reader can inspect them during handling.
    struct FlushWriter {
        bytes: Vec<u8>,
        flushes: Rc<Cell<usize>>,
    }

    impl Write for FlushWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.write(bytes)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes.set(self.flushes.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn formats_and_routes_output_events() {
        let cases = [
            (
                EngineEvent::Text {
                    message: "answer".into(),
                },
                "answer".into(),
                false,
            ),
            (
                EngineEvent::Reasoning {
                    message: "thinking".into(),
                },
                format!("{DIM_TEXT}{ITALICS}thinking{RESET}"),
                false,
            ),
            (
                EngineEvent::ToolCall {
                    name: "search".into(),
                    id: "call-1".into(),
                    args: serde_json::json!({"query": "rust"}),
                },
                format!("{DIM_TEXT}search {{\"query\":\"rust\"}}{RESET}"),
                false,
            ),
            (
                EngineEvent::ToolResponse {
                    name: "search".into(),
                    id: "call-1".into(),
                    response: Ok(serde_json::json!({"count": 1})),
                },
                format!("  {DIM_TEXT}-> (search) {{\n  \"count\": 1\n}}{RESET}"),
                false,
            ),
            (
                EngineEvent::ToolResponse {
                    name: "search".into(),
                    id: "call-1".into(),
                    response: Err("failed".into()),
                },
                format!("  {DIM_TEXT}-> (search) failed{RESET}"),
                false,
            ),
            (
                EngineEvent::StatusUpdate {
                    message: "working".into(),
                },
                format!("{BLUE}working{RESET}"),
                false,
            ),
            (
                EngineEvent::Warning {
                    message: "warning".into(),
                },
                format!("{YELLOW}warning{RESET}"),
                true,
            ),
            (
                EngineEvent::Error {
                    message: "error".into(),
                },
                format!("{RED}error{RESET}"),
                true,
            ),
            (
                EngineEvent::RecoverableWarning {
                    message: "hidden".into(),
                },
                String::new(),
                true,
            ),
            (
                EngineEvent::TokenUsage {
                    usage: UsageMetadata::default(),
                },
                String::new(),
                false,
            ),
        ];
        for (event, expected, is_error) in cases {
            test_eq!(event.to_string(), expected);
            let (mut out, mut err) = (Vec::new(), Vec::new());
            event
                .handle_event(&mut io::empty(), &mut out, &mut err, no_secret)
                .unwrap();
            let (actual, unused) = if is_error { (err, out) } else { (out, err) };
            test_eq!(actual, expected.as_bytes());
            assert!(unused.is_empty());
        }
    }

    #[test]
    fn confirmation_replies_and_retries() {
        for (input, default, expected, retries) in [
            ("\n", true, true, 0),
            ("\n", false, false, 0),
            (" Y \n", false, true, 0),
            ("n\n", true, false, 0),
            ("later\nN\n", true, false, 1),
        ] {
            let (reply, mut answer) = oneshot::channel();
            let mut input = input.as_bytes();
            let mut out = Vec::new();
            let event = EngineEvent::Confirm {
                message: "Continue?".into(),
                default,
                reply,
            };
            let prompt = format!("Continue? ({}): ", if default { "[y]/n" } else { "y/[n]" });
            test_eq!(event.to_string(), prompt);
            event
                .handle_event(&mut input, &mut out, &mut Vec::new(), no_secret)
                .unwrap();
            test_eq!(answer.try_recv().unwrap(), expected);
            test_eq!(
                String::from_utf8(out).unwrap(),
                format!("{prompt}{}", "Enter y or n.\n> ".repeat(retries))
            );
        }
    }

    #[test]
    fn choice_defaults_match_display_and_reply() {
        for (default, expected) in [(0, 0), (1, 1), (2, 0), (usize::MAX, 0)] {
            let (reply, mut answer) = oneshot::channel();
            let event = EngineEvent::Choose {
                message: "Pick".into(),
                options: vec!["first".into(), "second".into()],
                default,
                reply,
            };
            let options = if expected == 0 {
                "[1] first\n(2) second"
            } else {
                "(1) first\n[2] second"
            };
            let prompt = format!("Pick\n\n{options}\n\n(1 - 2, default: {}) > ", expected + 1);
            test_eq!(event.to_string(), prompt);
            let mut out = Vec::new();
            event
                .handle_event(&mut &b"\n"[..], &mut out, &mut Vec::new(), no_secret)
                .unwrap();
            test_eq!(answer.try_recv().unwrap(), expected);
            test_eq!(out, prompt.as_bytes());
        }
    }

    #[test]
    fn choice_retries_invalid_numbers_and_returns_zero_based_index() {
        let (reply, mut answer) = oneshot::channel();
        let event = EngineEvent::Choose {
            message: "Pick".into(),
            options: vec!["first".into(), "second".into()],
            default: 0,
            reply,
        };
        let prompt = event.to_string();
        let flushes = Rc::new(Cell::new(0));
        let mut out = FlushWriter {
            bytes: Vec::new(),
            flushes: Rc::clone(&flushes),
        };
        event
            .handle_event(
                &mut &b"0\n3\nnot a number\n 2 \n"[..],
                &mut out,
                &mut Vec::new(),
                no_secret,
            )
            .unwrap();
        test_eq!(answer.try_recv().unwrap(), 1);
        test_eq!(flushes.get(), 4);
        let expected = format!(
            "{prompt}{}",
            "Enter one of the numbered options.\n> ".repeat(3)
        );
        test_eq!(out.bytes, expected.as_bytes());
    }

    #[test]
    fn empty_choices_fail_without_reading_or_writing() {
        let (reply, mut answer) = oneshot::channel();
        let event = EngineEvent::Choose {
            message: "Pick".into(),
            options: Vec::new(),
            default: usize::MAX,
            reply,
        };
        test_eq!(event.to_string(), "Pick\n\n(no options available)\n");
        let mut input = &b"1\n"[..];
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let error = event
            .handle_event(&mut input, &mut out, &mut err, no_secret)
            .unwrap_err();
        test_eq!(error.kind(), io::ErrorKind::InvalidInput);
        test_eq!(input, b"1\n");
        assert!(out.is_empty());
        assert!(err.is_empty());
        test_eq!(answer.try_recv(), Err(oneshot::error::TryRecvError::Closed));
    }

    #[test]
    fn line_requests_preserve_input_and_render_one_prompt() {
        for message in [None, Some("Question".to_string())] {
            let (reply, mut answer) = oneshot::channel();
            let prompt = format!("{}\n\n> ", message.as_deref().unwrap_or_default());
            let event = EngineEvent::Line { message, reply };
            test_eq!(event.to_string(), prompt);
            let mut out = Vec::new();
            event
                .handle_event(
                    &mut &b" answer \r\n"[..],
                    &mut out,
                    &mut Vec::new(),
                    no_secret,
                )
                .unwrap();
            test_eq!(answer.try_recv().unwrap(), " answer \r\n");
            test_eq!(out, prompt.as_bytes());
        }
    }

    #[test]
    fn secret_reader_runs_after_prompt_flush_without_echo() {
        let (reply, mut answer) = oneshot::channel();
        let flushes = Rc::new(Cell::new(0));
        let mut out = FlushWriter {
            bytes: Vec::new(),
            flushes: Rc::clone(&flushes),
        };
        let event = EngineEvent::Secret {
            message: "API key".into(),
            reply,
        };
        test_eq!(event.to_string(), "API key\n\n> ");
        event
            .handle_event(&mut io::empty(), &mut out, &mut Vec::new(), || {
                test_eq!(flushes.get(), 1);
                Ok("secret".into())
            })
            .unwrap();
        test_eq!(answer.try_recv().unwrap(), "secret");
        test_eq!(out.bytes, b"API key\n\n> ");
    }

    #[test]
    fn secret_read_errors_propagate_without_reply() {
        let (reply, mut answer) = oneshot::channel();
        let event = EngineEvent::Secret {
            message: "API key".into(),
            reply,
        };
        let error = event
            .handle_event(&mut io::empty(), &mut Vec::new(), &mut Vec::new(), || {
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "terminal unavailable",
                ))
            })
            .unwrap_err();
        test_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        test_eq!(answer.try_recv(), Err(oneshot::error::TryRecvError::Closed));
    }

    #[test]
    fn input_errors_propagate_without_reply() {
        for (mut input, expected) in [
            (&b""[..], io::ErrorKind::UnexpectedEof),
            (&b"\xff\n"[..], io::ErrorKind::InvalidData),
        ] {
            let (reply, mut answer) = oneshot::channel();
            let event = EngineEvent::Line {
                message: None,
                reply,
            };
            let error = event
                .handle_event(&mut input, &mut Vec::new(), &mut Vec::new(), no_secret)
                .unwrap_err();
            test_eq!(error.kind(), expected);
            test_eq!(answer.try_recv(), Err(oneshot::error::TryRecvError::Closed));
        }
    }

    #[test]
    fn eof_does_not_accept_confirmation_or_choice_defaults() {
        let (confirm_reply, mut confirm_answer) = oneshot::channel();
        let (choose_reply, mut choose_answer) = oneshot::channel();
        let events = [
            EngineEvent::Confirm {
                message: "Continue?".into(),
                default: true,
                reply: confirm_reply,
            },
            EngineEvent::Choose {
                message: "Pick".into(),
                options: vec!["first".into()],
                default: 0,
                reply: choose_reply,
            },
        ];
        for event in events {
            let error = event
                .handle_event(
                    &mut io::empty(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                    no_secret,
                )
                .unwrap_err();
            test_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        }
        test_eq!(
            confirm_answer.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        );
        test_eq!(
            choose_answer.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        );
    }

    #[test]
    fn output_write_errors_prevent_input_reads() {
        let (reply, mut answer) = oneshot::channel();
        let event = EngineEvent::Secret {
            message: "API key".into(),
            reply,
        };
        let mut out = &mut [][..];
        let error = event
            .handle_event(&mut io::empty(), &mut out, &mut Vec::new(), no_secret)
            .unwrap_err();
        test_eq!(error.kind(), io::ErrorKind::WriteZero);
        test_eq!(answer.try_recv(), Err(oneshot::error::TryRecvError::Closed));
    }

    struct FlushError;

    impl Write for FlushError {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    #[test]
    fn output_flush_errors_prevent_input_reads() {
        let (reply, mut answer) = oneshot::channel();
        let event = EngineEvent::Secret {
            message: "API key".into(),
            reply,
        };
        let error = event
            .handle_event(
                &mut io::empty(),
                &mut FlushError,
                &mut Vec::new(),
                no_secret,
            )
            .unwrap_err();
        test_eq!(error.kind(), io::ErrorKind::Other);
        test_eq!(answer.try_recv(), Err(oneshot::error::TryRecvError::Closed));
    }

    #[test]
    fn dropped_reply_receivers_are_not_io_errors() {
        let (reply, answer) = oneshot::channel();
        drop(answer);
        EngineEvent::Line {
            message: None,
            reply,
        }
        .handle_event(
            &mut &b"answer\n"[..],
            &mut Vec::new(),
            &mut Vec::new(),
            no_secret,
        )
        .unwrap();
    }
}
