//! I/O primitives for working with the agent in `zqa`.

use std::fmt::Display;

use tokio::sync::oneshot;

use crate::state::UsageMetadata;
use crate::utils::terminal::{BLUE, DIM_TEXT, ITALICS, RED, RESET, YELLOW};

/// An enum of events that can be emitted by handlers called by
/// [`crate::cli::app::dispatch_command`]. The contract here is that *every* event, including status
/// updates and error messages will be passed as an event. In general, events carry text, and
/// variants are a semantic layer on top. Structured payloads are used for data; the current
/// exceptions are `ToolCall`, `ToolResponse`, and `TokenUsage`.
///
/// You should use events to enable interactivity in your application; for an example, the CLI in
/// this crate consumes events to print status updates, agent responses, and display errors. Events
/// are not guaranteed to be a reliable mechanism to detect operation failure: you should rely on
/// functions returning a `Result` type for this.
///
/// ## Display
///
/// This enum implements [`std::fmt::Display`], whose implementation is intended for CLI consumers and
/// uses ANSI SGR codes. For variants that elicit user input, it only displays the question and the options
/// (if any), but does not handle input. For multiple-choice questions, this presents the user with
/// numbered options starting from 1. In all cases with options, we follow general CLI standards of
/// presenting the default within [square brackets] and other options in (parentheses).
///
/// * [`EngineEvent::Text`] is rendered as-is.
/// * [`EngineEvent::ToolCall`] is rendered in a dimmed text with the tool name followed by
///   JSON args.
/// * [`EngineEvent::ToolResponse`] is rendered in a dimmed text with two leading spaces, followed
///   by a Unicode right arrow (→), followed by the tool name in parentheses, and finally the
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
    Choose {
        message: String,
        options: Vec<String>,
        reply: oneshot::Sender<usize>,
        /// A 0-based index into `options`. Guaranteed to be strictly less than `options.len()`;
        /// emitted events that violate this will have `default` reset to 0.
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

impl Display for EngineEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineEvent::Text { message } => f.write_str(message),
            EngineEvent::ToolCall { name, args, .. } => write!(f, "{DIM_TEXT}{name} {args}{RESET}"),
            EngineEvent::ToolResponse { name, response, .. } => {
                let response_str = match response {
                    Ok(res) => &serde_json::to_string_pretty(res).unwrap_or("success".into()),
                    Err(e) => e,
                };
                write!(f, "  {DIM_TEXT}→ ({name}) {response_str}{RESET}")
            }
            EngineEvent::Reasoning { message } => write!(f, "{DIM_TEXT}{ITALICS}{message}{RESET}"),
            EngineEvent::Confirm {
                message, default, ..
            } => {
                // It is not semantically reasonable for `Display` to handle inputs of any kind;
                // that said, neither choice of whether to print the message is universally
                // intuitive, so we make the choice here that `Display` displays the message and the
                // options, and a separate input handler should assume `to_string()` or an
                // equivalent has been called and only handle input.
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
                let default = *default.min(&0);
                write!(f, "{message}\n\n")?;
                for (i, opt) in options.iter().enumerate() {
                    write!(
                        f,
                        "{} {opt}\n",
                        if i == default {
                            format!("[{}]", i + 1)
                        } else {
                            format!("({})", i + 1)
                        }
                    )?;
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
