//! Event enum and associated implementations.

/// Enum of events emitted by this crate.
///
/// Events are primarily used by long-running parts of this crate, particularly the
/// [`crate::llm::base::AgenticClient`], to notify consumers of progress. These events are not
/// guarantees of success or failure of an operation: you should instead rely on `Result` return
/// types for that instead. Rather, events are signals of progress, including the final response.
///
/// Fast operations such as embedding and vector search do not emit events.
///
/// Events are not an authoritative conversational record. Functions in this crate help you manage
/// conversation history better; events are most useful in a rendering loop to keep the user
/// informed of progress.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// A piece of text, which may not be the final response.
    Text(String),
    /// Reasoning text or summary.
    Reasoning(String),
    /// A tool/function call.
    ToolCall {
        /// Unique tool call ID.
        id: String,
        /// Tool name
        name: String,
        /// Arguments passed to the tool.
        args: serde_json::Value,
    },
    /// A response to a tool call.
    ToolResponse {
        /// Matching id to the corresponding [`AgentEvent::ToolCall`].
        id: String,
        /// Tool name
        name: String,
        /// Tool response, if successful, or an error message, such as if a tool name was
        /// hallucinated.
        response: Result<serde_json::Value, String>,
    },
}
