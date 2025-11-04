use crate::llm::tools::Tool;

use super::errors::LLMError;
use serde::{Deserialize, Serialize};

/// A user-facing, generic, tool call request. This contains the tool name and the parameters
/// passed to that tool. It also includes the tool call id, which most providers use to
/// disambiguate each tool use response. Note that the *result* of the tool call is not stored
/// here; that is the `ToolCallResponse` struct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Tool calls typically contain an ID. This ID is used in the corresponding response to signal
    /// which results are for which tool calls.
    pub id: String,
    /// The name of the tool being called
    pub tool_name: String,
    /// The parameters passed to the tool
    pub args: serde_json::Value,
}

/// The user-facing version of a tool call result. This contains the id of the tool call request
/// made by the model, the tool that was called, and the result. Notably, this does not contain the
/// args passed to the tool; that is only present in `ToolCallRequest`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallResponse {
    /// Tool calls typically contain an ID. This ID matches the request for a specific tool call.
    pub id: String,
    /// The name of the tool being called
    pub tool_name: String,
    /// The tool call result
    pub result: serde_json::Value,
}

/// This enum specifies the main kinds of content that is passed between the servers and us,
/// the client. It records a single piece of content in the *chat history*, so it includes
/// an enum variant for tool call results as well. The type that is passed as a single *response*
/// from the model is `ContentType`, which does not include that variant. `ContentType` also
/// differs in that `ContentType::ToolCall` includes all the metadata for each tool call, while the
/// variants here only contain what is pertinent to common API schemas.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChatHistoryContent {
    Text(String),
    ToolCallRequest(ToolCallRequest),
    ToolCallResponse(ToolCallResponse),
}

/// An element of the chat history. This records all previous chat interactions, including tool
/// requests from the model and the results of those tool calls. This is a user-facing struct: this
/// is internally converted to provider-specific implementations.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatHistoryItem {
    /// Typically "user", "assistant", etc.
    pub role: String,
    /// The contents of this item. This is a `Vec` because some APIs expect tool call requests to
    /// be bundled together with the text preceding it.
    pub content: Vec<ChatHistoryContent>,
}

/// A user-facing struct that does not carry API-specific information. Clients should
/// convert from this to native message types.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UserMessage {
    pub chat_history: Vec<ChatHistoryItem>,
    pub max_tokens: Option<u32>,
    pub message: String,
}

/// Represents a request to the chat API, combining a user message with optional tools.
pub struct ChatRequest<'a> {
    pub message: &'a UserMessage,
    pub tools: Option<&'a [Box<dyn Tool>]>,
}

/// Ergonomic conversion from a `UserMessage` to a `ChatRequest` without tools.
impl<'a> From<&'a UserMessage> for ChatRequest<'a> {
    fn from(message: &'a UserMessage) -> Self {
        ChatRequest {
            message,
            tools: None,
        }
    }
}

/// A structure dedicated to a single tool call. This contains the tool called, the arguments
/// passed, and the result of that tool call. In the future, this may include additional
/// information such as the number of tokens used in each tool call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolUseStats {
    pub tool_call_id: String,
    pub tool_name: String,
    pub tool_args: serde_json::Value,
    pub tool_result: serde_json::Value,
}

/// A model response can contain multiple types of content, such as raw text and tool calls. This
/// captures those variants. Specifically, this is different from `ChatHistoryContent`, which is
/// more tailored to individual items in the chat history that is passed to providers.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ContentType {
    Text(String),
    ToolCall(ToolUseStats),
}

impl Default for ContentType {
    fn default() -> Self {
        Self::Text("".into())
    }
}

/// A user-facing struct representing API responses, containing only information users
/// would be interested in.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CompletionApiResponse {
    pub content: Vec<ContentType>,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A client that can interact with an LLM provider and get a response.
#[allow(async_fn_in_trait)]
pub trait ApiClient {
    async fn send_message<'a>(
        &self,
        request: &'a ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, LLMError>;
}
