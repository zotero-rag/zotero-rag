//! User-facing structs and traits for working with LLMs, including tool calling support. Most
//! structs used by the clients can be converted to/from the structs here.

use crate::llm::tools::Tool;

use super::errors::LLMError;
use serde::{Deserialize, Serialize};

/// The user role.
pub const USER_ROLE: &str = "user";
/// The assistant role.
pub const ASSISTANT_ROLE: &str = "assistant";

/// A user-facing, generic, tool call request. This contains the tool name and the parameters
/// passed to that tool. It also includes the tool call id, which most providers use to
/// disambiguate each tool use response. Note that the *result* of the tool call is not stored
/// here; that is the `ToolCallResponse` struct.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChatHistoryContent {
    /// A plain-text message
    Text(String),
    /// A tool call request
    ToolCallRequest(ToolCallRequest),
    /// A tool call response
    ToolCallResponse(ToolCallResponse),
}

/// An element of the chat history. This records all previous chat interactions, including tool
/// requests from the model and the results of those tool calls. This is a user-facing struct: this
/// is internally converted to provider-specific implementations.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ChatHistoryItem {
    /// The chat role. Every supported provider has a "user" (`USER_ROLE`) and an "assistant"
    /// (`ASSISTANT_ROLE`) role.
    pub role: String,
    /// The contents of this item. This is a `Vec` because some APIs expect tool call requests to
    /// be bundled together with the text preceding it.
    pub content: Vec<ChatHistoryContent>,
}

/// Represents a request to the chat API with optional tools.
pub struct ChatRequest<'a> {
    /// The chat history
    pub chat_history: Vec<ChatHistoryItem>,
    /// The maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// The message to send
    pub message: String,
    /// The tools to use
    pub tools: Option<&'a [Box<dyn Tool>]>,
}

/// A structure dedicated to a single tool call. This contains the tool called, the arguments
/// passed, and the result of that tool call. In the future, this may include additional
/// information such as the number of tokens used in each tool call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolUseStats {
    /// The provider-specified ID for the tool call
    pub tool_call_id: String,
    /// The name of the tool called
    pub tool_name: String,
    /// The arguments passed to the tool
    pub tool_args: serde_json::Value,
    /// The result of the tool call
    pub tool_result: serde_json::Value,
}

/// A model response can contain multiple types of content, such as raw text and tool calls. This
/// captures those variants. Specifically, this is different from `ChatHistoryContent`, which is
/// more tailored to individual items in the chat history that is passed to providers.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ContentType {
    /// A plain-text message
    Text(String),
    /// A tool call
    ToolCall(ToolUseStats),
}

impl Default for ContentType {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// A struct representing API responses, containing only information users
/// would be interested in.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CompletionApiResponse {
    /// The content of the response. Note that this is a *single* response from the API, which can
    /// contain multiple types of content.
    pub content: Vec<ContentType>,
    /// The number of tokens used in the input prompt
    pub input_tokens: u32,
    /// The number of tokens used in the output
    pub output_tokens: u32,
}

/// A client that can interact with an LLM provider and get a response.
#[allow(async_fn_in_trait)]
pub trait ApiClient {
    /// Send a request to the API and return the response.
    async fn send_message<'a>(
        &self,
        request: &'a ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, LLMError>;
}
