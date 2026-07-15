//! User-facing types for working with LLMs, including tool calling support. Most structs used by
//! the clients can be converted to/from the structs here.

use std::sync::Arc;

use http::HeaderMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::errors::LLMError;
use crate::{
    constants::DEFAULT_MAX_RETRIES,
    http_client::HttpClient,
    llm::tools::{CallbackFn, SerializedTool, Tool, get_owned_tools, process_tool_calls},
    pricing::ModelUsage,
    requests::request_with_backoff,
};

/// Roles for messages
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// A user message/response. Some providers fold tool responses into this.
    User,
    /// An assistant response.
    Assistant,
    /// Some providers, like OpenRouter, use a separate role for tool responses.
    Tool,
}

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
    pub role: MessageRole,
    /// The contents of this item. This is a `Vec` because some APIs expect tool call requests to
    /// be bundled together with the text preceding it.
    pub content: Vec<ChatHistoryContent>,
}

impl From<Vec<ContentType>> for ChatHistoryItem {
    /// Creates a [`ChatHistoryItem`] from a [`Vec<ContentType>`]. In general, the reason you would
    /// want to do this is to conveniently add the response from a model (which can include text
    /// and multiple tool calls) into [`ChatRequest`]. For this reason, this `from` implementation
    /// sets the `role` to [`ASSISTANT_ROLE`].
    fn from(value: Vec<ContentType>) -> Self {
        let content = value
            .into_iter()
            .map(|ct| match ct {
                ContentType::Text(s) => ChatHistoryContent::Text(s),
                ContentType::ToolCall(stats) => {
                    ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                        id: stats.tool_call_id,
                        tool_name: stats.tool_name,
                        result: stats.tool_result,
                    })
                }
            })
            .collect();

        Self {
            role: MessageRole::Assistant,
            content,
        }
    }
}

/// Provider-agnostic reasoning configuration
#[derive(Debug, Clone)]
pub struct ReasoningConfig {
    /// Token budget for thinking.
    pub max_tokens: Option<u32>,
    /// Reasoning effort for providers that support it.
    pub effort: Option<String>,
    /// Thinking summary length, for providers that support it.
    pub summary: Option<String>,
}

/// Represents a request to the chat API with optional tools.
#[derive(Default)]
pub struct ChatRequest<'a> {
    /// The chat history
    pub chat_history: Vec<ChatHistoryItem>,
    /// The maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// The message to send
    pub message: String,
    /// Reasoning config. Optional.
    pub reasoning: Option<ReasoningConfig>,
    /// The tools to use
    pub tools: Option<&'a [Box<dyn Tool>]>,
    /// Optional callback invoked each time a tool call completes.
    pub on_tool_call: Option<Arc<CallbackFn<ToolUseStats>>>,
    /// Optional callback invoked each time a text chunk is produced.
    pub on_text: Option<Arc<CallbackFn<str>>>,
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
    /// Token usage statistics for the request.
    pub usage: ModelUsage,
}

/// An abstraction over one model turn.
pub(crate) struct ProviderTurn<H> {
    /// The model's output in the provider-native format.
    pub(crate) native_items: Vec<H>,
    /// Provider-agnostic view of the same output, for tool dispatch and final-text extraction.
    pub(crate) contents: Vec<ChatHistoryContent>,
    /// Token usage
    pub(crate) usage: ModelUsage,
}

/// Internal contract for provider-specific generation adapters.
pub(crate) trait AgenticClient
where
    ChatHistoryItem: Into<Vec<Self::HistoryItem>>,
{
    type HistoryItem;
    const SCHEMA_KEY: &'static str;

    /// Build the initial conversation history in the provider-native format given a
    /// [`ChatRequest`].
    fn build_initial_history(&self, request: &ChatRequest<'_>) -> Vec<Self::HistoryItem>;

    /// Perform one provider request-response round trip and convert the response into native and
    /// provider-agnostic history items.
    async fn send_once(
        &self,
        history: &[Self::HistoryItem],
        tools: Option<&[SerializedTool<'_>]>,
        reasoning: Option<&ReasoningConfig>,
        max_tokens: Option<u32>,
    ) -> Result<ProviderTurn<Self::HistoryItem>, LLMError>;

    /// Send a chat request, executing requested tools until the provider returns a final response.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat request to send.
    ///
    /// # Returns
    ///
    /// The final response and accumulated usage across all provider turns.
    ///
    /// # Errors
    ///
    /// * Returns [`LLMError`] if a provider request or tool execution fails.
    async fn send_message(
        &self,
        request: &ChatRequest<'_>,
    ) -> Result<CompletionApiResponse, LLMError>
    where
        Self: Sized,
    {
        let mut history = self.build_initial_history(request);
        let tools = get_owned_tools(request.tools, Self::SCHEMA_KEY);
        let mut usage = ModelUsage::default();
        let mut contents = Vec::<ContentType>::new();

        // TODO: Add max tool iterations support
        loop {
            let turn = self
                .send_once(
                    &history,
                    tools.as_deref(),
                    request.reasoning.as_ref(),
                    request.max_tokens,
                )
                .await?;
            usage += turn.usage;
            history.extend(turn.native_items);

            let new_history_items = process_tool_calls(
                &mut contents,
                &turn.contents,
                tools.as_deref().unwrap_or_default(),
                request.on_tool_call.as_ref(),
                request.on_text.as_ref(),
            )
            .await?;

            if new_history_items.is_empty() {
                break;
            }

            history.extend(new_history_items.into_iter().flat_map(Into::into));
        }

        Ok(CompletionApiResponse {
            content: contents,
            usage,
        })
    }
}

pub(crate) async fn send_generation_request<R, S>(
    client: &impl HttpClient,
    request: R,
    headers: &HeaderMap,
    api_url: &str,
) -> Result<S, LLMError>
where
    R: Serialize + Send + Sync,
    S: DeserializeOwned,
{
    let res = request_with_backoff(client, api_url, headers, &request, DEFAULT_MAX_RETRIES).await?;

    let body = res.text().await?;
    let json: serde_json::Value = match serde_json::from_str(&body) {
        Ok(json) => json,
        Err(_) => return Err(LLMError::DeserializationError(body)),
    };

    let response: S = match serde_json::from_value(json) {
        Ok(response) => response,
        Err(_) => return Err(LLMError::DeserializationError(body)),
    };

    Ok(response)
}
