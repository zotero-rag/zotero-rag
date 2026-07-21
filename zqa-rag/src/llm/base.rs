//! User-facing types for working with LLMs, including tool calling support. Most structs used by
//! the clients can be converted to/from the structs here.

use std::sync::Arc;

use http::HeaderMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::errors::LLMError;
use crate::{
    constants::{DEFAULT_MAX_RETRIES, DEFAULT_MAX_TOOL_ITERATIONS},
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

// TODO: The driver options probably don't belong here; it might at some point make sense to make a
// new `Request<'a>` or similar type, that has this struct along with a `DriverOptions` or
// something. It's not yet clear *when* (i.e., at how many fields) that transition should happen.
/// Represents a request to the chat API with optional tools, along with options for the behavior of
/// the agentic loop itself.
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

    // Driver options: these must be `Option<T>`; semantically, it doesn't make sense to force
    // end-users into supplying these options.
    /// Optional callback invoked each time a tool call completes.
    pub on_tool_call: Option<Arc<CallbackFn<ToolUseStats>>>,
    /// Optional callback invoked each time a text chunk is produced.
    pub on_text: Option<Arc<CallbackFn<str>>>,
    /// Optional limit on the number of tool call iterations per user message.
    pub tool_iteration_limit: Option<usize>,
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
    /// * Returns [`LLMError`] if a provider request fails.
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

        let mut round_trips = 0;
        let iteration_limit = request
            .tool_iteration_limit
            .unwrap_or(DEFAULT_MAX_TOOL_ITERATIONS);
        while round_trips < iteration_limit {
            let is_last_turn = round_trips == iteration_limit.saturating_sub(1);
            let tools_passed = if is_last_turn {
                // On the last trip, disallow tool calls
                None
            } else {
                tools.as_deref()
            };

            let turn = self
                .send_once(
                    &history,
                    tools_passed,
                    request.reasoning.as_ref(),
                    request.max_tokens,
                )
                .await?;
            usage += turn.usage;
            history.extend(turn.native_items);

            let tool_call_results = process_tool_calls(
                &mut contents,
                &turn.contents,
                if is_last_turn {
                    &[]
                } else {
                    tools.as_deref().unwrap_or_default()
                },
                request.on_tool_call.as_ref(),
                request.on_text.as_ref(),
            )
            .await;

            if tool_call_results.is_empty() {
                break;
            } else if iteration_limit > 1 && round_trips == iteration_limit - 2 {
                // On the next turn, tool calls will be disallowed, so we should log this.
                log::warn!(
                    "Reached penultimate iteration ({round_trips}), last trip will not allow tool calling."
                );
            }

            history.extend(tool_call_results.into_iter().flat_map(Into::into));
            round_trips += 1;
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
    let json: serde_json::Value = serde_json::from_str(&body).map_err(|err| {
        log::error!("Failed to parse response body as JSON: {err}. Body: {body}");
        LLMError::DeserializationError(body.clone())
    })?;
    let response: S = serde_json::from_value(json).map_err(|err| {
        log::error!("Failed to deserialize response into target type: {err}. Body: {body}");
        LLMError::DeserializationError(body)
    })?;

    Ok(response)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
    };

    use super::*;
    use crate::llm::tools::test_utils::MockTool;

    struct TestHistoryItem;

    impl From<ChatHistoryItem> for Vec<TestHistoryItem> {
        fn from(_: ChatHistoryItem) -> Self {
            Vec::new()
        }
    }

    struct TestClient {
        turns: Mutex<VecDeque<ProviderTurn<TestHistoryItem>>>,
        tools_seen: Arc<Mutex<Vec<Option<usize>>>>,
    }

    impl AgenticClient for TestClient {
        type HistoryItem = TestHistoryItem;
        const SCHEMA_KEY: &'static str = "parameters";

        fn build_initial_history(&self, _: &ChatRequest<'_>) -> Vec<Self::HistoryItem> {
            Vec::new()
        }

        async fn send_once(
            &self,
            _: &[Self::HistoryItem],
            tools: Option<&[SerializedTool<'_>]>,
            _: Option<&ReasoningConfig>,
            _: Option<u32>,
        ) -> Result<ProviderTurn<Self::HistoryItem>, LLMError> {
            self.tools_seen.lock().unwrap().push(tools.map(<[_]>::len));
            Ok(self.turns.lock().unwrap().pop_front().unwrap())
        }
    }

    fn tool_call_turn() -> ProviderTurn<TestHistoryItem> {
        ProviderTurn {
            native_items: Vec::new(),
            contents: vec![ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                id: "call-1".into(),
                tool_name: "mock_tool".into(),
                args: serde_json::json!({"name": "Alice"}),
            })],
            usage: ModelUsage::default(),
        }
    }

    #[tokio::test]
    async fn tool_iteration_limit_disables_tools_on_last_turn() {
        let tools_seen = Arc::new(Mutex::new(Vec::new()));
        let client = TestClient {
            turns: Mutex::new(VecDeque::from([
                tool_call_turn(),
                ProviderTurn {
                    native_items: Vec::new(),
                    contents: vec![ChatHistoryContent::Text("done".into())],
                    usage: ModelUsage::default(),
                },
            ])),
            tools_seen: Arc::clone(&tools_seen),
        };
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };
        let request = ChatRequest {
            tools: Some(&[Box::new(tool)]),
            tool_iteration_limit: Some(2),
            ..ChatRequest::default()
        };

        let response = client.send_message(&request).await.unwrap();

        assert_eq!(*tools_seen.lock().unwrap(), vec![Some(1), None]);
        assert_eq!(*call_count.lock().unwrap(), 1);
        assert!(matches!(
            response.content.as_slice(),
            [ContentType::ToolCall(_), ContentType::Text(text)] if text == "done"
        ));
    }

    #[tokio::test]
    async fn tool_iteration_limit_of_one_does_not_underflow() {
        let tools_seen = Arc::new(Mutex::new(Vec::new()));
        let client = TestClient {
            turns: Mutex::new(VecDeque::from([ProviderTurn {
                native_items: Vec::new(),
                contents: vec![ChatHistoryContent::Text("done".into())],
                usage: ModelUsage::default(),
            }])),
            tools_seen: Arc::clone(&tools_seen),
        };
        let tool = MockTool {
            call_count: Arc::new(Mutex::new(0)),
        };
        let request = ChatRequest {
            tools: Some(&[Box::new(tool)]),
            tool_iteration_limit: Some(1),
            ..ChatRequest::default()
        };

        let response = client.send_message(&request).await.unwrap();

        assert_eq!(*tools_seen.lock().unwrap(), vec![None]);
        assert_eq!(response.content.len(), 1);
    }
}
