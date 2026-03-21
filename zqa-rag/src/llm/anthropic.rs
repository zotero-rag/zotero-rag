//! A module for interacting with Anthropic's API.

use std::env;

use http::HeaderMap;
use serde::{Deserialize, Serialize};

use super::base::{ApiClient, ChatHistoryItem, ChatRequest, CompletionApiResponse};
use super::errors::LLMError;
use crate::clients::anthropic::AnthropicClient;
use crate::common::request_with_backoff;
use crate::constants::{
    DEFAULT_ANTHROPIC_MAX_TOKENS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_MAX_RETRIES,
};
use crate::http_client::HttpClient;
use crate::llm::base::{
    ChatHistoryContent, ContentType, ToolCallRequest, ToolCallResponse, USER_ROLE,
};
use crate::llm::tools::{
    ANTHROPIC_SCHEMA_KEY, SerializedTool, get_owned_tools, process_tool_calls,
};
const DEFAULT_CLAUDE_MODEL: &str = DEFAULT_ANTHROPIC_MODEL;

/// An Anthropic-specific chat history object. This is pretty much the same as `ChatHistoryItem`,
/// except that `content` is now Anthropic-specific.
#[derive(Clone, Serialize)]
pub(crate) struct AnthropicChatHistoryItem {
    /// Either "user" or "assistant".
    pub role: String,
    /// The contents of this item.
    pub content: Vec<AnthropicResponseContent>,
}

impl From<ChatHistoryItem> for AnthropicChatHistoryItem {
    fn from(value: ChatHistoryItem) -> Self {
        let role = value.role.clone();
        Self {
            content: value
                .content
                .into_iter()
                .map(|ct| match ct {
                    ChatHistoryContent::Text(s) => {
                        AnthropicResponseContent::Text(AnthropicTextResponseContent {
                            r#type: "text".into(),
                            text: s,
                        })
                    }
                    ChatHistoryContent::ToolCallRequest(req) => {
                        AnthropicResponseContent::ToolCall(req.into())
                    }
                    ChatHistoryContent::ToolCallResponse(res) => {
                        AnthropicResponseContent::ToolResult(res.into())
                    }
                })
                .collect(),
            role,
        }
    }
}

impl From<&ChatHistoryItem> for AnthropicChatHistoryItem {
    fn from(value: &ChatHistoryItem) -> Self {
        value.clone().into()
    }
}

/// Represents a request to the Anthropic API
#[derive(Serialize)]
pub(crate) struct AnthropicRequest<'a> {
    /// The model to use for the request (e.g., "claude-4-5-sonnet")
    pub(crate) model: &'a str,
    /// The maximum number of tokens that can be generated in the response
    pub(crate) max_tokens: u32,
    /// The conversation history and current message
    pub(crate) messages: &'a [AnthropicChatHistoryItem],
    /// The tools passed in
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<&'a [SerializedTool<'a>]>,
}

/// Helper to build messages and tools from a ChatRequest.
/// Returns owned data that can then be borrowed by AnthropicRequest.
pub(crate) fn build_anthropic_messages_and_tools<'a>(
    req: &'a ChatRequest<'a>,
) -> (
    Vec<AnthropicChatHistoryItem>,
    Option<Vec<SerializedTool<'a>>>,
) {
    let mut messages: Vec<AnthropicChatHistoryItem> = req
        .chat_history
        .clone()
        .into_iter()
        .map(Into::into)
        .collect();

    messages.push(AnthropicChatHistoryItem {
        role: USER_ROLE.to_owned(),
        content: vec![req.message.clone().into()],
    });

    let owned_tools: Option<Vec<SerializedTool<'a>>> =
        get_owned_tools(req.tools, ANTHROPIC_SCHEMA_KEY);

    (messages, owned_tools)
}

/// Token usage statistics returned by the Anthropic API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicUsageStats {
    /// Number of tokens in the input prompt
    pub(crate) input_tokens: u32,
    /// Number of tokens in the generated response
    pub(crate) output_tokens: u32,
}

/// The result of a tool call. This is the Anthropic-specific result format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicToolUseResult {
    /// The content type. This should always be "tool_result".
    r#type: String,
    /// The ID of the tool call request that this is a result for.
    tool_use_id: String,
    /// The result from the tool call.
    content: String,
}

impl From<ToolCallResponse> for AnthropicToolUseResult {
    fn from(value: ToolCallResponse) -> Self {
        Self {
            r#type: "tool_result".into(),
            tool_use_id: value.id,
            content: match value.result {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            },
        }
    }
}

/// A thinking block returned by models with extended thinking enabled.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicThinkingResponseContent {
    /// The content type. Always "thinking".
    pub(crate) r#type: String,
    /// The thinking content from the model
    pub(crate) thinking: String,
}

/// A part of an Anthropic API response denoting some text from the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicTextResponseContent {
    /// The content type. Always "text".
    pub(crate) r#type: String,
    /// The text content from the model's response
    pub(crate) text: String,
}

/// A part of an Anthropic API response denoting a tool call (Anthropic uses the term "tool use" or
/// "function call"). This is *not* the struct containing the *result* of that tool: that is
/// `AnthropicToolUseResult`. Note that this struct is also used as part of
/// `AnthropicChatHistoryItem`, as one of the possible content types.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicToolUseResponseContent {
    /// Tool use ID
    pub(crate) id: String,
    /// The content type. Always "tool_use".
    pub(crate) r#type: String,
    /// The tool being called
    pub(crate) name: String,
    /// The input to the tool. Note that all we can guarantee is that the keys are strings
    pub(crate) input: serde_json::Map<String, serde_json::Value>,
}

impl From<ToolCallRequest> for AnthropicToolUseResponseContent {
    fn from(value: ToolCallRequest) -> Self {
        Self {
            id: value.id,
            r#type: "tool_use".into(),
            name: value.tool_name,
            input: serde_json::Value::as_object(&value.args)
                .unwrap_or(&serde_json::Map::new())
                .clone(),
        }
    }
}

/// Content block in an Anthropic API response. This is also used in the request as the chat
/// history.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum AnthropicResponseContent {
    /// A plain-text content, which is used in requests and the user parts of chat history.
    PlainText(String),
    /// A text part of a response
    Text(AnthropicTextResponseContent),
    ToolCall(AnthropicToolUseResponseContent),
    /// Only used in the chat history; this can never be in the response from the API.
    ToolResult(AnthropicToolUseResult),
    /// A thinking block from models with extended thinking enabled.
    Thinking(AnthropicThinkingResponseContent),
}

impl From<String> for AnthropicResponseContent {
    fn from(value: String) -> Self {
        Self::Text(AnthropicTextResponseContent {
            r#type: "text".into(),
            text: value,
        })
    }
}

/// Response from the Anthropic API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicResponse {
    /// Unique identifier for the response
    pub(crate) id: String,
    /// The model that generated the response
    pub(crate) model: String,
    /// The role of the message (usually "assistant")
    pub(crate) role: String,
    /// Why the model stopped generating (e.g., "end_turn")
    pub(crate) stop_reason: String,
    /// The stop sequence that caused generation to end, if any
    pub(crate) stop_sequence: Option<String>,
    /// Token usage statistics
    pub(crate) usage: AnthropicUsageStats,
    /// The type of the response (usually "message")
    pub(crate) r#type: String,
    /// The content blocks in the response
    pub(crate) content: Vec<AnthropicResponseContent>,
}

/// Send an API request to Anthropic.
///
/// This function takes references to a client, a set of headers, and an Anthropic-specific
/// request body, and sends a request with exponential backoff. This acts as a helper function so
/// that the calling code can be easier to follow.
///
/// # Arguments:
///
/// * `client`: An `HttpClient` implementation.
/// * `headers`: A set of headers to pass.
/// * `req`: The request body to send
///
/// # Returns
///
/// The Anthropic-specific response.
async fn send_anthropic_request(
    client: &impl HttpClient,
    headers: &HeaderMap,
    req: &AnthropicRequest<'_>,
) -> Result<AnthropicResponse, LLMError> {
    const MAX_RETRIES: usize = DEFAULT_MAX_RETRIES;
    let res = request_with_backoff(
        client,
        "https://api.anthropic.com/v1/messages",
        headers,
        req,
        MAX_RETRIES,
    )
    .await?;

    let body = res.text().await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let response: AnthropicResponse = serde_json::from_value(json.clone()).map_err(|err| {
        eprintln!("Failed to deserialize Anthropic response: we got the response {json}");

        LLMError::DeserializationError(err.to_string())
    })?;

    Ok(response)
}

/// Convert Anthropic response content into provider-agnostic `ChatHistoryContent` items.
///
/// Tool results should never appear in Anthropic API responses; if encountered, they are ignored
/// with a warning.
pub(crate) fn map_response_to_chat_contents(
    contents: &[AnthropicResponseContent],
) -> Vec<ChatHistoryContent> {
    let mut out = Vec::new();
    for c in contents {
        match c {
            AnthropicResponseContent::PlainText(s) => out.push(ChatHistoryContent::Text(s.clone())),
            AnthropicResponseContent::Text(t) => out.push(ChatHistoryContent::Text(t.text.clone())),
            AnthropicResponseContent::ToolCall(tc) => {
                out.push(ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                    id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    args: serde_json::Value::from(tc.input.clone()),
                }));
            }
            AnthropicResponseContent::ToolResult(_) => {
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be ignored."
                );
            }
            AnthropicResponseContent::Thinking(_) => {}
        }
    }
    out
}

impl<T: HttpClient> ApiClient for AnthropicClient<T> {
    /// Send a request to the Anthropic API, processing tool calls as necessary. Returns a final
    /// response after all tool calls are processed and sent back to the API.
    async fn send_message<'a>(
        &self,
        request: &'a ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, LLMError> {
        // Use config if available, otherwise fall back to env vars
        let (api_key, model, max_tokens) = if let Some(ref config) = self.config {
            (
                config.api_key.clone(),
                config.model.clone(),
                config.max_tokens,
            )
        } else {
            (
                env::var("ANTHROPIC_API_KEY")?,
                env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_CLAUDE_MODEL.to_string()),
                DEFAULT_ANTHROPIC_MAX_TOKENS,
            )
        };

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", api_key.parse()?);
        headers.insert("anthropic-version", "2023-06-01".parse()?);
        headers.insert("content-type", "application/json".parse()?);

        // Build the initial messages and tools (owned)
        let (mut chat_history, tools) = build_anthropic_messages_and_tools(request);
        let max_tokens_to_use = request.max_tokens.unwrap_or(max_tokens);

        // Create the initial request
        let req_body = AnthropicRequest {
            model: &model,
            max_tokens: max_tokens_to_use,
            messages: &chat_history,
            tools: tools.as_deref(),
        };

        let mut response = send_anthropic_request(&self.client, &headers, &req_body).await?;

        let mut has_tool_calls: bool = response
            .content
            .iter()
            .any(|c| matches!(c, AnthropicResponseContent::ToolCall { .. }));

        // Append the contents
        chat_history.push(AnthropicChatHistoryItem {
            role: "assistant".into(),
            content: response.content.clone(),
        });

        let mut contents: Vec<ContentType> = Vec::new();

        while has_tool_calls {
            let converted_contents = map_response_to_chat_contents(&response.content);
            process_tool_calls(
                &mut chat_history,
                &mut contents,
                &converted_contents,
                tools.as_ref().ok_or_else(|| {
                    LLMError::ToolCallError(
                        "Model returned tool calls, but no tools were provided.".to_string(),
                    )
                })?,
                request.on_tool_call.as_ref(),
                request.on_text.as_ref(),
            )
            .await?;

            // Create a new request borrowing the updated chat history
            let updated_req_body = AnthropicRequest {
                model: &model,
                max_tokens: max_tokens_to_use,
                messages: &chat_history,
                tools: tools.as_deref(),
            };

            response = send_anthropic_request(&self.client, &headers, &updated_req_body).await?;

            // Append the new response to chat history
            chat_history.push(AnthropicChatHistoryItem {
                role: "assistant".into(),
                content: response.content.clone(),
            });

            has_tool_calls = response
                .content
                .iter()
                .any(|c| matches!(c, AnthropicResponseContent::ToolCall { .. }));
        }

        // Process the final response (which has no tool calls) to extract text content
        for content in &response.content {
            match content {
                AnthropicResponseContent::PlainText(s) => {
                    if let Some(cb) = request.on_text.as_ref() {
                        cb(s);
                    }
                    contents.push(ContentType::Text(s.clone()));
                }
                AnthropicResponseContent::Text(text_content) => {
                    if let Some(cb) = request.on_text.as_ref() {
                        cb(&text_content.text);
                    }
                    contents.push(ContentType::Text(text_content.text.clone()));
                }
                _ => {}
            }
        }

        // TODO: Check if this metadata includes tool use
        Ok(CompletionApiResponse {
            content: contents,
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use dotenv::dotenv;
    use zqa_macros::{test_eq, test_ok};

    use crate::http_client::{MockHttpClient, ReqwestClient, SequentialMockHttpClient};
    use crate::llm::anthropic::{AnthropicTextResponseContent, DEFAULT_CLAUDE_MODEL};
    use crate::llm::base::{
        ApiClient, ChatHistoryContent, ChatHistoryItem, ChatRequest, ContentType, ToolCallResponse,
        USER_ROLE,
    };
    use crate::llm::tools::test_utils::MockTool;

    use super::{
        AnthropicClient, AnthropicResponse, AnthropicResponseContent,
        AnthropicToolUseResponseContent, AnthropicUsageStats,
    };

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = AnthropicClient::<ReqwestClient>::default();
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
            tools: None,
            on_tool_call: None,
            on_text: None,
        };

        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Anthropic test error: {:?}", res.as_ref().err());
        }

        test_ok!(res);
    }

    #[tokio::test]
    async fn test_request_with_mock() {
        // Load environment variables from .env file
        dotenv().ok();

        // Create a proper AnthropicResponse that matches the structure we expect to deserialize
        let mock_response = AnthropicResponse {
            id: "mock-id".to_string(),
            model: DEFAULT_CLAUDE_MODEL.to_string(),
            role: "assistant".to_string(),
            stop_reason: "end_turn".to_string(),
            stop_sequence: None,
            usage: AnthropicUsageStats {
                input_tokens: 9,
                output_tokens: 13,
            },
            r#type: "message".to_string(),
            content: vec![AnthropicResponseContent::Text(
                AnthropicTextResponseContent {
                    r#type: "text".into(),
                    text: "Hi there! How can I help you today?".into(),
                },
            )],
        };

        let mock_http_client = MockHttpClient::new(mock_response);
        let mock_client = AnthropicClient {
            client: mock_http_client,
            config: None,
        };

        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
            tools: None,
            on_tool_call: None,
            on_text: None,
        };

        let res = mock_client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Anthropic test error: {:?}", res.as_ref().err());
        }

        test_ok!(res);

        let res = res.unwrap();
        test_eq!(res.input_tokens, 9);
        test_eq!(res.output_tokens, 13);

        let content = &res.content[0];
        if let ContentType::Text(text) = content {
            test_eq!(text, "Hi there! How can I help you today?");
        } else {
            panic!("Expected Text content type");
        }
    }

    #[tokio::test]
    async fn test_request_with_tool_works() {
        dotenv().ok();

        let client = AnthropicClient::<ReqwestClient>::default();
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "This is a test. Call the `mock_tool`, passing in a `name`, and ensure it returns a greeting".into(),
            tools: Some(&[Box::new(tool)]),
            on_tool_call: None,
            on_text: None,
        };
        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Anthropic test error: {:?}", res.as_ref().err());
        }

        test_ok!(res);
        assert!(call_count.lock().unwrap().eq(&1_usize));
    }

    #[tokio::test]
    async fn test_callbacks_fire() {
        dotenv().ok();

        let tool_call_response = AnthropicResponse {
            id: "msg-1".into(),
            model: DEFAULT_CLAUDE_MODEL.into(),
            role: "assistant".into(),
            stop_reason: "tool_use".into(),
            stop_sequence: None,
            usage: AnthropicUsageStats {
                input_tokens: 10,
                output_tokens: 5,
            },
            r#type: "message".into(),
            content: vec![AnthropicResponseContent::ToolCall(
                AnthropicToolUseResponseContent {
                    id: "tool-1".into(),
                    r#type: "tool_use".into(),
                    name: "mock_tool".into(),
                    input: serde_json::json!({"name": "Alice"})
                        .as_object()
                        .unwrap()
                        .clone(),
                },
            )],
        };
        let text_response = AnthropicResponse {
            id: "msg-2".into(),
            model: DEFAULT_CLAUDE_MODEL.into(),
            role: "assistant".into(),
            stop_reason: "end_turn".into(),
            stop_sequence: None,
            usage: AnthropicUsageStats {
                input_tokens: 20,
                output_tokens: 8,
            },
            r#type: "message".into(),
            content: vec![AnthropicResponseContent::Text(
                AnthropicTextResponseContent {
                    r#type: "text".into(),
                    text: "Done!".into(),
                },
            )],
        };

        let call_count = Arc::new(Mutex::new(0_usize));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };

        let tool_call_count = Arc::new(Mutex::new(0_usize));
        let tool_call_count_cb = Arc::clone(&tool_call_count);
        let text_segments: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let text_segments_cb = Arc::clone(&text_segments);

        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Test".into(),
            tools: Some(&[Box::new(tool)]),
            on_tool_call: Some(Arc::new(move |_| {
                *tool_call_count_cb.lock().unwrap() += 1;
            })),
            on_text: Some(Arc::new(move |s| {
                text_segments_cb.lock().unwrap().push(s.to_string());
            })),
        };

        let mock_client = AnthropicClient {
            client: SequentialMockHttpClient::new([tool_call_response, text_response]),
            config: None,
        };
        let res = mock_client.send_message(&request).await;
        test_ok!(res);

        test_eq!(*tool_call_count.lock().unwrap(), 1_usize);
        let texts = text_segments.lock().unwrap();
        test_eq!(texts.len(), 1);
        test_eq!(texts[0].as_str(), "Done!");
    }
    #[test]

    fn test_tool_result_object_content_is_serialized_as_string() {
        // A JSON-object result must be serialized as a JSON string so the Anthropic API
        // receives `"content": "{...}"` rather than `"content": {...}`.
        let content = AnthropicResponseContent::ToolResult(
            ToolCallResponse {
                id: "tool-1".into(),
                tool_name: "mock_tool".into(),
                result: serde_json::json!({"answer": 42}),
            }
            .into(),
        );
        let json = serde_json::to_value(&content).unwrap();
        assert!(
            json["content"].is_string(),
            "content should be a JSON string, got: {}",
            json["content"]
        );
    }

    #[test]
    fn test_tool_result_string_content_passes_through() {
        let content = AnthropicResponseContent::ToolResult(
            ToolCallResponse {
                id: "tool-2".into(),
                tool_name: "mock_tool".into(),
                result: serde_json::json!("plain text"),
            }
            .into(),
        );
        let json = serde_json::to_value(&content).unwrap();
        test_eq!(json["content"].as_str().unwrap(), "plain text");
    }

    #[tokio::test]
    async fn test_followup_queries_work() {
        dotenv().ok();

        let client = AnthropicClient::<ReqwestClient>::default();
        let first_message = ChatRequest {
            message: "What is self-attention?".into(),
            ..ChatRequest::default()
        };

        let response = client.send_message(&first_message).await;
        test_ok!(response);

        let response = response.unwrap();
        let chat_history_contents: ChatHistoryItem = response.content.into();

        let second_message = ChatRequest {
            chat_history: vec![
                ChatHistoryItem {
                    role: USER_ROLE.into(),
                    content: vec![ChatHistoryContent::Text(first_message.message.clone())],
                },
                chat_history_contents,
            ],
            message: "What are the Q, K, and V matrices?".into(),
            ..ChatRequest::default()
        };

        let response = client.send_message(&second_message).await;
        test_ok!(response);
    }
}
