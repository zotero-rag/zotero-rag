use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use arrow_array;
use http::HeaderMap;
use lancedb::arrow::arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::base::{ApiClient, ChatHistoryItem, ChatRequest, CompletionApiResponse};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};
use crate::common::request_with_backoff;
use crate::constants::{
    DEFAULT_ANTHROPIC_MAX_TOKENS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_MAX_RETRIES,
    OPENAI_EMBEDDING_DIM,
};
use crate::embedding::openai::compute_openai_embeddings_sync;
use crate::llm::base::{ChatHistoryContent, ContentType, ToolUseStats};
use crate::llm::tools::{SerializedTool, Tool};
const DEFAULT_CLAUDE_MODEL: &str = DEFAULT_ANTHROPIC_MODEL;

/// A generic client class for now. We can add stuff here later if needed, for
/// example, features like Anthropic's native RAG thing
#[derive(Debug, Clone)]
pub struct AnthropicClient<T: HttpClient = ReqwestClient> {
    pub client: T,
    pub config: Option<crate::config::AnthropicConfig>,
}

impl<T: HttpClient + Default> Default for AnthropicClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AnthropicClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new AnthropicClient instance without configuration
    /// (will fall back to environment variables)
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new AnthropicClient instance with provided configuration
    pub fn with_config(config: crate::config::AnthropicConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }

    /// Internal implementation for computing embeddings using shared logic
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        compute_openai_embeddings_sync(source)
    }
}

impl From<ChatHistoryContent> for AnthropicResponseContent {
    fn from(value: ChatHistoryContent) -> Self {
        match value {
            ChatHistoryContent::Text(s) => Self::PlainText(s),
            ChatHistoryContent::ToolCallRequest(req) => {
                Self::ToolCall(AnthropicToolUseResponseContent {
                    id: req.id,
                    r#type: "tool_use".into(),
                    name: req.tool_name,
                    input: serde_json::Value::as_object(&req.args).unwrap().clone(),
                })
            }
            ChatHistoryContent::ToolCallResponse(res) => Self::ToolResult(AnthropicToolUseResult {
                r#type: "tool_result".into(),
                tool_use_id: res.id,
                content: res.result,
            }),
        }
    }
}

impl From<&ChatHistoryContent> for AnthropicResponseContent {
    fn from(value: &ChatHistoryContent) -> Self {
        match value {
            ChatHistoryContent::Text(s) => Self::PlainText(s.clone()),
            ChatHistoryContent::ToolCallRequest(req) => {
                Self::ToolCall(AnthropicToolUseResponseContent {
                    id: req.id.clone(),
                    r#type: "tool_use".into(),
                    name: req.tool_name.clone(),
                    input: serde_json::Value::as_object(&req.args).unwrap().clone(),
                })
            }
            ChatHistoryContent::ToolCallResponse(res) => Self::ToolResult(AnthropicToolUseResult {
                r#type: "tool_result".into(),
                tool_use_id: res.id.clone(),
                content: res.result.clone(),
            }),
        }
    }
}

/// An Anthropic-specific chat history object. This is pretty much the same as `ChatHistoryItem`,
/// except that `content` is now Anthropic-specific.
#[derive(Clone, Serialize)]
struct AnthropicChatHistoryItem {
    /// Either "user" or "assistant".
    pub role: String,
    /// The contents of this item.
    pub content: Vec<AnthropicResponseContent>,
}

impl From<ChatHistoryItem> for AnthropicChatHistoryItem {
    fn from(value: ChatHistoryItem) -> Self {
        Self {
            role: value.role,
            content: value
                .content
                .into_iter()
                .map(|f| f.into())
                .collect::<Vec<_>>(),
        }
    }
}

impl From<&ChatHistoryItem> for AnthropicChatHistoryItem {
    fn from(value: &ChatHistoryItem) -> Self {
        Self {
            role: value.role.clone(),
            content: value
                .content
                .clone()
                .into_iter()
                .map(|f| f.into())
                .collect::<Vec<_>>(),
        }
    }
}

/// Represents a request to the Anthropic API
#[derive(Serialize)]
struct AnthropicRequest<'a> {
    /// The model to use for the request (e.g., "claude-3-5-sonnet-20241022")
    model: String,
    /// The maximum number of tokens that can be generated in the response
    max_tokens: u32,
    /// The conversation history and current message
    messages: Vec<AnthropicChatHistoryItem>,
    /// The tools passed in
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<SerializedTool<'a>>>,
}

fn get_owned_tools<'a>(tools: Option<&'a [Box<dyn Tool>]>) -> Option<Vec<SerializedTool<'a>>> {
    let owned_tools: Option<Vec<SerializedTool>> = tools.as_ref().map(|iter| {
        iter.iter()
            .map(|f| SerializedTool(&**f))
            .collect::<Vec<SerializedTool>>()
    });

    owned_tools
}

impl<'a> AnthropicRequest<'a> {
    fn from_chat_request(req: &'a mut ChatRequest<'a>, model: String, max_tokens: u32) -> Self {
        let mut messages = req
            .message
            .chat_history
            .iter()
            .map(|f| f.into())
            .collect::<Vec<_>>();

        messages.push(AnthropicChatHistoryItem {
            role: "user".to_owned(),
            content: vec![req.message.message.clone().into()],
        });

        let owned_tools: Option<Vec<SerializedTool>> = get_owned_tools(req.tools);
        AnthropicRequest {
            model,
            max_tokens: req.message.max_tokens.unwrap_or(max_tokens),
            messages,
            tools: owned_tools,
        }
    }
}

/// Token usage statistics returned by the Anthropic API
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AnthropicUsageStats {
    /// Number of tokens in the input prompt
    input_tokens: u32,
    /// Number of tokens in the generated response
    output_tokens: u32,
}

/// The result of a tool call. This is the Anthropic-specific result format.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AnthropicToolUseResult {
    /// The content type. This should always be "tool_result".
    r#type: String,
    /// The ID of the tool call request that this is a result for.
    tool_use_id: String,
    /// The result from the tool call.
    content: serde_json::Value,
}

/// A part of an Anthropic API response denoting some text from the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AnthropicTextResponseContent {
    /// The content type. Always "text".
    r#type: String,
    /// The text content from the model's response
    text: String,
}

impl From<String> for AnthropicResponseContent {
    fn from(value: String) -> Self {
        Self::Text(AnthropicTextResponseContent {
            r#type: "text".into(),
            text: value,
        })
    }
}

/// A part of an Anthropic API response denoting a tool call (Anthropic uses the term "tool use" or
/// "function call"). This is *not* the struct containing the *result* of that tool: that is
/// `AnthropicToolUseResult`. Note that this struct is also used as part of
/// `AnthropicChatHistoryItem`, as one of the possible content types.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AnthropicToolUseResponseContent {
    /// Tool use ID
    id: String,
    /// The content type. Always "tool_use".
    r#type: String,
    /// The tool being called
    name: String,
    /// The input to the tool. Note that all we can guarantee is that the keys are strings
    input: serde_json::Map<String, serde_json::Value>,
}

/// Content block in an Anthropic API response. This is also used in the request as the chat
/// history.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicResponseContent {
    /// A plain-text content, which is used in requests and the user parts of chat history.
    PlainText(String),
    /// A text part of a response
    Text(AnthropicTextResponseContent),
    ToolCall(AnthropicToolUseResponseContent),
    /// Only used in the chat history; this can never be in the response from the API.
    ToolResult(AnthropicToolUseResult),
}

/// Response from the Anthropic API
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AnthropicResponse {
    /// Unique identifier for the response
    id: String,
    /// The model that generated the response
    model: String,
    /// The role of the message (usually "assistant")
    role: String,
    /// Why the model stopped generating (e.g., "end_turn")
    stop_reason: String,
    /// The stop sequence that caused generation to end, if any
    stop_sequence: Option<String>,
    /// Token usage statistics
    usage: AnthropicUsageStats,
    /// The type of the response (usually "message")
    r#type: String,
    /// The content blocks in the response
    content: Vec<AnthropicResponseContent>,
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
async fn send_anthropic_request<'a>(
    client: &impl HttpClient,
    headers: &HeaderMap,
    req: &AnthropicRequest<'a>,
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

/// Process tool calls in a single response from the Anthropic API.
///
/// This function processes tool calls in a single response from the Anthropic API, modifying the
/// `chat_history` with results as it proceeds. This function also accepts a mutable reference to a
/// vector of `ContentType`, which gets filled with text responses from the model and processed tool
/// calls and results.
///
/// # Arguments:
///
/// * `chat_history`: A list of Anthropic-specific chat history items, to be passed back to the API
///   after the tool calls are processed.
/// * `new_contents`: A list that should be populated with the model responses and tool uses.
/// * `response`: The API response to process.
/// * `tools`: A reference to a list of owned tools.
async fn process_tool_calls<'a>(
    chat_history: &mut Vec<AnthropicChatHistoryItem>,
    new_contents: &mut Vec<ContentType>,
    response: &AnthropicResponse,
    tools: &Vec<SerializedTool<'a>>,
) -> Result<(), LLMError> {
    for content in &response.content {
        match content {
            AnthropicResponseContent::PlainText(s) => {
                new_contents.push(ContentType::Text(s.clone()))
            }
            AnthropicResponseContent::Text(text_content) => {
                new_contents.push(ContentType::Text(text_content.text.clone()));
            }
            AnthropicResponseContent::ToolResult(tool_result) => {
                // This is invalid, but technically we can recover by just ignoring it--so
                // we will.
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be ignored. Tool result: {:#?}",
                    tool_result
                );
            }
            AnthropicResponseContent::ToolCall(tool_call) => {
                let tool_call_id = tool_call.id.clone();
                let called_tool = tools.iter().find(|tool| tool.0.name() == tool_call.name)
                    .ok_or_else(|| {
                        LLMError::ToolCallError(format!(
                            "Tool {} was called, but it does not exist in the passed list of tools.",
                            tool_call.name
                        ))
                    }
                )?;

                let tool_result = match called_tool.0.call(tool_call.input.clone().into()).await {
                    Ok(res) => res,
                    Err(e) => Value::String(format!("Error calling tool: {e}")),
                };

                new_contents.push(ContentType::ToolCall(ToolUseStats {
                    tool_name: tool_call.name.clone(),
                    tool_args: serde_json::Value::from(tool_call.input.clone()),
                    tool_result: tool_result.clone(),
                }));

                chat_history.push(AnthropicChatHistoryItem {
                    role: "user".into(),
                    content: vec![AnthropicResponseContent::ToolResult(
                        AnthropicToolUseResult {
                            r#type: "tool_result".into(),
                            tool_use_id: tool_call_id,
                            content: tool_result,
                        },
                    )],
                });
            }
        }
    }

    Ok(())
}

impl<T: HttpClient> ApiClient for AnthropicClient<T> {
    /// Send a request to the Anthropic API, processing tool calls as necessary. Returns a final
    /// response after all tool calls are processed and sent back to the API.
    async fn send_message<'a>(
        &self,
        request: &'a mut ChatRequest<'a>,
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

        let req_body = AnthropicRequest::from_chat_request(request, model, max_tokens);
        let response = send_anthropic_request(&self.client, &headers, &req_body).await?;

        let mut has_tool_calls: bool = response
            .content
            .iter()
            .any(|c| matches!(c, AnthropicResponseContent::ToolCall { .. }));

        let mut chat_history: Vec<AnthropicChatHistoryItem> = req_body.messages.clone();

        // Append the contents
        chat_history.push(AnthropicChatHistoryItem {
            role: "assistant".into(),
            content: response.content.clone(),
        });

        let mut contents: Vec<ContentType> = Vec::new();

        while has_tool_calls {
            process_tool_calls(
                &mut chat_history,
                &mut contents,
                &response,
                req_body.tools.as_ref().unwrap(),
            )
            .await?;

            let response = send_anthropic_request(&self.client, &headers, &req_body).await?;
            has_tool_calls = response
                .content
                .iter()
                .any(|c| matches!(c, AnthropicResponseContent::ToolCall { .. }));
        }

        Ok(CompletionApiResponse {
            content: contents,
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

/// Implements the LanceDB EmbeddingFunction trait for AnthropicClient. Note that Anthropic
/// does not have their own embeddings model, so we'll just use OpenAI's model instead. This
/// does mean users will need an API key from both--but there's really no other option here.
/// Anthropic's docs recommend Voyage AI--but users are more likely to have an OpenAI key than
/// a Voyage AI key.
///
/// Maintainers should note that any updates here should also be reflected in OpenAIClient.
impl<T: HttpClient + Default + std::fmt::Debug> EmbeddingFunction for AnthropicClient<T> {
    fn name(&self) -> &str {
        "Anthropic"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            OPENAI_EMBEDDING_DIM as i32, // text-embedding-3-small size
        )))
    }

    fn compute_source_embeddings(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        // Call our internal implementation and map LLMError to lancedb::Error
        match self.compute_embeddings_internal(source) {
            Ok(result) => Ok(result),
            Err(e) => Err(lancedb::Error::Other {
                message: e.to_string(),
                source: None,
            }),
        }
    }

    fn compute_query_embeddings(
        &self,
        input: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        // For queries, we don't need concurrency since it's typically a single query
        // Just reuse the same implementation with the expectation it's usually for one item
        match self.compute_embeddings_internal(input) {
            Ok(result) => Ok(result),
            Err(e) => Err(lancedb::Error::Other {
                message: e.to_string(),
                source: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::{Arc, Mutex};

    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;
    use schemars::{JsonSchema, schema_for};
    use serde::Deserialize;

    use crate::constants::OPENAI_EMBEDDING_DIM;
    use crate::llm::anthropic::{AnthropicTextResponseContent, DEFAULT_CLAUDE_MODEL};
    use crate::llm::base::{ApiClient, ChatRequest, ContentType, UserMessage};
    use crate::llm::http_client::{MockHttpClient, ReqwestClient};
    use crate::llm::tools::Tool;

    use super::{
        AnthropicClient, AnthropicResponse, AnthropicResponseContent, AnthropicUsageStats,
    };

    /// A mock tool that returns static content. We will test that tool calling works and that we
    /// can deserialize the responses using this.
    struct MockTool {
        call_count: Arc<Mutex<usize>>,
    }

    #[derive(Deserialize, JsonSchema)]
    struct MockToolInput {
        name: String,
    }
    impl Tool for MockTool {
        fn name(&self) -> String {
            "mock_tool".into()
        }

        fn description(&self) -> String {
            "A mock tool that you should call for testing, and pass in a name.".into()
        }

        fn parameters(&self) -> schemars::Schema {
            schema_for!(MockToolInput)
        }

        fn schema_key(&self) -> String {
            "input_schema".into()
        }

        fn call(
            &self,
            args: serde_json::Value,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send>>
        {
            *self.call_count.lock().unwrap() += 1;

            Box::pin(async move {
                let input: MockToolInput =
                    serde_json::from_value(args).map_err(|e| format!("Error: {e}"))?;
                let greeting = format!("Hello, {}!", input.name);

                serde_json::to_value(greeting).map_err(|e| format!("Error: {e}"))
            })
        }
    }

    //
    // End tool setup
    //

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = AnthropicClient::<ReqwestClient>::default();
        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
        };

        let mut request = ChatRequest::from(&message);
        let res = client.send_message(&mut request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Anthropic test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
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

        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: None,
            message: "Hello!".to_owned(),
        };

        let mut request = ChatRequest::from(&message);
        let res = mock_client.send_message(&mut request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Anthropic test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());

        let res = res.unwrap();
        assert_eq!(res.input_tokens, 9);
        assert_eq!(res.output_tokens, 13);

        let content = res.content[0];
        matches!(
            content,
            ContentType::Text("Hi there! How can I help you today?".into())
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings() {
        dotenv().ok();

        let array = arrow_array::StringArray::from(vec![
            "Hello, World!",
            "A second string",
            "A third string",
            "A fourth string",
            "A fifth string",
            "A sixth string",
        ]);

        let client = AnthropicClient::<ReqwestClient>::default();
        let embeddings = client.compute_source_embeddings(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Anthropic embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), OPENAI_EMBEDDING_DIM as i32);
    }

    #[tokio::test]
    async fn test_request_with_tool_works() {
        dotenv().ok();

        let client = AnthropicClient::<ReqwestClient>::default();
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };
        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "This is a test. Call the `mock_tool`, passing in a `name`, and ensure it returns a greeting".into()
        };

        let mut request = ChatRequest {
            message: &message,
            tools: Some(&mut [Box::new(tool)]),
        };
        let res = client.send_message(&mut request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Anthropic test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
        dbg!(&res);
        assert!(call_count.lock().unwrap().eq(&(1 as usize)));
    }
}
