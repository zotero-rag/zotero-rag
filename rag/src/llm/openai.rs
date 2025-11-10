//! OpenAI client implementation.
//!
//! This module implements an OpenAI client using the Responses API, including
//! support for tool calls and token usage reporting. It also implements the
//! LanceDB `EmbeddingFunction` by delegating to the shared OpenAI embeddings
//! implementation used across providers in this project.

use std::borrow::Cow;
use std::env;
use std::fmt::Debug;
use std::sync::Arc;

use http::HeaderMap;
use lancedb::arrow::arrow_schema::DataType;
use lancedb::embeddings::EmbeddingFunction;
use serde::{Deserialize, Serialize};

use super::base::{ApiClient, ChatHistoryItem, ChatRequest, CompletionApiResponse};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};
use crate::common::request_with_backoff;
use crate::constants::{DEFAULT_MAX_RETRIES, DEFAULT_OPENAI_MODEL, OPENAI_EMBEDDING_DIM};
use crate::embedding::openai::compute_openai_embeddings_sync;
use crate::llm::base::{ChatHistoryContent, ContentType, ToolUseStats};
use crate::llm::tools::{SerializedTool, get_owned_tools};
use serde_json::{Map, Value};

/// A client for OpenAI's chat completions (Responses) API.
#[derive(Debug, Clone)]
pub struct OpenAIClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub client: T,
    /// Optional configuration for the OpenAI client.
    pub config: Option<crate::config::OpenAIConfig>,
}

impl<T: HttpClient + Default> Default for OpenAIClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OpenAIClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new `OpenAIClient` instance without configuration.
    ///
    /// Uses environment variables for configuration.
    ///
    /// # Returns
    ///
    /// A client initialized with a default HTTP client and no config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new `OpenAIClient` instance with provided configuration.
    ///
    /// # Arguments:
    ///
    /// * `config` - Client configuration including API key and model.
    ///
    /// # Returns
    ///
    /// A client initialized with a default HTTP client and the given config.
    #[must_use]
    pub fn with_config(config: crate::config::OpenAIConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }

    /// Internal implementation for computing embeddings using shared logic.
    ///
    /// # Arguments:
    ///
    /// * `source` - An Arrow array of strings to embed.
    ///
    /// # Returns
    ///
    /// An Arrow array of `FixedSizeList<Float32>` containing the embeddings.
    ///
    /// # Errors
    ///
    /// * `LLMError::EnvError` - If the OPENAI_API_KEY environment variable is not set
    /// * `LLMError::TimeoutError` - If the HTTP request times out
    /// * `LLMError::CredentialError` - If the API returns 401 or 403 status
    /// * `LLMError::HttpStatusError` - If the API returns other unsuccessful HTTP status codes
    /// * `LLMError::NetworkError` - If a network connectivity error occurs
    /// * `LLMError::DeserializationError` - If the API response cannot be parsed
    /// * `LLMError::GenericLLMError` - If other HTTP errors occur or Arrow array creation fails
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        compute_openai_embeddings_sync(source)
    }
}

/// OpenAI-specific tool wrapper that adds the `type` and `strict` fields
/// required by OpenAI's API.
#[derive(Clone)]
pub struct OpenAITool<'a>(pub SerializedTool<'a>);

impl Serialize for OpenAITool<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        // Construct a map with OpenAI-specific fields
        let mut obj = Map::new();
        obj.insert("name".into(), Value::String(self.0.0.name()));
        obj.insert("description".into(), Value::String(self.0.0.description()));
        obj.insert("type".into(), Value::String("function".into()));
        obj.insert("strict".into(), Value::Bool(false));

        // Add the schema with the provider-specific key
        obj.insert(self.0.0.schema_key(), self.0.0.parameters().into());

        // Serialize the map
        let mut map = serializer.serialize_map(Some(obj.len()))?;
        for (k, v) in &obj {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

/// Convert a Vec of SerializedTools into OpenAI-specific wrappers.
fn wrap_tools_for_openai(tools: Vec<SerializedTool<'_>>) -> Vec<OpenAITool<'_>> {
    tools.into_iter().map(OpenAITool).collect()
}

/// Extract the inner SerializedTools from OpenAI-specific wrappers.
fn unwrap_openai_tools<'a>(tools: &[OpenAITool<'a>]) -> Vec<SerializedTool<'a>> {
    tools.iter().map(|t| t.0.clone()).collect()
}

/// A tool call input item for the Responses API.
///
/// We add tool call requests as items in the flattened input list when
/// constructing a request body for the Responses API.
#[derive(Clone, Debug, Serialize)]
struct OpenAIRequestToolCallInputItem {
    /// The unique tool call ID.
    call_id: String,
    /// Always "function_call".
    r#type: String,
    /// The name of the function to call.
    name: String,
    /// The arguments to the tool call.
    arguments: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
struct OpenAIRequestToolResultInputItem {
    /// Always "function_call_output".
    r#type: String,
    /// The tool call ID this is a result for.
    call_id: String,
    /// The tool output value.
    output: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
enum OpenAIRequestInputItem {
    /// A plain-text item.
    Text(String),
}

#[derive(Clone, Serialize)]
struct OpenAIChatHistoryItem {
    /// Either "user" or "assistant".
    pub role: String,
    /// The content item for this role.
    pub content: OpenAIRequestInputItem,
    /// Always "message".
    pub r#type: String,
}

#[derive(Clone, Serialize)]
#[serde(untagged)]
enum OpenAIRequestInput {
    Message(OpenAIChatHistoryItem),
    FunctionCall(OpenAIRequestToolCallInputItem),
    ToolResult(OpenAIRequestToolResultInputItem),
}

impl From<&ChatHistoryItem> for Vec<OpenAIRequestInput> {
    fn from(value: &ChatHistoryItem) -> Vec<OpenAIRequestInput> {
        value
            .content
            .iter()
            .map(|c| match c {
                ChatHistoryContent::Text(s) => OpenAIRequestInput::Message(OpenAIChatHistoryItem {
                    role: value.role.clone(),
                    r#type: "message".into(),
                    content: OpenAIRequestInputItem::Text(s.clone()),
                }),
                ChatHistoryContent::ToolCallRequest(req) => {
                    OpenAIRequestInput::FunctionCall(OpenAIRequestToolCallInputItem {
                        call_id: req.id.clone(),
                        r#type: "function_call".into(),
                        name: req.tool_name.clone(),
                        arguments: req.args.clone(),
                    })
                }
                ChatHistoryContent::ToolCallResponse(res) => {
                    OpenAIRequestInput::ToolResult(OpenAIRequestToolResultInputItem {
                        call_id: res.id.clone(),
                        r#type: "function_call_output".into(),
                        output: res.result.clone(),
                    })
                }
            })
            .collect::<Vec<_>>()
    }
}

#[derive(Serialize)]
struct OpenAIRequest<'a> {
    /// The model to use for the request (e.g., "gpt-4.1").
    model: &'a str,

    /// The flattened chat history and current message.
    input: &'a [OpenAIRequestInput],

    #[serde(skip_serializing_if = "Option::is_none")]
    /// Maximum tokens to generate in the response.
    max_output_tokens: Option<u32>,

    /// The tools passed in
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [OpenAITool<'a>]>,
}

/// Helper to build messages and tools from a ChatRequest.
/// Returns owned data that can then be borrowed by OpenAIRequest.
fn build_openai_messages_and_tools<'a>(
    req: &'a ChatRequest<'a>,
) -> (Vec<OpenAIRequestInput>, Option<Vec<OpenAITool<'a>>>) {
    let mut messages = req
        .chat_history
        .iter()
        .flat_map(<&ChatHistoryItem as Into<Vec<OpenAIRequestInput>>>::into)
        .collect::<Vec<_>>();

    messages.push(OpenAIRequestInput::Message(OpenAIChatHistoryItem {
        role: "user".to_owned(),
        r#type: "message".into(),
        content: OpenAIRequestInputItem::Text(req.message.clone()),
    }));

    let owned_tools: Option<Vec<OpenAITool<'a>>> =
        get_owned_tools(req.tools).map(wrap_tools_for_openai);

    (messages, owned_tools)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[allow(clippy::struct_field_names)]
struct OpenAIUsage {
    /// Number of tokens in the input prompt.
    input_tokens: u32,
    /// Number of tokens in the generated response.
    output_tokens: u32,
    /// Total token usage.
    total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIContent {
    /// The content type key.
    r#type: String,
    #[serde(default)]
    /// Message text, when present.
    text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
enum OpenAIOutput {
    /// A reasoning block with an optional summary.
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<String>,
    },
    /// A standard assistant message with text content.
    #[serde(rename = "message")]
    Message {
        id: String,
        status: String,
        content: Vec<OpenAIContent>,
        role: String,
    },
    /// A function call request to invoke a tool.
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIResponse {
    /// Unique identifier for the response.
    id: String,
    /// Timestamp the response was created.
    created_at: u64,
    /// The model that generated the response.
    model: String,
    /// Token usage details.
    usage: OpenAIUsage,
    /// The content parts produced by the model.
    output: Vec<OpenAIOutput>,
}

/// Process tool call outputs in an OpenAI response, updating chat history and
/// collecting provider-agnostic content summaries.
///
/// Tool results appearing directly in the model response are ignored with a warning.
///
/// Because OpenAI's responses specifically are in a format where we cannot directly `impl
/// From<ChatHistoryItem>`, we need a specialized version for this.
///
/// # Arguments:
///
/// * `chat_history` - The running chat history that will be appended to.
/// * `new_contents` - The collection of normalized content to return to callers.
/// * `contents` - The newly returned OpenAI chat items to process.
/// * `tools` - The tools available for invocation.
///
/// # Returns
///
/// `Ok(())` if tool calls are processed successfully, otherwise an `LLMError`.
async fn process_openai_tool_calls(
    chat_history: &mut Vec<OpenAIRequestInput>,
    new_contents: &mut Vec<ContentType>,
    contents: &[OpenAIRequestInput],
    tools: &[SerializedTool<'_>],
) -> Result<(), LLMError> {
    for content in contents {
        match content {
            OpenAIRequestInput::ToolResult(res_item) => {
                // This is invalid, but technically we can recover by just ignoring it--so
                // we will.
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be ignored. Tool result: {:#?}",
                    res_item.output
                );
            }
            OpenAIRequestInput::FunctionCall(tool_call) => {
                let tool_call_id = tool_call.call_id.clone();
                let called_tool = tools.iter().find(|tool| tool.0.name() == tool_call.name)
                .ok_or_else(|| {
                    LLMError::ToolCallError(format!(
                        "Tool {} was called, but it does not exist in the passed list of tools.",
                        tool_call.name
                    ))
                })?;

                // OpenAI returns arguments as a JSON string, so we need to parse it
                let parsed_arguments = match &tool_call.arguments {
                    serde_json::Value::String(s) => {
                        serde_json::from_str(s).unwrap_or_else(|_| tool_call.arguments.clone())
                    }
                    other => other.clone(),
                };

                let tool_result = match called_tool.0.call(parsed_arguments.clone()).await {
                    Ok(res) => res,
                    Err(e) => serde_json::Value::String(format!("Error calling tool: {e}")),
                };

                new_contents.push(ContentType::ToolCall(ToolUseStats {
                    tool_call_id: tool_call.call_id.clone(),
                    tool_name: tool_call.name.clone(),
                    tool_args: parsed_arguments.clone(),
                    tool_result: tool_result.clone(),
                }));

                let new_items: OpenAIRequestInput =
                    OpenAIRequestInput::ToolResult(OpenAIRequestToolResultInputItem {
                        r#type: "function_call_output".into(),
                        call_id: tool_call_id,
                        output: tool_result,
                    });
                chat_history.push(new_items);
            }
            OpenAIRequestInput::Message(chi) => match &chi.content {
                OpenAIRequestInputItem::Text(s) => new_contents.push(ContentType::Text(s.clone())),
            },
        }
    }

    Ok(())
}

/// Send a generation request to OpenAI's Responses API with retry/backoff.
///
/// # Arguments:
///
/// * `client` - The HTTP client implementation to use.
/// * `headers` - The HTTP headers to include (auth, content-type).
/// * `req` - The OpenAI-specific request body.
///
/// # Returns
///
/// The deserialized OpenAI response, or an `LLMError` on failure.
async fn send_openai_generation_request(
    client: &impl HttpClient,
    headers: &HeaderMap,
    req: &OpenAIRequest<'_>,
) -> Result<OpenAIResponse, LLMError> {
    const MAX_RETRIES: usize = DEFAULT_MAX_RETRIES;

    let res = request_with_backoff(
        client,
        "https://api.openai.com/v1/responses",
        headers,
        &req,
        MAX_RETRIES,
    )
    .await?;

    let body = res.text().await?;

    let json: serde_json::Value = match serde_json::from_str(&body) {
        Ok(json) => json,
        Err(_) => return Err(super::errors::LLMError::DeserializationError(body)),
    };

    let response: OpenAIResponse = match serde_json::from_value(json) {
        Ok(response) => response,
        Err(_) => return Err(super::errors::LLMError::DeserializationError(body)),
    };

    Ok(response)
}

/// Convert the OpenAI-specific response into OpenAI chat history items.
///
/// This is used to extend the chat history with assistant responses and
/// function call requests before running tool execution.
fn map_response_to_chat_history(response: &OpenAIResponse) -> Vec<OpenAIRequestInput> {
    response
        .output
        .iter()
        .filter_map(|c| match c {
            OpenAIOutput::Reasoning { .. } => None,
            OpenAIOutput::Message { content, .. } => {
                Some(OpenAIRequestInput::Message(OpenAIChatHistoryItem {
                    role: "assistant".into(),
                    r#type: "message".into(),
                    content: OpenAIRequestInputItem::Text(
                        content
                            .first()
                            .unwrap()
                            .text
                            .clone()
                            .unwrap_or_else(String::new),
                    ),
                }))
            }
            OpenAIOutput::FunctionCall {
                call_id,
                name,
                arguments,
            } => Some(OpenAIRequestInput::FunctionCall(
                OpenAIRequestToolCallInputItem {
                    call_id: call_id.clone(),
                    r#type: "function_call".into(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                },
            )),
        })
        .collect::<Vec<_>>()
}

impl<T: HttpClient> ApiClient for OpenAIClient<T> {
    /// Send a request to the OpenAI Responses API, processing tool calls as necessary.
    /// Returns a final response after all tool calls are processed and sent back.
    async fn send_message<'a>(
        &self,
        request: &'a ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, super::errors::LLMError> {
        // Use config if available, otherwise fall back to env vars
        let (api_key, model, _) = if let Some(ref config) = self.config {
            (
                config.api_key.clone(),
                config.model.clone(),
                Some(config.max_tokens),
            )
        } else {
            (
                env::var("OPENAI_API_KEY")?,
                env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string()),
                env::var("OPENAI_MAX_TOKENS")
                    .ok()
                    .and_then(|s| s.parse().ok()),
            )
        };

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
        headers.insert("content-type", "application/json".parse()?);

        // Build owned versions of the initial messages and tools
        let (mut chat_history, tools) = build_openai_messages_and_tools(request);
        let max_output_tokens = request.max_tokens;

        // Create the initial request
        let req_body = OpenAIRequest {
            model: &model,
            input: &chat_history,
            max_output_tokens,
            tools: tools.as_deref(),
        };

        let mut response =
            send_openai_generation_request(&self.client, &headers, &req_body).await?;

        let mut has_tool_calls: bool = response
            .output
            .iter()
            .any(|c| matches!(c, OpenAIOutput::FunctionCall { .. }));

        let mut response_messages = map_response_to_chat_history(&response);
        chat_history.append(&mut response_messages);
        let mut contents: Vec<ContentType> = Vec::new();

        while has_tool_calls {
            let converted_contents = map_response_to_chat_history(&response);
            let base_tools = tools.as_ref().map(|t| unwrap_openai_tools(t)).unwrap();
            process_openai_tool_calls(
                &mut chat_history,
                &mut contents,
                &converted_contents,
                &base_tools,
            )
            .await?;

            // Create a new request borrowing the updated chat history
            let updated_req_body = OpenAIRequest {
                model: &model,
                input: &chat_history,
                max_output_tokens,
                tools: tools.as_deref(),
            };

            response =
                send_openai_generation_request(&self.client, &headers, &updated_req_body).await?;
            let mut response_messages = map_response_to_chat_history(&response);

            // Append the new response to chat history
            chat_history.append(&mut response_messages);
            has_tool_calls = response
                .output
                .iter()
                .any(|c| matches!(c, OpenAIOutput::FunctionCall { .. }));
        }

        // Process the final response (which has no tool calls) to extract text content
        for content in &response.output {
            match content {
                OpenAIOutput::Message { content, .. } => {
                    if let Some(ct) = content.first() {
                        contents.push(ContentType::Text(ct.text.clone().unwrap()));
                    }
                }
                OpenAIOutput::Reasoning { summary, .. } => {
                    contents.push(ContentType::Text(format!(
                        "<reasoning>{}</reasoning>",
                        summary.join("\n")
                    )));
                }
                OpenAIOutput::FunctionCall { .. } => {}
            }
        }

        Ok(CompletionApiResponse {
            content: contents,
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

/// Implements the LanceDB EmbeddingFunction trait for OpenAI client. This is the same code
/// as the one in AnthropicClient verbatim--I made a judgement call that two copies are okay;
/// when we hit a place where we need a third copy, we'll refactor.
///
/// Maintainers should note that any updates here should also be reflected in AnthropicClient.
impl<T: HttpClient + Default + Debug> EmbeddingFunction for OpenAIClient<T> {
    fn name(&self) -> &'static str {
        "OpenAI"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(lancedb::arrow::arrow_schema::Field::new(
                "item",
                DataType::Float32,
                true,
            )),
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
    use super::{OpenAIClient, OpenAIContent, OpenAIOutput, OpenAIResponse, OpenAIUsage};
    use crate::constants::OPENAI_EMBEDDING_DIM;
    use crate::llm::base::{ApiClient, ChatRequest, ContentType};
    use crate::llm::http_client::{MockHttpClient, ReqwestClient};
    use crate::llm::tools::OPENAI_SCHEMA_KEY;
    use crate::llm::tools::test_utils::MockTool;
    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = OpenAIClient::<ReqwestClient>::default();
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
            tools: None,
        };
        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("OpenAI test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_request_with_mock() {
        // Load environment variables from .env file
        dotenv().ok();

        // Create a proper OpenAIResponse that matches the structure we expect to deserialize
        #[allow(clippy::unreadable_literal)]
        let mock_response = OpenAIResponse {
            id: "mock-id".to_string(),
            created_at: 1234567890,
            model: "gpt-4.1-2025-04-14".to_string(),
            usage: OpenAIUsage {
                input_tokens: 5,
                output_tokens: 10,
                total_tokens: 15,
            },
            output: vec![OpenAIOutput::Message {
                id: "msg_id".into(),
                status: "completed".into(),
                role: "assistant".into(),
                content: vec![OpenAIContent {
                    r#type: "output_text".into(),
                    text: Some("Hi there!".into()),
                }],
            }],
        };

        let mock_http_client = MockHttpClient::new(mock_response);
        let mock_client = OpenAIClient {
            client: mock_http_client,
            config: None,
        };

        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
            tools: None,
        };
        let res = mock_client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("OpenAI test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
        let res = res.unwrap();
        assert_eq!(res.content.len(), 1);
        if let ContentType::Text(text) = &res.content[0] {
            assert_eq!(text, "Hi there!");
        } else {
            panic!("Expected Text content type");
        }
        assert_eq!(res.input_tokens, 5);
        assert_eq!(res.output_tokens, 10);
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

        let client = OpenAIClient::<ReqwestClient>::default();
        let embeddings = client.compute_source_embeddings(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("OpenAI embedding error: {:?}", embeddings.as_ref().err());
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

        let client = OpenAIClient::<ReqwestClient>::default();
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
            schema_key: OPENAI_SCHEMA_KEY.into(),
        };
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "This is a test. Call the `mock_tool`, passing in a `name`, and ensure it returns a greeting".into(),
            tools: Some(&[Box::new(tool)]),
        };

        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("OpenAI test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
        assert!(call_count.lock().unwrap().eq(&1_usize));
    }
}
