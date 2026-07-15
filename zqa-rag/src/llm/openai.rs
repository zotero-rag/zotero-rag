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
use serde_json::{Map, Value};

use super::base::{ApiClient, ChatHistoryItem, ChatRequest, CompletionApiResponse};
use super::errors::LLMError;
use crate::clients::openai::OpenAIClient;
use crate::constants::{
    DEFAULT_OPENAI_EMBEDDING_DIM, DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_REASONING_EFFORT,
};
use crate::http_client::HttpClient;
use crate::llm::base::{
    AgenticClient, ChatHistoryContent, MessageRole, ProviderTurn, ReasoningConfig, ToolCallRequest,
    run_agentic_loop, send_generation_request,
};
use crate::llm::tools::{OPENAI_SCHEMA_KEY, SerializedTool};
use crate::pricing::ModelUsage;

/// OpenAI-specific tool wrapper that adds the `type` and `strict` fields
/// required by OpenAI's API.
#[derive(Clone)]
pub(crate) struct OpenAITool<'a>(pub SerializedTool<'a>);

impl Serialize for OpenAITool<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        // Construct a map with OpenAI-specific fields
        let mut obj = Map::new();
        obj.insert("name".into(), Value::String(self.0.tool.name()));
        obj.insert(
            "description".into(),
            Value::String(self.0.tool.description()),
        );
        obj.insert("type".into(), Value::String("function".into()));
        obj.insert("strict".into(), Value::Bool(false));

        // Add the schema with the provider-specific key
        obj.insert(self.0.schema_key.into(), self.0.tool.parameters().into());

        // Serialize the map
        let mut map = serializer.serialize_map(Some(obj.len()))?;
        for (k, v) in &obj {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

/// Convert serialized tools into OpenAI-specific wrappers.
fn wrap_tools_for_openai<'a>(tools: &[SerializedTool<'a>]) -> Vec<OpenAITool<'a>> {
    tools.iter().cloned().map(OpenAITool).collect()
}

/// A tool call input item for the Responses API.
///
/// We add tool call requests as items in the flattened input list when
/// constructing a request body for the Responses API.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpenAIRequestToolCallInputItem {
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
pub(crate) struct OpenAIRequestToolResultInputItem {
    /// Always "function_call_output".
    r#type: String,
    /// The tool call ID this is a result for.
    call_id: String,
    /// The tool output value, JSON-encoded as a string per the Responses API spec.
    output: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum OpenAIRequestInputItem {
    /// A plain-text item.
    Text(String),
}

#[derive(Clone, Serialize)]
pub(crate) struct OpenAIChatHistoryItem {
    /// Either USER_ROLE or "assistant".
    pub role: MessageRole,
    /// The content item for this role.
    pub content: OpenAIRequestInputItem,
    /// Always "message".
    pub r#type: String,
}

#[derive(Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum OpenAIRequestInput {
    Message(OpenAIChatHistoryItem),
    FunctionCall(OpenAIRequestToolCallInputItem),
    ToolResult(OpenAIRequestToolResultInputItem),
    Reasoning(OpenAIRequestReasoningItem),
}

#[derive(Clone, Serialize)]
pub(crate) struct OpenAIRequestReasoningItem {
    /// Always "reasoning".
    r#type: String,
    /// The provider-assigned reasoning item ID.
    id: String,
    /// Optional reasoning summaries returned by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<Vec<OpenAIOutputReasoningSummary>>,
    /// Opaque fields, such as encrypted reasoning content, that must be replayed unchanged.
    #[serde(flatten)]
    additional_fields: Map<String, Value>,
}

impl From<ChatHistoryItem> for Vec<OpenAIRequestInput> {
    fn from(value: ChatHistoryItem) -> Self {
        let role = value.role;

        value
            .content
            .into_iter()
            .map(|content| match content {
                ChatHistoryContent::Text(text) => {
                    OpenAIRequestInput::Message(OpenAIChatHistoryItem {
                        role,
                        r#type: "message".into(),
                        content: OpenAIRequestInputItem::Text(text),
                    })
                }
                ChatHistoryContent::ToolCallRequest(request) => {
                    OpenAIRequestInput::FunctionCall(OpenAIRequestToolCallInputItem {
                        call_id: request.id,
                        r#type: "function_call".into(),
                        name: request.tool_name,
                        arguments: request.args,
                    })
                }
                ChatHistoryContent::ToolCallResponse(response) => {
                    OpenAIRequestInput::ToolResult(OpenAIRequestToolResultInputItem {
                        call_id: response.id,
                        r#type: "function_call_output".into(),
                        output: if let Some(text) = response.result.as_str() {
                            text.to_string()
                        } else {
                            serde_json::to_string(&response.result)
                                .unwrap_or_else(|_| "null".to_string())
                        },
                    })
                }
            })
            .collect()
    }
}

#[derive(Serialize)]
pub(crate) struct OpenAIReasoning {
    /// Thinking effort. Currently supported values are 'none', 'minimal', 'low', 'medium',
    /// 'high', and 'xhigh'. Only supported by gpt-5 and o-series models. gpt-5.1 does not
    /// support 'xhigh'. gpt-5 (the OpenAI docs state "all models before gpt-5.1") does not
    /// support "none". gpt-5-pro only supports 'high'. 'xhigh' is supported for all models
    /// after gpt-5.1-codex-max. See [docs].
    ///
    /// [docs]: https://developers.openai.com/api/reference/resources/responses/methods/create
    effort: String,
    /// One of "auto", "concise", "detailed".
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
}

impl From<&ReasoningConfig> for OpenAIReasoning {
    fn from(value: &ReasoningConfig) -> Self {
        Self {
            effort: value
                .effort
                .clone()
                .unwrap_or(DEFAULT_OPENAI_REASONING_EFFORT.into()),
            summary: value.summary.clone(),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct OpenAIRequest<'a> {
    /// The model to use for the request (e.g., "gpt-5.2").
    model: &'a str,

    /// The flattened chat history and current message.
    input: &'a [OpenAIRequestInput],

    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAIReasoning>,

    #[serde(skip_serializing_if = "Option::is_none")]
    /// Maximum tokens to generate in the response.
    max_output_tokens: Option<u32>,

    /// The tools passed in
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [OpenAITool<'a>]>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct OpenAIInputTokensDetails {
    /// The number of input tokens that were written to the cache.
    #[serde(default)]
    cache_write_tokens: u32,
    /// The number of tokens that were retrieved from the cache.
    cached_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct OpenAIOutputTokensDetails {
    /// The number of reasoning tokens
    reasoning_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[allow(clippy::struct_field_names)]
struct OpenAIUsage {
    /// Number of tokens in the input prompt.
    input_tokens: u32,
    /// Breakdown of the input tokens.
    input_tokens_details: Option<OpenAIInputTokensDetails>,
    /// Number of tokens in the generated response.
    output_tokens: u32,
    /// Breakdown of the output tokens.
    output_tokens_details: Option<OpenAIOutputTokensDetails>,
    /// Total token usage.
    total_tokens: u32,
}

impl From<OpenAIUsage> for ModelUsage {
    fn from(val: OpenAIUsage) -> Self {
        ModelUsage {
            input_tokens: val.input_tokens,
            input_cache_written: val.input_tokens_details.map_or(0, |c| c.cache_write_tokens),
            input_cache_read: val.input_tokens_details.map_or(0, |c| c.cached_tokens),
            output_tokens: val.output_tokens,
            reasoning_tokens: val.output_tokens_details.map_or(0, |c| c.reasoning_tokens),
        }
    }
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
struct OpenAIOutputReasoningSummary {
    /// Always "summary_text"
    r#type: String,
    /// Reasoning summary
    text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
enum OpenAIOutput {
    /// A reasoning block with an optional summary.
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<Vec<OpenAIOutputReasoningSummary>>,
        /// Opaque reasoning fields, such as encrypted reasoning content.
        #[serde(flatten)]
        additional_fields: Map<String, Value>,
    },
    /// A standard assistant message with text content.
    #[serde(rename = "message")]
    Message {
        id: String,
        status: String,
        content: Vec<OpenAIContent>,
        role: MessageRole,
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

/// Normalize an OpenAI function call's arguments for generic tool dispatch.
fn parse_function_arguments(arguments: &Value) -> Value {
    match arguments {
        Value::String(arguments) => {
            serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.clone()))
        }
        arguments => arguments.clone(),
    }
}

/// Convert an OpenAI message's first text content item to provider-agnostic content.
fn map_message_to_chat_content(content: &[OpenAIContent]) -> Option<ChatHistoryContent> {
    content
        .first()
        .map(|content| ChatHistoryContent::Text(content.text.clone().unwrap_or_default()))
}

/// Convert OpenAI output into provider-agnostic content for tool dispatch and presentation.
fn map_response_to_chat_contents(response: &OpenAIResponse) -> Vec<ChatHistoryContent> {
    response
        .output
        .iter()
        .filter_map(|output| match output {
            OpenAIOutput::Reasoning { summary, .. } => {
                let summary = summary
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .map(|summary| summary.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                Some(ChatHistoryContent::Text(format!(
                    "<reasoning>{summary}</reasoning>"
                )))
            }
            OpenAIOutput::Message { content, .. } => map_message_to_chat_content(content),
            OpenAIOutput::FunctionCall {
                call_id,
                name,
                arguments,
            } => Some(ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                id: call_id.clone(),
                tool_name: name.clone(),
                args: parse_function_arguments(arguments),
            })),
        })
        .collect()
}

/// Convert the OpenAI-specific response into native input items for a continuation request.
///
/// Reasoning items are preserved alongside function calls and tool outputs so reasoning models can
/// continue their reasoning after a tool result.
fn map_response_to_chat_history(response: &OpenAIResponse) -> Vec<OpenAIRequestInput> {
    response
        .output
        .iter()
        .filter_map(|output| match output {
            OpenAIOutput::Reasoning {
                id,
                summary,
                additional_fields,
            } => Some(OpenAIRequestInput::Reasoning(OpenAIRequestReasoningItem {
                r#type: "reasoning".into(),
                id: id.clone(),
                summary: summary.clone(),
                additional_fields: additional_fields.clone(),
            })),
            OpenAIOutput::Message { content, role, .. } => {
                Some(OpenAIRequestInput::Message(OpenAIChatHistoryItem {
                    role: *role,
                    r#type: "message".into(),
                    content: OpenAIRequestInputItem::Text(
                        content.first()?.text.clone().unwrap_or_default(),
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
        .collect()
}

impl<T: HttpClient> AgenticClient for OpenAIClient<T> {
    type HistoryItem = OpenAIRequestInput;
    const SCHEMA_KEY: &'static str = OPENAI_SCHEMA_KEY;

    fn build_initial_history(&self, request: &ChatRequest<'_>) -> Vec<Self::HistoryItem> {
        let mut input: Vec<OpenAIRequestInput> = request
            .chat_history
            .clone()
            .into_iter()
            .flat_map(Vec::<OpenAIRequestInput>::from)
            .collect();

        input.push(OpenAIRequestInput::Message(OpenAIChatHistoryItem {
            role: MessageRole::User,
            r#type: "message".into(),
            content: OpenAIRequestInputItem::Text(request.message.clone()),
        }));

        input
    }

    async fn send_once(
        &self,
        history: &[Self::HistoryItem],
        tools: Option<&[SerializedTool<'_>]>,
        reasoning: Option<&ReasoningConfig>,
        max_tokens: Option<u32>,
    ) -> Result<ProviderTurn<Self::HistoryItem>, LLMError> {
        // Use config if available, otherwise fall back to env vars
        let (api_key, model) = if let Some(ref config) = self.config {
            (config.api_key.clone(), config.model.clone())
        } else {
            (
                env::var("OPENAI_API_KEY")?,
                env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string()),
            )
        };

        let wrapped_tools = tools.map(wrap_tools_for_openai);
        let request_body = OpenAIRequest {
            model: &model,
            input: history,
            reasoning: reasoning.map(Into::into),
            max_output_tokens: max_tokens,
            tools: wrapped_tools.as_deref(),
        };

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
        headers.insert("content-type", "application/json".parse()?);

        let response: OpenAIResponse = send_generation_request(
            &self.client,
            &request_body,
            &headers,
            "https://api.openai.com/v1/responses",
        )
        .await?;

        Ok(ProviderTurn {
            native_items: map_response_to_chat_history(&response),
            contents: map_response_to_chat_contents(&response),
            usage: response.usage.into(),
        })
    }
}

impl<T: HttpClient> ApiClient for OpenAIClient<T> {
    /// Send a request to the OpenAI Responses API, processing tool calls as necessary.
    /// Returns a final response after all tool calls are processed and sent back.
    async fn send_message(
        &self,
        request: &ChatRequest<'_>,
    ) -> Result<CompletionApiResponse, LLMError> {
        run_agentic_loop(self, request).await
    }
}

/// Implements the LanceDB EmbeddingFunction trait for OpenAI client.
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
            DEFAULT_OPENAI_EMBEDDING_DIM as i32, // text-embedding-3-small size
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
    use std::{
        collections::VecDeque,
        future::Future,
        pin::Pin,
        sync::{Arc, Mutex},
    };

    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;
    use zqa_macros::{test_eq, test_ok};

    use super::{
        OpenAIClient, OpenAIContent, OpenAIOutput, OpenAIOutputReasoningSummary, OpenAIResponse,
        OpenAIUsage,
    };
    use crate::config::OpenAIConfig;
    use crate::constants::DEFAULT_OPENAI_EMBEDDING_DIM;
    use crate::http_client::{HttpClient, MockHttpClient, ReqwestClient};
    use crate::llm::base::{
        ApiClient, ChatHistoryContent, ChatHistoryItem, ChatRequest, ContentType, MessageRole,
        ReasoningConfig,
    };
    use crate::llm::openai::{OpenAIInputTokensDetails, OpenAIOutputTokensDetails};
    use crate::llm::tools::test_utils::MockTool;

    #[derive(Clone)]
    struct RecordingSequentialMockHttpClient {
        responses: Arc<Mutex<VecDeque<String>>>,
        requests: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    impl RecordingSequentialMockHttpClient {
        fn new<T: serde::Serialize>(responses: impl IntoIterator<Item = T>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(
                    responses
                        .into_iter()
                        .map(|response| serde_json::to_string(&response).unwrap())
                        .collect(),
                )),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn requests(&self) -> Vec<serde_json::Value> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl HttpClient for RecordingSequentialMockHttpClient {
        fn post_json<'a, T: serde::Serialize + Send + Sync>(
            &'a self,
            _url: &'a str,
            _headers: http::HeaderMap,
            body: &'a T,
        ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>
        {
            self.requests
                .lock()
                .unwrap()
                .push(serde_json::to_value(body).unwrap());
            let responses = Arc::clone(&self.responses);

            Box::pin(async move {
                let body = responses
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("RecordingSequentialMockHttpClient: no more responses");
                let response = http::Response::builder()
                    .status(200)
                    .header("content-type", "application/json")
                    .body(bytes::Bytes::from(body))
                    .unwrap();

                Ok(reqwest::Response::from(response))
            })
        }

        fn post_form<'a>(
            &'a self,
            url: &'a str,
            headers: http::HeaderMap,
            _form_data: reqwest::multipart::Form,
        ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + '_>>
        {
            self.post_json(url, headers, &None::<usize>)
        }

        fn get_json<'a>(
            &'a self,
            url: &'a str,
            headers: http::HeaderMap,
        ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>
        {
            self.post_json(url, headers, &None::<usize>)
        }

        fn post_empty<'a>(
            &'a self,
            url: &'a str,
            headers: http::HeaderMap,
        ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>
        {
            self.post_json(url, headers, &None::<usize>)
        }
    }

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = OpenAIClient::<ReqwestClient>::default();
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
            reasoning: None,
            tools: None,
            on_tool_call: None,
            on_text: None,
        };
        let res = client.send_message(&request).await;

        test_ok!(res);
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
            model: "gpt-5.2-2025-12-11".to_string(),
            usage: OpenAIUsage {
                input_tokens: 5,
                output_tokens: 10,
                total_tokens: 15,
                input_tokens_details: Some(OpenAIInputTokensDetails {
                    cache_write_tokens: 0,
                    cached_tokens: 0,
                }),
                output_tokens_details: Some(OpenAIOutputTokensDetails {
                    reasoning_tokens: 0,
                }),
            },
            output: vec![OpenAIOutput::Message {
                id: "msg_id".into(),
                status: "completed".into(),
                role: MessageRole::Assistant,
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
            reasoning: None,
            tools: None,
            on_tool_call: None,
            on_text: None,
        };
        let res = mock_client.send_message(&request).await;

        test_ok!(res);
        let res = res.unwrap();
        test_eq!(res.content.len(), 1);
        if let ContentType::Text(text) = &res.content[0] {
            test_eq!(text, "Hi there!");
        } else {
            panic!("Expected Text content type");
        }
        test_eq!(res.usage.input_tokens, 5);
        test_eq!(res.usage.output_tokens, 10);
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

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        test_eq!(vector.len(), 6);
        test_eq!(vector.value_length(), DEFAULT_OPENAI_EMBEDDING_DIM as i32);
    }

    #[tokio::test]
    async fn test_request_with_tool_works() {
        dotenv().ok();

        let client = OpenAIClient::<ReqwestClient>::default();
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "This is a test. Call the `mock_tool`, passing in a `name`, and ensure it returns a greeting".into(),
            reasoning: None,
            tools: Some(&[Box::new(tool)]),
            on_tool_call: None,
            on_text: None,
        };

        let res = client.send_message(&request).await;

        test_ok!(res);
        assert!(call_count.lock().unwrap().eq(&1_usize));
    }

    fn mock_response(
        id: &str,
        input_tokens: u32,
        output_tokens: u32,
        output: Vec<OpenAIOutput>,
    ) -> OpenAIResponse {
        OpenAIResponse {
            id: id.into(),
            created_at: 0,
            model: "gpt-5".into(),
            usage: OpenAIUsage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens + output_tokens,
                input_tokens_details: Some(OpenAIInputTokensDetails {
                    cache_write_tokens: 0,
                    cached_tokens: 0,
                }),
                output_tokens_details: Some(OpenAIOutputTokensDetails {
                    reasoning_tokens: 0,
                }),
            },
            output,
        }
    }

    fn reasoning_output(
        id: &str,
        summary: &str,
        additional_fields: serde_json::Map<String, serde_json::Value>,
    ) -> OpenAIOutput {
        OpenAIOutput::Reasoning {
            id: id.into(),
            summary: Some(vec![OpenAIOutputReasoningSummary {
                r#type: "summary_text".into(),
                text: summary.into(),
            }]),
            additional_fields,
        }
    }

    fn text_output(text: &str) -> OpenAIOutput {
        OpenAIOutput::Message {
            id: "msg-1".into(),
            status: "completed".into(),
            role: MessageRole::Assistant,
            content: vec![OpenAIContent {
                r#type: "output_text".into(),
                text: Some(text.into()),
            }],
        }
    }

    fn assert_request_configuration(request: &serde_json::Value) {
        assert_eq!(request["reasoning"]["effort"].as_str(), Some("high"));
        assert_eq!(request["reasoning"]["summary"].as_str(), Some("detailed"));
        assert_eq!(request["tools"][0]["type"].as_str(), Some("function"));
    }

    fn input_item<'a>(input: &'a [serde_json::Value], item_type: &str) -> &'a serde_json::Value {
        input
            .iter()
            .find(|item| item["type"].as_str() == Some(item_type))
            .unwrap()
    }

    #[tokio::test]
    async fn test_callbacks_fire_and_reasoning_is_replayed() {
        let mut reasoning_fields = serde_json::Map::new();
        reasoning_fields.insert(
            "encrypted_content".into(),
            serde_json::Value::String("opaque-reasoning".into()),
        );
        let tool_call_response = mock_response(
            "resp-1",
            10,
            5,
            vec![
                reasoning_output("rs-1", "I need the tool result first.", reasoning_fields),
                OpenAIOutput::FunctionCall {
                    call_id: "call-1".into(),
                    name: "mock_tool".into(),
                    arguments: serde_json::Value::String(r#"{"name":"Alice"}"#.into()),
                },
            ],
        );
        let text_response = mock_response(
            "resp-2",
            20,
            8,
            vec![
                reasoning_output("rs-2", "I can now answer.", serde_json::Map::new()),
                text_output("Done!"),
            ],
        );
        let call_count = Arc::new(Mutex::new(0_usize));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };
        let tool_call_count = Arc::new(Mutex::new(0_usize));
        let text_segments = Arc::new(Mutex::new(Vec::new()));
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Test".into(),
            reasoning: Some(ReasoningConfig {
                max_tokens: None,
                effort: Some("high".into()),
                summary: Some("detailed".into()),
            }),
            tools: Some(&[Box::new(tool)]),
            on_tool_call: Some(Arc::new({
                let tool_call_count = Arc::clone(&tool_call_count);
                move |_| *tool_call_count.lock().unwrap() += 1
            })),
            on_text: Some(Arc::new({
                let text_segments = Arc::clone(&text_segments);
                move |text| text_segments.lock().unwrap().push(text.to_string())
            })),
        };
        let http_client =
            RecordingSequentialMockHttpClient::new([tool_call_response, text_response]);
        let mock_client = OpenAIClient {
            client: http_client.clone(),
            config: Some(OpenAIConfig {
                api_key: "test".into(),
                model: "gpt-5".into(),
                ..OpenAIConfig::default()
            }),
        };

        let res = mock_client.send_message(&request).await;
        test_ok!(res);
        let res = res.unwrap();
        test_eq!(res.usage.input_tokens, 30);
        test_eq!(res.usage.output_tokens, 13);
        test_eq!(*call_count.lock().unwrap(), 1_usize);
        test_eq!(*tool_call_count.lock().unwrap(), 1_usize);
        assert_eq!(
            *text_segments.lock().unwrap(),
            [
                "<reasoning>I need the tool result first.</reasoning>",
                "<reasoning>I can now answer.</reasoning>",
                "Done!"
            ]
        );

        let requests = http_client.requests();
        test_eq!(requests.len(), 2);
        requests.iter().for_each(assert_request_configuration);
        let second_input = requests[1]["input"].as_array().unwrap();
        let reasoning = input_item(second_input, "reasoning");
        assert_eq!(reasoning["id"].as_str(), Some("rs-1"));
        assert_eq!(
            reasoning["summary"][0]["text"].as_str(),
            Some("I need the tool result first.")
        );
        assert_eq!(
            reasoning["encrypted_content"].as_str(),
            Some("opaque-reasoning")
        );
        assert_eq!(
            input_item(second_input, "function_call")["arguments"].as_str(),
            Some(r#"{"name":"Alice"}"#)
        );
        assert_eq!(
            input_item(second_input, "function_call_output")["output"].as_str(),
            Some("Hello, Alice!")
        );
    }

    #[tokio::test]
    async fn test_followup_queries_work() {
        dotenv().ok();

        let client = OpenAIClient::<ReqwestClient>::default();
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
                    role: MessageRole::User,
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
