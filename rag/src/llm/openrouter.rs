//! Functions, structs, and trait implementations for interacting with the OpenRouter API. This
//! module includes support for text generation only.

use crate::common::request_with_backoff;
use crate::llm::base::{ChatHistoryContent, ContentType, ToolCallRequest};
use crate::llm::tools::{SerializedTool, get_owned_tools, process_tool_calls};
use std::collections::HashMap;
use std::env;

use http::HeaderMap;
use serde::{Deserialize, Serialize};

use super::base::{ApiClient, ChatHistoryItem, ChatRequest, CompletionApiResponse};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};

const DEFAULT_MODEL: &str = "openai/gpt-4o";

/// A generic client class for now. We can add stuff here later for
/// all the features OpenRouter supports.
#[derive(Debug, Clone)]
pub struct OpenRouterClient<T: HttpClient = ReqwestClient> {
    /// The HTTP client. The generic parameter allows for mocking in tests.
    pub client: T,
    /// Optional configuration for the OpenRouter client.
    pub config: Option<crate::config::OpenRouterConfig>,
}

impl<T: HttpClient + Default> Default for OpenRouterClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OpenRouterClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new OpenRouterClient instance without configuration
    /// (will fall back to environment variables)
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new OpenRouterClient instance with provided configuration
    pub fn with_config(config: crate::config::OpenRouterConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }
}

/// OpenRouter-specific message format
#[derive(Clone, Debug, Serialize)]
struct OpenRouterMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterToolCall>>,
    /// Tool call ID (only for tool role messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<ChatHistoryItem> for OpenRouterMessage {
    fn from(item: ChatHistoryItem) -> Self {
        convert_to_openrouter_messages(&item)
            .into_iter()
            .next()
            .unwrap_or(OpenRouterMessage {
                role: item.role,
                content: None,
                tool_calls: None,
                tool_call_id: None,
            })
    }
}

/// Wrapper for tools in OpenRouter format
#[derive(Serialize)]
struct OpenRouterTool<'a> {
    r#type: &'static str,
    function: &'a SerializedTool<'a>,
}

/// Represents a request to the OpenRouter API
#[derive(Serialize)]
struct OpenRouterRequest<'a> {
    /// The model to use for the request (e.g., "google/gemini-2.5-flash")
    model: &'a str,
    /// The conversation history and current message
    messages: &'a [OpenRouterMessage],
    /// The tools passed in
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenRouterTool<'a>>>,
}

/// Convert ChatHistoryItem to OpenRouterMessage
fn convert_to_openrouter_messages(item: &ChatHistoryItem) -> Vec<OpenRouterMessage> {
    let mut messages = Vec::new();
    let mut content_text = String::new();
    let mut tool_calls = Vec::new();

    for c in &item.content {
        match c {
            ChatHistoryContent::Text(text) => {
                if !content_text.is_empty() {
                    content_text.push(' ');
                }
                content_text.push_str(text);
            }
            ChatHistoryContent::ToolCallRequest(req) => {
                tool_calls.push(OpenRouterToolCall {
                    id: req.id.clone(),
                    r#type: "function".into(),
                    function: OpenRouterFunction {
                        name: req.tool_name.clone(),
                        arguments: serde_json::to_string(&req.args).unwrap_or_default(),
                    },
                });
            }
            ChatHistoryContent::ToolCallResponse(res) => {
                // Tool responses in OpenRouter are sent as a separate message with role "tool"
                messages.push(OpenRouterMessage {
                    role: "tool".into(),
                    content: Some(serde_json::to_string(&res.result).unwrap_or_default()),
                    tool_calls: None,
                    tool_call_id: Some(res.id.clone()),
                });
            }
        }
    }

    // Add the main message if it has text or tool calls
    if !content_text.is_empty() || !tool_calls.is_empty() {
        messages.insert(
            0,
            OpenRouterMessage {
                role: item.role.clone(),
                content: if content_text.is_empty() {
                    None
                } else {
                    Some(content_text)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
            },
        );
    }

    messages
}

/// Helper to build messages and tools from a ChatRequest.
/// Returns owned data that can then be borrowed by OpenRouterRequest.
fn build_openrouter_messages_and_tools<'a>(
    req: &'a ChatRequest<'a>,
) -> (Vec<OpenRouterMessage>, Option<Vec<SerializedTool<'a>>>) {
    let mut messages: Vec<OpenRouterMessage> = req
        .chat_history
        .iter()
        .flat_map(convert_to_openrouter_messages)
        .collect();

    messages.push(OpenRouterMessage {
        role: "user".to_owned(),
        content: Some(req.message.clone()),
        tool_calls: None,
        tool_call_id: None,
    });

    let owned_tools: Option<Vec<SerializedTool<'a>>> = get_owned_tools(req.tools);

    (messages, owned_tools)
}

/// Token usage statistics returned by the OpenRouter API
#[derive(Clone, Serialize, Deserialize)]
struct OpenRouterUsageStats {
    /// Number of tokens in the input prompt
    prompt_tokens: u32,
    /// Number of tokens in the generated response
    completion_tokens: u32,
    /// Total token usage. Includes reasoning tokens
    total_tokens: u32,
}

/// Represents a tool call in the OpenRouter API format
#[derive(Clone, Debug, Serialize, Deserialize)]
struct OpenRouterToolCall {
    /// Tool call ID
    id: String,
    /// Type (always "function")
    r#type: String,
    /// Function details
    function: OpenRouterFunction,
}

/// Function details for a tool call
#[derive(Clone, Debug, Serialize, Deserialize)]
struct OpenRouterFunction {
    /// Function name
    name: String,
    /// Function arguments as JSON string
    arguments: String,
}

/// The part of the API response containing the actual content.
#[derive(Clone, Serialize, Deserialize)]
struct OpenRouterResponseMessage {
    /// Usually "assistant"
    role: String,
    /// The model response
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Tool calls made by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterToolCall>>,
    /// Information on why a response was refused, if any
    refusal: Option<String>,
}

/// Content block in an OpenRouter API response
#[derive(Clone, Serialize, Deserialize)]
struct OpenRouterResponseChoices {
    /// The model response and related metadata
    message: OpenRouterResponseMessage,
    /// For multiple responses, a mapping to log-probabilities
    logprobs: Option<HashMap<String, f64>>,
    /// Finish reason, usually "stop", though it might indicate token limit
    finish_reason: String,
    /// Index of the message
    index: usize,
}

/// Response from the OpenRouter API
/// * `content` - The content blocks in the response
#[derive(Clone, Serialize, Deserialize)]
struct OpenRouterResponse {
    /// Unique identifier for the response
    id: String,
    /// The model that generated the response
    model: String,
    /// The model provider (e.g., "OpenAI")
    provider: String,
    /// The type of this message; we expect it to be "chat.completion"
    object: String,
    /// Timestamp this object was created
    created: u64,
    /// Token usage statistics
    usage: OpenRouterUsageStats,
    /// Model response choices
    choices: Vec<OpenRouterResponseChoices>,
}

/// Send an API request to OpenRouter.
///
/// # Arguments:
///
/// * `client`: An `HttpClient` implementation.
/// * `headers`: A set of headers to pass.
/// * `req`: The request body to send
///
/// # Returns
///
/// The OpenRouter-specific response.
async fn send_openrouter_request<'a>(
    client: &impl HttpClient,
    headers: &HeaderMap,
    req: &OpenRouterRequest<'a>,
) -> Result<OpenRouterResponse, LLMError> {
    const MAX_RETRIES: usize = 3;
    let res = request_with_backoff(
        client,
        "https://openrouter.ai/api/v1/chat/completions",
        headers,
        req,
        MAX_RETRIES,
    )
    .await?;

    let body = res.text().await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let response: OpenRouterResponse = serde_json::from_value(json.clone()).map_err(|err| {
        eprintln!("Failed to deserialize OpenRouter response: we got the response {json}");

        LLMError::DeserializationError(err.to_string())
    })?;

    Ok(response)
}

/// Convert OpenRouter response message into provider-agnostic `ChatHistoryContent` items.
fn map_response_to_chat_contents(message: &OpenRouterResponseMessage) -> Vec<ChatHistoryContent> {
    let mut contents = Vec::new();

    if let Some(text) = &message.content
        && !text.is_empty()
    {
        contents.push(ChatHistoryContent::Text(text.clone()));
    }

    if let Some(tool_calls) = &message.tool_calls {
        for tc in tool_calls {
            let args: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);

            contents.push(ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                id: tc.id.clone(),
                tool_name: tc.function.name.clone(),
                args,
            }));
        }
    }

    contents
}

impl<T: HttpClient> ApiClient for OpenRouterClient<T> {
    /// Send a request to the OpenRouter API, processing tool calls as necessary. Returns a final
    /// response after all tool calls are processed and sent back to the API.
    async fn send_message<'a>(
        &self,
        request: &'a ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, LLMError> {
        // Use config if available, otherwise fall back to env vars
        let (api_key, model) = if let Some(ref config) = self.config {
            (config.api_key.clone(), config.model.clone())
        } else {
            (
                env::var("OPENROUTER_API_KEY")?,
                env::var("OPENROUTER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            )
        };

        let mut headers = HeaderMap::new();
        let auth = format!("Bearer {api_key}");
        headers.insert("Authorization", auth.parse()?);
        headers.insert("Content-Type", "application/json".parse()?);

        // Build the initial messages and tools (owned)
        let (mut chat_history, tools) = build_openrouter_messages_and_tools(request);

        // Helper to create wrapped tools
        let make_wrapped_tools = || {
            tools.as_ref().map(|t| {
                t.iter()
                    .map(|tool| OpenRouterTool {
                        r#type: "function",
                        function: tool,
                    })
                    .collect()
            })
        };

        // Create the initial request
        let req_body = OpenRouterRequest {
            model: &model,
            messages: &chat_history,
            tools: make_wrapped_tools(),
        };

        let mut response = send_openrouter_request(&self.client, &headers, &req_body).await?;

        let mut choice = response.choices.into_iter().next().ok_or_else(|| {
            LLMError::DeserializationError("OpenRouter response contained no choices".to_string())
        })?;

        let mut has_tool_calls: bool = choice
            .message
            .tool_calls
            .as_ref()
            .map(|tc| !tc.is_empty())
            .unwrap_or(false);

        // Append the assistant's response to chat history
        chat_history.push(OpenRouterMessage {
            role: choice.message.role.clone(),
            content: choice.message.content.clone(),
            tool_calls: choice.message.tool_calls.clone(),
            tool_call_id: None,
        });

        let mut contents: Vec<ContentType> = Vec::new();

        while has_tool_calls {
            let converted_contents = map_response_to_chat_contents(&choice.message);
            process_tool_calls(
                &mut chat_history,
                &mut contents,
                &converted_contents,
                tools.as_ref().ok_or_else(|| {
                    LLMError::ToolCallError(
                        "Model returned tool calls, but no tools were provided.".to_string(),
                    )
                })?,
            )
            .await?;

            // Create a new request borrowing the updated chat history
            let updated_req_body = OpenRouterRequest {
                model: &model,
                messages: &chat_history,
                tools: make_wrapped_tools(),
            };

            response = send_openrouter_request(&self.client, &headers, &updated_req_body).await?;

            choice = response.choices.into_iter().next().ok_or_else(|| {
                LLMError::DeserializationError(
                    "OpenRouter response contained no choices".to_string(),
                )
            })?;

            // Append the new response to chat history
            chat_history.push(OpenRouterMessage {
                role: choice.message.role.clone(),
                content: choice.message.content.clone(),
                tool_calls: choice.message.tool_calls.clone(),
                tool_call_id: None,
            });

            has_tool_calls = choice
                .message
                .tool_calls
                .as_ref()
                .map(|tc| !tc.is_empty())
                .unwrap_or(false);
        }

        // Process the final response (which has no tool calls) to extract text content
        if let Some(text) = &choice.message.content
            && !text.is_empty()
        {
            contents.push(ContentType::Text(text.clone()));
        }

        Ok(CompletionApiResponse {
            content: contents,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use dotenv::dotenv;

    use super::*;
    use crate::llm::base::{ApiClient, ChatRequest};
    use crate::llm::http_client::{MockHttpClient, ReqwestClient};
    use crate::llm::tools::test_utils::MockTool;

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = OpenRouterClient::<ReqwestClient>::default();
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
            tools: None,
        };

        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("OpenRouter test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_request_with_mock() {
        // Load environment variables from .env file
        dotenv().ok();

        // Create a proper OpenRouterResponse that matches the structure we expect to deserialize
        let mock_response = OpenRouterResponse {
            id: String::from("test"),
            model: String::from("openai/gpt-3.5-turbo"),
            provider: String::from("OpenAI"),
            object: String::from("chat.completion"),
            created: 1000,
            usage: OpenRouterUsageStats {
                prompt_tokens: 14,
                completion_tokens: 163,
                total_tokens: 177,
            },
            choices: vec![OpenRouterResponseChoices {
                message: OpenRouterResponseMessage {
                    role: String::from("assistant"),
                    content: Some(String::from("Hi there! How can I help you today?")),
                    tool_calls: None,
                    refusal: Some(String::new()),
                },
                finish_reason: String::from("stop"),
                logprobs: Some(HashMap::new()),
                index: 0,
            }],
        };

        let mock_http_client = MockHttpClient::new(mock_response);
        let mock_client = OpenRouterClient {
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
            println!("OpenRouter test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());

        let res = res.unwrap();
        assert_eq!(res.input_tokens, 14);
        assert_eq!(res.output_tokens, 163);
        assert_eq!(res.content.len(), 1);
        if let ContentType::Text(text) = &res.content[0] {
            assert_eq!(text, "Hi there! How can I help you today?");
        } else {
            panic!("Expected Text content type");
        }
    }

    #[tokio::test]
    async fn test_request_with_tool_works() {
        dotenv().ok();

        let client = OpenRouterClient::<ReqwestClient>::default();
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
            schema_key: "parameters".into(),
        };

        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
            tools: Some(&[Box::new(tool)]),
        };

        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("OpenRouter test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
        assert!(call_count.lock().unwrap().eq(&1_usize));
    }
}
