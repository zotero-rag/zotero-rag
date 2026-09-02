//! Functions, structs, and trait implementations for interacting with the Gemini API. This module
//! includes support for both text generation and embedding, and tool calling is supported.

use std::env;

use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use super::base::ChatRequest;
use super::errors::LLMError;
use crate::clients::gemini::{GeminiClient, get_gemini_api_key};
use crate::constants::{DEFAULT_GEMINI_MODEL, DEFAULT_GEMINI_REASONING_BUDGET};
use crate::http_client::HttpClient;
use crate::llm::base::{
    AgenticClient, ChatHistoryContent, ChatHistoryItem, MessageRole, ProviderTurn, ReasoningConfig,
    ToolCallRequest, send_generation_request,
};
use crate::llm::tools::{GEMINI_SCHEMA_KEY, SerializedTool};
use crate::pricing::ModelUsage;

/// A function (tool) call request from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionCall {
    /// A unique ID for the function call (optional in responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    /// The name of the tool (function) to call
    name: String,
    /// The function parameters
    args: serde_json::Value,
}

/// A result of a tool call, to be sent to the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionResult {
    /// The ID of the corresponding function call
    id: String,
    /// The name of the function
    name: String,
    /// The function response in JSON format
    response: serde_json::Value,
}

/// A content part in a request to the Gemini API
#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub(crate) enum GeminiPart {
    Text {
        text: String,
        /// Whether this part is a thought summary rather than regular response text.
        #[serde(skip_serializing_if = "Option::is_none")]
        thought: Option<bool>,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResult {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResult,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
}

/// Gemini uses "model", not "assistant", so we need our own enum.
#[derive(Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum GeminiMessageRole {
    /// The model response.
    Model,
    /// The user message or response.
    User,
}

/// Content for requests to the Gemini API
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiContent {
    role: GeminiMessageRole,
    parts: Vec<GeminiPart>,
}

/// Instructions that guide Gemini for the entire request.
#[derive(Serialize, Clone)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

impl From<ChatHistoryItem> for Vec<GeminiContent> {
    fn from(value: ChatHistoryItem) -> Self {
        vec![value.into()]
    }
}

impl From<ChatHistoryItem> for GeminiContent {
    fn from(value: ChatHistoryItem) -> Self {
        Self {
            role: match value.role {
                MessageRole::User | MessageRole::Tool => GeminiMessageRole::User,
                MessageRole::Assistant => GeminiMessageRole::Model,
            },
            parts: value
                .content
                .into_iter()
                .filter_map(|c| match c {
                    ChatHistoryContent::Text(text) => Some(GeminiPart::Text {
                        text,
                        thought: None,
                        thought_signature: None,
                    }),
                    ChatHistoryContent::Reasoning(_) => None,
                    ChatHistoryContent::ToolCallRequest(tool_call) => {
                        Some(GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                id: Some(tool_call.id),
                                name: tool_call.tool_name,
                                args: tool_call.args,
                            },
                            thought_signature: None,
                        })
                    }
                    ChatHistoryContent::ToolCallResponse(tool_res) => {
                        // Wrap the result in an object with a "result" field if it's not already an object
                        let response = if tool_res.result.is_object() {
                            tool_res.result
                        } else {
                            serde_json::json!({ "result": tool_res.result })
                        };

                        Some(GeminiPart::FunctionResult {
                            function_response: GeminiFunctionResult {
                                id: tool_res.id,
                                name: tool_res.tool_name,
                                response,
                            },
                            thought_signature: None,
                        })
                    }
                })
                .collect::<Vec<_>>(),
        }
    }
}

/// Thinking config in case reasoning models are used
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
    /// Thinking level (`"minimal"`, `"low"`, `"medium"`, `"high"`) for Gemini 3+ models,
    /// which use this instead of a token budget. See
    /// <https://ai.google.dev/gemini-api/docs/thinking>.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<String>,
}

/// Whether the given Gemini model generation uses the `thinkingLevel` API (Gemini 3 and
/// later) rather than the older `thinkingBudget` API (Gemini 2.5). A model name of the form
/// `gemini-{major}...` is parsed for its major version; unrecognized names are assumed to
/// use `thinkingBudget`.
fn uses_thinking_level(model: &str) -> bool {
    model
        .to_lowercase()
        .strip_prefix("gemini-")
        .and_then(|version| version.split(['-', '.']).next())
        .and_then(|major| major.parse::<u32>().ok())
        .is_some_and(|major| major >= 3)
}

/// Map a reasoning effort to a Gemini `thinkingLevel` (`minimal`, `low`, `medium`, or
/// `high`). `xhigh` and `max` are mapped down to `high`, since Google's highest documented
/// level is `high`; `none` and unrecognized efforts map to `None`.
fn effort_to_thinking_level(effort: &str) -> Option<String> {
    match effort {
        "minimal" | "low" | "medium" | "high" => Some(effort.to_string()),
        "xhigh" | "max" => Some("high".to_string()),
        _ => None,
    }
}

/// Build the thinking config for a request. Gemini 3+ models use `thinkingLevel`; Gemini 2.5
/// models use `thinkingBudget`. When no level is specified for Gemini 3+, the provider chooses
/// its default level.
fn gemini_thinking_config(
    model: &str,
    reasoning: Option<&ReasoningConfig>,
) -> Option<GeminiThinkingConfig> {
    let reasoning = reasoning?;
    let level = if uses_thinking_level(model) {
        reasoning
            .effort
            .as_deref()
            .and_then(effort_to_thinking_level)
    } else {
        None
    };
    let budget = (!uses_thinking_level(model)).then(|| {
        reasoning
            .max_tokens
            .unwrap_or(DEFAULT_GEMINI_REASONING_BUDGET)
    });

    Some(GeminiThinkingConfig {
        include_thoughts: Some(true),
        thinking_budget: budget,
        thinking_level: level,
    })
}

/// Optional text generation configuration
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiToolDeclaration<'a> {
    function_declarations: Vec<SerializedTool<'a>>,
}

/// The request body for text generation
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiRequestBody<'a> {
    contents: &'a [GeminiContent],
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a GeminiToolDeclaration<'a>>,
}

/// Helper to build contents, config, and tools from a ChatRequest.
/// Returns owned data that can then be borrowed by GeminiRequestBody.
fn build_gemini_request_data<'a>(
    model: &str,
    max_tokens: Option<u32>,
    tools: Option<&'a [SerializedTool<'_>]>,
    reasoning: Option<&ReasoningConfig>,
) -> (
    Option<GeminiGenerationConfig>,
    Option<GeminiToolDeclaration<'a>>,
) {
    let model_max = max_tokens.or_else(|| {
        env::var("GEMINI_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
    });

    let generation_config = Some(GeminiGenerationConfig {
        max_output_tokens: model_max,
        temperature: Some(1.0),
        top_k: Some(1),
        top_p: Some(1.0),
        thinking_config: gemini_thinking_config(model, reasoning),
    });

    let tools = tools.map(|tools| GeminiToolDeclaration {
        function_declarations: tools.to_vec(),
    });

    (generation_config, tools)
}

/// Token details by modality in usage metadata
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiTokenDetails {
    modality: String,
    token_count: u32,
}

/// Usage metadata received from the Gemini text generation response.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cached_content_token_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_use_prompt_token_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thoughts_token_count: Option<u32>,
    candidates_token_count: u32,
    total_token_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<Vec<GeminiTokenDetails>>,
}

impl From<GeminiUsageMetadata> for ModelUsage {
    fn from(val: GeminiUsageMetadata) -> Self {
        ModelUsage {
            input_tokens: val.prompt_token_count,
            input_cache_read: val.cached_content_token_count.unwrap_or_default(),
            // The Gemini API doesn't seem to distinguish between cache reads/writes, and only gives
            // us one number. The `prompt_tokens_details` number is a split by modality.
            // See: https://ai.google.dev/api/generate-content#UsageMetadata
            input_cache_written: 0,
            output_tokens: val.candidates_token_count,
            reasoning_tokens: val.thoughts_token_count.unwrap_or_default(),
        }
    }
}

/// One of several response candidates.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseCandidate {
    content: GeminiContent,
    finish_reason: String,
}

/// Text generation response from the Gemini API.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseBody {
    candidates: Vec<GeminiResponseCandidate>,
    usage_metadata: GeminiUsageMetadata,
}

/// Convert Gemini response content into provider-agnostic `ChatHistoryContent` items.
///
/// Tool results should never appear in API responses; if encountered, they are ignored
/// with a warning.
fn map_response_to_chat_contents(contents: &[GeminiPart]) -> Vec<ChatHistoryContent> {
    contents.iter().enumerate().filter_map(|(idx, c)| {
        match c {
            GeminiPart::Text{text, thought, ..} => {
                if *thought == Some(true) {
                    Some(ChatHistoryContent::Reasoning(text.clone()))
                } else {
                    Some(ChatHistoryContent::Text(text.clone()))
                }
            },
            GeminiPart::FunctionCall{function_call: fc, ..} => Some(ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                    // Generate an ID if not provided by the API
                    id: fc.id.clone().unwrap_or_else(|| format!("{}_{}", fc.name, idx)),
                    tool_name: fc.name.clone(),
                    args: fc.args.clone()
                })),
            GeminiPart::FunctionResult {..} => {
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be ignored."
                );

                None
            }
        }
    }).collect::<Vec<_>>()
}

impl<T: HttpClient> AgenticClient for GeminiClient<T> {
    type HistoryItem = GeminiContent;
    const SCHEMA_KEY: &'static str = GEMINI_SCHEMA_KEY;

    fn build_initial_history(&self, request: &ChatRequest<'_>) -> Vec<Self::HistoryItem> {
        let mut contents: Vec<GeminiContent> = request
            .chat_history
            .iter()
            .cloned()
            .map(Into::into)
            .collect();

        contents.push(GeminiContent {
            role: GeminiMessageRole::User,
            parts: vec![GeminiPart::Text {
                text: request.message.clone(),
                thought: None,
                thought_signature: None,
            }],
        });

        contents
    }

    async fn send_once(
        &self,
        history: &[Self::HistoryItem],
        system_prompt: Option<&str>,
        tools: Option<&[SerializedTool<'_>]>,
        reasoning: Option<&ReasoningConfig>,
        max_tokens: Option<u32>,
    ) -> Result<super::base::ProviderTurn<Self::HistoryItem>, LLMError> {
        let key = get_gemini_api_key()?;
        let model = match &self.config {
            None => env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_GEMINI_MODEL.to_string()),
            Some(config) => config.model.clone(),
        };

        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse()?);
        headers.insert("x-goog-api-key", key.parse()?);

        // Build the initial contents, config, and tools (owned)
        let (generation_config, tools) =
            build_gemini_request_data(&model, max_tokens, tools, reasoning);

        // Create the initial request borrowing
        let request = GeminiRequestBody {
            contents: history,
            system_instruction: system_prompt.map(|text| GeminiSystemInstruction {
                parts: vec![GeminiPart::Text {
                    text: text.to_owned(),
                    thought: None,
                    thought_signature: None,
                }],
            }),
            generation_config: generation_config.clone(),
            tools: tools.as_ref(),
        };

        let response: GeminiResponseBody = send_generation_request(
            &self.client,
            request,
            &headers,
            &format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            ),
        )
        .await?;

        let first_candidate = response
            .candidates
            .first()
            .ok_or_else(|| LLMError::GenericLLMError("No candidates in Gemini response".into()))?;

        Ok(ProviderTurn {
            contents: map_response_to_chat_contents(&first_candidate.content.parts),
            native_items: vec![GeminiContent {
                role: GeminiMessageRole::Model,
                parts: first_candidate.content.parts.clone(),
            }],
            usage: response.usage_metadata.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;
    use zqa_macros::{test_eq, test_ok};

    use super::*;
    use crate::clients::gemini::GeminiClient;
    use crate::constants::DEFAULT_GEMINI_EMBEDDING_DIM;
    use crate::http_client::{MockHttpClient, RecordingSequentialMockHttpClient, ReqwestClient};
    use crate::llm::base::{AgenticClient, ChatHistoryItem, ChatRequest, ContentType};
    use crate::llm::tools::test_utils::MockTool;

    #[test]
    fn test_uses_thinking_level() {
        test_eq!(uses_thinking_level("gemini-3-pro-preview"), true);
        test_eq!(uses_thinking_level("gemini-3.7-flash"), true);
        test_eq!(uses_thinking_level("Gemini-3-flash-preview"), true);
        test_eq!(uses_thinking_level("gemini-2.5-pro"), false);
        test_eq!(uses_thinking_level("gemini-2.5-flash-lite"), false);
        // Unrecognized names conservatively use the budget API.
        test_eq!(uses_thinking_level("gemma-3-27b"), false);
        test_eq!(uses_thinking_level("gemini"), false);
    }

    #[test]
    fn test_effort_to_thinking_level() {
        test_eq!(
            effort_to_thinking_level("minimal"),
            Some("minimal".to_string())
        );
        test_eq!(effort_to_thinking_level("high"), Some("high".to_string()));
        // `xhigh` and `max` are mapped down, since Google caps at `high`.
        test_eq!(effort_to_thinking_level("xhigh"), Some("high".to_string()));
        test_eq!(effort_to_thinking_level("max"), Some("high".to_string()));
        test_eq!(effort_to_thinking_level("none"), None);
        test_eq!(effort_to_thinking_level("bogus"), None);
    }

    #[test]
    fn test_gemini_thinking_config() {
        // Gemini 3: an effort maps to a thinking level, with no budget by default.
        let reasoning = ReasoningConfig {
            max_tokens: None,
            effort: Some("high".into()),
            summary: None,
        };
        let config = gemini_thinking_config("gemini-3-pro-preview", Some(&reasoning)).unwrap();
        test_eq!(config.thinking_level, Some("high".to_string()));
        test_eq!(config.thinking_budget, None);
        test_eq!(config.include_thoughts, Some(true));

        // Gemini 3 does not send the legacy budget, even when one was requested.
        let reasoning = ReasoningConfig {
            max_tokens: Some(4096),
            effort: Some("low".into()),
            summary: None,
        };
        let config = gemini_thinking_config("gemini-3.7-flash", Some(&reasoning)).unwrap();
        test_eq!(config.thinking_level, Some("low".to_string()));
        test_eq!(config.thinking_budget, None);

        // Gemini 2.5: the budget API is used, regardless of effort.
        let reasoning = ReasoningConfig {
            max_tokens: Some(8192),
            effort: Some("high".into()),
            summary: None,
        };
        let config = gemini_thinking_config("gemini-2.5-pro", Some(&reasoning)).unwrap();
        test_eq!(config.thinking_level, None);
        test_eq!(config.thinking_budget, Some(8192));

        // No effort and no budget: fall back to the default budget.
        let reasoning = ReasoningConfig {
            max_tokens: None,
            effort: None,
            summary: None,
        };
        let config = gemini_thinking_config("gemini-2.5-flash", Some(&reasoning)).unwrap();
        test_eq!(
            config.thinking_budget,
            Some(DEFAULT_GEMINI_REASONING_BUDGET)
        );

        // No reasoning at all: no thinking config is sent.
        test_eq!(
            gemini_thinking_config("gemini-3-pro-preview", None).is_none(),
            true
        );
    }

    #[tokio::test]
    async fn test_send_message_with_mock() {
        dotenv().ok();

        let mock_response = GeminiResponseBody {
            candidates: vec![GeminiResponseCandidate {
                content: GeminiContent {
                    role: GeminiMessageRole::Model,
                    parts: vec![GeminiPart::Text {
                        text: "Hello from Gemini!".into(),
                        thought: None,
                        thought_signature: None,
                    }],
                },
                finish_reason: "stop".into(),
            }],
            usage_metadata: GeminiUsageMetadata {
                prompt_token_count: 7,
                candidates_token_count: 11,
                total_token_count: 18,
                thoughts_token_count: Some(0),
                prompt_tokens_details: None,
                cached_content_token_count: None,
                tool_use_prompt_token_count: None,
            },
        };

        let mock_http = MockHttpClient::new(mock_response);
        let client = GeminiClient {
            client: mock_http,
            config: None,
        };

        let request = ChatRequest {
            message: "foo".into(),
            chat_history: vec![ChatHistoryItem {
                role: MessageRole::Assistant,
                content: vec![ChatHistoryContent::Text("Prior".into())],
            }],
            max_tokens: Some(256),
            system_prompt: None,
            reasoning: None,
            tools: None,
            on_tool_call: None,
            on_text: None,
            tool_iteration_limit: None,
        };
        let res = client.send_message(&request).await;
        test_ok!(res);
        let res = res.unwrap();
        test_eq!(res.content.len(), 1);
        if let ContentType::Text(text) = &res.content[0] {
            test_eq!(text, "Hello from Gemini!");
        } else {
            panic!("Expected Text content type");
        }
        test_eq!(res.usage.input_tokens, 7);
        test_eq!(res.usage.output_tokens, 11);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings_mock() {
        dotenv().ok();

        // Build a deterministic mock response that returns a 3-length embedding
        #[derive(Debug, Serialize, Deserialize, Clone, Default)]
        struct MockEmbeddingResp {
            embedding: MockEmbeddingVec,
        }
        #[derive(Debug, Serialize, Deserialize, Clone, Default)]
        struct MockEmbeddingVec {
            values: Vec<f32>,
        }

        let mock = MockEmbeddingResp {
            embedding: MockEmbeddingVec {
                values: vec![1.0, 0.0, -1.0],
            },
        };

        let mock_http = MockHttpClient::new(mock);
        let client = GeminiClient {
            client: mock_http,
            config: None,
        };

        let array = arrow_array::StringArray::from(vec!["A", "B", " ", "C"]);
        let embeddings = client.compute_source_embeddings(Arc::new(array));

        test_ok!(embeddings);
        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);
        test_eq!(vector.len(), 4);
        // With mock 3-length vectors, value_length should be 3
        test_eq!(vector.value_length(), 3);
    }

    #[tokio::test]
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

        let client = GeminiClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_async(Arc::new(array)).await;

        test_ok!(embeddings);

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        test_eq!(vector.len(), 6);
        test_eq!(vector.value_length(), DEFAULT_GEMINI_EMBEDDING_DIM as i32);
    }

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = GeminiClient::<ReqwestClient>::default();
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
            system_prompt: None,
            reasoning: None,
            tools: None,
            on_tool_call: None,
            on_text: None,
            tool_iteration_limit: None,
        };
        let res = client.send_message(&request).await;

        test_ok!(res);
    }

    #[tokio::test]
    async fn test_request_works_with_tools() {
        dotenv().ok();

        let client = GeminiClient::<ReqwestClient>::default();
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "This is a test. Call the `mock_tool`, passing in a `name`, and ensure it returns a greeting".into(),
            system_prompt: None,
            reasoning: None,
            tools: Some(&[Box::new(tool)]),
            on_tool_call: None,
            on_text: None,
            tool_iteration_limit: None,
        };

        let res = client.send_message(&request).await;

        test_ok!(res);
    }

    #[tokio::test]
    async fn test_callbacks_fire() {
        dotenv().ok();

        let tool_call_response = GeminiResponseBody {
            candidates: vec![GeminiResponseCandidate {
                content: GeminiContent {
                    role: GeminiMessageRole::Model,
                    parts: vec![GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            id: Some("call-1".into()),
                            name: "mock_tool".into(),
                            args: serde_json::json!({"name": "Alice"}),
                        },
                        thought_signature: None,
                    }],
                },
                finish_reason: "STOP".into(),
            }],
            usage_metadata: GeminiUsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 5,
                total_token_count: 15,
                thoughts_token_count: None,
                prompt_tokens_details: None,
                cached_content_token_count: Some(0),
                tool_use_prompt_token_count: Some(0),
            },
        };
        let text_response = GeminiResponseBody {
            candidates: vec![GeminiResponseCandidate {
                content: GeminiContent {
                    role: GeminiMessageRole::Model,
                    parts: vec![GeminiPart::Text {
                        text: "Done!".into(),
                        thought: None,
                        thought_signature: None,
                    }],
                },
                finish_reason: "STOP".into(),
            }],
            usage_metadata: GeminiUsageMetadata {
                prompt_token_count: 20,
                candidates_token_count: 8,
                total_token_count: 28,
                thoughts_token_count: None,
                prompt_tokens_details: None,
                cached_content_token_count: Some(0),
                tool_use_prompt_token_count: None,
            },
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
            system_prompt: Some("Follow the system instructions.".into()),
            reasoning: None,
            tools: Some(&[Box::new(tool)]),
            on_tool_call: Some(Arc::new(move |_| {
                *tool_call_count_cb.lock().unwrap() += 1;
            })),
            on_text: Some(Arc::new(move |s| {
                text_segments_cb.lock().unwrap().push(s.to_string());
            })),
            tool_iteration_limit: None,
        };

        let http_client =
            RecordingSequentialMockHttpClient::new([tool_call_response, text_response]);
        let mock_client = GeminiClient {
            client: http_client.clone(),
            config: None,
        };
        let res = mock_client.send_message(&request).await;
        test_ok!(res);

        test_eq!(*tool_call_count.lock().unwrap(), 1_usize);
        let texts = text_segments.lock().unwrap();
        test_eq!(texts.len(), 1);
        test_eq!(texts[0].as_str(), "Done!");
        for request in http_client.requests() {
            test_eq!(
                request["systemInstruction"]["parts"][0]["text"],
                "Follow the system instructions."
            );
        }
    }

    #[tokio::test]
    async fn test_followup_queries_work() {
        dotenv().ok();

        let client = GeminiClient::<ReqwestClient>::default();
        let first_message = ChatRequest {
            message: "What is self-attention?".into(),
            ..ChatRequest::default()
        };

        let response = client.send_message(&first_message).await;
        test_ok!(response);

        let response = response.unwrap();
        let mut chat_history = vec![ChatHistoryItem {
            role: MessageRole::User,
            content: vec![ChatHistoryContent::Text(first_message.message.clone())],
        }];
        chat_history.extend(response.history_additions);

        let second_message = ChatRequest {
            chat_history,
            message: "What are the Q, K, and V matrices?".into(),
            ..ChatRequest::default()
        };

        let response = client.send_message(&second_message).await;
        test_ok!(response);
    }
}
