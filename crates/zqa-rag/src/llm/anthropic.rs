//! A module for interacting with Anthropic's API.

use std::env;

use http::HeaderMap;
use serde::{Deserialize, Serialize};

use super::base::{ChatHistoryItem, ChatRequest};
use super::errors::LLMError;
use crate::clients::anthropic::AnthropicClient;
use crate::constants::{
    DEFAULT_ANTHROPIC_MAX_TOKENS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_ANTHROPIC_REASONING_BUDGET,
};
use crate::http_client::HttpClient;
use crate::llm::base::{
    AgenticClient, ChatHistoryContent, MessageRole, ProviderTurn, ReasoningConfig, ToolCallRequest,
    ToolCallResponse, send_generation_request,
};
use crate::llm::tools::{ANTHROPIC_SCHEMA_KEY, SerializedTool};
use crate::pricing::ModelUsage;
const DEFAULT_CLAUDE_MODEL: &str = DEFAULT_ANTHROPIC_MODEL;

/// An Anthropic-specific chat history object. This is pretty much the same as `ChatHistoryItem`,
/// except that `content` is now Anthropic-specific.
#[derive(Clone, Serialize)]
pub(crate) struct AnthropicChatHistoryItem {
    /// Either "user" or "assistant".
    pub role: MessageRole,
    /// The contents of this item.
    pub content: Vec<AnthropicResponseContent>,
}

impl From<ChatHistoryItem> for Vec<AnthropicChatHistoryItem> {
    fn from(value: ChatHistoryItem) -> Self {
        vec![value.into()]
    }
}

impl From<ChatHistoryItem> for AnthropicChatHistoryItem {
    fn from(value: ChatHistoryItem) -> Self {
        let role = value.role;
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

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicThinkingConfig {
    /// Manual extended thinking with a fixed token budget (older models).
    Enabled {
        /// "summarized" or "omitted"
        display: &'static str,
        /// Token budget for thinking
        budget_tokens: u32,
    },
    /// Adaptive thinking: the model decides when and how deeply to think. This is the only
    /// thinking mode supported by Claude Opus 4.6+ and Claude 5 models; depth is steered via
    /// `output_config.effort` instead of a token budget.
    Adaptive {
        /// "summarized" or "omitted"
        display: &'static str,
    },
}

/// Output-level request configuration, currently only the reasoning effort.
#[derive(Serialize)]
pub(crate) struct AnthropicOutputConfig {
    /// Effort level ("low", "medium", "high", "xhigh", "max") used to steer adaptive thinking.
    effort: String,
}

/// Whether the given Claude model supports adaptive thinking (`thinking.type: "adaptive"`)
/// rather than manual extended thinking (`thinking.type: "enabled"`).
///
/// The model name is parsed into a model family (Opus, Sonnet, or Haiku) and a version, and
/// adaptive thinking is enabled for versions above the per-family threshold (Opus and
/// Sonnet 4.6+, Haiku 4.5+, and all 5.x+ models); the thresholds should not need updating as
/// new models are released. Unrecognized model names are assumed to support adaptive thinking.
///
/// # Arguments
///
/// * `model` - The model name (e.g., "claude-opus-4-8")
///
/// # Returns
///
/// `true` if the model supports adaptive thinking.
fn supports_adaptive_thinking(model: &str) -> bool {
    let lower = model.to_lowercase();
    let tokens: Vec<&str> = lower.split('-').collect();
    let Some(i) = tokens
        .iter()
        .position(|t| matches!(*t, "opus" | "sonnet" | "haiku"))
    else {
        return true;
    };
    let family = tokens[i];

    // The version can follow the family ("claude-opus-4-5") or precede it
    // ("claude-3-7-sonnet"); a date suffix ("claude-opus-4-5-20251001") is ignored.
    let version = |a: usize, b: usize| {
        Some((
            tokens.get(a)?.parse::<u32>().ok()?,
            tokens.get(b)?.parse::<u32>().ok()?,
        ))
    };
    let Some((major, minor)) = i
        .checked_sub(2)
        .and_then(|j| version(j, j + 1))
        .or_else(|| version(i + 1, i + 2))
        // A bare major version ("claude-opus-4") has minor 0.
        .or_else(|| tokens.get(i + 1)?.parse::<u32>().ok().map(|m| (m, 0)))
    else {
        return true;
    };

    if major >= 5 {
        return true;
    }
    if major == 4 {
        return minor >= if family == "haiku" { 5 } else { 6 };
    }
    false
}

/// Build the thinking and output configuration for a request. Adaptive-thinking models use
/// `output_config.effort` to steer thinking; older models use a fixed `budget_tokens`.
///
/// # Arguments
///
/// * `model` - The model the request will be sent to
/// * `reasoning` - The reasoning configuration, if thinking is enabled
///
/// # Returns
///
/// The `(thinking, output_config)` pair for the request; both `None` when thinking is disabled.
pub(crate) fn make_thinking_config(
    model: &str,
    reasoning: Option<&ReasoningConfig>,
) -> (
    Option<AnthropicThinkingConfig>,
    Option<AnthropicOutputConfig>,
) {
    let Some(mut reasoning) = reasoning.cloned() else {
        return (None, None);
    };

    reasoning.normalize();

    if supports_adaptive_thinking(model) {
        let output_config = reasoning
            .effort
            .clone()
            .map(|effort| AnthropicOutputConfig { effort });
        (
            Some(AnthropicThinkingConfig::Adaptive {
                display: "summarized",
            }),
            output_config,
        )
    } else {
        (
            Some(AnthropicThinkingConfig::Enabled {
                display: "summarized",
                budget_tokens: reasoning
                    .max_tokens
                    .unwrap_or(DEFAULT_ANTHROPIC_REASONING_BUDGET),
            }),
            None,
        )
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
    /// Instructions that guide the model for the entire request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system: Option<&'a str>,
    /// Thinking/reasoning configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking: Option<AnthropicThinkingConfig>,
    /// Output configuration (reasoning effort) for adaptive-thinking models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_config: Option<AnthropicOutputConfig>,
    /// The tools passed in
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<&'a [SerializedTool<'a>]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicOutputTokensDetails {
    /// Number of output tokens the model generated as internal reasoning, including the
    /// thinking-block delimiter token
    pub(crate) thinking_tokens: u32,
}

/// Token usage statistics returned by the Anthropic API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicUsageStats {
    /// Number of tokens in the input prompt
    pub(crate) input_tokens: u32,
    /// Number of input tokens used to create the cache entry
    #[serde(default)]
    pub(crate) cache_creation_input_tokens: u32,
    /// Number of input tokens read from the cache
    #[serde(default)]
    pub(crate) cache_read_input_tokens: u32,
    /// Number of tokens in the generated response
    pub(crate) output_tokens: u32,
    /// Breakdown of output tokens by category
    pub(crate) output_tokens_details: Option<AnthropicOutputTokensDetails>,
}

impl From<AnthropicUsageStats> for ModelUsage {
    fn from(val: AnthropicUsageStats) -> Self {
        ModelUsage {
            input_tokens: val.input_tokens
                + val.cache_creation_input_tokens
                + val.cache_read_input_tokens,
            input_cache_written: val.cache_creation_input_tokens,
            input_cache_read: val.cache_read_input_tokens,
            output_tokens: val.output_tokens,
            reasoning_tokens: val.output_tokens_details.map_or(0, |d| d.thinking_tokens),
        }
    }
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
    /// The thinking content from the model.
    pub(crate) thinking: String,
    /// Opaque signature that must be returned unchanged with a tool-use continuation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) signature: Option<String>,
}

/// A redacted thinking block returned by models with extended thinking enabled.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicRedactedThinkingResponseContent {
    /// The content type. Always "redacted_thinking".
    pub(crate) r#type: String,
    /// Opaque redacted thinking data that must be returned unchanged with a tool-use continuation.
    pub(crate) data: String,
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
                .cloned()
                .unwrap_or_default(),
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
    /// A redacted thinking block from models with extended thinking enabled.
    RedactedThinking(AnthropicRedactedThinkingResponseContent),
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
    pub(crate) role: MessageRole,
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
            AnthropicResponseContent::Thinking(_)
            | AnthropicResponseContent::RedactedThinking(_) => {}
        }
    }
    out
}

impl<T: HttpClient> AgenticClient for AnthropicClient<T> {
    type HistoryItem = AnthropicChatHistoryItem;
    const SCHEMA_KEY: &'static str = ANTHROPIC_SCHEMA_KEY;

    fn build_initial_history(&self, request: &ChatRequest<'_>) -> Vec<Self::HistoryItem> {
        let mut messages: Vec<AnthropicChatHistoryItem> = request
            .chat_history
            .clone()
            .into_iter()
            .map(Into::into)
            .collect();

        messages.push(AnthropicChatHistoryItem {
            role: MessageRole::User,
            content: vec![request.message.clone().into()],
        });

        messages
    }

    async fn send_once(
        &self,
        history: &[Self::HistoryItem],
        system_prompt: Option<&str>,
        tools: Option<&[SerializedTool<'_>]>,
        reasoning: Option<&ReasoningConfig>,
        max_tokens: Option<u32>,
    ) -> Result<super::base::ProviderTurn<Self::HistoryItem>, LLMError> {
        // Use config if available, otherwise fall back to env vars
        let (api_key, model, config_max_tokens) = if let Some(ref config) = self.config {
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

        let (thinking, output_config) = make_thinking_config(&model, reasoning);
        let request = AnthropicRequest {
            model: &model,
            max_tokens: max_tokens.unwrap_or(config_max_tokens),
            messages: history,
            system: system_prompt,
            thinking,
            output_config,
            tools,
        };

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", api_key.parse()?);
        headers.insert("anthropic-version", "2023-06-01".parse()?);
        headers.insert("content-type", "application/json".parse()?);

        let response: AnthropicResponse = send_generation_request(
            &self.client,
            &request,
            &headers,
            "https://api.anthropic.com/v1/messages",
        )
        .await?;

        Ok(ProviderTurn {
            contents: map_response_to_chat_contents(&response.content),
            native_items: vec![AnthropicChatHistoryItem {
                role: MessageRole::Assistant,
                content: response.content,
            }],
            usage: response.usage.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use dotenv::dotenv;
    use zqa_macros::{test_eq, test_ok};

    use super::{
        AnthropicClient, AnthropicRedactedThinkingResponseContent, AnthropicResponse,
        AnthropicResponseContent, AnthropicThinkingResponseContent,
        AnthropicToolUseResponseContent, AnthropicUsageStats, make_thinking_config,
        supports_adaptive_thinking,
    };
    use crate::config::AnthropicConfig;
    use crate::http_client::{MockHttpClient, RecordingSequentialMockHttpClient, ReqwestClient};
    use crate::llm::anthropic::{
        AnthropicOutputTokensDetails, AnthropicTextResponseContent, DEFAULT_CLAUDE_MODEL,
    };
    use crate::llm::base::{
        AgenticClient, ChatHistoryContent, ChatHistoryItem, ChatRequest, ContentType, MessageRole,
        ReasoningConfig, ToolCallResponse,
    };
    use crate::llm::tools::test_utils::MockTool;

    fn mock_response(
        id: &str,
        stop_reason: &str,
        input_tokens: u32,
        output_tokens: u32,
        content: Vec<AnthropicResponseContent>,
    ) -> AnthropicResponse {
        AnthropicResponse {
            id: id.into(),
            model: DEFAULT_CLAUDE_MODEL.into(),
            role: MessageRole::Assistant,
            stop_reason: stop_reason.into(),
            stop_sequence: None,
            usage: AnthropicUsageStats {
                input_tokens,
                output_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens_details: Some(AnthropicOutputTokensDetails { thinking_tokens: 0 }),
            },
            r#type: "message".into(),
            content,
        }
    }

    fn assert_thinking_config(request: &serde_json::Value) {
        test_eq!(request["thinking"]["type"].as_str(), Some("adaptive"));
        assert!(request["thinking"].get("budget_tokens").is_none());
    }

    #[test]
    fn test_supports_adaptive_thinking() {
        for model in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5",
            // Future and dated models work without changes to this function.
            "claude-opus-4-9-20260101",
            "claude-5-7-opus",
            "claude-haiku-5",
            "Claude-Opus-6",
            // Unrecognized names are assumed to support adaptive thinking.
            "claude-mythos-preview",
        ] {
            assert!(supports_adaptive_thinking(model), "{model}");
        }
        for model in [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-opus-4",
            "claude-sonnet-4",
            "claude-3-7-sonnet",
        ] {
            assert!(!supports_adaptive_thinking(model), "{model}");
        }
    }

    #[test]
    fn test_thinking_config_extended_only_model_uses_budget() {
        let reasoning = ReasoningConfig {
            max_tokens: Some(1024),
            effort: Some("high".into()),
            summary: None,
        };
        let (thinking, output_config) = make_thinking_config("claude-opus-4-5", Some(&reasoning));
        let thinking = serde_json::to_value(thinking).unwrap();
        test_eq!(thinking["type"].as_str(), Some("enabled"));
        test_eq!(thinking["budget_tokens"].as_u64(), Some(1024));
        test_eq!(thinking["display"].as_str(), Some("summarized"));
        assert!(output_config.is_none());
    }

    #[test]
    fn test_thinking_config_adaptive_model_uses_effort() {
        let reasoning = ReasoningConfig {
            max_tokens: Some(1024),
            effort: Some("low".into()),
            summary: None,
        };
        let (thinking, output_config) = make_thinking_config("claude-opus-4-8", Some(&reasoning));
        let thinking = serde_json::to_value(thinking).unwrap();
        test_eq!(thinking["type"].as_str(), Some("adaptive"));
        assert!(thinking.get("budget_tokens").is_none());
        let output_config = serde_json::to_value(output_config).unwrap();
        test_eq!(output_config["effort"].as_str(), Some("low"));
    }

    #[test]
    fn test_thinking_config_extended_only_model_derives_budget_from_effort() {
        let reasoning = ReasoningConfig {
            max_tokens: None,
            effort: Some("low".into()),
            summary: None,
        };
        let (thinking, output_config) = make_thinking_config("claude-opus-4-5", Some(&reasoning));
        let thinking = serde_json::to_value(thinking).unwrap();
        test_eq!(thinking["budget_tokens"].as_u64(), Some(4096));
        assert!(output_config.is_none());
    }

    #[test]
    fn test_thinking_config_adaptive_model_derives_effort_from_budget() {
        let reasoning = ReasoningConfig {
            max_tokens: Some(1024),
            effort: None,
            summary: None,
        };
        let (thinking, output_config) = make_thinking_config("claude-sonnet-5", Some(&reasoning));
        assert!(thinking.is_some());
        let output_config = serde_json::to_value(output_config).unwrap();
        test_eq!(output_config["effort"].as_str(), Some("minimal"));
    }

    #[test]
    fn test_thinking_config_adaptive_model_without_effort_omits_output_config() {
        let reasoning = ReasoningConfig {
            max_tokens: None,
            effort: None,
            summary: None,
        };
        let (thinking, output_config) = make_thinking_config("claude-sonnet-5", Some(&reasoning));
        assert!(thinking.is_some());
        assert!(output_config.is_none());

        let (thinking, output_config) = make_thinking_config("claude-sonnet-5", None);
        assert!(thinking.is_none());
        assert!(output_config.is_none());
    }

    fn content_block<'a>(
        content: &'a [serde_json::Value],
        content_type: &str,
    ) -> &'a serde_json::Value {
        content
            .iter()
            .find(|block| block["type"].as_str() == Some(content_type))
            .unwrap()
    }

    fn assert_thinking_continuation(request: &serde_json::Value) {
        assert_thinking_config(request);
        let messages = request["messages"].as_array().unwrap();
        let assistant_content = messages
            .iter()
            .find(|message| message["role"].as_str() == Some("assistant"))
            .unwrap()["content"]
            .as_array()
            .unwrap();
        let thinking = content_block(assistant_content, "thinking");
        assert_eq!(
            thinking["thinking"].as_str(),
            Some("I need the tool result.")
        );
        assert_eq!(thinking["signature"].as_str(), Some("thinking-signature"));
        assert_eq!(
            content_block(assistant_content, "redacted_thinking")["data"].as_str(),
            Some("redacted-thinking-data")
        );

        let tool_result = messages
            .iter()
            .filter_map(|message| message["content"].as_array())
            .flat_map(|content| content.iter())
            .find(|block| block["type"].as_str() == Some("tool_result"))
            .unwrap();
        assert_eq!(tool_result["content"].as_str(), Some("Hello, Alice!"));
    }

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = AnthropicClient::<ReqwestClient>::default();
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
    async fn test_request_with_mock() {
        // Load environment variables from .env file
        dotenv().ok();

        // Create a proper AnthropicResponse that matches the structure we expect to deserialize
        let mock_response = AnthropicResponse {
            id: "mock-id".to_string(),
            model: DEFAULT_CLAUDE_MODEL.to_string(),
            role: MessageRole::Assistant,
            stop_reason: "end_turn".to_string(),
            stop_sequence: None,
            usage: AnthropicUsageStats {
                input_tokens: 9,
                output_tokens: 13,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens_details: Some(AnthropicOutputTokensDetails { thinking_tokens: 0 }),
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
            system_prompt: None,
            reasoning: None,
            tools: None,
            on_tool_call: None,
            on_text: None,
            tool_iteration_limit: None,
        };

        let res = mock_client.send_message(&request).await;

        test_ok!(res);

        let res = res.unwrap();
        test_eq!(res.usage.input_tokens, 9);
        test_eq!(res.usage.output_tokens, 13);

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
            system_prompt: None,
            reasoning: None,
            tools: Some(&[Box::new(tool)]),
            on_tool_call: None,
            on_text: None,
            tool_iteration_limit: None,
        };
        let res = client.send_message(&request).await;

        test_ok!(res);
        assert!(call_count.lock().unwrap().eq(&1_usize));
    }

    #[tokio::test]
    async fn test_callbacks_fire() {
        dotenv().ok();

        let tool_call_response = AnthropicResponse {
            id: "msg-1".into(),
            model: DEFAULT_CLAUDE_MODEL.into(),
            role: MessageRole::Assistant,
            stop_reason: "tool_use".into(),
            stop_sequence: None,
            usage: AnthropicUsageStats {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens_details: Some(AnthropicOutputTokensDetails { thinking_tokens: 0 }),
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
            role: MessageRole::Assistant,
            stop_reason: "end_turn".into(),
            stop_sequence: None,
            usage: AnthropicUsageStats {
                input_tokens: 20,
                output_tokens: 8,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens_details: Some(AnthropicOutputTokensDetails { thinking_tokens: 0 }),
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
        let mock_client = AnthropicClient {
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
            test_eq!(request["system"], "Follow the system instructions.");
        }
    }

    #[test]
    fn test_thinking_blocks_round_trip() {
        let expected = serde_json::json!([
            {
                "type": "thinking",
                "thinking": "I need the tool result.",
                "signature": "thinking-signature"
            },
            {
                "type": "redacted_thinking",
                "data": "redacted-thinking-data"
            }
        ]);
        let content: Vec<AnthropicResponseContent> =
            serde_json::from_value(expected.clone()).unwrap();

        assert_eq!(serde_json::to_value(content).unwrap(), expected);
    }

    #[tokio::test]
    async fn test_thinking_blocks_are_replayed_after_tool_call() {
        let tool_call_response = mock_response(
            "msg-1",
            "tool_use",
            10,
            5,
            vec![
                AnthropicResponseContent::Thinking(AnthropicThinkingResponseContent {
                    r#type: "thinking".into(),
                    thinking: "I need the tool result.".into(),
                    signature: Some("thinking-signature".into()),
                }),
                AnthropicResponseContent::RedactedThinking(
                    AnthropicRedactedThinkingResponseContent {
                        r#type: "redacted_thinking".into(),
                        data: "redacted-thinking-data".into(),
                    },
                ),
                AnthropicResponseContent::ToolCall(AnthropicToolUseResponseContent {
                    id: "tool-1".into(),
                    r#type: "tool_use".into(),
                    name: "mock_tool".into(),
                    input: serde_json::json!({"name": "Alice"})
                        .as_object()
                        .unwrap()
                        .clone(),
                }),
            ],
        );
        let text_response = mock_response(
            "msg-2",
            "end_turn",
            20,
            8,
            vec![AnthropicResponseContent::Text(
                AnthropicTextResponseContent {
                    r#type: "text".into(),
                    text: "Done!".into(),
                },
            )],
        );
        let call_count = Arc::new(Mutex::new(0_usize));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };
        let http_client =
            RecordingSequentialMockHttpClient::new([tool_call_response, text_response]);
        let client = AnthropicClient {
            client: http_client.clone(),
            config: Some(AnthropicConfig {
                api_key: "test".into(),
                model: DEFAULT_CLAUDE_MODEL.into(),
                max_tokens: 2048,
                reasoning_budget: None,
                reasoning_effort: None,
            }),
        };
        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(2048),
            message: "Test".into(),
            system_prompt: None,
            reasoning: Some(ReasoningConfig {
                max_tokens: Some(1024),
                effort: None,
                summary: None,
            }),
            tools: Some(&[Box::new(tool)]),
            on_tool_call: None,
            on_text: None,
            tool_iteration_limit: None,
        };

        let response = client.send_message(&request).await;
        test_ok!(response);
        test_eq!(*call_count.lock().unwrap(), 1_usize);

        let requests = http_client.requests();
        assert_eq!(requests.len(), 2);
        assert_thinking_config(&requests[0]);
        assert_thinking_continuation(&requests[1]);
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
