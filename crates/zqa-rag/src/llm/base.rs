//! User-facing types for working with LLMs, including tool calling support. Most structs used by
//! the clients can be converted to/from the structs here.

use std::collections::HashMap;
use std::fmt::Display;
use std::sync::{Arc, LazyLock};

use http::HeaderMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use super::errors::LLMError;
use crate::constants::DEFAULT_MAX_TOOL_ITERATIONS;
use crate::events::AgentEvent;
use crate::http_client::HttpClient;
use crate::llm::tools::{CallbackFn, SerializedTool, Tool, get_owned_tools, process_tool_calls};
use crate::pricing::ModelUsage;
use crate::providers::ProviderId;
use crate::requests::request_with_backoff;

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
    pub result: Result<serde_json::Value, String>,
}

impl ToolCallResponse {
    /// The tool result as the text sent back to a provider.
    ///
    /// A string result or an error message is sent as-is, without JSON quotes, and any other result
    /// as compact JSON. Every provider uses this, so they all show the model the same text.
    pub(crate) fn content_text(&self) -> String {
        match &self.result {
            Ok(serde_json::Value::String(text)) | Err(text) => text.clone(),
            Ok(result) => result.to_string(),
        }
    }
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
    /// A user-visible reasoning summary (e.g. a thinking block). This remains distinct from answer
    /// text for presentation. Provider conversions may omit it because a summary does not contain
    /// the opaque metadata required to replay native reasoning blocks.
    Reasoning(String),
    /// A tool call request
    ToolCallRequest(ToolCallRequest),
    /// A tool call response
    ToolCallResponse(ToolCallResponse),
}

/// Wire API flavor that produced a [`NativePayload`].
///
/// Distinguishes endpoints within one provider and separates providers that share a
/// wire-compatible format (such as Anthropic and Ollama), so stored payloads are only replayed
/// natively by an adapter speaking the same API.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ApiFlavor {
    /// Anthropic Messages API (`POST /v1/messages`).
    #[serde(rename = "anthropic.messages")]
    AnthropicMessages,
    /// OpenAI Responses API (`POST /v1/responses`).
    #[serde(rename = "openai.responses")]
    OpenAIResponses,
    /// Gemini Interactions API (`POST /v1beta/interactions`).
    #[serde(rename = "gemini.interactions")]
    GeminiInteractions,
    /// OpenRouter Chat Completions API (`POST /api/v1/chat/completions`).
    #[serde(rename = "openrouter.chat_completions")]
    OpenRouterChatCompletions,
    /// Ollama Anthropic-compatible Messages API (`POST /v1/messages`).
    #[serde(rename = "ollama.anthropic_messages")]
    OllamaAnthropicMessages,
    /// Mock API flavor, used for testing (enabled by the `mock` feature).
    #[cfg(any(test, feature = "mock"))]
    #[serde(rename = "mock")]
    Mock,
}

impl ApiFlavor {
    /// Return the canonical wire identifier for this API flavor.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AnthropicMessages => "anthropic.messages",
            Self::OpenAIResponses => "openai.responses",
            Self::GeminiInteractions => "gemini.interactions",
            Self::OpenRouterChatCompletions => "openrouter.chat_completions",
            Self::OllamaAnthropicMessages => "ollama.anthropic_messages",
            #[cfg(any(test, feature = "mock"))]
            Self::Mock => "mock",
        }
    }
}

impl Display for ApiFlavor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Identifies the provider, wire API flavor, and serving model that produced a response.
///
/// `api` distinguishes endpoints within one provider (such as [`ApiFlavor::AnthropicMessages`] vs.
/// [`ApiFlavor::OllamaAnthropicMessages`], or [`ApiFlavor::GeminiInteractions`]), so stored
/// payloads are not misread if an adapter later switches APIs. `model` records the serving model
/// reported in the response when available.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    /// The provider that served the response.
    pub provider: ProviderId,
    /// The wire API flavor that served the response.
    pub api: ApiFlavor,
    /// The serving model identifier reported by the provider.
    pub model: String,
}

/// A deterministic `xxh3` digest of the provider-agnostic request frame (`system_prompt` and tool
/// specifications).
///
/// The digest hashes `(system_prompt, [{ name, description, parameters }])` without
/// provider-specific schema keys, so adapters can tell when the request prefix changed (for
/// example, to strip Anthropic thinking blocks bound to an older frame).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrameHash(pub String);

impl FrameHash {
    /// Compute the canonical `xxh3` hash for `system_prompt` and `tools`.
    ///
    /// Both `None` and `Some(&[])` for `tools` hash identically since neither sends tool
    /// declarations. JSON object keys inside tool parameter schemas are sorted before hashing so
    /// the digest is independent of map iteration order.
    ///
    /// # Arguments
    ///
    /// * `system_prompt` - The system prompt active for the request, if any.
    /// * `tools` - The tools exposed to the model, if any.
    ///
    /// # Returns
    ///
    /// A 32-character lowercase hexadecimal `xxh3_128` digest.
    #[must_use]
    pub(crate) fn compute(
        system_prompt: Option<&str>,
        tools: Option<&[SerializedTool<'_>]>,
    ) -> Self {
        let tool_specs: Vec<serde_json::Value> = tools
            .unwrap_or_default()
            .iter()
            .map(|serialized| {
                serde_json::json!({
                    "name": serialized.tool.name(),
                    "description": serialized.tool.description(),
                    "parameters": serde_json::Value::from(serialized.tool.parameters()),
                })
            })
            .collect();
        let canonical = serde_json::json!({
            "system_prompt": system_prompt,
            "tools": tool_specs,
        });

        let mut hasher = Xxh3::new();
        hash_canonical_json(&mut hasher, &canonical);

        Self(format!("{:032x}", hasher.digest128()))
    }
}

/// Feed a `serde_json::Value` into `hasher` with object keys sorted lexicographically.
fn hash_canonical_json(hasher: &mut Xxh3, value: &serde_json::Value) {
    match value {
        serde_json::Value::Null => hasher.update(b"null"),
        serde_json::Value::Bool(b) => hasher.update(if *b { b"true" } else { b"false" }),
        serde_json::Value::Number(n) => hasher.update(n.to_string().as_bytes()),
        serde_json::Value::String(s) => {
            hasher.update(serde_json::to_string(s).unwrap_or_default().as_bytes());
        }
        serde_json::Value::Array(items) => {
            hasher.update(b"[");

            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    hasher.update(b",");
                }
                hash_canonical_json(hasher, item);
            }

            hasher.update(b"]");
        }
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&str, &serde_json::Value)> =
                map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            entries.sort_unstable_by_key(|(k, _)| *k);

            hasher.update(b"{");

            for (idx, (k, v)) in entries.into_iter().enumerate() {
                if idx > 0 {
                    hasher.update(b",");
                }
                hasher.update(serde_json::to_string(k).unwrap_or_default().as_bytes());
                hasher.update(b":");
                hash_canonical_json(hasher, v);
            }

            hasher.update(b"}");
        }
    }
}

/// Provider-native response captured verbatim alongside canonical [`ChatHistoryContent`].
///
/// Storing the raw response JSON lets each adapter replay provider-opaque state (such as Anthropic
/// thinking signatures and redacted thinking, OpenAI encrypted reasoning, Gemini thought and
/// function-call signatures, and OpenRouter `reasoning_details`) across turns and disk reloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NativePayload {
    /// Provider, API flavor, and serving model that produced `response`.
    pub origin: ModelRef,
    /// Digest of the request frame (`system_prompt` and tools) active when `response` was produced.
    pub frame: FrameHash,
    /// The response body exactly as returned by the provider.
    pub response: serde_json::Value,
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
    /// Provider-native response payload for assistant turns, when available. Absent on user and
    /// tool-result items and on conversations saved before native capture existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<NativePayload>,
}

/// Provider-agnostic reasoning configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Token budget for thinking.
    pub max_tokens: Option<u32>,
    /// Reasoning effort for providers that support it.
    pub effort: Option<String>,
    /// Thinking summary length, for providers that support it.
    pub summary: Option<String>,
}

/// Reasoning effort levels
#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)]
pub enum ReasoningEffort {
    r#None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReasoningEffort::None => f.write_str("none"),
            ReasoningEffort::Minimal => f.write_str("minimal"),
            ReasoningEffort::Low => f.write_str("low"),
            ReasoningEffort::Medium => f.write_str("medium"),
            ReasoningEffort::High => f.write_str("high"),
            ReasoningEffort::Xhigh => f.write_str("xhigh"),
            ReasoningEffort::Max => f.write_str("max"),
        }
    }
}

impl TryFrom<String> for ReasoningEffort {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "none" => Ok(ReasoningEffort::None),
            "minimal" => Ok(ReasoningEffort::Minimal),
            "low" => Ok(ReasoningEffort::Low),
            "medium" => Ok(ReasoningEffort::Medium),
            "high" => Ok(ReasoningEffort::High),
            "xhigh" => Ok(ReasoningEffort::Xhigh),
            "max" => Ok(ReasoningEffort::Max),
            val => Err(format!(
                "invalid 'effort' parameter {val}, expected one of {}",
                "'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'"
            )),
        }
    }
}

pub(crate) static EFFORT_TO_TOKENS: LazyLock<HashMap<ReasoningEffort, u32>> = LazyLock::new(|| {
    let mut map = HashMap::new();

    map.insert(ReasoningEffort::None, 0);
    map.insert(ReasoningEffort::Minimal, 2048);
    map.insert(ReasoningEffort::Low, 4096);
    map.insert(ReasoningEffort::Medium, 10_000);
    map.insert(ReasoningEffort::High, 16_000);
    map.insert(ReasoningEffort::Xhigh, 32_000);
    map.insert(ReasoningEffort::Max, 32_000);

    map
});

pub(crate) static TOKENS_TO_EFFORT: [(u32, ReasoningEffort); 7] = [
    (0_u32, ReasoningEffort::None),
    (2048_u32, ReasoningEffort::Minimal),
    (4096_u32, ReasoningEffort::Low),
    (10_000_u32, ReasoningEffort::Medium),
    (16_000_u32, ReasoningEffort::High),
    (32_000_u32, ReasoningEffort::Xhigh),
    (32_000_u32, ReasoningEffort::Max),
];

impl ReasoningConfig {
    /// Normalize reasoning "effort" and token counts. If only one of the two is specified, then the
    /// other is automatically filled in based on [Inspect's mapping](https://inspect.aisi.org.uk/reasoning.html#anthropic-claude-3.7-4.0-4.1-4.5).
    /// In other cases, no changes are made.
    pub(crate) fn normalize(&mut self) {
        match (self.max_tokens, self.effort.as_deref()) {
            (Some(token_budget), None) => {
                self.effort = Some(
                    TOKENS_TO_EFFORT
                        .iter()
                        .find(|tup| tup.0 >= token_budget)
                        .map_or(ReasoningEffort::Max, |tup| tup.1)
                        .to_string(),
                );
            }
            (None, Some("none")) => self.effort = None,
            (None, Some(effort)) => {
                self.max_tokens = ReasoningEffort::try_from(effort.to_string())
                    .ok()
                    .map(|e| *EFFORT_TO_TOKENS.get(&e).unwrap());
            }
            _ => {}
        }
    }
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
    /// Instructions that guide the model for the entire request
    pub system_prompt: Option<String>,
    /// Reasoning config. Optional.
    pub reasoning: Option<ReasoningConfig>,
    /// The tools to use
    pub tools: Option<&'a [Box<dyn Tool>]>,

    // Driver options: these must be `Option<T>`; semantically, it doesn't make sense to force
    // end-users into supplying these options.
    /// Optional callback invoked for any event, such as model text, tool calls, or responses. For
    /// details, see [`crate::events::AgentEvent`].
    pub on_event: Option<Arc<CallbackFn<AgentEvent>>>,
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
    /// The result of the tool call, or an error message, such as if the tool name was hallucinated.
    pub tool_result: Result<serde_json::Value, String>,
}

/// A model response can contain multiple types of content, such as raw text and tool calls. This
/// captures those variants. Specifically, this is different from `ChatHistoryContent`, which is
/// more tailored to individual items in the chat history that is passed to providers.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ContentType {
    /// A plain-text message
    Text(String),
    /// A reasoning summary (e.g. a thinking block)
    Reasoning(String),
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
    /// The user-facing content accumulated across all provider turns. Use
    /// [`Self::history_additions`] when provider turn boundaries matter.
    pub content: Vec<ContentType>,
    /// The provider-agnostic history items produced while handling this request. This excludes the
    /// history and user message supplied in [`ChatRequest`].
    #[serde(default)]
    pub history_additions: Vec<ChatHistoryItem>,
    /// Token usage of each provider request made while handling this request, in order. Keep these
    /// separate when estimating costs, since prompt-size pricing tiers apply per provider request;
    /// see [`crate::pricing::ModelPricing::estimate_cost`]. Use [`Self::total_usage`] for totals.
    pub usage: Vec<ModelUsage>,
}

impl CompletionApiResponse {
    /// Sum the token usage of every provider request made for this response.
    ///
    /// # Returns
    ///
    /// The total token usage.
    #[must_use]
    pub fn total_usage(&self) -> ModelUsage {
        self.usage.iter().sum()
    }

    /// Join the response's text parts with newlines, leaving out reasoning and tool calls.
    ///
    /// Use this when the response is data, such as a title or a sub-agent's answer that goes back
    /// to another model, rather than something shown to a user.
    ///
    /// # Returns
    ///
    /// The text parts in order, or an empty string if the response has none.
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentType::Text(text) => Some(text.as_str()),
                ContentType::Reasoning(_) | ContentType::ToolCall(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Raw provider response and origin captured by one [`AgenticClient::send_once`] call.
pub(crate) struct RawProviderResponse {
    /// Provider, API flavor, and serving model reported by the response.
    pub(crate) origin: ModelRef,
    /// The exact JSON response body returned by the provider.
    pub(crate) body: serde_json::Value,
}

/// An abstraction over one model turn.
pub(crate) struct ProviderTurn {
    /// Raw response body and origin metadata, attached to the assistant [`ChatHistoryItem`] as
    /// [`NativePayload`]. Every real provider populates this; only mock loop tests omit it.
    pub(crate) raw: Option<RawProviderResponse>,
    /// Provider-agnostic view of the same output, for tool dispatch and final-text extraction.
    pub(crate) contents: Vec<ChatHistoryContent>,
    /// Token usage of each provider request in this turn. Usually one, but retried requests add
    /// more.
    pub(crate) usage: Vec<ModelUsage>,
}

impl ProviderTurn {
    /// Split this turn into its recorded usage and an assistant [`ChatHistoryItem`] stamped with
    /// `frame`, or `None` if neither canonical contents nor a raw response were produced.
    fn into_history_and_usage(
        self,
        frame: &FrameHash,
    ) -> (Option<ChatHistoryItem>, Vec<ModelUsage>) {
        let native = self.raw.map(|raw| NativePayload {
            origin: raw.origin,
            frame: frame.clone(),
            response: raw.body,
        });
        let item = (!self.contents.is_empty() || native.is_some()).then_some(ChatHistoryItem {
            role: MessageRole::Assistant,
            content: self.contents,
            native,
        });

        (item, self.usage)
    }
}

/// Internal contract for provider-specific generation adapters.
pub(crate) trait AgenticClient {
    type HistoryItem;
    const SCHEMA_KEY: &'static str;
    const API_FLAVOR: ApiFlavor;

    /// Project one history item into provider-native request items for `current_frame`.
    ///
    /// Both [`Self::build_initial_history`] and the in-loop continuation inside
    /// [`Self::send_message`] call this method, so in-loop replay and post-reload replay produce
    /// identical request items.
    fn project_history_item(
        &self,
        item: &ChatHistoryItem,
        current_frame: &FrameHash,
    ) -> Vec<Self::HistoryItem>;

    /// Build the initial conversation history in the provider-native format given a
    /// [`ChatRequest`] and the active `current_frame`.
    fn build_initial_history(
        &self,
        request: &ChatRequest<'_>,
        current_frame: &FrameHash,
    ) -> Vec<Self::HistoryItem>;

    /// Perform one provider request-response round trip and convert the response into raw and
    /// provider-agnostic history data.
    ///
    /// `tools` is the same on every turn of a tool loop; `allow_tool_calls` is `false` on the last
    /// turn. Providers that can forbid tool calls (e.g. a `tool_choice` of "none") should keep
    /// sending `tools` then, since changing them mid-loop invalidates the prompt cache and, on
    /// Claude, any preserved thinking blocks after the change. Other providers drop the tools.
    /// When the first turn is also the last (an iteration limit of 1), `tools` is `None`.
    async fn send_once(
        &self,
        history: &[Self::HistoryItem],
        system_prompt: Option<&str>,
        tools: Option<&[SerializedTool<'_>]>,
        allow_tool_calls: bool,
        reasoning: Option<&ReasoningConfig>,
        max_tokens: Option<u32>,
    ) -> Result<ProviderTurn, LLMError>;

    /// Send a chat request, executing requested tools until the provider returns a final response.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat request to send.
    ///
    /// # Returns
    ///
    /// The flattened response content, turn-preserving history additions, and accumulated usage
    /// across all provider turns.
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
        let tools = get_owned_tools(request.tools, Self::SCHEMA_KEY);
        let iteration_limit = request
            .tool_iteration_limit
            .unwrap_or(DEFAULT_MAX_TOOL_ITERATIONS);
        let initial_tools = tools.as_deref().filter(|_| iteration_limit > 1);
        let frame_hash = FrameHash::compute(request.system_prompt.as_deref(), initial_tools);
        let mut provider_history = self.build_initial_history(request, &frame_hash);
        let mut usage = Vec::new();
        let mut contents = Vec::<ContentType>::new();
        let mut history_additions = Vec::new();

        let mut round_trips = 0;
        while round_trips < iteration_limit {
            // On the last trip, disallow tool calls
            let is_last_turn = round_trips == iteration_limit.saturating_sub(1);

            // Keep sending the tools after an earlier tool round so the request prefix (and any
            // replayed thinking) stays unchanged; a single turn that cannot call tools omits them.
            let tools_passed = tools
                .as_deref()
                .filter(|_| !is_last_turn || round_trips > 0);

            let turn_start = std::time::Instant::now();
            log::debug!(
                "Generation turn {} of {iteration_limit}: history_items={}, tools_allowed={}",
                round_trips + 1,
                provider_history.len(),
                tools_passed.is_some() && !is_last_turn
            );
            let turn = self
                .send_once(
                    &provider_history,
                    request.system_prompt.as_deref(),
                    tools_passed,
                    !is_last_turn,
                    request.reasoning.as_ref(),
                    request.max_tokens,
                )
                .await
                .inspect_err(|error| {
                    log::debug!(
                        "Generation turn {} failed after {:.2?}: {}",
                        round_trips + 1,
                        turn_start.elapsed(),
                        crate::logging::preview(error)
                    );
                })?;
            log::debug!(
                "Generation turn {} completed in {:.2?}: content_items={}, usage={:?}",
                round_trips + 1,
                turn_start.elapsed(),
                turn.contents.len(),
                turn.usage
            );
            let (assistant_item, turn_usage) = turn.into_history_and_usage(&frame_hash);
            usage.extend(turn_usage);
            let turn_contents = assistant_item
                .as_ref()
                .map_or(&[][..], |item| item.content.as_slice());

            if let Some(item) = &assistant_item {
                provider_history.extend(self.project_history_item(item, &frame_hash));
            }

            let tool_call_results = process_tool_calls(
                &mut contents,
                turn_contents,
                if is_last_turn {
                    &[]
                } else {
                    tools.as_deref().unwrap_or_default()
                },
                request.on_event.as_ref(),
            )
            .await;

            if let Some(item) = assistant_item {
                history_additions.push(item);
            }

            let Some(tool_call_results) = tool_call_results else {
                log::debug!(
                    "Generation stopped: no tool calls on turn {}",
                    round_trips + 1
                );
                break;
            };

            if iteration_limit > 1 && round_trips == iteration_limit - 2 {
                // On the next turn, tool calls will be disallowed, so we should log this.
                log::warn!(
                    "Reached penultimate iteration ({round_trips}), last trip will not allow tool calling."
                );
            }

            provider_history.extend(self.project_history_item(&tool_call_results, &frame_hash));
            history_additions.push(tool_call_results);
            round_trips += 1;
        }

        if round_trips == iteration_limit {
            log::debug!("Generation stopped: iteration limit {iteration_limit} reached");
        }
        log::debug!(
            "Generation finished: history_additions={}, usage={usage:?}",
            history_additions.len()
        );

        Ok(CompletionApiResponse {
            content: contents,
            history_additions,
            usage,
        })
    }
}

pub(crate) async fn send_generation_request<R, S>(
    client: &impl HttpClient,
    request: R,
    headers: &HeaderMap,
    api_url: &str,
    max_retries: usize,
) -> Result<(S, serde_json::Value), LLMError>
where
    R: Serialize + Send + Sync,
    S: DeserializeOwned,
{
    let res = request_with_backoff(client, api_url, headers, &request, max_retries).await?;

    let body = res.text().await?;
    let json: serde_json::Value = serde_json::from_str(&body).map_err(|err| {
        log::error!(
            "Failed to parse response body as JSON: {}. Body: {}",
            crate::logging::preview(&err),
            crate::logging::preview(&body)
        );
        LLMError::DeserializationError(body.clone())
    })?;
    let response: S = serde_json::from_value(json.clone()).map_err(|err| {
        log::error!(
            "Failed to deserialize response into target type: {}. Body: {}",
            crate::logging::preview(&err),
            crate::logging::preview(&body)
        );
        LLMError::DeserializationError(body)
    })?;

    Ok((response, json))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::{Future, ready};
    use std::sync::{Arc, Mutex};

    use zqa_macros::test_eq;

    use super::*;
    use crate::events::AgentEvent;
    use crate::llm::tools::test_utils::MockTool;

    struct TestHistoryItem;

    /// The number of tools sent on a turn, and whether tool calls were allowed.
    type ToolsSeen = (Option<usize>, bool);

    struct TestClient {
        turns: Mutex<VecDeque<ProviderTurn>>,
        system_prompts_seen: Arc<Mutex<Vec<Option<String>>>>,
        tools_seen: Arc<Mutex<Vec<ToolsSeen>>>,
    }

    impl AgenticClient for TestClient {
        type HistoryItem = TestHistoryItem;
        const SCHEMA_KEY: &'static str = "parameters";
        const API_FLAVOR: ApiFlavor = ApiFlavor::Mock;

        fn project_history_item(
            &self,
            _: &ChatHistoryItem,
            _: &FrameHash,
        ) -> Vec<Self::HistoryItem> {
            Vec::new()
        }

        fn build_initial_history(
            &self,
            _: &ChatRequest<'_>,
            _: &FrameHash,
        ) -> Vec<Self::HistoryItem> {
            Vec::new()
        }

        fn send_once(
            &self,
            _: &[Self::HistoryItem],
            system_prompt: Option<&str>,
            tools: Option<&[SerializedTool<'_>]>,
            allow_tool_calls: bool,
            _: Option<&ReasoningConfig>,
            _: Option<u32>,
        ) -> impl Future<Output = Result<ProviderTurn, LLMError>> {
            self.system_prompts_seen
                .lock()
                .unwrap()
                .push(system_prompt.map(ToOwned::to_owned));
            self.tools_seen
                .lock()
                .unwrap()
                .push((tools.map(<[_]>::len), allow_tool_calls));
            ready(Ok(self.turns.lock().unwrap().pop_front().unwrap()))
        }
    }

    fn tool_call_turn(calls: &[(&str, &str)], usage: ModelUsage) -> ProviderTurn {
        ProviderTurn {
            raw: None,
            contents: calls
                .iter()
                .map(|&(id, name)| tool_call_request(id, name))
                .collect(),
            usage: vec![usage],
        }
    }

    fn tool_call_request(id: &str, name: &str) -> ChatHistoryContent {
        ChatHistoryContent::ToolCallRequest(ToolCallRequest {
            id: id.into(),
            tool_name: "mock_tool".into(),
            args: serde_json::json!({"name": name}),
        })
    }

    fn expected_history_additions() -> Vec<ChatHistoryItem> {
        vec![
            ChatHistoryItem {
                role: MessageRole::Assistant,
                content: vec![
                    ChatHistoryContent::Reasoning("Need information from both tools.".into()),
                    ChatHistoryContent::Text("I will check both sources.".into()),
                    ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                        id: "call-1".into(),
                        tool_name: "mock_tool".into(),
                        args: serde_json::json!({"name": "Alice"}),
                    }),
                    ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                        id: "call-2".into(),
                        tool_name: "mock_tool".into(),
                        args: serde_json::json!({"name": "Bob"}),
                    }),
                ],
                native: None,
            },
            ChatHistoryItem {
                role: MessageRole::User,
                content: vec![
                    ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                        id: "call-1".into(),
                        tool_name: "mock_tool".into(),
                        result: Ok(serde_json::json!("Hello, Alice!")),
                    }),
                    ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                        id: "call-2".into(),
                        tool_name: "mock_tool".into(),
                        result: Ok(serde_json::json!("Hello, Bob!")),
                    }),
                ],
                native: None,
            },
            ChatHistoryItem {
                role: MessageRole::Assistant,
                content: vec![ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                    id: "call-3".into(),
                    tool_name: "mock_tool".into(),
                    args: serde_json::json!({"name": "Carol"}),
                })],
                native: None,
            },
            ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                    id: "call-3".into(),
                    tool_name: "mock_tool".into(),
                    result: Ok(serde_json::json!("Hello, Carol!")),
                })],
                native: None,
            },
            ChatHistoryItem {
                role: MessageRole::Assistant,
                content: vec![ChatHistoryContent::Text("done".into())],
                native: None,
            },
        ]
    }

    #[tokio::test]
    async fn send_message_preserves_tool_turns_and_accumulates_usage() {
        let tools_seen = Arc::new(Mutex::new(Vec::new()));
        let first_turn = ProviderTurn {
            raw: None,
            contents: vec![
                ChatHistoryContent::Reasoning("Need information from both tools.".into()),
                ChatHistoryContent::Text("I will check both sources.".into()),
                tool_call_request("call-1", "Alice"),
                tool_call_request("call-2", "Bob"),
            ],
            usage: vec![ModelUsage {
                input_tokens: 10,
                input_cache_written: 1,
                input_cache_read: 2,
                output_tokens: 3,
                reasoning_tokens: 4,
            }],
        };
        let client = TestClient {
            turns: Mutex::new(VecDeque::from([
                first_turn,
                tool_call_turn(&[("call-3", "Carol")], ModelUsage::default()),
                ProviderTurn {
                    raw: None,
                    contents: vec![ChatHistoryContent::Text("done".into())],
                    usage: vec![ModelUsage {
                        input_tokens: 20,
                        input_cache_written: 5,
                        input_cache_read: 6,
                        output_tokens: 7,
                        reasoning_tokens: 8,
                    }],
                },
            ])),
            system_prompts_seen: Arc::new(Mutex::new(Vec::new())),
            tools_seen: Arc::clone(&tools_seen),
        };
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };
        let request = ChatRequest {
            system_prompt: Some("Follow the system instructions.".into()),
            tools: Some(&[Box::new(tool)]),
            tool_iteration_limit: Some(3),
            ..ChatRequest::default()
        };

        let response = client.send_message(&request).await.unwrap();

        assert_eq!(
            *tools_seen.lock().unwrap(),
            vec![(Some(1), true), (Some(1), true), (Some(1), false)]
        );
        assert_eq!(
            *client.system_prompts_seen.lock().unwrap(),
            vec![
                Some("Follow the system instructions.".into()),
                Some("Follow the system instructions.".into()),
                Some("Follow the system instructions.".into())
            ]
        );
        assert_eq!(*call_count.lock().unwrap(), 3);
        assert_eq!(response.usage.len(), 3);
        let usage = response.total_usage();
        assert_eq!(usage.input_tokens, 30);
        assert_eq!(usage.input_cache_written, 6);
        assert_eq!(usage.input_cache_read, 8);
        assert_eq!(usage.output_tokens, 10);
        assert_eq!(usage.reasoning_tokens, 12);
        assert!(matches!(
            response.content.as_slice(),
            [
                ContentType::Reasoning(reasoning),
                ContentType::Text(progress),
                ContentType::ToolCall(_),
                ContentType::ToolCall(_),
                ContentType::ToolCall(_),
                ContentType::Text(text)
            ] if reasoning == "Need information from both tools."
                && progress == "I will check both sources."
                && text == "done"
        ));
        assert_eq!(response.history_additions, expected_history_additions());
    }

    /// Reasoning, text, and tool callbacks follow provider content order across turns.
    #[tokio::test]
    async fn send_message_delivers_reasoning_in_callback_order() {
        let client = TestClient {
            turns: Mutex::new(VecDeque::from([
                ProviderTurn {
                    raw: None,
                    contents: vec![
                        ChatHistoryContent::Reasoning("Check the source.".into()),
                        ChatHistoryContent::Text("Looking it up.".into()),
                        tool_call_request("call-1", "Alice"),
                    ],
                    usage: vec![ModelUsage::default()],
                },
                ProviderTurn {
                    raw: None,
                    contents: vec![
                        ChatHistoryContent::Reasoning("The source agrees.".into()),
                        ChatHistoryContent::Text("Done.".into()),
                    ],
                    usage: vec![ModelUsage::default()],
                },
            ])),
            system_prompts_seen: Arc::default(),
            tools_seen: Arc::default(),
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&events);
        let request = ChatRequest {
            tools: Some(&[Box::new(MockTool {
                call_count: Arc::default(),
            })]),
            on_event: Some(Arc::new(move |event| {
                let line = match event {
                    AgentEvent::Text(text) => format!("text: {text}"),
                    AgentEvent::Reasoning(text) => format!("reasoning: {text}"),
                    AgentEvent::ToolCall { name, .. } => format!("tool call: {name}"),
                    AgentEvent::ToolResponse { name, .. } => format!("tool response: {name}"),
                };
                recorded.lock().unwrap().push(line);
            })),
            ..ChatRequest::default()
        };

        let response = client.send_message(&request).await.unwrap();

        assert_eq!(
            *events.lock().unwrap(),
            [
                "reasoning: Check the source.",
                "text: Looking it up.",
                "tool call: mock_tool",
                "tool response: mock_tool",
                "reasoning: The source agrees.",
                "text: Done.",
            ]
        );
        assert_eq!(response.text(), "Looking it up.\nDone.");
    }

    /// Text-only responses must not send reasoning events.
    #[tokio::test]
    async fn send_message_without_reasoning_only_sends_text_events() {
        let client = TestClient {
            turns: Mutex::new(VecDeque::from([ProviderTurn {
                raw: None,
                contents: vec![ChatHistoryContent::Text("done".into())],
                usage: vec![ModelUsage::default()],
            }])),
            system_prompts_seen: Arc::default(),
            tools_seen: Arc::default(),
        };
        let text_segments = Arc::new(Mutex::new(Vec::new()));
        let request = ChatRequest {
            on_event: Some(Arc::new({
                let text_segments = Arc::clone(&text_segments);
                move |event| match event {
                    AgentEvent::Text(text) => text_segments.lock().unwrap().push(text.clone()),
                    AgentEvent::Reasoning(_) => panic!("unexpected reasoning event"),
                    _ => {}
                }
            })),
            ..ChatRequest::default()
        };

        let response = client.send_message(&request).await.unwrap();

        assert_eq!(*text_segments.lock().unwrap(), ["done"]);
        assert_eq!(response.content, [ContentType::Text("done".into())]);
    }

    #[test]
    fn completion_response_defaults_missing_history_additions() {
        let mut serialized = serde_json::to_value(CompletionApiResponse::default()).unwrap();
        serialized
            .as_object_mut()
            .unwrap()
            .remove("history_additions");

        let response: CompletionApiResponse = serde_json::from_value(serialized).unwrap();

        assert_eq!(response.history_additions, [] as [ChatHistoryItem; 0]);
    }

    #[tokio::test]
    async fn tool_iteration_limit_of_one_does_not_underflow() {
        let tools_seen = Arc::new(Mutex::new(Vec::new()));
        let client = TestClient {
            turns: Mutex::new(VecDeque::from([ProviderTurn {
                raw: None,
                contents: vec![ChatHistoryContent::Text("done".into())],
                usage: vec![ModelUsage::default()],
            }])),
            system_prompts_seen: Arc::new(Mutex::new(Vec::new())),
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

        assert_eq!(*tools_seen.lock().unwrap(), vec![(None, false)]);
        assert_eq!(response.content.len(), 1);
    }

    #[test]
    fn test_all_reasoning_effort_values_mapped() {
        for effort in [
            ReasoningEffort::None,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max,
        ] {
            assert!(
                EFFORT_TO_TOKENS.contains_key(&effort),
                "{effort:?} missing from EFFORT_TO_TOKENS"
            );
        }

        // Exhaustiveness guard: a new `ReasoningEffort` variant fails to compile
        // here until it is added to the array above as well.
        match ReasoningEffort::None {
            ReasoningEffort::None
            | ReasoningEffort::Minimal
            | ReasoningEffort::Low
            | ReasoningEffort::Medium
            | ReasoningEffort::High
            | ReasoningEffort::Xhigh
            | ReasoningEffort::Max => {}
        }
    }

    #[test]
    fn test_effort_token_maps_have_same_length() {
        test_eq!(EFFORT_TO_TOKENS.len(), TOKENS_TO_EFFORT.len());
    }

    #[test]
    fn frame_hash_is_provider_agnostic_and_tracks_changes() {
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            call_count: Arc::default(),
        })];
        let anthropic_tools = get_owned_tools(Some(&tools), "input_schema");
        let openai_tools = get_owned_tools(Some(&tools), "parameters");

        let a = FrameHash::compute(Some("Be concise."), anthropic_tools.as_deref());
        let b = FrameHash::compute(Some("Be concise."), openai_tools.as_deref());
        let different_prompt = FrameHash::compute(Some("Be verbose."), openai_tools.as_deref());
        let no_tools = FrameHash::compute(Some("Be concise."), None);
        let empty_tools = FrameHash::compute(Some("Be concise."), Some(&[]));

        test_eq!(a, b);
        test_eq!(no_tools, empty_tools);
        assert_ne!(a, different_prompt);
        assert_ne!(a, no_tools);
    }

    #[test]
    fn chat_history_item_defaults_missing_native_and_omits_none() {
        let legacy = serde_json::json!({
            "role": "assistant",
            "content": [{ "Text": "Hello" }]
        });
        let parsed: ChatHistoryItem = serde_json::from_value(legacy.clone()).unwrap();

        assert!(parsed.native.is_none());
        test_eq!(serde_json::to_value(parsed).unwrap(), legacy);
    }
}
