//! Functions, structs, and trait implementations for interacting with the Gemini API. This module
//! includes support for both text generation and embedding, and tool calling is supported. Text
//! generation uses the stateless mode of the Interactions API: see
//! <https://ai.google.dev/gemini-api/docs/interactions>.

use std::env;

use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use super::base::ChatRequest;
use super::errors::LLMError;
use crate::clients::gemini::{GeminiClient, get_gemini_api_key};
use crate::constants::{DEFAULT_GEMINI_MODEL, DEFAULT_MAX_RETRIES};
use crate::http_client::HttpClient;
use crate::llm::base::{
    AgenticClient, ChatHistoryContent, ChatHistoryItem, MessageRole, ProviderTurn, ReasoningConfig,
    ToolCallRequest, send_generation_request,
};
use crate::llm::tools::{GEMINI_SCHEMA_KEY, SerializedTool};
use crate::pricing::ModelUsage;
use crate::requests::exponential_backoff_delay;

/// A content item inside a step. Only text is requested from or sent to the API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum GeminiContentItem {
    Text {
        text: String,
    },
    /// Content types we do not request, such as images. These are dropped.
    #[serde(other)]
    Unsupported,
}

/// One step of an interaction. The conversation history is sent as a list of steps, and the
/// response returns the steps the model generated, which must be replayed unchanged (including
/// `thought` signatures) on later turns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum GeminiStep {
    UserInput {
        content: Vec<GeminiContentItem>,
    },
    ModelOutput {
        #[serde(default)]
        content: Vec<GeminiContentItem>,
    },
    Thought {
        /// Opaque representation of the model's reasoning state.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        /// Thought summaries, present when `thinking_summaries` is enabled.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        summary: Vec<GeminiContentItem>,
    },
    FunctionCall {
        /// A unique ID for the function call
        id: String,
        /// The name of the tool (function) to call
        name: String,
        /// The function parameters
        arguments: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    FunctionResult {
        /// The ID of the corresponding function call
        call_id: String,
        /// The name of the function
        name: String,
        /// The function response in JSON format
        result: serde_json::Value,
    },
    /// Step types we do not request, such as built-in tool calls. These are dropped.
    #[serde(other)]
    Unsupported,
}

impl From<ChatHistoryItem> for Vec<GeminiStep> {
    fn from(value: ChatHistoryItem) -> Self {
        value
            .content
            .into_iter()
            .filter_map(|c| match c {
                ChatHistoryContent::Text(text) => {
                    let content = vec![GeminiContentItem::Text { text }];
                    Some(match value.role {
                        MessageRole::User | MessageRole::Tool => GeminiStep::UserInput { content },
                        MessageRole::Assistant => GeminiStep::ModelOutput { content },
                    })
                }
                ChatHistoryContent::Reasoning(_) => None,
                ChatHistoryContent::ToolCallRequest(tool_call) => Some(GeminiStep::FunctionCall {
                    id: tool_call.id,
                    name: tool_call.tool_name,
                    arguments: tool_call.args,
                    signature: None,
                }),
                ChatHistoryContent::ToolCallResponse(tool_res) => {
                    // Wrap the result in an object with a "result" field if it's not already an object
                    let result = if tool_res.result.is_object() {
                        tool_res.result
                    } else {
                        serde_json::json!({ "result": tool_res.result })
                    };

                    Some(GeminiStep::FunctionResult {
                        call_id: tool_res.id,
                        name: tool_res.tool_name,
                        result,
                    })
                }
            })
            .collect()
    }
}

/// Map a reasoning effort to a Gemini `thinking_level` (`low`, `medium`, or `high`). `xhigh` and
/// `max` are mapped down to `high`, since Google's highest documented level is `high`, and
/// `minimal` is mapped up to `low`, since only some Flash models support it. The Interactions API
/// has no token budget, so a missing or unrecognized effort maps to `None`, which uses the model's
/// default level.
///
/// # Arguments
///
/// * `reasoning` - The provider-neutral reasoning config.
///
/// # Returns
///
/// The Gemini thinking level, or `None` to use the model's default.
fn gemini_thinking_level(reasoning: &ReasoningConfig) -> Option<&'static str> {
    match reasoning.effort.as_deref() {
        Some("none") => {
            log::warn!(
                "Gemini does not consistently support disabled thinking; using low effort instead."
            );
            Some("low")
        }
        Some("minimal" | "low") => Some("low"),
        Some("medium") => Some("medium"),
        Some("high" | "xhigh" | "max") => Some("high"),
        _ => None,
    }
}

/// Optional text generation configuration
#[derive(Serialize, Clone)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<&'static str>,
    /// `"auto"` to return thought summaries, which are only requested when reasoning is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_summaries: Option<&'static str>,
    /// `"none"` on the last turn of a tool loop, so the tools stay in the request (keeping the
    /// prompt cache) but cannot be called.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

/// A function declaration in the format the Interactions API expects.
#[derive(Serialize, Clone)]
struct GeminiTool<'a> {
    r#type: &'static str,
    #[serde(flatten)]
    tool: SerializedTool<'a>,
}

/// The request body for text generation
#[derive(Serialize, Clone)]
struct GeminiRequestBody<'a> {
    model: &'a str,
    input: &'a [GeminiStep],
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<&'a str>,
    generation_config: GeminiGenerationConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool<'a>>>,
    /// The full history is sent on every turn, so the interaction is not stored server-side.
    store: bool,
}

/// Usage received from the Gemini Interactions API.
#[derive(Default, Serialize, Deserialize, Clone)]
#[serde(default)]
struct GeminiUsage {
    #[serde(rename = "total_input_tokens")]
    input: u32,
    #[serde(rename = "total_cached_tokens")]
    cached: u32,
    #[serde(rename = "total_output_tokens")]
    output: u32,
    #[serde(rename = "total_thought_tokens")]
    thought: u32,
}

impl From<GeminiUsage> for ModelUsage {
    fn from(val: GeminiUsage) -> Self {
        ModelUsage {
            input_tokens: val.input,
            input_cache_read: val.cached,
            // The Gemini API doesn't distinguish between cache reads and writes, and only gives
            // us one number.
            input_cache_written: 0,
            output_tokens: val.output,
            reasoning_tokens: val.thought,
        }
    }
}

/// Interaction response from the Gemini API.
#[derive(Serialize, Deserialize, Clone)]
struct GeminiResponseBody {
    status: String,
    #[serde(default)]
    steps: Vec<GeminiStep>,
    #[serde(default)]
    usage: GeminiUsage,
}

/// Convert Gemini response steps into provider-agnostic `ChatHistoryContent` items.
///
/// Tool results and user input should never appear in API responses; if encountered, they are
/// ignored with a warning.
fn map_response_to_chat_contents(steps: &[GeminiStep]) -> Vec<ChatHistoryContent> {
    fn texts(content: &[GeminiContentItem]) -> impl Iterator<Item = String> + '_ {
        content.iter().filter_map(|item| match item {
            GeminiContentItem::Text { text } => Some(text.clone()),
            GeminiContentItem::Unsupported => None,
        })
    }

    steps
        .iter()
        .flat_map(|step| match step {
            GeminiStep::ModelOutput { content } => {
                texts(content).map(ChatHistoryContent::Text).collect()
            }
            GeminiStep::Thought { summary, .. } => texts(summary)
                .filter(|text| !text.is_empty())
                .map(ChatHistoryContent::Reasoning)
                .collect(),
            GeminiStep::FunctionCall {
                id,
                name,
                arguments,
                ..
            } => vec![ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                id: id.clone(),
                tool_name: name.clone(),
                args: arguments.clone(),
            })],
            GeminiStep::UserInput { .. } | GeminiStep::FunctionResult { .. } => {
                log::warn!(
                    "Got a user input or tool result step from the API response. This is not expected, and will be ignored."
                );

                Vec::new()
            }
            GeminiStep::Unsupported => {
                log::warn!("Got an unsupported step type from the API response; ignoring it.");

                Vec::new()
            }
        })
        .collect()
}

impl<T: HttpClient> AgenticClient for GeminiClient<T> {
    type HistoryItem = GeminiStep;
    const SCHEMA_KEY: &'static str = GEMINI_SCHEMA_KEY;

    fn build_initial_history(&self, request: &ChatRequest<'_>) -> Vec<Self::HistoryItem> {
        let mut steps: Vec<GeminiStep> = request
            .chat_history
            .iter()
            .cloned()
            .flat_map(Vec::<GeminiStep>::from)
            .collect();

        steps.push(GeminiStep::UserInput {
            content: vec![GeminiContentItem::Text {
                text: request.message.clone(),
            }],
        });

        steps
    }

    async fn send_once(
        &self,
        history: &[Self::HistoryItem],
        system_prompt: Option<&str>,
        tools: Option<&[SerializedTool<'_>]>,
        allow_tool_calls: bool,
        reasoning: Option<&ReasoningConfig>,
        max_tokens: Option<u32>,
    ) -> Result<super::base::ProviderTurn<Self::HistoryItem>, LLMError> {
        let key = get_gemini_api_key()?;
        let (model, max_retries) = match &self.config {
            None => (
                env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_GEMINI_MODEL.to_string()),
                DEFAULT_MAX_RETRIES,
            ),
            Some(config) => (config.model.clone(), config.max_retries),
        };

        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse()?);
        headers.insert("x-goog-api-key", key.parse()?);

        let request = GeminiRequestBody {
            model: &model,
            input: history,
            system_instruction: system_prompt,
            generation_config: GeminiGenerationConfig {
                max_output_tokens: max_tokens.or_else(|| {
                    env::var("GEMINI_MAX_TOKENS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                }),
                thinking_level: reasoning.and_then(gemini_thinking_level),
                thinking_summaries: reasoning.map(|_| "auto"),
                tool_choice: (tools.is_some() && !allow_tool_calls).then_some("none"),
            },
            tools: tools.map(|tools| {
                tools
                    .iter()
                    .map(|tool| GeminiTool {
                        r#type: "function",
                        tool: tool.clone(),
                    })
                    .collect()
            }),
            store: false,
        };

        let url = "https://generativelanguage.googleapis.com/v1beta/interactions";
        let mut usage = Vec::new();
        let mut attempt = 0;
        loop {
            let response: GeminiResponseBody =
                send_generation_request(&self.client, &request, &headers, url, max_retries).await?;
            usage.push(response.usage.into());

            let status = response.status;
            if status == "failed" && attempt < max_retries {
                log::warn!("Gemini interaction failed; retrying generation");
                // Retry this turn before executing tools or modifying the conversation history.
                tokio::time::sleep(exponential_backoff_delay(attempt)).await;
                attempt += 1;
                continue;
            }

            // Drop unsupported steps and content so they are not replayed.
            let mut steps = response.steps;
            steps.retain(|step| *step != GeminiStep::Unsupported);
            for step in &mut steps {
                if let GeminiStep::ModelOutput { content }
                | GeminiStep::Thought {
                    summary: content, ..
                } = step
                {
                    content.retain(|item| *item != GeminiContentItem::Unsupported);
                }
            }

            // `incomplete` means the output was cut off, e.g. by `max_output_tokens`.
            let contents = map_response_to_chat_contents(&steps);
            if !matches!(
                status.as_str(),
                "completed" | "requires_action" | "incomplete"
            ) || contents.is_empty()
            {
                return Err(LLMError::GenerationError {
                    provider: "Gemini",
                    finish_reason: status,
                });
            }

            return Ok(ProviderTurn {
                contents,
                native_items: steps,
                usage,
            });
        }
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
    use crate::config::GeminiConfig;
    use crate::constants::DEFAULT_GEMINI_EMBEDDING_DIM;
    use crate::http_client::{MockHttpClient, RecordingSequentialMockHttpClient, ReqwestClient};
    use crate::llm::base::{AgenticClient, ChatHistoryItem, ChatRequest, ContentType};
    use crate::llm::tools::test_utils::MockTool;

    /// Empty thoughts produce no display block, while other text retains its content and order.
    #[test]
    fn empty_thoughts_are_omitted_from_display() {
        let steps: Vec<GeminiStep> = serde_json::from_value(serde_json::json!([
            {"type": "thought", "signature": "opaque"},
            {"type": "thought", "summary": [{"type": "text", "text": "Check the sources.\nThey agree."}]},
            {"type": "model_output", "content": [
                {"type": "text", "text": "The answer."},
                {"type": "text", "text": "More detail."}
            ]},
            {"type": "model_output", "content": [
                {"type": "text", "text": ""},
                {"type": "image", "data": "opaque", "mime_type": "image/png"}
            ]},
            {"type": "google_search_call", "arguments": {}}
        ]))
        .unwrap();

        assert_eq!(
            map_response_to_chat_contents(&steps[..1]),
            [] as [ChatHistoryContent; 0]
        );
        assert_eq!(
            map_response_to_chat_contents(&steps),
            [
                ChatHistoryContent::Reasoning("Check the sources.\nThey agree.".into()),
                ChatHistoryContent::Text("The answer.".into()),
                ChatHistoryContent::Text("More detail.".into()),
                ChatHistoryContent::Text(String::new()),
            ]
        );
        test_eq!(steps[4], GeminiStep::Unsupported);
    }

    #[test]
    fn test_gemini_thinking_level() {
        for (effort, budget, expected) in [
            // `minimal` is unsupported on Gemini 2.5 and 3.1 Pro (the default model).
            (Some("minimal"), None, Some("low")),
            (Some("high"), Some(4096), Some("high")),
            // `xhigh` and `max` are mapped down, since Google caps at `high`.
            (Some("xhigh"), None, Some("high")),
            (Some("max"), None, Some("high")),
            // Gemini does not consistently support disabled thinking, so use low effort instead.
            (Some("none"), None, Some("low")),
            (Some("bogus"), None, None),
            // A budget alone uses the model's default level.
            (None, Some(4096), None),
            (None, None, None),
        ] {
            let reasoning = ReasoningConfig {
                max_tokens: budget,
                effort: effort.map(str::to_owned),
                summary: None,
            };
            test_eq!(gemini_thinking_level(&reasoning), expected);
        }
    }

    #[tokio::test]
    async fn test_send_message_with_mock() {
        dotenv().ok();

        let mock_response: GeminiResponseBody = serde_json::from_value(serde_json::json!({
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Hello from Gemini!"}]}],
            "usage": {"total_input_tokens": 7, "total_output_tokens": 11, "total_tokens": 18}
        }))
        .unwrap();

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
            on_reasoning: None,
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
        test_eq!(res.total_usage().input_tokens, 7);
        test_eq!(res.total_usage().output_tokens, 11);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings_mock() {
        for (model, expected_values) in [
            ("gemini-embedding-2", [3.0, 4.0, 4.0, 3.0]),
            ("gemini-embedding-001", [0.6, 0.8, 0.8, 0.6]),
        ] {
            let config = GeminiConfig {
                api_key: "configured-key".into(),
                embedding_model: model.into(),
                embedding_dims: 768,
                ..GeminiConfig::default()
            };

            let mut first = vec![0.0; config.embedding_dims];
            first[..2].copy_from_slice(&[3.0, 4.0]);

            let mut second = vec![0.0; config.embedding_dims];
            second[..2].copy_from_slice(&[4.0, 3.0]);

            let first = serde_json::json!({"embedding": {"values": first}});
            let second = serde_json::json!({"embedding": {"values": second}});

            let http_client = RecordingSequentialMockHttpClient::new([
                first.clone(),
                second.clone(),
                first,
                second,
                serde_json::json!({"embedding": {"values": [1.0, 0.0, -1.0]}}),
            ]);

            let client = GeminiClient {
                client: http_client.clone(),
                config: Some(config),
            };
            let dest_type = client.dest_type().unwrap();

            for is_query in [false, true] {
                let input = Arc::new(arrow_array::StringArray::from(vec!["A", "B"]));

                let embeddings = if is_query {
                    client.compute_query_embeddings(input)
                } else {
                    client.compute_source_embeddings(input)
                }
                .unwrap();

                let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);
                test_eq!(vector.len(), 2);
                test_eq!(vector.value_length(), 768);
                test_eq!(embeddings.data_type(), dest_type.as_ref());

                let values = arrow_array::cast::as_primitive_array::<arrow_array::types::Float32Type>(
                    vector.values(),
                );

                for (index, expected) in [0, 1, 768, 769].into_iter().zip(expected_values) {
                    assert!((values.value(index) - expected).abs() < f32::EPSILON);
                }
            }

            let empty = client
                .compute_source_embeddings(Arc::new(arrow_array::StringArray::from(
                    Vec::<&str>::new(),
                )))
                .unwrap();

            test_eq!(empty.len(), 0);
            test_eq!(empty.data_type(), dest_type.as_ref());

            let requests = http_client.requests();
            test_eq!(requests.len(), 4);

            let expected_model = format!("models/{model}");

            for (request, text) in requests.iter().zip(["A", "B", "A", "B"]) {
                test_eq!(request["model"], expected_model);
                test_eq!(request["content"]["parts"][0]["text"], text);

                if model == "gemini-embedding-001" {
                    test_eq!(request["outputDimensionality"], 768);
                    test_eq!(request.get("embedContentConfig"), None);
                } else {
                    test_eq!(request["embedContentConfig"]["outputDimensionality"], 768);
                    test_eq!(request.get("outputDimensionality"), None);
                }
            }

            let malformed =
                client.compute_query_embeddings(Arc::new(arrow_array::StringArray::from(vec![
                    "wrong width",
                ])));

            assert!(malformed.is_err());
            test_eq!(http_client.requests().len(), 5);
        }
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
            on_reasoning: None,
            tool_iteration_limit: None,
        };
        let res = client.send_message(&request).await;

        test_ok!(res);
    }

    #[tokio::test]
    async fn test_request_works_with_tools() {
        dotenv().ok();

        let client = GeminiClient::<ReqwestClient>::default();
        let tool = MockTool {
            call_count: Arc::new(Mutex::new(0)),
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
            on_reasoning: None,
            tool_iteration_limit: None,
        };

        let res = client.send_message(&request).await;

        test_ok!(res);
    }

    const FAILED_RESPONSE: &str = r#"{
        "id": "v1_failed",
        "model": "gemini-3.1-pro-preview",
        "object": "interaction",
        "status": "failed",
        "steps": [],
        "usage": {
            "input_tokens_by_modality": [{"modality": "text", "tokens": 121}],
            "total_input_tokens": 121,
            "total_thought_tokens": 103,
            "total_tokens": 224
        }
    }"#;

    #[tokio::test(start_paused = true)]
    async fn test_generation_retry_exhaustion() {
        dotenv().ok();

        {
            let response: serde_json::Value = serde_json::from_str(FAILED_RESPONSE).unwrap();
            let max_retries = 1;
            let http_client = RecordingSequentialMockHttpClient::new(std::iter::repeat_n(
                response,
                max_retries + 1,
            ));
            let client = GeminiClient {
                client: http_client.clone(),
                config: Some(GeminiConfig {
                    max_retries,
                    ..GeminiConfig::default()
                }),
            };

            let result = client.send_message(&ChatRequest::default()).await;

            assert!(matches!(
                result,
                Err(LLMError::GenerationError { provider: "Gemini", finish_reason: reason })
                    if reason == "failed"
            ));
            let requests = http_client.requests();
            test_eq!(requests.len(), max_retries + 1);
            assert!(requests.iter().all(|request| request == &requests[0]));
        }

        // Rate-limited requests are retried up to the configured limit too, not the default.
        let max_retries = 1;
        let http_client = RecordingSequentialMockHttpClient::from_status_bodies(
            std::iter::repeat_n((429, String::from("{}")), max_retries + 1),
        );
        let client = GeminiClient {
            client: http_client.clone(),
            config: Some(GeminiConfig {
                max_retries,
                ..GeminiConfig::default()
            }),
        };

        let result = client.send_message(&ChatRequest::default()).await;

        assert!(matches!(result, Err(LLMError::HttpStatusError(_))));
        test_eq!(http_client.requests().len(), max_retries + 1);
    }

    #[tokio::test]
    async fn test_empty_generation_is_not_success_or_retried() {
        dotenv().ok();

        for (finish_reason, steps) in [
            ("cancelled", None),
            ("completed", None),
            ("incomplete", None),
            (
                "completed",
                Some(serde_json::json!([{"type": "model_output", "content": []}])),
            ),
        ] {
            let mut response: serde_json::Value = serde_json::from_str(FAILED_RESPONSE).unwrap();
            response["status"] = finish_reason.into();
            match steps {
                Some(steps) => response["steps"] = steps,
                None => _ = response.as_object_mut().unwrap().remove("steps"),
            }
            let http_client = RecordingSequentialMockHttpClient::new([response]);
            let client = GeminiClient {
                client: http_client.clone(),
                config: None,
            };

            let result = client.send_message(&ChatRequest::default()).await;

            assert!(matches!(
                result,
                Err(LLMError::GenerationError { provider: "Gemini", finish_reason: reason })
                    if reason == finish_reason
            ));
            test_eq!(http_client.requests().len(), 1);
        }
    }

    #[tokio::test]
    async fn test_callbacks_fire_once_across_generation_retries() {
        dotenv().ok();

        let tool_call_response = serde_json::json!({
            "status": "requires_action",
            "steps": [
                {"type": "thought", "signature": "test-signature"},
                {"type": "function_call", "id": "call_1", "name": "mock_tool", "arguments": {"name": "Alice"}}
            ],
            "usage": {"total_input_tokens": 10, "total_output_tokens": 5, "total_tokens": 15}
        });
        let text_response = serde_json::json!({
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Done!"}]}],
            "usage": {"total_input_tokens": 20, "total_output_tokens": 8, "total_tokens": 28}
        });

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
            on_reasoning: None,
            // The second turn is the last, which forbids tool calls.
            tool_iteration_limit: Some(2),
        };

        let failed_response: serde_json::Value = serde_json::from_str(FAILED_RESPONSE).unwrap();
        let http_client = RecordingSequentialMockHttpClient::new([
            failed_response.clone(),
            tool_call_response,
            failed_response,
            text_response,
        ]);
        let mock_client = GeminiClient {
            client: http_client.clone(),
            config: None,
        };
        let res = mock_client.send_message(&request).await;
        test_ok!(res);

        // Retried requests keep their own usage entries, since each one is priced on its own.
        let response = res.unwrap();
        let input_tokens: Vec<u32> = response.usage.iter().map(|u| u.input_tokens).collect();
        test_eq!(input_tokens, vec![121, 10, 121, 20]);

        let usage = response.total_usage();
        test_eq!(usage.input_tokens, 10 + 20 + 2 * 121);
        test_eq!(usage.output_tokens, 5 + 8);
        test_eq!(usage.reasoning_tokens, 2 * 103);
        test_eq!(*call_count.lock().unwrap(), 1_usize);
        let requests = http_client.requests();
        test_eq!(requests.len(), 4);
        test_eq!(&requests[0], &requests[1]);
        test_eq!(&requests[2], &requests[3]);
        assert!(
            requests[0]["generation_config"]
                .get("tool_choice")
                .is_none()
        );
        test_eq!(requests[2]["generation_config"]["tool_choice"], "none");
        test_eq!(
            &requests[2]["input"],
            &serde_json::json!([
                {"type": "user_input", "content": [{"type": "text", "text": "Test"}]},
                {"type": "thought", "signature": "test-signature"},
                {
                    "type": "function_call",
                    "id": "call_1", "name": "mock_tool", "arguments": {"name": "Alice"}
                },
                {
                    "type": "function_result",
                    "call_id": "call_1", "name": "mock_tool", "result": {"result": "Hello, Alice!"}
                }
            ])
        );

        test_eq!(*tool_call_count.lock().unwrap(), 1_usize);
        let texts = text_segments.lock().unwrap();
        test_eq!(texts.len(), 1);
        test_eq!(texts[0].as_str(), "Done!");
        for request in http_client.requests() {
            test_eq!(
                request["system_instruction"],
                "Follow the system instructions."
            );
            test_eq!(request["store"], false);
            test_eq!(request["tools"][0]["type"], "function");
            test_eq!(request["tools"][0]["name"], "mock_tool");
            test_eq!(request["tools"][0]["parameters"]["type"], "object");
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
