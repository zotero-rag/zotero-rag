//! Functions, structs, and trait implementations for interacting with the Gemini API. This module
//! includes support for both text generation and embedding, and tool calling is supported.

use std::env;

use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use super::base::{ApiClient, ChatRequest, CompletionApiResponse};
use super::errors::LLMError;
use crate::clients::gemini::{GeminiClient, get_gemini_api_key};
use crate::common::request_with_backoff;
use crate::constants::{DEFAULT_GEMINI_MODEL, DEFAULT_MAX_RETRIES};
use crate::http_client::HttpClient;
use crate::llm::base::{
    ChatHistoryContent, ChatHistoryItem, ContentType, ToolCallRequest, USER_ROLE,
};
use crate::llm::tools::{GEMINI_SCHEMA_KEY, SerializedTool, get_owned_tools, process_tool_calls};

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

/// Content for requests to the Gemini API
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

impl From<ChatHistoryItem> for GeminiContent {
    fn from(value: ChatHistoryItem) -> Self {
        Self {
            role: value.role,
            parts: value
                .content
                .into_iter()
                .map(|c| match c {
                    ChatHistoryContent::Text(text) => GeminiPart::Text {
                        text,
                        thought_signature: None,
                    },
                    ChatHistoryContent::ToolCallRequest(tool_call) => GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            id: Some(tool_call.id),
                            name: tool_call.tool_name,
                            args: tool_call.args,
                        },
                        thought_signature: None,
                    },
                    ChatHistoryContent::ToolCallResponse(tool_res) => {
                        // Wrap the result in an object with a "result" field if it's not already an object
                        let response = if tool_res.result.is_object() {
                            tool_res.result
                        } else {
                            serde_json::json!({ "result": tool_res.result })
                        };

                        GeminiPart::FunctionResult {
                            function_response: GeminiFunctionResult {
                                name: tool_res.tool_name,
                                response,
                            },
                            thought_signature: None,
                        }
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
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a GeminiToolDeclaration<'a>>,
}

/// Helper to build contents, config, and tools from a ChatRequest.
/// Returns owned data that can then be borrowed by GeminiRequestBody.
fn build_gemini_request_data<'a>(
    req: &'a ChatRequest<'a>,
) -> (
    Vec<GeminiContent>,
    Option<GeminiGenerationConfig>,
    Option<GeminiToolDeclaration<'a>>,
) {
    let model_max = req.max_tokens.or_else(|| {
        env::var("GEMINI_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
    });
    let mut contents: Vec<GeminiContent> =
        req.chat_history.iter().cloned().map(Into::into).collect();

    contents.push(GeminiContent {
        role: USER_ROLE.to_string(),
        parts: vec![GeminiPart::Text {
            text: req.message.clone(),
            thought_signature: None,
        }],
    });

    let owned_tools: Option<Vec<SerializedTool<'a>>> =
        get_owned_tools(req.tools, GEMINI_SCHEMA_KEY);

    let generation_config = Some(GeminiGenerationConfig {
        max_output_tokens: model_max,
        // TODO: Make these configurable
        temperature: Some(1.0),
        top_k: Some(1),
        top_p: Some(1.0),
        thinking_config: Some(GeminiThinkingConfig {
            include_thoughts: Some(false),
            thinking_budget: Some(1024),
        }),
    });

    let tools = owned_tools.map(|tools| GeminiToolDeclaration {
        function_declarations: tools,
    });

    (contents, generation_config, tools)
}

/// Send a Gemini text generation request with retry/backoff and parse the response.
///
/// This helper constructs the model endpoint URL, performs the HTTP request with
/// exponential backoff, and deserializes the response into a single response
/// candidate and usage metadata. If the response contains no candidates or cannot
/// be deserialized, an error is returned.
///
/// # Arguments:
///
/// * `client` - An `HttpClient` implementation used to execute the request.
/// * `headers` - HTTP headers including content type and API key.
/// * `req` - The Gemini request body including contents, config, and tools.
/// * `model` - The Gemini model name to target.
///
/// # Returns
///
/// On success, returns the first `GeminiResponseCandidate` and the
/// accompanying `GeminiUsageMetadata`.
async fn send_gemini_generation_request(
    client: &impl HttpClient,
    headers: &HeaderMap,
    req: &GeminiRequestBody<'_>,
    model: &str,
) -> Result<(GeminiResponseCandidate, GeminiUsageMetadata), LLMError> {
    let url =
        format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent");

    let res = request_with_backoff(client, &url, headers, req, DEFAULT_MAX_RETRIES).await?;

    let body = res.text().await?;
    let json: serde_json::Value = match serde_json::from_str(&body) {
        Ok(json) => json,
        Err(_) => return Err(LLMError::DeserializationError(body)),
    };

    let response: GeminiResponseBody = match serde_json::from_value(json) {
        Ok(response) => response,
        Err(_) => return Err(LLMError::DeserializationError(body)),
    };
    let first_candidate = response
        .candidates
        .first()
        .ok_or_else(|| LLMError::GenericLLMError("No candidates in Gemini response".into()))?;

    Ok((first_candidate.clone(), response.usage_metadata))
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
    thoughts_token_count: Option<u32>,
    candidates_token_count: u32,
    total_token_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<Vec<GeminiTokenDetails>>,
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
            GeminiPart::Text{text, ..} => Some(ChatHistoryContent::Text(text.clone())),
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

impl<T: HttpClient> ApiClient for GeminiClient<T> {
    async fn send_message<'a>(
        &self,
        request: &'a ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, LLMError> {
        let key = get_gemini_api_key()?;
        let model = match &self.config {
            None => env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_GEMINI_MODEL.to_string()),
            Some(config) => config.model.clone(),
        };

        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse()?);
        headers.insert("x-goog-api-key", key.parse()?);

        // Build the initial contents, config, and tools (owned)
        let (mut chat_history, generation_config, tools) = build_gemini_request_data(request);

        // Create the initial request borrowing
        let req_body = GeminiRequestBody {
            contents: &chat_history,
            generation_config: generation_config.clone(),
            tools: tools.as_ref(),
        };

        let (mut response, mut usage) =
            send_gemini_generation_request(&self.client, &headers, &req_body, &model).await?;

        let mut has_tool_calls: bool = response
            .content
            .parts
            .iter()
            .any(|c| matches!(c, GeminiPart::FunctionCall { .. }));

        // Append the contents
        chat_history.push(GeminiContent {
            role: "assistant".into(),
            parts: response.content.parts.clone(),
        });

        let mut contents: Vec<ContentType> = Vec::new();

        while has_tool_calls {
            let converted_contents = map_response_to_chat_contents(&response.content.parts);

            let owned_tools = tools.as_ref().ok_or_else(|| {
                LLMError::ToolCallError(
                    "Model returned tool calls, but no tools were provided.".to_string(),
                )
            })?;
            process_tool_calls(
                &mut chat_history,
                &mut contents,
                &converted_contents,
                &owned_tools.function_declarations,
                request.on_tool_call.as_ref(),
                request.on_text.as_ref(),
            )
            .await?;

            // Create a new request borrowing the updated chat history
            let updated_req_body = GeminiRequestBody {
                generation_config: generation_config.clone(),
                contents: &chat_history,
                tools: tools.as_ref(),
            };

            (response, usage) =
                send_gemini_generation_request(&self.client, &headers, &updated_req_body, &model)
                    .await?;

            // Append the new response to chat history
            chat_history.push(GeminiContent {
                role: "assistant".into(),
                parts: response.content.parts.clone(),
            });

            has_tool_calls = response
                .content
                .parts
                .iter()
                .any(|c| matches!(c, GeminiPart::FunctionCall { .. }));
        }

        // Process the final response (which has no tool calls) to extract text content
        for content in &response.content.parts {
            if let GeminiPart::Text { text, .. } = content {
                if let Some(cb) = request.on_text.as_ref() {
                    cb(text);
                }
                contents.push(ContentType::Text(text.clone()));
            }
        }

        // TODO: Check if this metadata includes tool use
        Ok(CompletionApiResponse {
            content: contents,
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;
    use zqa_macros::test_eq;
    use zqa_macros::test_ok;

    use super::*;
    use crate::clients::gemini::GeminiClient;
    use crate::constants::DEFAULT_GEMINI_EMBEDDING_DIM;
    use crate::http_client::ReqwestClient;
    use crate::http_client::{MockHttpClient, SequentialMockHttpClient};
    use crate::llm::base::{ApiClient, ChatHistoryItem, ChatRequest};
    use crate::llm::tools::test_utils::MockTool;

    #[tokio::test]
    async fn test_send_message_with_mock() {
        dotenv().ok();

        let mock_response = GeminiResponseBody {
            candidates: vec![GeminiResponseCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart::Text {
                        text: "Hello from Gemini!".into(),
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
                role: "assistant".into(),
                content: vec![ChatHistoryContent::Text("Prior".into())],
            }],
            max_tokens: Some(256),
            tools: None,
            on_tool_call: None,
            on_text: None,
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
        test_eq!(res.input_tokens, 7);
        test_eq!(res.output_tokens, 11);
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

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Gemini embedding error: {:?}", embeddings.as_ref().err());
        }

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
            tools: None,
            on_tool_call: None,
            on_text: None,
        };
        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Gemini test error: {:?}", res.as_ref().err());
        }

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
            tools: Some(&[Box::new(tool)]),
            on_tool_call: None,
            on_text: None,
        };

        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Gemini test error: {:?}", res.as_ref().err());
        }

        test_ok!(res);
    }

    #[tokio::test]
    async fn test_callbacks_fire() {
        dotenv().ok();

        let tool_call_response = GeminiResponseBody {
            candidates: vec![GeminiResponseCandidate {
                content: GeminiContent {
                    role: "model".into(),
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
            },
        };
        let text_response = GeminiResponseBody {
            candidates: vec![GeminiResponseCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart::Text {
                        text: "Done!".into(),
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
            tools: Some(&[Box::new(tool)]),
            on_tool_call: Some(Arc::new(move |_| {
                *tool_call_count_cb.lock().unwrap() += 1;
            })),
            on_text: Some(Arc::new(move |s| {
                text_segments_cb.lock().unwrap().push(s.to_string());
            })),
        };

        let mock_client = GeminiClient {
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
