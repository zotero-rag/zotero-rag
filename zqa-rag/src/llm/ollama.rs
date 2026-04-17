//! Functions, structs, and trait implementations for interacting with `ollama`. This
//! module includes support for text generation only.

use std::env;

use http::HeaderMap;

use super::base::{ApiClient, ChatRequest, CompletionApiResponse};
use super::errors::LLMError;
use crate::clients::ollama::OllamaClient;
use crate::common::request_with_backoff;
use crate::constants::{
    DEFAULT_MAX_RETRIES, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MAX_TOKENS, DEFAULT_OLLAMA_MODEL,
};
use crate::http_client::HttpClient;
use crate::llm::anthropic::{
    AnthropicChatHistoryItem, AnthropicRequest, AnthropicResponse, AnthropicResponseContent,
    build_anthropic_messages_and_tools, map_response_to_chat_contents,
};
use crate::llm::base::ContentType;
use crate::llm::tools::process_tool_calls;

/// Ollama supports the Anthropic Messages API, so we can reuse structs.
type OllamaRequest<'a> = AnthropicRequest<'a>;
type OllamaResponse = AnthropicResponse;

/// Send an API request to `ollama`.
///
/// # Arguments:
///
/// * `client`: An `HttpClient` implementation.
/// * `headers`: A set of headers to pass.
/// * `req`: The request body to send
///
/// # Returns
///
/// The `ollama`-specific response.
async fn send_ollama_request(
    client: &impl HttpClient,
    headers: &HeaderMap,
    req: &OllamaRequest<'_>,
    url: &str,
) -> Result<OllamaResponse, LLMError> {
    const MAX_RETRIES: usize = DEFAULT_MAX_RETRIES;
    let res = request_with_backoff(client, url, headers, req, MAX_RETRIES).await?;

    let body = res.text().await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let response: OllamaResponse = serde_json::from_value(json.clone()).map_err(|err| {
        eprintln!("Failed to deserialize ollama response: we got the response {json}");

        LLMError::DeserializationError(err.to_string())
    })?;

    Ok(response)
}

impl<T: HttpClient> ApiClient for OllamaClient<T> {
    /// Send a request to the `ollama` API, processing tool calls as necessary. Returns a final
    /// response after all tool calls are processed and sent back to the API.
    async fn send_message<'a>(
        &self,
        request: &'a ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, LLMError> {
        // Use config if available, otherwise fall back to env vars
        let (model, max_tokens, base_url) = if let Some(ref config) = self.config {
            (
                config.model.clone(),
                config.max_tokens,
                config.base_url.clone(),
            )
        } else {
            (
                env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.to_string()),
                DEFAULT_OLLAMA_MAX_TOKENS,
                env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_BASE_URL.to_string()),
            )
        };

        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        headers.insert("x-api-key", "ollama".parse()?);
        headers.insert("anthropic-version", "2023-06-01".parse()?);

        // Build the initial messages and tools (owned)
        let (mut chat_history, tools) = build_anthropic_messages_and_tools(request);
        let max_tokens_to_use = request.max_tokens.unwrap_or(max_tokens);

        // Create the initial request
        let req_body = OllamaRequest {
            model: &model,
            max_tokens: max_tokens_to_use,
            messages: &chat_history,
            tools: tools.as_deref(),
        };

        let url = format!("{base_url}/v1/messages");
        let mut response = send_ollama_request(&self.client, &headers, &req_body, &url).await?;

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
            let updated_req_body = OllamaRequest {
                model: &model,
                max_tokens: max_tokens_to_use,
                messages: &chat_history,
                tools: tools.as_deref(),
            };

            response = send_ollama_request(&self.client, &headers, &updated_req_body, &url).await?;

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

    use super::*;
    use crate::clients::ollama::OllamaClient;
    use crate::http_client::{ReqwestClient, SequentialMockHttpClient};
    use crate::llm::anthropic::{
        AnthropicTextResponseContent, AnthropicToolUseResponseContent, AnthropicUsageStats,
    };
    use crate::llm::base::{
        ApiClient, ChatHistoryContent, ChatHistoryItem, ChatRequest, USER_ROLE,
    };
    use crate::llm::tools::test_utils::MockTool;

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        if env::var("CI").is_ok() {
            // Skip this test in CI environments until we get ollama there
            return;
        }

        let client = OllamaClient::<ReqwestClient>::default();
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
            println!("Ollama test error: {:?}", res.as_ref().err());
        }

        test_ok!(res);
    }

    #[tokio::test]
    async fn test_request_with_tool_works() {
        dotenv().ok();

        if env::var("CI").is_ok() {
            // Skip this test in CI environments until we get ollama there
            return;
        }

        let client = OllamaClient::<ReqwestClient>::default();
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
        };

        let request = ChatRequest {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Call the mock_tool function with the name parameter set to 'Alice'"
                .to_owned(),
            tools: Some(&[Box::new(tool)]),
            on_tool_call: None,
            on_text: None,
        };

        let res = client.send_message(&request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Ollama test error: {:?}", res.as_ref().err());
        }

        test_ok!(res);
        assert!(call_count.lock().unwrap().eq(&1_usize));
    }

    #[tokio::test]
    async fn test_callbacks_fire() {
        dotenv().ok();

        let tool_call_response = AnthropicResponse {
            id: "msg-1".into(),
            model: DEFAULT_OLLAMA_MODEL.into(),
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
            model: DEFAULT_OLLAMA_MODEL.into(),
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

        let mock_client = OllamaClient {
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
    #[ignore = "CI not set up with ollama"]
    async fn test_followup_queries_work() {
        dotenv().ok();

        let client = OllamaClient::<ReqwestClient>::default();
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
