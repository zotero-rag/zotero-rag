//! Functions, structs, and trait implementations for interacting with `ollama`. This
//! module includes support for text generation only.

use std::env;

use http::HeaderMap;

use super::base::ChatRequest;
use super::errors::LLMError;
use crate::clients::ollama::OllamaClient;
use crate::constants::{
    DEFAULT_ANTHROPIC_REASONING_BUDGET, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MAX_TOKENS,
    DEFAULT_OLLAMA_MODEL,
};
use crate::http_client::HttpClient;
use crate::llm::anthropic::{
    AnthropicChatHistoryItem, AnthropicRequest, AnthropicResponse, AnthropicThinkingConfig,
    map_response_to_chat_contents,
};
use crate::llm::base::{
    AgenticClient, MessageRole, ProviderTurn, ReasoningConfig, send_generation_request,
};
use crate::llm::tools::{ANTHROPIC_SCHEMA_KEY, SerializedTool};

/// Ollama supports the Anthropic Messages API, so we can reuse structs.
type OllamaRequest<'a> = AnthropicRequest<'a>;
type OllamaResponse = AnthropicResponse;

impl<T: HttpClient> AgenticClient for OllamaClient<T> {
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
    ) -> Result<ProviderTurn<Self::HistoryItem>, LLMError> {
        // Use config if available, otherwise fall back to env vars
        let (model, config_max_tokens, base_url) = if let Some(ref config) = self.config {
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

        let request_body = OllamaRequest {
            model: &model,
            max_tokens: max_tokens.unwrap_or(config_max_tokens),
            messages: history,
            system: system_prompt,
            thinking: reasoning.map(|r| AnthropicThinkingConfig::Enabled {
                display: "summarized",
                budget_tokens: r.max_tokens.unwrap_or(DEFAULT_ANTHROPIC_REASONING_BUDGET),
            }),
            output_config: None,
            tools,
        };

        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        headers.insert("x-api-key", "ollama".parse()?);
        headers.insert("anthropic-version", "2023-06-01".parse()?);

        let url = format!("{base_url}/v1/messages");
        let response: OllamaResponse =
            send_generation_request(&self.client, &request_body, &headers, &url).await?;

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

    use super::*;
    use crate::clients::ollama::OllamaClient;
    use crate::http_client::{RecordingSequentialMockHttpClient, ReqwestClient};
    use crate::llm::anthropic::{
        AnthropicOutputTokensDetails, AnthropicResponseContent, AnthropicTextResponseContent,
        AnthropicToolUseResponseContent, AnthropicUsageStats,
    };
    use crate::llm::base::{
        AgenticClient, ChatHistoryContent, ChatHistoryItem, ChatRequest, MessageRole,
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
            model: DEFAULT_OLLAMA_MODEL.into(),
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
            model: DEFAULT_OLLAMA_MODEL.into(),
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
        let mock_client = OllamaClient {
            client: http_client.clone(),
            config: None,
        };
        let res = mock_client.send_message(&request).await;
        test_ok!(res);
        let res = res.unwrap();

        test_eq!(res.usage.input_tokens, 30);
        test_eq!(res.usage.output_tokens, 13);
        test_eq!(*tool_call_count.lock().unwrap(), 1_usize);
        let texts = text_segments.lock().unwrap();
        test_eq!(texts.len(), 1);
        test_eq!(texts[0].as_str(), "Done!");
        for request in http_client.requests() {
            test_eq!(request["system"], "Follow the system instructions.");
        }
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
