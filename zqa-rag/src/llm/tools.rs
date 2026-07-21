//! This module provides the `Tool` trait and its implementations. This trait is used to define
//! tools that can be called by the LLM. Tools are used to perform tasks such as summarizing
//! conversations, generating text, and performing calculations.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::future::join_all;
use schemars::Schema;
use serde::Serialize;
use serde::ser::SerializeMap;
use serde_json::{Map, Value};

use crate::llm::base::{
    ChatHistoryContent, ChatHistoryItem, ContentType, MessageRole, ToolCallResponse, ToolUseStats,
};

/// The key for the input schema for tools passed to Anthropic.
pub(crate) const ANTHROPIC_SCHEMA_KEY: &str = "input_schema";
/// The key for the input schema for tools passed to Gemini.
pub(crate) const GEMINI_SCHEMA_KEY: &str = "parametersJsonSchema";
/// The key for the input schema for tools passed to OpenAI.
pub(crate) const OPENAI_SCHEMA_KEY: &str = "parameters";
/// The key for the input schema for tools passed to OpenRouter.
pub(crate) const OPENROUTER_SCHEMA_KEY: &str = "parameters";

/// A trait for tools that can be called by the LLM. Implement this trait to create tools that can
/// be passed to models.
///
/// # Example
///
/// ```rust
///    use schemars::{JsonSchema, schema_for};
///    # use serde::Deserialize;
///    # use std::sync::{Arc, Mutex};
///    # use zqa_rag::llm::tools::Tool;
///
///    /// A mock tool that returns a greeting when given a name. Note that this derives the
///    /// `Debug` trait; this is useful for debugging.
///    #[derive(Debug)]
///    pub(crate) struct MockTool {
///        pub call_count: Arc<Mutex<usize>>,
///        pub schema_key: String,
///    }
///
///    /// The tool is simple and greets a user with a name; this is that input. It is important
///    /// that this derives the `JsonSchema` and `Deserialize` traits. The former is used to
///    /// convert your input type to a JSON schema.
///    #[derive(Deserialize, JsonSchema)]
///    pub(crate) struct MockToolInput {
///        /// The name of the person to greet.
///        name: String,
///    }
///
///    impl Tool for MockTool {
///        /// The name of the tool. This is used to identify the tool when calling it.
///        fn name(&self) -> String {
///            "mock_tool".into()
///        }
///
///        /// A description of the tool. This is used to provide more context to the user.
///        fn description(&self) -> String {
///            "A mock tool that you should call for testing, and pass in a name.".into()
///        }
///
///        /// The JSON schema for the tool's arguments.
///        fn parameters(&self) -> schemars::Schema {
///            schema_for!(MockToolInput)
///        }
///
///        /// The function to call when the tool is invoked. Note that even if your function does
///        /// not need to be async, you should still use `async move` to ensure that the function
///        /// is properly `Send` and `Sync`.
///        fn call(
///            &self,
///            args: serde_json::Value,
///        ) -> std::pin::Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send>>
///        {
///            *self.call_count.lock().unwrap() += 1;
///
///            Box::pin(async move {
///                let input: MockToolInput =
///                    serde_json::from_value(args).map_err(|e| format!("Error: {e}"))?;
///                let greeting = format!("Hello, {}!", input.name);
///
///                serde_json::to_value(greeting).map_err(|e| format!("Error: {e}"))
///            })
///        }
///    }
/// ```
pub trait Tool: Send + Sync {
    /// The name of the tool.
    fn name(&self) -> String;

    /// A description of the tool.
    fn description(&self) -> String;

    /// The JSON schema for the tool's arguments.
    fn parameters(&self) -> Schema;

    /// The function to call when the tool is invoked.
    fn call<'a>(
        &'a self,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send + 'a>>;
}

/// A newtype wrapper struct that lets us add a blanket implementation of `Serialize` to all tools.
#[derive(Clone)]
pub(crate) struct SerializedTool<'a> {
    pub(crate) tool: &'a dyn Tool,
    pub(crate) schema_key: &'static str,
}

impl<'a> SerializedTool<'a> {
    pub(crate) fn new(tool: &'a dyn Tool, schema_key: &'static str) -> Self {
        Self { tool, schema_key }
    }
}

impl Serialize for SerializedTool<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Construct a map describing the tool
        let mut obj = Map::new();
        obj.insert("name".into(), Value::String(self.tool.name()));
        obj.insert("description".into(), Value::String(self.tool.description()));

        // This key varies by model provider
        obj.insert(self.schema_key.into(), self.tool.parameters().into());

        // Serialize the map to get the tool's JSON schema description
        let mut map = serializer.serialize_map(Some(obj.len()))?;
        for (k, v) in &obj {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

/// Convert an optional slice of tools into serializable wrappers.
///
/// This helper takes an optional slice of boxed tools and produces an optional `Vec` of
/// `SerializedTool` wrappers. The wrappers hold references to the original tools and provide a
/// `Serialize` implementation suitable for provider request payloads. In general, `SerializedTool`
/// is more ergonomic: you can directly `.clone()` it and it serializes to a JSON schema as
/// expected by most providers.
///
/// # Arguments:
///
/// * `tools` - Optional slice of tools to expose to the model.
///
/// # Returns
///
/// `Some(Vec<SerializedTool>)` referencing the original tools, or `None` if no tools were
/// provided.
#[must_use]
pub(crate) fn get_owned_tools<'a>(
    tools: Option<&'a [Box<dyn Tool>]>,
    schema_key: &'static str,
) -> Option<Vec<SerializedTool<'a>>> {
    tools.as_ref().map(|iter| {
        iter.iter()
            .map(|f| SerializedTool::new(&**f, schema_key))
            .collect::<Vec<SerializedTool<'a>>>()
    })
}

/// A callback function that is passed a reference to a passed argument, and is thread-safe and
/// has a `'static` lifetime.
pub type CallbackFn<T> = dyn Fn(&T) + Send + Sync + 'static;

/// Process tool calls in a single model response (provider‑agnostic).
///
/// This function consumes a slice of provider‑agnostic `ChatHistoryContent` that represents the
/// model's latest message (text and/or tool call requests). It executes known tool calls and
/// turns unknown calls into error results, then returns the corresponding results as user history
/// items and pushes user-visible entries to `new_contents` (both text and tool call summaries).
///
/// # Arguments:
///
/// * `new_contents` - Accumulates user‑facing `ContentType` items produced while processing.
/// * `contents` - The model's latest message converted to `ChatHistoryContent` values.
/// * `tools` - The available tools for invocation.
/// * `on_tool_call` - Optional callback invoked after each tool call completes, receiving a
///   reference to the [`ToolUseStats`] for that call.
/// * `on_text` - Optional callback invoked for each text segment in the model's response,
///   receiving the text as a string slice.
///
/// # Returns
///
/// A vector of user history items containing tool call results.
pub(crate) async fn process_tool_calls(
    new_contents: &mut Vec<ContentType>,
    contents: &[ChatHistoryContent],
    tools: &[SerializedTool<'_>],
    on_tool_call: Option<&Arc<CallbackFn<ToolUseStats>>>,
    on_text: Option<&Arc<CallbackFn<str>>>,
) -> Vec<ChatHistoryItem> {
    let futures = contents.iter().map(|content| async move {
        match content {
            ChatHistoryContent::Text(s) => (Some(ContentType::Text(s.clone())), None),
            ChatHistoryContent::ToolCallResponse(tool_result) => {
                // This is invalid, but technically we can recover by just ignoring it--so
                // we will.
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be \
                    ignored. Tool result: {tool_result:#?}"
                );
                (None, None)
            }
            ChatHistoryContent::ToolCallRequest(tool_call) => {
                let tool_call_id = tool_call.id.clone();
                let tool_result = if let Some(tool) = tools
                    .iter()
                    .find(|tool| tool.tool.name() == tool_call.tool_name)
                {
                    tool.tool
                        .call(tool_call.args.clone())
                        .await
                        .unwrap_or_else(|e| Value::String(format!("Error calling tool: {e}")))
                } else {
                    let available_tool_names = tools
                        .iter()
                        .map(|tool| tool.tool.name())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let available_tools = if available_tool_names.is_empty() {
                        "none"
                    } else {
                        &available_tool_names
                    };

                    log::warn!(
                        "Model called unknown tool '{}'. Available tools: {available_tools}.",
                        tool_call.tool_name
                    );
                    Value::String(format!(
                        "Tool '{}' does not exist. Available tools: {available_tools}.",
                        tool_call.tool_name
                    ))
                };

                let tool_use_stats = ToolUseStats {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_call.tool_name.clone(),
                    tool_args: tool_call.args.clone(),
                    tool_result: tool_result.clone(),
                };

                let chat_history_item = ChatHistoryItem {
                    role: MessageRole::User,
                    content: vec![ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                        id: tool_call_id,
                        tool_name: tool_call.tool_name.clone(),
                        result: tool_result,
                    })],
                };

                (
                    Some(ContentType::ToolCall(tool_use_stats)),
                    Some(chat_history_item),
                )
            }
        }
    });

    let results = join_all(futures).await;

    let mut new_history_items = Vec::new();
    for (content_opt, history_opt) in results {
        if let Some(content) = content_opt {
            match &content {
                ContentType::ToolCall(stats) => {
                    if let Some(cb) = on_tool_call {
                        cb(stats);
                    }
                }
                ContentType::Text(s) => {
                    if let Some(cb) = on_text {
                        cb(s);
                    }
                }
            }
            new_contents.push(content);
        }
        if let Some(history) = history_opt {
            new_history_items.push(history);
        }
    }

    new_history_items
}

#[cfg(test)]
/// Test utilities for tool calling. This contains a `MockTool` that tests in this crate use to
/// check that tool calling works correctly.
pub(crate) mod test_utils {
    use std::sync::{Arc, Mutex};

    use schemars::{JsonSchema, schema_for};
    use serde::Deserialize;

    use super::Tool;

    /// A mock tool that returns static content. We will test that tool calling works and that we
    /// can deserialize the responses using this.
    #[derive(Debug)]
    pub(crate) struct MockTool {
        pub call_count: Arc<Mutex<usize>>,
    }

    /// The tool is simple and greets a user with a name; this is that input.
    #[derive(Deserialize, JsonSchema)]
    pub(crate) struct MockToolInput {
        /// The name of the person to greet.
        name: String,
    }

    impl Tool for MockTool {
        fn name(&self) -> String {
            "mock_tool".into()
        }

        fn description(&self) -> String {
            "A mock tool that you should call for testing, and pass in a name.".into()
        }

        fn parameters(&self) -> schemars::Schema {
            schema_for!(MockToolInput)
        }

        fn call(
            &self,
            args: serde_json::Value,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send>>
        {
            *self.call_count.lock().unwrap() += 1;

            Box::pin(async move {
                let input: MockToolInput =
                    serde_json::from_value(args).map_err(|e| format!("Error: {e}"))?;
                let greeting = format!("Hello, {}!", input.name);

                serde_json::to_value(greeting).map_err(|e| format!("Error: {e}"))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        sync::{Arc, Mutex},
        time::Instant,
    };

    use schemars::{JsonSchema, schema_for};
    use serde::Deserialize;
    use serde_json::{Value, json};
    use zqa_macros::test_eq;

    use super::*;
    use crate::llm::base::{
        ChatHistoryContent, ChatHistoryItem, ContentType, MessageRole, ToolCallRequest,
        ToolCallResponse, ToolUseStats,
    };
    use crate::llm::tools::test_utils::MockTool;

    #[derive(Debug)]
    pub(crate) struct SlowTool {
        pub delay: std::time::Duration,
    }

    #[derive(Deserialize, JsonSchema)]
    pub(crate) struct SlowToolInput;

    impl Tool for SlowTool {
        fn name(&self) -> String {
            "slow_tool".into()
        }

        fn description(&self) -> String {
            "A slow tool".into()
        }

        fn parameters(&self) -> Schema {
            schema_for!(SlowToolInput)
        }

        fn call(
            &self,
            _args: Value,
        ) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
            let delay = self.delay;
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                Ok(Value::String("Done".to_string()))
            })
        }
    }

    #[tokio::test]
    async fn test_concurrent_execution() {
        let tool = SlowTool {
            delay: std::time::Duration::from_millis(500),
        };
        let boxed_tool: Box<dyn Tool> = Box::new(tool);
        let tools = vec![boxed_tool];
        let serialized_tools = get_owned_tools(Some(&tools), OPENAI_SCHEMA_KEY).unwrap();

        let contents = vec![
            ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                id: "1".into(),
                tool_name: "slow_tool".into(),
                args: json!({}),
            }),
            ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                id: "2".into(),
                tool_name: "slow_tool".into(),
                args: json!({}),
            }),
        ];

        let mut new_contents: Vec<ContentType> = vec![];

        let start = Instant::now();
        let chat_history =
            process_tool_calls(&mut new_contents, &contents, &serialized_tools, None, None).await;
        let duration = start.elapsed();

        // Expect ~500ms for concurrent execution
        assert!(duration.as_millis() < 1000);
        // Ensure that we processed two tool calls
        test_eq!(chat_history.len(), 2);
        test_eq!(new_contents.len(), 2);
    }

    #[tokio::test]
    async fn unknown_tool_call_returns_an_error_result() {
        let call_count = Arc::new(Mutex::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            call_count: Arc::clone(&call_count),
        })];
        let serialized_tools = get_owned_tools(Some(&tools), OPENAI_SCHEMA_KEY).unwrap();
        let contents = vec![ChatHistoryContent::ToolCallRequest(ToolCallRequest {
            id: "call-1".into(),
            tool_name: "imaginary_tool".into(),
            args: json!({"query": "test"}),
        })];
        let mut new_contents = Vec::new();

        let chat_history =
            process_tool_calls(&mut new_contents, &contents, &serialized_tools, None, None).await;
        let expected_result = Value::String(
            "Tool 'imaginary_tool' does not exist. Available tools: mock_tool.".into(),
        );

        assert_eq!(*call_count.lock().unwrap(), 0);
        assert_eq!(
            chat_history,
            vec![ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                    id: "call-1".into(),
                    tool_name: "imaginary_tool".into(),
                    result: expected_result.clone(),
                })],
            }]
        );
        assert_eq!(
            new_contents,
            vec![ContentType::ToolCall(ToolUseStats {
                tool_call_id: "call-1".into(),
                tool_name: "imaginary_tool".into(),
                tool_args: json!({"query": "test"}),
                tool_result: expected_result,
            })]
        );
    }
}
