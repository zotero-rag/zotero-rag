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

use crate::{
    events::AgentEvent,
    llm::base::{
        ChatHistoryContent, ChatHistoryItem, ContentType, MessageRole, ToolCallRequest,
        ToolCallResponse, ToolUseStats,
    },
};

/// The key for the input schema for tools passed to Anthropic.
pub(crate) const ANTHROPIC_SCHEMA_KEY: &str = "input_schema";
/// The key for the input schema for tools passed to Gemini.
pub(crate) const GEMINI_SCHEMA_KEY: &str = "parameters";
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

/// Execute a requested tool, logging its outcome and converting failures into model-visible text.
///
/// # Arguments
///
/// * `tool_call` - The provider's tool name, call ID, and arguments.
/// * `tools` - The tools available on this generation turn.
///
/// # Returns
///
/// The tool's unmodified result, or an error string for failed or unknown calls.
async fn execute_tool_call(
    tool_call: &ToolCallRequest,
    tools: &[SerializedTool<'_>],
) -> Result<Value, String> {
    let tool_call_id = &tool_call.id;
    let tool_name = &tool_call.tool_name;
    let start = std::time::Instant::now();
    log::debug!(
        "Calling tool {tool_name} (id={tool_call_id}): args={}",
        crate::logging::preview(&tool_call.args)
    );
    if let Some(tool) = tools
        .iter()
        .find(|tool| tool.tool.name() == tool_call.tool_name)
    {
        let result = tool.tool.call(tool_call.args.clone()).await;
        log::debug!(
            "Tool {tool_name} (id={tool_call_id}): success={}, elapsed={:.2?}",
            result.is_ok(),
            start.elapsed()
        );
        result.map_err(|e| {
            log::debug!(
                "Tool {tool_name} (id={tool_call_id}) error: {}",
                crate::logging::preview(&e)
            );
            format!("Error calling tool: {e}")
        })
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
        Err(format!(
            "Tool '{}' does not exist. Available tools: {available_tools}.",
            tool_call.tool_name
        ))
    }
}

/// Process tool calls in a single model response (provider-agnostic).
///
/// This function consumes a slice of provider-agnostic `ChatHistoryContent` that represents the
/// model's latest message (text and/or tool call requests). It executes known tool calls and
/// turns unknown calls into error results, then returns the corresponding results as one user
/// history item and pushes user-visible entries to `new_contents` (both text and tool call
/// summaries).
///
/// # Arguments:
///
/// * `new_contents` - Accumulates user-facing `ContentType` items produced while processing.
/// * `contents` - The model's latest message converted to `ChatHistoryContent` values.
/// * `tools` - The available tools for invocation.
/// * `on_event` - Optional callback for the response's events. Text, reasoning and
///   [`AgentEvent::ToolCall`] events are sent in response order before any tool runs. Each
///   [`AgentEvent::ToolResponse`] is sent when its tool finishes, so responses arrive in completion
///   order, which can differ from the order of the calls; match them to their calls by ID.
///
/// # Returns
///
/// One user history item containing all tool call results, or [`None`] if no tools were called.
pub(crate) async fn process_tool_calls(
    new_contents: &mut Vec<ContentType>,
    contents: &[ChatHistoryContent],
    tools: &[SerializedTool<'_>],
    on_event: Option<&Arc<CallbackFn<AgentEvent>>>,
) -> Option<ChatHistoryItem> {
    // Ensure text, reasoning, and tool calls are handled in order. These are "instant" and can be
    // sent as events immediately, while the event for the tool response is sent later.
    if let Some(cb) = on_event {
        for content in contents {
            match content {
                ChatHistoryContent::Text(s) => cb(&AgentEvent::Text(s.clone())),
                ChatHistoryContent::Reasoning(s) => cb(&AgentEvent::Reasoning(s.clone())),
                ChatHistoryContent::ToolCallRequest(call) => cb(&AgentEvent::ToolCall {
                    id: call.id.clone(),
                    name: call.tool_name.clone(),
                    args: call.args.clone(),
                }),
                ChatHistoryContent::ToolCallResponse(_) => {}
            }
        }
    }

    let futures = contents.iter().map(|content| async move {
        match content {
            ChatHistoryContent::Text(s) => (Some(ContentType::Text(s.clone())), None),
            ChatHistoryContent::Reasoning(s) => (Some(ContentType::Reasoning(s.clone())), None),
            ChatHistoryContent::ToolCallResponse(tool_result) => {
                // This is invalid, but technically we can recover by just ignoring it--so
                // we will.
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be \
                    ignored. Tool result: {}",
                    crate::logging::preview(format_args!("{tool_result:?}"))
                );
                (None, None)
            }
            ChatHistoryContent::ToolCallRequest(tool_call) => {
                let tool_call_id = tool_call.id.clone();
                let tool_result = execute_tool_call(tool_call, tools).await;

                if let Some(cb) = on_event {
                    cb(&AgentEvent::ToolResponse {
                        id: tool_call_id.clone(),
                        name: tool_call.tool_name.clone(),
                        response: tool_result.clone(),
                    });
                }

                let tool_use_stats = ToolUseStats {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_call.tool_name.clone(),
                    tool_args: tool_call.args.clone(),
                    tool_result: tool_result.clone(),
                };

                let history_content = ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                    id: tool_call_id,
                    tool_name: tool_call.tool_name.clone(),
                    result: tool_result,
                });

                (
                    Some(ContentType::ToolCall(tool_use_stats)),
                    Some(history_content),
                )
            }
        }
    });

    let results = join_all(futures).await;

    let mut tool_call_responses = Vec::new();
    for (content_opt, history_opt) in results {
        if let Some(content) = content_opt {
            new_contents.push(content);
        }
        if let Some(history_content) = history_opt {
            tool_call_responses.push(history_content);
        }
    }

    (!tool_call_responses.is_empty()).then_some(ChatHistoryItem {
        role: MessageRole::User,
        content: tool_call_responses,
    })
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
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

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

    /// A tool that answers immediately, or waits for `gate` when called with `{"gated": true}`.
    #[derive(Debug)]
    pub(crate) struct SlowTool {
        pub gate: Arc<tokio::sync::Notify>,
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

        fn call(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
            let gate = args["gated"]
                .as_bool()
                .unwrap_or(false)
                .then(|| Arc::clone(&self.gate));
            Box::pin(async move {
                if let Some(gate) = gate {
                    gate.notified().await;
                }

                Ok(Value::String("Done".to_string()))
            })
        }
    }

    #[tokio::test]
    async fn test_concurrent_execution() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let tool = SlowTool {
            gate: Arc::clone(&gate),
        };
        let boxed_tool: Box<dyn Tool> = Box::new(tool);
        let tools = vec![boxed_tool];
        let serialized_tools = get_owned_tools(Some(&tools), OPENAI_SCHEMA_KEY).unwrap();

        let contents = vec![
            ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                id: "1".into(),
                tool_name: "slow_tool".into(),
                args: json!({"gated": true}),
            }),
            ChatHistoryContent::Text("Checking both.".into()),
            ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                id: "2".into(),
                tool_name: "slow_tool".into(),
                args: json!({}),
            }),
        ];

        let mut new_contents: Vec<ContentType> = vec![];
        let events = Arc::new(Mutex::new(Vec::new()));
        let on_event: Arc<CallbackFn<AgentEvent>> = Arc::new({
            let events = Arc::clone(&events);
            move |event: &AgentEvent| {
                let line = match event {
                    AgentEvent::Text(text) => format!("text: {text}"),
                    AgentEvent::Reasoning(text) => format!("reasoning: {text}"),
                    AgentEvent::ToolCall { id, .. } => format!("call: {id}"),
                    AgentEvent::ToolResponse { id, .. } => format!("response: {id}"),
                };
                events.lock().unwrap().push(line);
            }
        });

        let tool_call_results = {
            let processing = process_tool_calls(
                &mut new_contents,
                &contents,
                &serialized_tools,
                Some(&on_event),
            );
            tokio::pin!(processing);

            // The fast call's response must arrive while the gated call is still running, so tools
            // run concurrently and each response is sent as soon as its tool finishes.
            let fast_response = async {
                while !events
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|line| line == "response: 2")
                {
                    tokio::task::yield_now().await;
                }
            };
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                tokio::select! {
                    _ = &mut processing => panic!("processing finished while call 1 was gated"),
                    () = fast_response => {}
                }
            })
            .await
            .expect("call 2 should respond before call 1 is released");

            gate.notify_one();
            processing.await
        };

        // Ensure that we processed two tool calls
        test_eq!(tool_call_results.unwrap().content.len(), 2);
        test_eq!(new_contents.len(), 3);

        // Calls and text go out in content order before any tool finishes, and each response
        // follows when its tool does.
        assert_eq!(
            *events.lock().unwrap(),
            [
                "call: 1",
                "text: Checking both.",
                "call: 2",
                "response: 2",
                "response: 1"
            ]
        );
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

        let tool_call_results =
            process_tool_calls(&mut new_contents, &contents, &serialized_tools, None).await;
        let expected_result: Result<Value, String> =
            Err("Tool 'imaginary_tool' does not exist. Available tools: mock_tool.".into());

        assert_eq!(*call_count.lock().unwrap(), 0);
        assert_eq!(
            tool_call_results,
            Some(ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                    id: "call-1".into(),
                    tool_name: "imaginary_tool".into(),
                    result: expected_result.clone(),
                })],
            })
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
