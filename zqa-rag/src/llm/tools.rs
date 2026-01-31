//! This module provides the `Tool` trait and its implementations. This trait is used to define
//! tools that can be called by the LLM. Tools are used to perform tasks such as summarizing
//! conversations, generating text, and performing calculations.

use futures::future::join_all;
use schemars::Schema;
use serde::Serialize;
use serde::ser::SerializeMap;
use serde_json::{Map, Value};
use std::future::Future;
use std::pin::Pin;

use crate::llm::{
    base::{
        ChatHistoryContent, ChatHistoryItem, ContentType, ToolCallResponse, ToolUseStats, USER_ROLE,
    },
    errors::LLMError,
};

/// The key for the input schema for tools passed to Anthropic. Use this key as the return
/// value of the `schema_key` method in the `Tool` trait.
pub const ANTHROPIC_SCHEMA_KEY: &str = "input_schema";
/// The key for the input schema for tools passed to Gemini. Use this key as the return
/// value of the `schema_key` method in the `Tool` trait.
pub const GEMINI_SCHEMA_KEY: &str = "parametersJsonSchema";
/// The key for the input schema for tools passed to OpenAI. Use this key as the return
/// value of the `schema_key` method in the `Tool` trait.
pub const OPENAI_SCHEMA_KEY: &str = "parameters";

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
///        /// The key used by the API to describe the tool's parameters. See the method's docs for
///        /// more details.
///        fn schema_key(&self) -> String {
///            self.schema_key.clone()
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

    /// The key used by the API to describe the tool's parameters.
    ///
    /// Most API providers have a standardized name and description property for tools, but the key
    /// used to represent the input schema is *not* the same. This function should return that
    /// *key*. For example, Anthropic uses "input_schema" (see docs:
    /// https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview), and OpenAI uses
    /// "parameters" (see docs:
    /// https://platform.openai.com/docs/guides/function-calling#function-tool-example).
    fn schema_key(&self) -> String;

    /// The function to call when the tool is invoked.
    fn call(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;
}

/// A newtype wrapper struct that lets us add a blanket implementation of `Serialize` to all tools.
#[derive(Clone)]
pub struct SerializedTool<'a>(pub &'a dyn Tool);

impl Serialize for SerializedTool<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Construct a map describing the tool
        let mut obj = Map::new();
        obj.insert("name".into(), Value::String(self.0.name()));
        obj.insert("description".into(), Value::String(self.0.description()));

        // This key varies by model provider
        obj.insert(self.0.schema_key(), self.0.parameters().into());

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
pub fn get_owned_tools<'a>(tools: Option<&'a [Box<dyn Tool>]>) -> Option<Vec<SerializedTool<'a>>> {
    tools.as_ref().map(|iter| {
        iter.iter()
            .map(|f| SerializedTool(&**f))
            .collect::<Vec<SerializedTool<'a>>>()
    })
}

/// Process tool calls in a single model response (provider‑agnostic).
///
/// This function consumes a slice of provider‑agnostic `ChatHistoryContent` that represents the
/// model's latest message (text and/or tool call requests). It executes tool calls, appends the
/// corresponding tool results to `chat_history` as user messages, and pushes user‑visible entries
/// to `new_contents` (both text and tool call summaries).
///
/// # Arguments:
///
/// * `chat_history` - Mutable, provider‑specific chat history. Each element implements
///   `From<ChatHistoryItem>` so the function can insert tool results in the provider's native type.
/// * `new_contents` - Accumulates user‑facing `ContentType` items produced while processing.
/// * `contents` - The model's latest message converted to `ChatHistoryContent` values.
/// * `tools` - The available tools for invocation.
///
/// # Returns
///
/// `Ok(())` on success, or an `LLMError` if a tool dispatch fails.
///
/// # Errors
///
/// Returns an error if a tool is called that does not exist in the provided list of tools.
pub async fn process_tool_calls<CHI>(
    chat_history: &mut Vec<CHI>,
    new_contents: &mut Vec<ContentType>,
    contents: &[ChatHistoryContent],
    tools: &[SerializedTool<'_>],
) -> Result<(), LLMError>
where
    CHI: From<ChatHistoryItem>,
{
    let futures = contents.iter().map(|content| async move {
        match content {
            ChatHistoryContent::Text(s) => {
                Ok::<_, LLMError>((Some(ContentType::Text(s.clone())), None))
            }
            ChatHistoryContent::ToolCallResponse(tool_result) => {
                // This is invalid, but technically we can recover by just ignoring it--so
                // we will.
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be ignored. Tool result: {tool_result:#?}"
                );
                Ok((None, None))
            }
            ChatHistoryContent::ToolCallRequest(tool_call) => {
                let tool_call_id = tool_call.id.clone();
                let called_tool = tools
                    .iter()
                    .find(|tool| tool.0.name() == tool_call.tool_name)
                    .ok_or_else(|| {
                        LLMError::ToolCallError(format!(
                            "Tool {} was called, but it does not exist in the passed list of tools.",
                            tool_call.tool_name
                        ))
                    })?;

                let tool_result = match called_tool.0.call(tool_call.args.clone()).await {
                    Ok(res) => res,
                    Err(e) => Value::String(format!("Error calling tool: {e}")),
                };

                let tool_use_stats = ToolUseStats {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_call.tool_name.clone(),
                    tool_args: tool_call.args.clone(),
                    tool_result: tool_result.clone(),
                };

                let chat_history_item = ChatHistoryItem {
                    role: USER_ROLE.into(),
                    content: vec![ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                        id: tool_call_id,
                        tool_name: tool_call.tool_name.clone(),
                        result: tool_result,
                    })],
                };

                Ok((Some(ContentType::ToolCall(tool_use_stats)), Some(chat_history_item)))
            }
        }
    });

    let results = join_all(futures).await;

    for result in results {
        let (content_opt, history_opt) = result?;
        if let Some(content) = content_opt {
            new_contents.push(content);
        }
        if let Some(history) = history_opt {
            chat_history.push(history.into());
        }
    }

    Ok(())
}

#[cfg(test)]
/// Test utilities for tool calling. This contains a `MockTool` that tests in this crate use to
/// check that tool calling works correctly.
pub(crate) mod test_utils {
    use schemars::{JsonSchema, schema_for};
    use serde::Deserialize;

    use super::Tool;
    use std::sync::{Arc, Mutex};

    /// A mock tool that returns static content. We will test that tool calling works and that we
    /// can deserialize the responses using this.
    #[derive(Debug)]
    pub(crate) struct MockTool {
        pub call_count: Arc<Mutex<usize>>,
        pub schema_key: String,
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

        fn schema_key(&self) -> String {
            self.schema_key.clone()
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
    use super::*;
    use crate::llm::base::{ChatHistoryContent, ChatHistoryItem, ContentType, ToolCallRequest};
    use schemars::{JsonSchema, schema_for};
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Instant;

    #[derive(Debug)]
    pub(crate) struct SlowTool {
        pub delay: std::time::Duration,
    }

    #[derive(Deserialize, JsonSchema)]
    pub(crate) struct SlowToolInput {}

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

        fn schema_key(&self) -> String {
            "input_schema".into()
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
        let serialized_tools = get_owned_tools(Some(&tools)).unwrap();

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

        let mut chat_history: Vec<ChatHistoryItem> = vec![];
        let mut new_contents: Vec<ContentType> = vec![];

        let start = Instant::now();
        process_tool_calls(
            &mut chat_history,
            &mut new_contents,
            &contents,
            &serialized_tools,
        )
        .await
        .unwrap();
        let duration = start.elapsed();

        println!("Execution time: {:?}", duration);
        // Expect ~500ms for concurrent execution
        assert!(duration.as_millis() < 1000);
        // Ensure that we processed two tool calls
        assert_eq!(chat_history.len(), 2);
        assert_eq!(new_contents.len(), 2);
    }
}
