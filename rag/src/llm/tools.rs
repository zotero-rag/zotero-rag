use schemars::Schema;
use serde::Serialize;
use serde::ser::SerializeMap;
use serde_json::{Map, Value};
use std::future::Future;
use std::pin::Pin;

use crate::llm::{
    base::{ChatHistoryContent, ChatHistoryItem, ContentType, ToolCallResponse, ToolUseStats},
    errors::LLMError,
};

/// A trait for tools that can be called by the LLM.
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

impl<'a> Serialize for SerializedTool<'a> {
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
        for (k, v) in obj.iter() {
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
pub fn get_owned_tools<'a>(tools: Option<&'a [Box<dyn Tool>]>) -> Option<Vec<SerializedTool<'a>>> {
    let owned_tools: Option<Vec<SerializedTool>> = tools.as_ref().map(|iter| {
        iter.iter()
            .map(|f| SerializedTool(&**f))
            .collect::<Vec<SerializedTool>>()
    });

    owned_tools
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
pub async fn process_tool_calls<'a, CHI>(
    chat_history: &mut Vec<CHI>,
    new_contents: &mut Vec<ContentType>,
    contents: &[ChatHistoryContent],
    tools: &[SerializedTool<'a>],
) -> Result<(), LLMError>
where
    CHI: From<ChatHistoryItem>,
{
    for content in contents {
        match content {
            ChatHistoryContent::Text(s) => new_contents.push(ContentType::Text(s.clone())),
            ChatHistoryContent::ToolCallResponse(tool_result) => {
                // This is invalid, but technically we can recover by just ignoring it--so
                // we will.
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be ignored. Tool result: {:#?}",
                    tool_result
                );
            }
            ChatHistoryContent::ToolCallRequest(tool_call) => {
                let tool_call_id = tool_call.id.clone();
                let called_tool = tools.iter().find(|tool| tool.0.name() == tool_call.tool_name)
                    .ok_or_else(|| {
                        LLMError::ToolCallError(format!(
                            "Tool {} was called, but it does not exist in the passed list of tools.",
                            tool_call.tool_name
                        ))
                    }
                )?;

                let tool_result = match called_tool.0.call(tool_call.args.clone()).await {
                    Ok(res) => res,
                    Err(e) => Value::String(format!("Error calling tool: {e}")),
                };

                new_contents.push(ContentType::ToolCall(ToolUseStats {
                    tool_name: tool_call.tool_name.clone(),
                    tool_args: tool_call.args.clone(),
                    tool_result: tool_result.clone(),
                }));

                chat_history.push(
                    ChatHistoryItem {
                        role: "user".into(),
                        content: vec![ChatHistoryContent::ToolCallResponse(ToolCallResponse {
                            id: tool_call_id,
                            tool_name: tool_call.tool_name.clone(),
                            result: tool_result,
                        })],
                    }
                    .into(),
                );
            }
        }
    }

    Ok(())
}

/// A mock tool that returns static content. We will test that tool calling works and that we
/// can deserialize the responses using this.
#[cfg(test)]
pub(crate) mod test_utils {
    use schemars::{JsonSchema, schema_for};
    use serde::Deserialize;

    use super::Tool;
    use std::sync::{Arc, Mutex};

    pub(crate) struct MockTool {
        pub call_count: Arc<Mutex<usize>>,
    }

    #[derive(Deserialize, JsonSchema)]
    pub(crate) struct MockToolInput {
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
            "input_schema".into()
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
