use schemars::Schema;
use serde::Serialize;
use serde::ser::SerializeMap;
use serde_json::{Map, Value};
use std::future::Future;
use std::pin::Pin;

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
    fn call(&mut self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;
}

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
