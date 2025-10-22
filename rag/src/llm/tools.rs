use schemars::Schema;
use serde_json::Value;
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

    /// The function to call when the tool is invoked.
    fn call(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;
}
