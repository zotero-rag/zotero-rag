use std::pin::Pin;
use std::time::Instant;

use tokio::sync::mpsc::UnboundedSender;
use zqa_rag::llm::tools::Tool;

use crate::io::{EngineEvent, StyledSpan, TextStyle};

/// `Tool` wrapper that reports how long the wrapped tool took to run on a status channel
pub struct Timed<T: Tool> {
    inner: T,
    status_tx: UnboundedSender<EngineEvent>,
}

impl<T: Tool> Timed<T> {
    pub fn new(inner: T, status_tx: UnboundedSender<EngineEvent>) -> Self {
        Self { inner, status_tx }
    }
}

impl<T: Tool> Tool for Timed<T> {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn description(&self) -> String {
        self.inner.description()
    }

    fn parameters(&self) -> schemars::Schema {
        self.inner.parameters()
    }

    fn call<'a>(
        &'a self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();
            let result = self.inner.call(args).await;
            let elapsed = start.elapsed();

            let outcome = if result.is_ok() {
                "completed in"
            } else {
                "failed after"
            };
            let _ = self.status_tx.send(EngineEvent::StyledText {
                spans: vec![StyledSpan {
                    text: format!(
                        "{} {} {:.2}s\n",
                        self.inner.name(),
                        outcome,
                        elapsed.as_secs_f64()
                    ),
                    style: TextStyle::Secondary,
                }],
            });

            result
        })
    }
}

/// Extension trait that provides ergonomics for the tool wrappers in this module.
///
/// # Example
///
/// ```rust
///    # use std::str::FromStr;
///    # use std::pin::Pin;
///    # use schemars::schema_for;
///    use serde_json::json;
///    use zqa_rag::llm::tools::Tool;
///    use zqa::tools::mixins::ToolExt;
///
///    struct Foo;
///    #
///    # impl Foo {
///    #     fn new() -> Self {
///    #         Self
///    #     }
///    # }
///    #
///    impl Tool for Foo {
///        // ...
///        # fn name(&self) -> String {
///        #     "Foo".into()
///        # }
///        #
///        # fn description(&self) -> String {
///        #     "An example tool".into()
///        # }
///        #  
///        # fn parameters(&self) -> schemars::Schema {
///        #     schema_for!(())
///        # }
///        # fn call<'a>(
///        #     &'a self, args: serde_json::Value
///        # ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>> {
///        #     Box::pin(async move {
///        #         Ok(json!({ "message": "Foo called" }))
///        #     })
///        # }
///    }
///
///    let (status_tx, _status_rx) = tokio::sync::mpsc::unbounded_channel();
///    let tool: Box<dyn Tool> = Box::new(
///        Foo::new().timed(status_tx)
///    );
///    let _future = tool.call(serde_json::Value::Null);
/// ```
pub trait ToolExt: Tool + Sized {
    /// Report each call's duration and outcome on `status_tx`.
    fn timed(self, status_tx: UnboundedSender<EngineEvent>) -> Timed<Self> {
        Timed::new(self, status_tx)
    }
}

impl<T> ToolExt for T where T: Tool + Sized {}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    use schemars::schema_for;
    use serde_json::json;

    use super::*;

    /// Mock tool that stores shared state of passed args
    struct MockTool {
        seen_args: Arc<Mutex<Option<serde_json::Value>>>,
    }

    impl Tool for MockTool {
        fn name(&self) -> String {
            "mock_tool".into()
        }

        fn description(&self) -> String {
            "A mock tool".into()
        }

        fn parameters(&self) -> schemars::Schema {
            schema_for!(())
        }

        fn call<'a>(
            &'a self,
            args: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>> {
            Box::pin(async move {
                *self.seen_args.lock().unwrap() = Some(args);
                Ok(json!({ "ok": true }))
            })
        }
    }

    #[tokio::test]
    async fn timed_forwards_call_and_reports_duration() {
        let seen_args = Arc::new(Mutex::new(None));
        let (status_tx, mut status_rx) = tokio::sync::mpsc::unbounded_channel();
        let tool = MockTool {
            seen_args: Arc::clone(&seen_args),
        }
        .timed(status_tx);

        let args = json!({ "foo": "bar" });
        let result = tool.call(args.clone()).await.unwrap();

        assert_eq!(result, json!({ "ok": true }));
        assert_eq!(*seen_args.lock().unwrap(), Some(args));

        let line = status_rx.try_recv().unwrap().to_string();
        assert!(line.contains("mock_tool"));
        assert!(line.contains("completed in"));
    }

    #[test]
    fn timed_delegates_metadata() {
        let (status_tx, _status_rx) = tokio::sync::mpsc::unbounded_channel();
        let tool = MockTool {
            seen_args: Arc::new(Mutex::new(None)),
        }
        .timed(status_tx);

        assert_eq!(tool.name(), "mock_tool");
        assert_eq!(tool.description(), "A mock tool");
    }
}
