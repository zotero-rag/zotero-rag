use std::pin::Pin;
use std::time::Instant;

use tokio::sync::mpsc::UnboundedSender;
use zqa_rag::llm::tools::Tool;

use crate::utils::terminal::{DIM_TEXT, RESET};

/// `Tool` wrapper that reports its arguments on a status channel before delegating to the
/// wrapped tool.
///
/// The wrappers in this module send their lines over a channel rather than printing, so the
/// consumer decides where they land: the query handler forwards them to the context's error
/// stream, which is the terminal in the CLI and the transcript in the TUI.
pub struct Verbose<T: Tool> {
    inner: T,
    status_tx: UnboundedSender<String>,
}

impl<T: Tool> Verbose<T> {
    pub fn new(inner: T, status_tx: UnboundedSender<String>) -> Self {
        Self { inner, status_tx }
    }
}

impl<T: Tool> Tool for Verbose<T> {
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
            let printed_args = serde_json::to_string(&args).unwrap_or_default();

            // A send fails only when the consumer is gone, in which case the line has
            // nowhere useful to go anyway.
            let _ = self.status_tx.send(format!(
                "{DIM_TEXT}{} ({}){RESET}",
                self.inner.name(),
                printed_args
            ));
            self.inner.call(args).await
        })
    }
}

/// `Tool` wrapper that reports how long the wrapped tool took to run on a status channel
/// (see [`Verbose`] for where the lines end up).
pub struct Timed<T: Tool> {
    inner: T,
    status_tx: UnboundedSender<String>,
}

impl<T: Tool> Timed<T> {
    pub fn new(inner: T, status_tx: UnboundedSender<String>) -> Self {
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
            let _ = self.status_tx.send(format!(
                "{DIM_TEXT}{}{RESET} {} {:.2}s",
                self.inner.name(),
                outcome,
                elapsed.as_secs_f64()
            ));

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
///        Foo::new().verbose(status_tx.clone()).timed(status_tx)
///    );
///    let _future = tool.call(serde_json::Value::Null);
/// ```
pub trait ToolExt: Tool + Sized {
    /// Report the tool's arguments on `status_tx` before each call.
    fn verbose(self, status_tx: UnboundedSender<String>) -> Verbose<Self> {
        Verbose::new(self, status_tx)
    }

    /// Report each call's duration and outcome on `status_tx`.
    fn timed(self, status_tx: UnboundedSender<String>) -> Timed<Self> {
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
    async fn verbose_forwards_call_and_reports_args() {
        let seen_args = Arc::new(Mutex::new(None));
        let (status_tx, mut status_rx) = tokio::sync::mpsc::unbounded_channel();
        let tool = MockTool {
            seen_args: Arc::clone(&seen_args),
        }
        .verbose(status_tx);

        let args = json!({ "foo": "bar" });
        let result = tool.call(args.clone()).await.unwrap();

        assert_eq!(result, json!({ "ok": true }));
        assert_eq!(*seen_args.lock().unwrap(), Some(args));

        let line = status_rx.try_recv().unwrap();
        assert!(line.contains("mock_tool"));
        assert!(line.contains(r#"{"foo":"bar"}"#));
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

        let line = status_rx.try_recv().unwrap();
        assert!(line.contains("mock_tool"));
        assert!(line.contains("completed in"));
    }

    #[tokio::test]
    async fn chained_wrappers_forward_call_and_report_in_order() {
        let seen_args = Arc::new(Mutex::new(None));
        let (status_tx, mut status_rx) = tokio::sync::mpsc::unbounded_channel();
        let tool = MockTool {
            seen_args: Arc::clone(&seen_args),
        }
        .verbose(status_tx.clone())
        .timed(status_tx);

        let args = json!({ "foo": "bar" });
        let result = tool.call(args.clone()).await.unwrap();

        assert_eq!(result, json!({ "ok": true }));
        assert_eq!(*seen_args.lock().unwrap(), Some(args));

        // The argument announcement precedes the timing line.
        let first = status_rx.try_recv().unwrap();
        let second = status_rx.try_recv().unwrap();
        assert!(first.contains(r#"{"foo":"bar"}"#));
        assert!(second.contains("completed in"));
    }

    #[test]
    fn verbose_delegates_metadata() {
        let (status_tx, _status_rx) = tokio::sync::mpsc::unbounded_channel();
        let tool = MockTool {
            seen_args: Arc::new(Mutex::new(None)),
        }
        .verbose(status_tx);

        assert_eq!(tool.name(), "mock_tool");
        assert_eq!(tool.description(), "A mock tool");
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
