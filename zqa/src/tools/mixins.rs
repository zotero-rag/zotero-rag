use std::{pin::Pin, sync::Arc, time::Instant};

use zqa_rag::llm::tools::{CallbackFn, Tool};

use crate::utils::terminal::{DIM_TEXT, RESET};

/// The default trace sink: dimmed lines on `stderr`, matching the CLI's look.
fn stderr_trace_sink() -> Arc<CallbackFn<str>> {
    Arc::new(|line: &str| eprintln!("{DIM_TEXT}{line}{RESET}"))
}

/// `Tool` wrapper that reports its arguments to a trace sink before delegating to the wrapped
/// tool.
pub struct Verbose<T: Tool> {
    inner: T,
    trace: Arc<CallbackFn<str>>,
}

impl<T: Tool> Verbose<T> {
    pub fn new(inner: T, trace: Arc<CallbackFn<str>>) -> Self {
        Self { inner, trace }
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

            (self.trace)(&format!("{} ({})", self.inner.name(), printed_args));
            self.inner.call(args).await
        })
    }
}

/// `Tool` wrapper that reports how long the wrapped tool took to run to a trace sink.
pub struct Timed<T: Tool> {
    inner: T,
    trace: Arc<CallbackFn<str>>,
}

impl<T: Tool> Timed<T> {
    pub fn new(inner: T, trace: Arc<CallbackFn<str>>) -> Self {
        Self { inner, trace }
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

            let status = if result.is_ok() {
                "completed in"
            } else {
                "failed after"
            };
            (self.trace)(&format!(
                "{} {status} {:.2}s",
                self.inner.name(),
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
///    let tool: Box<dyn Tool> = Box::new(
///        Foo::new().verbose().timed()
///    );
///    let _future = tool.call(serde_json::Value::Null);
/// ```
pub trait ToolExt: Tool + Sized {
    /// Wrap this tool so its arguments are printed as dimmed text on `stderr`.
    fn verbose(self) -> Verbose<Self> {
        self.verbose_to(stderr_trace_sink())
    }

    /// Wrap this tool so its arguments are reported to `trace`.
    fn verbose_to(self, trace: Arc<CallbackFn<str>>) -> Verbose<Self> {
        Verbose::new(self, trace)
    }

    /// Wrap this tool so its run time is printed as dimmed text on `stderr`.
    fn timed(self) -> Timed<Self> {
        self.timed_to(stderr_trace_sink())
    }

    /// Wrap this tool so its run time is reported to `trace`.
    fn timed_to(self, trace: Arc<CallbackFn<str>>) -> Timed<Self> {
        Timed::new(self, trace)
    }
}

impl<T> ToolExt for T where T: Tool + Sized {}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::{Arc, Mutex},
    };

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
    async fn verbose_forwards_call_to_inner_tool() {
        let seen_args = Arc::new(Mutex::new(None));
        let tool = MockTool {
            seen_args: Arc::clone(&seen_args),
        }
        .verbose();

        let args = json!({ "foo": "bar" });
        let result = tool.call(args.clone()).await.unwrap();

        assert_eq!(result, json!({ "ok": true }));
        assert_eq!(*seen_args.lock().unwrap(), Some(args));
    }

    #[tokio::test]
    async fn timed_forwards_call_to_inner_tool() {
        let seen_args = Arc::new(Mutex::new(None));
        let tool = MockTool {
            seen_args: Arc::clone(&seen_args),
        }
        .timed();

        let args = json!({ "foo": "bar" });
        let result = tool.call(args.clone()).await.unwrap();

        assert_eq!(result, json!({ "ok": true }));
        assert_eq!(*seen_args.lock().unwrap(), Some(args));
    }

    #[tokio::test]
    async fn chained_wrappers_forward_call_to_inner_tool() {
        let seen_args = Arc::new(Mutex::new(None));
        let tool = MockTool {
            seen_args: Arc::clone(&seen_args),
        }
        .verbose()
        .timed();

        let args = json!({ "foo": "bar" });
        let result = tool.call(args.clone()).await.unwrap();

        assert_eq!(result, json!({ "ok": true }));
        assert_eq!(*seen_args.lock().unwrap(), Some(args));
    }

    #[test]
    fn verbose_delegates_metadata() {
        let tool = MockTool {
            seen_args: Arc::new(Mutex::new(None)),
        }
        .verbose();

        assert_eq!(tool.name(), "mock_tool");
        assert_eq!(tool.description(), "A mock tool");
    }

    #[test]
    fn timed_delegates_metadata() {
        let tool = MockTool {
            seen_args: Arc::new(Mutex::new(None)),
        }
        .timed();

        assert_eq!(tool.name(), "mock_tool");
        assert_eq!(tool.description(), "A mock tool");
    }
}
