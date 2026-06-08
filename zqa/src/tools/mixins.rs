use std::{pin::Pin, time::Instant};

use zqa_rag::llm::tools::Tool;

use crate::utils::terminal::{DIM_TEXT, RESET};

/// `Tool` wrapper that prints its arguments before delegating to the wrapped tool.
pub struct Verbose<T: Tool> {
    inner: T,
}

impl<T: Tool> Verbose<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
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

            eprintln!("{DIM_TEXT}{} ({}){RESET}", self.inner.name(), printed_args);
            self.inner.call(args).await
        })
    }
}

/// `Tool` wrapper that prints how long the wrapped tool took to run
pub struct Timed<T: Tool> {
    inner: T,
}

impl<T: Tool> Timed<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
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

            match result {
                Ok(_) => eprintln!(
                    "{DIM_TEXT}{}{RESET} completed in {:.2}s",
                    self.inner.name(),
                    elapsed.as_secs_f64()
                ),
                Err(_) => eprintln!(
                    "{DIM_TEXT}{}{RESET} failed after {:.2}s",
                    self.inner.name(),
                    elapsed.as_secs_f64()
                ),
            }

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
    fn verbose(self) -> Verbose<Self> {
        Verbose::new(self)
    }

    fn timed(self) -> Timed<Self> {
        Timed::new(self)
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
