use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};

use clap::Parser;
use log::LevelFilter;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
use zqa_pdftools::parse::ExtractedContent;
use zqa_rag::llm::base::ChatHistoryItem;
use {fern, humantime};

use crate::config::Config;
use crate::io::EngineEvent;
use crate::state::UsageMetadata;
use crate::store::lance::LanceZoteroStore;

#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Log level. Options: debug, info, warn, error, off (default)
    #[arg(long, default_value = "off")]
    pub log_level: log::LevelFilter,

    /// Whether to also print out the extracted snippets of the retrieved papers. Note that these
    /// are printed out at the INFO log level, so if the log level is unset or lower than INFO, this
    /// effectively does nothing.
    #[arg(short, long, default_value_t = false)]
    pub print_summaries: bool,
}

/// A user-imported document that is not from their Zotero library.
#[derive(Clone)]
pub(crate) struct UserDocument {
    /// The filename as on disk
    pub(crate) filename: String,
    /// The result of `extract_text`
    pub(crate) contents: ExtractedContent,
    /// The text before "Introduction"
    pub(crate) summary: String,
}

/// The application state. This is embedded in the context, and all state variables are
/// encapsulated in this struct to avoid polluting the `Context`.
#[derive(Default)]
pub(crate) struct State {
    /// The current conversation's chat history
    pub(crate) chat_history: Arc<Mutex<Vec<ChatHistoryItem>>>,
    /// Has the chat history been modified?
    pub(crate) dirty: AtomicBool,
    /// A title generated for the current conversation
    pub(crate) title: Arc<Mutex<Option<String>>>,
    /// The current conversation's usage
    pub(crate) usage: UsageMetadata,
    /// Extracted content for imported documents
    pub(crate) imports: Arc<RwLock<HashMap<String, Arc<UserDocument>>>>,
}

/// Filesystem paths the application resolves at runtime.
///
/// In production these use their defaults; tests override them to point at isolated, per-test
/// locations. Keeping these off of process-global state lets the tests that touch the store
/// and library run in parallel without `#[serial]`.
#[derive(Clone, Debug)]
pub(crate) struct PathOptions {
    /// Override for the Zotero library directory. When `None`, the path is resolved from the
    /// environment (the CI toy library, or the default Zotero location in the user's home directory).
    pub(crate) library_path: Option<PathBuf>,
    /// Path to the file used to persist parsed PDFs between `/process` and `/embed`.
    pub(crate) batch_iter_path: PathBuf,
}

impl Default for PathOptions {
    fn default() -> Self {
        Self {
            library_path: None,
            batch_iter_path: PathBuf::from(crate::cli::app::BATCH_ITER_FILE),
        }
    }
}

/// Application state, configuration, storage, and an optional event sender.
pub(crate) struct Context {
    /// Application state
    pub(crate) state: State,
    /// Optional channel for transmitting events. If `None`, no events are published.
    pub(crate) event_tx: Option<mpsc::Sender<EngineEvent>>,
    /// Config from TOML and env
    pub(crate) config: Config,
    /// The store to use for storage and retrieval
    pub(crate) store: LanceZoteroStore,
    /// Runtime filesystem path overrides (library location, batch-iter file)
    pub(crate) path_options: PathOptions,
}

impl Context {
    /// Emit an event to the channel, if it exists.
    ///
    /// # Arguments
    ///
    /// * `event` - The event to publish.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the event is sent, or if no sender is configured and the event is dropped.
    ///
    /// # Errors
    ///
    /// * `SendError<EngineEvent>` - If the receiver is closed; the error retains the event.
    pub(crate) async fn emit(&self, event: EngineEvent) -> Result<(), SendError<EngineEvent>> {
        let Some(tx) = &self.event_tx else {
            return Ok(());
        };
        tx.send(event).await
    }
}

/// Initialize the `fern` logger.
///
/// # Arguments
///
/// * `log_level` - The log level to use.
///
/// # Errors
///
/// `log::SetLoggerError` if the logger could not be initialized.
pub fn setup_logger(log_level: LevelFilter) -> Result<(), log::SetLoggerError> {
    // Set up logging via fern
    fern::Dispatch::new()
        // Perform allocation-free log formatting
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                humantime::format_rfc3339_millis(std::time::SystemTime::now()),
                record.level(),
                record.target(),
                message
            ));
        })
        .level(log_level)
        .level_for("rustyline", log::LevelFilter::Off)
        .chain(std::io::stdout())
        .apply()
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;
    use zqa_macros::test_eq;

    use super::test_support::{capture_events, create_test_context};
    use crate::io::EngineEvent;

    #[tokio::test]
    async fn test_capture_events_drains_and_restores_sender() {
        let mut ctx = create_test_context(vec![]);
        let (tx, mut rx) = mpsc::channel(1);
        ctx.event_tx = Some(tx);

        let (result, events) = capture_events(&mut ctx, async |ctx| {
            for index in 0..3 {
                ctx.emit(EngineEvent::Text {
                    message: index.to_string(),
                })
                .await
                .unwrap();
            }

            42
        })
        .await;

        test_eq!(result, 42);
        test_eq!(events.len(), 3);

        for (index, event) in events.into_iter().enumerate() {
            let EngineEvent::Text { message } = event else {
                panic!("expected a text event");
            };

            test_eq!(message, index.to_string());
        }

        assert!(rx.try_recv().is_err());
        ctx.emit(EngineEvent::Text {
            message: "restored".into(),
        })
        .await
        .unwrap();
        assert!(
            matches!(rx.try_recv().unwrap(), EngineEvent::Text { message } if message == "restored")
        );
    }

    #[tokio::test]
    async fn test_emit_without_sender() {
        let ctx = create_test_context(vec![]);

        let result = ctx
            .emit(EngineEvent::Text {
                message: "No receiver configured".into(),
            })
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_emit_to_active_receiver() {
        let mut ctx = create_test_context(vec![]);
        let (tx, mut rx) = mpsc::channel(1);
        ctx.event_tx = Some(tx);

        ctx.emit(EngineEvent::Text {
            message: "Delivered event".into(),
        })
        .await
        .unwrap();

        assert!(matches!(
            rx.try_recv().unwrap(),
            EngineEvent::Text { message } if message == "Delivered event"
        ));
    }

    #[tokio::test]
    async fn test_emit_to_closed_receiver_retains_event() {
        let mut ctx = create_test_context(vec![]);
        let (tx, rx) = mpsc::channel(1);
        ctx.event_tx = Some(tx);
        drop(rx);

        let error = ctx
            .emit(EngineEvent::Text {
                message: "Undelivered event".into(),
            })
            .await
            .unwrap_err();

        assert!(matches!(
            error.0,
            EngineEvent::Text { message } if message == "Undelivered event"
        ));
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::PathBuf;

    use tempfile::TempDir;
    use tokio::sync::mpsc;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };

    use super::{Context, PathOptions};
    use crate::LanceZoteroStore;
    use crate::common::State;
    use crate::config::{Config, MockConfig, VoyageAIConfig};
    use crate::io::EngineEvent;

    /// Run an output-only handler and collect the events it sends, in order.
    ///
    /// Events are drained while the handler runs, so a full channel won't block the test.
    /// After the handler finishes, any queued events are collected and the previous sender is
    /// restored. This helper doesn't answer input requests or wait for detached background tasks.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The test context whose event sender will be temporarily replaced.
    /// * `action` - The handler or async closure to run with that context.
    ///
    /// # Returns
    ///
    /// The handler's result and its captured events, including events sent before an error.
    pub(crate) async fn capture_events<T>(
        ctx: &mut Context,
        action: impl AsyncFnOnce(&mut Context) -> T,
    ) -> (T, Vec<EngineEvent>) {
        let (tx, mut rx) = mpsc::channel(1);
        let previous_sender = ctx.event_tx.replace(tx);
        let mut events = Vec::new();
        let result = {
            let action = action(ctx);
            tokio::pin!(action);

            loop {
                tokio::select! {
                    result = &mut action => break result,
                    Some(event) = rx.recv() => events.push(event),
                }
            }
        };

        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        ctx.event_tx = previous_sender;
        (result, events)
    }

    /// Create a config with the mock LLM provider.
    pub(crate) fn get_config(mock_config: MockConfig) -> Config {
        let mut config = Config {
            voyageai: Some(VoyageAIConfig {
                reranker: Some(DEFAULT_VOYAGE_RERANK_MODEL.into()),
                embedding_model: Some(DEFAULT_VOYAGE_EMBEDDING_MODEL.into()),
                embedding_dims: Some(DEFAULT_VOYAGE_EMBEDDING_DIM as usize),
                api_key: Some(String::new()),
            }),
            mock: Some(mock_config),
            ..Default::default()
        };

        config.read_env().unwrap();
        config
    }

    /// Create a context with mock LLM responses and no event sender.
    /// Tests that inspect output can install an event sender on the returned context.
    pub(crate) fn create_test_context(llm_responses: Vec<String>) -> Context {
        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]);

        let config = get_config(MockConfig {
            responses: llm_responses,
        });

        let embedding_config = config.get_embedding_config().unwrap();

        Context {
            state: State::default(),
            event_tx: None,
            store: LanceZoteroStore::from_schema(embedding_config, schema.into()),
            config,
            path_options: PathOptions::default(),
        }
    }

    /// Isolated, per-test filesystem locations that mirror the `temp_db` pattern used in the
    /// `zqa-rag` LanceDB tests: a unique temporary database directory, the toy Zotero library
    /// shipped under `assets/`, and a temporary batch-iter file.
    ///
    /// The [`TempDir`] guard is owned by this struct and must be kept alive for the duration of the
    /// test; dropping it deletes the temporary database. Multiple contexts built from the same
    /// `TestPaths` share one database, mirroring the setup/act split that many tests use (populate
    /// with one context, then assert with another).
    pub(crate) struct TestPaths {
        _dir: TempDir,
        pub(crate) db_uri: String,
        pub(crate) path_options: PathOptions,
    }

    impl TestPaths {
        pub(crate) fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let db_uri = dir
                .path()
                .join("lancedb-table")
                .to_str()
                .unwrap()
                .to_string();
            let library_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("assets")
                .join("Zotero");
            let batch_iter_path = dir.path().join("batch_iter.bin");

            Self {
                _dir: dir,
                db_uri,
                path_options: PathOptions {
                    library_path: Some(library_path),
                    batch_iter_path,
                },
            }
        }

        /// Build a [`Context`] bound to these isolated paths, seeded with the given mock LLM
        /// responses. Reuses [`create_test_context`](create_test_context) for config/schema
        /// setup, then points the store at the temp database and installs the isolated [`PathOptions`].
        pub(crate) fn context(&self, llm_responses: Vec<String>) -> Context {
            let mut ctx = create_test_context(llm_responses);
            ctx.store = ctx.store.with_uri(&self.db_uri);
            ctx.path_options = self.path_options.clone();
            ctx
        }
    }
}
