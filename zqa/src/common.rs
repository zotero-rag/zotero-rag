use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};

use clap::Parser;
use log::LevelFilter;
use zqa_pdftools::parse::ExtractedContent;
use zqa_rag::llm::base::ChatHistoryItem;
use {fern, humantime};

use crate::config::Config;
use crate::state::UsageMetadata;
use crate::store::lance::LanceZoteroStore;

#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Whether to use the tui interface. This is very unstable and is not feature-complete, and it
    /// should not be used for regular tasks.
    #[arg(long, default_value_t = false)]
    pub tui: bool,

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

/// A structure that holds the application context, including CLI arguments and writers
/// for `stdout` and `stderr`.
pub(crate) struct Context<OutStream: Write, ErrStream: Write> {
    /// Application state
    pub(crate) state: State,
    /// Config from TOML and env
    pub(crate) config: Config,
    /// The store to use for storage and retrieval
    pub(crate) store: LanceZoteroStore,
    /// Runtime filesystem path overrides (library location, batch-iter file)
    pub(crate) path_options: PathOptions,
    /// Abstraction for `stdin()`. Boxed rather than generic because, unlike `out`/`err` (which
    /// tests substitute with an inspectable `Cursor` to assert on), input is only ever *supplied*:
    /// no caller needs the concrete reader type. Boxing also keeps the input concern off every
    /// handler signature.
    pub(crate) input: Box<dyn BufRead>,
    /// Abstraction for `stdout()`
    pub(crate) out: OutStream,
    /// Abstraction for `stderr()`
    pub(crate) err: ErrStream,
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
pub(crate) mod test_support {
    use std::io::Cursor;
    use std::path::PathBuf;

    use tempfile::TempDir;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };

    use super::{Context, PathOptions};
    use crate::LanceZoteroStore;
    use crate::common::State;
    use crate::config::{Config, MockConfig, VoyageAIConfig};

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

    /// Create a default `Context` object where the output and error streams are buffers that can
    /// be written into. This allows for the output to be easily inspected in tests.
    pub(crate) fn create_test_context(
        llm_responses: Vec<String>,
    ) -> Context<Cursor<Vec<u8>>, Cursor<Vec<u8>>> {
        let out_buf: Vec<u8> = Vec::new();
        let out = Cursor::new(out_buf);

        let err_buf: Vec<u8> = Vec::new();
        let err = Cursor::new(err_buf);

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
            store: LanceZoteroStore::from_schema(embedding_config, schema.into()),
            config,
            path_options: PathOptions::default(),
            // Default to empty input (EOF); tests that drive prompts overwrite `ctx.input`.
            input: Box::new(Cursor::new(Vec::new())),
            out,
            err,
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
        pub(crate) fn context(
            &self,
            llm_responses: Vec<String>,
        ) -> Context<Cursor<Vec<u8>>, Cursor<Vec<u8>>> {
            let mut ctx = create_test_context(llm_responses);
            ctx.store = ctx.store.with_uri(&self.db_uri);
            ctx.path_options = self.path_options.clone();
            ctx
        }
    }
}
