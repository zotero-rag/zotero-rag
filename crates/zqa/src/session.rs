//! A minimal, embeddable driver for the zqa command handlers.
//!
//! [`Session`] owns a [`Context`] and forwards command strings to the same
//! [`dispatch_command`](crate::cli::app) the REPL uses, but with caller-supplied
//! output streams. It exists so out-of-crate front-ends (such as `zqa-gui`) can reuse
//! the full retrieval/generation pipeline without depending on the crate internals
//! ([`Context`], [`State`], the handler functions) being `pub`.

use tokio::sync::mpsc;

use crate::cli::app::dispatch_command;
use crate::cli::errors::CLIError;
use crate::cli::handlers::conversation::resume_conversation;
use crate::common::{Context, PathOptions, State};
use crate::config::Config;
use crate::io::EngineEvent;
use crate::state::SavedChatHistory;
use crate::store::lance::LanceZoteroStore;

/// An embeddable driver around the zqa command handlers.
///
/// A `Session` holds the same [`Context`] the CLI builds, so it carries conversation
/// state, config, and the vector store across successive [`dispatch`](Session::dispatch)
/// calls. Commands emit events via the [`mpsc::Sender<EngineEvent>`] supplied during construction.
pub struct Session {
    ctx: Context,
}

impl Session {
    /// Build a session from a config and a pair of output streams.
    ///
    /// # Arguments
    ///
    /// * `config` - The loaded application configuration (see [`crate::load_config`]).
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if a vector store cannot be constructed from `config`
    /// (for example, if no embedding provider is configured).
    pub fn new(
        config: Config,
        event_tx: Option<mpsc::Sender<EngineEvent>>,
    ) -> Result<Self, CLIError> {
        let store = LanceZoteroStore::from_config(&config)?;
        let ctx = Context {
            state: State::default(),
            event_tx,
            config,
            store,
            path_options: PathOptions::default(),
        };

        Ok(Self { ctx })
    }

    /// Dispatch a single command string (e.g. `"/help"` or a bare query) through the
    /// same pipeline as the REPL.
    ///
    /// # Arguments
    ///
    /// * `command` - The command or query to run. Output is written to the session's
    ///   `out`/`err` streams as it is produced.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the session should keep running, `Ok(false)` if the command
    /// requested exit (e.g. `/quit`).
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if the command cannot be parsed or a handler fails
    /// unrecoverably.
    pub fn dispatch(
        &mut self,
        command: &str,
    ) -> impl Future<Output = Result<bool, CLIError>> + Send {
        dispatch_command(command, &mut self.ctx)
    }

    /// Resume a saved conversation without interactive input.
    ///
    /// The current conversation is saved before state is replaced. If that save fails, the current
    /// conversation remains active.
    ///
    /// # Arguments
    ///
    /// * `conversation` - The saved conversation to resume.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if the current conversation cannot be saved or conversation state
    /// cannot be locked.
    pub fn resume_conversation(
        &mut self,
        conversation: &SavedChatHistory,
    ) -> impl Future<Output = Result<(), CLIError>> + Send {
        resume_conversation(&mut self.ctx, conversation)
    }
}
