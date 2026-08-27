//! A minimal, embeddable driver for the zqa command handlers.
//!
//! [`Session`] owns a [`Context`] and forwards command strings to the same
//! [`dispatch_command`](crate::cli::app) the REPL uses, but with caller-supplied
//! output streams. It exists so out-of-crate front-ends (such as `zqa-gui`) can reuse
//! the full retrieval/generation pipeline without depending on the crate internals
//! ([`Context`], [`State`], the handler functions) being `pub`.

use std::io::{Cursor, Write};

use crate::cli::app::dispatch_command;
use crate::cli::errors::CLIError;
use crate::cli::handlers::conversation::resume_conversation;
use crate::common::{Context, PathOptions, State};
use crate::config::Config;
use crate::state::SavedChatHistory;
use crate::store::lance::LanceZoteroStore;

/// An embeddable driver around the zqa command handlers.
///
/// A `Session` holds the same [`Context`] the CLI builds, so it carries conversation
/// state, config, and the vector store across successive [`dispatch`](Session::dispatch)
/// calls. Command output is written to the `out`/`err` streams supplied at construction;
/// front-ends typically pass channel-backed writers to stream results into a UI.
///
/// The type is deliberately not `Send`: the underlying [`Context`] holds non-`Send` state,
/// so a `Session` is meant to live on a single owning thread that drives it (for example, a
/// dedicated engine thread with its own async runtime).
pub struct Session<O: Write, E: Write> {
    ctx: Context<O, E>,
}

impl<O: Write, E: Write> Session<O, E> {
    /// Build a session from a config and a pair of output streams.
    ///
    /// # Arguments
    ///
    /// * `config` - The loaded application configuration (see [`crate::load_config`]).
    /// * `out` - The stream that command stdout is written to.
    /// * `err` - The stream that command stderr (status lines, warnings) is written to.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if a vector store cannot be constructed from `config`
    /// (for example, if no embedding provider is configured).
    pub fn new(config: Config, out: O, err: E) -> Result<Self, CLIError> {
        let store = LanceZoteroStore::from_config(&config)?;
        let ctx = Context {
            state: State::default(),
            config,
            store,
            path_options: PathOptions::default(),
            // The GUI never drives interactive prompts; feed EOF so any handler that
            // reads input terminates instead of blocking.
            input: Box::new(Cursor::new(Vec::new())),
            out,
            err,
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
    pub async fn dispatch(&mut self, command: &str) -> Result<bool, CLIError> {
        dispatch_command(command, &mut self.ctx).await
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
    pub fn resume_conversation(&mut self, conversation: &SavedChatHistory) -> Result<(), CLIError> {
        resume_conversation(&mut self.ctx, conversation)
    }
}
