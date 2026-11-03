//! A minimal, embeddable driver for the zqa command handlers.
//!
//! [`Session`] owns a [`Context`] and forwards command strings to the same
//! [`dispatch_command`](crate::cli::app) the REPL uses, sending [`EngineEvent`]s to a
//! caller-supplied channel. It exists so out-of-crate front-ends (such as `zqa-gui`) can reuse
//! the full retrieval/generation pipeline without depending on the crate internals
//! ([`Context`], [`State`], the handler functions) being `pub`.

use std::sync::{Arc, Mutex, Weak};

use chrono::Local;
use tokio::sync::mpsc;
use zqa_rag::llm::base::ChatHistoryItem;

use crate::cli::app::dispatch_command;
use crate::cli::errors::CLIError;
use crate::cli::handlers::conversation::resume_conversation;
use crate::common::{Context, PathOptions, State};
use crate::config::Config;
use crate::io::EngineEvent;
use crate::start;
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

/// An opaque identity for a session's current conversation, from [`Session::conversation_id`].
///
/// Two IDs are equal only if they were taken while the same conversation was active. Holding an ID
/// keeps a small allocation reserved, so a conversation started later never compares equal to it.
#[derive(Clone, Debug)]
pub struct ConversationId(Weak<Mutex<Vec<ChatHistoryItem>>>);

impl PartialEq for ConversationId {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ConversationId {}

impl Session {
    /// Build a session from a config and an optional event sender.
    ///
    /// # Arguments
    ///
    /// * `config` - The loaded application configuration (see [`crate::load_config`]).
    /// * `event_tx` - Where commands send their events. `None` discards every event, and commands
    ///   that prompt get no reply.
    ///
    /// # Errors
    ///
    /// * Returns a [`CLIError`] if a vector store cannot be constructed from `config` (for example,
    ///   if no embedding provider is configured).
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

    /// Run the startup steps that come before the first command.
    ///
    /// These are the first-run check, first-run setup, the missing API key notices and the pending
    /// batch notice; see the crate's `start` function for their order. The CLI and the GUI share the
    /// first-run marker in the state directory, so setup runs once, in whichever starts first.
    /// Prompts and notices are sent as events, so keep receiving and answering events while the
    /// returned future runs, as with [`Self::dispatch`]. After setup, the session uses the config it
    /// wrote.
    ///
    /// # Returns
    ///
    /// `Ok(true)` to continue, or `Ok(false)` if the user chose to quit when the first-run check
    /// failed.
    ///
    /// # Errors
    ///
    /// Returns a [`CLIError`] if an event can't be sent, a prompt is dropped without a reply, or
    /// the pending batches can't be counted.
    pub async fn start(&mut self) -> Result<bool, CLIError> {
        start(&mut self.ctx).await
    }

    /// Dispatch a single command string (e.g. `"/help"` or a bare query) through the
    /// same pipeline as the REPL.
    ///
    /// Keep receiving events while the returned future runs: the command waits whenever the
    /// event channel is full or a prompt has not been answered.
    ///
    /// # Arguments
    ///
    /// * `command` - The command or query to run. Its events are sent to the session's event
    ///   sender as they are produced.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the session should keep running, `Ok(false)` if the command
    /// requested exit (e.g. `/quit`).
    ///
    /// # Errors
    ///
    /// * Returns a [`CLIError`] if the command cannot be parsed or a handler fails unrecoverably.
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
    /// * Returns a [`CLIError`] if the current conversation cannot be saved or conversation state
    ///   cannot be locked.
    pub fn resume_conversation(
        &mut self,
        conversation: &SavedChatHistory,
    ) -> impl Future<Output = Result<(), CLIError>> + Send {
        resume_conversation(&mut self.ctx, conversation)
    }

    /// Identify the current conversation.
    ///
    /// The ID changes whenever the session switches conversations, as `/resume` and `/new` do, even
    /// when `/resume` picks the conversation that is already open. Commands that only add to the
    /// current conversation, including a title generated in the background, keep it. A front-end
    /// can compare IDs from before and after a command to learn whether it switched conversations,
    /// without copying the history.
    #[must_use]
    pub fn conversation_id(&self) -> ConversationId {
        ConversationId(Arc::downgrade(&self.ctx.state.chat_history))
    }

    /// Take a snapshot of the current conversation: its history, title and total usage.
    ///
    /// The snapshot is a copy, so later commands don't change it. The snapshot's `date` is the time
    /// it was taken, since the session does not track when a conversation started, and its `title`
    /// is empty until one is generated.
    ///
    /// # Errors
    ///
    /// * Returns a [`CLIError`] if the conversation state cannot be locked.
    pub fn conversation(&self) -> Result<SavedChatHistory, CLIError> {
        Ok(SavedChatHistory {
            history: self.ctx.state.chat_history.lock()?.clone(),
            date: Local::now(),
            title: self.ctx.state.title.lock()?.clone().unwrap_or_default(),
            usage: self.ctx.state.usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use chrono::Local;

    use super::Session;
    use crate::common::test_support::create_test_context;
    use crate::state::{SavedChatHistory, UsageMetadata};

    #[tokio::test]
    async fn conversation_id_changes_only_when_switching() {
        let mut session = Session {
            ctx: create_test_context(vec![]),
        };
        let before = session.conversation_id();

        session.dispatch("/help").await.unwrap();
        assert_eq!(session.conversation_id(), before);

        // Resuming changes the ID even though the saved conversation matches the open one.
        let current = SavedChatHistory {
            history: Vec::new(),
            date: Local::now(),
            title: String::new(),
            usage: UsageMetadata::default(),
        };
        session.resume_conversation(&current).await.unwrap();
        assert_ne!(session.conversation_id(), before);
    }
}
