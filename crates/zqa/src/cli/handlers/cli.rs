//! Command handlers for CLI-related operations.

use std::sync::{Arc, Mutex, atomic};

use crate::cli::errors::CLIError;
use crate::cli::handlers::conversation::save_current_conversation;
use crate::common::Context;
use crate::io::EngineEvent;
use crate::utils::terminal::{BOLD, RESET};

/// Save the current conversation, if needed, and prepare to exit the CLI.
///
/// # Arguments
///
/// * `ctx` - The application context, including conversation state and the event sender.
///
/// # Returns
///
/// `Ok(())` if the conversation was saved successfully or no save was needed.
///
/// # Errors
///
/// Returns a [`CLIError`] if conversation state could not be persisted.
pub(crate) async fn handle_quit_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    save_current_conversation(ctx).await.map(|_| ())
}

/// Emit the active CLI configuration as a text event.
///
/// # Arguments
///
/// * `ctx` - The application context, including configuration and the event sender.
///
/// # Returns
///
/// `Ok(())` if the event was sent or no event sender is configured.
///
/// # Errors
///
/// * `CLIError::ChannelError` - If the event receiver is closed.
pub(crate) async fn handle_config_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    ctx.emit(EngineEvent::Text {
        message: ctx.config.to_string(),
    })
    .await?;

    Ok(())
}

/// Save the current conversation and reset in-memory conversation state.
///
/// # Arguments
///
/// * `ctx` - The application context, including conversation state and the event sender.
///
/// # Returns
///
/// `Ok(())` if the current conversation was saved and state was reset.
///
/// # Errors
///
/// Returns [`CLIError::CommandError`] if the conversation could not be saved; the
/// in-memory conversation is then kept so no data is lost.
pub(crate) async fn handle_new_conversation_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    if !save_current_conversation(ctx).await? {
        return Err(CLIError::CommandError(
            "could not save the current conversation; keeping it active".into(),
        ));
    }

    ctx.state.dirty.store(false, atomic::Ordering::Relaxed);
    ctx.state.chat_history = Arc::new(Mutex::new(Vec::new()));
    ctx.state.title = Arc::new(Mutex::new(None));

    Ok(())
}

/// Emit the CLI help text as a text event.
///
/// # Arguments
///
/// * `ctx` - The application context and event sender.
///
/// # Returns
///
/// `Ok(())` if the event was sent or no event sender is configured.
///
/// # Errors
///
/// * `CLIError::ChannelError` - If the event receiver is closed.
pub(crate) async fn handle_help_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    ctx.emit(EngineEvent::Text {
        message: format!(
            "{BOLD}Basic usage:{RESET}\n\
             - If you haven't already done so, you should run `/process` or `/batch create` to set up an embedding database.\n\
             - Type in a question to ask your configured model, grounded in your Zotero library.\n\
             - Use @ to include a PDF file in your current directory in the conversation.\n\
             \n\
             {BOLD}Available commands:\n{RESET}\n\
             /help\t\t\tShow this help message\n\
             \n\
             {BOLD}Common commands:{RESET}\n\
             /process\t\tPre-process Zotero library. Use this to update the database.\n\
             /search\t\t\tSearch for papers without summarizing them. Usage: /search <query>\n\
             /config\t\t\tShow the currently used configuration.\n\
             /new\t\t\tSave the current conversation and switch to a new one.\n\
             /resume\t\t\tResume a previous conversation.\n\
             /index\t\t\tCreate or update indices.\n\
             /quit\t\t\tExit the program. You can also use Ctrl+C or just type 'quit'.\n\
             \n\
             {BOLD}Batch API commands:{RESET}\n\
             /batch create\t\tPre-process Zotero library, but use a batch embedding API instead.\n\
             /batch check\t\tCheck on the status of a submitted batch.\n\
             /batch cancel <id>\tCancel a pending batch.\n\
             \n\
             {BOLD}Session document commands:{RESET}\n\
             /docs clear\t\tClear all documents in this session.\n\
             /docs list\t\tList all documents in this session.\n\
             /docs remove <key>\tRemove a document with a specified key from the session.\n\
             \n\
             {BOLD}Repair and troubleshooting commands:{RESET}\n\
             /embed\t\t\tRepair failed DB creation by re-adding embeddings.\n\
             /checkhealth\t\tRun health checks on your LanceDB.\n\
             /doctor\t\t\tAttempt to fix issues spotted by /checkhealth.\n\
             /stats\t\t\tShow table statistics.\n\
             /dedup\t\t\tRemove duplicate items.\n\
             \n"
        ),
    })
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::{Arc, Mutex, atomic};

    use serial_test::serial;
    use tokio::sync::mpsc;
    use zqa_macros::{test_contains, test_eq};
    use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, MessageRole};

    use super::handle_help_cmd;
    use crate::common::test_support::create_test_context;
    use crate::io::EngineEvent;

    #[tokio::test]
    async fn test_handle_help_cmd() {
        let mut ctx = create_test_context(vec![]);
        let (tx, mut rx) = mpsc::channel(1);
        ctx.event_tx = Some(tx);
        handle_help_cmd(&mut ctx).await.unwrap();
        let EngineEvent::Text { message: output } = rx.try_recv().unwrap() else {
            panic!("help must emit a text event");
        };
        test_contains!(output, "Available commands:");
        test_contains!(output, "/help");
        test_contains!(output, "/checkhealth");
        test_contains!(output, "/doctor");
        test_contains!(output, "/embed");
        test_contains!(output, "/process");
        test_contains!(output, "/index");
        test_contains!(output, "/stats");
        test_contains!(output, "/dedup");
        test_contains!(output, "/resume");
        test_contains!(output, "/config");
        test_contains!(output, "/new");
        test_contains!(output, "/quit");
        test_contains!(output, "/docs clear");
        test_contains!(output, "/docs remove");
        test_contains!(output, "/docs list");
        test_contains!(output, "/batch check");
        test_contains!(output, "/batch create");
    }

    #[tokio::test]
    #[serial]
    async fn test_new_conversation_keeps_history_when_save_fails() {
        let temp_dir = tempfile::tempdir().unwrap();
        // A regular file where the state dir should be makes `create_dir_all` fail.
        let blocker = temp_dir.path().join("blocker");
        fs::write(&blocker, b"not a directory").unwrap();
        temp_env::async_with_vars([("ZQA_STATE_DIR", Some(blocker.as_path()))], async {
            let mut ctx = create_test_context(vec![]);
            ctx.state.chat_history = Arc::new(Mutex::new(vec![ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::Text("What is attention?".into())],
            }]));
            ctx.state.dirty.store(true, atomic::Ordering::Relaxed);

            let result = super::handle_new_conversation_cmd(&mut ctx).await;

            assert!(result.is_err());
            let history = ctx.state.chat_history.lock().unwrap();
            test_eq!(history.len(), 1);
        })
        .await;
    }
}
