//! Command handlers for conversation-related operations.

use std::io::BufRead;
use std::sync::{Arc, Mutex, atomic};

use chrono::Local;

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::io::EngineEvent;
use crate::state::{SavedChatHistory, get_conversation_history, save_conversation};

/// Resume a previous conversation selected by the user.
///
/// Emits a numbered list of saved conversations, prompts for a selection from standard input,
/// and loads the chosen conversation into the current session. If the current session is dirty,
/// it is saved first.
///
/// # Arguments
///
/// * `ctx` - The application context, including conversation state and the event sender.
///
/// # Returns
///
/// `Ok(())` if the resume flow completed successfully.
///
/// # Errors
///
/// * `CLIError::IOError` - If reading user input fails.
/// * `CLIError::ChannelError` - If the event receiver is closed.
/// * `CLIError::LockPoisoningError` - If a lock on conversation state could not be obtained.
pub(crate) async fn handle_resume_cmd(ctx: &mut Context) -> Result<(), CLIError> {
    match get_conversation_history() {
        Err(e) => {
            ctx.emit(EngineEvent::Error {
                message: format!("Failed to load conversations: {e}\n"),
            })
            .await?;
        }
        Ok(None) => {
            ctx.emit(EngineEvent::Text {
                message: "No saved conversations found.\n".into(),
            })
            .await?;
        }
        Ok(Some(ref v)) if v.is_empty() => {
            ctx.emit(EngineEvent::Text {
                message: "No saved conversations found.\n".into(),
            })
            .await?;
        }
        Ok(Some(histories)) => {
            ctx.emit(EngineEvent::Text {
                message: "\nSaved conversations:\n".into(),
            })
            .await?;

            for (i, h) in histories.iter().enumerate() {
                let msg_count = h.history.len();
                ctx.emit(EngineEvent::Text {
                    message: format!(
                        "  [{}] {} ({} message{})\n",
                        i + 1,
                        h.title,
                        msg_count,
                        if msg_count == 1 { "" } else { "s" }
                    ),
                })
                .await?;
            }

            ctx.emit(EngineEvent::Text {
                message: format!("\nEnter a number (1-{}): ", histories.len()),
            })
            .await?;

            let mut input = String::new();
            ctx.input.read_line(&mut input)?;
            let input = input.trim();

            match input.parse::<usize>() {
                Ok(n) if n >= 1 && n <= histories.len() => {
                    let selected = &histories[n - 1];
                    resume_conversation(ctx, selected).await?;
                    ctx.emit(EngineEvent::StatusUpdate {
                        message: format!("Resumed: {}\n", selected.title),
                    })
                    .await?;
                }
                _ => {
                    ctx.emit(EngineEvent::Error {
                        message: "Invalid selection.\n".into(),
                    })
                    .await?;
                }
            }
        }
    }

    Ok(())
}

/// Resume a saved conversation without prompting for input.
///
/// The current conversation is saved before state is replaced. If that save fails, the current
/// state remains active.
///
/// # Arguments
///
/// * `ctx` - The application context whose conversation state will be replaced.
/// * `conversation` - The saved conversation to resume.
///
/// # Errors
///
/// Returns a [`CLIError`] if the current conversation cannot be saved or conversation state cannot
/// be locked.
pub(crate) async fn resume_conversation(
    ctx: &mut Context,
    conversation: &SavedChatHistory,
) -> Result<(), CLIError> {
    if !save_current_conversation(ctx).await? {
        return Err(CLIError::CommandError(
            "could not save the current conversation; keeping it active".into(),
        ));
    }

    // Replace the state with the conversation's state. We can't just update `*ctx.state.*.lock()?`
    // since detached tasks in `cli` may still own clones of that `Arc`.
    ctx.state.title = Arc::new(Mutex::new(Some(conversation.title.clone())));
    ctx.state.chat_history = Arc::new(Mutex::new(conversation.history.clone()));
    ctx.state.dirty.store(false, atomic::Ordering::Relaxed);
    ctx.state.usage = conversation.usage;

    Ok(())
}

/// Save the current conversation if it has unsaved changes.
///
/// # Arguments
///
/// * `ctx` - The application context, including conversation state and the event sender.
///
/// # Returns
///
/// Whether the conversation is safe to discard: `true` when there was nothing to
/// save or the save succeeded, `false` when saving failed. On failure the cause is
/// reported through an error event and the caller should keep the conversation alive.
///
/// # Errors
///
/// Returns a [`CLIError`] if a state lock could not be obtained or the event receiver is closed.
pub(crate) async fn save_current_conversation(ctx: &mut Context) -> Result<bool, CLIError> {
    if ctx.state.dirty.load(atomic::Ordering::Relaxed) {
        let chat_history = Arc::clone(&ctx.state.chat_history);
        let date = Local::now();

        let conversation = {
            let history = chat_history.lock()?;
            SavedChatHistory {
                history: history.clone(),
                date,
                title: ctx.state.title.lock()?.clone().unwrap_or_else(|| {
                    format!("Conversation on {}", date.format("%Y-%m-%d %H:%M"))
                }),
                usage: ctx.state.usage,
            }
        };

        if let Err(e) = save_conversation(&conversation) {
            ctx.emit(EngineEvent::Error {
                message: format!("Error saving conversation: {e}\n"),
            })
            .await?;
            return Ok(false);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::atomic::Ordering;

    use chrono::Local;
    use serial_test::serial;
    use temp_env;
    use tokio::sync::mpsc;
    use zqa_macros::{test_contains, test_eq};
    use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, MessageRole};

    use super::{handle_resume_cmd, resume_conversation};
    use crate::common::test_support::create_test_context;
    use crate::state::{SavedChatHistory, UsageMetadata, save_conversation};

    #[tokio::test]
    #[serial]
    async fn test_resume_no_conversations() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::async_with_vars([("ZQA_STATE_DIR", Some(temp_dir.path()))], async {
            let mut ctx = create_test_context(vec![]);
            let (tx, mut rx) = mpsc::channel(1);
            ctx.event_tx = Some(tx);
            handle_resume_cmd(&mut ctx).await.unwrap();

            let output = rx.try_recv().unwrap().to_string();
            test_contains!(output, "No saved conversations found.");
        })
        .await;
    }

    #[tokio::test]
    async fn resume_conversation_replaces_session_state() {
        let history = vec![ChatHistoryItem {
            role: MessageRole::User,
            content: vec![ChatHistoryContent::Text("What is attention?".into())],
        }];
        let saved = SavedChatHistory {
            history: history.clone(),
            date: Local::now(),
            title: "Attention".into(),
            usage: UsageMetadata {
                input_tokens: 1000,
                output_tokens: 500,
                ..UsageMetadata::default()
            },
        };

        let mut ctx = create_test_context(vec![]);
        resume_conversation(&mut ctx, &saved).await.unwrap();

        test_eq!(*ctx.state.chat_history.lock().unwrap(), history);
        test_eq!(
            *ctx.state.title.lock().unwrap(),
            Some("Attention".to_string())
        );
        test_eq!(ctx.state.usage.input_tokens, 1000);
        assert!(!ctx.state.dirty.load(Ordering::Relaxed));
    }

    #[tokio::test]
    #[serial]
    async fn test_resume_loads_selected_conversation() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::async_with_vars([("ZQA_STATE_DIR", Some(temp_dir.path()))], async {
            let history_a = vec![
                ChatHistoryItem {
                    role: MessageRole::User,
                    content: vec![ChatHistoryContent::Text("What is attention?".into())],
                },
                ChatHistoryItem {
                    role: MessageRole::Assistant,
                    content: vec![ChatHistoryContent::Text(
                        "Attention is a mechanism...".into(),
                    )],
                },
            ];
            let history_b = vec![ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::Text(
                    "Tell me about transformers.".into(),
                )],
            }];

            save_conversation(&SavedChatHistory {
                history: history_a.clone(),
                date: Local::now(),
                title: "Conversation A".into(),
                usage: UsageMetadata {
                    input_tokens: 1000,
                    input_cache_read: 0,
                    input_cache_written: 0,
                    output_tokens: 1000,
                    reasoning_tokens: 100,
                    estimated_cost: 5,
                },
            })
            .unwrap();

            save_conversation(&SavedChatHistory {
                history: history_b.clone(),
                date: Local::now() + chrono::Duration::seconds(1),
                title: "Conversation B".into(),
                usage: UsageMetadata {
                    input_tokens: 2000,
                    input_cache_read: 0,
                    input_cache_written: 0,
                    output_tokens: 1000,
                    reasoning_tokens: 100,
                    estimated_cost: 5,
                },
            })
            .unwrap();

            let mut ctx = create_test_context(vec![]);
            let (tx, mut rx) = mpsc::channel(16);
            ctx.event_tx = Some(tx);
            ctx.input = Box::new(Cursor::new("1\n"));
            handle_resume_cmd(&mut ctx).await.unwrap();

            let out: String = std::iter::from_fn(|| rx.try_recv().ok())
                .map(|event| event.to_string())
                .collect();
            test_contains!(out, "Resumed:");

            let loaded_history = ctx.state.chat_history.lock().unwrap();
            let loaded_usage = ctx.state.usage;
            test_eq!(loaded_history.len(), history_b.len());
            test_eq!(loaded_usage.input_tokens, 2000);
            test_eq!(
                *ctx.state.title.lock().unwrap(),
                Some("Conversation B".to_string())
            );
            assert!(!ctx.state.dirty.load(std::sync::atomic::Ordering::Relaxed));
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn test_resume_invalid_selection() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::async_with_vars([("ZQA_STATE_DIR", Some(temp_dir.path()))], async {
            save_conversation(&SavedChatHistory {
                history: vec![ChatHistoryItem {
                    role: MessageRole::User,
                    content: vec![ChatHistoryContent::Text("Hello".into())],
                }],
                usage: UsageMetadata::default(),
                date: Local::now(),
                title: "Only Conversation".into(),
            })
            .unwrap();

            let mut ctx = create_test_context(vec![]);
            let (tx, mut rx) = mpsc::channel(16);
            ctx.event_tx = Some(tx);
            ctx.input = Box::new(Cursor::new("99\n"));
            handle_resume_cmd(&mut ctx).await.unwrap();

            let err: String = std::iter::from_fn(|| rx.try_recv().ok())
                .map(|event| event.to_string())
                .collect();
            test_contains!(err, "Invalid selection.");
        })
        .await;
    }
}
