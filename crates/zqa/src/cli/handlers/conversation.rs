//! Command handlers for conversation-related operations.

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex, atomic};

use chrono::Local;

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::state::{SavedChatHistory, get_conversation_history, save_conversation};

/// Resume a previous conversation selected by the user.
///
/// Displays a numbered list of saved conversations, prompts for a selection from standard input,
/// and loads the chosen conversation into the current session. If the current session is dirty,
/// it is saved first.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the resume flow completed successfully.
///
/// # Errors
///
/// * `CLIError::IOError` - If writing prompts or reading user input fails.
/// * `CLIError::LockPoisoningError` - If a lock on conversation state could not be obtained.
pub(crate) fn handle_resume_cmd<O: Write, E: Write>(
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    match get_conversation_history() {
        Err(e) => {
            writeln!(&mut ctx.err, "Failed to load conversations: {e}")?;
        }
        Ok(None) => {
            writeln!(&mut ctx.out, "No saved conversations found.")?;
        }
        Ok(Some(ref v)) if v.is_empty() => {
            writeln!(&mut ctx.out, "No saved conversations found.")?;
        }
        Ok(Some(histories)) => {
            writeln!(&mut ctx.out)?;
            writeln!(&mut ctx.out, "Saved conversations:")?;
            for (i, h) in histories.iter().enumerate() {
                let msg_count = h.history.len();
                writeln!(
                    &mut ctx.out,
                    "  [{}] {} ({} message{})",
                    i + 1,
                    h.title,
                    msg_count,
                    if msg_count == 1 { "" } else { "s" }
                )?;
            }
            writeln!(&mut ctx.out)?;
            write!(&mut ctx.out, "Enter a number (1-{}): ", histories.len())?;
            ctx.out.flush()?;

            let mut input = String::new();
            ctx.input.read_line(&mut input)?;
            let input = input.trim();

            match input.parse::<usize>() {
                Ok(n) if n >= 1 && n <= histories.len() => {
                    let selected = &histories[n - 1];
                    resume_conversation(ctx, selected)?;
                    writeln!(&mut ctx.out, "Resumed: {}", selected.title)?;
                }
                _ => {
                    writeln!(&mut ctx.err, "Invalid selection.")?;
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
pub(crate) fn resume_conversation<O, E>(
    ctx: &mut Context<O, E>,
    conversation: &SavedChatHistory,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    if !save_current_conversation(ctx)? {
        return Err(CLIError::CommandError(
            "could not save the current conversation; keeping it active".into(),
        ));
    }

    *ctx.state.title.lock()? = Some(conversation.title.clone());
    ctx.state.chat_history = Arc::new(Mutex::new(conversation.history.clone()));
    ctx.state.dirty.store(false, atomic::Ordering::Relaxed);
    ctx.state.usage = conversation.usage;

    Ok(())
}

/// Save the current conversation if it has unsaved changes.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// Whether the conversation is safe to discard: `true` when there was nothing to
/// save or the save succeeded, `false` when saving failed. On failure the cause is
/// reported on stderr and the caller should keep the conversation alive.
///
/// # Errors
///
/// Returns a [`CLIError`] if a state lock could not be obtained or the failure
/// could not be written to stderr.
pub(crate) fn save_current_conversation<O, E>(ctx: &mut Context<O, E>) -> Result<bool, CLIError>
where
    O: Write,
    E: Write,
{
    if ctx.state.dirty.load(atomic::Ordering::Relaxed) {
        let chat_history = Arc::clone(&ctx.state.chat_history);
        let history = chat_history.lock()?;
        let date = Local::now();

        let conversation =
            SavedChatHistory {
                history: history.clone(),
                date,
                title: ctx.state.title.lock()?.clone().unwrap_or_else(|| {
                    format!("Conversation on {}", date.format("%Y-%m-%d %H:%M"))
                }),
                usage: ctx.state.usage,
            };

        if let Err(e) = save_conversation(&conversation) {
            writeln!(&mut ctx.err, "Error saving conversation: {e}")?;
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
    use zqa_macros::{test_contains, test_eq};
    use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, MessageRole};

    use super::{handle_resume_cmd, resume_conversation};
    use crate::common::test_support::create_test_context;
    use crate::state::{SavedChatHistory, UsageMetadata, save_conversation};

    #[test]
    #[serial]
    fn test_resume_no_conversations() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let mut ctx = create_test_context(vec![]);
            ctx.input = Box::new(Cursor::new(""));
            handle_resume_cmd(&mut ctx).unwrap();

            let output = String::from_utf8(ctx.out.into_inner()).unwrap();
            test_contains!(output, "No saved conversations found.");
        });
    }

    #[test]
    fn resume_conversation_replaces_session_state() {
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
        resume_conversation(&mut ctx, &saved).unwrap();

        assert_eq!(*ctx.state.chat_history.lock().unwrap(), history);
        assert_eq!(
            *ctx.state.title.lock().unwrap(),
            Some("Attention".to_string())
        );
        assert_eq!(ctx.state.usage.input_tokens, 1000);
        assert!(!ctx.state.dirty.load(Ordering::Relaxed));
    }

    #[test]
    #[serial]
    fn test_resume_loads_selected_conversation() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
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
            ctx.input = Box::new(Cursor::new("1\n"));
            handle_resume_cmd(&mut ctx).unwrap();

            let out = String::from_utf8(ctx.out.into_inner()).unwrap();
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
        });
    }

    #[test]
    #[serial]
    fn test_resume_invalid_selection() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
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
            ctx.input = Box::new(Cursor::new("99\n"));
            handle_resume_cmd(&mut ctx).unwrap();

            let err = String::from_utf8(ctx.err.into_inner()).unwrap();
            test_contains!(err, "Invalid selection.");
        });
    }
}
