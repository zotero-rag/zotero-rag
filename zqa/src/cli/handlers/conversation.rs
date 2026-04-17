//! Command handlers for conversation-related operations.

use std::{
    io::{self, BufRead, Write},
    sync::{Arc, Mutex, atomic},
};

use chrono::Local;

use crate::{
    cli::errors::CLIError,
    common::Context,
    state::{SavedChatHistory, get_conversation_history, save_conversation},
};

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
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    resume_with_reader(ctx, &mut reader)
}

fn resume_with_reader<O, E, R>(ctx: &mut Context<O, E>, reader: &mut R) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
    R: BufRead,
{
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
        Ok(Some(mut histories)) => {
            histories.sort_by_key(|b| std::cmp::Reverse(b.date));

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
            reader.read_line(&mut input)?;
            let input = input.trim();

            match input.parse::<usize>() {
                Ok(n) if n >= 1 && n <= histories.len() => {
                    if ctx.state.dirty.load(atomic::Ordering::Relaxed) {
                        let chat_history = Arc::clone(&ctx.state.chat_history);
                        let history = chat_history.lock()?;
                        let date = Local::now();
                        let conversation = SavedChatHistory {
                            history: history.clone(),
                            date,
                            title: ctx.state.title.lock()?.clone().unwrap_or_else(|| {
                                format!("Conversation on {}", date.format("%Y-%m-%d %H:%M"))
                            }),
                        };
                        if let Err(e) = save_conversation(&conversation)
                            && let Err(write_err) =
                                writeln!(&mut ctx.err, "Error saving conversation: {e}")
                        {
                            log::error!("Failed to write to stderr: {write_err}");
                        }
                    }

                    let selected = histories.swap_remove(n - 1);
                    let title = selected.title.clone();
                    ctx.state.chat_history = Arc::new(Mutex::new(selected.history));
                    ctx.state.dirty.store(false, atomic::Ordering::Relaxed);
                    writeln!(&mut ctx.out, "Resumed: {title}")?;
                }
                _ => {
                    writeln!(&mut ctx.err, "Invalid selection.")?;
                }
            }
        }
    }

    Ok(())
}

pub(super) fn save_current_conversation<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
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
            };

        if let Err(e) = save_conversation(&conversation) {
            writeln!(&mut ctx.err, "Error saving conversation: {e}")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use chrono::Local;
    use temp_env;
    use zqa_macros::{test_contains, test_eq};
    use zqa_rag::llm::base::{ASSISTANT_ROLE, ChatHistoryContent, ChatHistoryItem, USER_ROLE};

    use super::resume_with_reader;
    use crate::{
        cli::app::tests::create_test_context,
        state::{SavedChatHistory, save_conversation},
    };

    #[test]
    fn test_resume_no_conversations() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let mut ctx = create_test_context();
            let mut reader = Cursor::new("");
            resume_with_reader(&mut ctx, &mut reader).unwrap();

            let output = String::from_utf8(ctx.out.into_inner()).unwrap();
            test_contains!(output, "No saved conversations found.");
        });
    }

    #[test]
    fn test_resume_loads_selected_conversation() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let history_a = vec![
                ChatHistoryItem {
                    role: USER_ROLE.into(),
                    content: vec![ChatHistoryContent::Text("What is attention?".into())],
                },
                ChatHistoryItem {
                    role: ASSISTANT_ROLE.into(),
                    content: vec![ChatHistoryContent::Text(
                        "Attention is a mechanism...".into(),
                    )],
                },
            ];
            let history_b = vec![ChatHistoryItem {
                role: USER_ROLE.into(),
                content: vec![ChatHistoryContent::Text(
                    "Tell me about transformers.".into(),
                )],
            }];

            save_conversation(&SavedChatHistory {
                history: history_a.clone(),
                date: Local::now(),
                title: "Conversation A".into(),
            })
            .unwrap();

            save_conversation(&SavedChatHistory {
                history: history_b.clone(),
                date: Local::now() + chrono::Duration::seconds(1),
                title: "Conversation B".into(),
            })
            .unwrap();

            let mut ctx = create_test_context();
            let mut reader = Cursor::new("1\n");
            resume_with_reader(&mut ctx, &mut reader).unwrap();

            let out = String::from_utf8(ctx.out.into_inner()).unwrap();
            test_contains!(out, "Resumed:");

            let loaded = ctx.state.chat_history.lock().unwrap();
            test_eq!(loaded.len(), history_b.len());
            assert!(!ctx.state.dirty.load(std::sync::atomic::Ordering::Relaxed));
        });
    }

    #[test]
    fn test_resume_invalid_selection() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            save_conversation(&SavedChatHistory {
                history: vec![ChatHistoryItem {
                    role: USER_ROLE.into(),
                    content: vec![ChatHistoryContent::Text("Hello".into())],
                }],
                date: Local::now(),
                title: "Only Conversation".into(),
            })
            .unwrap();

            let mut ctx = create_test_context();
            let mut reader = Cursor::new("99\n");
            resume_with_reader(&mut ctx, &mut reader).unwrap();

            let err = String::from_utf8(ctx.err.into_inner()).unwrap();
            test_contains!(err, "Invalid selection.");
        });
    }
}
