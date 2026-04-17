//! Command handlers for conversation-related operations.

use std::{
    io::{self, BufRead, Write},
    sync::{Arc, Mutex, atomic},
};

use chrono::Local;

use crate::{
    cli::{commands::Command, errors::CLIError},
    common::Context,
    state::{SavedChatHistory, get_conversation_history, save_conversation},
};

/// Resume a previous conversation selected by the user.
///
/// Displays a numbered list of saved conversations, prompts for a selection, and loads the
/// chosen conversation into the current session. If the current session is dirty, it is saved
/// first.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
/// * `reader` - A buffered reader used to read the user's selection.
///
/// # Errors
///
/// * `CLIError::IOError` - If `writeln!` or reading input fails.
/// * `CLIError::LockPoisoningError` - If a lock on the context chat history could not be
///   obtained.
pub(crate) async fn handle_resume_cmd<O: Write, E: Write>(
    cmd: Command,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    if !matches!(cmd, Command::Resume) {
        return Err(CLIError::CommandError(
            "resume command expected".to_string(),
        ));
    }

    let stdin = io::stdin();
    let mut reader = stdin.lock();

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
