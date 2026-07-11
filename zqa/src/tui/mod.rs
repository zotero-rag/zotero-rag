//! A terminal UI (TUI) alternative to the readline-based CLI.
//!
//! The TUI reuses the CLI command handlers unchanged: a worker task owns the
//! [`crate::common::Context`] and runs [`crate::cli::app::dispatch_command`], with the context's
//! output streams replaced by [`ChannelWriter`]s so all handler output, including streamed model
//! answers, flows to the UI event loop as [`WorkerEvent`]s. The one exception is `/resume`,
//! which swaps in a saved conversation without printing it; the worker detects the swap and
//! replays the loaded history into the transcript.
//!
//! The layout is a borderless scrollable transcript, a bordered query box with command
//! suggestions below it, and a sidebar with session information (cost, message count, and
//! imported documents).

#![allow(clippy::cast_possible_truncation)]

mod app;
mod editor;
mod writer;

use std::io::{Write, stdout};
use std::sync::Arc;
use std::time::Duration;

use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{DisableMouseCapture, EnableMouseCapture, Event};
use ratatui::crossterm::execute;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use zqa_rag::llm::base::{ChatHistoryContent, MessageRole};

use crate::cli::app::dispatch_command;
use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::tui::app::App;
pub(crate) use crate::tui::writer::ChannelWriter;
use crate::tui::writer::{SidebarSnapshot, WorkerEvent};

/// How often the UI redraws without input. This animates the spinner and repaints over any
/// stray writes to the real terminal (e.g., progress bars that bypass the context streams).
const TICK_INTERVAL: Duration = Duration::from_millis(150);

/// Run the TUI until the user quits.
///
/// # Arguments
///
/// * `ctx` - The application context. Its output streams must be [`ChannelWriter`]s created
///   from `worker_tx` so that handler output reaches the UI.
/// * `worker_tx` - The sender for worker events; used by the command worker itself.
/// * `worker_rx` - The receiving end drained by the UI event loop.
///
/// # Errors
///
/// * `CLIError::IOError` - If the terminal could not be initialized or drawing fails.
pub(crate) async fn run(
    ctx: Context<ChannelWriter, ChannelWriter>,
    worker_tx: UnboundedSender<WorkerEvent>,
    worker_rx: UnboundedReceiver<WorkerEvent>,
) -> Result<(), CLIError> {
    let vi = matches!(
        crate::cli::readline::get_edit_mode(),
        rustyline::EditMode::Vi
    );

    let mut terminal = ratatui::try_init()?;
    let _ = execute!(stdout(), EnableMouseCapture);
    crate::utils::terminal::set_cli_mode(false);

    let result = event_loop(&mut terminal, ctx, worker_tx, worker_rx, vi).await;

    crate::utils::terminal::set_cli_mode(true);
    let _ = execute!(stdout(), DisableMouseCapture);
    ratatui::restore();
    result
}

/// Spawn the command worker and the input-reading thread, then drive the UI.
async fn event_loop(
    terminal: &mut DefaultTerminal,
    ctx: Context<ChannelWriter, ChannelWriter>,
    worker_tx: UnboundedSender<WorkerEvent>,
    worker_rx: UnboundedReceiver<WorkerEvent>,
    vi: bool,
) -> Result<(), CLIError> {
    let (cmd_tx, cmd_rx) = unbounded_channel::<String>();
    let (event_tx, event_rx) = unbounded_channel::<Event>();

    // Crossterm's blocking `read` runs on a plain thread; the process exiting reclaims it.
    std::thread::spawn(move || {
        while let Ok(event) = ratatui::crossterm::event::read() {
            if event_tx.send(event).is_err() {
                break;
            }
        }
    });

    // The worker must run on a runtime worker thread (not a `LocalSet` on the main thread):
    // the embedding clients bridge into sync lancedb traits via `tokio::task::block_in_place`,
    // which panics anywhere else.
    tokio::spawn(worker(ctx, cmd_rx, worker_tx));

    ui_loop(terminal, App::new(vi), event_rx, worker_rx, &cmd_tx).await
}

/// The UI event loop: draw, then react to terminal events, worker events, and ticks.
async fn ui_loop(
    terminal: &mut DefaultTerminal,
    mut app: App,
    mut event_rx: UnboundedReceiver<Event>,
    mut worker_rx: UnboundedReceiver<WorkerEvent>,
    cmd_tx: &UnboundedSender<String>,
) -> Result<(), CLIError> {
    let mut tick = tokio::time::interval(TICK_INTERVAL);

    loop {
        terminal.draw(|frame| app.render(frame))?;

        tokio::select! {
            event = event_rx.recv() => match event {
                None => break,
                Some(Event::Key(key)) => {
                    if let Some(command) = app.on_key(key) {
                        let _ = cmd_tx.send(command);
                    }
                    if app.force_quit {
                        break;
                    }
                }
                Some(Event::Mouse(mouse)) => app.on_mouse(mouse),
                Some(_) => {}
            },
            event = worker_rx.recv() => match event {
                None => break,
                Some(event) => {
                    app.on_worker_event(event);
                    if app.should_quit {
                        break;
                    }
                }
            },
            _ = tick.tick() => app.tick(),
        }
    }

    Ok(())
}

/// The command worker: owns the context and dispatches submitted lines to the CLI handlers.
async fn worker(
    mut ctx: Context<ChannelWriter, ChannelWriter>,
    mut cmd_rx: UnboundedReceiver<String>,
    tx: UnboundedSender<WorkerEvent>,
) {
    let _ = tx.send(WorkerEvent::Sidebar(snapshot(&ctx)));

    while let Some(command) = cmd_rx.recv().await {
        let history_before = Arc::clone(&ctx.state.chat_history);
        let result = dispatch_command(&command, &mut ctx).await;

        // Handlers stream all output, including model answers, through the context's
        // writers, so the transcript is already up to date. The exception is a swapped-in
        // conversation (`/resume` replaces the chat history `Arc` wholesale): its contents
        // were never printed, so replay them. `/new` also swaps the `Arc`, but to an empty
        // history, making the replay a no-op.
        if !Arc::ptr_eq(&history_before, &ctx.state.chat_history) {
            replay_history(&ctx, &tx);
        }

        let (should_continue, error) = match result {
            Ok(cont) => (cont, None),
            // Unlike the CLI REPL, the TUI stays alive on non-command errors; there is no
            // outer shell to fall back to, so surfacing the error is more useful than dying.
            Err(e) => (true, Some(e.to_string())),
        };

        let _ = tx.send(WorkerEvent::Sidebar(snapshot(&ctx)));
        let _ = tx.send(WorkerEvent::Done {
            should_continue,
            error,
        });

        if !should_continue {
            break;
        }
    }
}

/// Send the whole chat history to the UI, e.g. after `/resume` loads a saved conversation.
fn replay_history<O: Write, E: Write>(ctx: &Context<O, E>, tx: &UnboundedSender<WorkerEvent>) {
    let Ok(history) = ctx.state.chat_history.lock() else {
        return;
    };

    for item in history.iter() {
        let text = item
            .content
            .iter()
            .filter_map(|content| match content {
                ChatHistoryContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if text.is_empty() {
            continue;
        }

        let is_user = match item.role {
            MessageRole::User => true,
            MessageRole::Assistant => false,
            MessageRole::Tool => continue,
        };

        let _ = tx.send(WorkerEvent::HistoryItem { is_user, text });
    }
}

/// The current chat history length, or 0 if the lock is poisoned.
fn history_len<O: Write, E: Write>(ctx: &Context<O, E>) -> usize {
    ctx.state
        .chat_history
        .lock()
        .map_or(0, |history| history.len())
}

/// Collect the sidebar data from the context.
fn snapshot<O: Write, E: Write>(ctx: &Context<O, E>) -> SidebarSnapshot {
    SidebarSnapshot {
        title: ctx.state.title.lock().ok().and_then(|title| title.clone()),
        model: ctx.config.get_generation_model_name(),
        session_cost_cents: ctx
            .state
            .session_cost
            .load(std::sync::atomic::Ordering::Relaxed),
        message_count: history_len(ctx),
        documents: ctx
            .state
            .imports
            .read()
            .map(|imports| imports.keys().cloned().collect())
            .unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, MessageRole};

    use super::{WorkerEvent, replay_history};
    use crate::cli::app::tests::create_test_context;

    #[test]
    fn test_replay_history_forwards_conversation() {
        let ctx = create_test_context(vec![]);
        {
            let mut history = ctx.state.chat_history.lock().unwrap();
            history.push(ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::Text("What is attention?".into())],
            });
            history.push(ChatHistoryItem {
                role: MessageRole::Assistant,
                content: vec![ChatHistoryContent::Text("A weighting mechanism.".into())],
            });
            // Items with no text content should not produce transcript lines.
            history.push(ChatHistoryItem {
                role: MessageRole::Assistant,
                content: vec![],
            });
        }

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        replay_history(&ctx, &tx);
        drop(tx);

        let mut received = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                WorkerEvent::HistoryItem { is_user, text } => received.push((is_user, text)),
                _ => panic!("unexpected worker event"),
            }
        }

        assert_eq!(
            received,
            vec![
                (true, "What is attention?".to_string()),
                (false, "A weighting mechanism.".to_string()),
            ]
        );
    }
}
