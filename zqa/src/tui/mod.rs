//! A terminal UI (TUI) alternative to the readline-based CLI.
//!
//! The TUI reuses the CLI command handlers unchanged: a worker task owns the
//! [`crate::common::Context`] and runs [`crate::cli::app::dispatch_command`], with the context's
//! output streams replaced by [`ChannelWriter`]s so all handler output flows to the UI event
//! loop as [`WorkerEvent`]s. Model answers, which handlers record in the chat history rather
//! than writing to stdout, are synced into the transcript after each command.
//!
//! The layout is a borderless scrollable transcript, a bordered query box with command
//! suggestions below it, and a sidebar with session information (cost, message count, and
//! imported documents).

#![allow(clippy::cast_possible_truncation)]

mod app;
mod editor;
mod writer;

use std::io::{Write, stdout};
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
    crate::utils::terminal::set_progress_bars_enabled(false);

    let result = event_loop(&mut terminal, ctx, worker_tx, worker_rx, vi).await;

    crate::utils::terminal::set_progress_bars_enabled(true);
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
    let mut seen_history_items = history_len(&ctx);
    let _ = tx.send(WorkerEvent::Sidebar(snapshot(&ctx)));

    while let Some(command) = cmd_rx.recv().await {
        let result = dispatch_command(&command, &mut ctx).await;

        // These commands replace the chat history wholesale, so the whole (new) history is
        // fresh from the transcript's point of view.
        if matches!(command.trim(), "/new" | "/resume") {
            seen_history_items = 0;
        }
        seen_history_items = sync_history(&ctx, &tx, seen_history_items, &command);

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

/// Send chat history items past `seen` to the UI, returning the new high-water mark.
///
/// The user message for the command just dispatched is skipped, since the UI already echoed
/// it; other user messages (e.g., from a resumed conversation) are forwarded.
fn sync_history<O: Write, E: Write>(
    ctx: &Context<O, E>,
    tx: &UnboundedSender<WorkerEvent>,
    seen: usize,
    command: &str,
) -> usize {
    let Ok(history) = ctx.state.chat_history.lock() else {
        return seen;
    };

    let seen = seen.min(history.len());
    for item in history.iter().skip(seen) {
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
        if is_user && text == command.trim() {
            continue;
        }

        let _ = tx.send(WorkerEvent::HistoryItem { is_user, text });
    }

    history.len()
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
