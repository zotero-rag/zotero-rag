//! Basic full-screen terminal interface.

mod app;
mod editor;
mod io;
mod terminal;

use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::Ordering,
    thread,
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender, TryRecvError, unbounded};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};
use rustyline::history::{DefaultHistory, History};
use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, MessageRole};

use crate::{
    cli::{app::dispatch_command, readline::get_edit_mode},
    common::{Context, InterfaceMode},
    utils::library::set_terminal_output_hidden,
};

use self::{
    app::{App, render},
    editor::Editor,
    io::{ChannelInput, ChannelWriter, WorkerEvent},
    terminal::TerminalGuard,
};

struct ParserOutputGuard;

impl ParserOutputGuard {
    fn hide() -> Self {
        set_terminal_output_hidden(true);
        Self
    }
}

impl Drop for ParserOutputGuard {
    fn drop(&mut self) {
        set_terminal_output_hidden(false);
    }
}

/// Run the full-screen terminal interface.
///
/// # Arguments
///
/// * `context` - Fully initialized application context.
///
/// # Errors
///
/// * Returns terminal setup, event, rendering, history, or worker errors.
pub(crate) fn run<O, E>(context: Context<O, E>) -> Result<(), Box<dyn std::error::Error>>
where
    O: Write + Send + 'static,
    E: Write + Send + 'static,
{
    let _parser_output_guard = ParserOutputGuard::hide();
    let history_path = history_path()?;
    let history = load_history(&history_path);
    let model = context
        .config
        .get_generation_model_name()
        .unwrap_or_else(|| "not configured".into());
    let (command_sender, command_receiver) = unbounded();
    let (input_sender, input_receiver) = unbounded();
    let (event_sender, event_receiver) = unbounded();
    let worker = spawn_worker(context, command_receiver, input_receiver, event_sender);

    let mut terminal = TerminalGuard::enter()?;
    let mut app = App::new(Editor::new(get_edit_mode(), history));
    let mut transcript_area = ratatui::layout::Rect::default();
    let mut should_exit = false;

    while !should_exit {
        drain_worker_events(&event_receiver, &command_sender, &mut app, &mut should_exit);
        terminal
            .terminal()
            .draw(|frame| transcript_area = render(frame, &mut app, &model))?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') {
                        if app.flow.prompt_pending {
                            let _ = input_sender.send(None);
                            app.flow.prompt_pending = false;
                            app.flow.quit_after_completion = true;
                            app.status = "Cancelling command...".into();
                        } else if app.flow.busy {
                            app.flow.quit_after_completion = true;
                            app.status = "Will quit when the command completes".into();
                        } else {
                            submit("/quit".into(), &command_sender, &mut app);
                        }
                    } else if key.modifiers == KeyModifiers::CONTROL
                        && key.code == KeyCode::Char('u')
                    {
                        app.scroll_half_page(false);
                    } else if key.modifiers == KeyModifiers::CONTROL
                        && key.code == KeyCode::Char('d')
                    {
                        app.scroll_half_page(true);
                    } else {
                        handle_key(key, &command_sender, &input_sender, &mut app);
                    }
                }
                Event::Mouse(mouse)
                    if App::transcript_contains(mouse.column, mouse.row, transcript_area) =>
                {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => app.scroll_half_page(false),
                        MouseEventKind::ScrollDown => app.scroll_half_page(true),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    save_history(&history_path, &app)?;
    drop(command_sender);
    drop(input_sender);
    worker.join().map_err(|_| "TUI command worker panicked")?;
    Ok(())
}

fn handle_key(
    key: crossterm::event::KeyEvent,
    commands: &Sender<String>,
    input: &Sender<Option<String>>,
    app: &mut App,
) {
    let has_suggestions = !app.suggestions().is_empty();
    match key.code {
        KeyCode::Enter => {
            let text = app.editor.take();
            if app.flow.prompt_pending {
                let _ = input.send(Some(text));
                app.flow.prompt_pending = false;
                app.status = "Working...".into();
            } else if !text.trim().is_empty() {
                if app.flow.busy {
                    app.editor.set_text(&text);
                } else {
                    app.editor.remember(&text);
                    app.submitted_history.push(text.clone());
                    submit(text, commands, app);
                }
            }
        }
        KeyCode::Tab if has_suggestions => app.accept_suggestion(),
        KeyCode::Up if has_suggestions => app.move_suggestion(false),
        KeyCode::Down if has_suggestions => app.move_suggestion(true),
        _ if !app.flow.busy || app.flow.prompt_pending => {
            app.editor.handle_key(key);
            app.reset_suggestion();
        }
        _ => {}
    }
}

fn submit(command: String, sender: &Sender<String>, app: &mut App) {
    app.append(&format!("\n> {command}\n"));
    if sender.send(command).is_ok() {
        app.flow.busy = true;
        app.status = "Working...".into();
    }
}

fn spawn_worker<O, E>(
    context: Context<O, E>,
    commands: Receiver<String>,
    input: Receiver<Option<String>>,
    events: Sender<WorkerEvent>,
) -> thread::JoinHandle<()>
where
    O: Write + Send + 'static,
    E: Write + Send + 'static,
{
    thread::spawn(move || {
        let Context {
            state,
            config,
            store,
            ..
        } = context;
        let stream_events = events.clone();
        let stream_output = std::sync::Arc::new(move |text: &str| {
            let _ = stream_events.send(WorkerEvent::Stream(format!("{text}\n")));
        });
        let mut context = Context {
            state,
            config,
            store,
            input: Box::new(ChannelInput::new(input, events.clone())),
            out: ChannelWriter::new(events.clone(), false),
            err: ChannelWriter::new(events.clone(), true),
            stream_output,
            interface_mode: InterfaceMode::Tui,
        };
        let runtime = match build_worker_runtime() {
            Ok(runtime) => runtime,
            Err(error) => {
                let _ = events.send(WorkerEvent::Error(format!(
                    "Failed to start command runtime: {error}"
                )));
                return;
            }
        };
        while let Ok(command) = commands.recv() {
            let result = runtime.block_on(dispatch_command(&command, &mut context));
            let should_continue = result.as_ref().copied().unwrap_or(true);
            let error = result.err().map(|error| error.to_string());
            let title = context
                .state
                .title
                .lock()
                .ok()
                .and_then(|title| title.clone());
            let (message_count, conversation) = context.state.chat_history.lock().map_or_else(
                |_| (0, None),
                |history| {
                    let conversation = match command.trim() {
                        "/new" => Some("Started a new conversation.\n".into()),
                        "/resume" if title.is_some() => Some(format_chat_history(&history)),
                        _ => None,
                    };
                    (history.len(), conversation)
                },
            );
            let mut documents = context.state.imports.read().map_or_else(
                |_| Vec::new(),
                |documents| documents.keys().cloned().collect(),
            );
            documents.sort_unstable();
            let _ = events.send(WorkerEvent::Finished {
                should_continue,
                error,
                input_tokens: context.state.input_tokens.load(Ordering::Relaxed),
                output_tokens: context.state.output_tokens.load(Ordering::Relaxed),
                session_cost: context.state.session_cost.load(Ordering::Relaxed),
                message_count,
                title,
                documents,
                conversation,
            });
            if !should_continue {
                break;
            }
        }
    })
}

fn format_chat_history(history: &[ChatHistoryItem]) -> String {
    let mut transcript = String::new();
    for item in history {
        let prefix = match item.role {
            MessageRole::User => "> ",
            MessageRole::Assistant => "",
            MessageRole::Tool => continue,
        };
        for content in &item.content {
            if let ChatHistoryContent::Text(text) = content {
                transcript.push_str(prefix);
                transcript.push_str(text);
                transcript.push_str("\n\n");
            }
        }
    }
    transcript
}

fn build_worker_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
}

fn drain_worker_events(
    events: &Receiver<WorkerEvent>,
    commands: &Sender<String>,
    app: &mut App,
    should_exit: &mut bool,
) {
    loop {
        let event = match events.try_recv() {
            Ok(event) => event,
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                if app.flow.busy && !*should_exit {
                    app.append("\nError: command worker stopped unexpectedly.\n");
                    *should_exit = true;
                }
                break;
            }
        };
        match event {
            WorkerEvent::Output(text) | WorkerEvent::Stderr(text) | WorkerEvent::Stream(text) => {
                app.append(&text);
            }
            WorkerEvent::Error(text) => app.append(&format!("\nError: {text}\n")),
            WorkerEvent::PromptPending => {
                app.flow.prompt_pending = true;
                app.status = "Input requested by command".into();
            }
            WorkerEvent::Finished {
                should_continue,
                error,
                input_tokens,
                output_tokens,
                session_cost,
                message_count,
                title,
                documents,
                conversation,
            } => {
                app.flow.busy = false;
                app.flow.prompt_pending = false;
                app.stats.input_tokens = input_tokens;
                app.stats.output_tokens = output_tokens;
                app.stats.session_cost = session_cost;
                app.stats.message_count = message_count;
                app.stats.title = title;
                app.stats.documents = documents;
                if let Some(conversation) = conversation {
                    app.replace_transcript(&conversation);
                }
                if let Some(error) = error {
                    app.append(&format!("\nError: {error}\n"));
                }
                if !should_continue {
                    *should_exit = true;
                } else if app.flow.quit_after_completion {
                    submit("/quit".into(), commands, app);
                } else {
                    app.status = "Ready".into();
                }
            }
        }
    }
}

fn history_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let directories = directories::UserDirs::new().ok_or("Could not get user directories")?;
    Ok(directories.home_dir().join(".zqa_history"))
}

fn load_history(path: &Path) -> Vec<String> {
    let mut history = DefaultHistory::new();
    if history.load(path).is_err() {
        return Vec::new();
    }
    history.iter().cloned().collect()
}

fn save_history(path: &Path, app: &App) -> std::io::Result<()> {
    let mut history = DefaultHistory::new();
    let _ = history.load(path);
    for line in &app.submitted_history {
        history
            .add(line)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
    }
    history
        .save(path)
        .map_err(|error| std::io::Error::other(error.to_string()))
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::unbounded;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use rustyline::EditMode;

    use super::{App, Editor, build_worker_runtime, handle_key, load_history, save_history};

    #[test]
    fn worker_runtime_supports_block_in_place() {
        let runtime = build_worker_runtime().unwrap();
        let value = runtime.block_on(async { tokio::task::block_in_place(|| 42) });
        assert_eq!(value, 42);
    }

    #[test]
    fn empty_submission_answers_pending_prompt() {
        let (command_sender, _command_receiver) = unbounded();
        let (input_sender, input_receiver) = unbounded();
        let mut app = App::new(Editor::new(EditMode::Emacs, Vec::new()));
        app.flow.prompt_pending = true;

        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &command_sender,
            &input_sender,
            &mut app,
        );

        assert_eq!(input_receiver.recv().unwrap(), Some(String::new()));
        assert!(!app.flow.prompt_pending);
    }

    #[test]
    fn history_round_trip_uses_rustyline_encoding() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("history");
        let mut app = App::new(Editor::new(EditMode::Emacs, Vec::new()));
        app.submitted_history = vec![r"Compare C:\new\paper.pdf".into()];

        save_history(&path, &app).unwrap();

        assert_eq!(load_history(&path), app.submitted_history);
    }
}
