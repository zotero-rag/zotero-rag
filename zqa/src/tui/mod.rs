//! A full-screen terminal UI for zqa, launched with `--tui`.
//!
//! The TUI reuses the CLI's command handlers unchanged. It is split across two threads:
//!
//! * The *UI thread* owns the terminal. It renders the conversation transcript, a sidebar with
//!   session statistics, a query box, and a suggestion list, and it translates key and mouse
//!   events into edits or [`app::UserAction`]s.
//! * The *command runner* (the calling async task) receives submitted lines and feeds them to
//!   [`crate::cli::app::dispatch_command`], exactly like the CLI's REPL loop does.
//!
//! The two sides communicate over channels; see [`io`] for the wiring. Handler output arrives
//! as [`io::UiMessage`]s, and interactive prompts (such as `/resume`) block the runner until
//! the UI thread supplies a reply line.

pub(crate) mod app;
pub(crate) mod input;
pub(crate) mod io;
pub(crate) mod suggest;
pub(crate) mod transcript;
pub(crate) mod ui;

pub(crate) use app::run_tui;
