//! Command handlers for CLI-related operations.

use std::{
    io::Write,
    sync::{Arc, Mutex, atomic},
};

use crate::{
    cli::{errors::CLIError, handlers::conversation::save_current_conversation},
    common::Context,
};

/// Save the current conversation, if needed, and prepare to exit the CLI.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the conversation was saved successfully or no save was needed.
///
/// # Errors
///
/// Returns a [`CLIError`] if conversation state could not be persisted.
pub(crate) fn handle_quit_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    save_current_conversation(ctx)
}

/// Print the active CLI configuration.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the configuration was written successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if writing to the output stream fails.
pub(crate) fn handle_config_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(&mut ctx.out, "{}", ctx.config)?;

    Ok(())
}

/// Save the current conversation and reset in-memory conversation state.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the current conversation was saved and state was reset.
///
/// # Errors
///
/// Returns a [`CLIError`] if conversation state could not be persisted.
pub(crate) fn handle_new_conversation_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    save_current_conversation(ctx)?;

    ctx.state.dirty.store(false, atomic::Ordering::Relaxed);
    ctx.state.chat_history = Arc::new(Mutex::new(Vec::new()));
    ctx.state.title = Arc::new(Mutex::new(None));

    Ok(())
}

/// Print the CLI help text.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the help text was written successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if writing to the output stream fails.
pub(crate) fn handle_help_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(&mut ctx.out, "Basic usage:")?;
    writeln!(
        &mut ctx.out,
        "- If you haven't already done so, you should run `/process` or `/batch create` to set up an embedding database."
    )?;
    writeln!(
        &mut ctx.out,
        "- Type in a question to ask your configured model, grounded in your Zotero library."
    )?;
    writeln!(
        &mut ctx.out,
        "- Use @ to include a PDF file in your current directory in the conversation."
    )?;
    writeln!(&mut ctx.out)?;
    writeln!(&mut ctx.out, "Available commands:\n")?;
    writeln!(&mut ctx.out, "/help\t\t\tShow this help message")?;
    writeln!(&mut ctx.out)?;
    writeln!(&mut ctx.out, "Common commands:")?;
    writeln!(
        &mut ctx.out,
        "/process\t\tPre-process Zotero library. Use this to update the database."
    )?;
    writeln!(
        &mut ctx.out,
        "/search\t\t\tSearch for papers without summarizing them. Usage: /search <query>"
    )?;
    writeln!(
        &mut ctx.out,
        "/config\t\t\tShow the currently used configuration."
    )?;
    writeln!(
        &mut ctx.out,
        "/new\t\t\tSave the current conversation and switch to a new one."
    )?;
    writeln!(&mut ctx.out, "/resume\t\t\tResume a previous conversation.")?;
    writeln!(&mut ctx.out, "/index\t\t\tCreate or update indices.")?;
    writeln!(
        &mut ctx.out,
        "/quit\t\t\tExit the program. You can also use Ctrl+C or just type 'quit'."
    )?;
    writeln!(&mut ctx.out)?;
    writeln!(&mut ctx.out, "Batch API commands:")?;
    writeln!(
        &mut ctx.out,
        "/batch create\t\tPre-process Zotero library, but use a batch embedding API instead."
    )?;
    writeln!(
        &mut ctx.out,
        "/batch check\t\tCheck on the status of a submitted batch."
    )?;
    writeln!(
        &mut ctx.out,
        "/batch cancel <id>\t\tCancel a pending batch."
    )?;
    writeln!(&mut ctx.out)?;
    writeln!(&mut ctx.out, "Session document commands:")?;
    writeln!(
        &mut ctx.out,
        "/docs clear\t\tClear all documents in this session."
    )?;
    writeln!(
        &mut ctx.out,
        "/docs list\t\tList all documents in this session."
    )?;
    writeln!(
        &mut ctx.out,
        "/docs remove <key>\tRemove a document with a specified key from the session."
    )?;
    writeln!(&mut ctx.out)?;
    writeln!(&mut ctx.out, "Repair and troubleshooting commands:")?;
    writeln!(
        &mut ctx.out,
        "/embed\t\t\tRepair failed DB creation by re-adding embeddings."
    )?;
    writeln!(
        &mut ctx.out,
        "/checkhealth\t\tRun health checks on your LanceDB."
    )?;
    writeln!(
        &mut ctx.out,
        "/doctor\t\t\tAttempt to fix issues spotted by /checkhealth."
    )?;
    writeln!(&mut ctx.out, "/stats\t\t\tShow table statistics.")?;
    writeln!(&mut ctx.out, "/dedup\t\t\tRemove duplicate items.")?;
    writeln!(&mut ctx.out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use zqa_macros::test_contains;

    use super::handle_help_cmd;
    use crate::cli::app::tests::create_test_context;

    #[test]
    fn test_handle_help_cmd() {
        let mut ctx = create_test_context();
        handle_help_cmd(&mut ctx).unwrap();
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
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
}
