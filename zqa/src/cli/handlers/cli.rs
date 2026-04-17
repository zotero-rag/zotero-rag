use std::{
    io::Write,
    sync::{Arc, Mutex, atomic},
};

use crate::{
    cli::{errors::CLIError, handlers::conversation::save_current_conversation},
    common::Context,
};

pub(crate) fn handle_quit_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    save_current_conversation(ctx)
}

pub(crate) fn handle_config_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(&mut ctx.out, "{}", ctx.config)?;

    Ok(())
}

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

pub(crate) fn handle_help_cmd<O, E>(ctx: &mut Context<O, E>) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(&mut ctx.out)?;
    writeln!(&mut ctx.out, "Available commands:\n")?;
    writeln!(&mut ctx.out, "/help\t\tShow this help message")?;
    writeln!(
        &mut ctx.out,
        "/process\tPre-process Zotero library. Use to update the database."
    )?;
    writeln!(
        &mut ctx.out,
        "/docs clear\tClear all documents in this session."
    )?;
    writeln!(
        &mut ctx.out,
        "/docs list\tList all documents in this session."
    )?;
    writeln!(
        &mut ctx.out,
        "/docs remove <key>\tRemove a document with a specified key from the session."
    )?;
    writeln!(
        &mut ctx.out,
        "/embed\t\tRepair failed DB creation by re-adding embeddings."
    )?;
    writeln!(
        &mut ctx.out,
        "/search\t\tSearch for papers without summarizing them. Usage: /search <query>"
    )?;
    writeln!(
        &mut ctx.out,
        "/config\t\tShow the currently used configuration."
    )?;
    writeln!(
        &mut ctx.out,
        "/new\t\tSave the current conversation and switch to a new one."
    )?;
    writeln!(&mut ctx.out, "/resume\t\tResume a previous conversation.")?;
    writeln!(&mut ctx.out, "/index\t\tCreate or update indices.")?;
    writeln!(
        &mut ctx.out,
        "/checkhealth\tRun health checks on your LanceDB."
    )?;
    writeln!(
        &mut ctx.out,
        "/doctor\t\tAttempt to fix issues spotted by /checkhealth."
    )?;
    writeln!(&mut ctx.out, "/stats\t\tShow table statistics.")?;
    writeln!(&mut ctx.out, "/dedup\t\tRemove duplicate items.")?;
    writeln!(&mut ctx.out, "/quit\t\tExit the program")?;
    writeln!(&mut ctx.out)?;
    Ok(())
}
