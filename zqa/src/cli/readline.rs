use std::{env, fs, path::PathBuf};

use rustyline::{Config, EditMode};

pub fn get_readline_config() -> Config {
    Config::builder().edit_mode(get_edit_mode()).build()
}

/// Get the edit mode (emacs or vi) from either the readline (GNU) or editline (BSD) configs. The
/// preference is:
///
/// GNU readline: INPUTRC > ~/.inputrc > /etc/inputrc
/// BSD editline: EDITRC > ./.editrc > ~/.editrc
///
/// # Returns
///
/// The current edit mode (emacs or vi)
fn get_edit_mode() -> EditMode {
    match env::consts::OS {
        "windows" => EditMode::Emacs, // No standard on Windows
        "macos" => match get_editrc_edit_mode() {
            Some(mode) => mode,
            _ => get_inputrc_edit_mode(),
        },
        "linux" => get_inputrc_edit_mode(),
        _ => EditMode::Emacs,
    }
}

/// Attempt to find an editline config by priority order.
///
/// # Returns
///
/// * If a config file can be found, a path to it
/// * Otherwise, `None`
fn find_editrc() -> Option<PathBuf> {
    let cwd = env::current_dir().ok()?;
    if cwd.join(".editrc").exists() {
        return Some(cwd.join(".editrc"));
    }

    let home_dir = env::home_dir()?;
    if home_dir.join(".editrc").exists() {
        return Some(home_dir.join(".editrc"));
    }

    None
}

/// Attempt to get the edit mode from the editline config.
///
/// # Returns
///
/// * If either EDITRC or an editline config is found, then the edit mode that is set (or emacs if
///   unset)
/// * Otherwise, `None`
fn get_editrc_edit_mode() -> Option<EditMode> {
    let path = env::var("EDITRC")
        .map(PathBuf::from)
        .ok()
        .or_else(find_editrc);

    match path {
        None => None,
        Some(editrc_path) => match fs::read_to_string(editrc_path) {
            Ok(contents) => {
                let has_vi = contents
                    .lines()
                    .filter(|line| !line.starts_with('#'))
                    .map(|line| line.trim())
                    .filter(|line| *line == "bind -v")
                    .count()
                    > 0;

                if has_vi {
                    return Some(EditMode::Vi);
                }
                Some(EditMode::Emacs)
            }
            _ => None,
        },
    }
}

/// Attempt to find an readline config by priority order.
///
/// # Returns
///
/// * If a config file can be found, a path to it
/// * Otherwise, `None`
fn find_inputrc() -> Option<PathBuf> {
    if let Ok(mode) = env::var("EDITRC") {
        return Some(PathBuf::from(mode));
    }

    let home_dir = env::home_dir()?;
    if home_dir.join(".inputrc").exists() {
        return Some(home_dir.join(".inputrc"));
    }

    if PathBuf::from("/etc/inputrc").exists() {
        return Some(PathBuf::from("/etc/inputrc"));
    }

    None
}

/// Attempt to get the edit mode from the readline config.
///
/// From `man 3 readline`:
/// The name of this file is taken from the value of the INPUTRC environment
/// variable.  If that variable is unset, the default is ~/.inputrc.  If that file  does not exist
/// or cannot be read, readline looks for /etc/inputrc.
///
/// # Returns
///
/// * If either INPUTRC or a readline config is found, then the edit mode that is set (or emacs if
///   unset)
/// * Otherwise, `None`
fn get_inputrc_edit_mode() -> EditMode {
    let path = env::var("INPUTRC")
        .map(PathBuf::from)
        .ok()
        .or_else(find_inputrc);

    match path {
        None => EditMode::Emacs,
        Some(inputrc_path) => match fs::read_to_string(inputrc_path) {
            Ok(contents) => {
                let has_vi = contents
                    .lines()
                    .filter(|line| !line.starts_with('#'))
                    .map(|line| line.trim())
                    .filter(|line| *line == "set editing-mode vi" || *line == "set -o vi")
                    .count()
                    > 0;

                if has_vi {
                    return EditMode::Vi;
                }
                EditMode::Emacs
            }
            _ => EditMode::Emacs,
        },
    }
}
