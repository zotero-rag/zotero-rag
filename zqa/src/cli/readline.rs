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
    if let Ok(path) = env::var("EDITRC").map(PathBuf::from) {
        return Some(path);
    }

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
    let path = find_editrc();

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
    if let Ok(path) = env::var("INPUTRC").map(PathBuf::from) {
        return Some(path);
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
    let path = find_inputrc();

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

#[cfg(test)]
mod tests {
    use std::fs::{self, File};

    use rustyline::EditMode;
    use serial_test::serial;
    use temp_env;

    use crate::cli::readline::{
        find_editrc, find_inputrc, get_editrc_edit_mode, get_inputrc_edit_mode,
    };

    #[test]
    #[serial]
    fn test_find_editrc() {
        File::create(".editrc").unwrap();

        let path = find_editrc();
        assert!(path.is_some());

        let path = path.unwrap();
        let filename = path.file_name();
        assert!(filename.is_some());

        let filename = filename.unwrap();
        assert_eq!(filename.to_str().unwrap(), ".editrc");

        fs::remove_file(".editrc").unwrap();
    }

    #[test]
    #[serial]
    fn test_find_editrc_in_home() {
        let home_dir = std::env::home_dir();
        assert!(home_dir.is_some());

        let home_dir = home_dir.unwrap();
        File::create(home_dir.join(".editrc")).unwrap();

        let path = find_editrc();
        assert!(path.is_some());

        let path = path.unwrap();
        let filename = path.file_name();
        assert!(filename.is_some());

        let filename = filename.unwrap();
        assert_eq!(filename.to_str().unwrap(), ".editrc");

        fs::remove_file(home_dir.join(".editrc")).unwrap();
    }

    #[test]
    #[serial]
    fn test_find_editrc_prioritizes_env() {
        let home_dir = std::env::home_dir();
        assert!(home_dir.is_some());

        let home_dir = home_dir.unwrap();
        File::create(home_dir.join(".editrc")).unwrap();

        let path = temp_env::with_var("EDITRC", Some("foo"), find_editrc);
        assert!(path.is_some());

        let path = path.unwrap();
        let filename = path.file_name();
        assert!(filename.is_some());

        let filename = filename.unwrap();
        assert_eq!(filename.to_str().unwrap(), "foo");

        fs::remove_file(home_dir.join(".editrc")).unwrap();
    }

    #[test]
    #[serial]
    fn test_find_inputrc_in_home() {
        let home_dir = std::env::home_dir();
        assert!(home_dir.is_some());

        let home_dir = home_dir.unwrap();
        File::create(home_dir.join(".inputrc")).unwrap();

        let path = find_inputrc();
        assert!(path.is_some());

        let path = path.unwrap();
        let filename = path.file_name();
        assert!(filename.is_some());

        let filename = filename.unwrap();
        assert_eq!(filename.to_str().unwrap(), ".inputrc");

        fs::remove_file(home_dir.join(".inputrc")).unwrap();
    }

    #[test]
    #[serial]
    fn test_find_inputrc_prioritizes_env() {
        let home_dir = std::env::home_dir();
        assert!(home_dir.is_some());

        let home_dir = home_dir.unwrap();
        File::create(home_dir.join(".inputrc")).unwrap();

        let path = temp_env::with_var("INPUTRC", Some("foo"), find_inputrc);
        assert!(path.is_some());

        let path = path.unwrap();
        let filename = path.file_name();
        assert!(filename.is_some());

        let filename = filename.unwrap();
        assert_eq!(filename.to_str().unwrap(), "foo");

        fs::remove_file(home_dir.join(".inputrc")).unwrap();
    }

    #[test]
    #[serial]
    fn test_get_editrc_edit_mode() {
        let no_config = get_editrc_edit_mode();
        assert!(no_config.is_none());

        fs::write(".editrc", "bind -v").unwrap();
        let with_vi_config = get_editrc_edit_mode();
        assert!(with_vi_config.is_some_and(|m| m == EditMode::Vi));
        fs::remove_file(".editrc").unwrap();

        fs::write(".editrc", "foo").unwrap();
        let with_emacs_config = get_editrc_edit_mode();
        assert!(with_emacs_config.is_some_and(|m| m == EditMode::Emacs));
        fs::remove_file(".editrc").unwrap();
    }

    #[test]
    #[serial]
    fn test_get_inputrc_edit_mode() {
        let no_config = get_inputrc_edit_mode();
        assert_eq!(no_config, EditMode::Emacs);

        fs::write(".inputrc", "foo\nset editing-mode vi").unwrap();
        let with_vi_config = temp_env::with_var("INPUTRC", Some(".inputrc"), get_inputrc_edit_mode);
        assert_eq!(with_vi_config, EditMode::Vi);
        fs::remove_file(".inputrc").unwrap();

        fs::write(".inputrc", "foo").unwrap();
        let with_emacs_config = get_inputrc_edit_mode();
        assert_eq!(with_emacs_config, EditMode::Emacs);
        fs::remove_file(".inputrc").unwrap();
    }
}
