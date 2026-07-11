use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::state::StateError;

/// Whether the process is in CLI mode, i.e. the terminal may be written to directly. Some
/// output (progress bars, the verbose tool announcements) bypasses the context's output
/// streams and draws straight on the terminal; while the TUI owns the (alternate) screen,
/// such output must be suppressed or it draws over the interface.
static CLI_MODE: AtomicBool = AtomicBool::new(true);

/// Returns whether the terminal may be written to directly (see [`CLI_MODE`]).
pub(crate) fn in_cli_mode() -> bool {
    CLI_MODE.load(Ordering::Relaxed)
}

/// Enter or leave CLI mode (see [`in_cli_mode`]). The TUI clears this while it owns the
/// screen and restores it on exit.
pub(crate) fn set_cli_mode(enabled: bool) {
    CLI_MODE.store(enabled, Ordering::Relaxed);
}

/// ANSI escape code for dimming text
pub const DIM_TEXT: &str = "\x1b[2m";

/// ANSI escape code for resetting text formatting
pub const RESET: &str = "\x1b[0m";

/// ANSI escape code for yellow text
pub const YELLOW: &str = "\x1b[33m";

/// ANSI escape code for yellow, bold text
pub const YELLOW_BOLD: &str = "\x1b[33;1m";

/// ANSI escape code for red text
pub const RED: &str = "\x1b[31m";

/// ANSI escape code for red, bold text
pub const RED_BOLD: &str = "\x1b[31;1m";

/// ANSI escape code for bold text
pub const BOLD: &str = "\x1b[1m";

/// Read a line of input.
pub(crate) fn read_line<R: BufRead>(reader: &mut R) -> String {
    let mut input = String::new();
    reader.read_line(&mut input).expect("Failed to read input");

    input
}

/// Read a password from standard input.
pub(crate) fn read_password<R: BufRead>(
    reader: &mut R,
    is_terminal: bool,
) -> Result<String, StateError> {
    if is_terminal {
        rpassword::read_password().map_err(|e| StateError::PasswordReadError(e.to_string()))
    } else {
        Ok(read_line(reader))
    }
}

/// Read a character from standard input and return it, handling Enter as a default.
///
/// This function does not distinguish between lower- and uppercase options, and returns the
/// lowercase version of the character.
///
/// # Arguments:
///
/// * `reader` - The input reader.
/// * `default` - The default if Enter is pressed.
/// * `valid_set` - The valid set of characters, in lowercase.
pub(crate) fn read_char<R: BufRead>(reader: &mut R, default: char, valid_set: &[char]) -> char {
    debug_assert!(
        valid_set.contains(&default),
        "`default` must be a member of `valid_set`"
    );

    loop {
        print!("> ");
        let input = read_line(reader);
        let choice = input.chars().next().unwrap_or(default).to_ascii_lowercase();

        if valid_set.contains(&choice) {
            return choice;
        }

        if choice == '\n' {
            return default;
        }
    }
}

/// Read an integer from standard input, and validate that it is within bounds.
///
/// # Arguments:
///
/// * `reader` - The input reader.
/// * `default` - The default value if Enter is pressed.
/// * `bounds` - Lower and upper bounds to accept. Lower bound is inclusive, upper is exclusive.
pub(crate) fn read_number<R: BufRead>(
    reader: &mut R,
    default: usize,
    bounds: (usize, usize),
) -> usize {
    loop {
        print!("> ");
        let _ = std::io::stdout().flush();

        let input = read_line(reader);
        let input = input.trim();
        if input.is_empty() {
            return default;
        }

        if let Ok(num) = input.parse::<usize>()
            && bounds.0 <= num
            && num < bounds.1
        {
            return num;
        }
        println!("Choice must be in [{}, {}).", bounds.0, bounds.1);
    }
}
