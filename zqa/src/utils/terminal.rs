use std::io::{BufRead, Write};

use crate::state::StateError;

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

/// Given a positive number, returns a thousands separator-formatted string representation
///
/// # Arguments:
///
/// * `num` - The number to format
///
/// # Returns
///
/// The thousands-separated string
pub(crate) fn format_number(num: u64) -> String {
    num.to_string()
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(std::str::from_utf8)
        .collect::<Result<Vec<&str>, _>>()
        .unwrap_or_default()
        .join(",")
}

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

/// Read a character from the input reader and return it, handling Enter as a default.
///
/// This function does not distinguish between lower- and uppercase options, and returns the
/// lowercase version of the character.
///
/// # Arguments:
///
/// * `reader` - The input reader.
/// * `out` - The writer the prompt is written to.
/// * `default` - The default if Enter is pressed.
/// * `valid_set` - The valid set of characters, in lowercase.
pub(crate) fn read_char<R: BufRead, W: Write>(
    reader: &mut R,
    out: &mut W,
    default: char,
    valid_set: &[char],
) -> char {
    debug_assert!(
        valid_set.contains(&default),
        "`default` must be a member of `valid_set`"
    );

    loop {
        let _ = write!(out, "> ");
        let _ = out.flush();

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

/// Read an integer from the input reader, and validate that it is within bounds.
///
/// # Arguments:
///
/// * `reader` - The input reader.
/// * `out` - The writer the prompt is written to.
/// * `default` - The default value if Enter is pressed.
/// * `bounds` - Lower and upper bounds to accept. Lower bound is inclusive, upper is exclusive.
pub(crate) fn read_number<R: BufRead, W: Write>(
    reader: &mut R,
    out: &mut W,
    default: usize,
    bounds: (usize, usize),
) -> usize {
    loop {
        let _ = write!(out, "> ");
        let _ = out.flush();

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
        let _ = writeln!(out, "Choice must be in [{}, {}).", bounds.0, bounds.1);
    }
}
