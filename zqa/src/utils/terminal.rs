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
/// # Arguments
///
/// * `reader` - The input reader.
/// * `default` - The default if Enter is pressed.
/// * `valid_set` - The valid set of characters, in lowercase.
pub(crate) fn read_char<R: BufRead>(reader: &mut R, default: char, valid_set: &[char]) -> char {
    read_char_with_writer(reader, &mut std::io::stdout(), default, valid_set)
        .expect("Failed to read input")
}

/// Read a character while routing the prompt through a supplied writer.
///
/// # Arguments
///
/// * `reader` - The input reader.
/// * `writer` - The output writer for prompt markers.
/// * `default` - The default if Enter is pressed.
/// * `valid_set` - The valid set of characters, in lowercase.
///
/// # Errors
///
/// * Returns an I/O error if the prompt cannot be written or input cannot be read.
pub(crate) fn read_char_with_writer<R, W>(
    reader: &mut R,
    writer: &mut W,
    default: char,
    valid_set: &[char],
) -> std::io::Result<char>
where
    R: BufRead + ?Sized,
    W: Write + ?Sized,
{
    debug_assert!(
        valid_set.contains(&default),
        "`default` must be a member of `valid_set`"
    );

    loop {
        write!(writer, "> ")?;
        writer.flush()?;
        let mut input = String::new();
        reader.read_line(&mut input)?;
        let choice = input.chars().next().unwrap_or(default).to_ascii_lowercase();

        if valid_set.contains(&choice) {
            return Ok(choice);
        }

        if choice == '\n' {
            return Ok(default);
        }
    }
}

/// Read an integer from standard input, and validate that it is within bounds.
///
/// # Arguments
///
/// * `reader` - The input reader.
/// * `default` - The default value if Enter is pressed.
/// * `bounds` - Lower and upper bounds to accept. Lower bound is inclusive, upper is exclusive.
pub(crate) fn read_number<R: BufRead>(
    reader: &mut R,
    default: usize,
    bounds: (usize, usize),
) -> usize {
    read_number_with_writer(reader, &mut std::io::stdout(), default, bounds)
        .expect("Failed to read input")
}

/// Read a bounded integer while routing prompts through a supplied writer.
///
/// # Arguments
///
/// * `reader` - The input reader.
/// * `writer` - The output writer for prompts and validation errors.
/// * `default` - The default value if Enter is pressed.
/// * `bounds` - Inclusive lower and exclusive upper bounds.
///
/// # Errors
///
/// * Returns an I/O error if output cannot be written or input cannot be read.
pub(crate) fn read_number_with_writer<R, W>(
    reader: &mut R,
    writer: &mut W,
    default: usize,
    bounds: (usize, usize),
) -> std::io::Result<usize>
where
    R: BufRead + ?Sized,
    W: Write + ?Sized,
{
    loop {
        write!(writer, "> ")?;
        writer.flush()?;

        let mut input = String::new();
        reader.read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() {
            return Ok(default);
        }

        if let Ok(num) = input.parse::<usize>()
            && bounds.0 <= num
            && num < bounds.1
        {
            return Ok(num);
        }
        writeln!(writer, "Choice must be in [{}, {}).", bounds.0, bounds.1)?;
    }
}
