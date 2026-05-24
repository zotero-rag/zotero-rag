use std::io::BufRead;

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
pub(crate) fn read_number<R: BufRead>(reader: &mut R, default: u8, bounds: (u8, u8)) -> u8 {
    loop {
        print!("> ");
        let input = read_line(reader);
        let input = input.trim();
        if input.is_empty() {
            return default;
        }

        let choice = input.parse::<u8>();

        match choice {
            Ok(num) => {
                if bounds.0 <= num && num < bounds.1 {
                    return num;
                }
                println!("Choice must be in [{}, {}).", bounds.0, bounds.1);
            }
            Err(_) => {
                println!("Choice must be in [{}, {}).", bounds.0, bounds.1);
            }
        }
    }
}
