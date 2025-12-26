//! Utilities to interface with the `rag` crate. This mostly contains type conversions.

use std::{
    borrow::Cow,
    fmt::{self, Write},
};
use zqa_rag::llm::base::ContentType;

/// ANSI escape code for dimming text
const DIM_TEXT: &str = "\x1b[2m";

/// ANSI escape code for resetting text formatting
const RESET: &str = "\x1b[0m";

/// A wrapper type over `Vec<ContentType>` that provides a nicer interface for some tasks. This can
/// be constructed from an owned value or a reference, and implements `Display` so that the model
/// response can be printed or converted to a `String` using `to_string()`.
pub(crate) struct ModelResponse<'a> {
    parts: Cow<'a, [ContentType]>,
}

impl From<Vec<ContentType>> for ModelResponse<'_> {
    fn from(value: Vec<ContentType>) -> Self {
        Self {
            parts: Cow::Owned(value),
        }
    }
}

impl<'a> From<&'a Vec<ContentType>> for ModelResponse<'a> {
    fn from(value: &'a Vec<ContentType>) -> Self {
        Self {
            parts: Cow::Borrowed(value),
        }
    }
}

impl fmt::Display for ModelResponse<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for part in self.parts.as_ref() {
            match part {
                ContentType::Text(s) => {
                    f.write_str(s)?;
                    f.write_char('\n')?;
                }
                ContentType::ToolCall(stats) => {
                    f.write_fmt(format_args!("{}🗸 {}{}\n", DIM_TEXT, stats.tool_name, RESET))?;

                    // Log details in DEBUG mode
                    log::debug!(
                        "Tool call:\n\tname: {}\n\targuments: {}\n\tresponse{}",
                        &stats.tool_name,
                        serde_json::to_string_pretty(&stats.tool_args).or(Err(fmt::Error {}))?,
                        serde_json::to_string_pretty(&stats.tool_result).or(Err(fmt::Error {}))?
                    );
                }
            }
        }

        Ok(())
    }
}
