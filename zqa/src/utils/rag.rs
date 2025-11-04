//! Utilities to interface with the `rag` crate. This mostly contains type conversions.

use rag::llm::base::ContentType;
use std::{
    borrow::Cow,
    fmt::{self, Write},
};

/// ANSI escape code for dimming text
const DIM_TEXT: &str = "\x1b[2m";

/// ANSI escape code for resetting text formatting
const RESET: &str = "\x1b[0m";

/// A wrapper type that contains all the parts of a single model response.
pub(crate) struct SingleResponse<'a> {
    pub parts: Cow<'a, [ContentType]>,
}

/// Convenience so users can simply `.into()` as needed. This *moves* the value.
impl<'a> From<Vec<ContentType>> for SingleResponse<'a> {
    fn from(value: Vec<ContentType>) -> Self {
        Self {
            parts: Cow::Owned(value),
        }
    }
}

/// Convenience so users can simply `.into()` as needed. This *moves* the value.
impl<'a> From<&'a Vec<ContentType>> for SingleResponse<'a> {
    fn from(value: &'a Vec<ContentType>) -> Self {
        Self {
            parts: Cow::Borrowed(value),
        }
    }
}

impl<'a> fmt::Display for SingleResponse<'a> {
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
