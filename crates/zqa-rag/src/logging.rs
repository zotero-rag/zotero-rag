//! Bounded previews for diagnostic logs that may contain document or provider text.

use std::fmt::{self, Write};

/// Maximum number of UTF-8 bytes shown before the truncation notice.
const PREVIEW_BYTES: usize = 512;

/// Format a value lazily, showing at most 512 bytes and the total formatted byte count
/// when truncated. The prefix always ends at a UTF-8 character boundary.
///
/// This adapter does not allocate an intermediate string. Formatting still visits the
/// entire value to count its bytes, so construct it inside the logging macro. It does
/// not redact secrets; credentials must never be passed to it.
///
/// # Arguments
///
/// * `value` - Text or formatting arguments to preview.
///
/// # Returns
///
/// A display adapter that leaves short values unchanged.
///
/// # Examples
///
/// ```
/// use zqa_rag::logging::preview;
///
/// assert_eq!(preview("short text").to_string(), "short text");
/// let chunks = ["a paper", "another paper"];
/// log::debug!("Retrieved chunks: {}", preview(format_args!("{chunks:?}")));
/// ```
#[must_use]
pub fn preview(value: impl fmt::Display) -> impl fmt::Display {
    Preview(value)
}

struct Preview<T>(T);

impl<T: fmt::Display> fmt::Display for Preview<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut writer = PreviewWriter {
            output: formatter,
            total_bytes: 0,
            shown_bytes: 0,
            truncated: false,
        };
        write!(writer, "{}", self.0)?;
        if writer.truncated {
            write!(
                writer.output,
                "... [truncated; {} bytes total]",
                writer.total_bytes
            )?;
        }
        Ok(())
    }
}

struct PreviewWriter<'a, 'b> {
    output: &'a mut fmt::Formatter<'b>,
    total_bytes: usize,
    shown_bytes: usize,
    truncated: bool,
}

impl Write for PreviewWriter<'_, '_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.total_bytes += text.len();
        if self.truncated {
            return Ok(());
        }

        let end = text.floor_char_boundary(text.len().min(PREVIEW_BYTES - self.shown_bytes));
        self.output.write_str(&text[..end])?;
        self.shown_bytes += end;
        self.truncated = end < text.len();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PREVIEW_BYTES, preview};

    #[test]
    fn short_and_exact_limit_values_are_unchanged() {
        for text in [
            String::new(),
            "short text".into(),
            "x".repeat(PREVIEW_BYTES),
        ] {
            assert_eq!(preview(&text).to_string(), text);
        }
    }

    #[test]
    fn long_text_reports_original_size() {
        let text = "x".repeat(100_000);
        assert_eq!(
            preview(&text).to_string(),
            format!(
                "{}... [truncated; 100000 bytes total]",
                "x".repeat(PREVIEW_BYTES)
            )
        );
        assert_eq!(text.len(), 100_000);
    }

    #[test]
    fn truncation_preserves_unicode_and_a_contiguous_prefix() {
        let prefix = "x".repeat(PREVIEW_BYTES - 1);
        let result = preview(format_args!("{prefix}{}{}", '\u{1f980}', "tail")).to_string();
        assert_eq!(result, format!("{prefix}... [truncated; 519 bytes total]"));
    }

    #[test]
    fn collection_preview_has_one_budget_for_all_items() {
        let chunks = vec!["paper".repeat(1000); 20];
        let full = format!("{chunks:?}");
        let result = preview(format_args!("{chunks:?}")).to_string();
        assert_eq!(
            result,
            format!(
                "{}... [truncated; {} bytes total]",
                &full[..PREVIEW_BYTES],
                full.len()
            )
        );
    }
}
