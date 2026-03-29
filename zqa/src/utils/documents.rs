//! Utilities for interacting with documents that are not from the user's Zotero library.

use std::path::PathBuf;
use thiserror::Error;

use zqa_pdftools::parse::extract_text;

use crate::common::UserDocument;

#[derive(Debug, Error)]
pub enum DocumentError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Path conversion failed: {0}")]
    PathConversionFailed(String),
    #[error("Text extraction failed: {0}")]
    TextExtractionFailed(#[from] Box<dyn std::error::Error>),
}

/// Parse an imported user document (one not from Zotero, but from the file system), and return a
/// [`UserDocument`] to store in state.
///
/// # Arguments
///
/// * `path` - The path to the file
///
/// # Returns
///
/// A [`UserDocument`] containing the "summary" (text before the Introduction) and extracted contents.
///
/// # Errors
///
/// * [`DocumentError::FileNotFound`] if `path` does not exist.
/// * [`DocumentError::PathConversionFailed`] if `path` could not be converted to an `&str`.
/// * [`DocumentError::TextExtractionFailed`] if `extract_text` returned an error.
pub fn parse_user_document(path: PathBuf) -> Result<UserDocument, DocumentError> {
    if !path.exists() {
        return Err(DocumentError::FileNotFound(
            path.to_string_lossy().to_string(),
        ));
    }

    let filename = path.to_str().ok_or(DocumentError::PathConversionFailed(
        path.to_string_lossy().to_string(),
    ))?;
    let contents = extract_text(filename)?;

    Ok(UserDocument {
        filename: filename.to_string(),
        contents,
        summary: String::new(),
    })
}

const DEFAULT_MIN_SUMMARY_SEC_LEN: usize = 100;
const DEFAULT_MAX_SUMMARY_SEC_POS: usize = 1000;

/// Configuration passed to [`get_summary_end_index`] with heuristic thresholds for the maximum
/// summary length, the minimum length of a summary section, and the maximum position for the
/// *beginning* of the summary section.
pub(crate) struct SummaryIndexConfig {
    /// The minimum length of a section to be considered a summary. Default: 100
    pub(crate) min_summary_sec_len: usize,
    /// The maximum position a summary section can start. Default: 1000
    pub(crate) max_summary_sec_pos: usize,
}

impl Default for SummaryIndexConfig {
    fn default() -> Self {
        Self {
            min_summary_sec_len: DEFAULT_MIN_SUMMARY_SEC_LEN,
            max_summary_sec_pos: DEFAULT_MAX_SUMMARY_SEC_POS,
        }
    }
}

/// Given a `parsed_doc`, return the index where the "summary" ends.
///
/// The "summary" is defined as the contents up to the "Introduction" text. If this does not exist
/// or is "too far" from the beginning of the document, the first document section within a
/// threshold is used instead. The returned index is the last index of the summary, so the next
/// index is where the Introduction starts.
///
/// Note that this does *not* guarantee that the index afer the returned position is valid. For
/// example, if the document is small and unformatted, it is likely that the returned index will be
/// the last valid index in the contents.
fn get_summary_end_index(
    parsed_doc: &UserDocument,
    summary_index_config: SummaryIndexConfig,
) -> usize {
    let text = &parsed_doc.contents.text_content;

    match text.to_ascii_lowercase().find("introduction") {
        Some(pos) if pos <= summary_index_config.max_summary_sec_pos => pos,
        _ => parsed_doc
            .contents
            .sections
            .iter()
            .zip(parsed_doc.contents.sections.iter().skip(1))
            .find(|(f, s)| {
                f.byte_index <= summary_index_config.max_summary_sec_pos
                    && s.byte_index - f.byte_index >= summary_index_config.min_summary_sec_len
            })
            .map_or_else(
                || {
                    parsed_doc
                        .contents
                        .len()
                        .saturating_sub(1)
                        .min(summary_index_config.max_summary_sec_pos)
                },
                |(_, s)| s.byte_index - 1,
            ),
    }
}
