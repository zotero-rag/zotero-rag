//! Utilities for interacting with documents that are not from the user's Zotero library.

use std::path::Path;
use thiserror::Error;

use zqa_pdftools::parse::{ExtractedContent, extract_text};

use crate::common::UserDocument;

/// Newtype wrapper for `zqa_pdftools` type-erased error.
#[derive(Debug, Error)]
pub struct TextExtractionError(String);

impl std::fmt::Display for TextExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Error)]
pub enum DocumentError {
    #[error("Path conversion failed: {0}")]
    PathConversionFailed(String),
    #[error("Text extraction failed: {0}")]
    TextExtractionFailed(TextExtractionError),
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
/// * [`DocumentError::PathConversionFailed`] if `path` could not be converted to an `&str`.
/// * [`DocumentError::TextExtractionFailed`] if `extract_text` returned an error.
pub(crate) fn parse_user_document(path: &Path) -> Result<UserDocument, DocumentError> {
    let filename = path.to_str().ok_or(DocumentError::PathConversionFailed(
        path.to_string_lossy().to_string(),
    ))?;
    let contents = extract_text(filename).map_err(|e| {
        DocumentError::TextExtractionFailed(TextExtractionError(format!("{filename}: {e}")))
    })?;
    let summary_end_index = get_summary_end_index(&contents, SummaryIndexConfig::default());

    Ok(UserDocument {
        filename: filename.to_string(),
        summary: contents.text_content[..summary_end_index].to_string(),
        contents,
    })
}

const DEFAULT_MIN_SUMMARY_SEC_LEN: usize = 100;
const DEFAULT_MAX_SUMMARY_SEC_POS: usize = 1000;

/// Configuration passed to [`get_summary_end_index`] with heuristic thresholds for the maximum
/// summary length, the minimum length of a summary section, and the maximum position for the
/// *beginning* of the summary section.
#[derive(Copy, Clone, Debug)]
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
    parsed_doc: &ExtractedContent,
    summary_index_config: SummaryIndexConfig,
) -> usize {
    const INTRODUCTION_LEN: usize = "introduction".len();
    let text = &parsed_doc.text_content;

    match text
        .as_bytes()
        .windows(INTRODUCTION_LEN)
        .position(|w| w.eq_ignore_ascii_case(b"introduction"))
    {
        Some(pos) if pos <= summary_index_config.max_summary_sec_pos => pos,
        _ => parsed_doc
            .sections
            .windows(2)
            .find(|w| {
                if let [f, s] = w {
                    f.byte_index <= summary_index_config.max_summary_sec_pos
                        && s.byte_index.saturating_sub(f.byte_index)
                            >= summary_index_config.min_summary_sec_len
                } else {
                    false
                }
            })
            .map_or_else(
                || {
                    parsed_doc
                        .len()
                        .min(summary_index_config.max_summary_sec_pos)
                },
                |w| w[1].byte_index,
            ),
    }
}
