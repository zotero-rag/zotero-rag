//! Utilities for interacting with documents that are not from the user's Zotero library.

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    pin::Pin,
};
use thiserror::Error;

use zqa_pdftools::{
    chunk::{Chunker, ChunkingStrategy},
    parse::{ExtractedContent, extract_text},
};
use zqa_rag::{
    embedding::common::{EmbeddingProviderConfig, get_embedding_provider},
    llm::{errors::LLMError, tools::Tool},
    reranking::common::{RerankProviderConfig, get_reranking_provider},
};

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
    #[error("Configuration issue: {0}")]
    BadConfig(String),
    #[error("Error getting embedding provider: {0}")]
    EmbeddingError(#[from] LLMError),
    #[error("Error computing embeddings: {0}")]
    LanceError(#[from] lancedb::Error),
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
/// threshold is used instead. The returned index is the exclusive end index of the summary, so the next
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
                w[0].byte_index <= summary_index_config.max_summary_sec_pos
                    && w[1].byte_index.saturating_sub(w[0].byte_index)
                        >= summary_index_config.min_summary_sec_len
            })
            .map_or_else(
                || {
                    // Snap to valid UTF-8 boundary
                    text.floor_char_boundary(
                        summary_index_config.max_summary_sec_pos.min(text.len()),
                    )
                },
                |w| w[1].byte_index,
            ),
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum QueryMethod {
    Embedding,
    SubAgent,
    Hybrid,
}

fn dot<'a, T>(a: T, b: T) -> f32
where
    T: Iterator<Item = &'a f32>,
{
    a.zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

#[derive(Deserialize, JsonSchema)]
struct UserDocumentToolInput {
    /// The filenames to use. Optional; if `None`, will use all files currently selected by the
    /// user.
    filenames: Option<Vec<String>>,
    /// A query to obtain relevant passages
    query: String,
    /// Query method. One of "embedding", "sub_agent", or "hybrid".
    query_method: QueryMethod,
}

/// A tool to get relevant contents from user-imported (non-Zotero) documents.
pub(crate) struct UserDocumentTool {
    /// The files currently in the session.
    pub(crate) filenames: Vec<String>,
    /// For [`QueryMethod::Embedding`] and [`QueryMethod::Hybrid`], the embedding config.
    pub(crate) embedding_config: Option<EmbeddingProviderConfig>,
    /// For [`QueryMethod::Embedding`] and [`QueryMethod::Hybrid`], the reranker config.
    pub(crate) reranker_config: Option<RerankProviderConfig>,
}

impl Tool for UserDocumentTool {
    fn name(&self) -> String {
        "user_document_tool".into()
    }

    fn description(&self) -> String {
        "A tool to get relevant contents from user-imported (non-Zotero) documents.".into()
    }

    fn parameters(&self) -> schemars::Schema {
        schema_for!(UserDocumentToolInput)
    }

    fn call<'a>(
        &'a self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>> {
        type FileFuture<'a> =
            Pin<Box<dyn Future<Output = Result<(String, Vec<String>), DocumentError>> + Send + 'a>>;

        // TODO: Tune this in a future story; possibly look at embedding provider
        // docs and create a constant or a function.
        const SCORE_THRESHOLD: f32 = 0.7;

        Box::pin(async move {
            let input: UserDocumentToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;

            if self.filenames.is_empty() {
                return Err("There are no documents in this session. Ask the user to add some by @ing the file name.".into());
            }

            let filenames = input.filenames.as_ref().unwrap_or(&self.filenames);
            let session_files: HashSet<_> = self.filenames.iter().collect();
            if let Some(f) = filenames.iter().find(|f| !session_files.contains(f)) {
                return Err(format!("File {f} does not exist in the session."));
            }

            let futures: FuturesUnordered<FileFuture> = FuturesUnordered::new();
            for filename in filenames {
                if matches!(
                    input.query_method,
                    QueryMethod::Embedding | QueryMethod::Hybrid
                ) {
                    let Some(ref embedding_config) = self.embedding_config else {
                        return Err("Query method {} was used, but no embedding config was provided during tool creation. This is likely a bug.".into());
                    };

                    let Some(ref reranker_config) = self.reranker_config else {
                        return Err("Query method {} was used, but no reranker config was provided during tool creation. This is likely a bug.".into());
                    };

                    futures.push(Box::pin(async {
                        let filename = filename.clone();

                        // 1. Chunk document using `Chunker`
                        // 2. Generate embeddings altogether
                        //      - possibly contextual embeddings?
                        // 3. Compute similarities (on our own here)
                        // 4. Use reranker
                        // 5. If `QueryMethod::Hybrid`
                        //   - Find "hot" zones
                        //   - For the top-k chunks, look at the overlap in sections
                        //   - If there's a strong overlap (maybe by some percentage threshold),
                        //     then use the parent link in the `SectionBoundary` to go one level up
                        //   - Instead of the entire document, ask the agent to find relevant parts
                        //     from these instead. You can have chunks at different levels passed
                        //     to the agent.
                        let contents = extract_text(&filename).map_err(|e| {
                            DocumentError::TextExtractionFailed(TextExtractionError(format!(
                                "{filename}: {e}"
                            )))
                        })?;
                        let chunker = Chunker::new(contents, ChunkingStrategy::SectionBased(2048));
                        let chunks = chunker.chunk();

                        let chunk_texts: Vec<String> =
                            chunks.into_iter().map(|c| c.content).collect();
                        let provider = get_embedding_provider(embedding_config.provider_name())?;

                        let chunk_embeddings =
                            provider.compute_source_embeddings(Arc::new(StringArray::from(
                                chunk_texts.iter().map(String::as_str).collect::<Vec<_>>(),
                            )))?;
                        let chunk_embeddings: &[f32] = chunk_embeddings
                            .as_ref()
                            .as_fixed_size_list()
                            .values()
                            .as_primitive::<Float32Type>()
                            .values();

                        let query_embeddings = provider.compute_query_embeddings(Arc::new(
                            StringArray::from(vec![input.query.as_ref()]),
                        ))?;
                        let query_embeddings: &[f32] = query_embeddings
                            .as_ref()
                            .as_fixed_size_list()
                            .values()
                            .as_primitive::<Float32Type>()
                            .values();
                        let dim = query_embeddings.len();

                        let query_normalized = normalize(query_embeddings);
                        let chunk_embeddings_normalized: Vec<f32> = chunk_embeddings
                            .chunks_exact(dim)
                            .flat_map(normalize)
                            .collect();

                        let scores: Vec<f32> = chunk_embeddings_normalized
                            .chunks_exact(dim)
                            .map(|chunk| dot(chunk.iter(), query_normalized.iter()))
                            .collect();

                        let kept_chunks: Vec<&String> = scores
                            .iter()
                            .enumerate()
                            .filter(|(_, v)| **v >= SCORE_THRESHOLD)
                            .map(|(i, _)| chunk_texts.get(i).unwrap())
                            .collect();

                        if kept_chunks.is_empty() {
                            return Ok((filename, Vec::new()));
                        }

                        let reranker = get_reranking_provider(reranker_config.provider_name())?;
                        let reranked_idx = reranker.rerank(&kept_chunks, &input.query).await?;
                        let reranked_chunks: Vec<&String> = reranked_idx
                            .iter()
                            .map(|i| *kept_chunks.get(*i).unwrap())
                            .collect();

                        Ok((
                            filename.clone(),
                            reranked_chunks.into_iter().cloned().collect(),
                        ))
                    }));
                }
            }

            let mut chunks_by_file = HashMap::<&String, Vec<String>>::new();
            while let Some((filename, chunks)) = file_futures.next().await {
                chunks_by_file.entry(filename).or_default().extend(chunks);
            }

            Ok(json!({
                "results": []
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use zqa_macros::{test_eq, test_ok};
    use zqa_pdftools::parse::{ExtractedContent, SectionBoundary};

    use super::*;

    fn make_doc(text: &str, sections: Vec<SectionBoundary>) -> UserDocument {
        UserDocument {
            filename: "test".to_string(),
            contents: ExtractedContent {
                text_content: text.to_string(),
                sections,
                page_count: 1,
            },
            summary: String::new(),
        }
    }

    fn make_section(byte_index: usize) -> SectionBoundary {
        SectionBoundary {
            page_number: 0,
            byte_index,
            level: 1,
            parent_idx: None,
            font_size: 12.0,
        }
    }

    #[test]
    fn test_parse_user_document_file_not_found() {
        let path = PathBuf::from("/nonexistent/path/file.pdf");
        let result = parse_user_document(&path);
        assert!(matches!(
            result,
            Err(DocumentError::TextExtractionFailed(_))
        ));
    }

    #[test]
    fn test_parse_user_document_success() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets/Zotero/storage/7R5XZ5PX/Yedida et al. - 2023 - An expert system for redesigning software for cloud applications.pdf");
        let result = parse_user_document(&path);
        test_ok!(result);

        let doc = result.unwrap();
        test_eq!(doc.filename, path.to_str().unwrap());
        assert!(!doc.contents.text_content.is_empty());
        assert!(!doc.summary.is_empty());
    }

    #[test]
    fn test_summary_end_index_introduction_within_threshold() {
        // "introduction" starts at byte 50, which is within max_summary_sec_pos=1000
        let text = format!("{}introduction rest of document", " ".repeat(50));
        let doc = make_doc(&text, vec![]);
        let config = SummaryIndexConfig::default();
        let idx = get_summary_end_index(&doc.contents, config);

        test_eq!(idx, 50);
    }

    #[test]
    fn test_summary_end_index_introduction_case_insensitive() {
        let text = format!("{}Introduction rest of document", " ".repeat(50));
        let doc = make_doc(&text, vec![]);
        let config = SummaryIndexConfig::default();
        let idx = get_summary_end_index(&doc.contents, config);

        test_eq!(idx, 50);
    }

    #[test]
    fn test_summary_end_index_introduction_beyond_threshold_falls_back_to_sections() {
        // "introduction" at 2000, beyond max_summary_sec_pos=1000
        // Section at byte 200, next section at byte 500 (diff=300 >= min_summary_sec_len=100)
        let text = format!("{}introduction", "x".repeat(2000));
        let doc = make_doc(&text, vec![make_section(200), make_section(500)]);
        let config = SummaryIndexConfig::default();
        let idx = get_summary_end_index(&doc.contents, config);

        // Should return s.byte_index = 500
        test_eq!(idx, 500);
    }

    #[test]
    fn test_summary_end_index_no_introduction_uses_sections() {
        let text = "x".repeat(2000);
        let doc = make_doc(&text, vec![make_section(100), make_section(400)]);
        let config = SummaryIndexConfig::default();
        let idx = get_summary_end_index(&doc.contents, config);

        test_eq!(idx, 400);
    }

    #[test]
    fn test_summary_end_index_no_introduction_section_too_small() {
        // Section gap is 10 < min_summary_sec_len=100, so falls through to the default
        let text = "x".repeat(2000);
        let doc = make_doc(&text, vec![make_section(100), make_section(110)]);
        let config = SummaryIndexConfig::default();
        let idx = get_summary_end_index(&doc.contents, config);

        // Falls to min(len, max_summary_sec_pos) = min(2000, 1000) = 1000
        test_eq!(idx, 1000);
    }

    #[test]
    fn test_summary_end_index_no_introduction_no_sections() {
        let text = "x".repeat(2000);
        let doc = make_doc(&text, vec![]);
        let config = SummaryIndexConfig::default();
        let idx = get_summary_end_index(&doc.contents, config);

        test_eq!(idx, 1000);
    }

    #[test]
    fn test_summary_end_index_short_document_no_sections() {
        // Document shorter than max_summary_sec_pos; returns len-1
        let text = "short".to_string();
        let doc = make_doc(&text, vec![]);
        let config = SummaryIndexConfig::default();
        let idx = get_summary_end_index(&doc.contents, config);

        test_eq!(idx, 5); // len=5, min(5, 1000)=5
    }

    #[test]
    fn test_summary_end_index_section_beyond_threshold_skipped() {
        // Section starts at 1500, beyond max_summary_sec_pos=1000, so no valid pair found
        let text = "x".repeat(3000);
        let doc = make_doc(&text, vec![make_section(1500), make_section(2000)]);
        let config = SummaryIndexConfig::default();
        let idx = get_summary_end_index(&doc.contents, config);

        test_eq!(idx, 1000);
    }
}
