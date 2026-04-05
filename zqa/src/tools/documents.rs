//! Tools for interacting with documents that are not from the user's Zotero library.

use arrow_array::StringArray;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use lancedb::embeddings::EmbeddingFunction;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, RwLock};
use std::{collections::HashMap, path::Path, pin::Pin};
use thiserror::Error;
use zqa_rag::llm::base::ChatRequest;

use zqa_pdftools::{
    chunk::{Chunker, ChunkingStrategy},
    parse::{ExtractedContent, extract_text},
};
use zqa_rag::{
    embedding::common::{EmbeddingProviderConfig, get_embedding_provider},
    llm::{base::ApiClient, errors::LLMError, factory::LLMClient, tools::Tool},
    reranking::common::{RerankProviderConfig, get_reranking_provider},
};

use crate::common::UserDocument;
use crate::utils::rag::ModelResponse;

// TODO: Tune this in a future story; possibly look at embedding provider
// docs and create a constant or a function.

/// Threshold for cosine similarity to keep a chunk
const SCORE_THRESHOLD: f32 = 0.7;
/// If no chunks are found, this is how many chunks we keep.
const FALLBACK_CHUNK_LIMIT: usize = 8;
/// Threshold for percentage of chunks falling in one section, to "zoom out" to the parent section.
const ZOOM_OUT_THRESHOLD: f32 = 0.25;

/// Newtype wrapper for `zqa_pdftools` type-erased error.
#[derive(Debug, Error)]
pub struct TextExtractionError(String);

impl std::fmt::Display for TextExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Errors arising from working with user-imported documents.
#[derive(Debug, Error)]
pub enum DocumentError {
    #[error("Error getting embedding provider: {0}")]
    ApiError(#[from] LLMError),
    #[error("Configuration issue: {0}")]
    BadConfig(String),
    #[error("Failed to process document: {0}")]
    DocumentProcessingFailed(String),
    #[error("Error computing embeddings: {0}")]
    LanceError(#[from] lancedb::Error),
    #[error("Path conversion failed: {0}")]
    PathConversionFailed(String),
    #[error("Text extraction failed: {0}")]
    TextExtractionFailed(TextExtractionError),
}

/// A context object containing a thread-safe reference to user-imported documents and configs
/// necessary to process them.
pub(crate) struct DocumentsToolContext {
    /// User documents. These are shared with the tools created by [`DocumentsToolFactory`].
    pub(crate) documents: Arc<RwLock<HashMap<String, Arc<UserDocument>>>>,
    /// The config for the embedding provider.
    pub(crate) embedding_config: EmbeddingProviderConfig,
    /// The config for the reranking provider. May be `None`, in which case reranking is skipped.
    pub(crate) reranker_config: Option<RerankProviderConfig>,
    /// The client to interface with the LLM provider's text generation endpoint.
    pub(crate) client: Option<LLMClient>,
}

/// Factory to create the canonical toolset used by agents to interact with the user-imported
/// documents. It holds a thread-safe reference to a [`DocumentsToolContext`], which is effectively
/// the single source of truth for what documents are visible to agents.
pub(crate) struct DocumentsToolFactory {
    ctx: Arc<DocumentsToolContext>,
}

impl DocumentsToolFactory {
    /// Create a new factory.
    pub(crate) fn new(
        imports: &Arc<RwLock<HashMap<String, Arc<UserDocument>>>>,
        embedding_config: EmbeddingProviderConfig,
        reranker_config: Option<RerankProviderConfig>,
        client: Option<LLMClient>,
    ) -> Self {
        let documents = Arc::clone(imports);

        Self {
            ctx: Arc::new(DocumentsToolContext {
                documents,
                embedding_config,
                reranker_config,
                client,
            }),
        }
    }

    /// Return a list of tools to interact with the user-imported documents. As of now, all the
    /// tools are effectively pure functions in that they do not modify anything in the context.
    pub(crate) fn build_tools(&self) -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(ListDocumentsTool::new(Arc::clone(&self.ctx))),
            Box::new(QueryDocumentsTool::new(Arc::clone(&self.ctx))),
        ]
    }
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

/// Default value for the minimum length of a section to be considered a summary.
const DEFAULT_MIN_SUMMARY_SEC_LEN: usize = 100;
/// Default value for the maximum position a summary section can start.
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

/// The query method to use.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum QueryMethod {
    /// Only rely on embeddings + optionally, reranking if a config was provided.
    Embedding,
    /// Use an agent to clean up the chunks fetched by the embedding + optional reranking pipeline.
    Hybrid,
}

/// SIMD-friendly dot-product.
fn dot<'a, T>(a: T, b: T) -> f32
where
    T: Iterator<Item = &'a f32>,
{
    a.zip(b).map(|(x, y)| x * y).sum()
}

/// Scale the provided vector to unit norm.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        vec![0.0; v.len()]
    }
}

/// Get the prompt used in [`QueryDocumentsTool`] when a sub-agent is enabled.
///
/// # Arguments
///
/// * `query` - A query from a model or the user
/// * `retrieved_chunks` - A list of relevant chunks retrieved via embeddings
/// * `summary` - Extracted summary. See [`get_summary_end_index`].
fn get_prompt(query: &str, retrieved_chunks: &[&str], summary: &str) -> String {
    let chunks = retrieved_chunks.join("\n-----\n");

    format!(
        "You are a research assistant tasked with finding relevant chunks in a user-provided document. Some chunks
have been retrieved for you. You are also provided with what is possibly the paper's abstract, though this might not be fully accurate. Your task is to refine the chunks to be relevant to the user's query.

    <user_query>
    {query}
    </user_query>

    <paper_abstract>
    {summary}
    </paper_abstract>

    <chunks>
    {chunks}
    </chunks>

    Your returned chunks do not need to be entire chunks from the list above, and you may choose subsets as
applicable. However, you MUST return the text verbatim.

    Place each chunk on a new line, and separate chunks using 5 dashes (-----).
"
    )
}

/// Get embeddings for a list of texts using the specified embedding provider.
fn get_embeddings(
    texts: &[&str],
    embedding_provider: &Arc<dyn EmbeddingFunction>,
) -> Result<Vec<f32>, DocumentError> {
    let embeddings = embedding_provider.compute_source_embeddings(Arc::new(StringArray::from(
        texts.iter().map(|s| &**s).collect::<Vec<_>>(),
    )))?;
    let embeddings: &[f32] = embeddings
        .as_ref()
        .as_fixed_size_list()
        .values()
        .as_primitive::<Float32Type>()
        .values();

    Ok(embeddings.to_vec())
}

/// Call a sub-agent LLM to refine retrieved chunks to be relevant to `query`.
///
/// # Arguments
///
/// * `query` - A query from a model or the user
/// * `retrieved_chunks` - A list of relevant chunks retrieved via embeddings
/// * `summary` - Extracted summary. See [`get_summary_end_index`].
/// * `client` - A configured [`LLMClient`] to use for the sub-agent call
///
/// # Returns
///
/// A string containing the refined chunks, separated by `-----`.
///
/// # Errors
///
/// * [`DocumentError::ApiError`] if the LLM client returns an error.
async fn call_subagent(
    query: &str,
    retrieved_chunks: &[&str],
    summary: &str,
    client: &LLMClient,
) -> Result<String, DocumentError> {
    let request = ChatRequest {
        message: get_prompt(query, retrieved_chunks, summary),
        ..ChatRequest::default()
    };

    client
        .send_message(&request)
        .await
        .map(|response| {
            let response = ModelResponse::from(&response.content).to_string();

            // TODO: In a future refactor, this will be updated to actually get
            // a `Vec` of chunks. That requires implementing structured
            // outputs, which is a larger-scale change

            response.trim().to_string()
        })
        .map_err(Into::<DocumentError>::into)
}

/// Perform embedding-only retrieval from unchunked `contents`, based on `query`.
///
/// # Arguments
///
/// * `query` - A query, either from a model or the user
/// * `contents` - The [`zqa_pdftools::parse::ExtractedContent`] from the PDF parser.
/// * `embedding_config` - Configuration for an embedding provider
/// * `reranker_config` - Configuration for an reranker provider
///
/// # Returns
///
/// A list of chunks.
///
/// # Errors
///
/// * `DocumentError::ApiError` if either the embedding or reranker provider could not be
///   obtained, or if the embeddings or reranked chunks could not be computed.
async fn embedding_retrieval(
    query: &str,
    contents: &ExtractedContent,
    embedding_config: &EmbeddingProviderConfig,
    reranker_config: Option<&RerankProviderConfig>,
) -> Result<Vec<String>, DocumentError> {
    let chunker = Chunker::new(contents.clone(), ChunkingStrategy::SectionBased(2048));
    let chunks = chunker.chunk();

    let chunk_texts: Vec<String> = chunks.into_iter().map(|c| c.content.clone()).collect();
    let chunk_text_refs: Vec<&str> = chunk_texts
        .iter()
        .map(std::string::String::as_str)
        .collect();

    let provider = get_embedding_provider(embedding_config.provider_name())?;
    let chunk_embeddings = get_embeddings(&chunk_text_refs, &provider)?;

    let query_embeddings = get_embeddings(&[query], &provider)?;
    let dim = query_embeddings.len();

    let query_normalized = normalize(&query_embeddings);
    let chunk_embeddings_normalized: Vec<f32> = chunk_embeddings
        .chunks_exact(dim)
        .flat_map(normalize)
        .collect();

    let scores: Vec<f32> = chunk_embeddings_normalized
        .chunks_exact(dim)
        .map(|chunk| dot(chunk.iter(), query_normalized.iter()))
        .collect();

    let mut kept_chunks: Vec<&str> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| *v >= SCORE_THRESHOLD)
        .map(|(i, _)| chunk_texts[i].as_str())
        .collect();

    log::debug!("[embedding_retrieval] Kept chunks: {kept_chunks:#?}");

    if kept_chunks.is_empty() {
        let mut scored_indices = scores.iter().copied().enumerate().collect::<Vec<_>>();
        scored_indices.sort_by(|a, b| b.1.total_cmp(&a.1));

        kept_chunks = scored_indices
            .into_iter()
            .take(FALLBACK_CHUNK_LIMIT)
            .map(|(i, _)| chunk_texts[i].as_str())
            .collect();

        log::debug!(
            "No chunks passed score threshold {}; falling back to top {} chunk(s) for query '{}'",
            SCORE_THRESHOLD,
            kept_chunks.len(),
            query
        );
    }

    if kept_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let Some(reranker_config) = reranker_config else {
        return Ok(kept_chunks
            .iter()
            .map(std::string::ToString::to_string)
            .collect());
    };

    let reranker = get_reranking_provider(reranker_config.provider_name())?;
    let reranked_idx = reranker.rerank(&kept_chunks, query).await?;

    let reranked_chunks: Vec<String> = reranked_idx
        .into_iter()
        .map(|i| kept_chunks[i].to_string())
        .collect();

    Ok(reranked_chunks)
}

/// Retrieve relevant chunks from one document.
///
/// It uses an embedding-based chunk retrieval from the user-provided file. To do so, it first
/// chunks it into 2048-token chunks. From here, it uses `embedding_config` to generate embeddings
/// for the query and the chunks, and performs a cosine similarity to prune chunks to those whose
/// similarity scores are greater than `SCORE_THRESHOLD` (currently: 0.7). It then uses the
/// `reranker_config` to perform reranking of the retrieved chunks.
///
/// For [`QueryMethod::Hybrid`], this function prepares to pass the chunks to an agent by aiming to
/// provide more context. In particular, it finds "hot zones" in the user document (sections that
/// have at least the ratio `ZOOM_OUT_THRESHOLD` (currently: 0.25) of the pruned chunks. In these
/// hot zones, it uses the parent pointer in [`zqa_pdftools::parse::SectionBoundary`] to show the
/// parent section instead.
///
/// # Arguments
///
/// * `document` - The document to process
/// * `query` - A query, either from a model or the user
/// * `query_method` - See [`QueryMethod`]
/// * `embedding_config` - Configuration for an embedding provider
/// * `reranker_config` - Configuration for an reranker provider
/// * `client` - An optional [`zqa_rag::llm::factory::LLMClient`]. Must be `Some(..)` if
///   `query_method` is [`QueryMethod::Hybrid`].
///
/// # Returns
///
/// A list of chunks.
///
/// # Errors
///
/// * `DocumentError::TextExtractionFailed` if [`zqa_pdftools::parse::extract_text`] returns an
///   error.
/// * `DocumentError::ApiError` if either the embedding or reranker provider could not be
///   obtained, or if the embeddings or reranked chunks could not be computed.
async fn process_document(
    document: &UserDocument,
    query: &str,
    query_method: QueryMethod,
    embedding_config: &EmbeddingProviderConfig,
    reranker_config: Option<&RerankProviderConfig>,
    client: Option<&LLMClient>,
) -> Result<Vec<String>, DocumentError> {
    let reranked_chunks =
        embedding_retrieval(query, &document.contents, embedding_config, reranker_config).await?;

    log::debug!("Reranked chunks: {reranked_chunks:#?}");

    if reranked_chunks.is_empty() {
        log::debug!(
            "No document chunks passed retrieval for query '{}' in file '{}'",
            query,
            document.filename
        );
        return Ok(Vec::new());
    }

    if let QueryMethod::Hybrid = query_method {
        let Some(client) = client else {
            return Err(DocumentError::BadConfig(
                "`client` cannot be `None` when `query_method` is 'hybrid'. This is likely a bug."
                    .to_string(),
            ));
        };

        let sections = &document.contents.sections;
        let text_content = &document.contents.text_content;
        let mut returned_chunks = Vec::new();

        let mut section_counts: HashMap<usize, usize> = HashMap::new();

        for chunk in &reranked_chunks {
            let Some(text_idx) = text_content.find(&**chunk) else {
                continue;
            };

            // Which section does this chunk belong to?
            // `contents.sections` is sorted.
            let section_idx = sections
                .partition_point(|sec| text_idx >= sec.byte_index)
                .saturating_sub(1);
            *section_counts.entry(section_idx).or_insert(0) += 1;
        }

        let total = section_counts.values().sum::<usize>() as f32;
        for (section_idx, count) in &section_counts {
            let pct = *count as f32 / total;
            let effective_idx = if pct >= ZOOM_OUT_THRESHOLD {
                // Zoom out to higher-level section, since this one seems
                // important
                sections
                    .get(*section_idx)
                    .and_then(|s| s.parent_idx)
                    .unwrap_or(*section_idx)
            } else {
                *section_idx
            };

            // Handle the case for the last section
            if sections.len() > effective_idx {
                let end_byte = sections
                    .get(effective_idx + 1)
                    .map_or(text_content.len(), |s| s.byte_index);
                returned_chunks
                    .push(text_content[sections[effective_idx].byte_index..end_byte].to_string());
            }
        }

        if returned_chunks.is_empty() {
            log::debug!(
                "No expanded section chunks found for query '{}' in file '{}'; falling back to reranked chunks",
                query,
                document.filename
            );
            return Ok(reranked_chunks);
        }

        let result = call_subagent(
            query,
            &returned_chunks
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            &document.summary,
            client,
        )
        .await?;

        // TODO: This is terribly hacky, and is just a stupid way to do this in general.
        // However, the right solution is to use structured outputs (see the other todo in this
        // file).
        let agent_chunks = result
            .split("-----")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        Ok(agent_chunks)
    } else {
        Ok(reranked_chunks)
    }
}

/// Result of the [`QueryDocumentsTool`].
#[derive(serde::Serialize)]
struct QueryDocumentsResult {
    /// The filename, corresponding to a key in the context's `documents`.
    filename: String,
    /// The chunks extracted from the file.
    chunks: Vec<String>,
}

/// Input to [`QueryDocumentsTool`].
#[derive(Deserialize, JsonSchema)]
struct QueryDocumentsToolInput {
    /// The filenames to use. Optional; if `None`, will use all files currently selected by the
    /// user.
    filenames: Option<Vec<String>>,
    /// A query to obtain relevant passages
    query: String,
    /// Query method. One of "embedding" or "hybrid".
    query_method: QueryMethod,
}

/// The [`ListDocumentsTool`] takes no input, but we need an empty struct as opposed to a unit type
/// to pass to `schema_for`.
#[derive(Deserialize, JsonSchema)]
struct ListDocumentsToolInput {}

/// A tool to list available user-imported documents.
pub(crate) struct ListDocumentsTool {
    ctx: Arc<DocumentsToolContext>,
}

impl ListDocumentsTool {
    pub(crate) fn new(ctx: Arc<DocumentsToolContext>) -> Self {
        Self { ctx }
    }
}

impl Tool for ListDocumentsTool {
    fn name(&self) -> String {
        "list_documents".into()
    }

    fn description(&self) -> String {
        "List all user-imported documents available in the session.".into()
    }

    fn parameters(&self) -> schemars::Schema {
        schema_for!(ListDocumentsToolInput)
    }

    fn call<'a>(
        &'a self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>> {
        Box::pin(async move {
            let args = if args.is_null() { json!({}) } else { args };
            let _: ListDocumentsToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;

            let docs = Arc::clone(&self.ctx.documents);
            let map = docs
                .read()
                .map_err(|e| format!("Failed to obtain RwLock in ListDocumentsTool: {e}"))?;
            let files: Vec<&str> = map.keys().map(String::as_str).collect();

            Ok(json!({ "documents": files }))
        })
    }
}

/// A tool to get relevant contents from user-imported (non-Zotero) documents.
pub(crate) struct QueryDocumentsTool {
    ctx: Arc<DocumentsToolContext>,
}

impl QueryDocumentsTool {
    pub(crate) fn new(ctx: Arc<DocumentsToolContext>) -> Self {
        Self { ctx }
    }
}

impl Tool for QueryDocumentsTool {
    fn name(&self) -> String {
        "user_document_tool".into()
    }

    fn description(&self) -> String {
        "A tool to get relevant contents from user-imported (non-Zotero) documents.".into()
    }

    fn parameters(&self) -> schemars::Schema {
        schema_for!(QueryDocumentsToolInput)
    }

    #[allow(clippy::too_many_lines)]
    fn call<'a>(
        &'a self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>> {
        type FileFuture<'a> =
            Pin<Box<dyn Future<Output = Result<QueryDocumentsResult, DocumentError>> + Send + 'a>>;

        log::debug!("QueryDocumentsTool called with args: {args}");

        Box::pin(async move {
            let input: QueryDocumentsToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;

            let embedding_config = self.ctx.embedding_config.clone();
            let reranker_config = self.ctx.reranker_config.clone();
            let client = self.ctx.client.clone();
            if client.is_none() && matches!(input.query_method, QueryMethod::Hybrid) {
                log::debug!(
                    "Invalid query method {:?} (no LLM client passed)",
                    input.query_method
                );

                return Err(format!(
                    "Query method {:?} was used, but no LLM client was provided during tool creation. This is likely a bug.",
                    input.query_method,
                ));
            }
            let selected_documents: Vec<(String, Arc<UserDocument>)> = {
                let docs_map = self.ctx.documents.read().map_err(|e| {
                    log::debug!("Failed to obtain RwLock in QueryDocumentsTool: {e}");
                    format!("Failed to obtain RwLock in QueryDocumentsTool: {e}")
                })?;

                if docs_map.is_empty() {
                    log::debug!("Error in QueryDocumentsTool: no documents in session.");

                    return Err(
                        "There are no documents in this session. Ask the user to add some by @ing the file name."
                        .into(),
                    );
                }

                if let Some(filenames) = &input.filenames {
                    filenames
                        .iter()
                        .map(|filename| {
                            docs_map
                                .get(filename)
                                .map(|doc| (filename.clone(), Arc::clone(doc)))
                                .ok_or_else(|| {
                                    log::debug!("File {filename} does not exist in the session.");
                                    format!("File {filename} does not exist in the session.")
                                })
                        })
                        .collect::<Result<_, _>>()?
                } else {
                    let mut docs = docs_map
                        .iter()
                        .map(|(filename, doc)| (filename.clone(), Arc::clone(doc)))
                        .collect::<Vec<_>>();
                    docs.sort_by(|(a, _), (b, _)| a.cmp(b));
                    docs
                }
            };

            let query = input.query;
            let query_method = input.query_method;

            let mut futures: FuturesUnordered<FileFuture> = FuturesUnordered::new();
            for (filename, doc) in selected_documents {
                let query = query.clone();
                let embedding_config = embedding_config.clone();
                let reranker_config = reranker_config.clone();
                let client = client.clone();

                futures.push(Box::pin(async move {
                    let chunks = process_document(
                        doc.as_ref(),
                        &query,
                        query_method,
                        &embedding_config,
                        reranker_config.as_ref(),
                        client.as_ref(),
                    )
                    .await
                    .map_err(|e| {
                        log::debug!("Embedding-based retrieval failed: {e}");
                        DocumentError::DocumentProcessingFailed(format!(
                            "Embedding-based retrieval failed: {e}"
                        ))
                    })?;

                    Ok(QueryDocumentsResult { filename, chunks })
                }));
            }

            let mut all_results = Vec::new();
            let mut errors = Vec::new();

            while let Some(res) = futures.next().await {
                match res {
                    Ok(result) => all_results.push(result),
                    Err(e) => errors.push(e.to_string()),
                }
            }

            log::debug!(
                "QueryDocumentsTool returned: {:?}",
                json!({
                    "results": all_results,
                    "errors": errors
                })
            );

            Ok(json!({
                "results": all_results,
                "errors": errors
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use dotenv::dotenv;
    use serde_json::json;
    use zqa_macros::{test_eq, test_ok};
    use zqa_pdftools::parse::{ExtractedContent, SectionBoundary};
    use zqa_rag::config::{
        CohereConfig, GeminiConfig, LLMClientConfig, OpenAIConfig, VoyageAIConfig,
    };
    use zqa_rag::constants::{
        DEFAULT_COHERE_EMBEDDING_DIM, DEFAULT_COHERE_EMBEDDING_MODEL, DEFAULT_COHERE_RERANK_MODEL,
        DEFAULT_GEMINI_EMBEDDING_DIM, DEFAULT_GEMINI_EMBEDDING_MODEL, DEFAULT_OPENAI_EMBEDDING_DIM,
        DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_OPENAI_MAX_TOKENS, DEFAULT_OPENAI_MODEL_SMALL,
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::llm::factory::get_client_with_config;

    use super::*;

    fn test_pdf() -> UserDocument {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets/Zotero/storage/7R5XZ5PX/Yedida et al. - 2023 - An expert system for redesigning software for cloud applications.pdf")
            .to_string_lossy()
            .to_string();
        let text = extract_text(&path);

        UserDocument {
            filename: "mono2micro.pdf".into(),
            contents: text.unwrap(),
            summary: String::new(),
        }
    }

    async fn run_user_document_tool_test(tool: QueryDocumentsTool, provider_name: &str) {
        let args = json!({
            "filenames": ["mono2micro.pdf"],
            "query": "What problem does this paper solve?",
            "query_method": "embedding",
        });

        let result = tool.call(args).await;
        assert!(
            result.is_ok(),
            "{} QueryDocumentsTool call failed: {:?}",
            provider_name,
            result.err()
        );

        let value = result.unwrap();
        let results = value["results"]
            .as_array()
            .expect("results should be an array");
        assert!(
            !results.is_empty(),
            "{provider_name} results should not be empty"
        );

        println!(
            "{} QueryDocumentsTool test passed. Got {} file result(s).",
            provider_name,
            results.len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_user_document_tool_openai_embedding() {
        dotenv().ok();

        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for this test");
        let voyage_api_key = std::env::var("VOYAGE_AI_API_KEY")
            .expect("VOYAGE_AI_API_KEY must be set for this test");

        let ctx = Arc::new(DocumentsToolContext {
            documents: Arc::new(RwLock::new(HashMap::from([(
                "mono2micro.pdf".into(),
                Arc::new(test_pdf()),
            )]))),
            embedding_config: EmbeddingProviderConfig::OpenAI(OpenAIConfig {
                api_key,
                model: String::new(),
                max_tokens: 0,
                embedding_model: DEFAULT_OPENAI_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_OPENAI_EMBEDDING_DIM as usize,
            }),
            reranker_config: Some(RerankProviderConfig::VoyageAI(VoyageAIConfig {
                api_key: voyage_api_key,
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.to_string(),
            })),
            client: None,
        });
        let tool = QueryDocumentsTool::new(ctx);

        run_user_document_tool_test(tool, "OpenAI+VoyageAI").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_user_document_tool_voyageai_embedding() {
        dotenv().ok();

        let api_key = std::env::var("VOYAGE_AI_API_KEY")
            .expect("VOYAGE_AI_API_KEY must be set for this test");

        let voyage_config = VoyageAIConfig {
            api_key,
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_VOYAGE_RERANK_MODEL.to_string(),
        };

        let ctx = Arc::new(DocumentsToolContext {
            documents: Arc::new(RwLock::new(HashMap::from([(
                "mono2micro.pdf".into(),
                Arc::new(test_pdf()),
            )]))),
            embedding_config: EmbeddingProviderConfig::VoyageAI(voyage_config.clone()),
            reranker_config: Some(RerankProviderConfig::VoyageAI(voyage_config)),
            client: None,
        });
        let tool = QueryDocumentsTool::new(ctx);

        run_user_document_tool_test(tool, "VoyageAI").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_user_document_tool_cohere_embedding() {
        dotenv().ok();

        let api_key =
            std::env::var("COHERE_API_KEY").expect("COHERE_API_KEY must be set for this test");

        let cohere_config = CohereConfig {
            api_key,
            embedding_model: DEFAULT_COHERE_EMBEDDING_MODEL.to_string(),
            embedding_dims: DEFAULT_COHERE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_COHERE_RERANK_MODEL.to_string(),
        };

        let ctx = Arc::new(DocumentsToolContext {
            documents: Arc::new(RwLock::new(HashMap::from([(
                "mono2micro.pdf".into(),
                Arc::new(test_pdf()),
            )]))),
            embedding_config: EmbeddingProviderConfig::Cohere(cohere_config.clone()),
            reranker_config: Some(RerankProviderConfig::Cohere(cohere_config)),
            client: None,
        });
        let tool = QueryDocumentsTool::new(ctx);

        run_user_document_tool_test(tool, "Cohere").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_user_document_tool_gemini_embedding() {
        dotenv().ok();

        let gemini_api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .expect("GEMINI_API_KEY or GOOGLE_API_KEY must be set for this test");
        let voyage_api_key = std::env::var("VOYAGE_AI_API_KEY")
            .expect("VOYAGE_AI_API_KEY must be set for this test");

        let ctx = Arc::new(DocumentsToolContext {
            documents: Arc::new(RwLock::new(HashMap::from([(
                "mono2micro.pdf".into(),
                Arc::new(test_pdf()),
            )]))),
            embedding_config: EmbeddingProviderConfig::Gemini(GeminiConfig {
                api_key: gemini_api_key,
                model: String::new(),
                embedding_model: DEFAULT_GEMINI_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_GEMINI_EMBEDDING_DIM as usize,
            }),
            reranker_config: Some(RerankProviderConfig::VoyageAI(VoyageAIConfig {
                api_key: voyage_api_key,
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.to_string(),
            })),
            client: None,
        });
        let tool = QueryDocumentsTool::new(ctx);

        run_user_document_tool_test(tool, "Gemini+VoyageAI").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_user_document_tool_hybrid_openai() {
        dotenv().ok();

        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for this test");
        let voyage_api_key = std::env::var("VOYAGE_AI_API_KEY")
            .expect("VOYAGE_AI_API_KEY must be set for this test");

        let client =
            get_client_with_config(LLMClientConfig::OpenAI(zqa_rag::config::OpenAIConfig {
                api_key: api_key.clone(),
                model: DEFAULT_OPENAI_MODEL_SMALL.to_string(),
                max_tokens: DEFAULT_OPENAI_MAX_TOKENS,
                embedding_model: DEFAULT_OPENAI_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_OPENAI_EMBEDDING_DIM as usize,
            }))
            .expect("Failed to create OpenAI client");

        let _documents = test_pdf();
        let ctx = Arc::new(DocumentsToolContext {
            documents: Arc::new(RwLock::new(HashMap::from([(
                "mono2micro.pdf".into(),
                Arc::new(test_pdf()),
            )]))),
            embedding_config: EmbeddingProviderConfig::OpenAI(OpenAIConfig {
                api_key,
                model: String::new(),
                max_tokens: 0,
                embedding_model: DEFAULT_OPENAI_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_OPENAI_EMBEDDING_DIM as usize,
            }),
            reranker_config: Some(RerankProviderConfig::VoyageAI(VoyageAIConfig {
                api_key: voyage_api_key,
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.to_string(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.to_string(),
            })),
            client: Some(client),
        });
        let tool = QueryDocumentsTool::new(ctx);

        let args = serde_json::json!({
            "filenames": ["mono2micro.pdf"],
            "query": "What problem does this paper solve?",
            "query_method": "hybrid",
        });

        let result = tool.call(args).await;
        assert!(
            result.is_ok(),
            "Hybrid OpenAI QueryDocumentsTool call failed: {:?}",
            result.err()
        );

        let value = result.unwrap();
        let results = value["results"]
            .as_array()
            .expect("results should be an array");
        assert!(
            !results.is_empty(),
            "Hybrid OpenAI results should not be empty"
        );

        println!(
            "Hybrid OpenAI QueryDocumentsTool test passed. Got {} file result(s).",
            results.len()
        );
    }

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

        // Falls to min(len-1, max_summary_sec_pos) = min(1999, 1000) = 1000
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
