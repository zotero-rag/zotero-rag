use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc::UnboundedSender;
use zqa_rag::embedding::common::EmbeddingProviderConfig;
use zqa_rag::llm::tools::Tool;
use zqa_rag::reranking::common::RerankProviderConfig;

use crate::state::{UsageMetadata, get_model_usage};
use crate::store::common::ZoteroStore;
use crate::tools::summarization::SUMMARIZATION_TOOL_NAME;
use crate::utils::library::get_authors;

pub(crate) const RETRIEVAL_TOOL_NAME: &str = "retrieval_tool";

/// A tool to perform vector search and reranking.
#[derive(Debug)]
pub(crate) struct RetrievalTool<T>
where
    T: ZoteroStore,
{
    /// The vector store abstraction
    pub(crate) store: Arc<T>,
    /// The embedding provider config to use.
    pub(crate) embedding_config: EmbeddingProviderConfig,
    /// The reranker provider to use.
    pub(crate) reranker_config: Option<RerankProviderConfig>,
    /// Optional override for the Zotero library directory, used when enriching results with author
    /// metadata. When `None`, the path is resolved from the environment.
    pub(crate) library_path: Option<PathBuf>,
    /// Sender for the priced usage of each search.
    cost_tx: UnboundedSender<UsageMetadata>,
}

impl<T> RetrievalTool<T>
where
    T: ZoteroStore,
{
    /// Create a new instance of the [`RetrievalTool`] given a backend, reranker config, and an
    /// optional Zotero library path override (see [`RetrievalTool::library_path`]).
    pub(crate) fn new(
        store: Arc<T>,
        embedding_config: EmbeddingProviderConfig,
        reranker_provider: Option<RerankProviderConfig>,
        library_path: Option<PathBuf>,
        cost_tx: UnboundedSender<UsageMetadata>,
    ) -> Self {
        Self {
            store,
            embedding_config,
            reranker_config: reranker_provider,
            library_path,
            cost_tx,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct RetrievalToolInput {
    /// Search query into LanceDB
    pub query: String,
}

impl<T> Tool for RetrievalTool<T>
where
    T: ZoteroStore + 'static,
{
    fn name(&self) -> String {
        RETRIEVAL_TOOL_NAME.into()
    }

    fn description(&self) -> String {
        format!(
            "Searches the user's Zotero library and returns the top 10 matching papers as metadata only: \
             title, authors, and paper ID. To retrieve paper passages, call `{SUMMARIZATION_TOOL_NAME}` \
             with promising IDs."
        )
    }

    fn parameters(&self) -> schemars::Schema {
        schema_for!(RetrievalToolInput)
    }

    /// Executes a vector search against LanceDB using the provided query arguments,
    /// then enriches the results with author metadata.
    ///
    /// # Arguments
    ///
    /// * `args` - A JSON value expected to deserialize into [`RetrievalToolInput`],
    ///   containing a `query` string.
    ///
    /// # Returns
    ///
    /// A JSON object with a `results` key mapping to a list of formatted strings,
    /// each containing the title, authors, and item ID of a matching paper.
    /// Returns an error string if argument deserialization, vector search, or author
    /// retrieval fails.
    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + '_>> {
        let reranker_config = self.reranker_config.clone();
        let store = Arc::clone(&self.store);
        let library_path = self.library_path.clone();

        Box::pin(async move {
            let input: RetrievalToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;
            let (results, stats) = store
                .vector_search(input.query, 10, reranker_config.as_ref())
                .await
                .map_err(|e| format!("Search failed: {e}"))?;

            // Calculate priced usage
            let mut usage = get_model_usage(
                self.embedding_config.provider_name(),
                self.embedding_config.model_name(),
                stats.embedding_tokens as u64,
            )
            .await;
            if let Some(reranker) = &reranker_config
                && stats.rerank_tokens > 0
            {
                usage += get_model_usage(
                    reranker.provider_name(),
                    reranker.model_name(),
                    stats.rerank_tokens as u64,
                )
                .await;
            }

            // A send fails only when the handler is gone, and then nobody is counting.
            let _ = self.cost_tx.send(usage);

            let author_lookup_start = Instant::now();
            // While Zotero is open, this copies its database and may wait between attempts.
            let (results, author_result) = tokio::task::spawn_blocking(move || {
                let mut results = results;
                let author_result = get_authors(&mut results, library_path.as_deref());
                (results, author_result)
            })
            .await
            .map_err(|e| format!("Author lookup failed: {e}"))?;
            log::debug!(
                "Zotero author lookup took {:.2?}",
                author_lookup_start.elapsed()
            );
            author_result.map_err(|e| {
                log::warn!("Failed to get authors: {e}");
                format!("Failed to get authors: {e}")
            })?;

            let tool_results = results
                .iter()
                .map(|item| {
                    let authors: String = item
                        .metadata
                        .authors
                        .as_deref()
                        .map_or("Unknown".into(), |v| v.join(", "));

                    format!(
                        "Title: {}\nAuthors: {}\nItem ID: {}",
                        item.metadata.title, authors, item.metadata.library_key
                    )
                })
                .collect::<Vec<_>>();

            Ok(json!({
                "results": tool_results
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::embedding::common::EmbeddingProviderConfig;

    use super::*;
    use crate::LanceZoteroStore;

    fn make_tool() -> RetrievalTool<LanceZoteroStore> {
        let config = zqa_rag::config::VoyageAIConfig {
            api_key: String::new(),
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            max_concurrent_requests: zqa_rag::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
            max_retries: zqa_rag::constants::DEFAULT_MAX_RETRIES,
        };
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]));
        let embedding_config = EmbeddingProviderConfig::VoyageAI(config.clone());
        let store = LanceZoteroStore::from_schema(embedding_config.clone(), schema);

        // These tests don't check usage, so the receiver is dropped and sends are discarded.
        let (cost_tx, _) = tokio::sync::mpsc::unbounded_channel();
        RetrievalTool::new(
            Arc::new(store),
            embedding_config,
            Some(RerankProviderConfig::VoyageAI(config)),
            None,
            cost_tx,
        )
    }

    #[test]
    fn test_name() {
        let tool = make_tool();
        assert_eq!(tool.name(), RETRIEVAL_TOOL_NAME);
    }

    #[test]
    fn test_description() {
        let tool = make_tool();
        assert_eq!(
            tool.description(),
            format!(
                "Searches the user's Zotero library and returns the top 10 matching papers as metadata only: \
                 title, authors, and paper ID. To retrieve paper passages, call `{SUMMARIZATION_TOOL_NAME}` \
                 with promising IDs."
            )
        );
    }

    #[test]
    fn test_parameters_contains_query_field() {
        let tool = make_tool();
        let schema = tool.parameters();
        let schema_value = serde_json::to_value(&schema).unwrap();
        let properties = &schema_value["properties"];
        assert!(
            properties.get("query").is_some(),
            "Expected 'query' in parameters schema, got: {schema_value}"
        );
    }

    #[tokio::test]
    async fn test_call_returns_error_on_invalid_args() {
        let tool = make_tool();
        // Pass an object that is missing the required `query` field.
        let result = tool.call(json!({"not_query": "value"})).await;
        assert!(result.is_err(), "Expected error on invalid args");
        let err = result.unwrap_err();
        assert!(
            err.contains("Invalid arguments"),
            "Unexpected error message: {err}"
        );
    }
}
