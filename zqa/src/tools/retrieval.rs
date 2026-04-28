use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use zqa_rag::{llm::tools::Tool, reranking::common::RerankProviderConfig};

use crate::store::common::ZoteroStore;
use crate::utils::{
    library::get_authors,
    terminal::{DIM_TEXT, RESET},
};

pub(crate) const RETRIEVAL_TOOL_NAME: &str = "retrieval_tool";

/// A tool to perform vector search and reranking.
#[derive(Debug)]
pub(crate) struct RetrievalTool<T>
where
    T: ZoteroStore,
{
    /// The vector store abstraction
    pub(crate) store: Arc<T>,
    /// The reranker provider to use.
    pub(crate) reranker_config: Option<RerankProviderConfig>,
    /// Accumulated token count of text sent to the embedding API across all calls.
    pub(crate) embedding_tokens: Arc<AtomicU64>,
    /// Accumulated token count of text sent to the reranker API across all calls.
    pub(crate) rerank_tokens: Arc<AtomicU64>,
}

impl<T> RetrievalTool<T>
where
    T: ZoteroStore,
{
    /// Create a new instance of the [`RetrievalTool`] given a backend and reranker config.
    pub(crate) fn new(store: Arc<T>, reranker_provider: Option<RerankProviderConfig>) -> Self {
        Self {
            store,
            reranker_config: reranker_provider,
            embedding_tokens: Arc::new(AtomicU64::new(0)),
            rerank_tokens: Arc::new(AtomicU64::new(0)),
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
        "Retrieves relevant papers from the user's Zotero library based on a search query".into()
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
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + '_>>
    {
        let start = Instant::now();
        let reranker_config = self.reranker_config.clone();
        let embedding_tokens = Arc::clone(&self.embedding_tokens);
        let rerank_tokens = Arc::clone(&self.rerank_tokens);
        let store = Arc::clone(&self.store);

        Box::pin(async move {
            let input: RetrievalToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;
            let (mut results, stats) = store
                .vector_search(input.query, 10, reranker_config.as_ref())
                .await
                .map_err(|e| format!("Search failed: {e}"))?;
            embedding_tokens.fetch_add(stats.embedding_tokens as u64, Ordering::Relaxed);
            rerank_tokens.fetch_add(stats.rerank_tokens as u64, Ordering::Relaxed);

            get_authors(&mut results).map_err(|e| format!("Failed to get authors: {e}"))?;
            log::info!(
                "{DIM_TEXT}Vector search took {}ms{RESET}.",
                start.elapsed().as_millis()
            );

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
                        &item.metadata.title, authors, &item.metadata.library_key
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
        };
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]));
        let store = LanceZoteroStore::from_schema(
            EmbeddingProviderConfig::VoyageAI(config.clone()),
            schema,
        );
        RetrievalTool::new(
            Arc::new(store),
            Some(RerankProviderConfig::VoyageAI(config)),
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
            "Retrieves relevant papers from the user's Zotero library based on a search query"
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
