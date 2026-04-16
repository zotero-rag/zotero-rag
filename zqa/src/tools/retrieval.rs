use std::time::Instant;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use zqa_rag::{
    embedding::common::EmbeddingProviderConfig, llm::tools::Tool,
    reranking::common::RerankProviderConfig,
};

use crate::utils::{
    arrow::vector_search,
    library::get_authors,
    terminal::{DIM_TEXT, RESET},
};

pub(crate) const RETRIEVAL_TOOL_NAME: &str = "retrieval_tool";

/// A tool to perform vector search and reranking.
#[derive(Debug)]
pub(crate) struct RetrievalTool {
    /// The embedding provider configuration. Note that this must be the same embedding provider
    /// used when initially creating the database.
    pub(crate) embedding_config: EmbeddingProviderConfig,
    /// The reranker provider to use.
    pub(crate) reranker_config: Option<RerankProviderConfig>,
}

impl RetrievalTool {
    /// Create a new instance of the [`RetrievalTool`] given an embedding config, the name of a
    /// reranker provider, and a schema key.
    pub(crate) fn new(
        embedding_config: EmbeddingProviderConfig,
        reranker_provider: Option<RerankProviderConfig>,
    ) -> Self {
        Self {
            embedding_config,
            reranker_config: reranker_provider,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct RetrievalToolInput {
    /// Search query into LanceDB
    pub query: String,
}

impl Tool for RetrievalTool {
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
        let embedding_config = self.embedding_config.clone();
        let reranker_config = self.reranker_config.clone();

        Box::pin(async move {
            let input: RetrievalToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;
            let mut results =
                vector_search(input.query, &embedding_config, reranker_config.as_ref())
                    .await
                    .map_err(|e| format!("Search failed: {e}"))?;

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
    use super::*;
    use serde_json::json;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::embedding::common::EmbeddingProviderConfig;

    fn make_tool() -> RetrievalTool {
        let config = zqa_rag::config::VoyageAIConfig {
            api_key: String::new(),
            embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
            embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
            reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
        };

        // Build a minimal tool; the embedding config is only used in `call`, not in the metadata
        // methods, so we use a dummy VoyageAI config here.
        RetrievalTool::new(
            EmbeddingProviderConfig::VoyageAI(config.clone()),
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
