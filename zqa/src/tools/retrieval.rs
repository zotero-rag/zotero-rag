use std::time::Instant;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use zqa_rag::{embedding::common::EmbeddingProviderConfig, llm::tools::Tool};

use crate::utils::{
    arrow::vector_search,
    library::get_authors,
    terminal::{DIM_TEXT, RESET},
};

/// A tool to perform vector search and reranking.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct RetrievalTool {
    /// The embedding provider configuration. Note that this must be the same embedding provider
    /// used when initially creating the database.
    pub embedding_config: EmbeddingProviderConfig,
    /// The reranker provider to use.
    pub reranker_provider: String,
    /// The key used by the API to describe the tool's parameters.
    pub schema_key: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct RetrievalToolInput {
    /// Search query into LanceDB
    pub query: String,
}

impl Tool for RetrievalTool {
    fn name(&self) -> String {
        "retrieval_tool".into()
    }

    fn description(&self) -> String {
        "Retrieves relevant papers from the user's Zotero library based on a search query".into()
    }

    fn schema_key(&self) -> String {
        self.schema_key.clone()
    }

    fn parameters(&self) -> schemars::Schema {
        schema_for!(RetrievalToolInput)
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send>> {
        let start = Instant::now();
        let embedding_config = self.embedding_config.clone();
        let reranker_provider = self.reranker_provider.clone();
        Box::pin(async move {
            let input: RetrievalToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;
            let mut results = vector_search(input.query, &embedding_config, reranker_provider)
                .await
                .map_err(|e| format!("Search failed: {e}"))?;

            get_authors(&mut results).map_err(|e| format!("Failed to get authors: {e}"))?;
            log::info!(
                "{DIM_TEXT}Vector search took {}s{RESET}.",
                start.elapsed().as_secs()
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

    fn make_tool(schema_key: &str) -> RetrievalTool {
        // Build a minimal tool; the embedding config is only used in `call`, not in the metadata
        // methods, so we use a dummy VoyageAI config here.
        RetrievalTool {
            embedding_config: EmbeddingProviderConfig::VoyageAI(zqa_rag::config::VoyageAIConfig {
                api_key: String::new(),
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            }),
            reranker_provider: "voyageai".into(),
            schema_key: schema_key.into(),
        }
    }

    #[test]
    fn test_name() {
        let tool = make_tool("input_schema");
        assert_eq!(tool.name(), "retrieval_tool");
    }

    #[test]
    fn test_description() {
        let tool = make_tool("input_schema");
        assert_eq!(
            tool.description(),
            "Retrieves relevant papers from the user's Zotero library based on a search query"
        );
    }

    #[test]
    fn test_schema_key() {
        for key in ["input_schema", "parameters", "parametersJsonSchema"] {
            let tool = make_tool(key);
            assert_eq!(tool.schema_key(), key);
        }
    }

    #[test]
    fn test_parameters_contains_query_field() {
        let tool = make_tool("input_schema");
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
        let tool = make_tool("input_schema");
        // Pass an object that is missing the required `query` field.
        let result = tool.call(json!({"not_query": "value"})).await;
        assert!(result.is_err(), "Expected error on invalid args");
        let err = result.unwrap_err();
        assert!(
            err.contains("Invalid arguments"),
            "Unexpected error message: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_call_returns_results_key() {
        dotenv::dotenv().ok();
        let _ = crate::common::setup_logger(log::LevelFilter::Info);

        let tool = make_tool("input_schema");
        let result = tool.call(json!({"query": "machine learning"})).await;
        assert!(result.is_ok(), "call failed: {:?}", result.err());
        let value = result.unwrap();
        assert!(
            value.get("results").is_some(),
            "Expected 'results' key in output, got: {value}"
        );
    }
}
