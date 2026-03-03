use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use zqa_rag::{embedding::common::EmbeddingProviderConfig, llm::tools::Tool};

use crate::utils::{arrow::vector_search, library::ZoteroItem};

#[derive(Debug)]
pub struct RetrievalTool {
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
        let embedding_config = self.embedding_config.clone();
        let reranker_provider = self.reranker_provider.clone();

        Box::pin(async move {
            let input: RetrievalToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;
            let results = vector_search(input.query, &embedding_config, reranker_provider)
                .await
                .map_err(|e| format!("Search failed: {e}"))?;

            let tool_results: Vec<String> = results
                .into_iter()
                .filter_map(|item| serde_json::to_string_pretty::<ZoteroItem>(&item).ok())
                .collect();

            Ok(json!({
                "results": tool_results
            }))
        })
    }
}
