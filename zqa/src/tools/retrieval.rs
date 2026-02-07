use crate::utils::arrow::vector_search;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use zqa_rag::embedding::common::EmbeddingProviderConfig;
use zqa_rag::llm::tools::Tool;

#[derive(Debug, Clone)]
pub struct RetrievalTool {
    pub embedding_config: EmbeddingProviderConfig,
    pub reranker_provider: String,
    pub schema_key: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct RetrievalToolInput {
    /// The query to search for in the library.
    pub query: String,
}

impl Tool for RetrievalTool {
    fn name(&self) -> String {
        "retrieve_papers".into()
    }

    fn description(&self) -> String {
        "Retrieves relevant papers from the user's Zotero library based on a query.".into()
    }

    fn parameters(&self) -> schemars::Schema {
        schema_for!(RetrievalToolInput)
    }

    fn schema_key(&self) -> String {
        self.schema_key.clone()
    }

    fn call(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
        let embedding_config = self.embedding_config.clone();
        let reranker = self.reranker_provider.clone();

        Box::pin(async move {
            let input: RetrievalToolInput = serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;
            let items = vector_search(input.query, &embedding_config, reranker).await.map_err(|e| format!("Search failed: {e}"))?;

            let results: Vec<Value> = items.into_iter().map(|item| {
                json!({
                    "title": item.metadata.title,
                    "text": item.text,
                    "authors": item.metadata.authors,
                })
            }).collect();

            Ok(json!(results))
        })
    }
}
