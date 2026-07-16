use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use tokio::task::JoinSet;
use zqa_rag::{
    llm::{
        base::{ChatRequest, CompletionApiResponse},
        errors::LLMError,
        factory::LLMClient,
        tools::Tool,
    },
    pricing::ModelUsage,
};

use crate::{
    cli::prompts::get_extraction_prompt,
    store::common::ZoteroStore,
    utils::{library::ZoteroItem, rag::ModelResponse},
};

pub(crate) const SUMMARIZATION_TOOL_NAME: &str = "summarization_tool";

/// A tool to summarize Zotero papers with a specified ID.
#[derive(Debug, Clone)]
pub(crate) struct SummarizationTool<T: ZoteroStore> {
    pub(crate) llm_client: LLMClient,
    /// Backend for searching stored Zotero papers.
    pub(crate) store: Arc<T>,
    /// The tool's token usage
    pub(crate) usage: Arc<Mutex<ModelUsage>>,
}

impl<T> SummarizationTool<T>
where
    T: ZoteroStore,
{
    /// Create a new [`SummarizationTool`] instance, given an LLM client and a backend.
    pub fn new(llm_client: LLMClient, store: Arc<T>, usage: Arc<Mutex<ModelUsage>>) -> Self {
        Self {
            llm_client,
            store,
            usage,
        }
    }
}

/// The input to `SummarizationTool`.
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct SummarizationToolInput {
    /// The query to choose relevant passages from the papers.
    query: String,
    /// The set of IDs of papers to summarize.
    ids: Vec<String>,
}

impl<T> Tool for SummarizationTool<T>
where
    T: ZoteroStore + 'static,
{
    fn name(&self) -> String {
        SUMMARIZATION_TOOL_NAME.into()
    }

    fn description(&self) -> String {
        "A tool to summarize Zotero papers with a specified ID.".into()
    }

    fn parameters(&self) -> schemars::Schema {
        schema_for!(SummarizationToolInput)
    }

    /// Executes the summarization tool by fetching passages for the given paper IDs
    /// and summarizing each one in parallel using the configured LLM client.
    ///
    /// # Arguments
    ///
    /// * `args` - A JSON value deserializable into [`SummarizationToolInput`], containing
    ///   a query string and a list of Zotero item IDs to summarize.
    ///
    /// # Returns
    ///
    /// A JSON object with a `"summaries"` key mapping to a list of summary strings,
    /// one per successfully processed paper, and an `"errors"` key mapping to a list
    /// of error messages for papers that failed to summarize.
    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + '_>> {
        let store = Arc::clone(&self.store);
        let llm_client = self.llm_client.clone();
        Box::pin(async move {
            let input: SummarizationToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;

            let results: Vec<ZoteroItem> = store
                .get_items_by_keys(&input.ids)
                .await
                .map_err(|e| format!("Search failed: {e}"))?;

            let mut set = JoinSet::new();
            for item in results {
                let client = llm_client.clone();
                let text = item.text;
                let metadata = item.metadata;
                let query_cloned = input.query.clone();

                set.spawn(async move {
                    let request = ChatRequest {
                        chat_history: Vec::new(),
                        max_tokens: None,
                        message: get_extraction_prompt(&query_cloned, &text, &metadata),
                        reasoning: client.get_reasoning_config(),
                        tools: None, // We ARE the tool :3
                        on_tool_call: None,
                        on_text: None,
                        tool_iteration_limit: None,
                    };

                    client.send_message(&request).await
                });
            }

            let summarization_results: Vec<Result<CompletionApiResponse, LLMError>> =
                set.join_all().await;

            let mut summaries = Vec::new();
            let mut errors = Vec::new();

            for result in summarization_results {
                match result {
                    Ok(response) => {
                        let summary = ModelResponse::from(&response.content).to_string();
                        summaries.push(summary);

                        if let Ok(mut usage) = self.usage.lock() {
                            *usage += response.usage;
                        }
                    }
                    Err(e) => {
                        log::warn!("Summarization failed: {e}");
                        errors.push(e.to_string());
                    }
                }
            }

            Ok(json!({
                "summaries": summaries,
                "errors": errors
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use temp_env;
    use tempfile;
    use zqa_macros::{test_contains, test_eq, test_ok};
    use zqa_rag::providers::registry::provider_registry;

    use super::*;
    use crate::{
        cli::app::tests::{create_test_context, get_config},
        store::lance::LanceZoteroStore,
    };
    use crate::{cli::handlers::library::handle_process_cmd, config::MockConfig};

    fn make_tool(llm_responses: Vec<String>) -> SummarizationTool<LanceZoteroStore> {
        let config = get_config(MockConfig {
            responses: llm_responses,
        });
        let client = provider_registry()
            .create_llm(&config.get_generation_config().unwrap())
            .unwrap();
        let embedding_config = config.get_embedding_config().unwrap();
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]));
        let store = LanceZoteroStore::from_schema(embedding_config, schema);
        SummarizationTool::new(
            client,
            Arc::new(store),
            Arc::new(Mutex::new(ModelUsage::default())),
        )
    }

    #[test]
    fn test_name() {
        let tool = make_tool(vec![]);
        test_eq!(tool.name(), SUMMARIZATION_TOOL_NAME);
    }

    #[test]
    fn test_description() {
        let tool = make_tool(vec![]);
        test_eq!(
            tool.description(),
            "A tool to summarize Zotero papers with a specified ID."
        );
    }

    #[test]
    fn test_parameters_schema() {
        let tool = make_tool(vec![]);
        let schema = tool.parameters();

        // The schema should be a valid JSON schema for SummarizationToolInput
        let schema_json = serde_json::to_value(&schema).unwrap();

        // Check that required fields are present in the schema
        if let Some(properties) = schema_json.get("properties").and_then(|p| p.as_object()) {
            assert!(properties.contains_key("query"));
            assert!(properties.contains_key("ids"));
        } else {
            panic!("Schema should have properties object");
        }
    }

    #[test]
    fn test_input_deserialization() {
        // Test valid input
        let valid_json = json!({
            "query": "What is the main contribution?",
            "ids": ["id1", "id2", "id3"]
        });

        let input: Result<SummarizationToolInput, _> = serde_json::from_value(valid_json);
        assert!(input.is_ok());
        let input = input.unwrap();
        test_eq!(input.query, "What is the main contribution?");
        test_eq!(input.ids, vec!["id1", "id2", "id3"]);

        // Test missing query field
        let invalid_json = json!({
            "ids": ["id1", "id2"]
        });
        let input: Result<SummarizationToolInput, _> = serde_json::from_value(invalid_json);
        assert!(input.is_err());

        // Test missing ids field
        let invalid_json = json!({
            "query": "test query"
        });
        let input: Result<SummarizationToolInput, _> = serde_json::from_value(invalid_json);
        assert!(input.is_err());

        // Test empty ids array (should be valid)
        let valid_json = json!({
            "query": "test query",
            "ids": []
        });
        let input: Result<SummarizationToolInput, _> = serde_json::from_value(valid_json);
        test_ok!(input);
        test_eq!(input.unwrap().ids, Vec::<String>::new());
    }

    #[tokio::test]
    async fn test_call_invalid_args() {
        let tool = make_tool(vec![]);

        // Test with invalid JSON
        let invalid_args = json!({
            "invalid_field": "value"
        });

        let result = tool.call(invalid_args).await;
        assert!(result.is_err());
        test_contains!(result.unwrap_err(), "Invalid arguments");
    }

    #[tokio::test]
    async fn test_call_with_empty_ids() {
        let tool = make_tool(vec![]);

        // Test with empty IDs array
        let args = json!({
            "query": "test query",
            "ids": []
        });

        let result = tool.call(args).await;
        // This should succeed and return empty results
        test_ok!(result);

        let response = result.unwrap();
        let summaries = response.get("summaries").unwrap().as_array().unwrap();
        let errors = response.get("errors").unwrap().as_array().unwrap();

        // Should have no summaries since no papers were provided
        test_eq!(summaries.len(), 0);
        // Should have no errors since the operation succeeded
        test_eq!(errors.len(), 0);
    }

    #[tokio::test]
    async fn test_call_successful_summarization() {
        // Set up test database with real data
        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context(vec![]);

        // Create test database with assets data
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_process_cmd(&mut setup_ctx),
        )
        .await;

        assert!(setup_result.is_ok(), "Failed to set up test database");

        let tool = make_tool(vec![
            r"<title>Some title</title>
            <authors>Last, Name</authors>
            <reference>Some reference</reference>
            <excerpt>foobar</excerpt>
            "
            .into(),
        ]);

        let args = json!({
            "query": "What is the main contribution of this paper?",
            "ids": ["5KWS383N"]  // Known working test ID
        });
        let result =
            temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], tool.call(args)).await;
        test_ok!(result);

        let response = result.unwrap();
        assert!(
            response.get("summaries").is_some(),
            "Response should contain 'summaries' field"
        );
        assert!(
            response.get("errors").is_some(),
            "Response should contain 'errors' field"
        );

        let summaries = response.get("summaries").unwrap().as_array().unwrap();
        let errors = response.get("errors").unwrap().as_array().unwrap();

        // Should have generated at least one summary and no errors
        assert!(
            !summaries.is_empty(),
            "Should generate at least one summary"
        );
        assert!(errors.is_empty(), "Should have no errors for valid data");

        // Verify summary content
        let summary = summaries[0].as_str().unwrap();
        assert!(
            summary.contains("<title>"),
            "Summary should contain title tag"
        );
        assert!(summary.len() > 50, "Summary should be substantial");
        dbg!(&summary[..100.min(summary.len())]);
    }
}
