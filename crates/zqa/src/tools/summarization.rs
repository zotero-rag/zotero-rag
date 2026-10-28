use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinSet;
use zqa_rag::llm::base::{ChatRequest, CompletionApiResponse};
use zqa_rag::llm::errors::LLMError;
use zqa_rag::llm::factory::LLMClient;
use zqa_rag::llm::tools::Tool;

use crate::cli::prompts::{get_extraction_prompt, get_extraction_system_prompt};
use crate::state::UsageMetadata;
use crate::store::common::ZoteroStore;
use crate::tools::retrieval::RETRIEVAL_TOOL_NAME;
use crate::utils::library::ZoteroItem;

pub(crate) const SUMMARIZATION_TOOL_NAME: &str = "summarization_tool";

/// A tool to summarize Zotero papers with a specified ID.
#[derive(Debug, Clone)]
pub(crate) struct SummarizationTool<T: ZoteroStore> {
    pub(crate) llm_client: LLMClient,
    /// Backend for searching stored Zotero papers.
    pub(crate) store: Arc<T>,
    /// Channel to send priced usage stats over.
    pub(crate) cost_tx: UnboundedSender<UsageMetadata>,
}

impl<T> SummarizationTool<T>
where
    T: ZoteroStore,
{
    /// Create a new [`SummarizationTool`] instance, given an LLM client and a backend.
    pub fn new(
        llm_client: LLMClient,
        store: Arc<T>,
        cost_tx: UnboundedSender<UsageMetadata>,
    ) -> Self {
        Self {
            llm_client,
            store,
            cost_tx,
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
        format!(
            "Given paper IDs from `{RETRIEVAL_TOOL_NAME}` and a query, returns passages from each \
             paper most relevant to the query. Call after `{RETRIEVAL_TOOL_NAME}` and pass all \
             promising IDs in one call; papers are processed in parallel."
        )
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
            log::debug!(
                "Summarization retrieval: requested_ids={}, found_papers={}",
                input.ids.len(),
                results.len()
            );

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
                        system_prompt: Some(get_extraction_system_prompt().to_owned()),
                        reasoning: client.get_reasoning_config(),
                        tools: None, // We ARE the tool :3
                        on_tool_call: None,
                        on_text: None,
                        on_reasoning: None,
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
                        summaries.push(response.text());

                        let _ = self.cost_tx.send(
                            UsageMetadata::from_rag_usage(
                                &response.usage,
                                self.llm_client.get_provider(),
                                &self.llm_client.get_model_name().unwrap_or_default(),
                            )
                            .await,
                        );
                    }
                    Err(e) => {
                        log::warn!("Summarization failed: {}", zqa_rag::logging::preview(&e));
                        errors.push(e.to_string());
                    }
                }
            }

            log::debug!(
                "Summarization completed: successes={}, failures={}",
                summaries.len(),
                errors.len()
            );
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
    use zqa_macros::{test_contains, test_eq, test_ok};
    use zqa_rag::providers::registry::provider_registry;

    use super::*;
    use crate::cli::handlers::library::handle_process_cmd;
    use crate::common::test_support::{TestPaths, get_config};
    use crate::config::MockConfig;
    use crate::store::lance::LanceZoteroStore;

    fn make_tool(llm_responses: Vec<String>) -> SummarizationTool<LanceZoteroStore> {
        // These tests don't check usage, so the receiver is dropped and sends are discarded.
        let (cost_tx, _) = tokio::sync::mpsc::unbounded_channel();
        make_tool_with_uri(llm_responses, None, cost_tx)
    }

    /// Like [`make_tool`], but optionally pins the store to an explicit database URI. Tests that
    /// actually query the store must pass `Some(uri)` so they read their own isolated database
    /// rather than the process-global `LANCEDB_URI` — that's what keeps them safe to run in
    /// parallel. Tests that never touch the store can use [`make_tool`].
    fn make_tool_with_uri(
        llm_responses: Vec<String>,
        db_uri: Option<&str>,
        cost_tx: UnboundedSender<UsageMetadata>,
    ) -> SummarizationTool<LanceZoteroStore> {
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
        let mut store = LanceZoteroStore::from_schema(embedding_config, schema);
        if let Some(uri) = db_uri {
            store = store.with_uri(uri);
        }
        SummarizationTool::new(client, Arc::new(store), cost_tx)
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
            format!(
                "Given paper IDs from `{RETRIEVAL_TOOL_NAME}` and a query, returns passages from each \
                 paper most relevant to the query. Call after `{RETRIEVAL_TOOL_NAME}` and pass all \
                 promising IDs in one call; papers are processed in parallel."
            )
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

    #[tokio::test(flavor = "multi_thread")]
    async fn test_call_successful_summarization() {
        // Set up an isolated test database with the toy library's data.
        let paths = TestPaths::new();
        let mut setup_ctx = paths.context(vec![]);

        let setup_result = handle_process_cmd(&mut setup_ctx).await;
        assert!(setup_result.is_ok(), "Failed to set up test database");

        let (cost_tx, mut cost_rx) = tokio::sync::mpsc::unbounded_channel();
        let tool = make_tool_with_uri(
            vec![
                r"<title>Some title</title>
            <authors>Last, Name</authors>
            <reference>Some reference</reference>
            <excerpt>foobar</excerpt>
            "
                .into(),
            ],
            Some(&paths.db_uri),
            cost_tx,
        );

        let args = json!({
            "query": "What is the main contribution of this paper?",
            "ids": ["5KWS383N"]  // Known working test ID
        });
        let result = tool.call(args).await;
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

        // The mock client reports no tokens, so only the number of updates is checked.
        let usage_updates = std::iter::from_fn(|| cost_rx.try_recv().ok()).count();
        assert_eq!(
            usage_updates,
            summaries.len(),
            "Expected one usage update per summary"
        );

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
