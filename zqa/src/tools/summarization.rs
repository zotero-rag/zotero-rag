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
        base::{ApiClient, ChatRequest, CompletionApiResponse},
        errors::LLMError,
        factory::LLMClient,
        tools::Tool,
    },
    vector::backends::{backend::VectorBackend, lance::LanceBackend},
};

use crate::{
    cli::prompts::get_extraction_prompt,
    utils::{
        arrow::DbFields,
        library::{ZoteroItem, ZoteroItemSet},
        rag::ModelResponse,
    },
};

pub(crate) const SUMMARIZATION_TOOL_NAME: &str = "summarization_tool";

/// A tool to summarize Zotero papers with a specified ID.
#[derive(Debug, Clone)]
pub(crate) struct SummarizationTool {
    pub(crate) llm_client: LLMClient,
    /// Backend for searching stored Zotero papers.
    pub(crate) backend: LanceBackend,
    /// The input tokens used
    pub(crate) input_tokens: Arc<Mutex<u32>>,
    /// The output tokens used
    pub(crate) output_tokens: Arc<Mutex<u32>>,
}

impl SummarizationTool {
    /// Create a new [`SummarizationTool`] instance, given an LLM client and a backend.
    pub fn new(llm_client: LLMClient, backend: LanceBackend) -> Self {
        Self {
            llm_client,
            backend,
            input_tokens: Arc::new(Mutex::new(0)),
            output_tokens: Arc::new(Mutex::new(0)),
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

impl Tool for SummarizationTool {
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
    fn call<'a>(
        &'a self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>> {
        Box::pin(async move {
            let input: SummarizationToolInput =
                serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;

            let results = self
                .backend
                .search_by_column(DbFields::LibraryKey.as_ref(), &input.ids)
                .await
                .map_err(|e| format!("Search failed: {e}"))?;

            let batches: ZoteroItemSet = results.into();
            let items: Vec<ZoteroItem> = batches.into();

            let mut set = JoinSet::new();
            for item in items {
                let client = self.llm_client.clone();
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

                        // Update token counts (with error handling for mutex poisoning)
                        if let Ok(mut input_tokens) = self.input_tokens.lock() {
                            *input_tokens += response.input_tokens;
                        }
                        if let Ok(mut output_tokens) = self.output_tokens.lock() {
                            *output_tokens += response.output_tokens;
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
    use std::env;
    use std::sync::Arc;

    use serde_json::json;
    use temp_env;
    use tempfile;
    use zqa_macros::{test_contains, test_eq, test_ok};
    use zqa_rag::{
        config::{AnthropicConfig, LLMClientConfig},
        constants::DEFAULT_ANTHROPIC_MODEL_SMALL,
        llm::factory::get_client_with_config,
        vector::backends::lance::LanceBackend,
    };

    use super::*;
    use crate::cli::app::tests::{create_test_context, get_config};
    use crate::cli::handlers::library::handle_process_cmd;

    fn make_tool() -> SummarizationTool {
        let client = get_client_with_config(&LLMClientConfig::Anthropic(AnthropicConfig {
            api_key: env::var("ANTHROPIC_API_KEY").unwrap(),
            model: DEFAULT_ANTHROPIC_MODEL_SMALL.into(),
            max_tokens: 8192,
            reasoning_budget: None,
        }))
        .unwrap();

        let config = get_config();
        let embedding_config = config.get_embedding_config().unwrap();
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
        ]));
        let backend = LanceBackend::new(embedding_config, schema, "pdf_text".into());
        SummarizationTool::new(client, backend)
    }

    #[test]
    fn test_name() {
        let tool = make_tool();
        test_eq!(tool.name(), SUMMARIZATION_TOOL_NAME);
    }

    #[test]
    fn test_description() {
        let tool = make_tool();
        test_eq!(
            tool.description(),
            "A tool to summarize Zotero papers with a specified ID."
        );
    }

    #[test]
    fn test_parameters_schema() {
        let tool = make_tool();
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
        let tool = make_tool();

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
        let tool = make_tool();

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

        let mut setup_ctx = create_test_context();

        // Create test database with assets data
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_process_cmd(&mut setup_ctx),
        )
        .await;

        assert!(setup_result.is_ok(), "Failed to set up test database");

        let tool = make_tool();

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
