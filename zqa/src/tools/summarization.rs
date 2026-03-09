use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use std::{future::Future, pin::Pin};
use tokio::task::JoinSet;
use zqa_rag::{
    llm::{
        base::{ApiClient, ChatRequest, CompletionApiResponse},
        errors::LLMError,
        factory::LLMClient,
        tools::Tool,
    },
    vector::lance::search_by_column,
};

use crate::{
    cli::prompts::get_extraction_prompt,
    utils::{
        arrow::DbFields,
        library::{ZoteroItem, ZoteroItemSet},
        rag::ModelResponse,
    },
};

/// A tool to summarize Zotero papers with a specified ID.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct SummarizationTool {
    pub(crate) llm_client: LLMClient,
    /// The key used by the API to describe the tool's parameters.
    pub(crate) schema_key: String,
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
        "summarization_tool".into()
    }

    fn description(&self) -> String {
        "A tool to summarize Zotero papers with a specified ID.".into()
    }

    fn schema_key(&self) -> String {
        self.schema_key.clone()
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

            let results = search_by_column(DbFields::LibraryKey.as_ref(), &input.ids)
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
                        tools: None, // We ARE the tool :3
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
                        let summary = Into::<ModelResponse>::into(response.content).to_string();
                        summaries.push(summary);
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

    use super::*;
    use crate::cli::app::process;
    use crate::cli::app::tests::create_test_context;
    use serde_json::json;
    use temp_env;
    use tempfile;
    use zqa_macros::{test_contains, test_eq, test_ok};
    use zqa_rag::{
        config::AnthropicConfig,
        constants::DEFAULT_ANTHROPIC_MODEL_SMALL,
        llm::{
            anthropic::AnthropicClient, factory::LLMClient, http_client::ReqwestClient,
            tools::ANTHROPIC_SCHEMA_KEY,
        },
    };

    fn make_tool(schema_key: &str) -> SummarizationTool {
        let client = AnthropicClient::<ReqwestClient>::with_config(AnthropicConfig {
            api_key: env::var("ANTHROPIC_API_KEY").unwrap(),
            model: DEFAULT_ANTHROPIC_MODEL_SMALL.into(),
            max_tokens: 8192,
        });

        SummarizationTool {
            llm_client: LLMClient::Anthropic(client),
            schema_key: schema_key.to_string(),
        }
    }

    #[test]
    fn test_name() {
        let tool = make_tool("input_schema");
        test_eq!(tool.name(), "summarization_tool");
    }

    #[test]
    fn test_description() {
        let tool = make_tool("input_schema");
        test_eq!(
            tool.description(),
            "A tool to summarize Zotero papers with a specified ID."
        );
    }

    #[test]
    fn test_schema_key() {
        for key in ["input_schema", "parameters", "parametersJsonSchema"] {
            let tool = make_tool(key);
            test_eq!(tool.schema_key(), key);
        }
    }

    #[test]
    fn test_parameters_schema() {
        let tool = make_tool("input_schema");
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
        let tool = make_tool("input_schema");

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
        let tool = make_tool("input_schema");

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
            process(&mut setup_ctx),
        )
        .await;

        assert!(setup_result.is_ok(), "Failed to set up test database");

        let tool = make_tool(ANTHROPIC_SCHEMA_KEY);

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
