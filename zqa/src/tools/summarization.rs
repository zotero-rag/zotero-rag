use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;
use std::pin::Pin;
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
