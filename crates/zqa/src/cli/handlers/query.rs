//! Command handlers for query operations.

use std::iter;
use std::path::Path;
use std::pin::pin;
use std::sync::{Arc, atomic};
use std::time::Instant;

use tokio::sync::mpsc::{self, UnboundedSender};
use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, ChatRequest, MessageRole};
use zqa_rag::llm::factory::{LLMClient, get_client_with_config};
use zqa_rag::llm::tools::{CallbackFn, Tool};
use zqa_rag::providers::registry::provider_registry;

use crate::cli::errors::CLIError;
use crate::cli::handlers::documents::{
    get_document_mentions, get_user_document_tools, import_document,
};
use crate::cli::prompts::{
    get_summarize_prompt, get_summarize_system_prompt, get_title_prompt, get_title_system_prompt,
};
use crate::common::Context;
use crate::io::{EngineEvent, StyledSpan, TextStyle};
use crate::state::UsageMetadata;
use crate::store::common::ZoteroStore;
use crate::tools::mixins::ToolExt;
use crate::tools::retrieval::RetrievalTool;
use crate::tools::summarization::SummarizationTool;
use crate::utils::rag::ModelResponse;

/// Reasoning and answer segments share one queue to preserve the provider's content order.
enum ResponseSegment {
    Text(String),
    Reasoning(String),
}

impl From<ResponseSegment> for EngineEvent {
    fn from(segment: ResponseSegment) -> Self {
        match segment {
            ResponseSegment::Text(text) => Self::Text {
                message: format!("{text}\n"),
            },
            ResponseSegment::Reasoning(reasoning) => Self::Reasoning {
                message: format!("{reasoning}\n"),
            },
        }
    }
}

/// Perform a vector search and emit matching titles as text events.
///
/// # Arguments
///
/// * `search_term` - The search string to run against the vector database.
/// * `ctx` - The application context, including the store, configuration, and event sender.
///
/// # Returns
///
/// `Ok(())` if the search and event publication completed without error.
///
/// # Errors
///
/// Returns a [`CLIError`] if provider configuration is invalid, vector search fails,
/// or the event receiver is closed.
pub(crate) async fn handle_search_cmd(
    search_term: String,
    ctx: &mut Context,
) -> Result<(), CLIError> {
    if search_term.is_empty() {
        ctx.emit(EngineEvent::Warning {
            message: "Please provide a search term after /search.\n".into(),
        })
        .await?;
        return Ok(());
    }

    let vector_search_start = Instant::now();
    let (search_results, _) = ctx
        .store
        .vector_search(
            search_term.clone(),
            10,
            ctx.config.get_reranker_config().as_ref(),
        )
        .await?;
    let vector_search_duration = vector_search_start.elapsed();
    ctx.emit_multiple(
        iter::once(EngineEvent::StyledText {
            spans: vec![StyledSpan {
                text: format!("Vector search completed in {vector_search_duration:.2?}\n\n"),
                style: TextStyle::Secondary,
            }],
        })
        .chain(search_results.iter().map(|item| EngineEvent::Text {
            message: format!("{}\n", item.metadata.title),
        }))
        .chain(iter::once(EngineEvent::Text {
            message: "\n".into(),
        })),
    )
    .await?;

    Ok(())
}

/// Generate a title for a conversation based on the given user query.
///
/// # Arguments
///
/// * `user_query` - The user query.
/// * `client` - A configured [`LLMClient`].
/// * `model` - The model string to use.
///
/// # Returns
///
/// A tuple consisting of the generated title and the usage statistics for the LLM call.
async fn generate_title(
    user_query: String,
    client: LLMClient,
    model: String,
) -> Option<(String, UsageMetadata)> {
    let prompt = get_title_prompt(&user_query);
    let request = ChatRequest {
        chat_history: Vec::new(),
        max_tokens: Some(20),
        message: prompt,
        system_prompt: Some(get_title_system_prompt().to_owned()),
        reasoning: None,
        tools: None,
        on_tool_call: None,
        on_text: None,
        on_reasoning: None,
        tool_iteration_limit: None,
    };

    let response = match client.send_message(&request).await {
        Ok(r) => r,
        Err(error) => {
            log::debug!(
                "Background title generation failed: {}",
                zqa_rag::logging::preview(error)
            );
            return None;
        }
    };

    let title = ModelResponse::from(&response.content).to_string();
    let title = title.trim().to_string();

    let usage = UsageMetadata::from_rag_usage(&response.usage, client.get_provider(), &model).await;
    Some((title, usage))
}

/// Build the tools available to the model for a query, each reporting on `status_tx`.
///
/// # Arguments
///
/// * `llm_client` - The client the summarization tool uses.
/// * `ctx` - The application context, for the store, configuration and imported documents.
/// * `status_tx` - The channel the tool wrappers send their status events on.
/// * `cost_tx` - The channel to send priced usage over.
///
/// # Returns
///
/// The retrieval and summarization tools, plus tools for any imported documents.
///
/// # Errors
///
/// Returns a [`CLIError`] if the document tools can't be built (see [`get_user_document_tools`]).
fn build_query_tools(
    llm_client: LLMClient,
    ctx: &mut Context,
    status_tx: UnboundedSender<EngineEvent>,
    cost_tx: UnboundedSender<UsageMetadata>,
) -> Result<Vec<Box<dyn Tool>>, CLIError> {
    let Some(embedding_config) = ctx.config.get_embedding_config() else {
        return Err(CLIError::ConfigError(
            "Embedding config is required before queries can be run.".into(),
        ));
    };

    let store = Arc::new(ctx.store.clone());
    let retrieval_tool = RetrievalTool::new(
        Arc::clone(&store),
        embedding_config,
        ctx.config.get_reranker_config(),
        ctx.path_options.library_path.clone(),
        cost_tx.clone(),
    );

    let summarization_tool = SummarizationTool::new(llm_client, store, cost_tx.clone());

    let mut tools: Vec<Box<dyn Tool>> = vec![
        Box::new(
            retrieval_tool
                .verbose(status_tx.clone())
                .timed(status_tx.clone()),
        ),
        Box::new(
            summarization_tool
                .verbose(status_tx.clone())
                .timed(status_tx.clone()),
        ),
    ];
    let document_tools = get_user_document_tools(ctx, status_tx, cost_tx)?;
    tools.extend(document_tools);

    Ok(tools)
}

/// Answer a user query using retrieval and generation over the user's Zotero library.
///
/// # Arguments
///
/// * `query` - The user query.
/// * `ctx` - The application context, including conversation state, providers, and the event sender.
///
/// # Returns
///
/// `Ok(())` if the query and response-metadata event publication completed without error.
///
/// # Errors
///
/// Returns a [`CLIError`] if configuration is invalid, document import fails,
/// provider calls fail before final response handling, or the event receiver is closed.
#[allow(clippy::too_many_lines)]
pub(crate) async fn handle_query_cmd(query: String, ctx: &mut Context) -> Result<(), CLIError> {
    ctx.emit(EngineEvent::Text {
        message: "\n".into(),
    })
    .await?;
    log::debug!(
        "Starting query: input_bytes={}, tool_iteration_limit={}",
        query.len(),
        ctx.config.tool_iteration_limit
    );

    for mention in get_document_mentions(&query) {
        let path = match import_document(ctx, Path::new(&mention)) {
            Ok(p) => p,
            Err(e) => {
                ctx.emit(EngineEvent::Error {
                    message: format!("Failed to import {mention}: {e}\n"),
                })
                .await?;
                continue;
            }
        };
        ctx.emit(EngineEvent::StatusUpdate {
            message: format!("Imported document: {path}\n"),
        })
        .await?;
    }

    let llm_client = ctx
        .config
        .get_generation_config()
        .map(|c| provider_registry().create_llm(&c))
        .transpose()?
        .ok_or(CLIError::ConfigError(
            "Failed to get LLM generation config in `run_query`".into(),
        ))?;

    // `build_query_tools` needs this config too. Checking here fails the query before the title
    // task is spawned and its tokens are spent.
    let _embedding_config = ctx
        .config
        .get_embedding_config()
        .ok_or(CLIError::ConfigError(
            "Could not get embedding config".into(),
        ))?;

    // Set up channels to communicate cost
    let (cost_tx, mut cost_rx) = mpsc::unbounded_channel::<UsageMetadata>();

    // Spawn a background title generation task from the query alone, in parallel with summarization.
    // Only generate a title if we don't already have one (i.e., first query in the conversation).
    let title_slot = Arc::clone(&ctx.state.title);
    if title_slot.lock()?.is_none()
        && let Some(small_config) = ctx.config.get_small_model_config()
        && let Ok(small_client) = get_client_with_config(&small_config)
    {
        let title_cost_tx = cost_tx.clone();
        let title_model_name = small_config.model_name().to_string();
        let query = query.clone();
        tokio::spawn(async move {
            if let Some((title, usage)) =
                generate_title(query, small_client, title_model_name).await
            {
                let _ = title_cost_tx.send(usage);
                if !title.is_empty()
                    && let Ok(mut slot) = title_slot.lock()
                {
                    *slot = Some(title);
                }
            }
        });
    }

    let (response_tx, mut response_rx) = mpsc::unbounded_channel::<ResponseSegment>();
    let (status_tx, mut status_rx) = mpsc::unbounded_channel::<EngineEvent>();
    let text_tx = response_tx.clone();
    let on_text: Arc<CallbackFn<str>> = Arc::new(move |text: &str| {
        let _ = text_tx.send(ResponseSegment::Text(text.to_string()));
    });
    let on_reasoning: Arc<CallbackFn<str>> = Arc::new(move |reasoning: &str| {
        let _ = response_tx.send(ResponseSegment::Reasoning(reasoning.to_string()));
    });

    let final_draft_start = Instant::now();
    let chat_history = Arc::clone(&ctx.state.chat_history);
    let mut query_usage = UsageMetadata::default();
    let result = {
        let tools = build_query_tools(llm_client.clone(), ctx, status_tx, cost_tx.clone())?;

        let request = {
            let history = chat_history
                .lock()
                .expect("Could not obtain lock on chat history.");
            ChatRequest {
                chat_history: history.clone(),
                max_tokens: None,
                message: get_summarize_prompt(&query),
                system_prompt: Some(get_summarize_system_prompt()),
                reasoning: llm_client.get_reasoning_config(),
                tools: Some(&tools),
                on_tool_call: None,
                on_text: Some(on_text),
                on_reasoning: Some(on_reasoning),
                tool_iteration_limit: Some(ctx.config.tool_iteration_limit),
            }
        };
        log::debug!(
            "Query ready: history_items={}, tools={}",
            request.chat_history.len(),
            tools.len()
        );

        let mut send_message = pin!(llm_client.send_message(&request));
        loop {
            tokio::select! {
                Some(segment) = response_rx.recv() => ctx.emit(segment.into()).await?,
                Some(event) = status_rx.recv() => ctx.emit(event).await?,
                Some(usage) = cost_rx.recv() => query_usage += usage,
                result = &mut send_message => break result,
            }
        }
    };
    // Once the above block ends, the remaining sender left is the title generation task's.
    drop(cost_tx);

    // Producers can race the response's completion; drain the leftovers.
    while let Ok(segment) = response_rx.try_recv() {
        ctx.emit(segment.into()).await?;
    }
    while let Ok(event) = status_rx.try_recv() {
        ctx.emit(event).await?;
    }
    while let Some(usage) = cost_rx.recv().await {
        query_usage += usage;
    }

    let final_draft_duration = final_draft_start.elapsed();

    match result {
        Ok(response) => {
            log::debug!(
                "Query completed in {final_draft_duration:.2?}: usage={:?}",
                response.total_usage()
            );
            ctx.emit(EngineEvent::StyledText {
                spans: vec![StyledSpan {
                    text: format!("Final draft completed in {final_draft_duration:.2?}\n"),
                    style: TextStyle::Secondary,
                }],
            })
            .await?;

            query_usage += UsageMetadata::from_rag_usage(
                &response.usage,
                ctx.config.model_provider,
                &ctx.config.get_generation_model_name().unwrap_or_default(),
            )
            .await;

            // Update state - re-acquire lock
            let mut history = chat_history
                .lock()
                .expect("Could not obtain lock on chat history.");
            history.push(ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::Text(query)],
            });

            history.extend(response.history_additions);
            ctx.state.dirty.store(true, atomic::Ordering::Relaxed);
        }
        Err(e) => {
            log::debug!(
                "Query failed in {final_draft_duration:.2?}: {}",
                zqa_rag::logging::preview(&e)
            );
            ctx.emit_multiple([
                EngineEvent::StyledText {
                    spans: vec![StyledSpan {
                        text: format!("Final draft failed in {final_draft_duration:.2?}\n"),
                        style: TextStyle::Secondary,
                    }],
                },
                EngineEvent::Error {
                    message: format!(
                        "Failed to call the LLM endpoint for the final response: {e}\n"
                    ),
                },
            ])
            .await?;
        }
    }

    ctx.state.usage += query_usage;
    ctx.emit(EngineEvent::TokenUsage { usage: query_usage })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use zqa_macros::{test_eq, test_ok};
    use zqa_macros_proc::retry;

    use super::{ResponseSegment, handle_query_cmd, handle_search_cmd};
    use crate::cli::handlers::library::handle_process_cmd;
    use crate::common::test_support::{TestPaths, capture_events};
    use crate::io::EngineEvent;

    /// Reasoning and answers retain distinct event types and trailing newlines.
    #[test]
    fn response_segments_keep_reasoning_separate_from_answers() {
        let reasoning = EngineEvent::from(ResponseSegment::Reasoning(
            "Compare the sources.\nThey agree.".into(),
        ));
        let answer = EngineEvent::from(ResponseSegment::Text("The answer.".into()));

        let EngineEvent::Reasoning { message } = reasoning else {
            panic!("Expected a reasoning event");
        };

        test_eq!(message, "Compare the sources.\nThey agree.\n");

        let EngineEvent::Text { message } = answer else {
            panic!("Expected a text event");
        };

        test_eq!(message, "The answer.\n");
    }

    /// A text-only response produces an answer event, not a reasoning event.
    #[test]
    fn text_response_leaves_reasoning_output_empty() {
        let event = EngineEvent::from(ResponseSegment::Text("The answer.".into()));

        let EngineEvent::Text { message } = event else {
            panic!("Expected a text event");
        };

        test_eq!(message, "The answer.\n");
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_search_only() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut setup_ctx = paths.context(vec![]);
        let result = handle_process_cmd(&mut setup_ctx).await;
        test_ok!(result);

        let mut ctx = paths.context(vec![]); // search doesn't use LLMs
        let (result, events) = capture_events(&mut ctx, async |ctx| {
            handle_search_cmd(
                "How should I oversample in defect prediction?".to_string(),
                ctx,
            )
            .await
        })
        .await;
        test_ok!(result);
        assert!(result.is_ok());

        let output_len: usize = events
            .iter()
            .map(|event| match event {
                EngineEvent::Text { message } => message.len(),
                _ => 0,
            })
            .sum();
        assert!(output_len > 20);
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_run_query() {
        dotenv::dotenv().ok();

        let paths = TestPaths::new();
        let mut setup_ctx = paths.context(vec![]);
        let _ = handle_process_cmd(&mut setup_ctx).await;

        // TODO: At some point, we'll want to add support for tool calls on these. Right now, the
        // underlying `TestClient` doesn't call `process_tool_calls
        let mut ctx = paths.context(vec!["You have papers X, Y, and Z.".into()]);
        let (result, events) = capture_events(&mut ctx, async |ctx| {
            handle_query_cmd(
                "What papers do I have about learning rate scheduling?".to_string(),
                ctx,
            )
            .await
        })
        .await;

        test_ok!(result);
        assert!(result.is_ok());
        test_eq!(
            events
                .iter()
                .filter(|event| matches!(event, EngineEvent::TokenUsage { .. }))
                .count(),
            1
        );
    }
}
