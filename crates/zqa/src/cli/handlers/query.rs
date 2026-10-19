//! Command handlers for query operations.

use std::path::Path;
use std::pin::pin;
use std::sync::atomic::{self, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::mpsc::{self, UnboundedSender};
use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, ChatRequest, MessageRole};
use zqa_rag::llm::factory::{LLMClient, get_client_with_config};
use zqa_rag::llm::tools::{CallbackFn, Tool};
use zqa_rag::pricing::{ModelUsage, get_model_pricing, get_usd_exchange_rate};
use zqa_rag::providers::registry::provider_registry;

use crate::cli::errors::CLIError;
use crate::cli::handlers::documents::{
    get_document_mentions, get_user_document_tools, import_document,
};
use crate::cli::prompts::{
    get_summarize_prompt, get_summarize_system_prompt, get_title_prompt, get_title_system_prompt,
};
use crate::common::Context;
use crate::io::EngineEvent;
use crate::state::UsageMetadata;
use crate::store::common::ZoteroStore;
use crate::tools::mixins::ToolExt;
use crate::tools::retrieval::RetrievalTool;
use crate::tools::summarization::SummarizationTool;
use crate::utils::rag::ModelResponse;
use crate::utils::terminal::{DIM_TEXT, RESET};

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

/// Given a positive number, returns a thousands separator-formatted string representation
///
/// # Arguments:
///
/// * `num` - The number to format
///
/// # Returns
///
/// The thousands-separated string
fn format_number(num: u32) -> String {
    num.to_string()
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(std::str::from_utf8)
        .collect::<Result<Vec<&str>, _>>()
        .unwrap_or_default()
        .join(",")
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
    ctx.emit(EngineEvent::StatusUpdate {
        message: format!(
            "{DIM_TEXT}Vector search completed in {vector_search_duration:.2?}{RESET}\n"
        ),
    })
    .await?;

    for item in &search_results {
        ctx.emit(EngineEvent::Text {
            message: format!("{}\n", item.metadata.title),
        })
        .await?;
    }

    ctx.emit(EngineEvent::Text {
        message: "\n".into(),
    })
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

/// For embedding and reranking, where we currently only have characters, estimate the cost if
/// possible and return the usage.
///
/// Cost lookup uses the LiteLLM JSON catalog of models, and uses `{provider_name}/{model}`.
///
/// TODO: We should probably just use the provider-sent usage directly instead of the approximation
/// here.
///
/// # Arguments
///
/// * `provider_name` - The provider name to use in the lookup.
/// * `model` - The model string to use in the lookup.
/// * `chars` - The character count to use in the estimate.
///
/// # Returns
///
/// A [`UsageMetadata`] with estimated cost if the model was found.
async fn get_model_usage(provider_name: &str, model: &str, chars: u64) -> UsageMetadata {
    let mut usage = UsageMetadata {
        #[allow(clippy::cast_possible_truncation)]
        input_tokens: (chars / 4) as u32,
        ..Default::default()
    };

    if let Some(ref p) = get_model_pricing(provider_name, model, None).await {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let cost = (p.estimate_cost(usage.into()) * 100.0) as u32;
        usage.estimated_cost += cost;
    }

    usage
}

/// Usage counters that the query tools update as they run. The handler reads them once the
/// response is complete to price the query.
struct ToolUsage {
    /// Characters embedded by the retrieval tool.
    embedding_tokens: Arc<AtomicU64>,
    /// Characters reranked by the retrieval tool.
    rerank_tokens: Arc<AtomicU64>,
    /// Usage of each LLM request made by the summarization tool.
    summarization_usage: Arc<Mutex<Vec<ModelUsage>>>,
}

/// Build the tools available to the model for a query, each reporting on `status_tx`.
///
/// # Arguments
///
/// * `llm_client` - The client the summarization tool uses.
/// * `ctx` - The application context, for the store, configuration and imported documents.
/// * `status_tx` - The channel the tool wrappers send their status events on.
///
/// # Returns
///
/// The retrieval and summarization tools, plus tools for any imported documents, and the
/// counters those tools update.
///
/// # Errors
///
/// Returns a [`CLIError`] if the document tools can't be built (see [`get_user_document_tools`]).
fn build_query_tools(
    llm_client: LLMClient,
    ctx: &mut Context,
    status_tx: UnboundedSender<EngineEvent>,
) -> Result<(Vec<Box<dyn Tool>>, ToolUsage), CLIError> {
    let store = Arc::new(ctx.store.clone());
    let reranker_config = ctx.config.get_reranker_config();
    let retrieval_tool = RetrievalTool::new(
        Arc::clone(&store),
        reranker_config,
        ctx.path_options.library_path.clone(),
    );

    let summarization_usage = Arc::new(Mutex::new(Vec::new()));
    let summarization_tool =
        SummarizationTool::new(llm_client, store, Arc::clone(&summarization_usage));
    let usage = ToolUsage {
        embedding_tokens: Arc::clone(&retrieval_tool.embedding_tokens),
        rerank_tokens: Arc::clone(&retrieval_tool.rerank_tokens),
        summarization_usage,
    };

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
    let document_tools = get_user_document_tools(ctx, status_tx)?;
    tools.extend(document_tools);

    Ok((tools, usage))
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
                    message: format!("{DIM_TEXT}Failed to import {mention}: {e}{RESET}\n"),
                })
                .await?;
                continue;
            }
        };
        ctx.emit(EngineEvent::StatusUpdate {
            message: format!("{DIM_TEXT}Imported document: {path}{RESET}\n"),
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

    let embedding_config = ctx
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
    drop(cost_tx);

    let embedding_provider_name = embedding_config.provider_name().to_string();
    let embedding_model_name = embedding_config.model_name().to_string();
    let reranker_provider_and_model = ctx
        .config
        .get_reranker_config()
        .as_ref()
        .map(|c| (c.provider_name().to_string(), c.model_name().to_string()));

    let (response_tx, mut response_rx) = mpsc::unbounded_channel::<ResponseSegment>();
    let (status_tx, mut status_rx) = mpsc::unbounded_channel::<EngineEvent>();
    let text_tx = response_tx.clone();
    let on_text: Arc<CallbackFn<str>> = Arc::new(move |text: &str| {
        let _ = text_tx.send(ResponseSegment::Text(text.to_string()));
    });
    let on_reasoning: Arc<CallbackFn<str>> = Arc::new(move |reasoning: &str| {
        let _ = response_tx.send(ResponseSegment::Reasoning(reasoning.to_string()));
    });

    let (tools, tool_usage) = build_query_tools(llm_client.clone(), ctx, status_tx)?;

    let chat_history = Arc::clone(&ctx.state.chat_history);

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

    let final_draft_start = Instant::now();
    let mut send_message = pin!(llm_client.send_message(&request));
    let result = loop {
        tokio::select! {
            Some(segment) = response_rx.recv() => ctx.emit(segment.into()).await?,
            Some(event) = status_rx.recv() => ctx.emit(event).await?,
            Some(title_cost) = cost_rx.recv() => ctx.state.usage += title_cost,
            result = &mut send_message => break result,
        }
    };
    // Producers can race the response's completion; drain the leftovers.
    while let Ok(segment) = response_rx.try_recv() {
        ctx.emit(segment.into()).await?;
    }
    while let Ok(event) = status_rx.try_recv() {
        ctx.emit(event).await?;
    }
    while let Some(usage) = cost_rx.recv().await {
        ctx.state.usage += usage;
    }

    let final_draft_duration = final_draft_start.elapsed();

    match result {
        Ok(response) => {
            log::debug!(
                "Query completed in {final_draft_duration:.2?}: usage={:?}",
                response.total_usage()
            );
            ctx.emit(EngineEvent::StatusUpdate {
                message: format!(
                    "{DIM_TEXT}Final draft completed in {final_draft_duration:.2?}{RESET}\n"
                ),
            })
            .await?;

            // Collect the usage of every request; `UsageMetadata::from_rag_usage` prices each one
            let mut request_usage = response.usage;
            request_usage.extend(
                tool_usage
                    .summarization_usage
                    .lock()
                    .map(|mut u| std::mem::take(&mut *u))?,
            );
            let usage = UsageMetadata::from_rag_usage(
                &request_usage,
                ctx.config.model_provider,
                &ctx.config.get_generation_model_name().unwrap_or_default(),
            )
            .await;
            ctx.state.usage += usage;

            // Add embedding cost to session cost
            let emb_chars = tool_usage.embedding_tokens.load(atomic::Ordering::Relaxed);
            if emb_chars > 0 {
                ctx.state.usage +=
                    get_model_usage(&embedding_provider_name, &embedding_model_name, emb_chars)
                        .await;
            }

            // Add reranker cost to session cost
            let rerank_chars_val = tool_usage.rerank_tokens.load(atomic::Ordering::Relaxed);
            if rerank_chars_val > 0
                && let Some((rerank_provider, rerank_model)) = reranker_provider_and_model
            {
                ctx.state.usage +=
                    get_model_usage(&rerank_provider, &rerank_model, rerank_chars_val).await;
            }

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
            ctx.emit(EngineEvent::StatusUpdate {
                message: format!(
                    "{DIM_TEXT}Final draft failed in {final_draft_duration:.2?}{RESET}\n"
                ),
            })
            .await?;

            ctx.emit(EngineEvent::Error {
                message: format!("Failed to call the LLM endpoint for the final response: {e}\n"),
            })
            .await?;
        }
    }

    ctx.emit(EngineEvent::Text {
        message: "\n-----\n".into(),
    })
    .await?;
    ctx.emit(EngineEvent::Text {
        message: format!("{DIM_TEXT}Total token usage:{RESET}\n"),
    })
    .await?;
    ctx.emit(EngineEvent::Text {
        message: format!(
            "\t{DIM_TEXT}Input tokens: {}{RESET}\n",
            format_number(ctx.state.usage.input_tokens)
        ),
    })
    .await?;
    ctx.emit(EngineEvent::Text {
        message: format!(
            "\t{DIM_TEXT}Output tokens: {}{RESET}\n\n",
            format_number(ctx.state.usage.output_tokens)
        ),
    })
    .await?;

    let cost = f64::from(ctx.state.usage.estimated_cost) / 100.0;
    if cost > 0.0 {
        let rate = *ctx
            .state
            .exchange_rate
            .get_or_init(|| get_usd_exchange_rate(&ctx.config.currency))
            .await;
        let (cost, currency) = match rate {
            Some(rate) => (cost * rate, ctx.config.currency.to_ascii_uppercase()),
            None => (cost, String::from("USD")),
        };

        ctx.emit(EngineEvent::Text {
            message: format!(
                "\t{DIM_TEXT}Session cost: {cost:.4} {currency} ({}){RESET}\n",
                ctx.config.get_generation_model_name().unwrap_or_default()
            ),
        })
        .await?;
    }

    ctx.emit(EngineEvent::Text {
        message: "\n".into(),
    })
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
        assert!(events.iter().any(|event| matches!(
            event,
            EngineEvent::Text { message } if message.contains("Total token usage:")
        )));
    }
}
