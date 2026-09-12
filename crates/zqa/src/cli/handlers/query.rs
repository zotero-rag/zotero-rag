//! Command handlers for query operations.

use std::io::Write;
use std::path::Path;
use std::pin::pin;
use std::sync::{Arc, Mutex, atomic};
use std::time::Instant;

use tokio::sync::mpsc;
use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, ChatRequest, MessageRole};
use zqa_rag::llm::factory::get_client_with_config;
use zqa_rag::llm::tools::{CallbackFn, Tool};
use zqa_rag::pricing::{ModelUsage, get_model_pricing};
use zqa_rag::providers::registry::provider_registry;

use crate::cli::errors::CLIError;
use crate::cli::handlers::documents::{
    get_document_mentions, get_user_document_tools, import_document,
};
use crate::cli::prompts::{
    get_summarize_prompt, get_summarize_system_prompt, get_title_prompt, get_title_system_prompt,
};
use crate::common::Context;
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

impl ResponseSegment {
    /// Write answer text to stdout and dimmed reasoning to stderr.
    ///
    /// The GUI maps stdout to answer rows and stderr to muted status rows. Routing reasoning to
    /// stderr reuses that styling and keeps reasoning out of stdout captures. The tradeoff is that
    /// reasoning shares diagnostic output and is hidden when stderr is discarded.
    ///
    /// TODO: Move to typed output events so the GUI can render reasoning rows directly and the CLI
    /// can print reasoning dimmed to stdout, with diagnostics on stderr.
    ///
    /// # Arguments
    ///
    /// * `out` - The answer output stream.
    /// * `err` - The diagnostic output stream, rendered as muted text by the GUI.
    ///
    /// # Errors
    ///
    /// * Returns an I/O error if writing the segment fails.
    fn write_to(&self, out: &mut impl Write, err: &mut impl Write) -> std::io::Result<()> {
        match self {
            Self::Text(text) => writeln!(out, "{text}"),
            Self::Reasoning(reasoning) => writeln!(err, "{DIM_TEXT}{reasoning}{RESET}"),
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

/// Perform a vector search and print matching titles.
///
/// # Arguments
///
/// * `search_term` - The search string to run against the vector database.
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the search completed and results were written successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if provider configuration is invalid, vector search fails,
/// or writing to an output stream fails.
pub(crate) async fn handle_search_cmd<O, E>(
    search_term: String,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    if search_term.is_empty() {
        writeln!(&mut ctx.err, "Please provide a search term after /search.")?;
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
    writeln!(
        &mut ctx.err,
        "{DIM_TEXT}Vector search completed in {vector_search_duration:.2?}{RESET}"
    )?;

    for item in &search_results {
        writeln!(&mut ctx.out, "{}", item.metadata.title)?;
    }
    writeln!(&mut ctx.out)?;

    Ok(())
}

/// Answer a user query using retrieval and generation over the user's Zotero library.
///
/// # Arguments
///
/// * `query` - The user query.
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the query was processed and response metadata was written successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if configuration is invalid, document import fails,
/// provider calls fail before final response handling, or writing to output streams fails.
#[allow(clippy::too_many_lines)]
pub(crate) async fn handle_query_cmd<O, E>(
    query: String,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    writeln!(&mut ctx.out)?;
    log::debug!(
        "Starting query: input_bytes={}, tool_iteration_limit={}",
        query.len(),
        ctx.config.tool_iteration_limit
    );

    for mention in get_document_mentions(&query) {
        let path = match import_document(ctx, Path::new(&mention)) {
            Ok(p) => p,
            Err(e) => {
                writeln!(
                    &mut ctx.err,
                    "{DIM_TEXT}Failed to import {mention}: {e}{RESET}"
                )?;
                continue;
            }
        };
        writeln!(&mut ctx.err, "{DIM_TEXT}Imported document: {path}{RESET}")?;
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
    let reranker_config = ctx.config.get_reranker_config();

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
        let config_clone = ctx.config.clone();
        let title_model_name = small_config.model_name().to_owned();
        let prompt = get_title_prompt(&query);
        tokio::spawn(async move {
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
            if let Ok(response) = small_client
                .send_message(&request)
                .await
                .inspect_err(|error| {
                    log::debug!(
                        "Background title generation failed: {}",
                        zqa_rag::logging::preview(error)
                    );
                })
            {
                let title = ModelResponse::from(&response.content).to_string();
                let title = title.trim().to_string();
                if !title.is_empty()
                    && let Ok(mut slot) = title_slot.lock()
                {
                    *slot = Some(title);
                }

                let usage = UsageMetadata::from_rag_usage(
                    response.usage,
                    config_clone.model_provider,
                    &title_model_name,
                )
                .await;
                let _ = title_cost_tx.send(usage);
            }
        });
    }
    drop(cost_tx);

    let embedding_provider_name = embedding_config.provider_name().to_string();
    let embedding_model_name = embedding_config.model_name().to_string();
    let reranker_provider_and_model = reranker_config
        .as_ref()
        .map(|c| (c.provider_name().to_string(), c.model_name().to_string()));

    let store_arc = Arc::new(ctx.store.clone());
    let retrieval_tool = RetrievalTool::new(
        std::sync::Arc::clone(&store_arc),
        reranker_config,
        ctx.path_options.library_path.clone(),
    );
    let retrieval_embedding_tokens = Arc::clone(&retrieval_tool.embedding_tokens);
    let retrieval_rerank_tokens = Arc::clone(&retrieval_tool.rerank_tokens);

    let (response_tx, mut response_rx) = mpsc::unbounded_channel::<ResponseSegment>();
    let (status_tx, mut status_rx) = mpsc::unbounded_channel::<String>();
    let text_tx = response_tx.clone();
    let on_text: Arc<CallbackFn<str>> = Arc::new(move |text: &str| {
        let _ = text_tx.send(ResponseSegment::Text(text.to_string()));
    });
    let on_reasoning: Arc<CallbackFn<str>> = Arc::new(move |reasoning: &str| {
        let _ = response_tx.send(ResponseSegment::Reasoning(reasoning.to_string()));
    });

    // Since `Box::new` moves the tool but we still need the modified usage after the tool runs, we
    // pass in an `Arc` that we create here. We use the `zqa-rag` struct `ModelUsage` since that
    // doesn't require a `Config` object.
    let summarization_usage = Arc::new(Mutex::new(ModelUsage::default()));
    let summarization_tool = SummarizationTool::new(
        llm_client.clone(),
        store_arc,
        Arc::clone(&summarization_usage),
    );
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
            reasoning: ctx.config.get_reasoning_config(),
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
            Some(segment) = response_rx.recv() => segment.write_to(&mut ctx.out, &mut ctx.err)?,
            Some(line) = status_rx.recv() => writeln!(ctx.err, "{line}")?,
            Some(title_cost) = cost_rx.recv() => ctx.state.usage += title_cost,
            result = &mut send_message => break result,
        }
    };
    // Both producers can race the response's completion; drain the leftovers.
    while let Ok(segment) = response_rx.try_recv() {
        segment.write_to(&mut ctx.out, &mut ctx.err)?;
    }
    while let Ok(line) = status_rx.try_recv() {
        writeln!(ctx.err, "{line}")?;
    }
    while let Some(usage) = cost_rx.recv().await {
        ctx.state.usage += usage;
    }

    let final_draft_duration = final_draft_start.elapsed();

    match result {
        Ok(response) => {
            log::debug!(
                "Query completed in {final_draft_duration:.2?}: usage={:?}",
                response.usage
            );
            writeln!(
                &mut ctx.err,
                "{DIM_TEXT}Final draft completed in {final_draft_duration:.2?}{RESET}"
            )?;

            // Accumulate token usage counts, then compute pricing using `UsageMetadata::from_rag_usage`
            let total_usage = response.usage + summarization_usage.lock().map(|u| *u)?;
            let usage = UsageMetadata::from_rag_usage(
                total_usage,
                ctx.config.model_provider,
                &ctx.config.get_generation_model_name().unwrap_or_default(),
            )
            .await;
            ctx.state.usage += usage;

            // Add embedding cost to session cost
            let emb_chars = retrieval_embedding_tokens.load(atomic::Ordering::Relaxed);
            if emb_chars > 0 {
                let emb_pricing =
                    get_model_pricing(&embedding_provider_name, &embedding_model_name, None).await;

                if let Some(ref p) = emb_pricing {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let emb_cost = (p.estimate_cost(ModelUsage {
                        // TODO: Do better
                        input_tokens: (emb_chars / 4) as u32,
                        ..Default::default()
                    }) * 100.0) as u32;
                    ctx.state.usage.estimated_cost += emb_cost;
                }
            }

            // Add reranker cost to session cost
            let rerank_chars_val = retrieval_rerank_tokens.load(atomic::Ordering::Relaxed);
            if rerank_chars_val > 0
                && let Some((rerank_provider, rerank_model)) = reranker_provider_and_model
            {
                let rerank_pricing = get_model_pricing(&rerank_provider, &rerank_model, None).await;

                if let Some(ref p) = rerank_pricing {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let rerank_cost = (p.estimate_cost(ModelUsage {
                        // TODO: Do better
                        input_tokens: (rerank_chars_val / 4) as u32,
                        ..Default::default()
                    }) * 100.0) as u32;

                    ctx.state.usage.estimated_cost += rerank_cost;
                }
            }

            // Update state - re-acquire lock
            let mut history = chat_history
                .lock()
                .expect("Could not obtain lock on chat history.");
            history.push(ChatHistoryItem {
                role: MessageRole::User,
                content: vec![ChatHistoryContent::Text(query.clone())],
            });

            history.extend(response.history_additions);
            ctx.state.dirty.store(true, atomic::Ordering::Relaxed);
        }
        Err(e) => {
            log::debug!(
                "Query failed in {final_draft_duration:.2?}: {}",
                zqa_rag::logging::preview(&e)
            );
            writeln!(
                &mut ctx.err,
                "{DIM_TEXT}Final draft failed in {final_draft_duration:.2?}{RESET}"
            )?;

            writeln!(
                &mut ctx.err,
                "Failed to call the LLM endpoint for the final response: {e}"
            )?;
        }
    }

    writeln!(&mut ctx.out, "\n-----")?;
    writeln!(&mut ctx.out, "{DIM_TEXT}Total token usage:{RESET}")?;
    writeln!(
        &mut ctx.out,
        "\t{DIM_TEXT}Input tokens: {}{RESET}",
        format_number(ctx.state.usage.input_tokens)
    )?;
    writeln!(
        &mut ctx.out,
        "\t{DIM_TEXT}Output tokens: {}{RESET}\n",
        format_number(ctx.state.usage.output_tokens)
    )?;

    let cost = f64::from(ctx.state.usage.estimated_cost) / 100.0;
    if cost > 0.0 {
        writeln!(
            &mut ctx.out,
            "\t{DIM_TEXT}Session cost: ${cost:.4} ({}){RESET}",
            ctx.config.get_generation_model_name().unwrap_or_default()
        )?;
    }
    writeln!(&mut ctx.out)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use zqa_macros::test_ok;
    use zqa_macros_proc::retry;

    use super::{ResponseSegment, handle_query_cmd, handle_search_cmd};
    use crate::cli::handlers::library::handle_process_cmd;
    use crate::common::test_support::TestPaths;

    /// Reasoning is dimmed on stderr, and answer text remains unformatted on stdout.
    #[test]
    fn response_segments_keep_reasoning_separate_from_answers() {
        let mut out = Vec::new();
        let mut err = Vec::new();

        ResponseSegment::Reasoning("Compare the sources.\nThey agree.".into())
            .write_to(&mut out, &mut err)
            .unwrap();
        ResponseSegment::Text("The answer.".into())
            .write_to(&mut out, &mut err)
            .unwrap();

        assert_eq!(out, b"The answer.\n");
        assert_eq!(err, b"\x1b[2mCompare the sources.\nThey agree.\x1b[0m\n");
    }

    /// A text-only response does not produce diagnostic output.
    #[test]
    fn text_response_leaves_reasoning_output_empty() {
        let mut out = Vec::new();
        let mut err = Vec::new();

        ResponseSegment::Text("The answer.".into())
            .write_to(&mut out, &mut err)
            .unwrap();

        assert_eq!(out, b"The answer.\n");
        assert!(err.is_empty());
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
        let result = handle_search_cmd(
            "How should I oversample in defect prediction?".to_string(),
            &mut ctx,
        )
        .await;
        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.len() > 20);
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
        let result = handle_query_cmd(
            "What papers do I have about learning rate scheduling?".to_string(),
            &mut ctx,
        )
        .await;

        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Total token usage:"));
    }
}
