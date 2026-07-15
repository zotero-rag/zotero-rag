//! Command handlers for query operations.

use std::{
    io::Write,
    path::Path,
    pin::pin,
    sync::{Arc, Mutex, atomic},
    time::Instant,
};

use tokio::sync::mpsc;
use zqa_rag::{
    llm::{
        base::{ApiClient, ChatHistoryContent, ChatHistoryItem, ChatRequest, MessageRole},
        factory::get_client_with_config,
        tools::{CallbackFn, Tool},
    },
    pricing::{ModelUsage, get_model_pricing},
    providers::registry::provider_registry,
};

use crate::{
    cli::{
        errors::CLIError,
        handlers::documents::{get_document_mentions, get_user_document_tools, import_document},
        prompts::{get_summarize_prompt, get_title_prompt},
    },
    common::Context,
    tools::{mixins::ToolExt, retrieval::RetrievalTool, summarization::SummarizationTool},
    utils::{
        library::get_authors,
        rag::ModelResponse,
        terminal::{DIM_TEXT, RESET},
    },
};
use crate::{state::UsageMetadata, store::common::ZoteroStore};

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

/// Perform a vector search for a user-provided search term.
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
    let (mut search_results, _) = ctx
        .store
        .vector_search(
            search_term.clone(),
            10,
            ctx.config.get_reranker_config().as_ref(),
        )
        .await?;
    let _ = get_authors(&mut search_results);

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
    let config_clone = ctx.config.clone();
    if title_slot.lock()?.is_none()
        && let Some(small_config) = ctx.config.get_small_model_config()
        && let Ok(small_client) = get_client_with_config(&small_config)
    {
        let title_cost_tx = cost_tx.clone();
        let title_model_name = small_config.model_name().to_owned();
        let prompt = get_title_prompt(&query);
        tokio::spawn(async move {
            let request = ChatRequest {
                chat_history: Vec::new(),
                max_tokens: Some(20),
                message: prompt,
                reasoning: None,
                tools: None,
                on_tool_call: None,
                on_text: None,
            };
            if let Ok(response) = small_client.send_message(&request).await {
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

    let store_arc = std::sync::Arc::new(ctx.store.clone());
    let retrieval_tool = RetrievalTool::new(std::sync::Arc::clone(&store_arc), reranker_config);
    let retrieval_embedding_tokens = std::sync::Arc::clone(&retrieval_tool.embedding_tokens);
    let retrieval_rerank_tokens = std::sync::Arc::clone(&retrieval_tool.rerank_tokens);

    let (text_tx, mut text_rx) = mpsc::unbounded_channel::<String>();
    let (status_tx, mut status_rx) = mpsc::unbounded_channel::<String>();
    let on_text: Arc<CallbackFn<str>> = Arc::new(move |text: &str| {
        let _ = text_tx.send(text.to_string());
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
            reasoning: ctx.config.get_reasoning_config(),
            tools: Some(&tools),
            on_tool_call: None,
            on_text: Some(on_text),
        }
    };

    let final_draft_start = Instant::now();
    let mut send_message = pin!(llm_client.send_message(&request));
    let result = loop {
        tokio::select! {
            Some(segment) = text_rx.recv() => writeln!(ctx.out, "{segment}")?,
            Some(line) = status_rx.recv() => writeln!(ctx.err, "{line}")?,
            Some(title_cost) = cost_rx.recv() => ctx.state.usage += title_cost,
            result = &mut send_message => break result,
        }
    };
    // Both producers can race the response's completion; drain the leftovers.
    while let Ok(segment) = text_rx.try_recv() {
        writeln!(ctx.out, "{segment}")?;
    }
    while let Ok(line) = status_rx.try_recv() {
        writeln!(ctx.err, "{line}")?;
    }
    while let Ok(usage) = cost_rx.try_recv() {
        ctx.state.usage += usage;
    }

    let final_draft_duration = final_draft_start.elapsed();

    match result {
        Ok(response) => {
            writeln!(
                &mut ctx.err,
                "{DIM_TEXT}Final draft completed in {final_draft_duration:.2?}{RESET}"
            )?;

            let model_response_text = ModelResponse::from(&response.content).to_string();

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
            history.push(ChatHistoryItem {
                role: MessageRole::Assistant,
                content: vec![ChatHistoryContent::Text(model_response_text)],
            });
            ctx.state.dirty.store(true, atomic::Ordering::Relaxed);
        }
        Err(e) => {
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
    use serial_test::serial;
    use temp_env;
    use zqa_macros::test_ok;
    use zqa_macros_proc::retry;

    use super::{handle_query_cmd, handle_search_cmd};
    use crate::cli::{app::tests::create_test_context, handlers::library::handle_process_cmd};

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_search_only() {
        dotenv::dotenv().ok();
        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context(vec![]);
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_process_cmd(&mut setup_ctx),
        )
        .await;
        test_ok!(result);

        let mut ctx = create_test_context(vec![]); // search doesn't use LLMs
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_search_cmd(
                "How should I oversample in defect prediction?".to_string(),
                &mut ctx,
            ),
        )
        .await;
        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.len() > 20);
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_run_query() {
        dotenv::dotenv().ok();
        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context(vec![]);
        let _ = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_process_cmd(&mut setup_ctx),
        )
        .await;

        // TODO: At some point, we'll want to add support for tool calls on these. Right now, the
        // underlying `TestClient` doesn't call `process_tool_calls
        let mut ctx = create_test_context(vec!["You have papers X, Y, and Z.".into()]);
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_query_cmd(
                "What papers do I have about learning rate scheduling?".to_string(),
                &mut ctx,
            ),
        )
        .await;

        test_ok!(result);
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Total token usage:"));
    }
}
