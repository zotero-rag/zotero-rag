use crate::cli::placeholder::PlaceholderText;
use crate::cli::prompts::{get_summarize_prompt, get_title_prompt};
use crate::cli::readline::get_readline_config;
use crate::state::{SavedChatHistory, save_conversation};
use crate::tools::retrieval::RetrievalTool;
use crate::tools::summarization::SummarizationTool;
use crate::utils::arrow::{DbFields, get_schema, library_to_arrow, vector_search};
use crate::utils::library::{ZoteroItem, ZoteroItemSet, get_authors, get_new_library_items};
use crate::utils::rag::ModelResponse;
use arrow_array::{self, RecordBatch, StringArray};
use arrow_schema::Schema;
use chrono::Local;
use lancedb::embeddings::EmbeddingDefinition;
use rustyline::error::ReadlineError;
use zqa_rag::capabilities::ModelProvider;
use zqa_rag::llm::base::{
    ASSISTANT_ROLE, ApiClient, ChatHistoryContent, ChatHistoryItem, ChatRequest, ToolUseStats,
    USER_ROLE,
};
use zqa_rag::llm::factory::get_client_with_config;
use zqa_rag::llm::tools::{
    ANTHROPIC_SCHEMA_KEY, CallbackFn, GEMINI_SCHEMA_KEY, OPENAI_SCHEMA_KEY, Tool,
};
use zqa_rag::pricing::get_model_pricing;
use zqa_rag::vector::checkhealth::lancedb_health_check;
use zqa_rag::vector::doctor::doctor as rag_doctor;
use zqa_rag::vector::lance::{
    create_or_update_indexes, db_statistics, dedup_rows, delete_rows, get_zero_vector_records,
    insert_records, lancedb_exists,
};

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::state::get_conversation_history;
use crate::utils::terminal::{DIM_TEXT, RESET};
use crate::{full_library_to_arrow, utils::library::parse_library_metadata};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use std::sync::{Arc, Mutex, atomic};
use std::{
    fs::File,
    io::{self, BufRead, Write},
    time::Instant,
};

/// A file that contains parsed PDF texts from the user's Zotero library. In case the
/// embedding generation fails, the user does not need to rerun the full PDF parsing,
/// and can simply retry the embedding. Note that this is *not* supposed to be user-facing
/// and all interaction with it is meant for use by the CLI.
const BATCH_ITER_FILE: &str = "batch_iter.bin";

/// Embed text from PDFs parsed, in case this step previously failed. This function reads
/// the `BATCH_ITER_FILE` and uses the data in there to compute embeddings and write out
/// the `LanceDB` table.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
/// * `fix_zeros` - If `true`, fixes zero-embedding vectors, but does not handle PDFs
///   parsed but not embedded.
async fn embed<O: Write, E: Write>(
    ctx: &mut Context<O, E>,
    fix_zeros: bool,
) -> Result<(), CLIError> {
    if fix_zeros {
        return fix_zero_embeddings(ctx).await;
    }

    let file = File::open(BATCH_ITER_FILE)?;
    let reader = FileReader::try_new(file, None)?;

    let mut batches = Vec::<RecordBatch>::new();
    for batch in reader {
        batches.push(batch?);
    }

    if batches.is_empty() {
        writeln!(
            &mut ctx.err,
            "(/embed) It seems {BATCH_ITER_FILE} contains no batches. Exiting early."
        )?;
        return Ok(());
    }

    let n_batches = batches.len();

    write!(ctx.out, "Successfully loaded {n_batches} batch")?;

    if n_batches > 1 {
        write!(&mut ctx.out, "es")?;
    }
    writeln!(ctx.out, ".")?;

    let embedding_provider = &ctx.config.embedding_provider;
    let db = insert_records(
        batches,
        Some(&[DbFields::LibraryKey.as_ref()]),
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        EmbeddingDefinition::new(
            &DbFields::PdfText.into(), // source column
            embedding_provider,
            Some(&DbFields::Embeddings.into()), // dest column
        ),
    )
    .await;

    if db.is_ok() {
        writeln!(ctx.out, "Successfully parsed library!")?;
        std::fs::remove_file(BATCH_ITER_FILE)?;
    } else if let Err(e) = db {
        writeln!(ctx.out, "Parsing library failed: {e}")?;
        writeln!(
            ctx.out,
            "Your {BATCH_ITER_FILE} file has been left untouched."
        )?;
    }

    Ok(())
}

/// Fix the zero-embedding problem. In some error cases, we store zero-vectors as embeddings in
/// `LanceDB`. This function fixes those errors by replacing them with "real" embeddings. Note that
/// there are cases where the embeddings are zeros not because there was an error, but because the
/// extracted text was empty. This could be the result of a failed attempt to parse, or some other
/// similar error. APIs like Voyage do accept empty strings, and simply return a zero vector.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
async fn fix_zero_embeddings<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    let healthcheck = lancedb_health_check(&ctx.config.embedding_provider).await?;

    if let Some(Ok(zero_items)) = healthcheck.zero_embedding_items {
        let num_zeros: usize = zero_items
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum();

        if num_zeros > 0 {
            writeln!(
                ctx.out,
                "{DIM_TEXT}Fixing {num_zeros} zero-embedding items.{RESET}"
            )?;
        }
    }

    let zero_batches: Vec<RecordBatch> =
        get_zero_vector_records(&ctx.config.get_embedding_config().ok_or(
            CLIError::ConfigError("Could not get embedding config".into()),
        )?)
        .await?;

    if zero_batches.is_empty() {
        writeln!(ctx.out, "{DIM_TEXT}Done!{RESET}")?;
        return Ok(());
    }

    let zero_subset: Vec<ZoteroItem> = ZoteroItemSet::from(zero_batches).into();
    let nonempty_zero_subset = zero_subset
        .iter()
        .filter(|&item| !item.text.is_empty())
        .cloned()
        .collect::<Vec<_>>();

    let num_empty_texts = zero_subset.len() - nonempty_zero_subset.len();

    let zero_subset_keys: Vec<_> = zero_subset
        .iter()
        .map(|item| item.metadata.library_key.as_str())
        .collect();
    let key_array = StringArray::from(zero_subset_keys);
    let delete_schema = Arc::new(Schema::new(vec![arrow_schema::Field::new(
        DbFields::LibraryKey,
        arrow_schema::DataType::Utf8,
        false,
    )]));
    let zero_subset_batch = RecordBatch::try_new(delete_schema, vec![Arc::new(key_array)])?;

    delete_rows(
        zero_subset_batch,
        DbFields::LibraryKey.as_ref(),
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
    )
    .await?;

    writeln!(
        ctx.out,
        "{num_empty_texts} items had empty texts, and will be deleted.\n"
    )?;

    if nonempty_zero_subset.is_empty() {
        return Ok(());
    }

    let nonempty_zero_subset_batch = library_to_arrow(
        nonempty_zero_subset,
        ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ArrowError(
                "Could not get embedding config".into(),
            ))?,
    )
    .await?;

    let batches = vec![nonempty_zero_subset_batch.clone()];

    insert_records(
        batches,
        Some(&[DbFields::LibraryKey.as_ref()]),
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        EmbeddingDefinition::new(
            &DbFields::PdfText.into(), // source column
            &ctx.config.embedding_provider,
            Some(&DbFields::Embeddings.into()), // dest column
        ),
    )
    .await?;

    writeln!(ctx.out, "Successfully fixed zero embeddings!\n",)?;

    Ok(())
}

async fn dedup<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<usize, CLIError> {
    Ok(dedup_rows(
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        get_schema(&ctx.config.embedding_provider).await,
        DbFields::Title.as_ref(),
        DbFields::LibraryKey.as_ref(),
    )
    .await?)
}

/// Performs comprehensive health checks on the `LanceDB` database and reports status
/// with colored output using ASCII escape codes.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
async fn checkhealth<O: Write, E: Write>(ctx: &mut Context<O, E>) {
    let _ = match lancedb_health_check(&ctx.config.embedding_provider).await {
        Ok(result) => writeln!(ctx.out, "{result}"),
        Err(e) => writeln!(ctx.err, "{e}"),
    };
}

async fn update_indices<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    writeln!(
        &mut ctx.out,
        "Updating indices. This may take a while depending on how many items need to be added."
    )?;

    create_or_update_indexes(DbFields::PdfText.as_ref(), DbFields::Embeddings.as_ref()).await?;

    writeln!(
        &mut ctx.out,
        "Done! You should verify the indices exist with /checkhealth."
    )?;

    Ok(())
}

/// Runs health checks on the `LanceDB` database and provides helpful suggestions to the user on how
/// to fix any issues, if that is possible. Automatically attempt to fix issues found. Currently,
/// only zero-embedding vectors can be fixed, since a lot of the other issues are possibly just DB
/// corruption. Maybe we can diagnose that in the future.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
async fn doctor<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    if let Err(e) = rag_doctor(&ctx.config.embedding_provider, &mut ctx.out).await {
        writeln!(ctx.err, "{e}")?;
    }

    // Currently, we can really only fix the zero-embeddings issue
    fix_zero_embeddings(ctx).await
}

/// Process a user's Zotero library. This acts as one of the main functions provided by the CLI.
/// This parses the library, extracts the text from each file, stores them in a `LanceDB`
/// table, and adds their embeddings. If the last step fails, the parsed texts are stored
/// in `BATCH_ITER_FILE`.
///
/// # Arguments:
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
pub(crate) async fn process<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    const WARNING_THRESHOLD: usize = 100;

    let item_metadata =
        if lancedb_exists().await {
            get_new_library_items(&ctx.config.get_embedding_config().ok_or(
                CLIError::ConfigError("Could not get embedding config".into()),
            )?)
            .await
        } else {
            parse_library_metadata(None, None)
        };

    if let Err(parse_err) = item_metadata {
        writeln!(
            &mut ctx.out,
            "Could not parse library metadata: {parse_err}"
        )?;
        return Ok(());
    }

    let item_metadata = item_metadata.unwrap();
    let metadata_length = item_metadata.len();
    if metadata_length >= WARNING_THRESHOLD {
        writeln!(
            &mut ctx.out,
            "Your library has {metadata_length} new items. Parsing may take a while. Continue?"
        )?;
        write!(&mut ctx.out, "(/process) >>> ")?;
        ctx.out.flush()?;

        let mut option = String::new();
        io::stdin().read_line(&mut option)?;

        let option = option.trim().to_lowercase();
        if ["n", "no", "false", "0"].contains(&option.as_str()) {
            return Ok(());
        }
    }

    let record_batch = full_library_to_arrow(&ctx.config, None, None).await?;
    let schema = record_batch.schema();
    let batches = vec![record_batch.clone()];

    // Write to binary file using Arrow IPC format
    let file = File::create(BATCH_ITER_FILE)?;
    let mut writer = FileWriter::try_new(file, &schema)?;

    writer.write(&record_batch)?;
    writer.finish()?;

    let embedding_provider = ctx.config.embedding_provider.as_str();
    let result = insert_records(
        batches,
        Some(&[DbFields::LibraryKey.as_ref()]),
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        EmbeddingDefinition::new(
            DbFields::PdfText.as_ref(), // source column
            embedding_provider,
            Some(DbFields::Embeddings.as_ref()), // dest column
        ),
    )
    .await;

    match result {
        Ok(_) => {
            writeln!(&mut ctx.out, "Successfully parsed library!")?;
            std::fs::remove_file(BATCH_ITER_FILE)?;
        }
        Err(e) => {
            writeln!(&mut ctx.out, "Parsing library failed: {e}")?;
            writeln!(
                &mut ctx.out,
                "The parsed PDFs have been saved in 'batch_iter.bin'. Run '/embed fix' to retry embedding."
            )?;
        }
    }

    Ok(())
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
        .unwrap()
        .join(",")
}

/// Given a user query and the runtime context (CLI args + a `io::Write` implementation), perform a
/// vector search.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
async fn search_for_papers<O: Write, E: Write>(
    query: String,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    let vector_search_start = Instant::now();
    let mut search_results = vector_search(
        query.clone(),
        &ctx.config
            .get_embedding_config()
            .ok_or(CLIError::ConfigError(
                "Could not get embedding config".into(),
            ))?,
        ctx.config.reranker_provider.clone(),
    )
    .await?;
    let _ = get_authors(&mut search_results);

    let vector_search_duration = vector_search_start.elapsed();
    writeln!(
        &mut ctx.err,
        "{DIM_TEXT}Vector search completed in {vector_search_duration:.2?}{RESET}"
    )?;

    for item in &search_results {
        writeln!(&mut ctx.out, "{}", item.metadata.title).unwrap_or_else(|_| {
            eprintln!("Could not write out search results.");
        });
    }
    writeln!(&mut ctx.out)?;

    Ok(())
}

/// Given a user query and the runtime context (CLI args + a `io::Write` implementation), perform a
/// vector search and generate an answer based on the user's Zotero library.
///
/// # Arguments
///
/// * `query` - The user query.
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
#[allow(clippy::too_many_lines)]
async fn run_query<O: Write, E: Write>(
    query: String,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    let model_provider = ctx.config.model_provider.clone();
    if !ModelProvider::contains(&model_provider) {
        return Err(CLIError::LLMError(format!(
            "Model provider {model_provider} is not valid."
        )));
    }

    // Create LLM client with config
    let llm_client = ctx
        .config
        .get_generation_config()
        .map(get_client_with_config)
        .transpose()?
        .ok_or(CLIError::ConfigError(
            "Could not get generation config".into(),
        ))?;

    let generation_model_name = ctx.config.get_generation_model_name();

    let embedding_config = ctx
        .config
        .get_embedding_config()
        .ok_or(CLIError::ConfigError(
            "Could not get embedding config".into(),
        ))?;
    let reranker_provider = ctx.config.reranker_provider.clone();
    let schema_key = match model_provider.as_str() {
        "anthropic" => ANTHROPIC_SCHEMA_KEY,
        "gemini" => GEMINI_SCHEMA_KEY,
        _ => OPENAI_SCHEMA_KEY,
    }
    .to_string();

    // Spawn a background title generation task from the query alone, in parallel with summarization.
    // Only generate a title if we don't already have one (i.e., first query in the conversation).
    let title_slot = Arc::clone(&ctx.state.title);
    if title_slot.lock()?.is_none()
        && let Some(small_config) = ctx.config.get_small_model_config()
        && let Ok(small_client) = get_client_with_config(small_config)
    {
        let prompt = get_title_prompt(&query);
        tokio::spawn(async move {
            let request = ChatRequest {
                chat_history: Vec::new(),
                max_tokens: Some(20),
                message: prompt,
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
            }
        });
    }

    let mut total_input_tokens: u32 = 0;
    let mut total_output_tokens: u32 = 0;

    let retrieval_tool =
        RetrievalTool::new(embedding_config, reranker_provider, schema_key.clone());
    let summarization_tool = SummarizationTool::new(llm_client.clone(), schema_key);
    let summarization_tool_clone = summarization_tool.clone();
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(retrieval_tool), Box::new(summarization_tool)];

    let chat_history = Arc::clone(&ctx.state.chat_history);

    let on_tool_call: Arc<CallbackFn<ToolUseStats>> = Arc::new(move |stats: &ToolUseStats| {
        let _ = writeln!(io::stderr(), "{}🗸 {}{}", DIM_TEXT, stats.tool_name, RESET);
    });
    let on_text: Arc<CallbackFn<str>> = Arc::new(move |text: &str| {
        let _ = writeln!(io::stdout(), "{text}");
    });

    let request = {
        let history = chat_history
            .lock()
            .expect("Could not obtain lock on chat history.");
        ChatRequest {
            chat_history: history.clone(),
            max_tokens: None,
            message: get_summarize_prompt(&query),
            tools: Some(&tools),
            on_tool_call: Some(on_tool_call),
            on_text: Some(on_text),
        }
    };

    let final_draft_start = Instant::now();
    let result = llm_client.send_message(&request).await;
    let final_draft_duration = final_draft_start.elapsed();

    // Invariant: by this point, `generation_model_name` cannot be `None`.
    let generation_model_name = generation_model_name.unwrap_or_default();
    let pricing = {
        let mp = model_provider.clone();
        let gmn = generation_model_name.clone();
        tokio::task::spawn_blocking(move || get_model_pricing(&mp, &gmn, None))
            .await
            .ok()
            .flatten()
    };

    match result {
        Ok(response) => {
            writeln!(
                &mut ctx.err,
                "{DIM_TEXT}Final draft completed in {final_draft_duration:.2?}{RESET}"
            )?;

            let model_response_text = ModelResponse::from(&response.content).to_string();

            total_input_tokens += response.input_tokens;
            if let Ok(summarization_input_tokens) = summarization_tool_clone.input_tokens.lock() {
                total_input_tokens += *summarization_input_tokens;
            }

            total_output_tokens += response.output_tokens;
            if let Ok(summarization_output_tokens) = summarization_tool_clone.output_tokens.lock() {
                total_output_tokens += *summarization_output_tokens;
            }

            // Update session cost
            if let Some(ref p) = pricing {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let call_cost =
                    (p.estimate_cost(total_input_tokens, total_output_tokens) * 100.0) as u64;
                ctx.state
                    .session_cost
                    .fetch_add(call_cost, atomic::Ordering::Relaxed);
            }

            // Update state - re-acquire lock
            let mut history = chat_history
                .lock()
                .expect("Could not obtain lock on chat history.");
            history.push(ChatHistoryItem {
                role: USER_ROLE.into(),
                content: vec![ChatHistoryContent::Text(query.clone())],
            });
            history.push(ChatHistoryItem {
                role: ASSISTANT_ROLE.into(),
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
        format_number(total_input_tokens)
    )?;
    writeln!(
        &mut ctx.out,
        "\t{DIM_TEXT}Output tokens: {}{RESET}\n",
        format_number(total_output_tokens)
    )?;

    if let Some(p) = &pricing {
        let cost = p.estimate_cost(total_input_tokens, total_output_tokens);
        if cost > 0.0 {
            writeln!(
                &mut ctx.out,
                "\t{DIM_TEXT}Estimated cost: ${cost:.4} ({generation_model_name}){RESET}"
            )?;
        }
    }
    let session_cost = ctx.state.session_cost.load(atomic::Ordering::Relaxed);
    if session_cost > 0 {
        let session_cost_dollars = session_cost as f64 / 100.0;
        writeln!(
            &mut ctx.out,
            "\t{DIM_TEXT}Session cost:   ${session_cost_dollars:.4}{RESET}"
        )?;
    }
    writeln!(&mut ctx.out)?;

    Ok(())
}

/// Prints out table statistics from the created DB. Fails if the database does not exist, could
/// not be read, or the statistics could not be computed.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
async fn stats<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    match db_statistics().await {
        Ok(stats) => writeln!(&mut ctx.out, "{stats}")?,
        Err(e) => writeln!(&mut ctx.err, "Could not get database statistics: {e}")?,
    }

    Ok(())
}

/// Resume a previous conversation selected by the user.
///
/// Displays a numbered list of saved conversations, prompts for a selection, and loads the
/// chosen conversation into the current session. If the current session is dirty, it is saved
/// first.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
/// * `reader` - A buffered reader used to read the user's selection.
///
/// # Errors
///
/// * `CLIError::IOError` - If `writeln!` or reading input fails.
/// * `CLIError::MutexPoisoningError` - If a lock on the context chat history could not be
///   obtained.
fn resume<O: Write, E: Write, R: BufRead>(
    ctx: &mut Context<O, E>,
    reader: &mut R,
) -> Result<(), CLIError> {
    match get_conversation_history() {
        Err(e) => {
            writeln!(&mut ctx.err, "Failed to load conversations: {e}")?;
        }
        Ok(None) => {
            writeln!(&mut ctx.out, "No saved conversations found.")?;
        }
        Ok(Some(ref v)) if v.is_empty() => {
            writeln!(&mut ctx.out, "No saved conversations found.")?;
        }
        Ok(Some(mut histories)) => {
            histories.sort_by(|a, b| b.date.cmp(&a.date));

            writeln!(&mut ctx.out)?;
            writeln!(&mut ctx.out, "Saved conversations:")?;
            for (i, h) in histories.iter().enumerate() {
                let msg_count = h.history.len();
                writeln!(
                    &mut ctx.out,
                    "  [{}] {} ({} message{})",
                    i + 1,
                    h.title,
                    msg_count,
                    if msg_count == 1 { "" } else { "s" }
                )?;
            }
            writeln!(&mut ctx.out)?;
            write!(&mut ctx.out, "Enter a number (1-{}): ", histories.len())?;
            ctx.out.flush()?;

            let mut input = String::new();
            reader.read_line(&mut input)?;
            let input = input.trim();

            match input.parse::<usize>() {
                Ok(n) if n >= 1 && n <= histories.len() => {
                    if ctx.state.dirty.load(atomic::Ordering::Relaxed) {
                        let chat_history = Arc::clone(&ctx.state.chat_history);
                        let history = chat_history.lock()?;
                        let date = Local::now();
                        let conversation = SavedChatHistory {
                            history: history.clone(),
                            date,
                            title: ctx.state.title.lock()?.clone().unwrap_or_else(|| {
                                format!("Conversation on {}", date.format("%Y-%m-%d %H:%M"))
                            }),
                        };
                        if let Err(e) = save_conversation(&conversation)
                            && let Err(write_err) =
                                writeln!(&mut ctx.err, "Error saving conversation: {e}")
                        {
                            log::error!("Failed to write to stderr: {write_err}");
                        }
                    }

                    let selected = histories.swap_remove(n - 1);
                    let title = selected.title.clone();
                    ctx.state.chat_history = Arc::new(Mutex::new(selected.history));
                    ctx.state.dirty.store(false, atomic::Ordering::Relaxed);
                    writeln!(&mut ctx.out, "Resumed: {title}")?;
                }
                _ => {
                    writeln!(&mut ctx.err, "Invalid selection.")?;
                }
            }
        }
    }

    Ok(())
}

fn save_current_conversation<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    if ctx.state.dirty.load(atomic::Ordering::Relaxed) {
        let chat_history = Arc::clone(&ctx.state.chat_history);
        let history = chat_history.lock()?;
        let date = Local::now();

        let conversation =
            SavedChatHistory {
                history: history.clone(),
                date,
                title: ctx.state.title.lock()?.clone().unwrap_or_else(|| {
                    format!("Conversation on {}", date.format("%Y-%m-%d %H:%M"))
                }),
            };

        if let Err(e) = save_conversation(&conversation) {
            writeln!(&mut ctx.err, "Error saving conversation: {e}")?;
        }
    }
    Ok(())
}

/// Handle a single command or query from the user.
///
/// # Arguments
///
/// * `command` - The command string entered by the user.
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(true)` if the CLI should continue running, `Ok(false)` if it should exit.
#[allow(clippy::too_many_lines)]
pub(crate) async fn handle_command<O: Write, E: Write>(
    command: &str,
    ctx: &mut Context<O, E>,
) -> Result<bool, CLIError> {
    let command = command.trim();

    match command {
        "" => Ok(true),
        "/help" | "help" | "?" => {
            writeln!(&mut ctx.out)?;
            writeln!(&mut ctx.out, "Available commands:\n")?;
            writeln!(&mut ctx.out, "/help\t\tShow this help message")?;
            writeln!(
                &mut ctx.out,
                "/process\tPre-process Zotero library. Use to update the database."
            )?;
            writeln!(
                &mut ctx.out,
                "/embed\t\tRepair failed DB creation by re-adding embeddings."
            )?;
            writeln!(
                &mut ctx.out,
                "/search\t\tSearch for papers without summarizing them. Usage: /search <query>"
            )?;
            writeln!(
                &mut ctx.out,
                "/config\t\tShow the currently used configuration."
            )?;
            writeln!(
                &mut ctx.out,
                "/new\t\tSave the current conversation and switch to a new one."
            )?;
            writeln!(&mut ctx.out, "/resume\t\tResume a previous conversation.")?;
            writeln!(&mut ctx.out, "/index\t\tCreate or update indices.")?;
            writeln!(
                &mut ctx.out,
                "/checkhealth\tRun health checks on your LanceDB."
            )?;
            writeln!(
                &mut ctx.out,
                "/doctor\t\tAttempt to fix issues spotted by /checkhealth."
            )?;
            writeln!(&mut ctx.out, "/stats\t\tShow table statistics.")?;
            writeln!(&mut ctx.out, "/dedup\t\tRemove duplicate items.")?;
            writeln!(&mut ctx.out, "/quit\t\tExit the program")?;
            writeln!(&mut ctx.out)?;
            Ok(true)
        }
        "/checkhealth" => {
            checkhealth(ctx).await;
            Ok(true)
        }
        "/doctor" => {
            doctor(ctx).await?;
            Ok(true)
        }
        "/embed" => {
            if let Err(e) = embed(ctx, false).await {
                writeln!(
                    &mut ctx.err,
                    "Failed to create embeddings. You may find relevant error messages below:\n\t{e}"
                )?;
            }
            Ok(true)
        }
        "/process" => {
            if let Err(e) = process(ctx).await {
                writeln!(
                    &mut ctx.err,
                    "Failed to create embeddings. You may find relevant error messages below:\n\t{e}"
                )?;
            }
            Ok(true)
        }
        "/index" => {
            if let Err(e) = update_indices(ctx).await {
                writeln!(
                    &mut ctx.err,
                    "Failed to update indexes. You may find the below information useful:\n\t{e}"
                )?;
            }
            Ok(true)
        }
        "/stats" => {
            if let Err(e) = stats(ctx).await {
                // This only errors on an I/O failure
                writeln!(&mut ctx.err, "Failed to write statistics to buffer. {e}")?;
            }
            Ok(true)
        }
        "/dedup" => {
            match dedup(ctx).await {
                Ok(ct) => {
                    writeln!(&mut ctx.out, "Removed {ct} items.")?;
                }
                Err(e) => {
                    writeln!(&mut ctx.err, "Deduplication failed: {e}")?;
                }
            }
            Ok(true)
        }
        "/resume" => {
            let stdin = io::stdin();
            let mut reader = stdin.lock();
            if let Err(e) = resume(ctx, &mut reader) {
                writeln!(&mut ctx.err, "Error resuming conversation: {e}")?;
            }
            Ok(true)
        }
        "/config" => {
            writeln!(&mut ctx.out, "{}", ctx.config)?;
            Ok(true)
        }
        "/quit" | "/exit" | "quit" | "exit" | "/new" => {
            save_current_conversation(ctx)?;

            if command == "/new" {
                ctx.state.dirty.store(false, atomic::Ordering::Relaxed);
                ctx.state.chat_history = Arc::new(Mutex::new(Vec::new()));
                ctx.state.title = Arc::new(Mutex::new(None));
                Ok(true)
            } else {
                Ok(false)
            }
        }
        query => {
            writeln!(&mut ctx.out)?;

            // Check for a threshold to ensure this isn't an accidental Enter-hit.
            #[allow(clippy::items_after_statements)]
            const MIN_QUERY_LENGTH: usize = 10;

            if query.len() < MIN_QUERY_LENGTH {
                writeln!(&mut ctx.out, "Invalid command: {query}")?;
                return Ok(true);
            }

            // Search queries have priority
            if query.starts_with("/search") {
                let search_term = query.strip_prefix("/search").unwrap().trim();
                if search_term.is_empty() {
                    writeln!(&mut ctx.err, "Please provide a search term after /search.")?;
                    return Ok(true);
                }

                if let Err(e) = search_for_papers(search_term.into(), ctx).await {
                    writeln!(
                        &mut ctx.err,
                        "Failed to perform a vector search. You may find relevant error messages below:\n\t{e}"
                    )?;
                }

                return Ok(true);
            } else if query.starts_with("/embed") {
                // Handle `/embed fix`
                let subcmd = query.strip_prefix("/embed").unwrap().trim();
                if subcmd != "fix" {
                    writeln!(&mut ctx.err, "Invalid subcommand to /embed: {subcmd}")?;
                    return Ok(true);
                }

                embed(ctx, true).await?;
                return Ok(true);
            }

            if let Err(e) = run_query(query.into(), ctx).await {
                writeln!(
                    &mut ctx.err,
                    "Failed to answer the question. You may find relevant error messages below:\n\t{e}",
                )?;
            }
            Ok(true)
        }
    }
}

/// The core CLI implementation that implements a REPL for user commands.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
///
/// # Errors
///
/// * `CLIError::ReadlineError` - If we could not get the history file path, a `readline` editor could not be created,
///   or the history could not be saved.
/// * `CLIError::IOError` - If `writeln!` fails.
///
/// # Panics
///
/// Cannot happen; `unwrap()` is called on `strip_prefix` result after checking that the prefix exists.
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_continue)]
pub(crate) async fn cli<O: Write, E: Write>(mut ctx: Context<O, E>) -> Result<(), CLIError> {
    // First, get the path to the history file used by the `readline` implementation.
    let user_dirs = directories::UserDirs::new().ok_or(CLIError::ReadlineError(
        "Could not get user directories".into(),
    ))?;
    let home_dir = user_dirs.home_dir();
    let history_path = home_dir.join(".zqa_history");

    // Create the `readline` "editor" with our `PlaceholderText` helper
    let mut rl =
        rustyline::Editor::<PlaceholderText, rustyline::history::DefaultHistory>::with_config(
            get_readline_config(),
        )
        .map_err(|e| CLIError::ReadlineError(e.to_string()))?;

    if rl.load_history(&history_path).is_err() {
        log::debug!("No previous history.");
    }

    rl.set_helper(Some(PlaceholderText {
        placeholder_text: "Type in a question or /help for options".to_string(),
    }));

    loop {
        let readline = rl.readline(">>> ");

        match readline {
            Ok(command) => {
                if !command.trim().is_empty()
                    && let Err(e) = rl.add_history_entry(command.trim())
                {
                    log::debug!("Failed to write history entry: {e}");
                }

                if !handle_command(&command, &mut ctx).await? {
                    break;
                }
            }
            Err(ReadlineError::Signal(rustyline::error::Signal::Resize)) => {
                // Handle SIGWINCH; we should just rewrite the prompt and continue
                continue;
            }
            Err(ReadlineError::Interrupted) => {
                // Handle SIGINT by saving the conversation if needed
                save_current_conversation(&mut ctx)?;
                break;
            }
            _ => break,
        }
    }

    rl.save_history(&history_path)
        .map_err(|e| CLIError::ReadlineError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::cli::app::{BATCH_ITER_FILE, handle_command, resume};
    use crate::common::State;
    use crate::config::{Config, VoyageAIConfig};
    use crate::state::{SavedChatHistory, save_conversation};
    use crate::tools::retrieval::RetrievalTool;
    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatchIterator};
    use arrow_array::{RecordBatch, StringArray};
    use arrow_ipc::writer::FileWriter;
    use chrono::Local;
    use lancedb::connect;
    use serde_json::json;
    use serial_test::serial;
    use std::fs::{self, File};
    use std::io::Cursor;
    use std::sync::Arc;
    use temp_env;
    use zqa_macros::{test_contains, test_eq, test_ok};
    use zqa_macros_proc::retry;
    use zqa_rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use zqa_rag::embedding::common::EmbeddingProviderConfig;
    use zqa_rag::llm::base::{ASSISTANT_ROLE, ChatHistoryContent, ChatHistoryItem, USER_ROLE};
    use zqa_rag::llm::tools::Tool;

    use crate::common::Context;

    pub(crate) fn get_config() -> Config {
        let mut config = Config {
            voyageai: Some(VoyageAIConfig {
                reranker: Some(DEFAULT_VOYAGE_RERANK_MODEL.into()),
                embedding_model: Some(DEFAULT_VOYAGE_EMBEDDING_MODEL.into()),
                embedding_dims: Some(DEFAULT_VOYAGE_EMBEDDING_DIM as usize),
                api_key: Some(String::new()),
            }),
            ..Default::default()
        };

        config.read_env().unwrap();
        config
    }

    /// Create a default `Context` object where the output and error streams are buffers that can
    /// be written into. This allows for the output to be easily inspected in tests.
    pub(crate) fn create_test_context() -> Context<Cursor<Vec<u8>>, Cursor<Vec<u8>>> {
        let out_buf: Vec<u8> = Vec::new();
        let out = Cursor::new(out_buf);

        let err_buf: Vec<u8> = Vec::new();
        let err = Cursor::new(err_buf);

        let config = get_config();

        Context {
            state: State::default(),
            config,
            out,
            err,
        }
    }

    fn make_retrieval_tool(schema_key: &str) -> RetrievalTool {
        // Build a minimal tool; the embedding config is only used in `call`, not in the metadata
        // methods, so we use a dummy VoyageAI config here.
        RetrievalTool::new(
            EmbeddingProviderConfig::VoyageAI(zqa_rag::config::VoyageAIConfig {
                api_key: String::new(),
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
            }),
            "voyageai".into(),
            schema_key.into(),
        )
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_embed() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();

        // Create `RecordBatch` object to write out
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "pdf_text",
            arrow_schema::DataType::Utf8,
            false,
        )]);
        let data = StringArray::from(vec!["Hello", "World"]);
        let record_batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(data)]).unwrap();

        // Write out the object to `BATCH_ITER_FILE`
        let file = File::create(BATCH_ITER_FILE).unwrap();
        let mut writer = FileWriter::try_new(file, &schema).unwrap();

        writer.write(&record_batch).unwrap();
        writer.finish().unwrap();

        // Actually call `handle_command`
        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_command("/embed", &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.unwrap()); // Should continue loop

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let err = String::from_utf8(ctx.err.into_inner()).unwrap();
        assert!(err.is_empty());

        // Clean up
        if fs::metadata(BATCH_ITER_FILE).is_ok() {
            fs::remove_file(BATCH_ITER_FILE).expect("Failed to clean up BATCH_ITER_FILE");
        }
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_process() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();

        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_command("/process", &mut ctx),
        )
        .await;

        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.clone().into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let stats = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_command("/stats", &mut ctx),
        )
        .await;
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_ok!(stats);
        assert!(stats.unwrap());
        assert!(output.contains("Table statistics:"));
        assert!(output.contains("Number of rows: 8"));

        // Cleanup
        if fs::metadata(BATCH_ITER_FILE).is_ok() {
            fs::remove_file(BATCH_ITER_FILE).expect("Failed to clean up BATCH_ITER_FILE");
        }
    }

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

        let mut setup_ctx = create_test_context();

        // `process` needs to be run before `search_for_papers`
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(result);

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_command(
                "/search How should I oversample in defect prediction?",
                &mut ctx,
            ),
        )
        .await;

        if let Err(e) = &result {
            dbg!(e);
        }

        test_ok!(result);
        assert!(result.unwrap());

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

        let mut setup_ctx = create_test_context();

        // `process` needs to be run before `run_query`
        let _ = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_command("/process", &mut setup_ctx),
        )
        .await;

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_command("How should I oversample in defect prediction?", &mut ctx),
        )
        .await;

        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Total token usage:"));
    }

    #[tokio::test]
    #[serial]
    async fn test_checkhealth_no_database() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut ctx = create_test_context();
        let output = temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], async move {
            handle_command("/checkhealth", &mut ctx).await.unwrap();
            String::from_utf8(ctx.out.into_inner()).unwrap()
        })
        .await;

        assert!(output.contains("directory does not exist"));
    }

    #[retry(3)]
    #[tokio::test]
    #[serial]
    async fn test_checkhealth_with_database() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        // First create a database by running process
        let mut setup_ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(result);

        // Now run health check
        let mut ctx = create_test_context();
        let output = temp_env::async_with_vars([("LANCEDB_URI", Some(&db_uri))], async move {
            handle_command("/checkhealth", &mut ctx).await.unwrap();
            String::from_utf8(ctx.out.into_inner()).unwrap()
        })
        .await;

        test_contains!(output, "LanceDB Health Check Results");
        test_contains!(output, "directory exists");
        test_contains!(output, "Table is accessible");
        test_contains!(output, "Table has");
    }

    #[test]
    fn test_resume_no_conversations() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let mut ctx = create_test_context();
            let mut reader = Cursor::new("");
            resume(&mut ctx, &mut reader).unwrap();

            let output = String::from_utf8(ctx.out.into_inner()).unwrap();
            test_contains!(output, "No saved conversations found.");
        });
    }

    #[test]
    fn test_resume_loads_selected_conversation() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let history_a = vec![
                ChatHistoryItem {
                    role: USER_ROLE.into(),
                    content: vec![ChatHistoryContent::Text("What is attention?".into())],
                },
                ChatHistoryItem {
                    role: ASSISTANT_ROLE.into(),
                    content: vec![ChatHistoryContent::Text(
                        "Attention is a mechanism...".into(),
                    )],
                },
            ];
            let history_b = vec![ChatHistoryItem {
                role: USER_ROLE.into(),
                content: vec![ChatHistoryContent::Text(
                    "Tell me about transformers.".into(),
                )],
            }];

            save_conversation(&SavedChatHistory {
                history: history_a.clone(),
                date: Local::now(),
                title: "Conversation A".into(),
            })
            .unwrap();

            save_conversation(&SavedChatHistory {
                history: history_b.clone(),
                // Avoid same filename
                date: Local::now() + chrono::Duration::seconds(1),
                title: "Conversation B".into(),
            })
            .unwrap();

            let mut ctx = create_test_context();
            // The list is sorted reverse-chronologically; B was saved last so it's [1].
            let mut reader = Cursor::new("1\n");
            resume(&mut ctx, &mut reader).unwrap();

            let out = String::from_utf8(ctx.out.into_inner()).unwrap();
            test_contains!(out, "Resumed:");

            let loaded = ctx.state.chat_history.lock().unwrap();
            test_eq!(loaded.len(), history_b.len());
            assert!(!ctx.state.dirty.load(std::sync::atomic::Ordering::Relaxed));
        });
    }

    #[test]
    fn test_resume_invalid_selection() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            save_conversation(&SavedChatHistory {
                history: vec![ChatHistoryItem {
                    role: USER_ROLE.into(),
                    content: vec![ChatHistoryContent::Text("Hello".into())],
                }],
                date: Local::now(),
                title: "Only Conversation".into(),
            })
            .unwrap();

            let mut ctx = create_test_context();
            let mut reader = Cursor::new("99\n");
            resume(&mut ctx, &mut reader).unwrap();

            let err = String::from_utf8(ctx.err.into_inner()).unwrap();
            test_contains!(err, "Invalid selection.");
        });
    }

    #[tokio::test]
    async fn test_handle_command_help() {
        let mut ctx = create_test_context();
        let result = handle_command("/help", &mut ctx).await.unwrap();
        assert!(result);
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "Available commands:");

        // Check that all the `match` arms are contained
        test_contains!(output, "/help");
        test_contains!(output, "/checkhealth");
        test_contains!(output, "/doctor");
        test_contains!(output, "/embed");
        test_contains!(output, "/process");
        test_contains!(output, "/index");
        test_contains!(output, "/stats");
        test_contains!(output, "/dedup");
        test_contains!(output, "/resume");
        test_contains!(output, "/config");
        test_contains!(output, "/new");
        test_contains!(output, "/quit"); // one of the variants suffices to show the user
    }

    /// Inserts a single row with a zero-valued embedding vector and empty `pdf_text` directly into
    /// the LanceDB table at `db_uri`. This bypasses the embedding function so we can deterministically
    /// create zero-embedding rows for testing `fix_zero_embeddings`.
    async fn insert_zero_embedding_row(db_uri: &str) {
        let dims = DEFAULT_VOYAGE_EMBEDDING_DIM as i32;
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("library_key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new(
                "embeddings",
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::Float32,
                        true,
                    )),
                    dims,
                ),
                false,
            ),
        ]));

        #[allow(clippy::cast_sign_loss)]
        let zeros = Float32Array::from(vec![0.0f32; dims as usize]);
        let embedding_col = FixedSizeListArray::try_new(
            Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                true,
            )),
            dims,
            Arc::new(zeros),
            None,
        )
        .unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["ZEROTEST001"])),
                Arc::new(StringArray::from(vec!["Zero Test Item"])),
                Arc::new(StringArray::from(vec!["/dev/null"])),
                Arc::new(StringArray::from(vec![""])), // will be deleted by fix
                Arc::new(embedding_col),
            ],
        )
        .unwrap();

        let db = connect(db_uri).execute().await.unwrap();
        let tbl = db.open_table("data").execute().await.unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        tbl.merge_insert(&["library_key"])
            .when_not_matched_insert_all()
            .clone()
            .execute(Box::new(reader))
            .await
            .unwrap();
    }

    /// When the database has no zero-embedding records, `fix_zero_embeddings` (via `/embed fix`)
    /// should complete immediately and report "Done!". Running `/embed fix` twice after `/process`
    /// guarantees the second run sees a clean database (the first clears any zeros produced by
    /// unparseable CI fixtures).
    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_fix_zero_embeddings_no_zeros() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(setup_result);

        // First run clears any zero-embedding rows produced by unparseable CI fixtures.
        let mut first_ctx = create_test_context();
        let first_result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_command("/embed fix", &mut first_ctx),
        )
        .await;
        test_ok!(first_result);
        assert!(first_result.unwrap());

        // Second run: the database is clean, so the function should exit early with "Done!".
        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_command("/embed fix", &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "Done!");
    }

    /// When zero-embedding rows whose `pdf_text` is empty are present, `fix_zero_embeddings` (via
    /// `/embed fix`) should delete them and report how many were removed.
    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_fix_zero_embeddings_with_zero_rows() {
        dotenv::dotenv().ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let db_uri = temp_dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let mut setup_ctx = create_test_context();
        let setup_result = temp_env::async_with_vars(
            [("CI", Some("true")), ("LANCEDB_URI", Some(&db_uri))],
            handle_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(setup_result);

        // Directly inject a row with a zero embedding vector and empty text into the DB.
        insert_zero_embedding_row(&db_uri).await;

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            handle_command("/embed fix", &mut ctx),
        )
        .await;
        test_ok!(result);
        assert!(result.unwrap());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        test_contains!(output, "items had empty texts, and will be deleted.");
    }

    #[retry(3)]
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_call_returns_results_key() {
        dotenv::dotenv().ok();
        let _ = crate::common::setup_logger(log::LevelFilter::Info);

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
            handle_command("/process", &mut setup_ctx),
        )
        .await;
        test_ok!(setup_result);

        // Now test the retrieval tool
        let tool = make_retrieval_tool("input_schema");

        let result = temp_env::async_with_vars(
            [("LANCEDB_URI", Some(&db_uri))],
            tool.call(json!({"query": "machine learning"})),
        )
        .await;
        test_ok!(result);

        let value = result.unwrap();
        assert!(
            value.get("results").is_some(),
            "Expected 'results' key in output, got: {value}"
        );
    }
}
