use crate::cli::placeholder::PlaceholderText;
use crate::cli::prompts::{get_extraction_prompt, get_summarize_prompt};
use crate::cli::readline::get_readline_config;
use crate::utils::arrow::{DbFields, library_to_arrow, vector_search};
use crate::utils::library::{ZoteroItem, ZoteroItemSet, get_authors, get_new_library_items};
use crate::utils::rag::SingleResponse;
use arrow_array::{self, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use lancedb::embeddings::EmbeddingDefinition;
use rag::capabilities::ModelProviders;
use rag::config::LLMClientConfig;
use rag::llm::base::{ApiClient, ChatRequest, CompletionApiResponse, ContentType};
use rag::llm::errors::LLMError;
use rag::llm::factory::{get_client_by_provider, get_client_with_config};
use rag::vector::checkhealth::lancedb_health_check;
use rag::vector::doctor::doctor as rag_doctor;
use rag::vector::lance::{
    create_or_update_indexes, db_statistics, delete_rows, get_zero_vector_records, insert_records,
    lancedb_exists,
};
use rustyline::error::ReadlineError;
use tokio::task::JoinSet;

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::{full_library_to_arrow, utils::library::parse_library_metadata};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use std::sync::Arc;
use std::{
    fs::File,
    io::{self, Write},
    time::Instant,
};

/// A file that contains parsed PDF texts from the user's Zotero library. In case the
/// embedding generation fails, the user does not need to rerun the full PDF parsing,
/// and can simply retry the embedding. Note that this is *not* supposed to be user-facing
/// and all interaction with it is meant for use by the CLI.
const BATCH_ITER_FILE: &str = "batch_iter.bin";

/// ANSI escape code for dimming text
const DIM_TEXT: &str = "\x1b[2m";

/// ANSI escape code for resetting text formatting
const RESET: &str = "\x1b[0m";

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

    let mut batches = Vec::<Result<RecordBatch, arrow_schema::ArrowError>>::new();
    for batch in reader {
        batches.push(batch);
    }

    if batches.is_empty() {
        writeln!(
            &mut ctx.err,
            "(/embed) It seems {BATCH_ITER_FILE} contains no batches. Exiting early."
        )?;
        return Ok(());
    }

    let n_batches = batches.len();

    // All batches should have the same schema, so we use the first batch
    let first_batch = batches
        .first()
        .ok_or(CLIError::MalformedBatchError)?
        .as_ref()?;
    let schema = first_batch.schema();
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema);

    write!(ctx.out, "Successfully loaded {n_batches} batch")?;

    if n_batches > 1 {
        write!(&mut ctx.out, "es")?;
    }
    writeln!(ctx.out, ".")?;

    let embedding_provider = &ctx.config.embedding_provider;
    let db = insert_records(
        batch_iter,
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

    let batches = vec![Ok(nonempty_zero_subset_batch.clone())];
    let batch_iter =
        RecordBatchIterator::new(batches.into_iter(), nonempty_zero_subset_batch.schema());

    insert_records(
        batch_iter,
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
    return fix_zero_embeddings(ctx).await;
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
async fn process<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
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
    let batches = vec![Ok(record_batch.clone())];
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());

    // Write to binary file using Arrow IPC format
    let file = File::create(BATCH_ITER_FILE)?;
    let mut writer = FileWriter::try_new(file, &schema)?;

    writer.write(&record_batch)?;
    writer.finish()?;

    let embedding_provider = ctx.config.embedding_provider.as_str();
    let result = insert_records(
        batch_iter,
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
            std::fs::remove_file("batch_iter.bin")?;
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
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
#[allow(clippy::too_many_lines)]
async fn run_query<O: Write, E: Write>(
    query: String,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    let model_provider = ctx.config.model_provider.clone();

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

    if !ModelProviders::contains(&model_provider) {
        return Err(CLIError::LLMError(format!(
            "Model provider {model_provider} is not valid."
        )));
    }

    // Create LLM client with config
    let llm_client = match model_provider.as_str() {
        "anthropic" => ctx
            .config
            .anthropic
            .clone()
            .map(|c| get_client_with_config(LLMClientConfig::Anthropic(c.into())))
            .transpose()?,
        "openai" => ctx
            .config
            .openai
            .clone()
            .map(|c| get_client_with_config(LLMClientConfig::OpenAI(c.into())))
            .transpose()?,
        "gemini" => ctx
            .config
            .gemini
            .clone()
            .map(|c| get_client_with_config(LLMClientConfig::Gemini(c.into())))
            .transpose()?,
        "openrouter" => ctx
            .config
            .openrouter
            .clone()
            .map(|c| get_client_with_config(LLMClientConfig::OpenRouter(c.into())))
            .transpose()?,
        _ => None,
    }
    .unwrap_or_else(|| get_client_by_provider(&model_provider).unwrap());

    let mut set = JoinSet::new();

    let summarization_start = Instant::now();
    for item in &search_results {
        let client = llm_client.clone();
        let text = item.text.clone();
        let query_clone = query.clone();
        let metadata = item.metadata.clone();

        set.spawn(async move {
            let request = ChatRequest {
                chat_history: Vec::new(),
                max_tokens: None,
                message: get_extraction_prompt(&query_clone, &text, &metadata),
                tools: None,
            };

            client.send_message(&request).await
        });
    }

    let results: Vec<Result<CompletionApiResponse, LLMError>> = set.join_all().await;
    let summarization_duration = summarization_start.elapsed();
    writeln!(
        &mut ctx.err,
        "{DIM_TEXT}Summarization completed in {summarization_duration:.2?}{RESET}"
    )?;

    if ctx.args.print_summaries && ctx.args.log_level >= log::LevelFilter::Info {
        for (paper, summary_result) in search_results.iter().zip(results.iter()) {
            if let Ok(summary) = summary_result {
                let title = &paper.metadata.title;
                let summary_text = summary
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ContentType::Text(text) => Some(text.as_str()),
                        ContentType::ToolCall(_) => None,
                    })
                    .collect::<String>();

                log::info!("Paper: {title}");
                log::info!("Summary: {summary_text}");
            }
        }
    }

    let err_results = results
        .iter()
        .filter_map(|res| res.as_ref().err())
        .collect::<Vec<_>>();

    if !err_results.is_empty() {
        writeln!(
            &mut ctx.err,
            "{}/{} LLM requests failed:",
            err_results.len(),
            search_results.len()
        )?;
        writeln!(
            &mut ctx.err,
            "Here is why the first one failed (the others may be similar):"
        )?;
        if let Some(first_error) = err_results.first() {
            writeln!(&mut ctx.err, "\t{first_error}")?;
        }
    }

    let (ok_contents, mut total_input_tokens, mut total_output_tokens) = results
        .iter()
        .filter_map(|res| res.as_ref().ok())
        .fold((Vec::new(), 0, 0), |mut acc, res| {
            acc.0.push(res.content.clone());
            acc.1 += res.input_tokens;
            acc.2 += res.output_tokens;
            acc
        });

    let search_results = ok_contents
        .iter()
        .map(|res| {
            format!(
                "<search_result>{}</search_result>",
                SingleResponse::from(res)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    log::debug!("Search results:\n{search_results}\n");

    let texts = ok_contents
        .iter()
        .map(|v| {
            v.iter()
                .filter_map(|f| match f {
                    ContentType::Text(s) => Some(s),
                    ContentType::ToolCall(_) => None,
                })
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .flatten() // We should only have one here anyway
        .collect::<Vec<_>>();

    let request = ChatRequest {
        chat_history: Vec::new(),
        max_tokens: None,
        message: get_summarize_prompt(&query, &texts),
        tools: None,
    };

    let final_draft_start = Instant::now();
    let result = llm_client.send_message(&request).await;
    let final_draft_duration = final_draft_start.elapsed();
    match result {
        Ok(response) => {
            writeln!(
                &mut ctx.err,
                "{DIM_TEXT}Final draft completed in {final_draft_duration:.2?}{RESET}"
            )?;

            writeln!(&mut ctx.out, "\n-----")?;
            writeln!(&mut ctx.out, "{}", SingleResponse::from(response.content))?;

            total_input_tokens += response.input_tokens;
            total_output_tokens += response.output_tokens;
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
pub async fn cli<O: Write, E: Write>(mut ctx: Context<O, E>) -> Result<(), CLIError> {
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
        )?;

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
                let command = command.trim();

                if !command.is_empty()
                    && let Err(e) = rl.add_history_entry(command)
                {
                    log::debug!("Failed to write history entry: {e}");
                }

                match command {
                    "" => {}
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
                        writeln!(&mut ctx.out, "/quit\t\tExit the program")?;
                        writeln!(&mut ctx.out)?;
                    }
                    "/checkhealth" => {
                        checkhealth(&mut ctx).await;
                    }
                    "/doctor" => {
                        doctor(&mut ctx).await?;
                    }
                    "/embed" => {
                        if let Err(e) = embed(&mut ctx, false).await {
                            writeln!(
                                &mut ctx.err,
                                "Failed to create embeddings. You may find relevant error messages below:\n\t{e}"
                            )?;
                        }
                    }
                    "/process" => {
                        if let Err(e) = process(&mut ctx).await {
                            writeln!(
                                &mut ctx.err,
                                "Failed to create embeddings. You may find relevant error messages below:\n\t{e}"
                            )?;
                        }
                    }
                    "/index" => {
                        if let Err(e) = update_indices(&mut ctx).await {
                            writeln!(
                                &mut ctx.err,
                                "Failed to update indexes. You may find the below information useful:\n\t{e}"
                            )?;
                        }
                    }
                    "/stats" => {
                        if let Err(e) = stats(&mut ctx).await {
                            // This only errors on an I/O failure
                            writeln!(&mut ctx.err, "Failed to write statistics to buffer. {e}")?;
                        }
                    }
                    "/quit" | "/exit" | "quit" | "exit" => {
                        break;
                    }
                    query => {
                        writeln!(&mut ctx.out)?;

                        // Check for a threshold to ensure this isn't an accidental Enter-hit.
                        #[allow(clippy::items_after_statements)]
                        const MIN_QUERY_LENGTH: usize = 10;

                        if query.len() < MIN_QUERY_LENGTH {
                            writeln!(&mut ctx.out, "Invalid command: {query}")?;
                            continue;
                        }

                        // Search queries have priority
                        if query.starts_with("/search") {
                            let search_term = query.strip_prefix("/search").unwrap().trim();
                            if search_term.is_empty() {
                                writeln!(
                                    &mut ctx.err,
                                    "Please provide a search term after /search."
                                )?;
                                continue;
                            }

                            if let Err(e) = search_for_papers(search_term.into(), &mut ctx).await {
                                writeln!(
                                    &mut ctx.err,
                                    "Failed to perform a vector search. You may find relevant error messages below:\n\t{e}"
                                )?;
                            }

                            continue;
                        } else if query.starts_with("/embed") {
                            // Handle `/embed fix`
                            let subcmd = query.strip_prefix("/embed").unwrap().trim();
                            if subcmd != "fix" {
                                writeln!(&mut ctx.err, "Invalid subcommand to /embed: {subcmd}")?;
                                continue;
                            }

                            embed(&mut ctx, true).await?;
                            continue;
                        }

                        if let Err(e) = run_query(query.into(), &mut ctx).await {
                            writeln!(
                                &mut ctx.err,
                                "Failed to answer the question. You may find relevant error messages below:\n\t{e}",
                            )?;
                        }
                    }
                }
            }
            Err(ReadlineError::Signal(rustyline::error::Signal::Resize)) => {
                // Handle SIGWINCH; we should just rewrite the prompt and continue
                continue;
            }
            _ => break,
        }
    }

    rl.save_history(&history_path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::cli::app::{BATCH_ITER_FILE, checkhealth, embed, search_for_papers, stats};
    use crate::common::State;
    use crate::config::{Config, VoyageAIConfig};
    use arrow_array::{RecordBatch, StringArray};
    use arrow_ipc::writer::FileWriter;
    use rag::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use rag::vector::lance::DB_URI;
    use serial_test::serial;
    use std::fs::{self, File};
    use std::io::Cursor;
    use std::sync::Arc;
    use temp_env;

    use crate::{
        cli::app::{process, run_query},
        common::{Args, Context},
    };

    fn get_config() -> Config {
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
    fn create_test_context() -> Context<Cursor<Vec<u8>>, Cursor<Vec<u8>>> {
        let args = Args {
            tui: false,
            print_summaries: false,
            log_level: log::LevelFilter::Off,
        };
        let out_buf: Vec<u8> = Vec::new();
        let out = Cursor::new(out_buf);

        let err_buf: Vec<u8> = Vec::new();
        let err = Cursor::new(err_buf);

        let config = get_config();

        Context {
            state: State::default(),
            config,
            args,
            out,
            err,
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_embed() {
        dotenv::dotenv().ok();

        // Clean up any existing data directories
        let _ = std::fs::remove_dir_all(format!("rag/{DB_URI}"));
        let _ = std::fs::remove_dir_all(DB_URI);

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

        // Actually call `embed`
        let result = embed(&mut ctx, false).await;
        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let err = String::from_utf8(ctx.err.into_inner()).unwrap();
        assert!(err.is_empty());

        // Clean up
        if fs::metadata(BATCH_ITER_FILE).is_ok() {
            fs::remove_file(BATCH_ITER_FILE).expect("Failed to clean up BATCH_ITER_FILE");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_process() {
        dotenv::dotenv().ok();

        // Clean up any existing data directories
        let _ = std::fs::remove_dir_all(format!("rag/{DB_URI}"));
        let _ = std::fs::remove_dir_all(DB_URI);

        let mut ctx = create_test_context();

        let result = temp_env::async_with_vars([("CI", Some("true"))], process(&mut ctx)).await;

        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.clone().into_inner()).unwrap();
        assert!(output.contains("Successfully parsed library!"));

        let stats = stats(&mut ctx).await;
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(stats.is_ok());
        assert!(output.contains("Table statistics:"));
        assert!(output.contains("Number of rows: 8"));

        // Cleanup
        if fs::metadata(BATCH_ITER_FILE).is_ok() {
            fs::remove_file(BATCH_ITER_FILE).expect("Failed to clean up BATCH_ITER_FILE");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_search_only() {
        dotenv::dotenv().ok();
        let mut setup_ctx = create_test_context();

        // Clean up any existing data directories
        let _ = std::fs::remove_dir_all(format!("rag/{DB_URI}"));
        let _ = std::fs::remove_dir_all(format!("zqa/{DB_URI}"));
        let _ = std::fs::remove_dir_all(DB_URI);

        // `process` needs to be run before `search_for_papers`
        let result =
            temp_env::async_with_vars([("CI", Some("true"))], process(&mut setup_ctx)).await;
        assert!(result.is_ok());

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true"))],
            search_for_papers(
                "How should I oversample in defect prediction?".into(),
                &mut ctx,
            ),
        )
        .await;

        if let Err(e) = &result {
            dbg!(e);
        }

        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.len() > 20);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_run_query() {
        dotenv::dotenv().ok();
        let mut setup_ctx = create_test_context();

        // `process` needs to be run before `run_query`
        let _ = temp_env::async_with_vars([("CI", Some("true"))], process(&mut setup_ctx)).await;

        let mut ctx = create_test_context();
        let result = temp_env::async_with_vars(
            [("CI", Some("true"))],
            run_query(
                "How should I oversample in defect prediction?".into(),
                &mut ctx,
            ),
        )
        .await;

        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Total token usage:"));
    }

    #[tokio::test]
    #[serial]
    async fn test_checkhealth_no_database() {
        dotenv::dotenv().ok();

        // Clean up any existing data directories
        let _ = std::fs::remove_dir_all(format!("rag/{DB_URI}"));
        let _ = std::fs::remove_dir_all(DB_URI);

        let mut ctx = create_test_context();
        checkhealth(&mut ctx).await;

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("directory does not exist"));
    }

    #[tokio::test]
    #[serial]
    async fn test_checkhealth_with_database() {
        dotenv::dotenv().ok();

        // Clean up any existing data directories
        let _ = std::fs::remove_dir_all(format!("rag/{DB_URI}"));
        let _ = std::fs::remove_dir_all(DB_URI);

        // First create a database by running process
        let mut setup_ctx = create_test_context();
        let result =
            temp_env::async_with_vars([("CI", Some("true"))], process(&mut setup_ctx)).await;
        assert!(result.is_ok());

        // Now run health check
        let mut ctx = create_test_context();
        checkhealth(&mut ctx).await;

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("LanceDB Health Check Results"));
        assert!(output.contains("directory exists"));
        assert!(output.contains("Table is accessible"));
        assert!(output.contains("Table has"));
    }
}
