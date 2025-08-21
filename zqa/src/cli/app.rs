use crate::cli::placeholder::PlaceholderText;
use crate::cli::prompts::{get_extraction_prompt, get_summarize_prompt};
use crate::cli::readline::get_readline_config;
use crate::utils::arrow::vector_search;
use crate::utils::library::get_new_library_items;
use arrow_array::{self, RecordBatch, RecordBatchIterator};
use lancedb::embeddings::EmbeddingDefinition;
use rag::llm::base::{ApiClient, ApiResponse, ModelProviders, UserMessage};
use rag::llm::errors::LLMError;
use rag::llm::factory::get_client_by_provider;
use rag::vector::lance::{create_initial_table, db_statistics, lancedb_exists, perform_health_check};
use rustyline::error::ReadlineError;
use tokio::task::JoinSet;

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::{library_to_arrow, utils::library::parse_library_metadata};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
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

/// ANSI escape code for red text (errors)
const RED_TEXT: &str = "\x1b[31m";

/// ANSI escape code for yellow text (warnings)
const YELLOW_TEXT: &str = "\x1b[33m";

/// ANSI escape code for blue text (informational)
const BLUE_TEXT: &str = "\x1b[34m";

/// Embed text from PDFs parsed, in case this step previously failed. This function reads
/// the `BATCH_ITER_FILE` and uses the data in there to compute embeddings and write out
/// the LanceDB table.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
async fn embed<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
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

    let embedding_provider = ctx.args.embedding.as_str();
    let db = create_initial_table(
        batch_iter,
        Some(&["library_key"]),
        EmbeddingDefinition::new(
            "pdf_text", // source column
            embedding_provider,
            Some("embeddings"), // dest column
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

/// Process a user's Zotero library. This acts as one of the main functions provided by the CLI.
/// This parses the library, extracts the text from each file, stores them in a LanceDB
/// table, and adds their embeddings. If the last step fails, the parsed texts are stored
/// in `BATCH_ITER_FILE`.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
async fn process<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    const WARNING_THRESHOLD: usize = 100;

    let embedding_name = ctx.args.embedding.clone();

    let item_metadata = match lancedb_exists().await {
        true => get_new_library_items(&embedding_name).await,
        false => parse_library_metadata(None, None),
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

    let record_batch = library_to_arrow(&embedding_name, None, None).await?;
    let schema = record_batch.schema();
    let batches = vec![Ok(record_batch.clone())];
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());

    // Write to binary file using Arrow IPC format
    let file = File::create(BATCH_ITER_FILE)?;
    let mut writer = FileWriter::try_new(file, &schema)?;

    writer.write(&record_batch)?;
    writer.finish()?;

    let embedding_provider = ctx.args.embedding.as_str();
    let result = create_initial_table(
        batch_iter,
        Some(&["library_key"]),
        EmbeddingDefinition::new(
            "pdf_text", // source column
            embedding_provider,
            Some("embeddings"), // dest column
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
                "The parsed PDFs have been saved in 'batch_iter.bin'. Run '/embed' to retry embedding."
            )?;
        }
    }

    Ok(())
}

/// Given a positive number, returns a thousands separator-formatted string representation
///
/// # Arguments
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
    let embedding_name = ctx.args.embedding.clone();

    let vector_search_start = Instant::now();
    let search_results = vector_search(query.clone(), embedding_name).await?;
    let vector_search_duration = vector_search_start.elapsed();
    writeln!(
        &mut ctx.err,
        "{DIM_TEXT}Vector search completed in {vector_search_duration:.2?}{RESET}"
    )?;

    search_results.iter().for_each(|item| {
        writeln!(&mut ctx.out, "{}", item.metadata.title).unwrap_or_else(|_| {
            eprintln!("Could not write out search results.");
        });
    });
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
async fn run_query<O: Write, E: Write>(
    query: String,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError> {
    let embedding_name = ctx.args.embedding.clone();
    let model_provider = ctx.args.model_provider.clone();

    let vector_search_start = Instant::now();
    let search_results = vector_search(query.clone(), embedding_name).await?;
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

    let mut set = JoinSet::new();

    let summarization_start = Instant::now();
    search_results.iter().for_each(|item| {
        let provider = model_provider.clone();
        let text = item.text.clone();
        let query_clone = query.clone();

        set.spawn(async move {
            let client = get_client_by_provider(&provider).unwrap();
            let message = UserMessage {
                chat_history: Vec::new(),
                max_tokens: None,
                message: get_extraction_prompt(&query_clone, &text),
            };

            client.send_message(&message).await
        });
    });

    let results: Vec<Result<ApiResponse, LLMError>> = set.join_all().await;
    let summarization_duration = summarization_start.elapsed();
    writeln!(
        &mut ctx.err,
        "{DIM_TEXT}Summarization completed in {summarization_duration:.2?}{RESET}"
    )?;

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
        .map(|res| format!("<search_result>{res}</search_result>"))
        .collect::<Vec<_>>()
        .join("\n");
    log::debug!("Search results:\n{search_results}\n");

    let client = get_client_by_provider(&model_provider).unwrap();
    let message = UserMessage {
        chat_history: Vec::new(),
        max_tokens: None,
        message: get_summarize_prompt(&query, ok_contents),
    };

    let final_draft_start = Instant::now();
    let result = client.send_message(&message).await;
    let final_draft_duration = final_draft_start.elapsed();
    match result {
        Ok(response) => {
            writeln!(
                &mut ctx.err,
                "{DIM_TEXT}Final draft completed in {final_draft_duration:.2?}{RESET}"
            )?;

            writeln!(&mut ctx.out, "\n-----")?;
            writeln!(&mut ctx.out, "{}", response.content)?;

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

/// Format file size in a human-readable format
fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    
    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

/// Performs comprehensive health checks on the LanceDB database and reports status
/// with colored output using ASCII escape codes.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
async fn checkhealth<O: Write, E: Write>(ctx: &mut Context<O, E>) -> Result<(), CLIError> {
    let embedding_name = &ctx.args.embedding;
    
    match perform_health_check(embedding_name).await {
        Ok(health_result) => {
            writeln!(&mut ctx.out, "\nLanceDB Health Check Results:\n")?;

            // Check directory existence and size
            if health_result.directory_exists {
                writeln!(
                    &mut ctx.out,
                    "{BLUE_TEXT}✓ Directory exists: {} ({}){RESET}",
                    "data/lancedb-table",
                    format_file_size(health_result.directory_size)
                )?;
            } else {
                writeln!(
                    &mut ctx.out,
                    "{RED_TEXT}✗ Directory does not exist: data/lancedb-table{RESET}"
                )?;
                return Ok(());
            }

            // Check table accessibility
            if health_result.table_accessible {
                writeln!(
                    &mut ctx.out,
                    "{BLUE_TEXT}✓ Table can be opened successfully{RESET}"
                )?;
            } else {
                writeln!(
                    &mut ctx.out,
                    "{RED_TEXT}✗ Table cannot be opened{RESET}"
                )?;
                return Ok(());
            }

            // Check row count
            if health_result.num_rows > 0 {
                writeln!(
                    &mut ctx.out,
                    "{BLUE_TEXT}✓ Table has {} rows{RESET}",
                    health_result.num_rows
                )?;
            } else {
                writeln!(
                    &mut ctx.out,
                    "{RED_TEXT}✗ Table has 0 rows{RESET}"
                )?;
                return Ok(());
            }

            // Check for zero embeddings
            if health_result.zero_embedding_count == 0 {
                writeln!(
                    &mut ctx.out,
                    "{BLUE_TEXT}✓ No rows with all-zero embeddings found{RESET}"
                )?;
            } else {
                writeln!(
                    &mut ctx.out,
                    "{RED_TEXT}✗ Found {} rows with all-zero embeddings{RESET}",
                    health_result.zero_embedding_count
                )?;
                
                // Write problematic titles to file
                if !health_result.zero_embedding_titles.is_empty() {
                    if let Err(e) = std::fs::write(
                        "bad_embeddings.txt",
                        health_result.zero_embedding_titles.join("\n") + "\n"
                    ) {
                        writeln!(&mut ctx.err, "Warning: Could not write bad_embeddings.txt: {e}")?;
                    } else {
                        writeln!(
                            &mut ctx.out,
                            "{BLUE_TEXT}  → Written problematic titles to bad_embeddings.txt{RESET}"
                        )?;
                    }
                    
                    writeln!(
                        &mut ctx.out,
                        "{YELLOW_TEXT}  → Suggestion: Run /repair to fix zero embeddings{RESET}"
                    )?;
                }
            }

            // Check index status
            if health_result.total_indexed_rows == health_result.num_rows {
                writeln!(
                    &mut ctx.out,
                    "{BLUE_TEXT}✓ All {} rows are indexed{RESET}",
                    health_result.total_indexed_rows
                )?;
                
                for (index_type, indexed_count) in &health_result.index_info {
                    writeln!(
                        &mut ctx.out,
                        "{BLUE_TEXT}  → {} index: {} rows{RESET}",
                        index_type, indexed_count
                    )?;
                }
            } else if health_result.total_indexed_rows > 0 {
                writeln!(
                    &mut ctx.out,
                    "{YELLOW_TEXT}⚠ Partially indexed: {}/{} rows indexed{RESET}",
                    health_result.total_indexed_rows,
                    health_result.num_rows
                )?;
                
                for (index_type, indexed_count) in &health_result.index_info {
                    writeln!(
                        &mut ctx.out,
                        "{YELLOW_TEXT}  → {} index: {} rows{RESET}",
                        index_type, indexed_count
                    )?;
                }
                
                writeln!(
                    &mut ctx.out,
                    "{YELLOW_TEXT}  → Suggestion: Run /index to index remaining rows{RESET}"
                )?;
            } else {
                writeln!(
                    &mut ctx.out,
                    "{YELLOW_TEXT}⚠ No indexes found{RESET}"
                )?;
                writeln!(
                    &mut ctx.out,
                    "{YELLOW_TEXT}  → Suggestion: Run /index to create indexes{RESET}"
                )?;
            }

            writeln!(&mut ctx.out)?;
        }
        Err(e) => {
            writeln!(
                &mut ctx.err,
                "{RED_TEXT}Health check failed: {e}{RESET}"
            )?;
        }
    }

    Ok(())
}

/// The core CLI implementation that implements a REPL for user commands.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   `std::io::Write` for `stdout` and `stderr`.
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
                if !command.trim().is_empty()
                    && let Err(e) = rl.add_history_entry(command.as_str())
                {
                    log::debug!("Failed to write history entry: {e}");
                }

                match command.as_str() {
                    "" => {}
                    "/help" => {
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
                        writeln!(&mut ctx.out, "/stats\t\tShow table statistics.")?;
                        writeln!(&mut ctx.out, "/checkhealth\tPerform database health checks.")?;
                        writeln!(&mut ctx.out, "/quit\t\tExit the program")?;
                        writeln!(&mut ctx.out)?;
                    }
                    "/embed" => {
                        if let Err(e) = embed(&mut ctx).await {
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
                    "/stats" => {
                        if let Err(e) = stats(&mut ctx).await {
                            // This only errors on an I/O failure
                            writeln!(&mut ctx.err, "Failed to write statistics to buffer. {e}")?;
                        }
                    }
                    "/checkhealth" => {
                        if let Err(e) = checkhealth(&mut ctx).await {
                            writeln!(
                                &mut ctx.err,
                                "Health check failed. You may find relevant error messages below:\n\t{e}"
                            )?;
                        }
                    }
                    "/quit" | "/exit" | "quit" | "exit" => {
                        break;
                    }
                    query => {
                        writeln!(&mut ctx.out)?;

                        // Check for a threshold to ensure this isn't an accidental Enter-hit.
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
                                    "Failed to perform a vector search. You may find relevant error messages below:\n\t{}",
                                    e
                                )?;
                            }

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
    use crate::cli::app::{BATCH_ITER_FILE, embed, search_for_papers, stats, checkhealth};
    use arrow_array::{RecordBatch, StringArray};
    use arrow_ipc::writer::FileWriter;
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

    fn create_test_context() -> Context<Cursor<Vec<u8>>, Cursor<Vec<u8>>> {
        let args = Args {
            tui: false,
            log_level: "none".into(),
            embedding: "voyageai".into(),
            model_provider: "anthropic".into(),
        };
        let out_buf: Vec<u8> = Vec::new();
        let out = Cursor::new(out_buf);

        let err_buf: Vec<u8> = Vec::new();
        let err = Cursor::new(err_buf);

        Context { args, out, err }
    }

    #[tokio::test]
    #[serial]
    async fn test_embed() {
        dotenv::dotenv().ok();

        // Clean up any existing data directories
        let _ = std::fs::remove_dir_all(format!("rag/{}", DB_URI));
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
        let result = embed(&mut ctx).await;
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
        let _ = std::fs::remove_dir_all(format!("rag/{}", DB_URI));
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
        let _ = std::fs::remove_dir_all(format!("rag/{}", DB_URI));
        let _ = std::fs::remove_dir_all(DB_URI);

        let mut ctx = create_test_context();
        let result = checkhealth(&mut ctx).await;

        assert!(result.is_ok());
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Directory does not exist"));
    }

    #[tokio::test]
    #[serial]
    async fn test_checkhealth_with_database() {
        dotenv::dotenv().ok();

        // Clean up any existing data directories
        let _ = std::fs::remove_dir_all(format!("rag/{}", DB_URI));
        let _ = std::fs::remove_dir_all(DB_URI);

        // First create a database by running process
        let mut setup_ctx = create_test_context();
        let result = temp_env::async_with_vars([("CI", Some("true"))], process(&mut setup_ctx)).await;
        assert!(result.is_ok());

        // Now run health check
        let mut ctx = create_test_context();
        let result = checkhealth(&mut ctx).await;

        assert!(result.is_ok());
        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("LanceDB Health Check Results"));
        assert!(output.contains("Directory exists"));
        assert!(output.contains("Table can be opened successfully"));
        assert!(output.contains("Table has"));
    }
}
