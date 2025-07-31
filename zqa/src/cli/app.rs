use crate::cli::prompts::{get_extraction_prompt, get_summarize_prompt};
use crate::utils::arrow::vector_search;
use arrow_array::{self, RecordBatch, RecordBatchIterator};
use lancedb::embeddings::EmbeddingDefinition;
use rag::llm::base::{ApiClient, ApiResponse, ModelProviders, UserMessage};
use rag::llm::errors::LLMError;
use rag::llm::factory::get_client_by_provider;
use rag::vector::lance::{create_initial_table, db_statistics};
use tokio::task::JoinSet;

use crate::cli::errors::CLIError;
use crate::common::Context;
use crate::{library_to_arrow, utils::library::parse_library_metadata};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use std::{
    fs::File,
    io::{self, Write},
};

/// A file that contains parsed PDF texts from the user's Zotero library. In case the
/// embedding generation fails, the user does not need to rerun the full PDF parsing,
/// and can simply retry the embedding. Note that this is *not* supposed to be user-facing
/// and all interaction with it is meant for use by the CLI.
const BATCH_ITER_FILE: &str = "batch_iter.bin";

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
    let item_metadata = parse_library_metadata(None, None);

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
            "Your library has {metadata_length} items. Parsing may take a while. Continue?"
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

    let record_batch = library_to_arrow(None, None)?;
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

    let search_results = vector_search(query.clone(), embedding_name).await?;

    if !ModelProviders::contains(&model_provider) {
        return Err(CLIError::LLMError(format!(
            "Model provider {model_provider} is not valid."
        )));
    }

    let mut set = JoinSet::new();

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

    let client = get_client_by_provider(&model_provider).unwrap();
    let message = UserMessage {
        chat_history: Vec::new(),
        max_tokens: None,
        message: get_summarize_prompt(&query, ok_contents),
    };
    match client.send_message(&message).await {
        Ok(response) => {
            writeln!(&mut ctx.out, "{}", response.content)?;

            total_input_tokens += response.input_tokens;
            total_output_tokens += response.output_tokens;
        }
        Err(e) => {
            writeln!(
                &mut ctx.err,
                "Failed to call the LLM endpoint for the final response: {e}"
            )?;
        }
    }

    writeln!(&mut ctx.out, "\nTotal token usage:")?;
    writeln!(&mut ctx.out, "\tInput tokens: {total_input_tokens}")?;
    writeln!(&mut ctx.out, "\tOutput tokens: {total_output_tokens}")?;

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
pub async fn cli<O: Write, E: Write>(mut ctx: Context<O, E>) -> Result<(), CLIError> {
    loop {
        write!(&mut ctx.out, ">>> ")?;
        ctx.out.flush()?;

        let mut command = String::new();
        io::stdin()
            .read_line(&mut command)
            .expect("Failed to read command.");

        let command = command.trim();

        match command {
            "" => {}
            "/help" => {
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
                writeln!(&mut ctx.out, "/stats\t\tShow table statistics.")?;
                writeln!(&mut ctx.out, "/quit\t\tExit the program")?;
                writeln!(&mut ctx.out)?;
            }
            "/embed" => {
                if embed(&mut ctx).await.is_err() {
                    eprintln!(
                        "Failed to create embeddings. You may find relevant error messages above."
                    );
                }
            }
            "/process" => {
                if process(&mut ctx).await.is_err() {
                    eprintln!(
                        "Failed to create embeddings. You may find relevant error messages above."
                    );
                }
            }
            "/stats" => {
                if stats(&mut ctx).await.is_err() {
                    // This only errors on an I/O failure
                    eprintln!("Failed to write statistics to buffer.");
                }
            }
            "/quit" | "/exit" | "quit" | "exit" => {
                break;
            }
            query => {
                // Check for a threshold to ensure this isn't an accidental Enter-hit.
                const MIN_QUERY_LENGTH: usize = 10;

                if query.len() < MIN_QUERY_LENGTH {
                    writeln!(&mut ctx.out, "Invalid command: {query}")?;
                    continue;
                }

                if run_query(query.into(), &mut ctx).await.is_err() {
                    eprintln!(
                        "Failed to answer the question. You may find relevant error messages above."
                    );
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::cli::app::{BATCH_ITER_FILE, embed, stats};
    use arrow_array::{RecordBatch, StringArray};
    use arrow_ipc::writer::FileWriter;
    use rag::vector::lance::TABLE_NAME;
    use serial_test::serial;
    use std::fs::{self, File};
    use std::io::Cursor;
    use std::sync::Arc;

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
        if fs::metadata(TABLE_NAME).is_ok() {
            fs::remove_dir_all(TABLE_NAME).expect("Failed to clean up test database");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_process() {
        dotenv::dotenv().ok();

        // This test needs the CI variable to be set. Some other tests rely on it, so we don't need
        // to worry about unsetting it.
        unsafe { std::env::set_var("CI", "true") };

        let mut ctx = create_test_context();

        let result = process(&mut ctx).await;

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

        // `TABLE_NAME` is also used as the DB directory
        if fs::metadata(TABLE_NAME).is_ok() {
            fs::remove_dir_all(TABLE_NAME).expect("Failed to clean up test database");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_run_query() {
        dotenv::dotenv().ok();

        // `process` needs to be run before `run_query`
        let mut setup_ctx = create_test_context();
        let _ = process(&mut setup_ctx).await;

        let mut ctx = create_test_context();
        let result = run_query(
            "How should I oversample in defect prediction?".into(),
            &mut ctx,
        )
        .await;

        assert!(result.is_ok());

        let output = String::from_utf8(ctx.out.into_inner()).unwrap();
        assert!(output.contains("Total token usage:"));
    }
}
