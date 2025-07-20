use arrow_array::{self, RecordBatch, RecordBatchIterator};
use lancedb::embeddings::EmbeddingDefinition;
use rag::vector::lance::create_initial_table;

use crate::cli::errors::CLIError;
use crate::common::Args;
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
async fn embed(args: &Args) -> Result<(), CLIError> {
    let file = File::open(BATCH_ITER_FILE)?;
    let reader = FileReader::try_new(file, None)?;

    let mut batches = Vec::<Result<RecordBatch, arrow_schema::ArrowError>>::new();
    for batch in reader {
        batches.push(batch);
    }

    if batches.is_empty() {
        eprintln!("(/embed) It seems {BATCH_ITER_FILE} contains no batches. Exiting early.");
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

    print!("Successfully loaded {n_batches} batch");

    if n_batches > 1 {
        print!("es");
    }
    println!(".");

    let embedding_provider = args.embedding.as_str();
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
        println!("Successfully parsed library!");
        std::fs::remove_file(BATCH_ITER_FILE)?;
    } else if let Err(e) = db {
        println!("Parsing library failed: {e}");
        println!("Your {BATCH_ITER_FILE} file has been left untouched.");
    }

    Ok(())
}

/// Process a user's Zotero library. This acts as the main function provided by the CLI.
/// This parses the library, extracts the text from each file, stores them in a LanceDB
/// table, and adds their embeddings. If the last step fails, the parsed texts are stored
/// in `BATCH_ITER_FILE`.
async fn process(args: &Args) -> Result<(), CLIError> {
    const WARNING_THRESHOLD: usize = 100;
    let item_metadata = parse_library_metadata(None, Some(50));

    if let Err(parse_err) = item_metadata {
        println!("Could not parse library metadata: {parse_err}");
        return Ok(());
    }

    let item_metadata = item_metadata.unwrap();
    let metadata_length = item_metadata.len();
    if metadata_length >= WARNING_THRESHOLD {
        println!("Your library has {metadata_length} items. Parsing may take a while. Continue?");
        print!("(/process) >>> ");
        let _ = io::stdout().flush();

        let mut option = String::new();
        io::stdin().read_line(&mut option)?;

        let option = option.trim().to_lowercase();
        if ["n", "no", "false", "0"].contains(&option.as_str()) {
            return Ok(());
        }
    }

    let record_batch = library_to_arrow(None, Some(50))?;
    let schema = record_batch.schema();
    let batches = vec![Ok(record_batch.clone())];
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());

    // Write to binary file using Arrow IPC format
    let file = File::create("batch_iter.bin")?;
    let mut writer = FileWriter::try_new(file, &schema)?;

    writer.write(&record_batch)?;
    writer.finish()?;

    let embedding_provider = args.embedding.as_str();
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
            println!("Successfully parsed library!");
            std::fs::remove_file("batch_iter.bin")?;
        }
        Err(e) => {
            println!("Parsing library failed: {e}");
            println!(
                "The parsed PDFs have been saved in 'batch_iter.bin'. Run '/embed' to retry embedding."
            );
        }
    }

    Ok(())
}

pub async fn cli(args: Args) {
    loop {
        print!(">>> ");
        let _ = io::stdout().flush();

        let mut command = String::new();
        io::stdin()
            .read_line(&mut command)
            .expect("Failed to read command.");

        let command = command.trim();

        match command {
            "" => {}
            "/help" => {
                println!("Available commands:\n");
                println!("/help\t\tShow this help message");
                println!("/process\tPre-process Zotero library. Use to update the database.");
                println!("/embed\t\tRepair failed DB creation by re-adding embeddings.");
                println!("/quit\t\tExit the program");
                println!();
            }
            "/embed" => {
                if embed(&args).await.is_err() {
                    eprintln!(
                        "Failed to create embeddings. You may find relevant error messages above."
                    );
                }
            }
            "/process" => {
                if process(&args).await.is_err() {
                    eprintln!(
                        "Failed to create embeddings. You may find relevant error messages above."
                    );
                }
            }
            "/quit" | "/exit" | "quit" | "exit" => {
                break;
            }
            invalid => {
                println!("Invalid command: {invalid}");
            }
        }
    }
}
