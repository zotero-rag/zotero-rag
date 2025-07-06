use arrow_array::{self, RecordBatch, RecordBatchIterator};
use lancedb::embeddings::EmbeddingDefinition;
use rag::vector::lance::create_initial_table;

use crate::cli::errors::CLIError;
use crate::{library_to_arrow, utils::library::parse_library_metadata};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use std::{
    fs::File,
    io::{self, Write},
};

async fn embed() -> Result<(), CLIError> {
    let file = File::open("batch_iter.bin")?;
    let reader = FileReader::try_new(file, None)?;

    let mut batches = Vec::<Result<RecordBatch, arrow_schema::ArrowError>>::new();
    for batch in reader {
        let batch = batch.unwrap();

        batches.push(Ok(batch));
    }

    if batches.len() == 0 {
        return Ok(());
    }

    // All batches should have the same schema, so we use the first batch
    let first_batch = batches.get(0).unwrap().as_ref()?;
    let schema = first_batch.schema();
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema);

    let db = create_initial_table(
        batch_iter,
        EmbeddingDefinition::new(
            "pdf_text",         // source column
            "anthropic",        // embedding name, either "openai" or "anthropic"
            Some("embeddings"), // dest column
        ),
    )
    .await;

    if db.is_ok() {
        println!("Successfully parsed library!");
    } else {
        println!("Parsing library failed: {}", db.err().unwrap());
    }

    Ok(())
}

async fn process() -> Result<(), CLIError> {
    const WARNING_THRESHOLD: usize = 100;
    let item_metadata = parse_library_metadata(None, None);

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
        if ["n", "no"].contains(&option.as_str()) {
            return Ok(());
        }
    }

    let record_batch = library_to_arrow(None, None)?;
    let schema = record_batch.schema();
    let batches = vec![Ok(record_batch.clone())];
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());

    // Write to binary file using Arrow IPC format
    let file = File::create("batch_iter.bin")?;
    let mut writer = FileWriter::try_new(file, &schema)?;

    writer.write(&record_batch)?;
    writer.finish()?;

    let db = create_initial_table(
        batch_iter,
        EmbeddingDefinition::new(
            "pdf_text",         // source column
            "anthropic",        // embedding name, either "openai" or "anthropic"
            Some("embeddings"), // dest column
        ),
    )
    .await;

    if db.is_ok() {
        println!("Successfully parsed library!");
        std::fs::remove_file("batch_iter.bin")?;
    } else {
        if let Err(e) = db {
            println!("Parsing library failed: {}", e);
            println!(
                "The parsed PDFs have been saved in 'batch_iter.bin'. Run '/embed' to retry embedding."
            );
        }
    }

    Ok(())
}

pub async fn cli() {
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
                if embed().await.is_err() {
                    eprintln!(
                        "Failed to create embeddings. You may find relevant error messages above."
                    );
                }
            }
            "/process" => {
                if process().await.is_err() {
                    eprintln!(
                        "Failed to create embeddings. You may find relevant error messages above."
                    );
                }
            }
            "/quit" => {
                break;
            }
            invalid => {
                println!("Invalid command: {invalid}");
            }
        }
    }
}
