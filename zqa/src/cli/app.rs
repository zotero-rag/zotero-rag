use arrow_array::{self, RecordBatchIterator};
use lancedb::embeddings::EmbeddingDefinition;
use rag::vector::lance::create_initial_table;

use crate::{library_to_arrow, utils::library::parse_library_metadata};
use arrow_ipc::writer::FileWriter;
use std::{
    fs::File,
    io::{self, Write},
};

pub async fn process() {
    const WARNING_THRESHOLD: usize = 100;
    let item_metadata = parse_library_metadata(None, None);

    if let Err(parse_err) = item_metadata {
        println!("Could not parse library metadata: {parse_err}");
        return;
    }

    let item_metadata = item_metadata.unwrap();
    let metadata_length = item_metadata.len();
    if metadata_length >= WARNING_THRESHOLD {
        println!("Your library has {metadata_length} items. Parsing may take a while. Continue?");
        print!("(/process) >>> ");
        let _ = io::stdout().flush();

        let mut option = String::new();
        io::stdin()
            .read_line(&mut option)
            .expect("Failed to read input.");

        if ["n", "no"].contains(&option.as_str()) {
            return;
        }
    }

    let record_batch = library_to_arrow(None, None).expect("Failed to parse library");
    let schema = record_batch.schema();
    let batches = vec![Ok(record_batch.clone())];
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());

    // Write to binary file using Arrow IPC format
    let file = File::create("batch_iter.bin").expect("Failed to create output file");
    let mut writer = FileWriter::try_new(file, &schema).expect("Failed to create writer");

    writer.write(&record_batch).expect("Failed to write batch");
    writer.finish().expect("Failed to finish writing");

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
                println!("/quit\t\tExit the program");
                println!();
            }
            "/process" => {
                process().await;
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
