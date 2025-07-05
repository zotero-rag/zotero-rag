use lancedb::embeddings::EmbeddingDefinition;
use rag::vector::lance::create_initial_table;

use crate::{library_to_arrow, utils::library::parse_library_metadata};
use std::io::{self, Write};

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

        let mut option = String::new();
        io::stdin()
            .read_line(&mut option)
            .expect("Failed to read input.");

        if ["n", "no"].contains(&option.as_str()) {
            return;
        }
    }

    let batch_iter = library_to_arrow(None, None).expect("Failed to parse library");
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
        println!("Parsing library failed: {}", db.err().unwrap().to_string())
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
                println!();
            }
            "/process" => {
                process().await;
            }
            invalid => {
                println!("Invalid command: {invalid}");
            }
        }
    }
}
