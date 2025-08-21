use arrow_array::RecordBatchIterator;
use dotenv::dotenv;
use lancedb::embeddings::EmbeddingDefinition;
use log::LevelFilter;
use zqa::common::setup_logger;

use std::env;

use rag::vector::lance::create_initial_table;
use zqa::library_to_arrow;

#[ignore]
#[tokio::test]
async fn test_integration_works() {
    dotenv().ok();
    setup_logger(LevelFilter::Info).unwrap();

    if env::var("CI").is_ok() {
        // Skip this test in CI environments
        return;
    }

    let record_batch = library_to_arrow("voyageai", None, None).await;
    assert!(record_batch.is_ok());

    let record_batch = record_batch.unwrap();
    let schema = record_batch.schema();
    let batches = vec![Ok(record_batch.clone())];
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());

    let db = create_initial_table(
        batch_iter,
        &[],
        EmbeddingDefinition::new(
            "pdf_text",         // source column
            "openai",           // embedding name, either "openai" or "anthropic"
            Some("embeddings"), // dest column
        ),
    )
    .await;

    assert!(db.is_ok());
}
