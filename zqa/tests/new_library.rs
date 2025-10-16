use arrow_array::RecordBatchIterator;
use dotenv::dotenv;
use lancedb::embeddings::EmbeddingDefinition;
use log::LevelFilter;
use rag::vector::lance::insert_records;
use std::env;
use zqa::common::setup_logger;
use zqa::full_library_to_arrow;

#[tokio::test]
async fn test_integration_works() {
    dotenv().ok();
    setup_logger(LevelFilter::Info).unwrap();

    if !env::var("INTEGRATION_TESTS").is_ok() {
        // Only enable if integration testing is desired
        return;
    }

    let record_batch = full_library_to_arrow("voyageai", None, None).await;
    assert!(record_batch.is_ok());

    let record_batch = record_batch.unwrap();
    let schema = record_batch.schema();
    let batches = vec![Ok(record_batch.clone())];
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());

    let db = insert_records(
        batch_iter,
        None,
        EmbeddingDefinition::new(
            "pdf_text",         // source column
            "openai",           // embedding name, either "openai" or "anthropic"
            Some("embeddings"), // dest column
        ),
    )
    .await;

    assert!(db.is_ok());
}
