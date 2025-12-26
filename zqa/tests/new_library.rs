use arrow_array::RecordBatchIterator;
use dotenv::dotenv;
use lancedb::embeddings::EmbeddingDefinition;
use log::LevelFilter;
use std::env;
use zqa::common::setup_logger;
use zqa::config::{AnthropicConfig, Config, VoyageAIConfig};
use zqa::full_library_to_arrow;
use zqa_rag::constants::{
    DEFAULT_MAX_CONCURRENT_REQUESTS, DEFAULT_MAX_RETRIES, DEFAULT_VOYAGE_EMBEDDING_DIM,
};
use zqa_rag::vector::lance::insert_records;

#[tokio::test]
async fn test_integration_works() {
    dotenv().ok();
    setup_logger(LevelFilter::Info).unwrap();

    if env::var("INTEGRATION_TESTS").is_err() {
        // Only enable if integration testing is desired
        return;
    }

    let config = Config {
        model_provider: "anthropic".into(),
        embedding_provider: "voyageai".into(),
        reranker_provider: "voyageai".into(),
        max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        max_retries: DEFAULT_MAX_RETRIES,
        gemini: None,
        openai: None,
        cohere: None,
        openrouter: None,
        anthropic: Some(AnthropicConfig {
            model: Some("claude-sonnet-4-5".into()),
            api_key: env::var("ANTHROPIC_API_KEY").unwrap().into(),
            max_tokens: 8192,
        }),
        voyageai: Some(VoyageAIConfig {
            reranker: Some("rerank-2.5".into()),
            embedding_model: Some("voyage-3-large".into()),
            embedding_dims: Some(DEFAULT_VOYAGE_EMBEDDING_DIM as usize),
            api_key: Some(env::var("VOYAGE_AI_API_KEY").unwrap()),
        }),
    };

    let record_batch = full_library_to_arrow(&config, None, None).await;
    assert!(record_batch.is_ok());

    let record_batch = record_batch.unwrap();
    let schema = record_batch.schema();
    let batches = vec![Ok(record_batch.clone())];
    let batch_iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());

    let db = insert_records(
        batch_iter,
        None,
        &config.get_embedding_config().unwrap(),
        EmbeddingDefinition::new("pdf_text", "voyageai", Some("embeddings")),
    )
    .await;

    assert!(db.is_ok());
}
