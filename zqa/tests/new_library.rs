use std::env;

use dotenv::dotenv;
use log::LevelFilter;
use zqa::common::setup_logger;
use zqa::config::{AnthropicConfig, Config, VoyageAIConfig};
use zqa::full_library_to_arrow;
use zqa_macros::test_ok;
use zqa_rag::capabilities::{EmbeddingProvider, ModelProvider, RerankerProvider};
use zqa_rag::constants::{
    DEFAULT_MAX_CONCURRENT_REQUESTS, DEFAULT_MAX_RETRIES, DEFAULT_VOYAGE_EMBEDDING_DIM,
};
use zqa_rag::vector::backends::{backend::VectorBackend, lance::LanceBackend};

#[tokio::test]
async fn test_integration_works() {
    dotenv().ok();
    setup_logger(LevelFilter::Info).unwrap();

    if env::var("INTEGRATION_TESTS").is_err() {
        // Only enable if integration testing is desired
        return;
    }

    let config = Config {
        model_provider: ModelProvider::Anthropic,
        embedding_provider: EmbeddingProvider::VoyageAI,
        reranker_provider: Some(RerankerProvider::VoyageAI),
        max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        max_retries: DEFAULT_MAX_RETRIES,
        gemini: None,
        ollama: None,
        openai: None,
        cohere: None,
        openrouter: None,
        zeroentropy: None,
        anthropic: Some(AnthropicConfig {
            model: Some("claude-sonnet-4-5".into()),
            model_small: Some("claude-haiku-4-5".into()),
            api_key: env::var("ANTHROPIC_API_KEY").unwrap().into(),
            max_tokens: 8192,
            reasoning_budget: None,
        }),
        voyageai: Some(VoyageAIConfig {
            reranker: Some("rerank-2.5".into()),
            embedding_model: Some("voyage-3-large".into()),
            embedding_dims: Some(DEFAULT_VOYAGE_EMBEDDING_DIM as usize),
            api_key: Some(env::var("VOYAGE_AI_API_KEY").unwrap()),
        }),
    };

    let embedding_config = config.get_embedding_config().unwrap();
    let schema = zqa::utils::arrow::get_schema(embedding_config.provider()).await;
    let backend = LanceBackend::new(
        embedding_config,
        std::sync::Arc::new(schema),
        "pdf_text".into(),
    );

    let record_batch = full_library_to_arrow(&backend, None, None).await;
    test_ok!(record_batch);

    let record_batch = record_batch.unwrap();
    let batches = vec![record_batch.clone()];

    let backend = LanceBackend::new(
        config.get_embedding_config().unwrap(),
        record_batch.schema(),
        "pdf_text".into(),
    );
    let db = backend.insert_items(batches, None).await;

    test_ok!(db);
}
