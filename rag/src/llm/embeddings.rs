use std::env;
use std::sync::Arc;

use arrow_array;
use futures::StreamExt;
use futures::stream;
use lancedb::arrow::arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;

use super::errors::LLMError;
use crate::common;
use crate::constants::VOYAGE_EMBEDDING_DIM;
use crate::constants::{DEFAULT_MAX_CONCURRENT_REQUESTS, OPENAI_EMBEDDING_DIM};
use crate::llm::http_client::ReqwestClient;
use crate::llm::openai::OpenAIClient;
use crate::llm::voyage::VoyageAIClient;

/// Shared embedding computation logic for OpenAI embeddings
/// This eliminates code duplication between OpenAI and Anthropic clients
pub async fn compute_openai_embeddings_async(
    source: Arc<dyn arrow_array::Array>,
) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
    let source_array = arrow_array::cast::as_string_array(&source);
    let texts: Vec<String> = source_array
        .iter()
        .filter_map(|s| Some(s?.to_owned()))
        .collect();

    // Create a stream of futures
    let futures = texts
        .iter()
        .map(|text| common::get_openai_embedding(text.clone()));

    // Convert to a stream and process with buffer_unordered to limit concurrency
    let max_concurrent = env::var("MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT_REQUESTS);

    // Process futures with limited concurrency
    let results = stream::iter(futures)
        .buffer_unordered(max_concurrent)
        .collect::<Vec<_>>()
        .await;

    // Process results and construct Arrow array
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for result in results {
        match result {
            Ok(embedding) => embeddings.push(embedding),
            Err(e) => return Err(e),
        }
    }

    // Convert to Arrow FixedSizeListArray
    let embedding_dim = if embeddings.is_empty() {
        OPENAI_EMBEDDING_DIM as usize // default for text-embedding-3-small
    } else {
        embeddings[0].len()
    };

    let flattened: Vec<f32> = embeddings.iter().flatten().copied().collect();
    let values = arrow_array::Float32Array::from(flattened);

    let list_array = arrow_array::FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        embedding_dim as i32,
        Arc::new(values),
        None,
    )
    .map_err(|e| LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {e}")))?;

    Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
}

/// Synchronous wrapper for embedding computation
pub fn compute_openai_embeddings_sync(
    source: Arc<dyn arrow_array::Array>,
) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(compute_openai_embeddings_async(source))
    })
}

/// Returns the embedding dimension given an embedding provider. Note that for Anthropic, we
/// actually use OpenAI's embeddings.
///
/// # Arguments
///
/// * `embedding_name` - Embedding provider name. Must be one of "openai", "anthropic", or
///   "voyageai".
///
/// # Returns
///
/// The dimensions of the embedding provider.
pub fn get_embedding_dims_by_provider(embedding_name: &str) -> u32 {
    match embedding_name {
        "openai" => OPENAI_EMBEDDING_DIM,
        "anthropic" => OPENAI_EMBEDDING_DIM,
        "voyageai" => VOYAGE_EMBEDDING_DIM,
        _ => panic!("Invalid embedding provider."),
    }
}

/// Gets an embedding provider, i.e., an atomically reference-counted, heap-allocated
/// `EmbeddingFunction` implementation.
///
/// # Arguments
///
/// * `embedding_name`: Embedding provider name. Must be one of "openai", "anthropic", or
///   "voyageai".
///
/// # Returns
///
/// An thread-safe object that can compute query embeddings.
pub fn get_embedding_provider(embedding_name: &str) -> Arc<dyn EmbeddingFunction> {
    match embedding_name {
        "openai" => Arc::new(OpenAIClient::<ReqwestClient>::default()),
        "anthropic" => Arc::new(OpenAIClient::<ReqwestClient>::default()),
        "voyageai" => Arc::new(VoyageAIClient::<ReqwestClient>::default()),
        _ => panic!("Invalid embedding provider."),
    }
}
