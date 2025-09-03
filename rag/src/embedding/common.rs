use std::sync::Arc;

use lancedb::embeddings::EmbeddingFunction;

use crate::constants::{GEMINI_EMBEDDING_DIM, OPENAI_EMBEDDING_DIM, VOYAGE_EMBEDDING_DIM};
use crate::embedding::voyage::VoyageAIClient;
use crate::llm::errors::LLMError;
use crate::llm::gemini::GeminiClient;
use crate::llm::http_client::ReqwestClient;
use crate::llm::openai::OpenAIClient;

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
        "gemini" => GEMINI_EMBEDDING_DIM,
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
pub fn get_embedding_provider(
    embedding_name: &str,
) -> Result<Arc<dyn EmbeddingFunction>, LLMError> {
    match embedding_name {
        "openai" => Ok(Arc::new(OpenAIClient::<ReqwestClient>::default())),
        "anthropic" => Ok(Arc::new(OpenAIClient::<ReqwestClient>::default())),
        "voyageai" => Ok(Arc::new(VoyageAIClient::<ReqwestClient>::default())),
        "gemini" => Ok(Arc::new(GeminiClient::<ReqwestClient>::default())),
        _ => Err(LLMError::InvalidProviderError(embedding_name.to_string())),
    }
}

/// A trait indicating reranking capabilities. This is made generic since it is expected that users
/// will pass in whatever type they convert `RecordBatch` to, as long as we can convert it into a
/// string in a non-consuming way. A user may also choose to `map` their `Vec<RecordBatch>` with
/// custom logic if they prefer (or if, for some reason, their struct's `AsRef<str>` is implemented
/// with a different purpose, but the resulting string isn't useful for reranking purposes).
pub trait Rerank<T: AsRef<str>> {
    fn rerank(items: Vec<T>, query: &str) -> Vec<T>;
}
