//! Shared functionality for generating OpenAI embeddings. Presently, this is also used by the
//! Anthropic client, but that will likely be changed, at which point this module will move to
//! `llm`.

use crate::constants::DEFAULT_OPENAI_EMBEDDING_MODEL;
use crate::constants::{DEFAULT_MAX_CONCURRENT_REQUESTS, OPENAI_EMBEDDING_DIM};
use crate::llm::errors::LLMError;
use arrow_array;
use futures::StreamExt;
use futures::stream;
use lancedb::arrow::arrow_schema::{DataType, Field};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;

/// Process the actual embedding request with the OpenAI API.
///
/// This function solely interacts with the OpenAI API; as such, it is not intended to be used
/// without the other wrapper functions provided in this file.
///
/// # Arguments:
///
/// * `text` - The text to embed.
///
/// # Returns
///
/// If successful, a `Vec<f32>` containing the embeddings.
async fn get_openai_embedding(text: String) -> Result<Vec<f32>, LLMError> {
    #[derive(Serialize)]
    struct EmbeddingRequest {
        model: String,
        input: String,
        encoding_format: String,
    }

    // Adding #[allow(dead_code)] to suppress warnings for fields required by the API
    // but not used directly in our code
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct EmbeddingResponseUsage {
        prompt_tokens: u32,
        total_tokens: u32,
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct EmbeddingResponseData {
        object: String,
        embedding: Vec<f32>,
        index: u16,
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct EmbeddingResponse {
        model: String,
        usage: EmbeddingResponseUsage,
        object: String,
        data: Vec<EmbeddingResponseData>,
    }

    let key = env::var("OPENAI_API_KEY")?;
    let model =
        env::var("OPENAI_EMBEDDING_MODEL").unwrap_or(DEFAULT_OPENAI_EMBEDDING_MODEL.to_string());

    let client = reqwest::Client::new();
    let request_body = EmbeddingRequest {
        model,
        input: text,
        encoding_format: "float".to_string(),
    };

    let response = client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(key)
        .header("content-type", "application/json")
        .json(&request_body)
        .send()
        .await?;

    let body = response.text().await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let response: EmbeddingResponse = serde_json::from_value(json.clone()).map_err(|e| {
        eprintln!("Failed to deserialize OpenAI embedding response: {e}");
        eprintln!(
            "Response body: {}",
            serde_json::to_string_pretty(&json).unwrap_or_else(|_| body.clone())
        );
        e
    })?;

    Ok(response.data[0].embedding.clone())
}

/// Shared embedding computation logic for OpenAI embeddings
/// This eliminates code duplication between OpenAI and Anthropic clients
///
/// # Errors
///
/// * `LLMError::EnvError` - If the OPENAI_API_KEY environment variable is not set
/// * `LLMError::TimeoutError` - If the HTTP request times out
/// * `LLMError::CredentialError` - If the API returns 401 or 403 status
/// * `LLMError::HttpStatusError` - If the API returns other unsuccessful HTTP status codes
/// * `LLMError::NetworkError` - If a network connectivity error occurs
/// * `LLMError::DeserializationError` - If the API response cannot be parsed
/// * `LLMError::GenericLLMError` - If other HTTP errors occur or Arrow array creation fails
pub async fn compute_openai_embeddings_async(
    source: Arc<dyn arrow_array::Array>,
) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
    let source_array = arrow_array::cast::as_string_array(&source);
    let texts: Vec<String> = source_array
        .iter()
        .filter_map(|s| Some(s?.to_owned()))
        .collect();

    // Create a stream of futures
    let futures = texts.iter().map(|text| get_openai_embedding(text.clone()));

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
        Arc::new(Field::new("item", DataType::Float32, true)),
        embedding_dim as i32,
        Arc::new(values),
        None,
    )
    .map_err(|e| LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {e}")))?;

    Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
}

/// Synchronous wrapper for embedding computation
///
/// # Errors
///
/// * `LLMError::EnvError` - If the OPENAI_API_KEY environment variable is not set
/// * `LLMError::TimeoutError` - If the HTTP request times out
/// * `LLMError::CredentialError` - If the API returns 401 or 403 status
/// * `LLMError::HttpStatusError` - If the API returns other unsuccessful HTTP status codes
/// * `LLMError::NetworkError` - If a network connectivity error occurs
/// * `LLMError::DeserializationError` - If the API response cannot be parsed
/// * `LLMError::GenericLLMError` - If other HTTP errors occur or Arrow array creation fails
pub fn compute_openai_embeddings_sync(
    source: Arc<dyn arrow_array::Array>,
) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(compute_openai_embeddings_async(source))
    })
}
