use serde::{Deserialize, Serialize};
use std::env;

use crate::llm::errors::LLMError;

pub async fn get_openai_embedding(text: String) -> Result<Vec<f32>, LLMError> {
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

    let key = env::var("OPENAI_API_KEY").expect("OpenAI API key not set");
    let model = env::var("OPENAI_EMBEDDING_MODEL").unwrap_or("text-embedding-3-small".to_string());

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
    let json: serde_json::Value = match serde_json::from_str(&body) {
        Ok(json) => json,
        Err(_) => return Err(LLMError::DeserializationError(body)),
    };
    let response: EmbeddingResponse = match serde_json::from_value(json) {
        Ok(response) => response,
        Err(_) => return Err(LLMError::DeserializationError(body)),
    };

    Ok(response.data[0].embedding.clone())
}
