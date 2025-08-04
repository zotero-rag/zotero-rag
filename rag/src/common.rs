use http::{HeaderMap, StatusCode};
use reqwest::Response;
use serde::{Deserialize, Serialize};
use std::{env, time::Duration};

use crate::llm::{errors::LLMError, http_client::HttpClient};

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

    let key = env::var("OPENAI_API_KEY")?;
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

fn calculate_backoff_delay(attempt: usize, response: &Response) -> Duration {
    if let Some(retry_after) = response.headers().get("retry-after") {
        if let Ok(wait_time_str) = retry_after.to_str() {
            if let Ok(wait_time) = wait_time_str.parse::<u64>() {
                return Duration::from_secs(wait_time);
            } else {
                log::warn!("Retry-After value {wait_time_str} could not be parsed as a u64");
            }
        } else {
            log::warn!("Retry-After value {retry_after:?} could not be converted to a string");
        }
    }

    // Adding a jitter helps mitigate the thundering herd problem
    let jittered_exp = 2_f64 * (1.0 + rand::random::<f64>());
    let delay = 1000.0 * jittered_exp.powi(attempt as i32);

    Duration::from_millis(delay.round() as u64)
}

pub async fn request_with_backoff<T: HttpClient>(
    client: &T,
    url: &str,
    headers: &HeaderMap,
    request: impl Serialize + Sync + Send,
    max_retries: usize,
) -> Result<Response, LLMError> {
    let mut attempt = 0;

    loop {
        let response = client.post_json(url, headers.clone(), &request).await?;

        if response.status().is_success() {
            return Ok(response);
        }

        if response.status() == StatusCode::TOO_MANY_REQUESTS && attempt < max_retries {
            let delay = calculate_backoff_delay(attempt, &response);
            let _ = tokio::time::sleep(delay).await;
            attempt += 1;
            continue;
        }

        return Ok(response);
    }
}
