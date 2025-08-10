use http::{HeaderMap, StatusCode};
use reqwest::Response;
use serde::{Deserialize, Serialize};
use std::{env, time::Duration};

use crate::constants::DEFAULT_OPENAI_EMBEDDING_MODEL;
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
    let model = env::var("OPENAI_EMBEDDING_MODEL").unwrap_or(DEFAULT_OPENAI_EMBEDDING_MODEL.to_string());

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
    let base_delay = 1000.0 * 2.0_f64.powi(attempt as i32);
    let jitter = base_delay * rand::random::<f64>();

    Duration::from_millis((base_delay + jitter).round() as u64)
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

        return Err(response.error_for_status().unwrap_err().into());
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        common::{calculate_backoff_delay, request_with_backoff},
        llm::http_client::HttpClient,
    };
    use http::HeaderMap;
    use reqwest::Response;
    use serde::Serialize;
    use serde_json::json;
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    struct MockRateLimitClient {
        call_count: Arc<Mutex<usize>>,
        max_failures: usize,
    }

    impl MockRateLimitClient {
        fn new(max_failures: usize) -> Self {
            Self {
                call_count: Arc::new(Mutex::new(0)),
                max_failures,
            }
        }
    }

    impl HttpClient for MockRateLimitClient {
        async fn post_json<T: Serialize + Send + Sync>(
            &self,
            _url: &str,
            _headers: HeaderMap,
            _body: &T,
        ) -> Result<Response, reqwest::Error> {
            let mut count = { self.call_count.lock().unwrap() };
            *count += 1;

            if *count <= self.max_failures {
                // Return 429 response with retry-after header
                let json = json!({"error": "Rate limit exceeded"});
                let bytes = bytes::Bytes::from(json.to_string());

                let http_response = http::Response::builder()
                    .status(429)
                    .header("content-type", "application/json")
                    .header("retry-after", "2")
                    .body(bytes)
                    .unwrap();

                Ok(Response::from(http_response))
            } else {
                // Return successful response
                let json = json!({"success": true});
                let bytes = bytes::Bytes::from(json.to_string());

                let http_response = http::Response::builder()
                    .status(200)
                    .header("content-type", "application/json")
                    .body(bytes)
                    .unwrap();

                Ok(Response::from(http_response))
            }
        }
    }

    #[tokio::test]
    async fn test_request_with_backoff_handles_429() {
        let client = MockRateLimitClient::new(2); // Fail twice, then succeed
        let headers = HeaderMap::new();
        let request = json!({"test": "data"});

        let result = request_with_backoff(&client, "http://test.com", &headers, request, 3).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.status().is_success());

        // Verify we made 3 calls (2 failures + 1 success)
        let call_count = *client.call_count.lock().unwrap();
        assert_eq!(call_count, 3);
    }

    #[tokio::test]
    async fn test_request_with_backoff_exceeds_max_retries() {
        let client = MockRateLimitClient::new(5); // Always fail
        let headers = HeaderMap::new();
        let request = json!({"test": "data"});

        let result = request_with_backoff(&client, "http://test.com", &headers, request, 2).await;

        assert!(result.is_err());

        // Verify we made max_retries + 1 calls (3 total: initial + 2 retries)
        let call_count = *client.call_count.lock().unwrap();
        assert_eq!(call_count, 3);
    }

    #[tokio::test]
    async fn test_calculate_backoff_delay_with_retry_after() {
        let json = json!({"error": "Rate limit exceeded"});
        let bytes = bytes::Bytes::from(json.to_string());

        let http_response = http::Response::builder()
            .status(429)
            .header("retry-after", "5")
            .body(bytes)
            .unwrap();

        let response = Response::from(http_response);
        let delay = calculate_backoff_delay(0, &response);

        assert_eq!(delay, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_calculate_backoff_delay_exponential() {
        let json = json!({"error": "Rate limit exceeded"});
        let bytes = bytes::Bytes::from(json.to_string());

        let http_response = http::Response::builder().status(429).body(bytes).unwrap();

        let response = Response::from(http_response);
        let delay = calculate_backoff_delay(1, &response);

        // Should be between 2000ms and 4000ms (base 2000ms + jitter)
        assert!(delay >= Duration::from_millis(2000));
        assert!(delay <= Duration::from_millis(4000));
    }
}
