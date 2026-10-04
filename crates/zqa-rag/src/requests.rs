//! Common utility functions for the crate.

use std::time::Duration;

use http::{HeaderMap, StatusCode};
use reqwest::Response;
use serde::Serialize;

use crate::http_client::HttpClient;
use crate::llm::errors::LLMError;
use crate::logging::preview;

/// Calculate the backoff delay given an attempt number and a retryable response. If the response
/// has a "Retry-After" header in seconds, that is used. Otherwise, it uses exponential backoff with
/// a base delay of 1000 milliseconds.
fn calculate_backoff_delay(attempt: usize, response: &Response) -> Duration {
    if let Some(retry_after) = response.headers().get("retry-after") {
        if let Ok(wait_time_str) = retry_after.to_str() {
            if let Ok(wait_time) = wait_time_str.parse::<u64>() {
                return Duration::from_secs(wait_time);
            }

            log::warn!("Retry-After value {wait_time_str} could not be parsed as a u64");
        } else {
            log::warn!("Retry-After value {retry_after:?} could not be converted to a string");
        }
    }

    exponential_backoff_delay(attempt)
}

/// The largest exponent used for exponential backoff. With a base delay of one second, this caps
/// the delay before jitter at 64 seconds, so a large configured `max_retries` cannot lead to
/// absurdly long sleeps.
const MAX_BACKOFF_EXPONENT: u32 = 6;

/// Compute an exponential backoff delay, with jitter, for a retry attempt.
///
/// # Arguments
///
/// * `attempt` - The zero-based retry attempt.
///
/// # Returns
///
/// A delay of `2^attempt` seconds plus up to 100% jitter, with the exponent capped at
/// `MAX_BACKOFF_EXPONENT`.
pub(crate) fn exponential_backoff_delay(attempt: usize) -> Duration {
    let exponent = u32::try_from(attempt)
        .unwrap_or(u32::MAX)
        .min(MAX_BACKOFF_EXPONENT);
    let base_delay = f64::from(1000_u32 << exponent);

    // Adding a jitter helps mitigate the thundering herd problem
    let jitter = base_delay * rand::random::<f64>();

    #[allow(clippy::cast_sign_loss)]
    Duration::from_millis((base_delay + jitter).round() as u64)
}

/// Perform a request with exponential backoff. This allows for retries without overwhelming the
/// server with too many requests. Rate-limited (429) responses and server errors (5xx, e.g., a 502
/// from a gateway or Anthropic's 529 "overloaded") are retried up to `max_retries` times, with a
/// delay of `2^attempt * base_delay` milliseconds (the exponent is capped at
/// `MAX_BACKOFF_EXPONENT`), where `base_delay` is 1000 milliseconds by default. If the response has
/// a "Retry-After" header, that is respected instead.
///
/// # Errors
///
/// * `LLMError::TimeoutError` if there was a timeout.
/// * `LLMError::CredentialError` if we receive an HTTP 401 Unauthorized or HTTP 403 Forbidden.
/// * `LLMError::HttpStatusError` for other 4xx and 5xx responses.
/// * `LLMError::NetworkError` if there was a network connectivity issue.
/// * `LLMError::GenericError` for all other errors.
pub(crate) async fn request_with_backoff<T: HttpClient>(
    client: &T,
    url: &str,
    headers: &HeaderMap,
    request: &(impl Serialize + Sync + Send),
    max_retries: usize,
) -> Result<Response, LLMError> {
    let mut attempt = 0;

    loop {
        log::debug!(
            "Provider request: attempt {} of {}",
            attempt + 1,
            max_retries + 1
        );
        let response = client.post_json(url, headers.clone(), &request).await?;
        let status = response.status();
        log::debug!(
            "Provider request: attempt {} returned {status}",
            attempt + 1
        );

        if response.status().is_success() {
            return Ok(response);
        }

        let is_retryable = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
        if is_retryable && attempt < max_retries {
            let delay = calculate_backoff_delay(attempt, &response);
            log::debug!(
                "Got {status} on attempt {}; retrying after {delay:.2?}",
                attempt + 1
            );
            let _ = tokio::time::sleep(delay).await;
            attempt += 1;
            continue;
        }

        let body = response.text().await?;
        log::debug!(
            "Provider request failed: status={status}, attempts={}, retries_exhausted={}, body={}",
            attempt + 1,
            is_retryable && attempt == max_retries,
            preview(&body)
        );

        return Err(LLMError::HttpStatusError(body));
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use http::HeaderMap;
    use reqwest::Response;
    use serde::Serialize;
    use serde_json::json;
    use zqa_macros::{test_eq, test_ok};

    use crate::http_client::HttpClient;
    use crate::requests::{calculate_backoff_delay, request_with_backoff};

    struct MockRateLimitClient {
        call_count: Arc<Mutex<usize>>,
        max_failures: usize,
        failure_status: u16,
    }

    impl MockRateLimitClient {
        fn new(max_failures: usize) -> Self {
            Self {
                call_count: Arc::new(Mutex::new(0)),
                max_failures,
                failure_status: 429,
            }
        }

        fn with_failure_status(mut self, failure_status: u16) -> Self {
            self.failure_status = failure_status;
            self
        }
    }

    impl HttpClient for MockRateLimitClient {
        fn post_json<'a, T: Serialize + Send + Sync>(
            &'a self,
            _url: &'a str,
            _headers: HeaderMap,
            _body: &'a T,
        ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>
        {
            Box::pin(async move {
                let mut count = { self.call_count.lock().unwrap() };
                *count += 1;

                if *count <= self.max_failures {
                    // Return 429 response with retry-after header
                    let json = json!({"error": "Rate limit exceeded"});
                    let bytes = bytes::Bytes::from(json.to_string());

                    let http_response = http::Response::builder()
                        .status(self.failure_status)
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
            })
        }

        fn get_json<'a>(
            &'a self,
            url: &'a str,
            headers: HeaderMap,
        ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>
        {
            self.post_json(url, headers, &())
        }

        fn post_form<'a>(
            &'a self,
            url: &'a str,
            headers: HeaderMap,
            _form_data: reqwest::multipart::Form,
        ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + '_>>
        {
            self.post_json(url, headers, &())
        }

        fn post_empty<'a>(
            &'a self,
            url: &'a str,
            headers: HeaderMap,
        ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>
        {
            self.post_json(url, headers, &())
        }
    }

    #[tokio::test]
    async fn test_request_with_backoff_handles_429() {
        let headers = HeaderMap::new();
        let request = json!({"test": "data"});

        // Rate limits and server errors (e.g., a 502 from a gateway) are both retried.
        for status in [429, 502] {
            let client = MockRateLimitClient::new(2).with_failure_status(status); // Fail twice, then succeed

            let result =
                request_with_backoff(&client, "http://test.com", &headers, &request, 3).await;

            test_ok!(result);
            let response = result.unwrap();
            assert!(response.status().is_success());

            // Verify we made 3 calls (2 failures + 1 success)
            let call_count = *client.call_count.lock().unwrap();
            test_eq!(call_count, 3);
        }
    }

    #[tokio::test]
    async fn test_request_with_backoff_exceeds_max_retries() {
        let client = MockRateLimitClient::new(5); // Always fail
        let headers = HeaderMap::new();
        let request = json!({"test": "data"});

        let result = request_with_backoff(&client, "http://test.com", &headers, &request, 2).await;

        assert!(result.is_err());

        // Verify we made max_retries + 1 calls (3 total: initial + 2 retries)
        let call_count = *client.call_count.lock().unwrap();
        test_eq!(call_count, 3);
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

        test_eq!(delay, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_calculate_backoff_delay_exponential() {
        let json = json!({"error": "Rate limit exceeded"});
        let bytes = bytes::Bytes::from(json.to_string());

        let http_response = http::Response::builder().status(429).body(bytes).unwrap();

        let response = Response::from(http_response);
        let delay = calculate_backoff_delay(1, &response);

        // Should be between 2000ms and 4000ms (base 2000ms + jitter)
        assert!(delay >= Duration::from_secs(2));
        assert!(delay <= Duration::from_secs(4));

        // Large attempt numbers (from a large configured `max_retries`) are capped at 64s + jitter
        let delay = calculate_backoff_delay(100, &response);
        assert!(delay >= Duration::from_secs(64));
        assert!(delay <= Duration::from_secs(128));
    }
}
