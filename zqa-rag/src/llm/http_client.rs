//! This module provides the `HttpClient` trait and its implementations. This trait is used to
//! abstract away the HTTP client used for making requests to LLM providers. The `ReqwestClient`
//! implementation is provided as a default implementation, but it can be easily replaced with a
//! `MockHttpClient` implementation for testing.

use http;
use reqwest::header::HeaderMap;
use std::{future::Future, pin::Pin};

/// A trait that represents an HTTP client for making requests to LLM providers.
/// This abstraction enables real HTTP requests to API endpoints while also
/// supporting mock implementations for testing.
pub trait HttpClient: Send + Sync {
    /// Send a POST request to the specified URL with the given body and headers.
    ///
    /// # Arguments:
    ///
    /// * `url` - The URL to send the request to.
    /// * `headers` - The headers to include in the request.
    /// * `body` - The body of the request.
    ///
    /// # Returns
    ///
    /// A `Future` that resolves to a `Result` containing the response from the server.
    fn post_json<'a, T: serde::Serialize + Send + Sync>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        body: &'a T,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>;
}

/// A default implementation of the `HttpClient` trait using the `reqwest` crate.
#[derive(Debug, Clone)]
pub struct ReqwestClient {
    client: reqwest::Client,
}

impl Default for ReqwestClient {
    fn default() -> Self {
        ReqwestClient {
            client: reqwest::Client::new(),
        }
    }
}

impl HttpClient for ReqwestClient {
    fn post_json<'a, T: serde::Serialize + Send + Sync>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        body: &'a T,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>> {
        Box::pin(async move {
            self.client
                .post(url)
                .json(&body)
                .headers(headers)
                .send()
                .await
        })
    }
}

/// A mock implementation of the `HttpClient` trait for testing purposes.
#[derive(Debug)]
pub struct MockHttpClient<T: Send + Sync + Clone> {
    /// The response to return when the mock client is called.
    pub response: T,
}

impl<T: serde::Serialize + Send + Sync + Clone> MockHttpClient<T> {
    /// Create a new `MockHttpClient` with the given response.
    pub fn new(response: T) -> Self {
        Self { response }
    }
}

impl<T: serde::Serialize + Send + Sync + Clone> Default for MockHttpClient<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            response: T::default(),
        }
    }
}

impl<T: serde::Serialize + Send + Sync + Clone> HttpClient for MockHttpClient<T> {
    #[allow(unused_variables)]
    fn post_json<'a, U: serde::Serialize + Send + Sync>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        body: &'a U,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>> {
        let response = self.response.clone();
        Box::pin(async move {
            // Serialize the response to JSON and create a reqwest::Response
            let json = serde_json::to_string(&response).unwrap();
            let bytes = bytes::Bytes::from(json);

            // Create a builder and set the body
            let builder = http::Response::builder()
                .status(200)
                .header("content-type", "application/json");

            let http_response = builder.body(bytes).unwrap();

            Ok(reqwest::Response::from(http_response))
        })
    }
}
