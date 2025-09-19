use http;
use reqwest::header::HeaderMap;
use std::{future::Future, pin::Pin};

/// A trait that represents an HTTP client for making requests to LLM providers.
/// This abstraction enables real HTTP requests to API endpoints while also
/// supporting mock implementations for testing.
pub trait HttpClient: Send + Sync {
    fn post_json<'a, T: serde::Serialize + Send + Sync>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        body: &'a T,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>;
}

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

#[derive(Debug)]
pub struct MockHttpClient<T: Send + Sync + Clone> {
    pub response: T,
}

impl<T: serde::Serialize + Send + Sync + Clone> MockHttpClient<T> {
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
