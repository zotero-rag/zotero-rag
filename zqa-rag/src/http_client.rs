//! This module provides the `HttpClient` trait and its implementations. This trait is used to
//! abstract away the HTTP client used for making requests to LLM providers. The `ReqwestClient`
//! implementation is provided as a default implementation, but it can be easily replaced with a
//! `MockHttpClient` implementation for testing.

#[cfg(test)]
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use std::{future::Future, pin::Pin};

use reqwest::{header::HeaderMap, multipart::Form};

/// A trait that represents an HTTP client for making requests to LLM providers.
/// This abstraction enables real HTTP requests to API endpoints while also
/// supporting mock implementations for testing.
pub trait HttpClient: Send + Sync {
    /// Send a GET request to the specified URL with the given headers.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to send the request to.
    /// * `headers` - The headers to include in the request.
    ///
    /// # Returns
    ///
    /// A `Future` that resolves to a `Result` containing the response from the server.
    fn get_json<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>>;

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

    /// Submit a request with a `multipart/form-data` body to `url` with the specified `headers`.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to send the request to.
    /// * `headers` - The headers to include in the request.
    /// * `form_data` - The form data of the request.
    fn post_form<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        form_data: Form,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + '_>>;
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
        let serialized_body = serde_json::to_string_pretty(&body);
        log::debug!(
            "Sending request to {url} with body: {:#?}",
            &serialized_body
        );

        Box::pin(async move {
            self.client
                .post(url)
                .json(&body)
                .headers(headers)
                .send()
                .await
        })
    }

    fn post_form<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        form_data: Form,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + '_>> {
        Box::pin(async move {
            self.client
                .post(url)
                .headers(headers)
                .multipart(form_data)
                .send()
                .await
        })
    }

    fn get_json<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>> {
        Box::pin(async move { self.client.get(url).headers(headers).send().await })
    }
}

/// A mock implementation of the `HttpClient` trait for testing purposes.
#[derive(Debug, Clone)]
#[cfg(test)]
pub(crate) struct MockHttpClient<T: Send + Sync + Clone> {
    /// The response to return when the mock client is called.
    pub response: T,
}

#[cfg(test)]
impl<T: serde::Serialize + Send + Sync + Clone> MockHttpClient<T> {
    /// Create a new `MockHttpClient` with the given response.
    pub(crate) fn new(response: T) -> Self {
        Self { response }
    }
}

#[cfg(test)]
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

/// A mock HTTP client that returns responses from a pre-loaded queue, one per call. Useful for
/// testing multi-turn interactions (e.g. a tool-call response followed by a final text response).
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct SequentialMockHttpClient {
    /// The queue of JSON-serialized responses to return, in order.
    responses: Arc<Mutex<VecDeque<String>>>,
}

#[cfg(test)]
impl SequentialMockHttpClient {
    /// Create a new `SequentialMockHttpClient` from an iterator of serializable responses.
    ///
    /// # Panics
    ///
    /// * If `serde_json::to_string` returns an `Err`.
    pub(crate) fn new<T: serde::Serialize>(responses: impl IntoIterator<Item = T>) -> Self {
        let queue = responses
            .into_iter()
            .map(|r| serde_json::to_string(&r).unwrap())
            .collect();
        Self {
            responses: Arc::new(Mutex::new(queue)),
        }
    }
}

#[cfg(test)]
impl HttpClient for SequentialMockHttpClient {
    fn post_json<'a, U: serde::Serialize + Send + Sync>(
        &'a self,
        _url: &'a str,
        _headers: HeaderMap,
        _body: &'a U,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>> {
        let responses = Arc::clone(&self.responses);
        Box::pin(async move {
            let json = responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("SequentialMockHttpClient: no more responses in queue");
            let bytes = bytes::Bytes::from(json);
            let http_response = http::Response::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(bytes)
                .unwrap();
            Ok(reqwest::Response::from(http_response))
        })
    }

    fn post_form<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        _form_data: Form,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + '_>> {
        self.post_json(url, headers, &None::<usize>)
    }

    fn get_json<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>> {
        self.post_json(url, headers, &None::<usize>)
    }
}

#[cfg(test)]
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

    fn post_form<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        _form_data: Form,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + '_>> {
        // We can use dummy data as a placeholder for the body
        self.post_json(url, headers, &None::<usize>)
    }

    fn get_json<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
    ) -> Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send + 'a>> {
        // We can use dummy data as a placeholder for the body
        self.post_json(url, headers, &None::<usize>)
    }
}
