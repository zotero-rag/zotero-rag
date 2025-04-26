use http;
use reqwest::header::HeaderMap;

/// A trait that represents an HTTP client for making requests to LLM providers.
/// This abstraction enables real HTTP requests to API endpoints while also
/// supporting mock implementations for testing.
pub trait HttpClient: Send + Sync {
    #[allow(async_fn_in_trait)]
    async fn post_json<T: serde::Serialize + Send + Sync>(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &T,
    ) -> Result<reqwest::Response, reqwest::Error>;
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
    async fn post_json<T: serde::Serialize + Send + Sync>(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &T,
    ) -> Result<reqwest::Response, reqwest::Error> {
        self.client
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
    }
}

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
    async fn post_json<U: serde::Serialize + Send + Sync>(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &U,
    ) -> Result<reqwest::Response, reqwest::Error> {
        // Serialize the response to JSON and create a reqwest::Response
        let json = serde_json::to_string(&self.response).unwrap();
        let bytes = bytes::Bytes::from(json);
        
        // Create a builder and set the body
        let builder = http::Response::builder()
            .status(200)
            .header("content-type", "application/json");
            
        let http_response = builder.body(bytes).unwrap();
        
        Ok(reqwest::Response::from(http_response))
    }
}
