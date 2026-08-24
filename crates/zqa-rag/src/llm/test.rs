use http::HeaderMap;

use crate::clients::test::TestClient;
use crate::http_client::HttpClient;
use crate::llm::base::{ChatRequest, CompletionApiResponse, ContentType};
use crate::llm::errors::LLMError;
use crate::pricing::ModelUsage;

impl TestClient {
    /// Return the next queued mock response.
    ///
    /// # Arguments
    ///
    /// Takes one `&ChatRequest` argument, but it is ignored.
    ///
    /// # Returns
    ///
    /// Always returns `Ok` with the next element in the provided deque. Token counts
    /// returned are always 0.
    ///
    /// # Panics
    ///
    /// * If a lock could not be obtained on the client's underlying mutex
    /// * If the deque has been exhausted
    /// * If the body in the next element is not UTF-8 (though this is unlikely)
    pub(crate) async fn send_message(
        &self,
        _: &ChatRequest<'_>,
    ) -> Result<CompletionApiResponse, LLMError> {
        let result = self
            .client
            .post_json("", HeaderMap::new(), &None::<usize>)
            .await;

        let result = result.expect("mock http client should not propagate errors");
        Ok(CompletionApiResponse {
            content: vec![ContentType::Text(result.text().await.unwrap())],
            usage: ModelUsage {
                input_tokens: 0,
                input_cache_written: 0,
                input_cache_read: 0,
                output_tokens: 0,
                reasoning_tokens: 0,
            },
        })
    }
}
