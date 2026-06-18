use http::HeaderMap;

use crate::{
    clients::test::TestClient,
    http_client::HttpClient,
    llm::base::{ApiClient, ChatRequest, CompletionApiResponse, ContentType},
};

impl ApiClient for TestClient {
    async fn send_message(
        &self,
        _: &ChatRequest<'_>,
    ) -> Result<CompletionApiResponse, super::errors::LLMError> {
        let result = self
            .client
            .post_json("", HeaderMap::new(), &None::<usize>)
            .await;

        match result {
            Err(_) => unreachable!("`SequentialMockHttpClient` does not propagate errors."),
            Ok(res) => Ok(CompletionApiResponse {
                content: vec![ContentType::Text(res.text().await.unwrap())],
                input_tokens: 100,
                output_tokens: 100,
            }),
        }
    }
}
