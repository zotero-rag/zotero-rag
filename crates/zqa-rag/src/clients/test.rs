use crate::http_client::SequentialMockHttpClient;

#[derive(Debug, Clone)]
pub struct TestClient {
    pub(crate) client: SequentialMockHttpClient,
}

impl TestClient {
    #[must_use]
    pub fn new(responses: impl IntoIterator<Item = String>) -> Self {
        Self {
            client: SequentialMockHttpClient::from_bodies(responses),
        }
    }
}
