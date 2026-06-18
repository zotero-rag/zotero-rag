use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use crate::http_client::SequentialMockHttpClient;

#[derive(Debug, Clone)]
pub struct TestClient {
    pub(crate) client: SequentialMockHttpClient,
}

impl TestClient {
    #[must_use]
    pub fn new(responses: &Arc<Mutex<VecDeque<String>>>) -> Self {
        let responses = Arc::clone(responses);
        let raw_responses = responses.lock().unwrap();

        Self {
            client: SequentialMockHttpClient::new(raw_responses.iter()),
        }
    }
}
