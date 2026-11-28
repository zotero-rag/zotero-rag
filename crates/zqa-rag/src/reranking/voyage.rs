use std::pin::Pin;
use std::time::Instant;

use http::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::embedding::voyage::VoyageAIClient;
use crate::http_client::HttpClient;
use crate::llm::errors::LLMError;
use crate::reranking::common::Rerank;

#[derive(Serialize, Debug)]
struct VoyageAIRerankRequest<'a> {
    query: String,
    documents: &'a [&'a str],
    model: String,
}

#[derive(Clone, Deserialize)]
struct VoyageAIRerankedDoc {
    index: usize,
}

#[derive(Deserialize)]
struct VoyageAIRerankSuccess {
    data: Vec<VoyageAIRerankedDoc>,
}

#[derive(Deserialize)]
struct VoyageAIError {
    detail: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum VoyageAIRerankResponse {
    Success(VoyageAIRerankSuccess),
    Error(VoyageAIError),
}

impl<T: HttpClient> Rerank for VoyageAIClient<T> {
    fn rerank<'a>(
        &'a self,
        items: &'a [&str],
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<usize>, LLMError>> + Send + 'a>> {
        Box::pin(async move {
            const RERANK_API_URL: &str = "https://api.voyageai.com/v1/rerank";

            let config = &self.config;
            let request = VoyageAIRerankRequest {
                model: config.reranker.clone(),
                query: query.into(),
                documents: items,
            };

            let mut headers = HeaderMap::new();
            headers.insert(
                "Authorization",
                format!("Bearer {}", config.api_key).parse()?,
            );
            headers.insert("Content-Type", "application/json".parse()?);
            headers.insert("Accept", "application/json".parse()?);

            let start_time = Instant::now();
            let response = self
                .client
                .post_json(RERANK_API_URL, headers, &request)
                .await?;

            let status = response.status();
            let body = response.text().await?;
            log::debug!("Voyage AI rerank request took {:.1?}", start_time.elapsed());
            if !status.is_success() {
                return Err(LLMError::from_status(status, body));
            }

            let voyage_response: VoyageAIRerankResponse =
                serde_json::from_str(&body).map_err(|e| {
                    log::warn!(
                        "Error deserializing Voyage AI reranker response: {e}. Response: {body}"
                    );
                    LLMError::DeserializationError(e.to_string())
                })?;

            match voyage_response {
                VoyageAIRerankResponse::Success(success) => {
                    let res = success
                        .data
                        .iter()
                        .map(|result| result.index)
                        .collect::<Vec<_>>();

                    Ok(res)
                }
                VoyageAIRerankResponse::Error(err) => Err(LLMError::HttpStatusError(err.detail)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use dotenv::dotenv;
    use zqa_macros::{test_eq, test_ok};

    use super::VoyageAIClient;
    use crate::config::VoyageAIConfig;
    use crate::http_client::{ReqwestClient, SequentialMockHttpClient};
    use crate::llm::errors::LLMError;
    use crate::reranking::common::Rerank;

    #[tokio::test]
    async fn test_rerank_classifies_error_statuses() {
        for (status, is_credential_error) in [(401, true), (403, true), (500, false)] {
            let client = VoyageAIClient {
                client: SequentialMockHttpClient::from_status_bodies([(
                    status,
                    String::from("request rejected"),
                )]),
                config: VoyageAIConfig {
                    api_key: "test-key".into(),
                    embedding_model: String::new(),
                    embedding_dims: 0,
                    reranker: "test-reranker".into(),
                    max_concurrent_requests: crate::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
                    max_retries: crate::constants::DEFAULT_MAX_RETRIES,
                },
            };

            let result = client.rerank(&["a document"], "a query").await;
            if is_credential_error {
                assert!(matches!(result, Err(LLMError::CredentialError(_))));
            } else {
                assert!(matches!(result, Err(LLMError::HttpStatusError(_))));
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_rerank() {
        dotenv().ok();

        let array = ["Hello, World!", "A second string", "A third string"];
        let query = "A string";

        let client =
            VoyageAIClient::<ReqwestClient>::with_config(crate::config::test_env::voyage());
        let reranked = client.rerank(&array, query).await;

        test_ok!(reranked);

        let reranked = reranked.unwrap();
        test_eq!(reranked.len(), array.len());
    }
}
