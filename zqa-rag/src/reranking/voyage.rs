use std::env;
use std::pin::Pin;
use std::time::Instant;

use http::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::constants::DEFAULT_VOYAGE_RERANK_MODEL;
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

            // Use config if available, otherwise fall back to env vars
            let (api_key, reranker_model) = if let Some(ref config) = self.config {
                (config.api_key.clone(), config.reranker.clone())
            } else {
                (
                    env::var("VOYAGE_AI_API_KEY")?,
                    env::var("VOYAGE_AI_RERANK_MODEL")
                        .unwrap_or(DEFAULT_VOYAGE_RERANK_MODEL.into()),
                )
            };

            let request = VoyageAIRerankRequest {
                model: reranker_model,
                query: query.into(),
                documents: items,
            };

            let mut headers = HeaderMap::new();
            headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
            headers.insert("Content-Type", "application/json".parse()?);
            headers.insert("Accept", "application/json".parse()?);

            let start_time = Instant::now();
            let response = self
                .client
                .post_json(RERANK_API_URL, headers, &request)
                .await?;

            let body = response.text().await?;
            log::debug!("Voyage AI rerank request took {:.1?}", start_time.elapsed());

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
    use crate::http_client::ReqwestClient;
    use crate::reranking::common::Rerank;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_rerank() {
        dotenv().ok();

        let array = ["Hello, World!", "A second string", "A third string"];
        let query = "A string";

        let client = VoyageAIClient::<ReqwestClient>::default();
        let reranked = client.rerank(&array, query).await;

        test_ok!(reranked);

        let reranked = reranked.unwrap();
        test_eq!(reranked.len(), array.len());
    }
}
