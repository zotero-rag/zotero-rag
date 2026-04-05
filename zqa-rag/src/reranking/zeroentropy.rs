use std::{env, pin::Pin, time::Instant};

use http::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::{
    constants::DEFAULT_ZEROENTROPY_RERANK_MODEL, embedding::zeroentropy::ZeroEntropyClient,
    http_client::HttpClient, llm::errors::LLMError, reranking::common::Rerank,
};

// Like with the embedding case, we just use the default `latency` option because it's the best
// default.
#[derive(Serialize)]
struct ZeroEntropyRerankRequest<'a> {
    model: String,
    query: String,
    documents: &'a [&'a str],
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
}

#[derive(Deserialize)]
struct ZeroEntropyRerankResult {
    index: usize,
    #[allow(dead_code)]
    relevance_score: f32,
}

#[derive(Deserialize)]
struct ZeroEntropyRerankResponse {
    results: Vec<ZeroEntropyRerankResult>,
}

impl<T: HttpClient> Rerank for ZeroEntropyClient<T> {
    fn rerank<'a>(
        &'a self,
        items: &'a [&str],
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<usize>, LLMError>> + Send + 'a>> {
        Box::pin(async move {
            const RERANK_API_URL: &str = "https://api.zeroentropy.dev/v1/models/rerank";

            let (api_key, reranker_model) = if let Some(ref config) = self.config {
                (config.api_key.clone(), config.reranker.clone())
            } else {
                (
                    env::var("ZEROENTROPY_API_KEY")?,
                    env::var("ZEROENTROPY_RERANK_MODEL")
                        .unwrap_or_else(|_| DEFAULT_ZEROENTROPY_RERANK_MODEL.into()),
                )
            };

            let request = ZeroEntropyRerankRequest {
                model: reranker_model,
                query: query.into(),
                documents: items,
                top_n: None,
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
            log::debug!(
                "ZeroEntropy rerank request took {:.1?}",
                start_time.elapsed()
            );

            let ze_response: ZeroEntropyRerankResponse =
                serde_json::from_str(&body).map_err(|e| {
                    log::warn!("Error deserializing ZeroEntropy reranker response: {e}");
                    LLMError::DeserializationError(e.to_string())
                })?;

            Ok(ze_response.results.iter().map(|r| r.index).collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ZeroEntropyClient;
    use crate::http_client::ReqwestClient;
    use crate::reranking::common::Rerank;
    use dotenv::dotenv;
    use zqa_macros::{test_eq, test_ok};

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_rerank() {
        dotenv().ok();

        let items = ["Hello, World!", "A second string", "A third string"];
        let query = "A string";

        let client = ZeroEntropyClient::<ReqwestClient>::default();
        let reranked = client.rerank(&items, query).await;

        if reranked.is_err() {
            println!("ZeroEntropy reranker error: {:?}", reranked.as_ref().err());
        }

        test_ok!(reranked);

        let reranked = reranked.unwrap();
        test_eq!(reranked.len(), items.len());
    }
}
