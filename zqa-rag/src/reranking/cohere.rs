use std::{env, pin::Pin, time::Instant};

use http::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::{
    constants::DEFAULT_COHERE_RERANK_MODEL, embedding::cohere::CohereClient,
    http_client::HttpClient, llm::errors::LLMError, reranking::common::Rerank,
};

#[derive(Serialize)]
struct CohereRerankRequest<'a> {
    model: String,
    query: String,
    top_n: Option<usize>,
    documents: &'a [&'a str],
}

#[derive(Clone, Deserialize)]
struct CohereRerankedDocument {
    index: usize,
}

#[derive(Deserialize)]
struct CohereRerankResponse {
    results: Vec<CohereRerankedDocument>,
}

impl<T: HttpClient> Rerank for CohereClient<T> {
    fn rerank<'a>(
        &'a self,
        items: &'a [&str],
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<usize>, LLMError>> + Send + 'a>> {
        Box::pin(async move {
            const RERANK_API_URL: &str = "https://api.cohere.com/v2/rerank";

            // Use config if available, otherwise fall back to env vars
            let (api_key, reranker_model) = if let Some(ref config) = self.config {
                (config.api_key.clone(), config.reranker.clone())
            } else {
                (
                    env::var("COHERE_API_KEY")?,
                    env::var("COHERE_RERANKER").unwrap_or(DEFAULT_COHERE_RERANK_MODEL.into()),
                )
            };

            let request = CohereRerankRequest {
                model: reranker_model,
                query: query.into(),
                top_n: None,
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
            log::debug!("Cohere rerank request took {:.1?}", start_time.elapsed());

            let cohere_response: CohereRerankResponse =
                serde_json::from_str(&body).map_err(|e| {
                    log::warn!("Error deserializing Cohere reranker response: {e}");
                    LLMError::DeserializationError(e.to_string())
                })?;

            let cohere_response = cohere_response.results;
            let res = cohere_response
                .iter()
                .map(|result| result.index)
                .collect::<Vec<_>>();

            Ok(res)
        })
    }
}

#[cfg(test)]
mod tests {
    use dotenv::dotenv;
    use zqa_macros::{test_eq, test_ok};

    use super::CohereClient;
    use crate::http_client::ReqwestClient;
    use crate::reranking::common::Rerank;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_rerank() {
        dotenv().ok();

        let array = ["Hello, World!", "A second string", "A third string"];
        let query = "A string";

        let client = CohereClient::<ReqwestClient>::default();
        let reranked = client.rerank(&array, query).await;

        // Debug the error if there is one
        if reranked.is_err() {
            println!("Cohere reranker error: {:?}", reranked.as_ref().err());
        }

        test_ok!(reranked);

        let reranked = reranked.unwrap();
        test_eq!(reranked.len(), array.len());
    }
}
