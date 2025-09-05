use crate::constants::{COHERE_EMBEDDING_DIM, COHERE_EMBEDDING_MODEL};
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    env, fs,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use reqwest::header::HeaderMap;

use crate::llm::errors::LLMError;
use crate::llm::http_client::{HttpClient, ReqwestClient};

#[derive(Debug, Clone)]
pub struct CohereClient<T: HttpClient = ReqwestClient> {
    pub client: T,
}

impl<T: HttpClient + Default> Default for CohereClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CohereClient<T>
where
    T: HttpClient + Default,
{
    pub fn new() -> Self {
        Self {
            client: T::default(),
        }
    }

    /// Internal method to compute embeddings that can be reused by both source and query embedding functions
    async fn compute_embeddings_async(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        let source_array = arrow_array::cast::as_string_array(&source);
        let texts: Vec<Option<String>> = source_array
            .iter()
            .map(|s| s.map(|s| s.to_owned()))
            .collect();

        log::info!("Processing {} input texts.", texts.len());
        let bar = ProgressBar::new(texts.len().try_into().unwrap());

        let api_key = env::var("COHERE_API_KEY")?;

        let mut all_embeddings: Vec<Vec<f32>> = Vec::new();

        // For a non-trial API key, Cohere's RPM is 2000 and the Embed model has a 128k context
        // window. It does not appear that there is a TPM limit. In theory, we could therefore send
        // 2000 requests spread over 1 minute (so 30 requests/second), each with one input.
        const BATCH_SIZE: usize = 30;
        const WAIT_AFTER_REQUEST_S: u64 = 1;

        // Gather failed texts
        let mut fail_count = 0;
        let mut total_masked = 0;
        let mut failed_texts: Vec<String> = Vec::new();

        let chunks = texts.chunks(BATCH_SIZE);
        let num_chunks = chunks.len();

        for (i, batch) in chunks.enumerate() {
            // Build mask for non-empty strings
            let mask: Vec<bool> = batch
                .iter()
                .map(|opt| opt.as_ref().is_some_and(|s| !s.trim().is_empty()))
                .collect();

            // Extract non-empty current texts
            let cur_texts: Vec<String> = batch
                .iter()
                .filter_map(|opt| opt.clone().filter(|s| !s.trim().is_empty()))
                .collect();

            // If none are real, just push zeros for whole batch
            if cur_texts.is_empty() {
                all_embeddings.extend(std::iter::repeat_n(
                    vec![0.0; COHERE_EMBEDDING_DIM as usize],
                    batch.len(),
                ));
            } else {
                let request = CohereEmbedRequest::from_texts(cur_texts);
                let mut headers = HeaderMap::new();
                headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
                headers.insert("Content-Type", "application/json".parse()?);
                headers.insert("Accept", "application/json".parse()?);

                let start_time = Instant::now();
                let response = self
                    .client
                    .post_json("https://api.cohere.com/v2/embed", headers, &request)
                    .await?;

                let body = response.text().await?;
                log::debug!("Cohere embedding request took {:.1?}", start_time.elapsed());

                let cohere_response: CohereAIResponse = serde_json::from_str(&body)?;

                match cohere_response {
                    CohereAIResponse::Success(success) => {
                        let mut it = success.embeddings.float.into_iter();

                        // 4. Weave the real embeddings back into the right spots, zero‐padding empties
                        let mut batch_embs = Vec::with_capacity(batch.len());
                        for &is_real in &mask {
                            if is_real && let Some(embedding) = it.next() {
                                batch_embs.push(embedding);
                            } else {
                                batch_embs.push(vec![0.0_f32; COHERE_EMBEDDING_DIM as usize]);
                            }
                        }
                        all_embeddings.extend(batch_embs);

                        total_masked += mask.iter().filter(|mask| !**mask).count();
                    }
                    CohereAIResponse::Error(err) => {
                        eprintln!("Got a 400 response from Cohere AI: {}\n", err.message);
                        eprintln!("We tried sending the request: {request:#?}\n");

                        fail_count += batch.len();
                        failed_texts.extend(batch.iter().filter_map(|text| text.as_ref()).cloned());

                        let zeros: Vec<Vec<f32>> = std::iter::repeat_n(
                            Vec::from([0.0; COHERE_EMBEDDING_DIM as usize]),
                            batch.len(),
                        )
                        .collect();
                        all_embeddings.extend(zeros);
                    }
                }
            }

            bar.inc(BATCH_SIZE as u64);

            if i < num_chunks - 1 {
                tokio::time::sleep(Duration::from_secs(WAIT_AFTER_REQUEST_S)).await;
            }
        }

        if fail_count > 0 {
            let failed = FailedTexts {
                embedding_provider: String::from("cohere"),
                texts: failed_texts,
            };
            let encoded = serde_json::to_string_pretty(&failed)?;

            if let Err(e) = fs::write("failed.json", encoded) {
                eprintln!("We could not write out the failed texts to 'failed.json': {e}");
            } else {
                println!(
                    "We have written the failed texts to 'failed.json'. Consider using /repair to fix this."
                );
            }
        }

        log::info!(
            "Processing finished. Statistics:\n{fail_count} items failed.\n{total_masked} items were empty."
        );

        // Convert to Arrow FixedSizeListArray
        let flattened: Vec<f32> = all_embeddings.iter().flatten().copied().collect();
        let values = arrow_array::Float32Array::from(flattened);

        let list_array = arrow_array::FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            COHERE_EMBEDDING_DIM as i32,
            Arc::new(values),
            None,
        )
        .map_err(|e| {
            LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {e}"))
        })?;

        Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
    }

    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.compute_embeddings_async(source))
        })
    }
}

/// A request to the Cohere embeddings API.
#[derive(Serialize, Debug)]
struct CohereEmbedRequest {
    texts: Vec<String>,
    model: String,
    input_type: String,
    output_dimension: u32,
    // Requesting float vectors explicitly for newer APIs; ignored by older
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_types: Option<Vec<String>>,
}

impl CohereEmbedRequest {
    pub fn from_texts(texts: Vec<String>) -> Self {
        Self {
            texts,
            model: COHERE_EMBEDDING_MODEL.to_string(),
            input_type: "search_document".into(),
            output_dimension: COHERE_EMBEDDING_DIM,
            embedding_types: Some(vec!["float".into()]),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct CohereAIEmbeddings {
    float: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub struct CohereAISuccess {
    embeddings: CohereAIEmbeddings,
}

#[derive(Serialize, Deserialize)]
pub struct CohereAIError {
    message: String,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum CohereAIResponse {
    Success(CohereAISuccess),
    Error(CohereAIError),
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FailedTexts {
    pub embedding_provider: String,
    pub texts: Vec<String>,
}

impl<T: HttpClient + Default + std::fmt::Debug> EmbeddingFunction for CohereClient<T> {
    fn name(&self) -> &str {
        "Cohere"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            COHERE_EMBEDDING_DIM as i32,
        )))
    }

    fn compute_source_embeddings(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        match self.compute_embeddings_internal(source) {
            Ok(result) => Ok(result),
            Err(e) => Err(lancedb::Error::Other {
                message: e.to_string(),
                source: None,
            }),
        }
    }

    fn compute_query_embeddings(
        &self,
        input: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, lancedb::Error> {
        match self.compute_embeddings_internal(input) {
            Ok(result) => Ok(result),
            Err(e) => Err(lancedb::Error::Other {
                message: e.to_string(),
                source: None,
            }),
        }
    }
}
