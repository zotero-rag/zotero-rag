use indicatif::ProgressBar;
use std::{borrow::Cow, env, fs, sync::Arc, time::Duration};

use arrow_schema::{DataType, Field};
use lancedb::embeddings::EmbeddingFunction;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use super::errors::LLMError;
use crate::llm::http_client::{HttpClient, ReqwestClient};

const VOYAGE_EMBEDDING_DIM: u32 = 2048;
const VOYAGE_EMBEDDING_MODEL: &str = "voyage-3-large";

/// A client for Voyage AI's embedding API.
#[derive(Debug, Clone)]
pub struct VoyageAIClient<T: HttpClient = ReqwestClient> {
    pub client: T,
}

impl<T: HttpClient + Default> Default for VoyageAIClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> VoyageAIClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new VoyageAIClient instance
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

        println!("Processing {} input texts.", texts.len());
        let bar = ProgressBar::new(texts.len().try_into().unwrap());

        let api_key = env::var("VOYAGE_AI_API_KEY")?;

        let mut all_embeddings = Vec::new();

        // Wait for two seconds after each batch to avoid RPM and TPM throttling. At the base
        // tier, Voyage AI has a 3M TPM and a 2K RPM. However, although the context of their
        // models is 32k, we can't just sent 3M / 32k ~= 93 requests at a time and then wait a
        // minute, because there is also a 120k token per request limit. This actually means we
        // can only send floor(120k / 32k) = 3 requests at a time. Now our effective requests
        // per minute is 3M (tokens / min) / 96k (tokens / request) = 31.25 RPM. Rounding down,
        // we can send 30 RPM, so we wait 2s between requests.
        const BATCH_SIZE: usize = 3;
        const WAIT_AFTER_REQUEST_S: u64 = 2;

        // Gather failed texts
        let mut fail_count = 0;
        let mut total_masked = 0;
        let mut failed_texts: Vec<String> = Vec::new();

        for batch in texts.chunks(BATCH_SIZE) {
            // For every batch, we need to handle the case of empty/whitespace strings, since Voyage AI
            // does not like handling them.
            // 1. Build a mask of "real" vs "empty" slots
            let mask: Vec<bool> = batch
                .iter()
                .map(|opt| opt.as_ref().is_some_and(|s| !s.trim().is_empty()))
                .collect();

            // 2. Extract only the non-empty strings to send
            let cur_texts: Vec<String> = batch
                .iter()
                .filter_map(|opt| opt.clone().filter(|s| !s.trim().is_empty()))
                .collect();

            // 3. If none are real, just push zeros for whole batch
            if cur_texts.is_empty() {
                all_embeddings.extend(
                    std::iter::repeat_n(vec![0.0; VOYAGE_EMBEDDING_DIM as usize], batch.len()),
                );
            } else {
                let request = VoyageAIRequest::from_texts(cur_texts);

                let mut headers = HeaderMap::new();
                headers.insert("Authorization", format!("Bearer {api_key}").parse()?);
                headers.insert("Content-Type", "application/json".parse()?);

                let response = self
                    .client
                    .post_json("https://api.voyageai.com/v1/embeddings", headers, &request)
                    .await?;

                let body = response.text().await?;
                let voyage_response: VoyageAIResponse = serde_json::from_str(&body)?;

                match voyage_response {
                    VoyageAIResponse::Success(success) => {
                        let mut it = success.data.into_iter().map(|d| d.embedding);

                        // 4. Weave the real embeddings back into the right spots, zero‐padding empties
                        let mut batch_embs = Vec::with_capacity(batch.len());
                        for &is_real in &mask {
                            // TODO: Refactor to use `let` in the condition when this is not an
                            // unstable feature anymore.
                            if is_real {
                                if let Some(embedding) = it.next() {
                                    batch_embs.push(embedding);
                                } else {
                                    batch_embs.push(vec![0.0_f32; VOYAGE_EMBEDDING_DIM as usize]);
                                }
                            } else {
                                batch_embs.push(vec![0.0_f32; VOYAGE_EMBEDDING_DIM as usize]);
                            }
                        }
                        all_embeddings.extend(batch_embs);

                        total_masked += mask.iter().filter(|mask| !**mask).count();
                    }
                    VoyageAIResponse::Error(err) => {
                        eprintln!("Got a 400 response from Voyage AI: {}\n", err.detail);
                        eprintln!("We tried sending the request: {request:#?}\n");

                        fail_count += batch.len();
                        failed_texts.extend(batch.iter().filter_map(|text| text.as_ref()).cloned());

                        let zeros: Vec<Vec<f32>> = std::iter::repeat_n(
                            Vec::from([0.0; VOYAGE_EMBEDDING_DIM as usize]),
                            batch.len(),
                        )
                        .collect();
                        all_embeddings.extend(zeros);
                    }
                }
            }

            bar.inc(BATCH_SIZE as u64);
            tokio::time::sleep(Duration::from_secs(WAIT_AFTER_REQUEST_S)).await;
        }

        println!("Processing finished. Statistics:\n{fail_count} items failed.\n{total_masked} items were empty.");

        if fail_count > 0 {
            let failed = FailedTexts {
                embedding_provider: String::from("voyageai"),
                texts: failed_texts,
            };
            let encoded = serde_json::to_string_pretty(&failed)?;

            if let Err(e) = fs::write("failed.json", encoded) {
                eprintln!("We could not write out the failed texts to 'failed.json': {e}");
            } else {
                println!("We have written the failed texts to 'failed.json'. Consider using /repair to fix this.");
            }
        }

        let n_embeddings = all_embeddings.len();
        let emb_shape = all_embeddings.first()
            .ok_or(LLMError::GenericLLMError(String::from(
                "Could not compute embedding shape--this should not happen.",
            )))?
            .len();
        println!("Embedding dim: ({n_embeddings}, {emb_shape})");

        // Convert to Arrow FixedSizeListArray
        let flattened: Vec<f32> = all_embeddings.iter().flatten().copied().collect();
        let values = arrow_array::Float32Array::from(flattened);

        let list_array = arrow_array::FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            VOYAGE_EMBEDDING_DIM as i32,
            Arc::new(values),
            None,
        )
        .map_err(|e| {
            LLMError::GenericLLMError(format!("Failed to create FixedSizeListArray: {e}"))
        })?;

        Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
    }

    /// Internal method to compute embeddings that works with LLMError
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            LLMError::GenericLLMError(format!("Could not create tokio runtime: {e}"))
        })?;

        rt.block_on(self.compute_embeddings_async(source))
    }
}

/// A request to Voyage AI's embedding endpoint. This struct should not be created directly.
/// Instead, use `from_texts` for good defaults.
#[derive(Serialize, Debug, Deserialize)]
struct VoyageAIRequest {
    pub input: Vec<String>,
    pub model: String,
    input_type: Option<String>,
    truncation: bool,
    output_dimension: u32,
    output_dtype: String,
}

impl VoyageAIRequest {
    pub fn from_texts(texts: Vec<String>) -> Self {
        Self {
            input: texts,
            model: VOYAGE_EMBEDDING_MODEL.to_string(),
            input_type: None, // Directly convert to vector
            truncation: true,
            output_dimension: VOYAGE_EMBEDDING_DIM, // Matryoshka embeddings
            output_dtype: "float".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct VoyageAIEmbedding {
    object: String,
    embedding: Vec<f32>,
    index: u32,
}

#[derive(Serialize, Deserialize)]
struct VoyageAIUsage {
    total_tokens: u32,
}

#[derive(Serialize, Deserialize)]
struct VoyageAISuccess {
    object: String,
    data: Vec<VoyageAIEmbedding>,
    model: String,
    usage: VoyageAIUsage,
}

#[derive(Serialize, Deserialize)]
struct VoyageAIError {
    pub detail: String,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum VoyageAIResponse {
    Success(VoyageAISuccess),
    Error(VoyageAIError),
}

#[derive(Serialize, Deserialize)]
pub struct FailedTexts {
    pub embedding_provider: String,
    pub texts: Vec<String>,
}

/// Implements the LanceDB EmbeddingFunction trait for VoyageAIClient. Since VoyageAI has the
/// highest token limit for their embedding model (32k instead of OpenAI's 8k), we prefer this
/// instead.
impl<T: HttpClient + Default + std::fmt::Debug> EmbeddingFunction for VoyageAIClient<T> {
    fn name(&self) -> &str {
        "Voyage AI"
    }

    fn source_type(&self) -> Result<Cow<DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            VOYAGE_EMBEDDING_DIM as i32,
        )))
    }

    /// The most basic tier of Voyage AI has a TPM of 3M, and an RPM of 2000. Since we have
    /// truncation enabled, we have 3M / 32k ~= 93, so we send 90 requests at a time, before
    /// waiting for one minute.
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

#[cfg(test)]
mod tests {
    use crate::llm::http_client::ReqwestClient;
    use crate::llm::voyage::{VoyageAIClient, VOYAGE_EMBEDDING_DIM};
    use arrow_array::Array;
    use dotenv::dotenv;
    use std::sync::Arc;

    #[test]
    fn test_compute_embeddings() {
        dotenv().ok();

        let array = arrow_array::StringArray::from(vec![
            "Hello, World!",
            "A second string",
            "A third string",
            "A fourth string",
            "A fifth string",
            "A sixth string",
        ]);

        let client = VoyageAIClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_internal(Arc::new(array));

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Voyage AI embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), VOYAGE_EMBEDDING_DIM as i32);
    }
}
