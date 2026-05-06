//! Provider-agnostic adapter for batch embedding APIs.
//!
//! To add support for a new batch embedding provider, add a new variant to [`BatchProviderAdapter`]
//! and implement the three high-level methods (`submit_texts`, `get_status`, `collect_embeddings`)
//! for it.

use serde::{Deserialize, Serialize};

use crate::{
    capabilities::{BatchAPIProvider, BatchJobState},
    embedding::{
        common::EmbeddingProviderConfig,
        voyage::{VoyageAIClient, VoyageAIFilesRequest, VoyageAIFilesRequestBody},
    },
    http_client::ReqwestClient,
    llm::errors::LLMError,
};

/// Metadata about a pending batch embedding job.
///
/// This is serialized to disk so the user can check status and collect results in a later CLI
/// session, without needing to keep the CLI running while the batch processes.
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchJobMetadata {
    /// The provider-assigned batch identifier.
    pub batch_id: String,
    /// The name of the embedding provider that owns this batch (e.g. `"voyageai"`).
    pub provider: String,
    /// Number of texts in each submitted request, in submission order.
    ///
    /// Used to reconstruct the flat embedding list from per-request results when collecting.
    pub request_sizes: Vec<usize>,
    /// Total number of texts submitted.
    pub total_texts: usize,
}

/// A provider-agnostic adapter for batch embedding APIs.
///
/// Each variant wraps the concrete client for that provider. To support a new batch embedding
/// provider, add a variant here and implement the three methods below.
///
/// The inner client types are `pub(crate)` implementation details; external code constructs
/// instances exclusively via [`BatchProviderAdapter::from_embedding_config`].
#[allow(private_interfaces)]
pub enum BatchProviderAdapter {
    /// Voyage AI batch API.
    VoyageAI(VoyageAIClient<ReqwestClient>),
}

impl BatchProviderAdapter {
    /// Texts per request when submitting to the Voyage AI batch API.
    ///
    /// Each Voyage AI request is capped at 120K tokens with a 32K context window per document,
    /// so three documents per request is the safe maximum.
    const VOYAGE_BATCH_SIZE: usize = 3;

    /// Construct a [`BatchProviderAdapter`] from an [`EmbeddingProviderConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`LLMError::GenericLLMError`] if the configured provider does not expose a batch API.
    pub fn from_embedding_config(config: &EmbeddingProviderConfig) -> Result<Self, LLMError> {
        match config {
            EmbeddingProviderConfig::VoyageAI(cfg) => {
                Ok(Self::VoyageAI(VoyageAIClient::with_config(cfg.clone())))
            }
            _ => Err(LLMError::GenericLLMError(format!(
                "Batch API is not supported for provider '{}'",
                config.provider_name()
            ))),
        }
    }

    /// Submit a flat list of texts to the batch embedding API.
    ///
    /// Internally, texts are grouped into provider-appropriate request chunks. The returned
    /// [`BatchJobMetadata`] records the batch ID and the chunk layout so results can be
    /// reassembled in the original order during [`Self::collect_embeddings`].
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying provider's [`BatchAPIProvider::submit_batch`].
    pub async fn submit_texts(&self, texts: Vec<String>) -> Result<BatchJobMetadata, LLMError> {
        match self {
            Self::VoyageAI(client) => {
                let batch_size = Self::VOYAGE_BATCH_SIZE;
                let request_sizes: Vec<usize> =
                    texts.chunks(batch_size).map(<[_]>::len).collect();
                let total_texts = texts.len();

                let requests: Vec<VoyageAIFilesRequest> = texts
                    .chunks(batch_size)
                    .enumerate()
                    .map(|(i, chunk)| VoyageAIFilesRequest {
                        custom_id: format!("req-{i}"),
                        body: VoyageAIFilesRequestBody {
                            input: chunk.to_vec(),
                        },
                    })
                    .collect();

                let response = client.submit_batch(requests).await?;

                Ok(BatchJobMetadata {
                    batch_id: response.id,
                    provider: "voyageai".to_string(),
                    request_sizes,
                    total_texts,
                })
            }
        }
    }

    /// Return the current [`BatchJobState`] for the given `batch_id`.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying provider's [`BatchAPIProvider::get_batch_status`].
    pub async fn get_status(&self, batch_id: &str) -> Result<BatchJobState, LLMError> {
        match self {
            Self::VoyageAI(client) => BatchAPIProvider::get_batch_status(client, batch_id).await,
        }
    }

    /// Collect embedding results for a completed batch.
    ///
    /// Returns a flat `Vec<Vec<f32>>` in the original text submission order, using the
    /// `request_sizes` recorded in `metadata` to reconstruct position. Failed individual
    /// requests are substituted with zero vectors.
    ///
    /// # Errors
    ///
    /// * [`LLMError::BatchNotCompleted`] — if the batch has not yet reached a terminal state.
    /// * Propagates all other errors from the underlying provider.
    pub async fn collect_embeddings(
        &self,
        metadata: &BatchJobMetadata,
    ) -> Result<Vec<Vec<f32>>, LLMError> {
        match self {
            Self::VoyageAI(client) => {
                client
                    .collect_batch_embeddings_ordered(
                        &metadata.batch_id,
                        &metadata.request_sizes,
                    )
                    .await
            }
        }
    }
}
