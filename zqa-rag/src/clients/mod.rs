//! Definitions for clients to interact with API providers. This module only contains the structs
//! themselves, as well as helper functions that are useful for trait implementations.
//! The [`lancedb::embeddings::EmbeddingFunction`] and the [`crate::llm::base::ApiClient`] traits
//! are implemented in [`crate::embedding`] and [`crate::llm`] respectively.

pub(crate) mod anthropic;
pub(crate) mod gemini;
pub(crate) mod ollama;
pub(crate) mod openai;
pub(crate) mod openrouter;
