//! Definitions for clients to interact with API providers. This module only contains the structs
//! themselves, as well as helper functions that are useful for trait implementations.
//! The [`lancedb::embeddings::EmbeddingFunction`] and the [`crate::llm::base::ApiClient`] traits
//! are implemented in [`crate::embedding`] and [`crate::llm`] respectively.

pub mod anthropic;
pub mod gemini;
pub mod ollama;
pub mod openai;
pub mod openrouter;
