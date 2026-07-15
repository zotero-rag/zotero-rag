//! Definitions for clients that interact with API providers.
//!
//! This module contains provider client structs and helpers. Their generation adapters live in
//! [`crate::llm`], while embedding implementations live in [`crate::embedding`].

pub(crate) mod anthropic;
pub(crate) mod gemini;
pub(crate) mod ollama;
pub(crate) mod openai;
pub(crate) mod openrouter;
#[cfg(any(test, feature = "mock"))]
pub(crate) mod test;
