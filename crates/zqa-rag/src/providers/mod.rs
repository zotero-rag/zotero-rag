//! Provider implementations, traits, and registry

pub mod anthropic;
pub mod cohere;
pub mod gemini;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod provider_id;
pub mod registry;
/// Mock LLM provider for use in tests (enabled by the `mock` feature).
#[cfg(any(test, feature = "mock"))]
pub mod test;
pub mod voyage;
pub mod zeroentropy;

pub use provider_id::ProviderId;
