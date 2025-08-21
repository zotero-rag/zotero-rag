use super::errors::LLMError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatHistoryItem {
    pub role: String,
    pub content: String,
}

/// A user-facing struct that does not carry API-specific information. Clients should
/// convert from this to native message types.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UserMessage {
    pub chat_history: Vec<ChatHistoryItem>,
    pub max_tokens: Option<u32>,
    pub message: String,
}

/// A user-facing struct representing API responses, containing only information users
/// would be interested in.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ApiResponse {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[allow(async_fn_in_trait)]
pub trait ApiClient {
    async fn send_message(&self, message: &UserMessage) -> Result<ApiResponse, LLMError>;
}

#[derive(Clone, Debug)]
pub enum ModelProviders {
    OpenAI,
    Anthropic,
    OpenRouter,
}

impl ModelProviders {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelProviders::OpenAI => "openai",
            ModelProviders::Anthropic => "anthropic",
            ModelProviders::OpenRouter => "openrouter",
        }
    }

    pub fn contains(provider: &str) -> bool {
        [
            ModelProviders::OpenAI.as_str(),
            ModelProviders::Anthropic.as_str(),
            ModelProviders::OpenRouter.as_str(),
        ]
        .contains(&provider)
    }
}

#[derive(Clone, Debug)]
pub enum EmbeddingProviders {
    OpenAI,
    Anthropic,
    VoyageAI,
}

impl EmbeddingProviders {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingProviders::OpenAI => "openai",
            EmbeddingProviders::Anthropic => "anthropic",
            EmbeddingProviders::VoyageAI => "voyageai",
        }
    }

    pub fn contains(provider: &str) -> bool {
        [
            EmbeddingProviders::OpenAI.as_str(),
            EmbeddingProviders::Anthropic.as_str(),
            EmbeddingProviders::VoyageAI.as_str(),
        ]
        .contains(&provider)
    }
}
