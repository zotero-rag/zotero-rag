use crate::llm::anthropic::AnthropicClient;
use crate::llm::base::ApiClient;

pub fn get_client_by_provider(provider: &str) -> impl ApiClient {
    match provider {
        "anthropic" => AnthropicClient {},
        _ => panic!("Invalid provider"),
    }
}
