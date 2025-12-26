use dotenv::dotenv;
use log::LevelFilter;
use zqa::common::setup_logger;
use zqa::utils::library::ZoteroItemMetadata;

use std::{env, fs};

use zqa::cli::prompts::get_extraction_prompt;
use zqa::config::{AnthropicConfig, GeminiConfig, OpenAIConfig};
use zqa_rag::config::LLMClientConfig;
use zqa_rag::llm::base::{ApiClient, ChatRequest, ContentType};
use zqa_rag::llm::factory::get_client_with_config;

async fn run_extraction_test(client: zqa_rag::llm::factory::LLMClient, provider_name: &str) {
    // Sample query and PDF text
    let query = "What is the main contribution of this paper?";
    let pdf_text = fs::read_to_string("assets/Zotero/storage/5KWS383N/.zotero-ft-cache")
        .expect("Failed to read cached Zotero file.");

    // Get the extraction prompt
    let prompt = get_extraction_prompt(
        query,
        &pdf_text,
        &ZoteroItemMetadata {
            library_key: "5KWS383N".into(),
            title: "Baydin et al. - 2018 - Online Learning Rate Adaptation with Hypergradient Descent.pdf".into(),
            file_path: "assets/Zotero/storage/5KWS383N/.zotero-ft-cache".into(),
            authors: None,
        },
    );

    // Create and send the message
    let request = ChatRequest {
        chat_history: Vec::new(),
        max_tokens: None,
        message: prompt,
        tools: None,
    };

    let result = client.send_message(&request).await;

    // Verify the request succeeded
    assert!(
        result.is_ok(),
        "{} API request failed: {:?}",
        provider_name,
        result.err()
    );

    let response = result.unwrap();

    // Verify we got some content back
    assert!(
        !response.content.is_empty(),
        "Response content should not be empty"
    );

    let content = response.content.first().unwrap();
    match content {
        ContentType::ToolCall(_) => {
            panic!("failed assertion: content is a tool call");
        }
        ContentType::Text(s) => {
            // Verify the response contains expected XML tags from the prompt format
            assert!(
                s.contains("<title>") || s.contains("<excerpt>"),
                "Response should contain structured output with title or excerpt tags"
            );
        }
    }

    println!(
        "{} extraction test passed. Token usage: input={}, output={}",
        provider_name, response.input_tokens, response.output_tokens
    );
}

/// Test the extraction prompt with OpenAI API
#[tokio::test]
async fn test_extraction_prompt_openai() {
    dotenv().ok();
    let _ = setup_logger(LevelFilter::Info);

    if env::var("INTEGRATION_TESTS").is_err() {
        // Only enable this in integration tests
        return;
    }

    // Ensure API key is available
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for this test");

    // Create OpenAI client config
    let config = OpenAIConfig {
        model: Some("gpt-5-mini-2025-08-07".to_string()),
        api_key: Some(api_key),
        max_tokens: 8192,
        embedding_model: None,
        embedding_dims: None,
    };

    let client = get_client_with_config(LLMClientConfig::OpenAI(config.into()))
        .expect("Failed to create OpenAI client");

    run_extraction_test(client, "OpenAI").await;
}

/// Test the extraction prompt with Anthropic API
#[tokio::test]
async fn test_extraction_prompt_anthropic() {
    dotenv().ok();
    let _ = setup_logger(LevelFilter::Info);

    if env::var("INTEGRATION_TESTS").is_err() {
        return;
    }

    // Ensure API key is available
    let api_key =
        env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set for this test");

    // Create Anthropic client config
    let config = AnthropicConfig {
        model: Some("claude-haiku-4-5-20251001".to_string()),
        api_key: Some(api_key),
        max_tokens: 8192,
    };

    let client = get_client_with_config(LLMClientConfig::Anthropic(config.into()))
        .expect("Failed to create Anthropic client");

    run_extraction_test(client, "Anthropic").await;
}

/// Test the extraction prompt with Gemini API
#[tokio::test]
async fn test_extraction_prompt_gemini() {
    dotenv().ok();
    let _ = setup_logger(LevelFilter::Info);

    if env::var("INTEGRATION_TESTS").is_err() {
        return;
    }

    // Ensure API key is available (Gemini can use either GEMINI_API_KEY or GOOGLE_API_KEY)
    let api_key = env::var("GEMINI_API_KEY")
        .or_else(|_| env::var("GOOGLE_API_KEY"))
        .expect("GEMINI_API_KEY or GOOGLE_API_KEY must be set for this test");

    // Create Gemini client config
    let config = GeminiConfig {
        model: Some("gemini-2.5-flash".to_string()),
        api_key: Some(api_key),
        embedding_model: None,
        embedding_dims: None,
    };

    let client = get_client_with_config(LLMClientConfig::Gemini(config.into()))
        .expect("Failed to create Gemini client");

    run_extraction_test(client, "Gemini").await;
}
