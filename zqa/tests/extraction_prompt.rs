use dotenv::dotenv;
use log::LevelFilter;
use zqa::common::setup_logger;

use std::{env, fs};

use rag::llm::base::{ApiClient, UserMessage};
use rag::llm::factory::{LLMClientConfig, get_client_with_config};
use zqa::cli::prompts::get_extraction_prompt;
use zqa::config::{AnthropicConfig, GeminiConfig, OpenAIConfig};

/// Test the extraction prompt with OpenAI API
#[tokio::test]
async fn test_extraction_prompt_openai() {
    dotenv().ok();
    setup_logger(LevelFilter::Info).unwrap();

    if !env::var("INTEGRATION_TESTS").is_ok() {
        // Only enable this in integration tests
        return;
    }

    // Ensure API key is available
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for this test");

    // Create OpenAI client config
    let config = OpenAIConfig {
        model: "gpt-4o-mini".to_string(),
        api_key: Some(api_key),
        max_tokens: 8192,
        embedding_model: None,
        embedding_dims: None,
    };

    let client = get_client_with_config(LLMClientConfig::OpenAI(config.into()))
        .expect("Failed to create OpenAI client");

    // Sample query and PDF text
    let query = "What is the main contribution of this paper?";
    let pdf_text = fs::read_to_string("assets/Zotero/storage/5KWS383N/.zotero-ft-cache")
        .expect("Failed to read cached Zotero file.");

    // Get the extraction prompt
    let prompt = get_extraction_prompt(query, &pdf_text);

    // Create and send the message
    let message = UserMessage {
        chat_history: Vec::new(),
        max_tokens: None,
        message: prompt,
    };

    let result = client.send_message(&message).await;

    // Verify the request succeeded
    assert!(
        result.is_ok(),
        "OpenAI API request failed: {:?}",
        result.err()
    );

    let response = result.unwrap();

    // Verify we got some content back
    assert!(
        !response.content.is_empty(),
        "Response content should not be empty"
    );

    // Verify the response contains expected XML tags from the prompt format
    assert!(
        response.content.contains("<title>") || response.content.contains("<excerpt>"),
        "Response should contain structured output with title or excerpt tags"
    );

    println!(
        "OpenAI extraction test passed. Token usage: input={}, output={}",
        response.input_tokens, response.output_tokens
    );
}

/// Test the extraction prompt with Anthropic API
#[tokio::test]
async fn test_extraction_prompt_anthropic() {
    dotenv().ok();
    setup_logger(LevelFilter::Info).unwrap();

    if !env::var("INTEGRATION_TESTS").is_ok() {
        return;
    }

    // Ensure API key is available
    let api_key =
        env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set for this test");

    // Create Anthropic client config
    let config = AnthropicConfig {
        model: "claude-3-5-haiku-latest".to_string(),
        api_key: Some(api_key),
        max_tokens: 8192,
    };

    let client = get_client_with_config(LLMClientConfig::Anthropic(config.into()))
        .expect("Failed to create Anthropic client");

    // Sample query and PDF text
    let query = "What is the main contribution of this paper?";
    let pdf_text = fs::read_to_string("assets/Zotero/storage/5KWS383N/.zotero-ft-cache")
        .expect("Failed to read cached Zotero file.");

    // Get the extraction prompt
    let prompt = get_extraction_prompt(query, &pdf_text);

    // Create and send the message
    let message = UserMessage {
        chat_history: Vec::new(),
        max_tokens: None,
        message: prompt,
    };

    let result = client.send_message(&message).await;

    // Verify the request succeeded
    assert!(
        result.is_ok(),
        "Anthropic API request failed: {:?}",
        result.err()
    );

    let response = result.unwrap();

    // Verify we got some content back
    assert!(
        !response.content.is_empty(),
        "Response content should not be empty"
    );

    // Verify the response contains expected XML tags from the prompt format
    assert!(
        response.content.contains("<title>") || response.content.contains("<excerpt>"),
        "Response should contain structured output with title or excerpt tags"
    );

    println!(
        "Anthropic extraction test passed. Token usage: input={}, output={}",
        response.input_tokens, response.output_tokens
    );
}

/// Test the extraction prompt with Gemini API
#[tokio::test]
async fn test_extraction_prompt_gemini() {
    dotenv().ok();
    setup_logger(LevelFilter::Info).unwrap();

    if !env::var("INTEGRATION_TESTS").is_ok() {
        return;
    }

    // Ensure API key is available (Gemini can use either GEMINI_API_KEY or GOOGLE_API_KEY)
    let api_key = env::var("GEMINI_API_KEY")
        .or_else(|_| env::var("GOOGLE_API_KEY"))
        .expect("GEMINI_API_KEY or GOOGLE_API_KEY must be set for this test");

    // Create Gemini client config
    let config = GeminiConfig {
        model: "gemini-2.5-flash".to_string(),
        api_key: Some(api_key),
        embedding_model: None,
        embedding_dims: None,
    };

    let client = get_client_with_config(LLMClientConfig::Gemini(config.into()))
        .expect("Failed to create Gemini client");

    // Sample query and PDF text
    let query = "What is the main contribution of this paper?";
    let pdf_text = fs::read_to_string("assets/Zotero/storage/5KWS383N/.zotero-ft-cache")
        .expect("Failed to read cached Zotero file.");

    // Get the extraction prompt
    let prompt = get_extraction_prompt(query, &pdf_text);

    // Create and send the message
    let message = UserMessage {
        chat_history: Vec::new(),
        max_tokens: None,
        message: prompt,
    };

    let result = client.send_message(&message).await;

    // Verify the request succeeded
    assert!(
        result.is_ok(),
        "Gemini API request failed: {:?}",
        result.err()
    );

    let response = result.unwrap();

    // Verify we got some content back
    assert!(
        !response.content.is_empty(),
        "Response content should not be empty"
    );

    // Verify the response contains expected XML tags from the prompt format
    assert!(
        response.content.contains("<title>") || response.content.contains("<excerpt>"),
        "Response should contain structured output with title or excerpt tags"
    );

    println!(
        "Gemini extraction test passed. Token usage: input={}, output={}",
        response.input_tokens, response.output_tokens
    );
}
