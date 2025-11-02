use futures::StreamExt;
use std::borrow::Cow;
use std::env;
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use futures::stream;
use lancedb::embeddings::EmbeddingFunction;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::common::request_with_backoff;
use crate::constants::{
    DEFAULT_GEMINI_EMBEDDING_MODEL, DEFAULT_GEMINI_MODEL, DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_RETRIES, GEMINI_EMBEDDING_DIM,
};
use crate::llm::base::{ChatHistoryContent, ChatHistoryItem, ContentType, ToolCallRequest};
use crate::llm::tools::{SerializedTool, get_owned_tools, process_tool_calls};

use super::base::{ApiClient, ChatRequest, CompletionApiResponse};
use super::errors::LLMError;
use super::http_client::{HttpClient, ReqwestClient};

/// A client for Google's Gemini APIs (chat + embeddings)
#[derive(Debug, Clone)]
pub struct GeminiClient<T: HttpClient = ReqwestClient> {
    pub client: T,
    pub config: Option<crate::config::GeminiConfig>,
}

impl<T: HttpClient + Default> Default for GeminiClient<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Call the Gemini embeddings API.
///
/// # Arguments:
///
/// * `client`: An `HTTPClient` implementation.
/// * `text`: The text to embed.
///
/// # Returns
///
/// An embedding vector if the request was successful.
async fn call_gemini_embedding_api(
    client: &impl HttpClient,
    text: String,
) -> Result<Vec<f32>, LLMError> {
    let api_key = get_gemini_api_key()?;
    let model = env::var("GEMINI_EMBEDDING_MODEL")
        .ok()
        .unwrap_or_else(|| DEFAULT_GEMINI_EMBEDDING_MODEL.to_string());

    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse()?);
    headers.insert("x-goog-api-key", api_key.parse()?);

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent",
        model
    );
    let request_body = GeminiEmbeddingRequest::from_text(text);

    let res =
        request_with_backoff(client, &url, &headers, &request_body, DEFAULT_MAX_RETRIES).await?;
    let body = res.text().await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let parsed: GeminiEmbeddingResponse = serde_json::from_value(json).map_err(|e| {
        LLMError::GenericLLMError(format!(
            "Failed to deserialize Gemini embedding response: {e}"
        ))
    })?;

    Ok(parsed.embedding.values)
}

impl<T> GeminiClient<T>
where
    T: HttpClient + Default,
{
    /// Creates a new GeminiClient instance without configuration
    /// (will fall back to environment variables)
    pub fn new() -> Self {
        Self {
            client: T::default(),
            config: None,
        }
    }

    /// Creates a new GeminiClient instance with provided configuration
    pub fn with_config(config: crate::config::GeminiConfig) -> Self {
        Self {
            client: T::default(),
            config: Some(config),
        }
    }

    /// Internal method to compute embeddings that works with LLMError
    pub fn compute_embeddings_internal(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.compute_embeddings_async(source))
        })
    }

    /// Compute embeddings asynchronously using the Gemini API.
    ///
    /// # Arguments:
    ///
    /// * `source`: An Arrow array
    ///
    /// # Returns
    ///
    /// If successful, an Arrow array containing the embeddings for each source text.
    async fn compute_embeddings_async(
        &self,
        source: Arc<dyn arrow_array::Array>,
    ) -> Result<Arc<dyn arrow_array::Array>, LLMError> {
        let source_array = arrow_array::cast::as_string_array(&source);
        let texts: Vec<String> = source_array
            .iter()
            .filter_map(|s| Some(s?.to_owned()))
            .collect();

        // Create a stream of futures
        let futures = texts
            .iter()
            .map(|text| call_gemini_embedding_api(&self.client, text.clone()));

        // Convert to a stream and process with buffer_unordered to limit concurrency
        let max_concurrent = env::var("MAX_CONCURRENT_REQUESTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_CONCURRENT_REQUESTS);

        // Process futures with limited concurrency
        let results = stream::iter(futures)
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        // Process results and construct Arrow array
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for result in results {
            match result {
                Ok(embedding) => embeddings.push(embedding),
                Err(e) => return Err(e),
            }
        }

        // Convert to Arrow FixedSizeListArray
        let embedding_dim = if embeddings.is_empty() {
            GEMINI_EMBEDDING_DIM as usize
        } else {
            embeddings[0].len()
        };

        let flattened: Vec<f32> = embeddings.iter().flatten().copied().collect();
        let values = arrow_array::Float32Array::from(flattened);

        let list_array = arrow_array::FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            embedding_dim as i32,
            Arc::new(values),
            None,
        )
        .map_err(|e| {
            LLMError::GenericLLMError(format!(
                "Failed to create FixedSizeListArray in Gemini embeddings: {e}"
            ))
        })?;

        Ok(Arc::new(list_array) as Arc<dyn arrow_array::Array>)
    }
}

/// Get the Gemini API key from environment variables.
fn get_gemini_api_key() -> Result<String, LLMError> {
    // Prefer GEMINI_API_KEY, fallback to GOOGLE_API_KEY if present
    match env::var("GEMINI_API_KEY") {
        Ok(v) => Ok(v),
        Err(_) => Ok(env::var("GOOGLE_API_KEY")?),
    }
}

/// A function (tool) call request from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionCall {
    /// A unique ID for the function call
    id: String,
    /// The name of the tool (function) to call
    name: String,
    /// The function parameters
    args: serde_json::Value,
}

/// A result of a tool call, to be sent to the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionResult {
    /// The ID corresponding to the tool call request
    id: String,
    /// The name of the function
    name: String,
    /// The function response in JSON format
    response: serde_json::Value,
}

/// A content part in a request to the Gemini API
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase", untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        function_call: GeminiFunctionCall,
    },
    FunctionResult {
        function_response: GeminiFunctionResult,
    },
}

/// Content for requests to the Gemini API
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

impl From<ChatHistoryItem> for GeminiContent {
    fn from(value: ChatHistoryItem) -> Self {
        Self {
            role: value.role,
            parts: value
                .content
                .into_iter()
                .map(|c| match c {
                    ChatHistoryContent::Text(text) => GeminiPart::Text { text },
                    ChatHistoryContent::ToolCallRequest(tool_call) => GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            id: tool_call.id,
                            name: tool_call.tool_name,
                            args: tool_call.args,
                        },
                    },
                    ChatHistoryContent::ToolCallResponse(tool_res) => GeminiPart::FunctionResult {
                        function_response: GeminiFunctionResult {
                            id: tool_res.id,
                            name: tool_res.tool_name,
                            response: tool_res.result,
                        },
                    },
                })
                .collect::<Vec<_>>(),
        }
    }
}

/// Thinking config in case reasoning models are used
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
}

/// Optional text generation configuration
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiToolDeclaration<'a> {
    function_declarations: Vec<SerializedTool<'a>>,
}

/// The request body for text generation
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiRequestBody<'a> {
    contents: &'a [GeminiContent],
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a GeminiToolDeclaration<'a>>,
}

/// Helper to build contents, config, and tools from a ChatRequest.
/// Returns owned data that can then be borrowed by GeminiRequestBody.
fn build_gemini_request_data<'a>(
    req: &'a ChatRequest<'a>,
) -> (
    Vec<GeminiContent>,
    Option<GeminiGenerationConfig>,
    Option<GeminiToolDeclaration<'a>>,
) {
    let model_max = req.message.max_tokens.or_else(|| {
        env::var("GEMINI_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
    });
    let mut contents: Vec<GeminiContent> = req
        .message
        .chat_history
        .iter()
        .map(|c| {
            let content = c.content[0].clone();

            GeminiContent {
                role: map_role(&c.role),
                parts: vec![GeminiPart::Text {
                    text: match content {
                        ChatHistoryContent::Text(s) => s,
                        _ => "".into(),
                    },
                }],
            }
        })
        .collect();

    contents.push(GeminiContent {
        role: "user".to_string(),
        parts: vec![GeminiPart::Text {
            text: req.message.message.clone(),
        }],
    });

    let owned_tools: Option<Vec<SerializedTool>> = get_owned_tools(req.tools);

    let generation_config = Some(GeminiGenerationConfig {
        max_output_tokens: model_max,
        // TODO: Make these configurable
        temperature: Some(1.0),
        top_k: Some(1),
        top_p: Some(1.0),
        thinking_config: Some(GeminiThinkingConfig {
            include_thoughts: Some(false),
            thinking_budget: Some(1024),
        }),
    });

    let tools = owned_tools.map(|tools| GeminiToolDeclaration {
        function_declarations: tools,
    });

    (contents, generation_config, tools)
}

/// Helper function to change "assistant" roles to "model" for Gemini's API.
fn map_role(role: &str) -> String {
    match role {
        // Gemini uses "model" instead of "assistant"
        "assistant" => "model".into(),
        _ => "user".into(),
    }
}

/// Send a Gemini text generation request with retry/backoff and parse the response.
///
/// This helper constructs the model endpoint URL, performs the HTTP request with
/// exponential backoff, and deserializes the response into a single response
/// candidate and usage metadata. If the response contains no candidates or cannot
/// be deserialized, an error is returned.
///
/// # Arguments:
///
/// * `client` - An `HttpClient` implementation used to execute the request.
/// * `headers` - HTTP headers including content type and API key.
/// * `req` - The Gemini request body including contents, config, and tools.
/// * `model` - The Gemini model name to target.
///
/// # Returns
///
/// On success, returns the first `GeminiResponseCandidate` and the
/// accompanying `GeminiUsageMetadata`.
async fn send_gemini_generation_request<'a>(
    client: &impl HttpClient,
    headers: &HeaderMap,
    req: &GeminiRequestBody<'a>,
    model: &str,
) -> Result<(GeminiResponseCandidate, GeminiUsageMetadata), LLMError> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model
    );

    let res = request_with_backoff(client, &url, headers, req, DEFAULT_MAX_RETRIES).await?;

    let body = res.text().await?;
    let json: serde_json::Value = match serde_json::from_str(&body) {
        Ok(json) => json,
        Err(_) => return Err(LLMError::DeserializationError(body)),
    };

    let response: GeminiResponseBody = match serde_json::from_value(json) {
        Ok(response) => response,
        Err(_) => return Err(LLMError::DeserializationError(body)),
    };
    let first_candidate = response
        .candidates
        .first()
        .ok_or_else(|| LLMError::GenericLLMError("No candidates in Gemini response".into()))?;

    Ok((first_candidate.clone(), response.usage_metadata))
}

/// Usage metadata received from the Gemini text generation response.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    thoughts_token_count: u32,
    candidates_token_count: u32,
    total_token_count: u32,
}

/// One of several response candidates.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseCandidate {
    content: GeminiContent,
    finish_reason: String,
}

/// Text generation response from the Gemini API.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseBody {
    candidates: Vec<GeminiResponseCandidate>,
    usage_metadata: GeminiUsageMetadata,
}

/// Convert Gemini response content into provider-agnostic `ChatHistoryContent` items.
///
/// Tool results should never appear in API responses; if encountered, they are ignored
/// with a warning.
fn map_response_to_chat_contents(contents: &[GeminiPart]) -> Vec<ChatHistoryContent> {
    contents.iter().filter_map(|c| {
        match c {
            GeminiPart::Text{text} => Some(ChatHistoryContent::Text(text.clone())),
            GeminiPart::FunctionCall{function_call: fc} => Some(ChatHistoryContent::ToolCallRequest(ToolCallRequest {
                    id: fc.id.clone(),
                    tool_name: fc.name.clone(),
                    args: fc.args.clone()
                })),
            _ => {
                log::warn!(
                    "Got a tool result from the API response. This is not expected, and will be ignored."
                );

                None
            }
        }
    }).collect::<Vec<_>>()
}

impl<T: HttpClient> ApiClient for GeminiClient<T> {
    async fn send_message<'a>(
        &self,
        request: &'a mut ChatRequest<'a>,
    ) -> Result<CompletionApiResponse, LLMError> {
        // TODO: Implement tool support for Gemini
        let key = get_gemini_api_key()?;
        let model = match &self.config {
            None => env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_GEMINI_MODEL.to_string()),
            Some(config) => config.model.clone(),
        };

        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse()?);
        headers.insert("x-goog-api-key", key.parse()?);

        // Build the initial contents, config, and tools (owned)
        let (mut chat_history, generation_config, tools) = build_gemini_request_data(request);

        // Create the initial request borrowing
        let req_body = GeminiRequestBody {
            contents: &chat_history,
            generation_config: generation_config.clone(),
            tools: tools.as_ref(),
        };

        let (mut response, mut usage) =
            send_gemini_generation_request(&self.client, &headers, &req_body, &model).await?;

        let mut has_tool_calls: bool = response
            .content
            .parts
            .iter()
            .any(|c| matches!(c, GeminiPart::FunctionCall { .. }));

        // Append the contents
        chat_history.push(GeminiContent {
            role: "assistant".into(),
            parts: response.content.parts.clone(),
        });

        let mut contents: Vec<ContentType> = Vec::new();

        while has_tool_calls {
            let converted_contents = map_response_to_chat_contents(&response.content.parts);

            // `unwrap` is likely to be safe here since tool calls exist
            let owned_tools = tools.as_ref().unwrap();
            process_tool_calls(
                &mut chat_history,
                &mut contents,
                &converted_contents,
                &owned_tools.function_declarations,
            )
            .await?;

            // Create a new request borrowing the updated chat history
            let updated_req_body = GeminiRequestBody {
                generation_config: generation_config.clone(),
                contents: &chat_history,
                tools: tools.as_ref(),
            };

            (response, usage) =
                send_gemini_generation_request(&self.client, &headers, &updated_req_body, &model)
                    .await?;

            // Append the new response to chat history
            chat_history.push(GeminiContent {
                role: "assistant".into(),
                parts: response.content.parts.clone(),
            });

            has_tool_calls = response
                .content
                .parts
                .iter()
                .any(|c| matches!(c, GeminiPart::FunctionCall { .. }));
        }

        // Process the final response (which has no tool calls) to extract text content
        for content in &response.content.parts {
            if let GeminiPart::Text { text } = content {
                contents.push(ContentType::Text(text.clone()))
            }
        }

        // TODO: Check if this metadata includes tool use
        Ok(CompletionApiResponse {
            content: contents,
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
        })
    }
}

/// Content for an embedding API request
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbeddingRequestContent {
    parts: Vec<GeminiPart>,
}

/// A request to embed texts using the Gemini API
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbeddingRequest {
    model: String,
    content: GeminiEmbeddingRequestContent,
}

impl GeminiEmbeddingRequest {
    fn from_text(text: String) -> Self {
        Self {
            model: env::var("GEMINI_EMBEDDING_MODEL")
                .ok()
                .unwrap_or_else(|| DEFAULT_GEMINI_EMBEDDING_MODEL.to_string()),
            content: GeminiEmbeddingRequestContent {
                parts: vec![GeminiPart::Text { text }],
            },
        }
    }
}

/// A vector containing the embeddings, returned as a nested object by Gemini's embedding API.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbeddingVector {
    values: Vec<f32>,
}

/// The full embedding API response
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbeddingResponse {
    embedding: GeminiEmbeddingVector,
}

/// Implements the LanceDB EmbeddingFunction trait for Gemini client.
impl<T: HttpClient + Default + std::fmt::Debug> EmbeddingFunction for GeminiClient<T> {
    fn name(&self) -> &str {
        "Gemini"
    }

    fn source_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    fn dest_type(&self) -> Result<Cow<'_, DataType>, lancedb::Error> {
        Ok(Cow::Owned(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            GEMINI_EMBEDDING_DIM as i32,
        )))
    }

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
    use std::sync::Mutex;

    use super::*;
    use crate::llm::base::{ApiClient, ChatHistoryItem, ChatRequest, UserMessage};
    use crate::llm::http_client::MockHttpClient;
    use crate::llm::tools::test_utils::MockTool;
    use arrow_array::Array;
    use dotenv::dotenv;
    use lancedb::embeddings::EmbeddingFunction;

    #[tokio::test]
    async fn test_send_message_with_mock() {
        dotenv().ok();

        let mock_response = GeminiResponseBody {
            candidates: vec![GeminiResponseCandidate {
                content: GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart::Text {
                        text: "Hello from Gemini!".into(),
                    }],
                },
                finish_reason: "stop".into(),
            }],
            usage_metadata: GeminiUsageMetadata {
                prompt_token_count: 7,
                candidates_token_count: 11,
                total_token_count: 18,
                thoughts_token_count: 0,
            },
        };

        let mock_http = MockHttpClient::new(mock_response);
        let client = GeminiClient {
            client: mock_http,
            config: None,
        };

        let message = UserMessage {
            message: "foo".into(),
            chat_history: vec![ChatHistoryItem {
                role: "assistant".into(),
                content: vec![ChatHistoryContent::Text("Prior".into())],
            }],
            max_tokens: Some(256),
        };
        let mut request = ChatRequest::from(&message);
        let res = client.send_message(&mut request).await;
        assert!(res.is_ok());
        let res = res.unwrap();
        assert_eq!(res.content.len(), 1);
        if let ContentType::Text(text) = &res.content[0] {
            assert_eq!(text, "Hello from Gemini!");
        } else {
            panic!("Expected Text content type");
        }
        assert_eq!(res.input_tokens, 7);
        assert_eq!(res.output_tokens, 11);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_compute_embeddings_mock() {
        dotenv().ok();

        // Build a deterministic mock response that returns a 3-length embedding
        #[derive(Debug, Serialize, Deserialize, Clone, Default)]
        struct MockEmbeddingResp {
            embedding: MockEmbeddingVec,
        }
        #[derive(Debug, Serialize, Deserialize, Clone, Default)]
        struct MockEmbeddingVec {
            values: Vec<f32>,
        }

        let mock = MockEmbeddingResp {
            embedding: MockEmbeddingVec {
                values: vec![1.0, 0.0, -1.0],
            },
        };

        let mock_http = MockHttpClient::new(mock);
        let client = GeminiClient {
            client: mock_http,
            config: None,
        };

        let array = arrow_array::StringArray::from(vec!["A", "B", " ", "C"]);
        let embeddings = client.compute_source_embeddings(Arc::new(array));

        assert!(embeddings.is_ok());
        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);
        assert_eq!(vector.len(), 4);
        // With mock 3-length vectors, value_length should be 3
        assert_eq!(vector.value_length(), 3);
    }

    #[tokio::test]
    async fn test_compute_embeddings() {
        dotenv().ok();

        let array = arrow_array::StringArray::from(vec![
            "Hello, World!",
            "A second string",
            "A third string",
            "A fourth string",
            "A fifth string",
            "A sixth string",
        ]);

        let client = GeminiClient::<ReqwestClient>::default();
        let embeddings = client.compute_embeddings_async(Arc::new(array)).await;

        // Debug the error if there is one
        if embeddings.is_err() {
            println!("Gemini embedding error: {:?}", embeddings.as_ref().err());
        }

        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        let vector = arrow_array::cast::as_fixed_size_list_array(&embeddings);

        assert_eq!(vector.len(), 6);
        assert_eq!(vector.value_length(), GEMINI_EMBEDDING_DIM as i32);
    }

    #[tokio::test]
    async fn test_request_works() {
        dotenv().ok();

        let client = GeminiClient::<ReqwestClient>::default();
        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "Hello!".to_owned(),
        };
        let mut request = ChatRequest::from(&message);
        let res = client.send_message(&mut request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Gemini test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_request_works_with_tools() {
        dotenv().ok();

        let client = GeminiClient::<ReqwestClient>::default();
        let call_count = Arc::new(Mutex::new(0));
        let tool = MockTool {
            call_count: Arc::clone(&call_count),
            schema_key: "parametersJsonSchema".into(),
        };
        let message = UserMessage {
            chat_history: Vec::new(),
            max_tokens: Some(1024),
            message: "This is a test. Call the `mock_tool`, passing in a `name`, and ensure it returns a greeting".to_owned(),
        };
        let mut request = ChatRequest {
            message: &message,
            tools: Some(&[Box::new(tool)]),
        };

        let res = client.send_message(&mut request).await;

        // Debug the error if there is one
        if res.is_err() {
            println!("Gemini test error: {:?}", res.as_ref().err());
        }

        assert!(res.is_ok());
    }
}
