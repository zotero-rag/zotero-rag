# zqa-rag

A provider-agnostic RAG and agentic library supporting multiple LLM, embedding, and reranking backends, backed by a LanceDB vector store.

## Features

- **Multi-provider LLM support**: Anthropic, OpenAI, Ollama, Gemini, OpenRouter
- **Embedding providers**: OpenAI, Gemini, VoyageAI, Cohere
- **Reranking**: VoyageAI, Cohere
- **Tool calling**: Trait-based tool definitions, see `llm::tools::Tool`.
- **Vector store**: LanceDB integration with backup, health-check, and diagnostic utilities
- **Rate limiting**: Exponential backoff with jitter, respects `Retry-After` headers for HTTP 429 responses.
- **Pricing estimation**: Fetches and caches LiteLLM pricing data from GitHub
- **Testable**: Generic `HttpClient` trait with `MockHttpClient` and `SequentialMockHttpClient` test implementations

## Core Abstractions

### LLM

Create a `ChatRequest`:

```rust
let request = ChatRequest {
    chat_history: Vec::new(),
    max_tokens: None,
    message: prompt,
    tools: None,
    on_tool_call: None,
    on_text: None,
};
```

Select a provider via the `LLMClient` factory enum:

```rust
let config = AnthropicConfig { api_key: "...".into(), model: "claude-sonnet-4-6", max_tokens: 8192 };
let client = get_client_with_config(config).expect("Failed to create Anthropic client");
let response = client.send_message(request).await?;
```

### Tool Calling

Implement the `Tool` trait; schemas are derived from `schemars::JsonSchema`:

```rust
struct MyTool;

impl Tool for MyTool {
    type Input = MyInput;   // must impl JsonSchema + DeserializeOwned
    type Output = String;

    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Does something useful" }
    async fn call(&self, input: Self::Input) -> Self::Output { ... }
}
```

### Embeddings

```rust
let embeddings = compute_embeddings_async(
    &texts,
    EmbeddingProviderConfig::OpenAI(config),
    batch_size,
).await?;
```

Empty strings return embeddings filled with zeros.

### Vector Store

```rust
let records: Vec<RecordBatch> = ...;

// Insert, query, and delete from LanceDB
vector::lance::insert_records(&db, &table_name, records).await?;
let results = vector::lance::query(&db, &table_name, embedding, top_k).await?;
```

## Providers & Capabilities

The enums in `capabilities` tell you which clients support each operation.

## Configuration

Each provider has a typed config struct (e.g., `AnthropicConfig`, `OpenAIConfig`, `OllamaConfig`). Credentials are read from environment variables via `dotenv`.

## MSRV

Rust **1.91** (edition 2024).
