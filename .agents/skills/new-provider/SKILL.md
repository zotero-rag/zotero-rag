---
name: new-provider
description: Scaffold a new LLM, embedding, or reranking provider in zotero-rag-rs. Use this skill when adding support for a new AI provider to the zqa-rag crate. Covers all required touch-points across the client module, config, factory enum, capabilities enum, embedding factory, and constants.
metadata:
  author: zotero-rag-rs
  version: "1.0"
allowed-tools: Read Write Edit Grep Glob Bash(cargo fmt --all:*) Bash(cargo clippy --all-targets --all-features:*)
---

# new-provider

Scaffold all required touch-points for a new provider in `zqa-rag`.

## Step 1: Gather information

Before making any changes, confirm the following with the user:

1. **Provider name** — the human-readable name (e.g. "Mistral AI")
2. **Key** — the lowercase string key used in match arms and env vars (e.g. `"mistral"`)
3. **Capabilities** — which of these the provider supports:
   - LLM text generation (implements `ApiClient` from `llm/base.rs`)
   - Embedding (implements `EmbeddingFunction` from LanceDB)
   - Reranking (implements `Rerank` from `embedding/common.rs`)
4. **Default constants** needed per capability:
   - LLM: default model name
   - Embedding: default model name, embedding dimensions
   - Reranking: default reranker model name
5. **API key env var** (e.g. `MISTRAL_API_KEY`)

## Step 2: Add constants

In `zqa-rag/src/constants.rs`, add constants for each capability. Follow the naming pattern `DEFAULT_<NAME>_*`, only for capabilities this provider actually has:

```rust
/// Default <Name> model for chat completions
pub const DEFAULT_<NAME>_MODEL: &str = "<model-id>";

/// Default <Name> embedding model
pub const DEFAULT_<NAME>_EMBEDDING_MODEL: &str = "<embedding-model-id>";

/// Default <Name> embedding dimension
pub const DEFAULT_<NAME>_EMBEDDING_DIM: u32 = <dim>;

/// Default <Name> rerank model
pub const DEFAULT_<NAME>_RERANK_MODEL: &str = "<reranker-id>";
```

## Step 3: Add a config struct

In `zqa-rag/src/config.rs`, add a `<Name>Config` struct. Include only fields relevant to this provider's capabilities. Reference patterns:
- LLM-only: see `OpenRouterConfig` (just `api_key` and `model`)
- Embedding + reranking: see `CohereConfig` (`api_key`, `embedding_model`, `embedding_dims`, `reranker`)
- LLM + embedding: see `GeminiConfig`

All fields should have `/// doc` comments and the struct should derive `Debug, Clone`.

## Step 4: LLM text generation (if applicable)

**New file: `zqa-rag/src/llm/<key>.rs`**

Model on `zqa-rag/src/llm/openrouter.rs`. Requirements:

- Module-level `//!` doc comment
- `<Name>Client<T: HttpClient>` struct with `pub client: T` and `pub config: Option<<Name>Config>`
- `impl<T: HttpClient + Default> Default` delegating to `new()`
- `new()` and `with_config(config: <Name>Config)` constructors (both `#[must_use]`)
- Private request/response structs with `#[derive(Serialize)]` / `#[derive(Deserialize)]`
- `impl ApiClient for <Name>Client<T>`: `send_message` reads credentials from `self.config` first, then env vars
- Use `request_with_backoff` from `crate::common` for the HTTP call
- Tests: at minimum one mock test (using `MockHttpClient`) and one live integration test (using `ReqwestClient`, loads `.env` via `dotenv`)

**Register in three places:**

- `zqa-rag/src/llm/mod.rs` — add `pub mod <key>;`
- `zqa-rag/src/llm/factory.rs`:
  - Import the new client
  - Add `/// <Name> client` + `<Name>(<Name>Client)` variant to `LLMClient`
  - Add delegation arm to the `ApiClient for LLMClient` match
  - Add `"<key>" => Ok(LLMClient::<Name>(<Name>Client::new()))` to `get_client_by_provider`
  - Add `LLMClientConfig::<Name>(cfg) => Ok(LLMClient::<Name>(<Name>Client::with_config(cfg)))` to `get_client_with_config`
- `zqa-rag/src/config.rs` — add `/// <Name> client configuration` + `<Name>(crate::config::<Name>Config)` to `LLMClientConfig`

**`zqa-rag/src/capabilities.rs` — `ModelProvider`:**

- Add `/// <Name> model provider` + `<Name>` variant
- Add arm to `as_str()`: `ModelProvider::<Name> => "<key>"`
- Add `ModelProvider::<Name>.as_str()` to the array in `contains()`

## Step 5: Embedding (if applicable)

**File location:**
- Provider has *only* embedding/reranking: create `zqa-rag/src/embedding/<key>.rs` and add `pub mod <key>;` to `zqa-rag/src/embedding/mod.rs`
- Provider shares a module with its LLM client (e.g. Gemini, OpenAI): add the embedding impl to that existing LLM file

**Client implementation** — model on `zqa-rag/src/embedding/cohere.rs`:

- `compute_embeddings_internal(&self, source: Arc<dyn arrow_array::Array>)` calls `compute_embeddings_async` from `embedding::common` with provider-specific batch parameters
- Define `<Name>EmbedRequest` (`#[derive(Serialize, Debug)]`) and an untagged `<Name>Response` enum (`Success`/`Error` variants), implementing `EmbeddingApiResponse`
- `impl EmbeddingFunction for <Name>Client<T>`:
  - `name()` returns the display name string
  - `source_type()` returns `Ok(Cow::Owned(DataType::Utf8))`
  - `dest_type()` returns `Ok(Cow::Owned(DataType::FixedSizeList(..., DEFAULT_<NAME>_EMBEDDING_DIM as i32)))`
  - `compute_source_embeddings` and `compute_query_embeddings` both delegate to `compute_embeddings_internal`, mapping `LLMError` to `lancedb::Error::Other`

**Register in `zqa-rag/src/embedding/common.rs`:**

- Import the new client
- Add `"<key>" => Ok(Arc::new(<Name>Client::<ReqwestClient>::default()))` to `get_embedding_provider`
- Add arm to `get_embedding_provider_with_config`
- Add `/// Configuration for <Name> embedding provider` + `<Name>(crate::config::<Name>Config)` to `EmbeddingProviderConfig`
- Add arm to `provider_name()`

**`zqa-rag/src/capabilities.rs` — `EmbeddingProvider`:**

- Add `/// <Name> embedding provider` + `<Name>` variant
- Update `as_str()`, `contains()`, and `recommended_chunking_strategy()`
- For `recommended_chunking_strategy`: use `ChunkingStrategy::WholeDocument` for large context windows, or `ChunkingStrategy::SectionBased(<token_limit>)` for smaller ones (leave a ~20% buffer below the true limit)

## Step 6: Reranking (if applicable)

**Add `Rerank` impl to the provider's file** — model on the `Rerank<U>` impl in `zqa-rag/src/embedding/cohere.rs`:

- `impl<T: HttpClient, U: AsRef<str> + Send + Clone> Rerank<U> for <Name>Client<T>`
- Reads credentials from `self.config` first, then env vars
- Returns `Pin<Box<dyn Future<...>>>`

**Register in `zqa-rag/src/embedding/common.rs`:**

- Add `"<key>" => Ok(Arc::new(<Name>Client::<ReqwestClient>::default()))` to `get_reranking_provider`
- Add arm to `get_reranking_provider_with_config`
- Add `/// Configuration for <Name> reranking provider` + `<Name>(crate::config::<Name>Config)` to `RerankProviderConfig`

**`zqa-rag/src/capabilities.rs` — `RerankerProviders`:**

- Add `/// <Name> reranking provider` + `<Name>` variant
- Update `as_str()` and `contains()`

## Step 7: Add to `zqa`

**Add configs in `zqa/src/config.rs`:**

- Add an example of the config to the docstring at the top of the file.
- Add `<name>: Option<NameConfig>` to the `Config` struct.
- Add a `<Name>Config` struct near the other provider-specific config structs.
- Add a `From<NameConfig>` impl to map the `zqa-rag` structs to the `zqa` structs.
- Add a `match` arm to `get_generation_config` and `get_small_model_config` for generation models, `get_embedding_config` for embedding models, and `get_reranker_config` for rerankers.
- Add a new section to `read_env` in `zqa/src/config.rs`.
- Add a `match` arm to `run_query` in `zqa/src/cli/app.rs`.
- Add the config example to the `README.md`.
- Add a section to the `oobe` function in `state.rs`.
- Add the new config to the initializer in `zqa/tests/new_library.rs`.

## Step 8: Post-scaffold checklist

- [ ] All public types and functions have `///` doc comments; `# Arguments`, `# Returns`, and `# Errors` sections are present where applicable (see `STYLE.md`)
- [ ] No `anyhow` used — errors propagate via `LLMError` using `thiserror`
- [ ] No `println!` or `stdout` in `zqa-rag` — use `log::info!`/`log::debug!`/`eprintln!` for warnings only
- [ ] At least one mock test and one live integration test per new module
- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] `.env.tmpl` updated if a new API key env var was introduced
- [ ] Commit message follows Conventional Commits: `feat(rag): add <name> provider`
