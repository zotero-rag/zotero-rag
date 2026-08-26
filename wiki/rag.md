---
type: Crate
title: zqa-rag
description: The reusable library for provider clients, embeddings, reranking, and LanceDB-backed vector retrieval.
tags: [rag, providers, lancedb]
timestamp: 2026-08-26T00:59:13-04:00
---

# Responsibilities

`zqa-rag` separates provider-specific API details from the application. It
owns configuration types, LLM client creation, embedding and reranking
factories, the vector-backend abstraction, LanceDB integration, pricing, and
storage diagnostics. [The CLI](/cli.md) supplies Zotero-specific records and
user interaction on top of those interfaces.

# Provider registry

`ProviderRegistry` maps a canonical `ProviderId` to factories for each
capability. A provider configuration selects a factory, which creates one of
these objects:

| Capability | Registered providers |
| --- | --- |
| Generation | Anthropic, OpenAI, OpenRouter, Gemini, Ollama |
| Embeddings | OpenAI, Voyage AI, Cohere, Gemini, Ollama |
| Reranking | Voyage AI, Cohere |
| Batch embeddings | Voyage AI |

The registry also registers embedding implementations with LanceDB. This keeps
provider selection in configuration rather than in application-specific code.

# LLM and tool interface

`ChatRequest` carries chat history, an optional system prompt, reasoning
settings, streaming callbacks, a list of `Tool` trait objects, and an
optional `tool_iteration_limit`. The system prompt is mapped onto each
provider's native system-instruction field (Anthropic's `system`, Gemini's
`system_instruction`, OpenAI's first-message roles, and so on).

Reasoning is configured through `ReasoningConfig`, which accepts a token
budget, an effort level (`none` through `max`), and a summary preference.
`normalize` fills in whichever of budget or effort is missing using a fixed
mapping, so providers can consume either representation. Providers adapt
that neutral form to their native API: Anthropic maps effort onto the
output-level `effort` parameter of its adaptive-thinking models (Claude Opus
4.6+ and Claude 5), OpenAI always requests a reasoning summary (defaulting
to `auto`), and Gemini preserves model `thought` content when replaying
history. Each
tool supplies a name, description, JSON Schema for its arguments, and an
asynchronous call.

All generation providers share one agentic loop. Each provider implements the
internal `AgenticClient` trait: it builds an initial history in its native
message format and performs one request-response round trip (`send_once`) that
returns both native items and a provider-agnostic view. The shared
`send_message` default method drives the loop: it dispatches tool calls until
the model returns text without tool requests, accumulates token usage across
turns, and serializes tools into each provider's schema format. The loop stops
after `tool_iteration_limit` round trips (15 by default), withholding tools on
the final turn so the model must answer in text. Tool calls to unknown names
are recovered as error results rather than aborting the loop.

Provider-specific behaviors worth noting: Anthropic preserves thinking-block
signatures (including redacted thinking) across history replays, and Gemini
records function-call IDs so tool results pair correctly on retried or
multi-call turns.

The CLI uses this interface for retrieval, paper summarization, and
session-imported document tools.

# Usage and pricing

Every completion returns a `ModelUsage` covering input, cache-write,
cache-read, output, and reasoning tokens; usage from each turn of the agentic
loop is accumulated into the final response. `ModelPricing::estimate_cost`
combines usage with per-token rates loaded asynchronously by
`get_model_pricing`, which caches fetched price sheets on disk with a TTL and
memoizes them in-process.

# Retrieval pipeline

Embedding providers implement LanceDB's embedding-function interface. The
common embedding layer selects a provider from `EmbeddingProviderConfig` and
handles shared batching behavior. Rerankers implement the `Rerank` trait,
which returns result ordering for a query and candidate texts.

`VectorBackend` is the generic persistence boundary: it defines connection,
indexing, insertion, deduplication, metadata, and vector-search operations.
`LanceBackend` is the current implementation. It stores data and metadata
tables, records the embedding provider and model, and tracks data-table version
drift so health checks can detect out-of-band updates.

# Storage operations

The LanceDB URI is configurable through `LANCEDB_URI`. Health checking is
backend-agnostic: `checkhealth` defines the neutral `HealthCheckResult`
type and the `HealthCheckable` trait, and each backend implements the checks
that make sense for it. Checks that do not apply to a backend (for example,
storage size for a remote store) are reported as absent rather than assumed,
and the same `Option`/`Result` fields distinguish \"not run\", \"succeeded\",
and \"failed\". `LanceBackend`'s implementation reports table access, size,
row counts, zero-vector rows, index state, and version drift; the `doctor`
module provides deeper diagnostics. [The CLI](/cli.md) surfaces these
operations through `/checkhealth` and `/doctor`.

# Related concepts

[Runtime configuration](/configuration.md) selects providers and models.
[PDF processing](/pdf-processing.md) produces the text chunks that the
application converts into vector-store records.
