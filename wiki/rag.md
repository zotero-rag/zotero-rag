---
type: Crate
title: zqa-rag
description: The reusable library for provider clients, embeddings, reranking, and LanceDB-backed vector retrieval.
tags: [rag, providers, lancedb]
timestamp: 2026-07-12T19:56:20-07:00
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
| Embeddings | OpenAI, Voyage AI, Cohere, ZeroEntropy, Gemini, Ollama |
| Reranking | Voyage AI, Cohere, ZeroEntropy |
| Batch embeddings | Voyage AI |

The registry also registers embedding implementations with LanceDB. This keeps
provider selection in configuration rather than in application-specific code.

# LLM and tool interface

`LLMClient::send_message` sends `ChatRequest` values and returns normalized completion
responses. A request can include chat history, reasoning settings,
streaming callbacks, and a list of `Tool` trait objects. Each tool supplies a
name, description, JSON Schema for its arguments, and an asynchronous call.
Provider adapters serialize that shared interface into their own tool schema
format.

The CLI uses this interface for retrieval, paper summarization, and
session-imported document tools.

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

The LanceDB URI is configurable through `LANCEDB_URI`. The library exposes
health and doctor modules for checking table access, size, row counts,
zero-vector rows, index state, and version drift. The CLI surfaces these
operations through `/checkhealth` and `/doctor`.

# Related concepts

[Runtime configuration](/configuration.md) selects providers and models.
[PDF processing](/pdf-processing.md) produces the text chunks that the
application converts into vector-store records.
