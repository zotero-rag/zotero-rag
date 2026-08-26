---
type: Configuration
title: Runtime Configuration
description: Provider settings, API credentials, and runtime defaults loaded from user configuration and environment variables.
tags: [configuration, providers, security]
timestamp: 2026-08-26T01:32:28-04:00
---

# Configuration sources

The user configuration file is `~/.config/zqa/config.toml`. It selects the
providers and models used for generation, embeddings, and optional reranking,
along with runtime limits such as concurrent requests and retries. At startup,
`zqa` loads TOML values and then overwrites supported values from the
environment. Its executable loads a local `.env` file first, so it
can supply those environment values without putting credentials in TOML.

Keep real credentials out of version control. A configuration file that is
managed with dotfiles should leave secret values blank and rely on environment
variables or a local `.env` file instead.

# Provider roles

| Role | Purpose |
| --- | --- |
| Generation | Produces conversational answers and tool calls. |
| Embedding | Converts library chunks and queries into vectors for retrieval. |
| Reranking | Optionally refines retrieved results before they reach the answer model. |

The provider registry supports Anthropic, OpenAI, OpenRouter, Gemini, and
Ollama for generation; OpenAI, Voyage AI, Cohere, Gemini, and
Ollama for embeddings; and Voyage AI, and Cohere for reranking.
Reranking is disabled when `reranker_provider` is omitted. Model names,
dimensions, token limits, and provider-specific options belong to each
provider's configuration section. `embedding_dims` (for OpenAI, also the
`OPENAI_EMBEDDING_DIMS` environment variable) must match the embedding
model's output width, or a reduced width for models that support dimension
reduction (OpenAI's text-embedding-3 family, Voyage AI); changing it or the
embedding model requires rebuilding the index under `data/lancedb-table/`,
since old and new vectors are incompatible. See [zqa-rag](/rag.md) for the provider and
retrieval abstractions.

# Reasoning settings

Reasoning is off by default for all providers. Each generation provider's
section can set `reasoning_budget` (a token budget) or `reasoning_effort`
(`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`); effort is
the more widely supported form. When only one of the two is given, the other
is derived from a fixed mapping. Anthropic validates the configured effort at
config load and routes it through the output-level `effort` parameter of its
adaptive-thinking models rather than a token budget. Invalid effort values
are rejected with an error listing the accepted levels.

# Operational settings

`max_concurrent_requests` controls concurrent embedding requests, while
`max_retries` controls retries after network failures.
`tool_iteration_limit` bounds how many tool-call round trips a single user
message may trigger before the model is forced to answer without tools
(default 15). `LANCEDB_URI` overrides
the database location. When it is unset, the CLI places the database under its
state directory before initializing the vector store.

# Related concepts

[System overview](/system-overview.md) shows where these choices affect the
runtime pipeline. [The CLI](/cli.md) owns user-facing setup and command flows.
