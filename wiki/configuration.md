---
type: Configuration
title: Runtime Configuration
description: Provider settings, API credentials, and runtime defaults loaded from user configuration and environment variables.
tags: [configuration, providers, security]
timestamp: 2026-07-12T19:58:42-07:00
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
Ollama for generation; OpenAI, Voyage AI, Cohere, ZeroEntropy, Gemini, and
Ollama for embeddings; and Voyage AI, Cohere, and ZeroEntropy for reranking.
Reranking is disabled when `reranker_provider` is omitted. Model names,
dimensions, token limits, and provider-specific options belong to each
provider's configuration section. See [zqa-rag](/rag.md) for the provider and
retrieval abstractions.

# Operational settings

`max_concurrent_requests` controls concurrent embedding requests, while
`max_retries` controls retries after network failures. `LANCEDB_URI` overrides
the database location. When it is unset, the CLI places the database under its
state directory before initializing the vector store.

# Related concepts

[System overview](/system-overview.md) shows where these choices affect the
runtime pipeline. [The CLI](/cli.md) owns user-facing setup and command flows.
