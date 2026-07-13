---
type: Crate
title: zqa CLI
description: The interactive application that processes Zotero libraries, manages sessions, and orchestrates search and RAG queries.
tags: [cli, zotero, rag]
timestamp: 2026-07-12T19:56:20-07:00
---

# Startup and state

`zqa/src/main.rs` loads `.env` and starts the Tokio runtime. `zqa::run` sets a
state-directory LanceDB location when `LANCEDB_URI` is not already set, loads
[runtime configuration](/configuration.md), performs first-run setup when
needed, creates a `LanceZoteroStore`, and enters the interactive CLI.

The REPL uses `rustyline` with history in `~/.zqa_history`. It keeps the
current chat history, title, imported documents, and session state in the
application context; interrupting the REPL saves the current conversation.

# Commands

| Command | Purpose |
| --- | --- |
| `/process` | Read new Zotero items, parse their PDFs, and add embeddings to the local store. |
| `/embed` and `/embed fix` | Resume a saved embedding batch or repair zero-vector rows. |
| `/search <query>` | Run vector retrieval and print paper titles without generation. |
| Natural-language input | Run a full RAG query; inputs shorter than ten characters are rejected as accidental prompts. |
| `/index`, `/dedup`, `/stats` | Maintain and inspect the local vector store. |
| `/checkhealth`, `/doctor` | Inspect storage health and diagnose database problems. |
| `/new`, `/resume` | Start a fresh conversation or load a saved one. |
| `/docs list`, `/docs remove <key>`, `/docs clear` | Inspect or remove session-scoped imported PDFs. |
| `/batch create`, `/batch check`, `/batch cancel <id>` | Manage supported provider batch-embedding jobs. |
| `/config`, `/help`, `/quit` | Display configuration, help, or leave the REPL. |

# Library processing

`/process` reads Zotero metadata, converts parsed PDFs to Arrow record batches,
and asks `LanceZoteroStore` to embed and upsert them. Before the upsert, it
writes the batch to `batch_iter.bin`. If embedding fails, that recovery file is
kept so `/embed` can replay the parsed records rather than parsing the library
again.

The store is the application-specific layer over [zqa-rag](/rag.md)'s
`LanceBackend`. Its records include Zotero library keys, titles, PDF paths, and
PDF text.

# Query execution

`/search` retrieves up to ten results through the local store and optionally
reranks them. A natural-language query constructs an LLM request with retrieval
and summarization tools, plus tools for any session-imported PDFs. The CLI
streams model text and tool status to the terminal, then records the completed
turn and its token and cost information in session state.

Imported PDFs are parsed with [zqa-pdftools](/pdf-processing.md), but live only
for the current session. This lets a question refer to a local PDF without
adding it to Zotero or the vector index.

# Related concepts

[System overview](/system-overview.md) describes the full data flow.
[Runtime configuration](/configuration.md) documents provider selection, and
[macros](/macros.md) documents the test helpers used by this crate.
