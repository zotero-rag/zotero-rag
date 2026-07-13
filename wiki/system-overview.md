---
type: System
title: Zotero RAG QA System
description: A Rust command-line system that indexes a local Zotero library and answers grounded questions over its PDFs.
tags: [zotero, rag, rust]
timestamp: 2026-07-12T19:56:20-07:00
---

# Purpose

`zqa` makes a local Zotero library searchable and queryable with natural-language
questions. It combines PDF text extraction, semantic indexing, optional
reranking, and model-generated answers grounded in retrieved library content.

# End-to-end flow

1. The user configures generation, embedding, and optional reranking providers.
2. The CLI processes Zotero entries and their PDFs.
3. [zqa-pdftools](/pdf-processing.md) extracts text and splits it into chunks.
4. [zqa-rag](/rag.md) embeds those chunks and persists searchable vectors in
   LanceDB.
5. A search or conversation retrieves relevant chunks; an LLM uses that context
   and its available tools to produce a response. Search-only mode returns
   relevant papers without generating an answer.

# Workspace boundaries

| Crate | Responsibility |
| --- | --- |
| [zqa](/cli.md) | Interactive command-line application, session state, library processing, and query orchestration. |
| [zqa-rag](/rag.md) | Reusable provider clients, embeddings, reranking, and vector retrieval. |
| [zqa-pdftools](/pdf-processing.md) | PDF parsing, text extraction, and chunking for academic papers. |
| [zqa-macros](/macros.md) | Declarative test assertions with diagnostic output. |
| [zqa-macros-proc](/macros.md) | Procedural test helpers, including the async `#[retry]` attribute. |

# Operating model

The vector index is stored locally in LanceDB. Generation, embedding, and
reranking can use local or hosted providers, so credentials and model choices
belong in [runtime configuration](/configuration.md), not in the repository.
See the [development workflow](/development.md) for local build and test
commands.
