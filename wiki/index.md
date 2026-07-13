---
okf_version: "0.1"
source_commit: f47a18f77f5209c956dc5da83a0e6dd5729771eb
---

# Zotero RAG QA System

* [Zotero RAG QA System](system-overview.md) - A Rust command-line system that indexes a local Zotero library and answers grounded questions over its PDFs.

# Application and libraries

* [zqa CLI](cli.md) - The interactive application that processes Zotero libraries, manages sessions, and orchestrates search and RAG queries.
* [zqa-rag](rag.md) - The reusable library for provider clients, embeddings, reranking, and LanceDB-backed vector retrieval.
* [zqa-pdftools](pdf-processing.md) - The academic-PDF parser that extracts structured text, detects sections, and produces retrieval chunks.
* [zqa Test Macros](macros.md) - Test-support crates that provide diagnostic assertions and retrying asynchronous test helpers.

# Operations

* [Runtime Configuration](configuration.md) - Provider settings, API credentials, and runtime defaults loaded from user configuration and environment variables.
* [Development Workflow](development.md) - Workspace layout, local requirements, and commands for building, testing, and linting the project.
