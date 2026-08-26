---
okf_version: "0.1"
source_commit: 349c3fb4bea81a10d2be676f70df71f150ae2d62
---

# Zotero RAG QA System

* [Zotero RAG QA System](system-overview.md) - A Rust system that indexes a local Zotero library and answers grounded questions over its PDFs, through a CLI and a native GUI.

# Application and libraries

* [zqa CLI](cli.md) - The interactive application that processes Zotero libraries, manages sessions, and orchestrates search and RAG queries.
* [zqa-rag](rag.md) - The reusable library for provider clients, embeddings, reranking, and LanceDB-backed vector retrieval.
* [zqa-gui](gui.md) - The native GPUI desktop front-end that reuses the zqa engine, configuration, and LanceDB database.
* [zqa-pdftools](pdf-processing.md) - The academic-PDF parser that extracts structured text, detects sections, and produces retrieval chunks.
* [zqa Test Macros](macros.md) - Test-support crates that provide diagnostic assertions and retrying asynchronous test helpers.

# Operations

* [Runtime Configuration](configuration.md) - Provider settings, API credentials, and runtime defaults loaded from user configuration and environment variables.
* [Development Workflow](development.md) - Workspace layout, local requirements, and commands for building, testing, and linting the project.
