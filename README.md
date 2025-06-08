# Zotero RAG QA System

> [!NOTE]
> This project is still active work-in-progress!

A Rust-based system for answering questions from your Zotero library using Retrieval-Augmented Generation (RAG).

## Overview

This project provides a command-line interface for querying your Zotero library with natural language questions. It uses RAG to search and retrieve relevant information from your academic papers, then generates answers grounded in your library content--at least, that's the end goal, anyway. A lot of the functionality doesn't quite yet: 

* We can currently extract text from PDFs and ignore tables/figures (they're unlikely to have useful context for LLMs)
* We can call LLMs (currently OpenAI and Anthropic) for embedding text and generating text.
* We can embed text using LanceDB.
* These parts are not strung together yet--the `zqa` crate is currently a Hello World app.

## Project Structure

The project is organized into three Rust crates:

- **pdftools**: PDF parsing and text extraction
- **rag**: Core RAG implementation with vector database and LLM clients
- **zqa**: Command-line interface and query processing

## Requirements

- Rust (2021 edition or higher)
- Zotero with a local library
- API key for OpenAI and Anthropic (for LLM capabilities)

## Installation

```bash
# Clone the repository
git clone https://github.com/yrahul3910/zotero-rag.git
cd zotero-rag

# Build the project
cargo build --release
```

## Usage

```bash
# Set up environment variables (API keys, etc.)

# Run the CLI
cargo run --bin zqa
```

## Status

This project is a work in progress. Features and API may change significantly between versions.

## License

MIT License - See [LICENSE](LICENSE) for details.
