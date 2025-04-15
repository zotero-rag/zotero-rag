# Zotero RAG QA System

A Rust-based system for answering questions from your Zotero library using Retrieval-Augmented Generation (RAG).

## Overview

This project provides a command-line interface for querying your Zotero library with natural language questions. It uses RAG techniques to search and retrieve relevant information from your academic papers, then generates answers grounded in your library content.

## Features

- PDF parsing with special handling for academic papers and mathematical symbols
- Integration with local Zotero SQLite databases
- Vector embeddings for semantic search using LanceDB
- LLM integration with Anthropic's Claude models (more on the way!)
- Flexible task chain system for processing queries

## Project Structure

The project is organized into three Rust crates:

- **pdftools**: PDF parsing and text extraction
- **rag**: Core RAG implementation with vector database and LLM clients
- **zqa**: Command-line interface and query processing

## Requirements

- Rust (2021 edition)
- Zotero with a local library
- API key for Anthropic Claude (for LLM capabilities)

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
