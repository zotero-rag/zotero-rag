# Zotero RAG QA System

> [!NOTE]
> This project is still active work-in-progress!

A Rust-based system for answering questions from your Zotero library using Retrieval-Augmented Generation (RAG).

## Overview

This project provides a command-line interface for querying your Zotero library with natural language questions. It uses RAG to search and retrieve relevant information from your academic papers, then generates answers grounded in your library content.

### Features

* We can currently extract text from PDFs and ignore tables/figures (they're unlikely to have useful context for LLMs)
* We can call LLMs (currently OpenAI and Anthropic) for embedding text and generating text.
* We can embed text using LanceDB.
* We can embed queries and perform vector search.

### Limitations

* The `zqa` CLI is limited--it can create embeddings for your library and perform vector search, but that's pretty much it.
* We only support OpenAI and Anthropic for LLMs, and only OpenAI and Voyage AI for embeddings. Future plans include Gemini, Vertex AI, ollama, and possibly OpenRouter and Groq.
* We do not perform chunking, so OpenAI embeddings are unlikely to work. Use Voyage AI in the meantime.

## Project Structure

The project is organized into three Rust crates:

- **pdftools**: PDF parsing and text extraction
- **rag**: Core RAG implementation with vector database and LLM clients
- **zqa**: Command-line interface and query processing

## Requirements

- Rust (2024 edition or higher)
- Zotero with a local library
- API key for OpenAI and Anthropic (for LLM capabilities), and Voyage AI (for embeddings)

## Installation

```bash
# Clone the repository
git clone https://github.com/yrahul3910/zotero-rag.git
cd zotero-rag

# Build the project
cargo build --release
```

### Note for building on Linux

On Linux, the project is configured to use the `mold` linker for faster linking. You can either install `mold` by following the [repo's instructions](https://github.com/rui314/mold) or by simply removing the `rustflags` line from `.cargo/config.toml`.

## Usage

### Set up environment variables

Create a `.env` in the root of the project with the following structure:
```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
VOYAGE_AI_API_KEY=
GEMINI_API_KEY=

MAX_CONCURRENT_REQUESTS=5
```

The model choices and the maximum concurrent requests above are defaults, and you can omit them. By default, the app uses Anthropic for generation and Voyage AI for embedding. This is currently the recommended settings (and so you won't need an OpenAI API key set up). Note that Voyage AI embeddings are the only ones that are known to work in a real setting; it is quite unlikely for the OpenAI embeddings to work at this moment, because we do not perform any chunking.

### Running the program

You can run the code using

```bash
cargo run --bin zqa
```

The binary exposes some options, but most of them are unlikely to be useful at the moment for end-users. The one that you might want to play with is `--model-provider`, which lets you choose the model provider. Note that this must be set for the model settings in your `.env` to have any effect.

### Using the program

Assuming you have Zotero installed with your library set up, you should first run `/process` to generate vector embeddings to store in LanceDB. Note that this will quite likely take a long time, especially if you have a large library. For example, I have a library with about 1100 items. This took about 45 minutes to extract text from the PDFs, and another 2-3 hours to generate embeddings (though I ran it in debug mode, so the text extraction step was likely considerably slower than if you use `--release`). The long time to generate embeddings cannot be worked around: this is a limitation on requests per minute and tokens per minute that Voyage AI has to prevent abuse.

Once you have embeddings set up, you can use `/stats` to check how many papers were successfully indexed. At this time, there is no way to update the embeddings with newly-added papers, but this is on the roadmap. You can also just ask any research question, and you will receive a response grounded in your library's research.

At any time, use `/help` to see available commands. You can use `quit`, `/quit`, or `q` to quit.

### Reporting issues

To report a bug, please re-run using `--log-level debug` and add the logs to your issue.

## Status

This project is a work in progress. Features and API may change significantly between versions.

## License

MIT License - See [LICENSE](LICENSE) for details.
