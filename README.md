# Zotero RAG QA System

> [!NOTE]
> This project is still active work-in-progress!

A Rust-based system for answering questions from your Zotero library using Retrieval-Augmented Generation (RAG).

## Overview

This project provides a command-line interface for querying your Zotero library with natural language questions. It uses RAG to search and retrieve relevant information from your academic papers, then generates answers grounded in your library content.

### Features

* We can currently extract text from PDFs and ignore tables/figures (they're unlikely to have useful context for LLMs)
* We can call LLMs for embedding text and generating text.
* We can embed text using LanceDB.
* We can embed queries and perform vector search.

### Limitations

* We do not perform chunking, so OpenAI embeddings are unlikely to work. Use Voyage AI or Cohere in the meantime, both of which perform truncation.
* The equation parsing currently leaves a lot to be desired. This part is particularly under active work, but this is the most likely place LLMs will make mistakes due to the PDF parsing not being particularly great for this yet.
* There are several other feature requests/bugs currently being tracked in [Issues](https://github.com/zotero-rag/zotero-rag/issues).

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

# Install the CLI
cargo install --path zqa
```

This will install the `zqa` binary in `~/.cargo/bin/`, so you should make sure this is in your `$PATH`. If you prefer, you can change this location to be, for example, `/usr/local/bin`, like so:

```
cargo install --path zqa --root /usr/local/bin
```

Note that wherever you run `zqa` later, the program will look for API keys and your choices of models in a `.env` file. A future release will make these configurable globally.

### Note for building on Linux

On Linux, the project is configured to use the `mold` linker for faster linking. You can either install `mold` by following the [repo's instructions](https://github.com/rui314/mold) or simply remove the `rustflags` line from `.cargo/config.toml`.

## Usage

### Set up environment variables

Create a `.env` in the root of the project with the following structure:
```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MODEL=o4-mini-2025-04-16
VOYAGE_AI_API_KEY=
VOYAGE_AI_RERANK_MODEL=rerank-2.5
COHERE_API_KEY=
COHERE_RERANK_MODEL=rerank-v3.5
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-pro
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
OPENROUTER_MODEL=
OPENROUTER_API_KEY=

MAX_CONCURRENT_REQUESTS=5
```

The model choices and the maximum concurrent requests above are defaults, and you can omit them. By default, the app uses Anthropic for generation and Voyage AI for embedding. This is currently the recommended settings (and so you won't need an OpenAI API key set up). Note that Voyage AI embeddings are the only ones that are known to work in a real setting; it is quite unlikely for the OpenAI embeddings to work at this moment, because we do not perform any chunking. For `OPENAI_MODEL`, o4-mini is recommended, unless you have a Usage Tier 3 account--in which case you may prefer gpt-4.1.

You likely don't need _all_ of these set (or even mentioned) in your `.env` file: add just the ones you need, and defaults will be used for all the others. The exception to this is if you're _contributing_--in which case you will need _all_ of them set so that tests can run locally. Because the LLM space changes often, most tests are written without mocking so that any breaking changes caused by APIs changing is caught early.

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

## License

MIT License - See [LICENSE](LICENSE) for details.
