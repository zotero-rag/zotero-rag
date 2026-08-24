# Zotero RAG QA System

A Rust-based system for answering questions from your Zotero library using Retrieval-Augmented Generation (RAG).

## Overview

This project provides a command-line interface for querying your Zotero library with natural language questions. It uses RAG to search and retrieve relevant information from your academic papers, then generates answers grounded in your library content.

### Features

* Extracts text from PDFs and ignores tables/figures (they're unlikely to have useful context for LLMs)
* Support for OpenAI, Ollama, Anthropic, Gemini, and OpenRouter models for text generation, Cohere, Gemini, Voyage AI, and ZeroEntropy models for embedding, and Cohere, Voyage AI, and ZeroEntropy models for reranking.
* Locally-stored embeddings using LanceDB.
* Search-only mode retrieves papers

### Limitations

* The equation parsing currently leaves a lot to be desired. Specifically, the most likely cases to fail involve equations with the `bmatrix`, `cases`, and `align` environments. This part is particularly under active work, but this is the most likely place LLMs will make mistakes due to the PDF parsing not being particularly great for this yet. However, Sonnet 4+ models in particular seem to be relatively decent at this.
* There are several other feature requests/bugs currently being tracked in [Issues](https://github.com/zotero-rag/zotero-rag/issues). Note that internally, issues are prioritized and stories are planned using [Linear](https://linear.app), so some metadata such as the priorities may not be up-to-date on GitHub Issues. This is not to discourage you from opening a GitHub issue yourself; those automatically sync to the Linear board, so I will see them.

## Usage

When you first run the CLI, you will be guided through setting up a config. This config file (see Configuration below for more details) contains details such as your preferred model provider and API keys. 

If you use a program like `stow` to manage your dotfiles, I recommend you either don't store your API keys in the config, or ensure your `~/.config/zqa/config.toml` is in `.gitignore`. If you prefer the former option, leave the API keys in the config file blank, and use environment variables or a `.env` file where you're working. For example, [my config](https://github.com/yrahul3910/dotfiles/blob/master/.config/zqa/config.toml) leaves placeholders for the API keys, and I have `.env` files in the directories where I actually use this. Now, I can safely use `stow` to keep my config synced across machines.

Once you have a config set up, you should first run `/process`, which will use your embedding provider to create a vector database. This step can take a long time! On my machine, with about 1500 papers in my Zotero library, this took about 3 hours when using `ollama` locally for embeddings, and about 40 minutes when using Voyage AI (the default). A lot of the delay with the APIs is mostly to respect their rate limits; the PDF parsing itself took about 3 seconds. It's possible that some PDFs fail to process when you're back; this is fine, and you can re-run `/process` at any time to handle failed/new Zotero entries.

The easiest way to get started is to run `/help`. This gives you a list of things you can do. If you only need to use this to ask questions, you simply type in your question and hit Enter. It's likely you might want to do other things as well; for example, `/search <query>` gives you papers in your library that are most relevant, with no further processing. This is particularly useful if you know some relevant keywords or ideas used by that paper. This also takes about 3 seconds and is *very* cheap, so it's an appealing option.

When asking questions, you can also use @ to select a file in your local directory, and that file will be parsed (but not added to either Zotero or to the vector database). You can then ask questions based on that file, and the model will perform searches to ground its answer.

The CLI has a `readline` implementation, so it respects your `.inputrc`. You can use

```
set editing-mode vi
```

in your `~/.inputrc` to use vim motions (the default is emacs bindings).

It is unlikely that you will run into a scenario where everything seems broken, but you can run `/checkhealth` to run health checks on your LanceDB database and `/doctor` to attempt to provide suggestions (but note that this does not actually apply the suggested fixes).

## Configuration

The TOML config goes in `~/.config/zqa/config.toml`, and has the following structure:

```toml
model_provider = "anthropic"  # Generation model provider
embedding_provider = "voyageai"  # Embedding/reranker model provider
reranker_provider = "voyageai"  # Omit this to disable reranking
max_concurrent_requests = 5  # Max concurrent embedding requests
max_retries = 3  # Max retries when network requests fail
tool_iteration_limit = 15  # Max tool call-processing iterations per user message

# `log_level` is a CLI-only arg so it isn't applied inadvertently.

# Provider-specific configs. This allows you to merely change the `model_provider`
# above and have the settings for that provider applied.
[anthropic]
model = "claude-sonnet-4-5"
model_small = "claude-haiku-4-5"
api_key = "sk-ant-..."
max_tokens = 64000
reasoning_budget = 2048

[ollama]
model = "qwen3.5"
model_small = "qwen3.5:0.8b"
max_tokens = 8192
embedding_model = "qwen3-embedding"
embedding_dims = 4096
reasoning_budget = 2048
base_url = "http://localhost:11434"  # Defaults to local ollama instance

[openai]
model = "gpt-5.2"
model_small = "gpt-5-mini"
api_key = "sk-proj-..."
max_tokens = 8192
embedding_model = "text-embedding-3-small"
embedding_dims = 1536
reasoning_effort = "high"  # Supported by the most models

[gemini]
model = "gemini-3.1-pro-preview"
model_small = "gemini-3-flash-preview"
api_key = "AI..."
embedding_model = "gemini-embedding-001"
embedding_dims = 3072
reasoning_budget = 2048

[voyageai]
reranker = "rerank-2.5"
embedding_model = "voyage-3-large"
embedding_dims = 2048
api_key = "..."

[cohere]
reranker = "rerank-v3.5"
embedding_model = "embed-v4.0"
embedding_dims = 1536
api_key = "..."

[openrouter]
api_key = "..."
model = "anthropic/claude-sonnet-4.5"
model_small = "anthropic/claude-haiku-4.5"
reasoning_effort = "high"
reasoning_budget = 2048
```

### Set up environment variables

The program will use a local `.env` to override values from your `config.toml`. This is best-used for API keys so that you can `stow` your `config.toml` without committing API keys.

```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=
ANTHROPIC_MODEL_SMALL=

OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=
OPENAI_MODEL=
OPENAI_MODEL_SMALL=

VOYAGE_AI_API_KEY=
VOYAGE_AI_MODEL=
VOYAGE_AI_RERANK_MODEL=

COHERE_API_KEY=
COHERE_MODEL=
COHERE_RERANKER=

GEMINI_API_KEY=
GEMINI_MODEL=
GEMINI_MODEL_SMALL=
GEMINI_EMBEDDING_MODEL=

LANCEDB_URI=

OPENROUTER_MODEL=
OPENROUTER_MODEL_SMALL=
OPENROUTER_API_KEY=

OLLAMA_MODEL=
OLLAMA_MODEL_SMALL=
OLLAMA_BASE_URL=
OLLAMA_EMBEDDING_MODEL=
OLLAMA_EMBEDDING_DIMS=

ZEROENTROPY_API_KEY=
ZEROENTROPY_RERANK_MODEL=
ZEROENTROPY_EMBEDDING_MODEL=

MAX_CONCURRENT_REQUESTS=5
```

The model choices and the maximum concurrent requests above are defaults, and you can omit them. By default, the app uses Anthropic for generation and Voyage AI for embedding. This is currently the recommended settings (and so you won't need an OpenAI API key set up). Note that Voyage AI embeddings are the only ones that are known to work in a real setting; it is quite unlikely for the OpenAI embeddings to work at this moment, because we do not perform any chunking. Cohere embeddings should also work.

You likely don't need _all_ of these set (or even mentioned) in your `.env` file: add just the ones you need, and defaults will be used for all the others.

# Reporting issues

To report a bug, please re-run using `--log-level debug` and add the logs to your issue.

