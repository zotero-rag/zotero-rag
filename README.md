# Zotero RAG QA System

A Rust-based system for answering questions from your Zotero library using Retrieval-Augmented Generation (RAG).

## Overview

This project provides a command-line interface for querying your Zotero library with natural language questions. It uses RAG to search and retrieve relevant information from your academic papers, then generates answers grounded in your library content.

### Features

* Extracts text from PDFs and ignores tables/figures (they're unlikely to have useful context for LLMs)
* Support for OpenAI, Anthropic, Gemini, and OpenRouter models for text generation, and Cohere and Voyage AI models for embedding (technically, OpenAI models are supported, but are unlikely to work as of now, see Limitations below).
* Locally-stored embeddings using LanceDB.
* Search-only mode retrieves papers

### Limitations

* We do not perform chunking, so OpenAI embeddings are unlikely to work. Use Voyage AI or Cohere in the meantime, both of which perform truncation. In testing, truncation did not change the quality of the generated output.
* The equation parsing currently leaves a lot to be desired. Specifically, the most likely cases to fail involve equations that involve the `bmatrix`, `cases`, and `align` environments. This part is particularly under active work, but this is the most likely place LLMs will make mistakes due to the PDF parsing not being particularly great for this yet. However, Sonnet 4+ models in particular seem to be relatively decent at this.
* There are several other feature requests/bugs currently being tracked in [Issues](https://github.com/zotero-rag/zotero-rag/issues). Note that internally, issues are prioritized and stories are planned using [Linear](https://linear.app), so some metadata such as the priorities may not be up-to-date on GitHub Issues.

## Installation

The most straightforward way to install this is using `cargo` directly:

```bash
cargo install --git https://github.com/zotero-rag/zotero-rag --tag v0.1.0-beta.1
```

It should be safe to omit the `--tag` and install the `HEAD` version, if you prefer. Contributors might prefer the below version instead, since you'll already have run a `git clone`:

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

## Usage

When you first run the CLI, you will be guided through setting up a config. This config file (see Configuration below for more details) contains details such as your preferred model provider and API keys. 

Although you *could* store your API keys in this config, I recommend you don't. Instead, leave the API keys in the config file blank, and use environment variables or a `.env` file where you're working. The reason is that this is stored in `~/.config/zqa`, so if you have a tool like GNU Stow managing your dotfiles, it could pick it up accidentally (technically you'd have to set it up to do this, but I digress), and you could inadvertently commit your keys to GitHub! For example, [my config](https://github.com/yrahul3910/dotfiles/blob/master/.config/zqa/config.toml) leaves placeholders for the API keys, and I have `.env` files in the directories where I actually use this. Now, I can safely use `stow` to keep my config synced across machines. If you insist on having your API keys in your config, however, I strongly encourage that you add `zqa/config.toml` to a `.gitignore`, just in case.

Once you have a config set up, you should first run `/process`, which will use your embedding provider to create a vector database. **This step takes a long time!** On my machine, with about 1100 papers in my Zotero library, this took about 4 hours. This is mostly to respect the API's rate limits; the PDF parsing itself took about 40 minutes (though if you're using a release build, this will likely be much faster). I recommend leaving this running in the background, possibly while you're sleeping. It's possible that some PDFs fail to process when you're back; this is fine, and you can re-run `/process` at any time to handle failed/new Zotero entries.

The easiest way to get started is to run `/help`. This gives you a list of things you can do. If you only need to use this to ask questions, you simply type in your question and hit Enter. It's likely you might want to do other things as well; for example, `/search <query>` gives you papers in your library that are most relevant, with no further processing. This is particularly useful if you know some relevant keywords or ideas used by that paper. This also takes about 3 seconds and is *very* cheap, so it's an appealing option.

The CLI has a `readline` implementation, so it respects your `.inputrc`! You can use

```
set editing-mode vi
```

in your `~/.inputrc` to use vim motions (the default is emacs bindings).

It is unlikely that you will run into a scenario where everything seems broken, but you can run `/checkhealth` to run health checks on your LanceDB database and `/doctor` to attempt to provide suggestions (but note that this does not actually apply the suggested fixes).

## Security

You should know that this comes with no warranty and I assume no liability if this bankrupts you, burns your house down, etc. etc. I can, however, give you reasonable security precautions that I use (in descending order of importance):

* **Set spending limits on your API keys.** If the provider does not have this option, set up email alerts for costs (and *please*, read your email; an alert mechanism is only as useful as the attention you pay to it). I have mine set to about $15 for Anthropic and $5 for Voyage AI.
* **Turn off automatic top-up.** This makes denial-of-wallet attacks harder.

# Developers Developers Developers

## Project Structure

The project is organized into three Rust crates:

- **zqa-pdftools**: PDF parsing and text extraction
- **zqa-rag**: Core RAG implementation with vector database and LLM clients
- **zqa**: Command-line interface and query processing

## Requirements

- Rust (2024 edition or higher)
- Zotero with a local library
- API key for OpenAI and Anthropic (for LLM capabilities), and Voyage AI (for embeddings)

### Note for building on Linux

On Linux, the project is configured to use the `mold` linker for faster linking. You can either install `mold` by following the [repo's instructions](https://github.com/rui314/mold) or simply remove the `rustflags` line from `.cargo/config.toml`.

## Contributing

If you wish to contribute, comment on an issue you want to work on (or create an issue and then comment on that), and open a PR. The [CONTRIBUTING.md](./CONTRIBUTING.md) file has some basic troubleshooting steps, especially for issues that I've faced repeatedly. Make sure you run `cargo clippy --all-targets --all-features` and `cargo fmt --all` (this also runs in CI).

A good first issue is one that interests you as a user. The best issues to work on (in open-source in general) are ones that you personally have with the project. If you're stuck or need help getting started, just comment on the issue you want to work on, and I'll give you pointers. This project being entirely in Rust, however, I will assume you have some experience with it.

## Configuration

The TOML config goes in `~/.config/zqa/config.toml`, and has the following structure:

```toml
model_provider = "anthropic"  # Generation model provider
embedding_provider = "voyageai"  # Embedding/reranker model provider
reranker_provider = "voyageai"  # Usually this will be the same as your `embedding_provider`
max_concurrent_requests = 5  # Max concurrent embedding requests
max_retries = 3  # Max retries when network requests fail

# `log_level` is a CLI-only arg so it isn't applied inadvertently.

# Provider-specific configs. This allows you to merely change the `model_provider`
# above and have the settings for that provider applied.
[anthropic]
model = "claude-sonnet-4-5"
api_key = "sk-ant-..."
max_tokens = 64000

[openai]
model = "gpt-5.2"
api_key = "sk-proj-..."
max_tokens = 8192
embedding_model = "text-embedding-3-small"
embedding_dims = 1536

[gemini]
model = "gemini-2.5-pro"
api_key = "AI..."
embedding_model = "gemini-embedding-001"
embedding_dims = 3072

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
```

### Set up environment variables

Create a `.env` in the root of the project with the following structure:

```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-5
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MODEL=o4-mini-2025-04-16
VOYAGE_AI_API_KEY=
VOYAGE_AI_MODEL=voyage-3-large
VOYAGE_AI_RERANK_MODEL=rerank-2.5
COHERE_API_KEY=
COHERE_MODEL=
COHERE_RERANKER=rerank-v3.5
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-pro
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
OPENROUTER_MODEL=
OPENROUTER_API_KEY=

MAX_CONCURRENT_REQUESTS=5
```

The model choices and the maximum concurrent requests above are defaults, and you can omit them. By default, the app uses Anthropic for generation and Voyage AI for embedding. This is currently the recommended settings (and so you won't need an OpenAI API key set up). Note that Voyage AI embeddings are the only ones that are known to work in a real setting; it is quite unlikely for the OpenAI embeddings to work at this moment, because we do not perform any chunking. Cohere embeddings should also work.

You likely don't need _all_ of these set (or even mentioned) in your `.env` file: add just the ones you need, and defaults will be used for all the others. The exception to this is if you're _contributing_--in which case you will need _all_ of them set so that tests can run locally. Because the LLM space changes often, most tests are written without mocking so that any breaking changes caused by APIs changing is caught early.

## Running the program

You can run the code using

```bash
cargo run --bin zqa
```

The binary exposes some options, but most of them are unlikely to be useful at the moment for end-users. The one that you might want to play with is `--model-provider`, which lets you choose the model provider. Note that this must be set for the model settings in your `.env` to have any effect.

## Testing

To run unit tests, a simple `cargo test` should suffice. Note that some tests are configured to not run in CI by checking for the `CI` environment variable. Similarly, integration tests do not run by default, and are enabled via the `INTEGRATION_TESTS` environment variable.

# Reporting issues

To report a bug, please re-run using `--log-level debug` and add the logs to your issue.

# FAQ

**Does this support vim motions?** Yes, obviously. Set up your `.inputrc` accordingly (on macOS, the program will also respect `.editrc`, and this takes precedence over `.inputrc`, which acts as a fallback).

**Will you make a GUI?** No, but there is an [open issue](https://github.com/zotero-rag/zotero-rag/issues/61) for a headless mode that will let you, or anyone else develop a UI easier (but this is a *very* low priority at the moment). However, this project is open-source, so feel free to build on top of it. Might I recommend [Dioxus](https://github.com/dioxuslabs/dioxus)?

**I'm getting a 400 Forbidden!** Check that you set an API key in your `config.toml`, `.env`, or in your environment variables. Then, check that the API key has access to the API and is valid. Finally, check that your account has credits loaded.

**What does the `--tui` option do?** Right now, nothing; it's disabled. A TUI is planned, but it's relatively low on priorities.

**How do I write a config file for this?** Read [the docs](./README.md).

**Is this blazing fast?** Of course, it's written entirely in Rust. Here are your obligatory rocket emojis 🚀🚀

# License

MIT License - See [LICENSE](LICENSE) for details.
