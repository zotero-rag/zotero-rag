# Project Overview

This is a Rust-based Zotero RAG QA System for answering questions from academic libraries.

## Project Structure

- **pdftools**: PDF parsing and text extraction utilities
- **rag**: Core RAG implementation with vector database (LanceDB) and LLM clients (OpenAI, Anthropic)
- **zqa**: Command-line interface and query processing (currently Hello World)

## Development Commands

### Building and Testing

```bash
# Build entire workspace
cargo build

# Build release version
cargo build --release

# Run tests
cargo test

# Run specific crate
cargo run --bin zqa
```

### Environment Setup

- Requires API keys for OpenAI and Anthropic
- Use `.env` file for environment variables (dotenv crate is included)

## Coding Standards

### Rust Conventions

- Use standard Rust formatting: `cargo fmt`
- Follow Rust naming conventions (snake_case for functions/variables, PascalCase for types)
- Use 4-space indentation (Rust standard)
- Prefer explicit error handling with `Result<T, E>`
- As much as possible, use idiomatic Rust patterns. `cargo clippy` is helpful here

### Project-Specific Patterns

- Error types defined in respective modules (e.g., `rag/src/llm/errors.rs`)
    - Prefer to inline error definitions if there are only a few, but if there are many kinds of errors, use a dedicated file like the above.
    - In general, prefer to handle errors explicitly as in idiomatic Rust. Errors from external sources should be wrapped appropriately and propagated until they are handled.
- Factory pattern for LLM clients (`rag/src/llm/factory.rs`)
- Trait-based design for extensibility (base traits in `rag/src/llm/base.rs`)
- In general, functions should have documentation above them. This does not need to be done for trait implementations, if the trait is standard in Rust (e.g., `From<...>`, `Copy`, etc.).
- In general, the library crates `pdftools` and `rag` should not have side-effects such as printing to `stdout`, _unless_ that side-effect provides useful information to the user (e.g., warnings, specific error messages, etc.).
- Although `cargo clippy` is automatically run and will block PR merging, you should also perform checks for idiomatic Rust, especially for code that reimplements functions that are built-in. However, if the user notes, or you believe, that Clippy marked that instance as okay, this is fine, and Clippy's ruling should be followed.

## PR Review

- Assess that the PR code follows idiomatic Rust and the coding standards set above.
- In general, bias for efficiency. However, there may be cases where some efficiency is traded off for readability or better UX; but this should be limited.
- PRs should, generally speaking, contain tests for the code they add. This should be exempted in very limited situations where there is a good reason.
- Minimize the use of emojis unless you need to strongly emphasize something; use standard Markdown instead.

## Architecture Notes

### Dependencies

- **lancedb**: Vector database for embeddings
- **reqwest**: HTTP client for API calls
- **serde/serde_json**: JSON serialization
- **tokio**: Async runtime
- **arrow-schema/arrow-array**: Data schema handling

### Current Status

- PDF text extraction working
- LLM clients (OpenAI/Anthropic) implemented
- Vector embeddings with LanceDB functional
- Components not yet integrated in `zqa` crate

## Important Files

- `rag/src/llm/`: LLM client implementations
- `rag/src/vector/`: Vector database operations
- `pdftools/src/`: PDF parsing utilities
- `zqa/src/`: CLI interface (work in progress)

## Testing Notes

- Integration tests in `zqa/tests/`--currently disabled since we need a Zotero mocker.
- Use `cargo test` to run available tests. If you are debugging, use `RUST_BACKTRACE=1`.
