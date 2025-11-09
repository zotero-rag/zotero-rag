# Project Overview

This is a Rust-based Zotero RAG QA System for answering questions from academic libraries.

## Project Structure

- **pdftools**: PDF parsing and text extraction utilities
- **rag**: Core RAG implementation with vector database (LanceDB) and LLM clients (OpenAI, Anthropic)
- **zqa**: Command-line interface and query processing (currently Hello World)

## Development Commands

### Building and Testing

- Build (release): `cargo build --release`
- Run CLI: `cargo run --bin zqa`
- Tests (workspace): `cargo test --workspace`
- Tests (per crate): `cargo test -p rag` (or `-p zqa`, `-p pdftools`)
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- Format: `cargo fmt --all`
- Bench (pdftools): `cargo bench -p pdftools`
- Faster Linux linking: uses `mold` via `.cargo/config.toml` (install or remove the flag).

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
* Documentation comments for functions follow this pattern:
```rs
/// Description
///
/// # Arguments:
///
/// * `some_arg` - Description
///
/// # Returns
///
/// Description
```
In particular, there is no colon after `# Returns`, but there is one after `# Arguments`. The exception to the no-colon rule is for when a return type is a tuple of at least three elements.

## PR Review

- Assess that the PR code follows idiomatic Rust and the coding standards set above.
- In general, bias for performance. However, there may be cases where some efficiency is traded off for readability or better UX; but this should be limited.
- PRs should, generally speaking, contain tests for the code they add. This should be exempted in very limited situations where there is a good reason.
- Minimize the use of emojis unless you need to strongly emphasize something; use standard Markdown instead.
- Do not leave inline comments unless you have specific recommendations for improvements.
- Do not leave inline comments to state that something has improved or is better than before.
- Keep your overall comment concise. In a paragraph or two, describe the overall PR quality and the recommendations in your comments.
- If an inline comment you leave is pedantic or otherwise minor, prefix it with "nit: ", and keep it short, about one sentence.

## Important Files

- `rag/src/llm/`: LLM client implementations
- `rag/src/embedding/`: Embedding client implementations
- `rag/src/vector/`: Vector database operations
- `pdftools/src/`: PDF parsing utilities
- `zqa/src/`: CLI interface (work in progress)

## Testing Notes

- Integration tests in `zqa/tests/`--currently disabled
- Use `cargo test` to run available tests. If you are debugging, use `RUST_BACKTRACE=1`.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits, e.g.: `feat(rag): add checkhealth`, `fix(ci): ...`, `perf(pdftools): ...`, `lint: cargo clippy`.
- PRs should include: concise description, rationale, linked issues, and test notes/output (use `--log-level debug` when relevant).
- Ensure `cargo fmt`, `cargo clippy`, and `cargo test --workspace` pass.

## Security & Configuration Tips

- Secrets: configure via `.env` (see `.env.tmpl`); never commit real keys.
- Recommended defaults: Anthropic for generation, Voyage AI for embeddings.
- Data location: LanceDB under `data/lancedb-table/`. Remove the folder to reset the index.
