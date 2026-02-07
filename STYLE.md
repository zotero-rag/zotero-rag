# Style Guide

## Formatting

- 4-space indentation, rustfmt defaults.
- Wrap long lines when it improves readability; doc comments often wrap.
- `use` blocks are grouped with blank lines separating external crates, std, and local modules when it reads better.

## Modules and file headers

- Module-level docs use `//!` at the top of the file describing the module's purpose.
- Public modules and re-exports are organized cleanly; `mod.rs` files provide structure.

## Naming and visibility

- `snake_case` for functions/variables, `PascalCase` for types/enums/traits, `SCREAMING_SNAKE_CASE` for constants.
- Public API is explicit; public structs/enums have documented fields.
- Avoid abbreviations unless standard (e.g., `LLM`, `API`).

## Docs and comments

- Public types and functions have `///` docs.
- Function docs follow this pattern:
  - Summary line.
  - Optional paragraph for context.
  - `# Arguments`, `# Returns`, `# Errors`, `# Panics`, `# Safety` sections as appropriate.
- Section formatting:
  - Insert a blank `///` line after each section header (`# Arguments`, `# Returns`, `# Errors`, `# Panics`, `# Safety`).
  - `# Arguments`,  `# Panics`, and `# Errors` are always bulleted lists, even with a single item.
  - `# Returns` is a sentence for a single value; use bullets only if returning multiple values (e.g., tuples).
  - Add `# Panics` when the function can panic (e.g., uses `unwrap()`).
  - Add `# Safety` for any `unsafe` usage.

- `# Arguments` bullet items should follow the format: 
```rs
/// * `arg_name` - Description
```

- Bulleted wrapping alignment:
  - Continuation lines align with the text, not the bullet:

```rs
/// * some long bulleted item...
///   start here, not at the bullet.
```

## Interfaces

- Interfaces should be flexible, unsurprising, obvious, and constrained (see "Rust for Rustaceans").

## Types and derivations

- Common derives: `Debug`, `Clone`, `PartialEq`, `Serialize`, `Deserialize`, `Default` where useful.
- Enums for configuration and provider selection; wrapper enums for errors.

## Error handling

- `anyhow` is disallowed.
- Central error enums per domain (e.g., `LLMError`) using `thiserror::Error`.
- Error variants represent common classes of errors.
- Prefer `#[from]` for conversion where possible. If multiple error types map to one variant,
  use explicit `impl From<..>` blocks.

## Traits and APIs

- Traits for extensibility; generic over client/request/response where needed.
- Async traits are allowed with `#[allow(async_fn_in_trait)]` when necessary.

## Clippy and lints

- Clippy allowances are specific and minimal (e.g., `too_many_arguments`, `too_many_lines` for complex functions).

## Logging and side effects

- Library crates generally avoid stdout, but `log` is used for progress/debug info and `eprintln!` for notable warnings.
- CLI prints are in `zqa`, not in `zqa-rag` or `zqa-pdftools`.

## Concurrency and synchronization

- Prefer channels over `Arc<Mutex<...>>` when a channel is simple enough; this is a preference,
  not a strict ban.
- Prefer atomics for shared primitive types when an `Atomic*` exists (e.g., `AtomicUsize`,
  `AtomicBool`) over `Arc<Mutex<...>>`.

## Ownership and allocation

- Avoid cloning where possible; prefer borrowing.
- In `zqa-pdftools`, avoid heap allocation as much as possible.

## Pull requests

- PR titles must be Conventional Commit format (`feat(...)`, `fix(...)`, `perf(...)`, etc.).
- Descriptions can be blank; if present, keep them minimal and factual, avoiding exaggeration
  (e.g., "production grade").
- Performance PRs:
  - Use `perf` as the Conventional Commit type.
  - Must include benchmark results.
  - Should focus entirely on performance (bug fixes are okay).
  - In general, LLMs and coding assistants should avoid making `perf` PRs.
  - This guidance is mostly restricted to `zqa-pdftools`; it does not apply to the other crates.
- In `zqa-pdftools`, three convenience functions are provided as tests:
  - `test_get_content_around_object` searches for text on a specific page of a document and gets some text around it.
  - `test_font_properties` gets the font properties for a specific font on a page of a document.
  - `test_pdf_content` gets the content stream of a page (defaults to first).
  In any PR, it is okay to modify these, generally speaking, as long as the comments match the new code. These should remain `#[ignore]`d, however.
