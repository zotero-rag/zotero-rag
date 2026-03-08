# Design: `#[tool]` Attribute Macro

**Author(s):** Rahul Yedida + Claude Code
**Date:** 2026-03-07
**Status:** Draft

## Glossary

- **Attribute macro:** A Rust proc-macro invoked as `#[name(...)]` that receives and can transform the annotated item's token stream.
- **Derive macro:** A Rust proc-macro invoked via `#[derive(Name)]` that only appends generated code without modifying the input.
- **Token stream:** The raw representation of Rust source code that proc-macros operate on.

## Context and Scope

`zqa` supports tool calling across multiple LLM providers (Anthropic, OpenAI, Gemini). Each tool implements the `Tool` trait from `zqa-rag`, which has five methods: `name`, `description`, `parameters`, `schema_key`, and `call`. The first four are pure boilerplate derived from static properties of the struct — only `call` contains meaningful business logic.

As the number of tools grows, manually implementing the same four boilerplate methods on every tool becomes error-prone and noisy. This doc describes a `#[tool]` attribute macro in `zqa-macros` that generates those four methods automatically.

### Goals and Non-Goals

* **Goals:**
  * Eliminate boilerplate for `name`, `description`, `parameters`, and `schema_key` on every `Tool` impl.
  * Keep the call site as minimal as possible — one annotation, no extra derives.
  * Enforce that tools always have a description (doc comment) at compile time.

* **Non-Goals:**
  * Generating the `call` method — this is custom async logic and cannot be derived.
  * Supporting tools without a `schema_key` field — all tools must support multiple providers at runtime, so this field is always required.
  * Adding any dependency from `zqa-macros` to `zqa-rag` or other domain crates.

## Design

The `#[tool]` attribute macro is added to a tool struct. It reads the struct's doc comment and ident, accepts the input type and schema key field name as macro arguments, and emits the original struct definition followed by a partial `impl Tool` block covering the four boilerplate methods. The user then writes a separate `impl Tool` block containing only `call`.

```
zqa-macros   ->  depends only on syn, quote, proc-macro2
zqa-rag      ->  no dependency on zqa-macros
zqa          ->  depends on both; macro expands here, all paths resolve
```

All paths in generated code are fully qualified (e.g. `::schemars::schema_for!`, `::zqa_rag::llm::tools::Tool`) so that `zqa-macros` requires no knowledge of those crates' APIs. This follows the design of built-in macros such as `#[derive(Debug)]`, which is in `rustc`, but emits references to `::core::fmt::Debug` and `::core::fmt::Formatter`.

### APIs

The macro is invoked as:

```rust
/// Description of the tool, used as the LLM-facing description.
#[tool(input = MyToolInput, schema_key_field = schema_key)]
pub struct MyTool {
    // ...
    pub schema_key: String,
}
```

The macro arguments are:

- `input` — the type implementing `JsonSchema` and `Deserialize` that describes the tool's parameters. Required.
- `schema_key_field` — the name of the field on the struct that holds the provider-specific schema key string. Required.

As is done currently, the `schema_key_field` can be passed one of `OPENAI_SCHEMA_KEY`, `ANTHROPIC_SCHEMA_KEY`, or `GEMINI_SCHEMA_KEY` from `zqa_rag::llm::tools`.

The macro derives:

- `name` — the struct ident converted to snake_case (e.g. `RetrievalTool` → `"retrieval_tool"`).
- `description` — the concatenated text of all `#[doc = "..."]` attributes on the struct. A missing doc comment is a compile error.
- `parameters` — `::schemars::schema_for!(#input)`.
- `schema_key` — `self.#schema_key_field.clone()`.

The user still writes an `impl Tool` block, but only for `call`:

```rust
impl Tool for MyTool {
    fn call(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> {
        // tool logic
    }
}
```

## Rejected Alternatives

### `#[derive(Tool)]` with a separate `ToolCall` trait

A derive macro could generate the four boilerplate methods and delegate `call` to a separate `ToolCall` trait that the user implements. This cleanly separates metadata from behavior.

Rejected because it splits what is conceptually one trait into two, complicates the `Tool` trait boundary (callers would need to bound on both), and requires a breaking change to `zqa-rag`. The attribute macro achieves the same DX improvement without touching the existing trait.

### `macro_rules! impl_tool!` declarative macro

A `macro_rules!` macro accepting name/description/input/schema_key literals and a closure for `call` could cover the same ground without proc-macro infrastructure.

Rejected because passing an async closure through a declarative macro is syntactically awkward and produces confusing borrow checker errors. The attribute macro is more readable at the call site and produces better error messages.

### `#[derive(Tool)]` without a trait split

A derive macro that generates all four boilerplate methods but leaves `call` to a manual `impl Tool` block. This works but requires two annotations (`#[derive(Tool)]` and `#[tool(...)]`) — one for the derive and one to pass arguments like `input` and `schema_key_field`. The attribute macro achieves the same result with a single annotation.
