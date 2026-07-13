---
type: Development Guide
title: Development Workflow
description: Workspace layout, local requirements, and commands for building, testing, and linting the project.
tags: [development, rust, ci]
timestamp: 2026-07-12T19:49:45-07:00
---

# Workspace

The Cargo workspace contains five crates:

| Crate | Role |
| --- | --- |
| `zqa` | Command-line application. |
| `zqa-rag` | RAG, provider, and vector-storage library. |
| `zqa-pdftools` | PDF parsing and text-extraction library. |
| `zqa-macros` | Declarative macro crate. |
| `zqa-macros-proc` | Procedural macro crate. |

See [system overview](/system-overview.md) for their runtime relationship and
[macros](/macros.md) for the supporting crates.

# Local requirements

Use Rust 2024 edition or later. A local Zotero library and provider credentials
are needed for end-to-end use; see [runtime configuration](/configuration.md).
On Linux, `.cargo/config.toml` uses the `mold` linker for faster linking. Install
`mold` or remove that linker setting when it is unavailable.

Some CI jobs install `protobuf-compiler` before building. The continuous
integration checks also run formatting, Clippy with warnings denied, and
`cargo-deny` dependency checks.

# Common commands

| Purpose | Command |
| --- | --- |
| Build a release binary | `cargo build --release` |
| Run the CLI | `cargo run --bin zqa` |
| Test the workspace | `cargo test --workspace` |
| Test one crate | `cargo test -p zqa-rag` |
| Format | `cargo fmt --all` |
| Check formatting | `cargo fmt --all -- --check` |
| Lint | `cargo clippy --all-targets --all-features -- -D warnings` |
| Benchmark PDF tools | `cargo bench -p zqa-pdftools` |

# Test notes

Integration tests in `zqa/tests/` are disabled by default and use the
`INTEGRATION_TESTS` environment variable. When debugging PDF behavior, the
ignored PDF diagnostics can be run with `cargo test -p zqa-pdftools <test_name>
-- --ignored --nocapture`.

Do not commit API keys. Keep secrets in `.env` or the user configuration file,
and use `RUST_BACKTRACE=1` when investigating Rust failures.
