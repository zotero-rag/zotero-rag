---
type: Supporting Crates
title: zqa Test Macros
description: Test-support crates that provide diagnostic assertions and retrying asynchronous test helpers.
tags: [testing, macros, rust]
timestamp: 2026-07-12T19:56:20-07:00
---

# zqa-macros

`zqa-macros` exports declarative test assertions with more useful failure
output than their minimal equivalents:

| Macro | Check |
| --- | --- |
| `test_eq!` | Equality, printing received and expected values on failure. |
| `test_contains!` | Membership in a container that supports `contains`. |
| `test_contains_all!` | Membership of every value in an iterable. |
| `test_ok!` | That a `Result` is `Ok`, including the error in a failing message. |

The macros borrow their operands before checking them, which avoids evaluating
an expression twice.

# zqa-macros-proc

`zqa-macros-proc` currently exports `#[retry(n)]` for asynchronous tests. It
requires a positive retry count and an `async fn`; each attempt catches a panic
and retries until the final attempt, which rethrows the panic normally.

Place `#[retry(n)]` before `#[tokio::test]` so the retry wrapper is applied
first. The attribute is intended for tests with transient failures, not for
hiding deterministic regressions.

# Usage in the workspace

The macros are used by tests in [zqa-pdftools](/pdf-processing.md) and the
[zqa CLI](/cli.md). They are support crates rather than part of the runtime RAG
pipeline; see [system overview](/system-overview.md) for the application and
library boundaries.
