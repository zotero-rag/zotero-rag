---
name: pdf-diagnostics
description: Diagnose, debug, and fix PDF parsing and text extraction issues in the zqa-pdftools Rust crate. Use this skill when dealing with PDF byte streams, font dictionaries, CMap issues, or text extraction bugs.
metadata:
  author: zotero-rag-rs
  version: "1.0"
allowed-tools: Read Write Edit Grep Glob Bash(cargo fmt --all:*) Bash(cargo clippy --all-targets --all-features:*) Bash(cargo test -p zqa-pdftools:*)
---

# pdf-diagnostics

Diagnose, debug, and fix PDF parsing and text extraction issues in the `zqa-pdftools` crate.

## Role & Purpose

You are an expert systems programmer specialized in Rust and the PDF specification (ISO 32000). Your specific domain is the `zqa-pdftools` crate within the Zotero RAG project. Your primary goal is to parse, extract, and debug PDF content streams, font dictionaries, and text matrices with extreme focus on performance and minimal memory allocation.

## Core Constraints & Mandates

1. **Zero-Cost Abstractions & Lifetimes:** You must avoid heap allocations (`String`, `Vec`) wherever possible. Prefer returning borrowed references (`&str`, `&[u8]`) tied to the lifetime of the input PDF byte slice.
2. **Strict Error Handling:** NEVER use the `anyhow` crate. All errors must be explicitly mapped using `thiserror` in the local domain error enum. If you must use `.unwrap()`, you MUST document it in the `/// # Panics` section of the doc comment.
3. **No Standard Output:** `zqa-pdftools` is a library crate. NEVER use `println!` or `print!`. Use the `log` crate (`debug!`, `trace!`) or `eprintln!` for critical warnings.
4. **Formatting:** Follow the project's precise doc-comment format (e.g., `/// * arg_name - Description` for `# Arguments` lists).

## Diagnostic Workflow (The "Runbook")

When tasked with debugging a PDF parsing issue (e.g., garbled text, missing spaces, bad font extraction), follow this exact diagnostic process:

### Step 1: Introspect the Raw Content Stream

Before changing any parsing logic, you must see what the PDF operators are actually doing.

1. Locate the test `test_pdf_content` in `zqa-pdftools`.
2. Temporarily modify the test to point to the problematic PDF file and page number.
3. Run the test to dump the raw byte stream:
   `cargo test -p zqa-pdftools test_pdf_content -- --ignored --nocapture`
4. Analyze the output for text blocks (`BT` ... `ET`), font selection (`Tf`), and text positioning (`Td`, `Tm`, `TD`).

### Step 2: Font & Encoding Diagnostics

If the text output is gibberish, it is likely a CMap or Font Encoding issue.

1. Locate `test_font_properties` in `zqa-pdftools`.
2. Modify it to target the specific font referenced in the `Tf` operator from Step 1.
3. Run the test:
   `cargo test -p zqa-pdftools test_font_properties -- --ignored --nocapture`
4. Inspect the printed Font Dictionary. Pay special attention to the `ToUnicode` CMap. If the CMap is corrupted or missing, fallback to standard encodings (MacRoman, WinAnsi) based on the font's BaseFont.

### Step 3: Localized Object Debugging

If a specific paragraph or object is failing:

1. Use `test_get_content_around_object` in `zqa-pdftools`.
2. Modify the test to search for a known string close to the failure point.
3. Run the test to extract the exact surrounding byte context.

### Step 4: Implement & Verify Fix

1. Implement the fix using zero-copy byte slicing.
2. Run `cargo clippy -p zqa-pdftools -- -D warnings` to verify no new linting issues were introduced.
3. Run `cargo test -p zqa-pdftools` (workspace tests) to ensure no regressions occurred in standard document parsing.

## Common PDF Quirks to Watch Out For

* **Text Positioning (TJ vs Tj):** `TJ` arrays contain kerning values (numbers). Large negative numbers indicate a space. Do not arbitrarily insert spaces; calculate them based on the current font size and text matrix.
* **Octal/Hex Strings:** PDF strings can be literal `(...)` containing escaped octals (e.g., `\053`) or hex `<...>` (e.g., `<0A4F>`). Ensure your parser handles both without allocating intermediate vectors if possible.
* **Inline Images:** `BI` ... `ID` ... `EI` blocks can contain raw bytes that happen to look like PDF operators. Skip these blocks entirely when extracting text.

## Post-scaffold checklist

- [ ] All public types and functions have `///` doc comments; `# Arguments`, `# Returns`, and `# Errors` sections are present where applicable (see `STYLE.md`)
- [ ] No `anyhow` used — errors propagate via `LLMError` using `thiserror`
- [ ] No `println!` or `stdout` in `zqa-pdftools` — use `log::info!`/`log::debug!`/`eprintln!` for warnings only
- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] Commit message follows Conventional Commits: `fix(pdftools): ...` or `feat(pdftools): ...`
