---
type: Crate
title: zqa-pdftools
description: The academic-PDF parser that extracts structured text, detects sections, and produces retrieval chunks.
tags: [pdf, parsing, chunking]
timestamp: 2026-07-12T19:56:20-07:00
---

# Purpose

`zqa-pdftools` extracts text from academic PDFs for the retrieval pipeline. Its
parser is tuned for papers: it skips images and uses heuristics to remove table
content, recognizes common Computer Modern math fonts, and records
font-derived section boundaries.

# Extraction pipeline

| Stage | Responsibility |
| --- | --- |
| Load pages | `extract_text` loads the PDF with `lopdf` and processes each page. |
| Tokenize content streams | `tokenizer` recognizes PDF literals, hex strings, names, numbers, and operators. |
| Decode text | `parse` handles text operators and spacing; `fonts` resolves simple and CID-keyed encodings, including ToUnicode CMaps. |
| Repair and classify | Edits repair ligatures and script markers; font size and boldness identify section boundaries. |
| Produce content | `ExtractedContent` returns text, section boundaries, and page count. |

For supported Computer Modern fonts, math mappings convert selected symbols to
LaTeX. Section levels are inferred from font sizes, and only the first four
levels are retained.

# Chunking

`Chunker` turns `ExtractedContent` into `DocumentChunk` values with a
one-based chunk ID, total chunk count, text, byte range, page range, and the
strategy used. Two strategies are available:

| Strategy | Behavior |
| --- | --- |
| `WholeDocument` | Keep the complete extracted document in one chunk. |
| `SectionBased(max_chars)` | Preserve detected section boundaries where possible and split text to the maximum character budget. |

Embedding providers can recommend a strategy, but callers remain free to
choose one. [The CLI](/cli.md) uses these chunks when it processes Zotero and
session-imported PDFs for [zqa-rag](/rag.md).

# Heuristics and limits

Table detection relies on text-position operators and alignment thresholds, so
unusual tables or multi-column layouts can be misclassified. Font and CMap
coverage varies by PDF; CID-keyed fonts without usable ToUnicode data cannot be
decoded reliably. Spacing, superscripts, and subscripts are heuristic, and
complex mathematical environments such as `bmatrix`, `cases`, and `align` can
extract poorly.

# Diagnostics

Ignored debugging tests in `parse.rs` print raw page content, inspect font
properties and CMaps, or show content around anchor text. Run one with:

```sh
cargo test -p zqa-pdftools <test_name> -- --ignored --nocapture
```

The crate also has Criterion benchmarks for parsing throughput. See
[development workflow](/development.md) for the standard benchmark command and
[macros](/macros.md) for test helpers used by the parser tests.
