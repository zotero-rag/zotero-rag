# Plan: ZOT-106 — Ignore rotated text in PDF parser

**Linear**: [ZOT-106](https://linear.app/zotero-rag/issue/ZOT-106/ignore-rotated-text-in-pdf-parser)
**GitHub**: [#113](https://github.com/zotero-rag/zotero-rag/issues/113)
**Milestone**: v0.2.0b

**Author:** Rahul Yedida + GLM-5.2 (pi)

## Summary

The `zqa-pdftools` parser currently extracts all text from a PDF regardless of how it is rendered. When a page contains rotated text (arXiv sidebar IDs, watermarks, image labels, rotated captions), that text is mixed in with the main body and pollutes the RAG index. This issue is to detect rotated text at parse time and drop those `BT..ET` blocks.

The parser already walks every token in `parse_content` (`zqa-pdftools/src/parse.rs`, ~line 1000) and recognizes `BT`, `ET`, `TJ`, `Tf`, `Td`. It does **not** recognize `cm`, `Tm`, `q`, or `Q`, so the current transformation matrix (CTM) and graphics-state stack are completely invisible to it.

## PDF Spec Background

PDF achieves rotated text by applying a non-axis-aligned transform to the text matrix. The relevant operators are:

| Op | Args | Where | Effect |
| - | - | - | - |
| `cm` | `a b c d e f` | Anywhere, often inside `q..Q` | `CTM := CTM · [a b c d e f]` |
| `Tm` | `a b c d e f` | Inside `BT..ET` | `Tm := Tlm := [a b c d e f]` (text + line matrix) |
| `q` | — | Anywhere | Push graphics state (including CTM) |
| `Q` | — | Anywhere | Pop graphics state |

A 2-D transform matrix `[a b c d e f]` is **purely axis-aligned (no rotation, no skew)** iff `b = 0 ∧ c = 0`. A 90° rotation looks like `[0 1 -1 0 x y]`. We will treat `|b|` or `|c|` above a small epsilon as "rotated."

## Approach

Track whether the current text is rotated using a small **graphics-state stack of booleans** on `PdfParser`:

* `q` pushes a copy of the current top onto the stack.
* `Q` pops the stack.
* `cm` and `Tm` each parse 6 numbers and replace the top of the stack with a derived "is rotated" flag based on the matrix's off-diagonal elements.
* On `BT`, snapshot the top-of-stack into a `skip_block: bool` that suppresses `TJ`/`Tj` processing until the matching `ET`.
* On `ET`, clear `skip_block`.

The current `parse_content` already iterates tokens linearly; we just need to add new `Token::Op(...)` arms. We do **not** need a real 3×3 matrix library — the decision reduces to two float comparisons.

### Threshold constant

Add a module-level constant (mirroring the existing `DEFAULT_SAME_WORD_THRESHOLD` style):

```rust
/// Off-diagonal magnitude above which a 2-D transform is considered rotated.
pub const DEFAULT_ROTATION_TOLERANCE: f32 = 1e-3;
```

A pure translation `[1 0 0 1 x y]` is always well below this; a 90° rotation is exactly `1.0`. A 0.1° rotation is `sin(0.1°) ≈ 0.0017`, slightly above; if the test fixture triggers that, lower the tolerance. Keeping it tight avoids false positives from font-rendering noise (which only ever uses `b = c = 0`).

## File-by-File Changes

### 1. `zqa-pdftools/src/parse.rs`

* **Add** `pub const DEFAULT_ROTATION_TOLERANCE: f32 = 1e-3;` near the other `DEFAULT_*` constants (top of file).
* **Extend** `PdfParser` with two new fields:
  * `rotation_stack: Vec<bool>` — the graphics-state stack for "is text rotated?"
  * `skip_block: bool` — true while we are inside a `BT..ET` we want to ignore
  * Initialise both in `PdfParser::new` and `Default`.
* **Extend** the `match` in `parse_content` with three new `Token::Op` arms:
  * `Token::Op(b"cm")` — extract 6 numbers via `get_params_from_tokens::<6>`, set the top of `rotation_stack` to `is_rotated_matrix(a, b, c, d, e, f)`. (If the stack is empty, push the result — this is defensive; under normal PDFs there is always an outer `q..Q`.)
  * `Token::Op(b"Tm")` — same as `cm` but only relevant inside `BT..ET`. Set the top of the stack to the same flag.
  * `Token::Op(b"q")` — `rotation_stack.push(*rotation_stack.last().unwrap_or(&false))`.
  * `Token::Op(b"Q")` — `rotation_stack.pop();` (ignore underflow, mirroring how the parser treats malformed input elsewhere).
* **Modify** the `Token::Op(b"BT")` arm: if `*rotation_stack.last().unwrap_or(&false) || self.skip_block`, set `self.skip_block = true`. This handles both the "rotated since the last `q`" case and the (paranoid) "nested BT inside a rotated BT" case.
* **Modify** the `Token::Op(b"ET")` arm: only run `get_table_bounds` and the `parsed.truncate` rewinds when `!self.skip_block`. Clear `self.skip_block = false` unconditionally on `ET`.
* **Guard** the existing `Token::Op(b"TJ")` arm with `if !self.skip_block`. This is the actual data-suppression point — without it, all the state tracking above is cosmetic.
* **Guard** the existing `Token::Op(b"Tf")` arm similarly so that font-size markers are not recorded for rotated text (otherwise `extract_text` would emit phantom `SectionBoundary` records).
* **Add a small private helper** near the other helpers:

  ```rust
  /// Returns `true` if the 2-D transform `[a b c d e f]` contains rotation or skew.
  fn is_rotated_matrix(a: f32, b: f32, c: f32, d: f32, _e: f32, _f: f32) -> bool {
      b.abs() > DEFAULT_ROTATION_TOLERANCE || c.abs() > DEFAULT_ROTATION_TOLERANCE
  }
  ```

  Note: a non-uniform scale (`a ≠ d`) is *not* rotation; per the issue ("ignore rotated text"), we keep scaled-only text. If false positives appear for `a ≠ d` matrices, a follow-up can add that check.

* **No changes** to `extract_text`, `chunk.rs`, `Edits`, or the tokenizer.

### 2. `zqa-pdftools/assets/rotated.pdf` (new test fixture)

A small PDF that contains:

1. A line of normal body text (e.g., `"Section 1: Introduction"`).
2. A `BT..ET` block with `Tm` whose matrix has `b = 1, c = -1` (90° rotation), containing a distinctive sentinel string (e.g., `"arXiv:2501.00001v1 [cs.CL]"`).
3. A second line of normal body text.

Easiest path to generate: a one-off Python script (not committed) using `reportlab` or a hand-written PDF (PDF 1.4 is small enough to write by hand — roughly 1.5 KB). The script is throwaway and does not need to be checked in. The maintainer is more familiar with `reportlab`; if unavailable in the dev env, write a minimal raw PDF.

A reasonable template (sketch):

```
%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
            /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length ... >>
stream
BT /F1 12 Tf 72 720 Td (Section 1: Introduction) Tj ET
BT 0 1 -1 0 540 720 Tm /F1 10 Tf (arXiv:2501.00001v1 [cs.CL]) Tj ET
BT /F1 12 Tf 72 700 Td (Body text after rotated label) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
```

(Exact contents to be finalized when implementing — the above is just a structural sketch.)

### 3. `zqa-pdftools/src/parse.rs` tests (in the existing `mod tests` block)

* `test_rotated_text_is_ignored` — extracts `assets/rotated.pdf`, asserts the sentinel `arXiv:2501.00001v1` is **not** present, and asserts that `"Section 1: Introduction"` and `"Body text after rotated label"` **are** present. Also asserts the resulting `sections` vec is empty (no font markers recorded for the rotated block).
* `test_is_rotated_matrix` — table-driven test of the helper: pure translation returns `false`; 90°/180°/270°/45°/arbitrary skew return `true`; tiny `b = 1e-4` returns `false` (within tolerance); tiny `b = 1e-2` returns `true`.
* `test_graphics_state_stack_balance` — feeds a hand-tokenized content stream `q 0 1 -1 0 0 0 cm BT (...) Tj ET Q BT (...) Tj ET` and asserts that the first `BT` is skipped and the second is processed. This isolates the `q/Q` push/pop logic from any real PDF.

### 4. No changes elsewhere

`chunk.rs`, `edits.rs`, `tokenizer.rs`, `fonts.rs`, the math modules, and the `zqa-rag`/`zqa` crates do not need to change. The fix is entirely contained in `parse.rs` plus one new asset.

## Edge Cases Considered

* **`cm` outside any `q..Q`**: The current parser sees every operator in a flat stream. If `cm` appears without a wrapping `q..Q`, we still update `rotation_stack[0]` (or push if empty), which is the correct behaviour for "the page-level transform is rotated."
* **Nested `q`**: We push the parent's flag, so a child `cm` cannot accidentally un-rotate a parent.
* **`Tm` resetting rotation mid-BT**: Each `Tm` updates the top of the stack. A `BT` that mixes rotated and non-rotated `Tm`s will be conservatively skipped if the *first* relevant matrix is rotated; in practice, `Tm` appears at most once at the start of a `BT` block, so this is fine.
* **`TJ` arrays**: A `TJ` in a rotated block can span several `Td` and additional `Tm` operators before `ET`. Guarding the whole `BT..ET` (not just the first `Tm`) is the safe choice and matches the issue's "ignore" intent.
* **Non-rotation transforms (mirror, skew)**: Per spec, `b = -c` for pure rotation, but `b ≠ 0` also flags skew and horizontal mirroring. The issue is to "ignore rotated text"; treating skew/horizontal-mirror as "rotated" is acceptable and conservative.
* **Performance**: Adds one `Vec<bool>` push/pop and a few `f32::abs` checks per relevant operator. No allocations on the hot path beyond the stack growth (which is bounded by nesting depth, typically <10). No `String`/`Vec` allocations introduced.

## Verification Checklist

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test --workspace` passes, including the new `test_rotated_text_is_ignored` and the existing `test_real_papers_parse_without_errors` (no regressions on `art2a.pdf`, `curriculum.pdf`, `hypergradient.pdf`, `mono2micro.pdf`, `rnanomaly.pdf`, `deeply.pdf`, `images.pdf`, `subtables.pdf`, `table.pdf`, `hyperlinks.pdf`, `sections.pdf`, `manifold.pdf`, `ntk.pdf`, `symbols.pdf`, `test1.pdf`)
- [ ] Manual check: the seven `test_papers/*.pdf` outputs are byte-identical (or near-identical, allowing for harmless whitespace) before/after, since none of them are expected to have rotation in the body text.
- [ ] `assets/rotated.pdf` extracts cleanly and the rotated sentinel is absent.

## Commit & PR

* **Branch already created**: `ryedida/zot-106-ignore-rotated-text-in-pdf-parser` (per Linear).
* **Commit message**: `fix(pdftools): ignore rotated text in PDF parser`
* **PR title**: `fix(pdftools): ignore rotated text in PDF parser` (Conventional Commits).
* **PR body**: link ZOT-106 and #113, list the files changed, paste the output of `cargo test -p zqa-pdftools` showing both the new tests and the unchanged results on existing fixtures, and confirm `cargo clippy --all-targets --all-features -- -D warnings` is clean.

## Out of Scope

* Tracking non-uniform scale (`a ≠ d`) as a reason to skip text. The current spec only asks about rotation; scale-only transforms are kept.
* Detecting "sideways" text via character width heuristics (e.g., very wide kerning) — the matrix-based check is the right primary signal.
* Refactoring `parse_content` to use a real matrix library. The boolean stack is sufficient and avoids pulling in a dependency.
* Adding a config knob for the rotation threshold. If users later need to tune it, lift `DEFAULT_ROTATION_TOLERANCE` into `PdfParserThresholds`.
