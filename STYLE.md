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

# Section for AI Agents (Claude Code, Pi, Codex, etc.)

You are expected to write code at the level of a senior engineer who cares about
the codebase. Optimize for the person who reads this code next, not for finishing
the diff.

## Fit in before you stand out

- **Read the surrounding code first.** Match its naming, structure, error handling, logging, and idioms. Local consistency beats any external "best practice." When in Rome.
- **Follow the project's rules.** If a style guide, `CONTRIBUTING.md`, `.editorconfig`, `biome.json`, `rustfmt.toml`, `pyproject.toml`, etc. exists, obey it. Run the formatter and linter before declaring done; don't hand back code that fails them.
- **Use the repo's established way of doing things.** If there's already a pattern for config, HTTP calls, dates, validation, or DI, use it. Introducing a second way to do the same thing is a regression even if your way is "nicer."

However: if there is a nicer way and the effort to change it is not significant, it's worth bring it up to the user.

## Don't reinvent or duplicate

- **Search before you write.** Before adding a helper, grep for an existing one. Most "utility" functions you're about to write already exist somewhere in the repo or its dependencies.
- **Rule of Three.** Copying once is fine. The third occurrence earns an abstraction. When you extract, generalize it properly--a shared helper with a special-case flag bolted on is worse than two copies.
- **Don't add a dependency** for something the standard library or an existing dependency already does well.
- **Don't reimplement the language or framework.** Reach for the built-in before the hand-rolled loop.

## Earn every abstraction

- **Don't extract a function that's used once** unless it names a genuinely non-obvious step or removes deep nesting. A one-shot helper usually just adds a layer to chase.
- **No pass-through one-liners.** A function that only renames or forwards to another function should be inlined and deleted.
- **A function justifies itself** by one of: reuse, a name that documents non-obvious intent, or a real testing/composition seam. "It's more granular" is not a justification.
- **YAGNI.** No parameters, options, config hooks, or interfaces added "in case we need them later." Build for what's in front of you; generalize when the second real caller appears.

## Be honest with the tools

- **Never silence the type checker or linter to make an error go away.** No `# noqa`, `# type: ignore`, `@ts-ignore`, `// biome-ignore`, blanket `any`, gratuitous casts, or `!` non-null assertions used as escape hatches. Fix the root cause.
- A suppression is acceptable only when the tool is genuinely wrong, it is **narrowly scoped to the single line**, and it carries a comment explaining *why*. This should be rare.
- **Don't weaken types to compile.** Loosening a type to `any`/`object`/`unknown` to get past an error is hiding a bug, not fixing one.
- **Don't make tests pass by deleting, skipping, or weakening them.** If a test is genuinely wrong, say so and explain--don't quietly gut it.

## Code shape smells

- **Data clumps**: always passing `(userId, orgId, accountId)` together → make a struct. Same for `(start, end)`, `(x, y)`, `(key, value)`.
- Don't return different shapes from one function (sometimes a list, sometimes a single item, sometimes `null`). Pick one, or expose two functions.
- Python-specific: avoid using dicts to represent data: new readers have no context what the valid keys are, and it only adds scope for typos. Prefer a `TypedDict`, a Pydantic model, etc.
- Rely on invariants in the code. If an argument is typed `usize`, there is no need for a `if (arg < 0)`. If an argument is typed as an `int`, don't "make sure", believe it, and let the type checker bring up issues. If a function is called after some invariants are checked, don't recheck inside the function. Either avoid the pre-call check, or remove it from the function and document that invariant in the function's docstring.
- For the items in this section, if the current code has these smells, follow the current code style, but bring up the possible refactor to the user if your own code has to use these smells.

## Comments

- **No decorative separators or banners** (`# -------- Section --------`, ASCII art headers). Well-structured code is navigable without them; if a file needs visual dividers, it needs splitting.
- **Comments explain *why*, not *what*.** Don't narrate code that already says what it does. Explain the non-obvious: a tricky invariant, a workaround, a reason for an unusual choice.
- **Match the tone of the codebase.** Read a few existing comments before writing your own. If this codebase's comments are whimsical or full of references, write in that register; if they are strictly design-and-algorithm notes, keep yours dry and technical. Don't impose your own voice on a codebase that has one.
- **Don't reference the pre-refactor state.** Write changed code as if the current version is the only one that ever existed. No "old schema", "previously now uses", "changed from X", "formerly". The reader has no access to what was there before and no reason to care; that context lives in git, not the source.
- **No changelog comments** (`// added X`, `// fixed bug`). Git records history.
- **Delete commented-out code.** It's dead weight; git remembers it.
- No comment that just paraphrases the function name. The name is the comment. Either write a real docstring, or don't write one at all.

## Characters and punctuation

- **Plain ASCII in code, comments, commit messages, and output.** No emojis and no Unicode decoration.
- **Don't use the em-dash character.** Use `-`, or `--` where an em-dash genuinely reads better, and let the editor's font and ligatures handle rendering. Follow the Chicago Manual of Style: no spaces around the em-dash.
- **No other "smart" punctuation** either: use straight quotes, `...` for an ellipsis, and plain hyphens. Let tooling render glyphs; don't paste them in.

## Naming

- Names reveal intent and match the codebase's conventions and casing.
- Avoid abbreviations unless they're already standard in this repo or domain.
- A name that needs a comment to explain what it holds is the wrong name.
- Booleans read as yes/no questions: `isActive`, `hasAccess`, `canEdit`--not `activeFlag`, `access`, `edit`.

## Taste: make special cases disappear

This is the part that separates competent from good.

- **Restructure to eliminate edge cases, don't pile on branches to handle them.** The mark of taste is the special case that vanishes after you pick the right data structure or formulation--not the function that grows another `if` for every input.
- **Get the data model right and the code follows.** Most ugly code is a symptom of the wrong data structure. Fix that first.
- **Make illegal states unrepresentable.** Prefer types/structures where the bad case can't be constructed over runtime checks that hope to catch it.
- **Reduce nesting.** Guard clauses and early returns over deep `if`/`else` pyramids.

## Stay in scope

- **Do what was asked--and the cleanup it directly requires--but don't gold-plate.** Don't refactor unrelated code, rename things wholesale, or "while I'm here" your way into a sprawling diff.
- **Keep diffs surgical.** A reviewer should be able to see exactly what changed and why. Smaller, focused changes over large opportunistic ones.
- **Leave it at least as clean as you found it**, but separate genuine drive-by improvements from the task and call them out rather than burying them.

## Errors and edges

- **Match the repo's error strategy** (exceptions vs. result types vs. error returns). Don't introduce a competing one.
- **Don't swallow errors.** No empty `catch`, no catch-log-continue that hides failure. Fail where failure is meaningful and let callers decide.
- **Handle the real edge cases** (empty, null, boundary, concurrent)--but via structure where possible (see taste), not a thicket of defensive checks.

## Before you call it done

- Run the build, the formatter, the linter, the type checker, and the tests. Report honestly: if something fails or you skipped a step, say so with the output--don't claim green when it's not.
- Re-read your own diff as a reviewer would. If anything in it would make *you* leave a comment, fix it first.

