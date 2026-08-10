# Contributing Guidelines, Development Settings, and More

This file is meant for developers interested in contributing, or for existing maintainers who need a refresher on development settings or other common troubleshooting tips.

## Contributing Guidelines

- Make sure to open an issue before submitting a pull request. This helps with issue tracking.
- You may find the `AGENTS.md` file helpful. This has brief style and design guides as well as some useful information on what directories/crates do what.
- Make sure you review the [policy on AI use](./AI_POLICY.md) in this repo.
- Large PRs are auto-closed and require maintainer approval. "Large" means a net change (whether additions or deletions) of 1,000 lines, or a PR that has both additions and deletions of over 1,000 lines. Please communicate with a maintainer (ideally, through an issue) if you plan on submitting a large PR.

## Troubleshooting Tips

### `cargo clippy --fix` errors with "unsupported mandatory extension: 'link'"

This is likely due to having `core.splitindex` set in your `.gitconfig`. To fix this, disable this setting for this repo:

```sh
git config core.splitindex false
git update-index --no-split-index
```

### warning: linker stderr: ld: __eh_frame section too large (max 16MB) to encode dwarf unwind offsets in compact unwind table, performance of exception handling might be affected

This is a known issue on macOS, but is ultimately harmless.
