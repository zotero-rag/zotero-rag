# Contributing Guidelines, Development Settings, and More

This file is meant for developers interested in contributing, or for existing maintainers who need a refresher on development settings or other common troubleshooting tips.

## Contributing Guidelines

- Make sure to open an issue before submitting a pull request. This helps with issue tracking.
- You may find the `AGENTS.md` file helpful. This has brief style and design guides as well as some useful information on what directories/crates do what.

## Troubleshooting Tips

### `cargo clippy --fix` errors with "unsupported mandatory extension: 'link'"

This is likely due to having `core.splitindex` set in your `.gitconfig`. To fix this, disable this setting for this repo:

```sh
git config core.splitindex false
git update-index --no-split-index
```
