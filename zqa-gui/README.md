# zqa-gui

A native GUI front-end for `zqa`, built on [GPUI](https://gpui.rs) and
[gpui-component](https://github.com/longbridge/gpui-component).

This is a separate, **unpublished** workspace crate rather than a feature of `zqa`, because
`gpui` is a git-only dependency and crates.io rejects git dependencies (even optional ones).
Keeping the GUI here lets `zqa` itself stay publishable.

## Running

    cargo run -p zqa-gui

The GUI reuses the same configuration and LanceDB database as the CLI. Configure providers and
API keys as described in the top-level README before running real queries; `/help` works with
no configuration.

## System dependencies

### macOS

Xcode command line tools (`xcode-select --install`). No other system libraries are required.

### Linux (Ubuntu 24.04)

GPUI needs font, display-backend, and related libraries:

    sudo apt-get install -y \
      clang libfontconfig-dev libwayland-dev \
      libxkbcommon-x11-dev libx11-xcb-dev libzstd-dev libvulkan1

This mirrors what CI installs (see `.github/workflows/rust-checks.yml`).

## Notes

- The `gpui`, `gpui_platform`, and `gpui-component` dependencies are unpinned git deps; exact
  commits are recorded in the workspace `Cargo.lock`.
- GPUI's dependency graph is large, so the first build is slow. If you use `sccache` as a
  `RUSTC_WRAPPER`, make sure `SCCACHE_DIR`/`TMPDIR` point at stable locations, otherwise
  GPUI's build can fail while creating temp files.
