# zqa-gui

A native GUI front-end for `zqa`, built on [GPUI Kit](https://gpui-kit.com/).

This is a separate workspace crate rather than a feature of `zqa`. GPUI Kit provides a
single crates.io dependency that pins and re-exports compatible GPUI, component, and asset crates.

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

- The `gpui-kit` dependency pins a mutually compatible GPUI, component, and asset stack.
- GPUI's dependency graph is large, so the first build is slow. If you use `sccache` as a
  `RUSTC_WRAPPER`, make sure `SCCACHE_DIR`/`TMPDIR` point at stable locations, otherwise
  GPUI's build can fail while creating temp files.
