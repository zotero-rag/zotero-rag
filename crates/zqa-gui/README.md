# zqa-gui

A native GUI front-end for `zqa`, built on [GPUI Kit](https://gpui-kit.com/).

## Running

```
cargo run -p zqa-gui
```

The GUI reuses the same configuration and LanceDB database as the CLI. Configure providers and API keys as described in the top-level README before running real queries; `/help` works with no configuration.

## Testing the UI

Run the whole-app interaction tests with:

```sh
cargo test -p zqa-gui gui_
```

The inline `GuiHarness` in `src/main.rs` constructs `ZqaApp` with synthetic history and in-memory engine channels. It does not start an engine thread or load the initial history from disk. Tests can set private view state with `harness.app.update(...)`, inject `UiEvent`s through `harness.events`, and inspect outgoing commands and cancellation signals.

Use `gpui_kit::test::TestWindowExt` to click controls by ID, type into the focused input, and press keys. Call `cx.run_until_parked()` after injecting events to let the GUI consume them before asserting state. Keep backend responses synthetic; paths that explicitly refresh saved history still access the filesystem.

The `prompt` test filter runs the prompt model and card interaction tests. GPUI's headless windows exercise layout and input handling, but do not verify native macOS materials or rendered pixels; those still need a real-window check.

## System dependencies

### macOS

Xcode command line tools (`xcode-select --install`). No other system libraries are required.

The sidebar uses a translucent tint over GPUI's native blurred window. `src/macos.rs` selects AppKit's `Sidebar` material for the existing backdrop: on macOS 27, GPUI's `Selection` material can remain transparent without blurring. The hook uses public AppKit APIs and does not replace GPUI's view or modify its layers.

### Linux

GPUI needs font, display-backend, and related libraries:

#### Ubuntu

```
sudo apt-get install -y \
  clang libfontconfig-dev libwayland-dev \
  libxkbcommon-x11-dev libx11-xcb-dev libzstd-dev libvulkan1
```

#### Fedora

```
sudo dnf install -y \
  clang fontconfig-devel wayland-devel \
  libxkbcommon-x11-devel libX11-devel libzstd-devel vulkan-loader
```

#### Arch

```
sudo pacman -Syu --needed \
  clang fontconfig wayland \
  libxkbcommon-x11 libx11 zstd vulkan-icd-loader
```
