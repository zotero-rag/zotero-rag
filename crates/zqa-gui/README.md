# zqa-gui

A native GUI front-end for `zqa`, built on [GPUI Kit](https://gpui-kit.com/).

## Running

```
cargo run -p zqa-gui
```

The GUI reuses the same configuration and LanceDB database as the CLI. Configure providers and API keys as described in the top-level README before running real queries; `/help` works with no configuration.

## System dependencies

### macOS

Xcode command line tools (`xcode-select --install`). No other system libraries are required.

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
