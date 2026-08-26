---
type: Crate
title: zqa-gui
description: The native GPUI desktop front-end that reuses the zqa engine, configuration, and LanceDB database.
tags: [gui, gpui, rag]
timestamp: 2026-08-26T00:59:13-04:00
---

# Purpose

`zqa-gui` is a native desktop front-end for the RAG pipeline, built on
[GPUI](https://gpui.rs) and `gpui-component`. It lives in the workspace as a
separate, unpublished crate rather than a feature of `zqa`, because `gpui` is
a git-only dependency and crates.io rejects git dependencies. Keeping the GUI
separate lets [the CLI crate](/cli.md) stay publishable.

Unlike the rest of the workspace, `zqa-gui` is licensed GPL-3.0-or-later.

# Architecture

The window is a chat-style harness over the same engine the CLI drives. There
is no second implementation of the pipeline: `zqa::session::Session` (see
[the CLI](/cli.md)) forwards command strings to the same command dispatcher
the REPL uses, writing output to caller-supplied streams. `zqa-gui`'s `bridge`
module runs that session on a dedicated thread with its own Tokio runtime and
streams output back to the view as `UiEvent`s, which are folded into a
transcript of turns.

The layout follows desktop AI-harness conventions: a translucent sidebar with
session and library commands, an opaque main pane with the conversation, and
no system title bar. The GUI opens the same LanceDB database and reads the
same [runtime configuration](/configuration.md) as the CLI, so `/help` works
unconfigured and real queries need the same provider credentials.

# Platform requirements

Building GPUI requires system libraries. On macOS, only the Xcode command
line tools are needed. On Linux, font, display-backend, and related packages
are required (`clang`, `libfontconfig-dev`, `libwayland-dev`,
`libxkbcommon-x11-dev`, `libx11-xcb-dev`, `libzstd-dev`, `libvulkan1`); CI
installs the same set. GPUI's dependency graph is large, so the first build
is slow. See [development workflow](/development.md) for workspace-wide
commands.

# Related concepts

[System overview](/system-overview.md) describes where the GUI sits relative
to the CLI and libraries. [The CLI](/cli.md) documents the command set the
GUI dispatches and the embeddable `Session` driver it builds on.
