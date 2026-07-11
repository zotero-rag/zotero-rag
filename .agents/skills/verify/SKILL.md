---
name: verify
description: Build, launch, and drive the zqa CLI/TUI to verify changes end-to-end.
---

# Verifying zqa changes

## Build

```bash
cargo build -p zqa        # binary at target/debug/zqa
```

## Launch the TUI (or CLI) in an isolated pty

Use a dedicated tmux server so you don't touch the host session. Always set
`ZQA_STATE_DIR` to a temp dir (isolates history/conversations/LanceDB) and
`CI=true` (uses the toy library in `zqa/assets/Zotero` instead of `~/Zotero`):

```bash
STATE=$(mktemp -d)
tmux -L zqa-verify new-session -d -x 110 -y 32 \
  "ZQA_STATE_DIR=$STATE CI=true ./target/debug/zqa --tui"
tmux -L zqa-verify send-keys "/help" Enter
tmux -L zqa-verify capture-pane -p
tmux -L zqa-verify kill-server   # cleanup
```

Gotchas:

- A fresh `ZQA_STATE_DIR` triggers the first-run OOBE on the plain terminal
  before the TUI starts; answer `n` + Enter to skip it.
- Config/API keys come from the repo's `.env`; plain-text queries (10+ chars)
  hit real APIs and cost real money. Prefer slash commands for driving.
- Force vi keybindings with `EDITRC=<file>` where the file contains `bind -v`.
- Simulate a mouse wheel tick with `tmux send-keys -l $'\e[<64;5;5M'`.
- To exercise the `/resume` reply prompt, drop a JSON file into
  `$STATE/conversations/` shaped like `SavedChatHistory` (see `zqa/src/state.rs`);
  roles serialize lowercase (`"user"`, `"assistant"`).
- Quitting: `/quit`, or two Ctrl-C presses within 2s on an empty input line.

## Tests (CI parity, not verification)

```bash
CI=true cargo test -p zqa --lib     # without CI=true, tests parse ~/Zotero and fail
cargo clippy --all-targets --all-features -- -D warnings
```
