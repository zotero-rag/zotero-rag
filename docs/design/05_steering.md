# Design: Steering in zqa

**Linear issues:** ZOT-219 (related: ZOT-248, ZOT-249, ZOT-230, ZOT-211, ZOT-78)  
**GitHub issues:** [#315](https://github.com/zotero-rag/zotero-rag/issues/315)  
**Author(s):** Rahul Yedida + Claude Code  
**`master` as of writing:** b1ac77a  
**Code read at:** `feat/migrate-zqa-gui-events` @ `ebaabba` (tip of the event-architecture stack)  
**Date:** 2026-10-29  
**Status:** Approved

## Summary

Steering should live inside the tool loop in `AgenticClient::send_message`, fed by one inbound control channel that every frontend writes to. This is ZOT-219's Phase 1, made concrete against the event architecture now in review.

- **What steering means here:** while a query is running, the user can **Stop** it, **Steer** it with guidance the model sees before its next call, or **Queue** a message that runs as its own query once this one finishes. Steer and Queue mirror pi's steering and follow-up queues.
- **Why inside the loop:** cancelling the dispatch and re-sending a new query would rebuild history from canonical `ChatHistoryItem`s, which drop signed reasoning, and would lose the run's tool results, which only come back on success. Only the loop holds the native `provider_history`.
- **Decided in review (Oct 29):** steer text is a plain user message, as in pi; only Stop interrupts anything, so a steer waits for the model call and any running tools to finish, as in pi; Stop keeps the finished tool rounds; a steer during the final, no-tools iteration buys one more tool round; the GUI defaults to Steer, configurable to Queue.
- **Size:** one new, breaking field on `ChatRequest` (`send_message` takes `&mut ChatRequest`), a `select!` in the loop, Steer, Queue and Stop commands in the GUI bridge, a Ctrl-C path in the CLI, and three new events (see Where the events live). No provider adapter changes.

## Context: what runs today

A query is one long future with no way in once it starts, except prompt replies.

- **The loop** (`zqa-rag/src/llm/base.rs`, `send_message`): builds `provider_history` once, then repeats `send_once` (one full, non-streaming request) and `process_tool_calls`. Tools in a turn run concurrently through `join_all`. `on_text` and `on_reasoning` fire only after the whole turn and its tools finish. The last iteration sends tools with tool calls disallowed.
- **The query handler** (`zqa/src/cli/handlers/query.rs`, `handle_query_cmd`): `select!`s over `send_message` and three channels (response segments, tool status, cost) and forwards them as `EngineEvent`s. On success it pushes the raw query plus `response.history_additions` into `ctx.state.chat_history`. On error nothing from the run is kept.
- **What the user sees mid-run:** the `Verbose` tool wrapper emits a `StatusUpdate` with the tool name and args *before* each tool runs, then `Timed` reports its duration. Model text and reasoning appear only per completed turn. So in Phase 1 the user steers mostly on seeing a bad retrieval query.
- **Events** (`zqa/src/io.rs`, `EngineEvent`): outbound only. Input flows back solely through `oneshot` reply senders on `Confirm`, `Choose`, `Secret` and `Line`.
- **GUI** (`zqa-gui/src/bridge.rs`): the engine thread races `session.dispatch()` against `cancel_rx` and `cmd_rx`. Cancel drops the dispatch future. A command sent while one runs is parked in `pending_command`. A `TODO(ZOT-219)` notes the detached title task survives cancellation.
- **CLI** (`zqa/src/cli/app.rs`, `run_command`): `select!`s dispatch against the event channel and renders each event with `handle_event`, which reads stdin synchronously for prompts. Rustyline is not active during a command, and nothing handles Ctrl-C there, so a Ctrl-C mid-query most likely ends the process (not tested).
- **Storage plan** (`docs/design/04_conversation_storage.md`, Approved): one immutable node per provider message, committed through a new `on_history_item` callback, and append-only requests so Anthropic preserved thinking stays valid. It already says steering injections fit as ordinary nodes.

## Goals and non-goals

- **Goals**
  - Stop a running query from the GUI and the CLI, with the conversation left in a valid, resumable state.
  - Steer a running query with free text that the model sees before its next call.
  - Keep requests append-only, so preserved thinking and prompt caches survive a steer.
  - One mechanism in `zqa-rag`, shared by all five providers and by both frontends.
- **Non-goals**
  - Token streaming and mid-stream steering. That is Phase 2 (the `TurnEvent` stream); this design leaves room for it.
  - Steering in headless mode (ZOT-78). Piped stdin is the command stream, so there is no second input; SIGINT stops, nothing steers.
  - Changing the frame mid-run (system prompt, tool set, reasoning settings). A steer is conversation content only.
  - Steering subrequests: title generation and the summarization subagent never get a control channel.

## Design

Frontends write `Control` messages into one channel per session; the tool loop is the only reader, and it reports what it did back through `EngineEvent`s.

```mermaid
flowchart LR
    gui["GUI: Enter / Stop"] -->|Control| ch(["control channel"])
    cli["CLI: Ctrl-C prompt"] -->|Control| ch
    ch --> call
    subgraph loop["tool loop (send_message)"]
        call["until_stop: model call / tool batch"]:::effect
        drain["drain point between rounds"]:::effect
        call --> drain --> call
    end
    loop -->|AgentEvent| ev["EngineEvent::Agent"]
    ev --> gui
    ev --> cli
    classDef effect fill:#fde68a,stroke:#b45309
```

The highlighted boxes are the only two places a control takes effect; everything else only forwards it.

### `zqa-rag`: the control channel

```rust
pub enum Control {
    Stop,
    Steer(String),
    Queue(String),
}

pub struct ChatRequest<'a> {
    // ...existing fields...
    /// Inbound control for this run. `None` for subrequests (titles, summarization).
    pub control: Option<&'a mut mpsc::UnboundedReceiver<Control>>,
}

// Breaking: the loop needs `&mut` to call `recv`.
async fn send_message(&self, request: &mut ChatRequest<'_>) -> Result<CompletionApiResponse, LLMError>;

pub struct CompletionApiResponse {
    // ...existing fields...
    pub outcome: RunOutcome,    // Completed | Stopped
    pub steered: Vec<String>,   // steers folded in, in order
    pub leftover: Vec<Control>, // unrun steers first, then Queue items
}
```

- **A plain field, no `Mutex`** (decided in review). Every caller changes `&request` to `&mut request`: `generate_title`, the summarization tool, `handle_query_cmd` and the per-provider tests.
- **Unbounded:** controls are rare and tiny, and a frontend must never block on sending one.
- **`Queue` is not the loop's business.** The loop never acts on it; it buffers each `Queue` it reads and hands it back in `response.leftover`, so nothing is lost. `zqa` runs them (wiring, below). A second channel would keep `Queue` out of `zqa-rag` entirely, but one ordered channel keeps the order the user typed in.

### The loop

One helper, `until_stop`, wraps both the model call and the tool batch. Only Stop interrupts; everything else waits.

```rust
/// Polls `fut` to completion unless Stop arrives first (then `None`).
/// Steer and Queue never interrupt: they are buffered in arrival order.
async fn until_stop<F: Future>(
    ctrl: &mut UnboundedReceiver<Control>,
    buf: &mut VecDeque<Control>,
    fut: F,
) -> Option<F::Output> {
    tokio::pin!(fut);
    loop {
        select! {
            biased;
            Some(c) = ctrl.recv() => match c {
                Control::Stop => return None,
                other => buf.push_back(other),
            },
            out = &mut fut => return Some(out),
        }
    }
}

while round_trips < limit {
    let Some(turn) = until_stop(ctrl, &mut buf, self.send_once(..)).await
        else { return Ok(self.stopped(..)) };
    let turn = turn.map_err(|e| e.with_leftover(take_all(&mut buf)))?; // keep buffered input on error
    // ...extend history, emit as today...

    let mut tail = if turn.has_tool_calls() {
        let Some(results) = until_stop(ctrl, &mut buf, process_tool_calls(..)).await
            else { return Ok(self.stopped(..)) };
        results                         // generic User item holding the tool results
    } else {
        ChatHistoryItem::user_empty()   // the model answered; a steer opens a new user message
    };

    if drain(ctrl, &mut buf).is_stop() {   // try_recv what arrived since
        if turn.has_tool_calls() {         // keep the round that just finished
            provider_history.extend(Vec::from(tail.clone()));
            history_additions.push(tail);
        }
        return Ok(self.stopped(..));
    }
    let steer = if is_last_turn && extra_round_used { None } else { take_oldest_steer(&mut buf) };
    if steer.is_none() && !turn.has_tool_calls() { break } // answered, nothing to fold in: done
    if let Some(text) = steer {
        tail.push_text(text);           // one at a time, as in pi
        if is_last_turn { limit += 2; extra_round_used = true; } // one tool round, then the final answer
    }
    provider_history.extend(Vec::from(tail.clone()));
    history_additions.push(tail);
    round_trips += 1;
}
// response.leftover = steers still in `buf`, then its Queue items
```

- **Steers wait, as in pi** (decided in review). `until_stop` buffers Steer and Queue and keeps polling the same future, so a model call or tool batch always runs to completion unless the user stops. A steer that arrives mid-call costs that call; in return there is one code path and nothing is ever thrown away but a stopped call.
- **Where a steer lands.** After tool results, it is appended as text to the same tool-results user message. After an answer with no tool calls, it opens a new user message and the loop continues instead of returning. pi does the same: its loop runs while there are tool calls or steering messages.
- **Nothing is rebuilt.** The steer is added to the generic `ChatHistoryItem` before it is converted with the existing `Into<Vec<Self::HistoryItem>>`, so no native item is ever popped or re-converted. Anthropic gets text blocks after the `tool_result` blocks in one user message, and OpenAI gets `function_call_output` items followed by a user message (checked in `openai.rs`).
- **One extra round for a late steer** (decided in review). A steer that arrives during the final, no-tools iteration waits for that answer, then raises the limit by two, once per run: one round with tools, so "also look for X" can be acted on, then the usual final answer without tools. Raising it by one would not be enough, because `is_last_turn` is recomputed as `round_trips == limit - 1` and the next iteration would be the last one again.
- **A Stop between rounds keeps the round.** A Stop seen at the drain point after a tool batch first pushes that batch's results (without a steer), so the finished round is kept, as decided in review.
- **Leftovers.** Steers still buffered when the loop ends (because the extra round is spent) and every `Queue` item go back in `response.leftover`, steers first, since pi also runs leftover steers before follow-ups.
- **Errors keep buffered input.** If a model call fails after `until_stop` has buffered a Steer or Queue, that text is no longer in the channel. So the error carries it out the same way, in a `leftover` field on the loop's error, and the handler treats it as on Stop (below).

### `zqa`: wiring

- **`Context`** gains `control_rx: Option<UnboundedReceiver<Control>>`; **`Session`** exposes a cloneable `SteerHandle` wrapping the sender. The CLI builds the pair in `run()` next to the event channel.
- **`handle_query_cmd`** takes the receiver out with `.take()` for the run and passes `control: Some(&mut rx)` on the main request only. It has to take it, because `ctx.emit` borrows all of `ctx` while the request holds the receiver. A small guard puts it back on every exit path, since the handler returns early with `?`.
- **Queued and leftover messages run as their own commands.** After `send_message` returns, the handler pushes `response.leftover` (unrun steers first, then `Queue` items) onto a new `ctx.state.pending_inputs` queue. After every command, query or not, `Session::dispatch` and the CLI's `run_command` read what is still in the channel: they drop stale `Stop`s (the same reason the bridge drains `cancel_rx`) and push each `Steer` or `Queue` onto `pending_inputs`, in arrival order. They then keep dispatching until it is empty, so a queued `/search` works too, and text typed while any command ran is neither lost nor run out of order. This replaces the bridge's single `pending_command` slot.
- **On Stop or an error**, unrun steers and queued messages are not run. They go back to the frontend in `EngineEvent::Unsent { messages }`, the way pi puts queued text back into the editor when a run is aborted. This covers the whole chain, not just the current query: a `Stopped` or error outcome also moves everything left on `pending_inputs` into `Unsent`. So does a Stop that the sweep finds while `pending_inputs` is not empty, which is how a Stop sent during a non-query command (a queued `/search`) ends the chain.
- **New events:** `Steered { message }` when the loop folds a steer in, so the transcript shows it where the model saw it; `Stopped` when a run ends by Stop; `Unsent { messages }` as above. Which crate owns them is the next section.

### Frontends

- **GUI:** add `EngineCommand::Steer(String)`, `EngineCommand::Queue(String)` and `EngineCommand::Stop`. Enter while running sends Steer by default; a config setting (for example `running_input = "steer" | "queue"`) flips the default, and a modifier sends the other one, like pi's Alt+Enter for follow-ups. While idle, both become a `Dispatch`. The existing cancel path (drop the future) stays as the hard stop for a second click, because a soft Stop waits for the loop to reach a `select!` point and keeps the partial run.
- **CLI:** add a `tokio::signal::ctrl_c()` arm to the `select!` in `run_command`. On Ctrl-C, stop polling the dispatch and prompt on stdin: a line is sent as the configured default (Steer unless set to Queue), and an empty line sends Stop. That read is synchronous like `handle_event`'s prompt reads, and while it blocks the dispatch is not polled, so the run is effectively paused. This avoids a second concurrent stdin reader, which would fight with prompt replies.

## Where the events live

Two sets: `zqa-rag` owns an `AgentEvent` enum for what the loop knows, and `zqa`'s `EngineEvent` wraps it in one variant beside the app-level events.

- **`zqa-rag` already emits events, as callbacks.** `on_text`, `on_reasoning` and `on_tool_call` today; design doc 04 adds `on_history_item`; Phase 2 adds `TurnEvent`. Much of `handle_query_cmd` exists to turn those callbacks into `EngineEvent`s (`ResponseSegment` plus three unbounded channels).
- **Some `EngineEvent`s describe the loop, not the app.** `ToolCall` and `ToolResponse` are never constructed anywhere in `zqa`; only the GUI matches them, and their own TODO says the `zqa-rag` hooks should change.
- **Steering adds loop-only facts.** Only the loop knows when a steer was folded in or a run stopped.

```rust
// zqa-rag: replaces on_text, on_reasoning, on_tool_call (and the planned on_history_item)
pub enum AgentEvent {
    Text(String),
    Reasoning(String),
    ToolCallStarted { id: String, name: String, args: Value },
    ToolCallFinished { id: String, name: String, result: Result<Value, String> },
    HistoryItem(ChatHistoryItem),
    Usage(ModelUsage),
    Steered(String),
    Stopped,
}
// ChatRequest gets `events: Option<mpsc::UnboundedSender<AgentEvent>>` beside `control`:
// one channel in, one channel out.

// zqa
pub enum EngineEvent {
    Agent(AgentEvent),
    StyledText { .. }, Text { .. }, StatusUpdate { .. }, Warning { .. }, Error { .. },
    TokenUsage { usage: UsageMetadata }, // priced, so it stays in zqa
    ProgressStart { .. }, ProgressUpdate { .. }, ProgressComplete { .. },
    Confirm { .. }, Choose { .. }, Secret { .. }, Line { .. },
    Unsent { messages: Vec<String> },    // zqa dispatches queued input, so zqa owns this
}
```

- **Cost of wrapping:** frontends match two levels (`EngineEvent::Agent(AgentEvent::Text(..))`), and `Display` delegates for that arm. The alternative, flattening with a `From` impl in `zqa`, keeps one level but needs every new loop event added in both crates plus a translation arm, which is what `handle_query_cmd` does today.
- **Subagents get it for free.** The summarization tool can give its sub-request its own sender and choose what to forward, the same concern as the branch's "keep reasoning and tool calls out of sub-request results" fix.
- **When (decided Oct 30):** after ZOT-248 and ZOT-249 merge, since that stack still has work left. `AgentEvent` is the first change after the merge, and steering builds on it, so its three variants land in the right crate from the start.

## Semantics: ZOT-219's three questions

| Question | Recommendation | Why |
| --- | --- | --- |
| 1. Does steering text become a new user turn? | User-role text. After tool results it is appended to the same user message; after an answer with no tool calls it opens a new user message and the loop goes on. Either way it is not a new user turn in `zqa`'s sense: no new rendered summarize prompt, no new title, same run. | Both are placements every provider accepts mid-loop without synthetic messages, and both keep the request append-only. |
| 2. Cancel outstanding tool calls, or wait? | **Steer waits** for tools already running. **Stop drops** them. | Every tool today is read-only, and their results are useful context for the steered call. Dropping them would leave a `tool_use` with no result, which every provider rejects, so a steer would need fake "cancelled" results. Stop ends the run, so nothing needs a result. |
| 3. Mid-stream, or only at the next model call? | At the next model call, **after the call in flight finishes** (decided in review, as pi does). | Nothing but a stopped call is ever thrown away, so there is one code path and no half-sent request to reason about. The cost is paying for a turn the user may already have rejected; once Phase 2 streams that turn, the user can see what they are steering. |

A steer is sent as plain user text with no wrapper, as pi does (decided in review). The transcript shows the same text.

Queued steers are delivered one at a time, also as in pi, whose default `steeringMode` is `one-at-a-time` (the alternative is `all`). Each drain point takes the oldest steer and the rest wait for the next call. A config switch can offer `all` later.

Steers wait, as in pi (decided in review). pi never aborts an in-flight call: a steer waits until the current assistant turn and its tools finish, and its loop keeps going while there are tool calls or steering messages, so a steer that arrives during a final answer still gets a reply. This design does the same.

## History persistence and reasoning preservation

A steer only ever appends after the last native item, and the only call ever discarded is one cut off by Stop, so the request stays append-only and nothing signed is lost.

### Reasoning (ZOT-230)

- **Inside the loop it just works.** `provider_history` holds native items, including Anthropic `thinking` blocks with signatures, OpenAI reasoning items and Gemini thought signatures. A steer adds text after the tool results or after the answer; the prefix, and so every preserved thinking block, is unchanged. The frame is unchanged too, because a steer never touches tools, system prompt or reasoning settings.
- **A stopped call leaves no gap.** Its response was never appended, so there is nothing to strip. Anthropic allows dropping thinking only from the start of history, never from the middle, and this design never needs to.
- **Nothing native is rebuilt.** A steer is added to the generic item before it is converted, and tool-result items are built locally from generic `ToolCallResponse`s anyway, so no native payload is ever popped or re-converted.
- **This is the main reason steering cannot be cancel-and-redispatch.** A new dispatch rebuilds history through `From<ChatHistoryItem>`, which drops `ChatHistoryContent::Reasoning` (`anthropic.rs`, line 54 on the branch). The steered call would then continue a `tool_use` turn without its thinking block.

### Persistence, with today's JSON history

- **Steer:** the text lives in a user `ChatHistoryItem`, either the tool-results item or a new one after an answer, so `history_additions` carries it to disk with no format change. It is added before the item is pushed, so what is saved matches what was sent.
- **Stop:** `send_message` returns `Ok` with `RunOutcome::Stopped` instead of an error. `handle_query_cmd` saves the query plus `history_additions`, trimmed to the last complete round: a trailing assistant item whose tool calls have no results is dropped. Usage is still recorded. Today, a GUI cancel loses the whole run, the user's query included.
- **Aside on ZOT-211:** both `master` and the branch already do `history.extend(response.history_additions)`, which includes tool calls and results. The issue text (only the query and final answer are kept) looks out of date. Inferred from the code; worth confirming before closing it.

### Persistence, with the conversation graph (design doc 04)

- **A steer after tool results** is part of the `ToolResults` node, since it is in the same provider message. The node gains `steering: Vec<String>` as display text, next to `provenance`.
- **A steer after an answer with no tool calls** is its own `User` node, a child of that `Assistant` node, and the loop's next call continues from it.
- **A Stop keeps the finished tool rounds** (decided in review). The stopped call becomes a `Failed { error: "stopped" }` node with the attempt marked `cancelled`, and the head moves to the last complete `ToolResults` node, so a follow-up can reuse what was retrieved.
- **Design doc 04 needs two amendments** (agreed in review, not yet made):
  - Under *Settled nodes*, count a `ToolResults` node whose calls all have results as settled, so a user message may follow it after a Stop.
  - Under *Write path*, step 4, a Stop moves the head to that node instead of leaving it where the run started; errors and refusals keep today's rule.

## Edge cases

| When | Steer | Queue | Stop |
| --- | --- | --- | --- |
| During a model call | Waits for the call. If the model asks for tools, folded in after them; if it answers, appended as a new user message and the loop continues. | Held; runs after the query. | Abort the call. On the first call, save the query alone; later, keep completed rounds. |
| While tools run | Waits; folded in after the tools finish, before the next call. | Held. | Drop the `join_all` future. Retrieval and summarization futures go with it, which also stops summarization's own LLM calls. |
| During the final, no-tools iteration | Waits for the answer, then continues with one extra tool round, once per run. | Held. | Abort; keep completed rounds. |
| After the extra round is spent, or after the loop returns | Left over; runs as its own command, before queued ones. | Runs as its own command. | Ends the chain: anything on `pending_inputs` comes back in `Unsent`. With nothing pending, ignored. |
| While a prompt (`Confirm`, `Line`) waits for a reply | Waits until the loop next polls. No prompt is raised in the query path today. | Same. | Same, unless the user also cancels the prompt (GUI: dropping the reply sender). |
| Several in quick succession | One per drain point, oldest first; one `Steered` event each. | All run, in order. | First Stop wins; unrun steers, queued items and everything on `pending_inputs` come back in `Unsent`, and nothing more runs. |
| Provider request already sent when stopped | n/a: a steer never aborts a call. | n/a | The provider may still bill for it. Record the usage as unknown; it never enters history. |
| Background title task | Unaffected. | Unaffected. | Keep it running: it is cheap and the stopped conversation still needs a title. This resolves the bridge's `TODO(ZOT-219)` by decision rather than code. |
| Subrequests (title, summarization subagent) | Never see the channel (`control: None`). | Same. | Same. |
| Headless mode (ZOT-78) | Not supported. | Not supported. | SIGINT, through the same `ctrl_c()` arm, with no prompt. |

## Phase 2: streaming

Nothing here blocks the `AgentDriver` / `TurnEvent` split in ZOT-219; the control channel and the `select!` points move into the driver unchanged.

- **Mid-stream Stop** drops the turn stream exactly as Phase 1 drops `send_once`. The partial text was already shown, so it should stay in the transcript, marked interrupted, but never enter history. A mid-stream Steer still waits for the turn to end, as in pi; streaming only lets the user see what they are steering.
- **Partial reasoning must be discarded.** A thinking block cut off mid-stream has no signature and cannot be replayed. Storing it as `Failed { partial }` (design doc 04) keeps it visible without sending it.
- **The `ToolCall` event TODO in `io.rs` gets fixed for free:** a streamed `ToolCall` arrives before execution, so `EngineEvent::ToolCall` can fire at the right time instead of after the tool runs.

## Rejected alternatives

- **Cancel and re-dispatch with the steer as a new query.** No `zqa-rag` change, but it rebuilds history from canonical items, losing reasoning and, on error paths, the run's tool results (see History persistence).
- **Steering through the `EngineEvent` reply channels** (a standing `Line` prompt the frontend can answer at any time). Events are outbound by contract and replies are one-shot; a standing prompt would also block the CLI's single stdin reader.
- **A `CancellationToken` for Stop alone, deferring Steer.** Simpler, but Steer then needs a second mechanism later. One channel covers both for almost the same cost.
- **Cancelling running tools on Steer.** Needs synthetic tool results to keep the request valid, and throws away work the model probably wants.
- **Aborting the in-flight model call on Steer** (rejected in review). It saves paying for a turn the user has rejected, but needs a second path (abort, re-convert the tail, resend, a `CallAborted` event) and differs from pi.

## Open questions

1. **Provider layout of text after tool results.** Anthropic and OpenAI are covered above; Gemini (text parts beside `functionResponse` parts in one content), OpenRouter (role `tool` messages, then a user message) and Ollama need a fixture test each before this ships.

### Decided in review (Oct 29 and 30)

- Stop keeps the finished tool rounds; design doc 04 gets the amendments listed under History persistence.
- A steer during the final, no-tools iteration buys one more tool round, once per run.
- Steer text follows pi: plain user text, no wrapper, delivered one at a time.
- Steers wait, as in pi: a model call or tool batch always finishes unless the user stops, and a steer after a plain answer continues the loop.
- The GUI defaults to Steer while running, configurable to Queue, with a modifier for the other.
- `Control` gains `Queue`, and the receiver is a plain `&mut` field on `ChatRequest`; breaking callers is fine.
- `AgentEvent` lands after the ZOT-248/249 stack merges (Oct 30), as the first change before steering.
