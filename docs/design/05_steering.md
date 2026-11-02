# Design: Steering in zqa

**Linear issues:** ZOT-219 (related: ZOT-248, ZOT-249, ZOT-230, ZOT-211, ZOT-78)  
**GitHub issues:** [#315](https://github.com/zotero-rag/zotero-rag/issues/315)  
**Author(s):** Rahul Yedida + Claude Code (TODO: model, high effort)  
**`master` as of writing:** b1ac77a  
**Code read at:** `feat/migrate-zqa-gui-events` @ `ebaabba` (tip of the event-architecture stack)  
**Date:** 2026-10-29  
**Status:** Approved

## Summary

Once a query starts in zqa, the user can only wait for it or kill it. This doc proposes a way to talk to a query while it runs. The user can **Stop** it, **Steer** it with guidance that the model sees before its next call, or **Queue** a message that runs as its own query once this one is done. Steer and Queue work like pi's steering and follow-up queues, and we follow pi's choices in most places below.

The core idea is that steering lives inside the tool loop in `AgenticClient::send_message`, fed by one inbound control channel that every frontend writes to. It has to live there because only the loop holds the native `provider_history`. The obvious alternative, cancelling the dispatch and sending a new query with the steer in it, would rebuild history from canonical `ChatHistoryItem`s. That drops signed reasoning, and it loses the run's tool results, which today only come back when a run succeeds. This is ZOT-219's Phase 1, made concrete against the event architecture that is now in review.

A few decisions came out of review on Oct 29. Steer text is sent as a plain user message, as in pi. Only Stop interrupts anything: a steer waits for the model call and any running tools to finish. Stop keeps the tool rounds that already finished. A steer that arrives during the final, no-tools iteration buys one more round with tools. And the GUI sends Steer by default while a query runs, with a setting to make Queue the default instead.

The change is fairly small. `ChatRequest` gets one new field, which is breaking because `send_message` now takes `&mut ChatRequest`. `send_message`'s error type gains a wrapper so a failed run can hand back buffered input, which is also breaking. The loop gets a `select!`, the GUI bridge gets Steer, Queue and Stop commands, the CLI gets a Ctrl-C path, and there are three new events (see Where the events live). No provider adapter changes.

## Context

Today a query is one long future, and once it starts the only way in is a reply to a prompt.

The loop itself is `send_message` in `zqa-rag/src/llm/base.rs`. It builds `provider_history` once, then repeats `send_once` (one full, non-streaming request) and `process_tool_calls`. Tools in a turn run concurrently through `join_all`. The `on_text` and `on_reasoning` callbacks fire only after the whole turn and its tools are done, and the last iteration sends tools with tool calls disallowed.

On the `zqa` side, `handle_query_cmd` (`zqa/src/cli/handlers/query.rs`) `select!`s over `send_message` and three channels (response segments, tool status and cost) and forwards them as `EngineEvent`s. On success it pushes the raw query plus `response.history_additions` into `ctx.state.chat_history`. On error, nothing from the run is kept.

While a query runs, the user sees less than you might expect. The `Verbose` tool wrapper emits a `StatusUpdate` with the tool name and args before each tool runs, and `Timed` reports its duration afterwards, but model text and reasoning only appear once a turn has completed. So in Phase 1, the moment a user will most often want to steer is when they see a bad retrieval query go by.

`EngineEvent` (`zqa/src/io.rs`) is outbound only. Input flows back solely through the `oneshot` reply senders on `Confirm`, `Choose`, `Secret` and `Line`.

The two frontends handle a running command differently. In the GUI (`zqa-gui/src/bridge.rs`), the engine thread races `session.dispatch()` against `cancel_rx` and `cmd_rx`. Cancelling drops the dispatch future, and a command sent while another runs is parked in a single `pending_command` slot. There is also a `TODO(ZOT-219)` noting that the detached title task survives cancellation. The CLI (`zqa/src/cli/app.rs`, `run_command`) `select!`s the dispatch against the event channel and renders each event with `handle_event`, which reads stdin synchronously for prompts. Rustyline is not active during a command and nothing handles Ctrl-C there, so a Ctrl-C mid-query most likely ends the process. We haven't tested that.

Finally, the storage plan in design doc 04 (Approved) matters here. It stores one immutable node per provider message, committed through a new `on_history_item` callback, and keeps requests append-only so that Anthropic's preserved thinking stays valid. It already says that steering injections fit in as ordinary nodes.

## Goals and non-goals

We want to be able to stop a running query from both the GUI and the CLI, and leave the conversation in a valid state that can be resumed. We want to steer a running query with free text that the model sees before its next call. Requests should stay append-only, so that preserved thinking and prompt caches survive a steer. And all of this should be one mechanism in `zqa-rag`, shared by all five providers and both frontends.

Some things are deliberately out of scope. Token streaming, and steering in the middle of a stream, is Phase 2 (the `TurnEvent` stream); this design leaves room for it. Headless mode (ZOT-78) won't support steering, because piped stdin is the command stream and there is no second input. SIGINT still stops a query there. A steer is conversation content only, so it can't change the frame mid-run: the system prompt, tool set and reasoning settings stay as they were. And subrequests, meaning title generation and the summarization subagent, never get a control channel.

## Design

Frontends write `Control` messages into one channel per session. The tool loop is the only reader, and it reports what it did back through `EngineEvent`s.

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

The highlighted boxes are the only two places where a control takes effect. Everything else just forwards it.

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
async fn send_message(&self, request: &mut ChatRequest<'_>) -> Result<CompletionApiResponse, SendError>;

pub struct CompletionApiResponse {
    // ...existing fields...
    pub outcome: RunOutcome,    // Completed | Stopped
    pub steered: Vec<String>,   // steers folded in, in order
    pub leftover: Vec<Control>, // unrun steers first, then Queue items
}

/// Breaking: wraps `LLMError` so a failed run can hand back input it had already buffered.
pub struct SendError {
    pub source: LLMError,
    pub leftover: Vec<Control>,
}
```

The receiver is a plain `&mut` field on the request, with no `Mutex` in the field's type (decided in review; how `zqa` stores the receiver between runs is up to `zqa`, see the wiring below). That means every caller changes `&request` to `&mut request`: `generate_title`, the summarization tool, `handle_query_cmd` and the per-provider tests. The channel is unbounded because controls are rare and tiny, and a frontend should never block on sending one. `LLMError` is a flat `thiserror` enum, so rather than adding a `leftover` field to every variant, `send_message` returns a small `SendError` wrapper.

`Queue` isn't really the loop's business. The loop never acts on one; it buffers each `Queue` it reads and hands it back in `response.leftover`, so nothing is lost, and `zqa` runs them (see the wiring below). A second channel would keep `Queue` out of `zqa-rag` entirely, but a single ordered channel preserves the order the user typed things in, which matters more.

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
    let turn = turn.map_err(|e| SendError { source: e, leftover: take_all(&mut buf) })?; // keep buffered input
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

Steers wait, as they do in pi (decided in review). `until_stop` buffers Steer and Queue and keeps polling the same future, so a model call or tool batch always runs to completion unless the user stops. That means a steer that arrives mid-call costs us that call. In return there is a single code path, and nothing is ever thrown away except a call the user explicitly stopped.

Where a steer lands depends on what the model just did. If it asked for tools, the steer is appended as text to the same user message that carries the tool results. If it answered with no tool calls, the steer opens a new user message and the loop carries on instead of returning. pi does the same thing: its loop keeps running while there are tool calls or steering messages.

Importantly, nothing is ever rebuilt. The steer is added to the generic `ChatHistoryItem` before it goes through the existing `Into<Vec<Self::HistoryItem>>` conversion, so no native item is popped or re-converted. Anthropic ends up with text blocks after the `tool_result` blocks in one user message, and OpenAI gets `function_call_output` items followed by a user message (checked in `openai.rs`).

A steer that arrives during the final, no-tools iteration gets one extra round (decided in review). It waits for that final answer, then raises the limit by two, once per run. The first of those two iterations has tools, so "also look for X" can actually be acted on, and the second is the usual final answer without tools. Raising it by one wouldn't be enough: `is_last_turn` is recomputed as `round_trips == limit - 1`, so the next iteration would be the last one again.

A Stop that shows up at the drain point, right after a tool batch finished, first pushes that batch's results (without any steer) and then returns. That way the round that just finished is kept, which is what review decided Stop should do.

When the loop ends, any steers still buffered (because the extra round was already used) and every `Queue` item go back in `response.leftover`, steers first, since pi also runs leftover steers before follow-ups. Errors get the same treatment. If a model call fails after `until_stop` has already pulled a Steer or Queue out of the channel, that text would otherwise be lost, so `SendError` carries it out in its `leftover` field and the handler treats it the same way as a Stop (below).

### `zqa`: wiring

`Context` gains a `control_rx: Option<Arc<tokio::sync::Mutex<UnboundedReceiver<Control>>>>`, and `Session` exposes a cloneable `SteerHandle` that wraps the sender. The CLI builds the pair in `run()`, next to the event channel.

`handle_query_cmd` clones the `Arc`, takes an owned lock on it with `lock_owned()` for the duration of the run, and passes `control: Some(&mut *guard)` on the main request only. The owned guard doesn't borrow `ctx`, so `ctx.emit` stays usable while the request holds the receiver. More importantly, the receiver itself never leaves `ctx`. If the run's future is dropped, whether by an early `?` return or by the GUI's hard cancel below, only the guard goes with it, so the lock is released and the next query finds the same receiver, and the frontend's `SteerHandle` keeps working. Taking the receiver out with `.take()` and putting it back with a guard would look simpler, but a dropped future would take the receiver with it and leave the session unable to steer.

Queued and leftover messages run as their own commands. After `send_message` returns, the handler pushes `response.leftover` (unrun steers first, then `Queue` items) onto a new `ctx.state.pending_inputs` queue. Then, after every command, query or not, `Session::dispatch` and the CLI's `run_command` look at whatever is still in the channel. They drop stale `Stop`s (for the same reason the bridge drains `cancel_rx`) and push each `Steer` or `Queue` onto `pending_inputs` in arrival order. They then keep dispatching until `pending_inputs` is empty, so a queued `/search` works too, and text typed while any command was running is neither lost nor run out of order. This replaces the bridge's single `pending_command` slot.

When a run ends by Stop or by an error, the unrun steers and queued messages don't run. They go back to the frontend in `EngineEvent::Unsent { messages }`, the same way pi puts queued text back into the editor when a run is aborted. This applies to the whole chain, not just the current query: a `Stopped` or error outcome also moves everything left on `pending_inputs` into `Unsent`. So does a Stop that the sweep finds while `pending_inputs` isn't empty, which is how a Stop sent during a non-query command, such as a queued `/search`, ends the chain.

That leaves three new events. `Steered { message }` fires when the loop folds a steer in, so the transcript shows it at the point where the model saw it. `Stopped` fires when a run ends by Stop, and `Unsent { messages }` is described above. Which crate owns them is the subject of the next section.

### Frontends

In the GUI, we add `EngineCommand::Steer(String)`, `EngineCommand::Queue(String)` and `EngineCommand::Stop`. Pressing Enter while a query runs sends a Steer by default. A config setting (for example `running_input = "steer" | "queue"`) flips that default, and a modifier sends the other one, like pi's Alt+Enter for follow-ups. While nothing is running, both simply become a `Dispatch`. The existing cancel path, which drops the future, stays around as the hard stop on a second click. A soft Stop has to wait for the loop to reach a `select!` point, and it keeps the partial run; sometimes you just want it gone. A hard cancel ends the chain the same way a Stop does. Because the cancel drops `Session::dispatch` before its usual sweep runs, the bridge does that cleanup itself after `UiEvent::Cancelled`. It locks the receiver (free by then, since the owned guard went with the future), drops any `Stop`s, and sends any `Steer` or `Queue` still in the channel to `Unsent`, along with everything on `pending_inputs`. Otherwise the soft Stop from the first click could stop the user's next query the moment it starts, and a stale steer could end up folded into an unrelated query. Steers and queued messages that the loop had already pulled into its own buffer are lost on a hard cancel. That seems acceptable for an action whose whole point is to throw the run away.

In the CLI, we add a `tokio::signal::ctrl_c()` arm to the `select!` in `run_command`. On Ctrl-C, it stops polling the dispatch and prompts on stdin. A line of text is sent as the configured default (Steer, unless set to Queue), and an empty line sends Stop. That read is synchronous, like `handle_event`'s prompt reads, and the dispatch isn't polled while it blocks, so the run is effectively paused while the user types. Doing it this way avoids a second concurrent stdin reader, which would fight with prompt replies.

## Where the events live

There are two sets of events. `zqa-rag` owns an `AgentEvent` enum for the things the loop knows about, and `zqa`'s `EngineEvent` wraps it in one variant, alongside the app-level events.

The reasoning is that `zqa-rag` already emits events; it just does it through callbacks. Today there are `on_text`, `on_reasoning` and `on_tool_call`, design doc 04 adds `on_history_item`, and Phase 2 adds `TurnEvent`. A good part of `handle_query_cmd` exists only to turn those callbacks into `EngineEvent`s, through `ResponseSegment` and three unbounded channels. Meanwhile some `EngineEvent`s really describe the loop rather than the app: `ToolCall` and `ToolResponse` are never constructed anywhere in `zqa`, only the GUI matches on them, and their own TODO says the `zqa-rag` hooks should change. Steering adds more facts that only the loop knows, namely when a steer was folded in and when a run stopped.

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

Wrapping has a cost: frontends match two levels deep (`EngineEvent::Agent(AgentEvent::Text(..))`), and `Display` has to delegate for that arm. The alternative is to flatten with a `From` impl in `zqa`. That keeps one level, but every new loop event then has to be added in both crates plus a translation arm, which is exactly what `handle_query_cmd` does today. Wrapping also gives subagents something for free: the summarization tool can give its sub-request its own sender and choose what to forward, which is the same concern as the branch's "keep reasoning and tool calls out of sub-request results" fix.

As for timing, `AgentEvent` comes after ZOT-248 and ZOT-249 merge, since that stack still has some work left (decided Oct 30). It is the first change after the merge, and steering builds on top of it, so the three steering variants land in the right crate from the start.

## Semantics: ZOT-219's three questions

ZOT-219 asks three questions about what steering means. Here is how this design answers them.

| Question | Answer | Why |
| --- | --- | --- |
| 1. Does steering text become a new user turn? | It is user-role text. After tool results it is appended to the same user message; after an answer with no tool calls it opens a new user message and the loop goes on. Either way it is not a new user turn in `zqa`'s sense: no new rendered summarize prompt, no new title, same run. | Both are placements that every provider accepts mid-loop without synthetic messages, and both keep the request append-only. |
| 2. Cancel outstanding tool calls, or wait? | **Steer waits** for tools that are already running. **Stop drops** them. | Every tool today is read-only, and its results are useful context for the steered call. Dropping them would leave a `tool_use` with no result, which every provider rejects, so a steer would need fake "cancelled" results. Stop ends the run, so nothing needs a result. |
| 3. Mid-stream, or only at the next model call? | At the next model call, **after the call in flight finishes** (decided in review, as pi does). | Nothing but a stopped call is ever thrown away, so there is one code path and no half-sent request to reason about. The cost is paying for a turn the user may already have rejected; once Phase 2 streams that turn, the user can at least see what they are steering. |

A steer is sent as plain user text with no wrapper, as in pi (decided in review), and the transcript shows the same text. When several steers are queued, they are delivered one at a time, also following pi, whose default `steeringMode` is `one-at-a-time` (the alternative is `all`). Each drain point takes the oldest steer and the rest wait for the next call. A config switch could offer `all` later.

It's worth spelling out how pi handles timing, since we copy it. pi never aborts an in-flight call. A steer waits until the current assistant turn and its tools finish, and the loop keeps going while there are tool calls or steering messages, so a steer that arrives during a final answer still gets a reply. This design behaves the same way.

## History persistence and reasoning preservation

A steer only ever appends after the last native item, and the only call that is ever discarded is one cut off by Stop. So the request stays append-only and nothing signed is lost.

### Reasoning (ZOT-230)

Inside the loop, this just works. `provider_history` holds native items, including Anthropic `thinking` blocks with their signatures, OpenAI reasoning items and Gemini thought signatures. A steer adds text after the tool results or after the answer, so the prefix, and every preserved thinking block in it, is unchanged. The frame is unchanged too, because a steer never touches the tools, the system prompt or the reasoning settings.

A stopped call doesn't leave a gap either. Its response was never appended, so there is nothing to strip out. That matters because Anthropic only allows dropping thinking from the start of history, never from the middle, and this design never needs to. Nothing native is rebuilt along the way: a steer is added to the generic item before conversion, and tool-result items are built locally from generic `ToolCallResponse`s anyway.

This is also the main reason steering can't be done as cancel-and-redispatch. A new dispatch rebuilds history through `From<ChatHistoryItem>`, which drops `ChatHistoryContent::Reasoning` (`anthropic.rs`, line 54 on the branch). The steered call would then continue a `tool_use` turn without its thinking block.

### Persistence, with today's JSON history

A steer needs no format change. Its text lives in a user `ChatHistoryItem`, either the tool-results item or a new one after an answer, so `history_additions` carries it to disk. It's added before the item is pushed, so what gets saved matches what was sent.

For a Stop, `send_message` returns `Ok` with `RunOutcome::Stopped` rather than an error. `handle_query_cmd` saves the query plus `history_additions`, trimmed to the last complete round, which means dropping a trailing assistant item whose tool calls have no results. Usage is still recorded. This is an improvement over today, where a GUI cancel loses the whole run, including the user's query.

As an aside on ZOT-211: both `master` and the branch already do `history.extend(response.history_additions)`, which includes tool calls and results. The issue text says only the query and final answer are kept, which looks out of date. That's inferred from the code, so it's worth confirming before closing the issue.

### Persistence, with the conversation graph (design doc 04)

With the conversation graph, a steer that follows tool results becomes part of the `ToolResults` node, since it's in the same provider message. The node gains a `steering: Vec<String>` field for display text, next to `provenance`. A steer that follows an answer with no tool calls is its own `User` node, a child of that `Assistant` node, and the loop's next call continues from it.

A Stop keeps the finished tool rounds (decided in review). The stopped call becomes a `Failed { error: "stopped" }` node with the attempt marked `cancelled`, and the head moves to the last complete `ToolResults` node, so a follow-up question can reuse whatever was retrieved.

That needs two amendments to design doc 04, which were agreed in review but haven't been made yet. Under *Settled nodes*, a `ToolResults` node whose calls all have results should count as settled, so that a user message may follow it after a Stop. And under *Write path*, step 4, a Stop should move the head to that node instead of leaving it where the run started. Errors and refusals keep today's rule.

## Edge cases

The table below walks through when each control can arrive and what happens.

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

Nothing here gets in the way of the `AgentDriver` / `TurnEvent` split that ZOT-219 plans. The control channel and the `select!` points move into the driver unchanged.

With streaming, a mid-stream Stop drops the turn stream exactly the way Phase 1 drops `send_once`. The partial text has already been shown, so it should stay in the transcript, marked as interrupted, but it should never enter history. A mid-stream Steer still waits for the turn to end, as in pi; streaming only lets the user see what they are steering.

Partial reasoning has to be thrown away, though. A thinking block that's cut off mid-stream has no signature and can't be replayed. Storing it as `Failed { partial }` (design doc 04) keeps it visible without ever sending it. As a bonus, the `ToolCall` event TODO in `io.rs` gets fixed for free: a streamed `ToolCall` arrives before the tool runs, so `EngineEvent::ToolCall` can finally fire at the right time.

## Rejected alternatives

The simplest option would be to cancel the run and re-dispatch with the steer as a new query. It needs no `zqa-rag` change at all, but it rebuilds history from canonical items, which loses reasoning and, on error paths, the run's tool results (see History persistence).

We also considered steering through the existing `EngineEvent` reply channels, as a standing `Line` prompt the frontend could answer at any time. But events are outbound by contract and replies are one-shot, and a standing prompt would block the CLI's single stdin reader.

A `CancellationToken` for Stop alone, with Steer deferred, would be simpler at first. Steer would then need a second mechanism later, though, and one channel covers both for almost the same cost.

Cancelling running tools when a steer arrives was rejected because it needs synthetic tool results to keep the request valid, and it throws away work the model probably wants. Aborting the in-flight model call on a steer was rejected in review. It would save paying for a turn the user has already rejected, but it needs a second path (abort, re-convert the tail, resend, plus a `CallAborted` event), and it differs from pi.

## Open questions

The main open question is how each provider lays out text that follows tool results. Anthropic and OpenAI are covered above. Gemini (text parts beside `functionResponse` parts in one content), OpenRouter (role `tool` messages, then a user message) and Ollama each need a fixture test before this ships.

### Decided in review (Oct 29 and 30)

- Stop keeps the finished tool rounds; design doc 04 gets the amendments listed under History persistence.
- A steer during the final, no-tools iteration buys one more tool round, once per run.
- Steer text follows pi: plain user text, no wrapper, delivered one at a time.
- Steers wait, as in pi: a model call or tool batch always finishes unless the user stops, and a steer after a plain answer continues the loop.
- The GUI defaults to Steer while running, configurable to Queue, with a modifier for the other.
- `Control` gains `Queue`, and the receiver is a plain `&mut` field on `ChatRequest`; breaking callers is fine.
- `AgentEvent` lands after the ZOT-248/249 stack merges (Oct 30), as the first change before steering.
