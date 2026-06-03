# Conversational Inferencer — Unified Inbox & Event Loop Plan

> **Status:** Draft **v1.5** — implementer-ready (code-correctness + audit-table cleanup pass; HIGH bug in `enable_inbox` body fixed; +AC-I12 for re-entrancy guard; +chronological audit table)
> **Author:** Tony Chen (synthesized from session discussion Thu 2026‑05‑28 15:40‑15:50)
> **Created:** 2026‑05‑28 15:52; **v1.1:** 15:59; **v1.2:** 17:17; **v1.3:** 17:48; **v1.4:** 18:22; **v1.5:** 18:53 (HIGH bug fix: `enable_inbox` body now stores `_auto_shutdown_on_sop_complete`; +AC-I12; +chronological audit table)
> **Empirical grounding:** All cited line numbers verified against the live tree on 2026‑05‑28 15:52 via 3 parallel exploration subagents + 1 confirmation grep pass.
> **Supersedes:** N/A
> **Companion plans (compatible):** `workflows_and_sop/sop_model_a_with_commands_and_resumability_plan.md` v2.6 (uses but does not require this work); `workflows_and_sop/multi_sop_focus_and_tool_concurrency_plan.md` v1.1 (cleanly composes with the inbox).
> **Integration history:** v1.1 over-corrected by adopting 4 changes from Claude's v1 plan. Claude's v2 plan (2026‑05‑28 16:10) reversed Claude's own position on 3 of those 4 and explicitly states "If I had to pick one plan: Tony's plan." v1.2 reverts to v1.0's architecture (typed InboxItem with 3 variants; `run_agentic_loop` unchanged; ToolCompletion as wake-up signal not content carrier; `request_shutdown()` flag) while KEEPING from v1.1 the one change Claude's v2 still endorses: **v1 scope-cut** (AgentFoundation only; OpenStartup ConversationService deferred to a future v2 of this plan).

---

## §‑1 Provenance and audit history

| v# | Date (UTC) | What |
|---|---|---|
| 1.0 | 2026‑05‑28 15:52 | Initial draft from session discussion. The discussion identified a **real architectural fault**: ConversationalInferencer (CI) is request‑response only (`run_agentic_loop(content) → AgenticResult`); async tools fire a background `asyncio.create_task(_run_async())` (CI:1049), set `_async_tool_dispatched=True` (CI:1050), and the loop exits (CI:483/526/586). When the background task finishes, it appends a "user" tool‑results message to `_messages` and applies `context_updates`, but **the CI is dormant** — no subsequent loop is triggered. For an SOP yolo run (or any user awaiting a tool they initiated), this is broken UX: the user must type "continue" before the LLM ever sees the result they asked for. The fix is a unified **inbox queue**: all input sources — user messages, async tool completions, synthetic SOP continuations — drop items onto a single `asyncio.Queue`; the CI runs a long‑lived event loop that drains it. |
| **1.1** | **2026‑05‑28 15:59** | **Round-1 integration with Claude's parallel-authored plan.** Adopted 4 changes from Claude: (1) Extract `_run_inner_loop` from `run_agentic_loop` (the latter becomes a wrapper); (2) Async tool result as content-in-queue (not bare signal); (3) Renamed `inbox_put`/`inbox_put_user` → `submit(content, source=...)`; (4) Replaced `_auto_shutdown_on_sop_complete` flag with 3 inline exit conditions. Retained from v1.0: typed `InboxItem`; back-compat guard; 12 ACs / 7 risks / phased rollout / §10 alternatives table. Reduced server-side scope: Phase 4 (OpenStartup) deferred to v2. |
| **1.2** | **2026‑05‑28 17:17** | **Round-2 integration; REVERTS 3 of v1.1's 4 changes.** Claude's plan expanded (124L → 259L at 16:10) and explicitly reversed its earlier position on changes 1–3 of v1.1, with Claude itself now stating: "If I had to pick one plan: **Tony's plan**." Critical re-examination confirms Claude's v2 position is architecturally more honest: (1) **Reverted `_run_inner_loop` extraction** — keeping `run_agentic_loop` unchanged means zero risk to the existing 115 tests, no inner/outer ambiguity, and `run()` simply layers on top. (2) **Reverted content-in-item** — `ToolCompletion` is now a pure wake-up signal; the tool result is already in `_messages` (where `_run_async`'s back-compat path puts it, and where the next `run_agentic_loop` render naturally reads it). This avoids two parallel paths through which tool results enter the system. (3) **Reverted `submit()` rename** — restored `inbox_put(item)` + convenience `inbox_put_user(content)`; matches the "what" + "for what type" naming pattern. (4) **Reverted 3-condition inline exit** — restored `request_shutdown()` flag, which is decoupled from SOP specifics and reusable for any termination policy. Kept from v1.1: the **v1 scope-cut** (OpenStartup ConversationService deferred to a future v2 of this plan; v1 ships AgentFoundation-only) — Claude's v2 still endorses this. Net effect: plan returns to v1.0's architecture but ships with the smaller v1 scope. |
| **1.3** | **2026‑05‑28 17:48** | **Empirical-fidelity pass.** External-review agent ran 12 checks against the live CI code and surfaced 10 valid issues + 2 rejected. Applied corrections: (1) Issue 1: §0.1 now correctly describes cli.py's two **branches** (yolo + interactive `while True`), not "two sequential calls." (2) Issues 2+3: §2.3 `_run_async` pseudocode now matches live code byte-for-byte — `add_message` is CONDITIONAL on `hasattr(result, "result")` (CI:1040), and the order is add → check (not check → add). (3) Issue 4: Risk R7 corrected — `_NON_SERIALIZABLE_PRIOR_CONTEXT_KEYS` was removed during the SOPState refactor; pause/resume now uses `SOPState.to_dict/from_dict`. The mitigation is now to declare `_inbox` / `_default_interactive` with `attrib(init=False, repr=False)` and have `_serialize_pause_state` filter them explicitly. (4) Issue 5: corrected `run_agentic_loop` size from "165 lines" to "~450 lines (CI:166-618)". (5) Issue 7: §0.1 contradiction fixed. (6) Issue 8: `_CONTINUE_AFTER_TOOLS` stays at CI:81; the new `inbox.py` IMPORTS it. (7) Issue 11: attrs-style alignment. (8) Issue 12: off-by-one fixed. Added Risk R8 for `_active_async_task` single-slot. |
| **1.4** | **2026‑05‑28 18:22** | **Internal-consistency pass.** External-review agent ran 7 internal-contradiction checks against v1.3. All 7 valid; applied as in-place corrections: (1) Issues 1+2: §3.1 now declares all 6 NEW CI fields (was: 1 in code, 5 only in narrative); §3.2 `enable_inbox` row matches §3.1 signature. (2) Issue 3: §9.1 inbox.py correctly omits `_CONTINUE_AFTER_TOOLS` (stays at CI:81); only NEW `_SYNTHETIC_CONTINUE` lives in inbox.py. (3) Issues 4+5: §2.4 "Today" col honestly describes cli.py TWO branches (yolo @170 + interactive @187); "After" col uses `inbox_put_user(request)` + `await ci.run()` (no params). (4) Issue 6: §12 v1.0 → v1.4. (5) Issue 7: removed `_shutdown_requested = False` auto-reset; added Risk R9. Zero architectural change. |
| **1.5** | **2026‑05‑28 18:51** | **Code-correctness + audit-table cleanup pass.** External-review agent ran 4 checks against v1.4. All 4 valid; applied as in-place corrections: (1) Issue 1 (HIGH bug): `enable_inbox` body in §3.1 added `self._auto_shutdown_on_sop_complete = auto_shutdown_on_sop_complete` — the parameter was accepted but never stored, which would have made SOP CLI yolo branch hang forever (AC-I3 would catch on first run). (2) Issue 2: §9.2 corrected "+5 new fields / +6 new methods" → "+6 new fields / +7 new methods" — `_running` was in §3.1 but missing from §9.2's list. (3) Issue 3: audit history table reordered to strict chronological order (1.0 → 1.1 → 1.2 → 1.3 → 1.4 → 1.5); v1.4 was previously inserted out-of-order at top. (4) Issue 4: added AC-I13 explicitly testing the `_running` re-entrancy guard — Q5 in §11 said "Add AC for this" but no AC had been added (AC-I12 was already taken for the SyntheticContinue test). Zero architectural changes; one real bug fix. |

---

## §0 Scope and non‑scope

### §0.1 In scope (v1)

- Add an **inbox** to `ConversationalInferencer`: a single `asyncio.Queue[InboxItem]` field that any input source can put items onto.
- Add an **event loop** entry point `await ci.run()` (long‑lived) that drains the inbox sequentially and invokes the existing agentic logic per item.
- Keep `run_agentic_loop(content)` as a **public single‑turn API** for callers that want request‑response semantics (CLI today, tests, BTA workers). The new `run()` is additive, not replacement.
- When an async tool completes, `_run_async()` puts a `ToolCompletion` item onto the inbox. The event loop dequeues it and re‑enters the agentic loop with the canonical "continue from tool result" content.
- An **idle gate**: if the inbox is empty AND no LLM call is in‑flight AND no interactive prompt is awaiting user response, the dequeue returns immediately and processes the item. Otherwise items wait FIFO.
- **One caller wired in v1** (v1.3 fix from v1.0's two-caller scope; v1.2 already scope-cut server-side; v1.3 fixes the §0.1 contradiction): SOP CLI at `resources/tools/sop/cli.py:170` (the `if yolo:` branch) and `:187` (the `else: while True:` interactive branch). The v1 migration replaces BOTH branches' explicit `run_agentic_loop(...)` calls — yolo branch becomes `enable_inbox(auto_shutdown_on_sop_complete=True)` + `inbox_put_user(request)` + `await run()`; interactive branch becomes `enable_inbox()` + a small input-reader loop that calls `inbox_put_user(user_line)` per typed line + `await run()` running concurrently in the background. **OpenStartup `ConversationService` is NOT changed in v1** — deferred to a future v2 of this plan; see §2.4.

### §0.2 Out of scope (v1, deferred)

- Multi‑SOP focus routing across the inbox (lives in `multi_sop_focus_and_tool_concurrency_plan.md` v1.1; this plan does NOT preempt that work).
- Cross‑process inbox (e.g. Redis, SQS). v1 is in‑process `asyncio.Queue` only.
- Inbox persistence across CI restarts (the inbox is ephemeral; tool completions that race against a shutdown are still recovered via the existing `_messages` append in `_run_async`).
- Reordering, deduplication, priority lanes — v1 is strict FIFO.
- Removing `run_agentic_loop` — it remains the single‑turn primitive that the event loop itself calls.

### §0.3 What this plan deliberately does NOT do

- **DO NOT** make `_run_async` recursively call `run_agentic_loop` (rejected in discussion at 15:43 — re‑entrant, can stack).
- **DO NOT** add a callback/event mechanism in the server layer ("server listens for task completion") — that was the second rejected option; it scatters async‑completion logic across two repos.
- **DO NOT** poll. The inbox uses `asyncio.Queue.get()` which **blocks the coroutine until an item arrives** — event‑driven, zero wasted cycles.

---

## §1 Problem statement (empirical)

| Layer | Today | Pain |
|---|---|---|
| CI lifecycle | Created fresh per turn by `ConversationService` (`conversation_service.py:639` calls `run_agentic_loop` once and returns) | Async tool background tasks outlive the CI's loop; their result message lands on `_messages` but nothing triggers re‑inference. |
| Async tool path | `_run_async` at `conversational_inferencer.py:1034‑1046` appends tool result to messages and applies `context_updates`, but does NOT re‑enter the agentic loop | SOP yolo with `task`/`create_role`/`role_setup` (all `asynchronous: true`, verified via tool.json grep) stalls after every async tool — the SOP cannot self‑drive. |
| Multiple input sources | Only one: the `content` argument to `run_agentic_loop` | No clean place to enqueue: (a) async tool completions, (b) synthetic SOP "continue" messages, (c) future scheduled events / webhooks. Each ad‑hoc fix would special‑case its own callback path. |
| User mental model | "I asked the tool to run; when it's done, the LLM should respond" | "I asked, the tool ran for 15 min, now I have to type 'continue' to see the result I already asked for" — the discussion identified this as broken UX even outside SOPs (15:48). |

The unifying root cause: **the CI has no way to receive input other than an explicit `run_agentic_loop` call**. Async tools, SOP auto‑continues, and user messages are conceptually all "input events," but only one is plumbed.

---

## §2 Target architecture

### §2.1 The inbox

A single field on `ConversationalInferencer`:

```python
self._inbox: asyncio.Queue[InboxItem] = asyncio.Queue()
```

`InboxItem` is a discriminated-union of 3 typed dataclasses (v1.2 restored from v1.0, after Claude's v2 reversed position and endorsed this design):

```python
from typing import Literal
import attrs

@attrs.frozen        # NOTE v1.3: aliased re-export from `attr`; we use the modern aliases
                     # ONLY for these inbox value-types (they are NEW dataclasses and do not
                     # participate in the InferencerBase classic-attrs hierarchy).
                     # All NEW fields added to ConversationalInferencer itself MUST use the
                     # classic `attrib(...)` style (see §7 file inventory) — see Issue 11
                     # in v1.3 audit row.
class UserMessage:
    """A real user input event (typed message, voice transcript, etc.)."""
    kind: Literal["user_message"] = "user_message"
    content: str = ""
    source: str = "user"             # for logging: "user", "cli", "websocket"

@attrs.frozen
class ToolCompletion:
    """An async tool finished. Result ALREADY in self._messages (added by _run_async).
    This item is a pure wake-up signal for the event loop — NOT a content carrier."""
    kind: Literal["tool_completion"] = "tool_completion"
    tool_name: str = ""              # for logging only

@attrs.frozen
class SyntheticContinue:
    """An internal driver event (e.g. SOP yolo phase-advance with no tool call)."""
    kind: Literal["synthetic_continue"] = "synthetic_continue"
    reason: str = ""                 # for logging: "sop_phase_advance", "test_inject"

InboxItem = UserMessage | ToolCompletion | SyntheticContinue
```

**Why `ToolCompletion` carries NO content (single source of truth for tool results)** — this is the v1.2 reversal of v1.1's "content-in-item" idea: the result is already in `_messages` (put there by `_run_async`, which today already does `add_message("user", f"{_TOOL_RESULTS_PREFIX}\n{name}: {result}")` at CI:1041). The inbox item is purely "wake up and re-enter the agentic loop." This means: ONE path through which tool results enter the system (`_messages`); the inbox item is a pure scheduling event. Two parallel paths would risk inconsistency.

**Multi-concurrent-tool correctness:** if N tools complete near-simultaneously, each `_run_async` calls `add_message` (appending its result to `_messages`) and then `inbox.put_nowait(ToolCompletion(name))`. The agentic loop wakes once per item; the first wake sees ALL accumulated tool results in `_messages` (and the LLM responds to all of them in one turn — which is the correct, natural batching behaviour). Subsequent wakes are idempotent no-ops (the agentic loop re-renders with no new messages and finds the LLM has nothing new to say). This is the right behaviour: the LLM doesn't need to be re-prompted for each tool individually; the messages are conversational state, not events.

### §2.2 The event loop (v1.2 — restored v1.0 design; `run_agentic_loop` UNCHANGED)

A new public method `async def run()` on `ConversationalInferencer`. **`run_agentic_loop` is NOT extracted, NOT renamed, NOT moved.** It stays exactly as today's ~450-line method (CI:166-618; v1.3 corrected from v1.2's incorrect "165-line" estimate). `run()` is purely additive:

```python
async def run(self) -> AgenticResult | None:
    """
    Long-lived event loop: drain the inbox sequentially, invoking the
    existing run_agentic_loop per item. Returns when request_shutdown()
    is called.

    Returns the most recent AgenticResult, or None if no item was processed.
    """
    if self._inbox is None:
        raise RuntimeError("Inbox not enabled; call enable_inbox() first")
    if self._running:
        raise RuntimeError("run() is already executing")
    self._running = True
    try:
        last_result: AgenticResult | None = None
        while not self._shutdown_requested:
            item = await self._inbox.get()
            try:
                content = self._content_for_item(item)
                if content is None:
                    continue  # filtered (e.g. logging-only event)
                last_result = await self.run_agentic_loop(
                    content=content,
                    interactive=self._default_interactive,
                    turn_number=self._next_turn_number(),
                )
                # If the agentic loop dispatched an async tool, do NOT re-loop.
                # The loop already exited with _async_tool_dispatched=True.
                # The eventual _run_async will put a ToolCompletion onto the
                # inbox and the next get() will pick it up.
            except Exception as e:
                logger.exception("Inbox item %r failed: %s", item, e)
                # v1: log + continue. v2 can route to an error handler.
            finally:
                self._inbox.task_done()
        return last_result
    finally:
        self._running = False
        # NOTE v1.4: we deliberately do NOT reset `_shutdown_requested` here.
        # Callers that need to know WHY `run()` returned must read `self.sop_state`,
        # the returned `AgenticResult`, or any of the explicit exit conditions —
        # not this flag. Resetting here would mask whether the caller asked for
        # shutdown vs. natural completion. To re-enable `run()` for a new
        # lifecycle, the caller explicitly sets `self._shutdown_requested = False`
        # (or constructs a fresh CI). See Risk R9.

def _content_for_item(self, item: InboxItem) -> str | None:
    if isinstance(item, UserMessage):
        return item.content
    if isinstance(item, ToolCompletion):
        # Result already in _messages; this is a wake-up. Use the canonical
        # "continue and process tool results" content so the next LLM turn
        # explicitly knows it was triggered by tool completion.
        return _CONTINUE_AFTER_TOOLS   # existing constant at CI module level
    if isinstance(item, SyntheticContinue):
        return _SYNTHETIC_CONTINUE     # NEW constant added by this plan
    return None
```

**Idle gate semantics** (the "if queue empty and no ongoing event, fire directly" rule from the discussion at 15:50): satisfied automatically by `asyncio.Queue.get()` — when empty, the coroutine suspends until an item arrives; no CPU spin. When an item arrives during an in-flight LLM call, it waits in the queue because `run_agentic_loop` is still running; the next `get()` picks it up after the prior turn completes. No explicit lock or "is busy" flag is needed — Python's single-threaded asyncio model serializes for us.

**Why `request_shutdown()` flag, not inline exit conditions** (v1.2 reversal): v1.1 inlined 3 conditions (PausedResult / SOP completed / queue-empty-and-no-SOP) inside `run()`. Two problems: (a) couples `run()` to SOP semantics (`self.sop_state` / `PhaseStatus.COMPLETED`) — but `run()` should be reusable for any persistent CI scenario, not SOP-specific; (b) hard-codes the policy ("exit when queue empty"), preventing legitimate idle-and-wait use cases like a server that wants to keep `run()` alive across user pauses. The flag-based design lets each caller set its own termination policy via a small `_check_phase_completion` hook (see §3.3) — SOPs flip the flag on completion; servers keep it false forever; tests flip it explicitly.

### §2.3 Routing async tool completion onto the inbox

The single change in `_run_async()` (CI:1035‑1049; v1.3 corrected from v1.2's off-by-one "1034" — line 1034 is blank, the inner function starts at 1035). **v1.3 restores live-code fidelity**: result append remains CONDITIONAL on `hasattr(result, "result")` (matching CI:1040), the order is `add_message` → `_check_phase_completion` (matching CI:1040-1045), and we additionally drop a bare `ToolCompletion` wake-up signal IF AND ONLY IF the inbox is enabled:

```python
async def _run_async() -> None:
    try:
        result = await executor(canonical, tool_call.arguments)
        if hasattr(result, "context_updates") and result.context_updates:
            self.update_prior_context(**result.context_updates)

        # ── UNCHANGED from today (CI:1040-1044): add_message is CONDITIONAL on
        #     the executor returning a result-bearing object. v1.3: matches live code
        #     exactly — including the `if hasattr(result, "result"):` guard. ──
        if hasattr(result, "result"):
            self.add_message(
                "user",
                f"{_TOOL_RESULTS_PREFIX}\n{canonical}: {result.result}",
            )

        # ── UNCHANGED from today (CI:1045): phase-completion check AFTER add_message ──
        self._check_phase_completion(tool_name=canonical)

        # ── NEW (v1.0/v1.2 — only if inbox enabled): wake the event loop ──
        if self._inbox is not None:
            try:
                self._inbox.put_nowait(ToolCompletion(tool_name=canonical))
            except asyncio.QueueFull:
                logger.warning(
                    "Inbox full; ToolCompletion wake for %s dropped — result still "
                    "in _messages and will be picked up by next user message",
                    canonical,
                )
    except Exception as e:
        logger.error("Async tool %s failed: %s", canonical, e)
```

**Back‑compat:** If `self._inbox is None` (no event loop running — e.g. legacy `run_agentic_loop`‑only callers), behaviour is **byte‑identical** to today: result is appended to `_messages`, context updated, no continuation. This is the current "stall" behaviour, retained intentionally for callers that did not opt in. **Verified the always-execute `add_message` line matches today's CI:1041 exactly.**

**Why this design is correct under concurrent completions** (v1.2 reversal of v1.1's "content-in-item" concern): if N tools complete near-simultaneously, each `_run_async`:
1. Appends its result to `_messages` (CONVERSATION STATE — accumulates correctly across all N completions)
2. Drops a `ToolCompletion(name)` wake-up signal on the inbox (EVENT — one per tool)

The event loop dequeues wake-up signals; each wake-up runs `run_agentic_loop` once. The first wake-up sees ALL accumulated results in `_messages` (because `add_message` was called by all N tasks before this LLM turn rendered). The LLM responds to ALL of them in one turn — which is the **correct** batching behaviour. Subsequent wake-ups are idempotent: the agentic loop renders again, the LLM sees the same conversation state, has nothing new to say, and `run_agentic_loop` returns with no tool calls. The next `_inbox.get()` then blocks normally.

**Why this is better than v1.1's content-carrier design:** the v1.1 design would have N parallel LLM turns each processing ONE tool result, which (a) wastes N-1 LLM calls (the model would still see all results in `_messages` on each call anyway, since `_messages` is the canonical conversation state), and (b) creates a subtle inconsistency where the same result text would appear both in the inbox item AND in `_messages` — double-bookkeeping.

### §2.4 The two opt‑in entry points

**v1.1 scope-cut (Claude's pragmatic call):** v1 ships AgentFoundation-only — just the CI and SOP CLI changes. OpenStartup's `ConversationService` is left unchanged because today it already gets correct behaviour (it just calls `run_agentic_loop` per turn; async tool results sit in `_messages` and the next user message naturally consumes them via the back-compat branch in §2.3). The server-side migration to a persistent `await ci.run()` is deferred to v2 of this plan and gated behind `OPENTEAM_CI_INBOX=1`. This keeps v1 cross-repo-change-free.

| Caller | Today | After v1.1 |
|---|---|---|
| SOP CLI (`resources/tools/sop/cli.py:170,187`) | TWO BRANCHES (v1.4 fix): the `if yolo:` branch (line 170) calls `await ci.run_agentic_loop(...)` ONCE with the full `/sop ...` request; the `else: while True:` interactive branch (line 187) calls `await ci.run_agentic_loop(...)` once per user-typed input. Neither branch handles async tool completion — `[Continue...]` user-typed in interactive mode is what makes it work today. | YOLO branch: `ci.enable_inbox(interactive, auto_shutdown_on_sop_complete=True)` + `ci.inbox_put_user(request)` + `await ci.run()`. INTERACTIVE branch: `ci.enable_inbox(interactive)` + spawn a background `run()` task + read user input in a foreground loop and call `ci.inbox_put_user(line)` per line; on `Ctrl‑D` call `ci.request_shutdown()` and `await` the background task. **`run()` itself takes NO parameters** — initial content is delivered via `inbox_put_user()` (v1.4 fix from v1.3's stale `initial_content=` reference). |
| ConversationService (`OpenStartup .../conversation_service.py:639`) | Per‑turn: creates CI, calls `await inferencer.run_agentic_loop(...)`, discards CI | **No change in v1.** Per-turn lifecycle preserved. Async tool results land in `_messages` via the back-compat branch in §2.3 (because the server never calls `enable_inbox()`), exactly as today. v2 will add an opt-in `OPENTEAM_CI_INBOX=1` path for persistent CI + queue-driven turns. |
| BTA worker subagents | Construct CI, call `run_agentic_loop` once, return | **No change.** Never call `enable_inbox()` → take legacy path automatically. |
| Tests mocking `run_agentic_loop` | Patch / mock `run_agentic_loop` directly | **No change.** v1.2 reverted v1.1's extraction; `run_agentic_loop` stays as today's ~450-line method (signature byte-identical). v1.4: removed unfounded "~50 tests" count — no empirical verification was ever done for it. |

---

## §3 Detailed design

### §3.1 New CI fields

```python
# In ConversationalInferencer.__attrs_post_init__ or factory:
self._inbox: asyncio.Queue[InboxItem] | None = None  # set by enable_inbox()
self._shutdown_requested: bool = False                # set by request_shutdown()
self._default_interactive: InteractiveBase | None = None  # captured in enable_inbox()
self._auto_shutdown_on_sop_complete: bool = False     # v1.4: declare alongside other fields
self._turn_counter: int = 0                           # incremented by _next_turn_number()
self._running: bool = False                           # re-entrancy guard for run()
```

`self._inbox` defaults to `None` (back‑compat). Callers explicitly opt in via:

```python
def enable_inbox(
    self,
    interactive: InteractiveBase | None = None,
    *,
    auto_shutdown_on_sop_complete: bool = False,
    maxsize: int = 0,
) -> None:
    """Enable the inbox event loop. Must be called before run()."""
    if self._inbox is not None:
        raise RuntimeError("Inbox already enabled")
    self._inbox = asyncio.Queue(maxsize=maxsize)
    self._default_interactive = interactive
    self._auto_shutdown_on_sop_complete = auto_shutdown_on_sop_complete
```

**v1.5 fix (HIGH bug from v1.4 audit):** the `auto_shutdown_on_sop_complete` parameter MUST be assigned to the corresponding field. Without this line, the field stays at its init default of `False` and the §3.3 `_check_phase_completion` bridge would never fire `request_shutdown()`, causing the SOP CLI yolo branch to hang forever after SOP completion. AC-I3 explicitly verifies the auto-shutdown pathway; without the assignment that test would catch the bug at first-run.

### §3.2 Public API additions (v1.2 — restored v1.0 design)

| Method | Signature | Purpose |
|---|---|---|
| `enable_inbox(interactive=None, *, auto_shutdown_on_sop_complete=False, maxsize=0)` | → None | Opt-in setup; creates `self._inbox` (asyncio.Queue with given maxsize); captures `interactive` in `self._default_interactive`; sets `self._auto_shutdown_on_sop_complete` flag (consumed by `_check_phase_completion` bridge §3.3). Second call raises `RuntimeError` (AC-I1). v1.4: this signature matches the code in §3.1; older drafts mismatched. |
| `inbox_put(item)` | `InboxItem` → None | Non-blocking enqueue (uses `put_nowait` internally). Producers of `ToolCompletion` / `SyntheticContinue` use this directly. |
| `inbox_put_user(content, source="user")` | `(str, str)` → None | Convenience wrapper for the common case: `self.inbox_put(UserMessage(content=content, source=source))`. |
| `request_shutdown()` | none → None | Cooperative termination: sets `self._shutdown_requested = True`; current `run_agentic_loop` finishes its iterations, then `run()` returns. |
| `run()` | → `AgenticResult \| None` | Long-lived event loop. Drains inbox; returns when `request_shutdown()` was called. Raises `RuntimeError` if `enable_inbox` not called or if `_running` already True. |
| `_content_for_item(item)` | `InboxItem` → `str \| None` | Private; maps each typed inbox item to the content string passed into `run_agentic_loop`. Returns `None` for events that should be silently skipped (reserved for v2 — currently never returns None). |
| `_next_turn_number()` | → int | Private; reads/increments `self._turn_counter` for per-turn artefact directory names. |

**Not in v1.2:** `submit()` (v1.1's renamed `inbox_put_user`) — reverted. `inbox_put` / `inbox_put_user` better matches the codebase's two-axis naming (the "what" — put — and "for what type" — user). `_run_inner_loop` extraction — reverted; `run_agentic_loop` stays unchanged.

### §3.3 Phase-completion → shutdown bridge (v1.2 — restored v1.0 design)

The SOP CLI needs `run()` to return when the SOP finishes. v1.2 uses a small hook in `_check_phase_completion` that flips the shutdown flag when the SOP tracker reports completion AND the caller opted in to auto-shutdown:

```python
# Inside _check_phase_completion, after advancing/completing:
if (
    self._auto_shutdown_on_sop_complete
    and self.sop_state is not None
    and self.sop_state.phase_status == PhaseStatus.COMPLETED
    and self.sop_state.current_phase is None
):
    self.request_shutdown()
```

This keeps shutdown policy decoupled from SOP specifics — the SOP code only sets a single flag at `enable_inbox` time. Any future caller (test harness, server, REPL) can compose its own termination policy by reading `self._inbox`, `self.sop_state`, etc. and calling `request_shutdown()` from wherever it wants — without `run()` itself needing to know about SOP semantics.

### §3.4 Interaction with the existing `_async_tool_dispatched` exit

The exit blocks at CI:483/526/586 stay unchanged: when an async tool fires, the current `run_agentic_loop` invocation returns to the outer `run()` loop. `await self._inbox.get()` then suspends — the inbox is empty because the tool hasn't finished yet. When `_run_async` finally completes and calls `self._inbox.put_nowait(ToolCompletion(tool_name=canonical))`, the suspended `get()` wakes up; the next `run_agentic_loop` call uses `_content_for_item(ToolCompletion)` → `_CONTINUE_AFTER_TOOLS`, and the LLM renders with the tool result already in `_messages` (placed there by the same `_run_async` BEFORE the wake-up signal).

**Important:** the `AgenticResult` returned by the per‑turn `run_agentic_loop` after an async dispatch carries `_async_tool_dispatched=True` (or equivalent — verify with grep on the result fields). The event loop sees this and **does not** loop back immediately; it lets the inbox wake it up. This avoids a busy spin while the tool runs.

### §3.5 Default interactive plumbing

`run_agentic_loop` already takes an optional `interactive` parameter. The event loop passes `self._default_interactive` (set in `enable_inbox`) so all turns share one InteractiveBase instance — exactly what the WebSocket / CLI / Rich terminal cases need. This naturally composes with the InteractionSerializer design from `sop_runtime_enablement_plan.md` v3.1 §9.11 (one serializer instance shared across all turns).

### §3.6 Concurrent inbox writers — serialization guarantee

`asyncio.Queue` is safe for multiple producers within the same event loop (Python docs guarantee FIFO ordering of `put_nowait` calls from the same loop). The producers in v1:

1. **The user input handler** (CLI keyboard reader OR `ConversationService.handle_user_message`) calls `ci.inbox_put_user(content)`.
2. **`_run_async`** (CI's own background task) calls `self._inbox.put_nowait(ToolCompletion(tool_name=canonical))`.
3. **(future)** SOP yolo synthetic injector if needed.

All three run in the same event loop — no thread‑safety concerns.

### §3.7 What happens to `run_agentic_loop()`'s public contract?

**Unchanged.** It remains the per‑turn primitive. Callers that don't opt in keep using it. The event loop is layered on top:

```
                ┌─────────────────────────────────┐
                │       ci.run()  (NEW)           │  ← long-lived event loop
                │   while not _shutdown:          │
                │     item = await _inbox.get()   │
                │     await run_agentic_loop(...) │  ← unchanged per-turn primitive
                └─────────────────────────────────┘
```

This minimises blast radius: existing tests, BTA workers, and the rankevolve message_handler all keep working unchanged.

---

## §4 Phased rollout

| Phase | What | Risk | LoC | PRs |
|---|---|---|---|---|
| **0** | Empirical baseline: write all 12 RED tests in §6 (they should all fail today) | LOW | +300 test | 1 |
| **1** | Add `conversational/inbox.py` (the 3 `InboxItem` types: `UserMessage`, `ToolCompletion`, `SyntheticContinue`) + CI fields (`_inbox`, `_shutdown_requested`, `_default_interactive`, `_auto_shutdown_on_sop_complete`, `_turn_counter`) + methods (`enable_inbox`, `inbox_put`, `inbox_put_user`, `request_shutdown`, `_next_turn_number`). NO `run()` yet. Behaviour unchanged. | LOW | +60 src | 1 |
| **2** | Wire `_run_async` to drop a bare `ToolCompletion` wake-up onto the inbox **after** appending the tool result to `_messages` (today's behaviour preserved). Add the 6-line `_check_phase_completion` auto-shutdown bridge (§3.3). v1.2: NO `_run_inner_loop` extraction — `run_agentic_loop` stays unchanged. | LOW | +8 LoC modified | 1 |
| **3** | Add `run()` long-lived event loop (drains inbox; exits when `_shutdown_requested`); add `_content_for_item` dispatcher. | LOW | +50 src | 1 (could merge with Phase 1+2 if confidence is high) |
| **4** | Migrate SOP CLI (`resources/tools/sop/cli.py`) to use `ci.enable_inbox(interactive, auto_shutdown_on_sop_complete=True)` + `ci.inbox_put_user(request)` + `await ci.run()`. Remove the two explicit `run_agentic_loop` calls at lines 170,187. | MED | +20 / ‑20 | 1 |

**v1.1 scope reduction:** v1.0 had 7 phases including OpenStartup server migration (Phase 4–6). v1.1 drops those to v2 — the server is already correct (back-compat branch covers it). v1 ships as **4 phases, all in AgentFoundation, zero cross-repo coordination**.

**v2 (deferred, separate plan iteration):** OpenStartup `ConversationService` switch to persistent CI + `await ci.run()`, gated behind `OPENTEAM_CI_INBOX=1`. Followed by 1-week soak, then default-ON flip, then back-compat branch removal. To be planned when the inbox has demonstrated stability under SOP CLI yolo end-to-end.

---

## §5 Compatibility & invariants

| Invariant | How preserved |
|---|---|
| Existing tests calling `run_agentic_loop(content)` directly still work | `run_agentic_loop` is unchanged. The event loop is additive. |
| `_run_async` callers that don't enable the inbox see no behaviour change | `if self._inbox is not None` guard. |
| BTA worker subagents (which construct CIs and call `run_agentic_loop`) need no changes | They never call `enable_inbox`; they take the legacy path. |
| Existing per‑turn streaming via `interactive.stream_token_batches()` keeps working | `interactive` parameter is threaded through `run_agentic_loop` unchanged. |
| Tool completion message ordering | FIFO by `asyncio.Queue` semantics; multi‑async‑tool case: each `_run_async` puts its own `ToolCompletion` in completion order; agentic loop drains them one at a time (it sees both results in `_messages` on the first wake). |

---

## §6 Acceptance criteria (testable)

### §6.1 Plumbing

- **AC‑I1** `enable_inbox(interactive)` creates `self._inbox`; second call raises `RuntimeError`.
- **AC‑I2** `inbox_put_user("hello")` from outside the running loop is non‑blocking and the item arrives in FIFO order with other puts.
- **AC‑I3** `request_shutdown()` causes `run()` to return after the current `run_agentic_loop` invocation completes — NOT mid-LLM-call.

### §6.2 Event loop semantics

- **AC‑I4** `run()` with one `UserMessage("/help")` in the inbox processes it; when followed by `request_shutdown()`, `run()` returns the last `AgenticResult`.
- **AC‑I5** `run()` with three queued `UserMessage`s processes them in order (assert via mock LLM that sees the content of each turn).
- **AC‑I6** `run()` with an empty inbox suspends (verify via `asyncio.wait_for(run(), timeout=0.05)` raising `TimeoutError` and then a put completing it).

### §6.3 Async‑tool completion

- **AC‑I7** With inbox enabled, dispatching an async tool causes the per‑turn `run_agentic_loop` to exit, the inbox to remain empty until `_run_async` finishes, then a `ToolCompletion` to arrive and trigger a fresh agentic loop. Verify via a fake async executor that takes 100 ms and a mock LLM that asserts the tool result is in its prompt on turn 2.
- **AC‑I8** With inbox NOT enabled (back‑compat), `_run_async` does NOT call `inbox.put_nowait` (verify via `unittest.mock.patch.object` on `asyncio.Queue.put_nowait`). Existing behaviour preserved.
- **AC‑I9** Two concurrent async tools complete; both `ToolCompletion` items land on the inbox in completion order; the first dequeue triggers a turn whose `_messages` contains BOTH tool results.

### §6.4 SOP integration

- **AC‑I10** SOP CLI with `--yolo`: `await ci.run()` returns when `tracker.status == "completed"`; total elapsed = sum of tool runtimes (no idle waiting between phases).
- **AC‑I11** SOP CLI yolo run produces the expected `<repo_root>/_runtime/sop/role_creation__<TS>__<uuid8>/turns/turn_001…turn_N/` directory tree (the `_runtime/sop/` layout from the user's earlier session).
- **AC‑I12** Inject `inbox_put(SyntheticContinue(reason="sop_phase_advance"))` after a no‑tool‑call LLM response — the next turn fires with `_SYNTHETIC_CONTINUE` content; mock LLM asserts it received the right prompt.
- **AC‑I13** (v1.5 — closes Q5): Re-entrancy guard. Calling `ci.run()` while a prior `ci.run()` is still in-flight raises `RuntimeError("run() already in progress")`. The `_running` flag is set to `True` at the top of `run()` (before the `try`), reset to `False` in the `finally` block, and checked at the very beginning. Test: launch one `run()` as a background task; immediately attempt a second `run()`; assert `RuntimeError`; cancel the background task; verify a third `run()` succeeds (proves the guard is properly reset on exit, including on cancellation/exception). Also covers the inverse: calling `run()` without prior `enable_inbox()` raises `RuntimeError("Inbox not enabled")` from the same guard block.

---

## §7 Risk register

| ID | Risk | Sev | Mitigation |
|---|---|---|---|
| R1 | Multiple `enable_inbox` calls in tests (or production) silently re‑initialise and lose queued items | MED | Second call raises `RuntimeError` (AC‑I1); tests must reset by constructing a new CI |
| R2 | Unbounded inbox memory growth if a runaway producer (e.g. tool dispatch loop) puts faster than the LLM drains | LOW | v1 ships unbounded for simplicity; v2 can bound via `maxsize=` and have `put_nowait` raise `QueueFull` (already logged); add a separate AC if needed |
| R3 | `_run_async` puts `ToolCompletion` BEFORE `add_message` if exception ordering goes wrong; the next turn would see an empty result | LOW | The `if self._inbox is not None` block is positioned AFTER `add_message`/`update_prior_context`/`_check_phase_completion` (§2.3). Add an assertion test: AC‑I7 verifies the message is in `_messages` before the inbox item is dequeued. |
| R4 | `request_shutdown()` races with a pending async tool dispatch — tool completes after `run()` already returned | MED | Document as known limitation in v1; the background task still applies `context_updates` and `add_message` (its existing behaviour) — the result is preserved in CI state but no follow‑up turn fires. v2 can add an explicit "drain pending async tasks" gate inside `run()` before exit. The companion `sop_model_a` pause/resume design covers this: resumed CIs naturally re-render with the persisted `_messages` including any post-shutdown tool results. |
| R5 | `ConversationService` switching to persistent CI changes lifecycle assumptions in downstream code (e.g. `_compute_session_context` may assume fresh CI) | MED | Gate behind `OPENTEAM_CI_INBOX=1` env flag for one release; collect logs; then flip. Document in OpenStartup release notes. |
| R6 | The CLI's existing `agentic_result = await ci.run_agentic_loop(...)` at `cli.py:170,187` becomes `result = await ci.run()` — the caller can no longer easily inspect per‑turn results | LOW | The CLI logs each turn via the existing per‑turn artefact writer (`turn_NNN/{input.md,prompt.md,response.md}`); the aggregated `result` is sufficient for the CLI's needs. Tests verify the artefact files exist (AC‑I11). |
| R7 | `asyncio.Queue` is not pickleable, breaking any code that snapshots the CI for resume | LOW | v1.3 correction: the `_NON_SERIALIZABLE_PRIOR_CONTEXT_KEYS` constant referenced in v1.0/v1.2 was REMOVED during the SOPState refactor (verified: zero matches in source). The pause/resume design in `sop_model_a_with_commands_and_resumability_plan.md` v2.6 now uses `SOPState`'s own `to_dict()` / `from_dict()` for serialization (per its §3.7). For the inbox: simply declare `_inbox` and `_default_interactive` with `attrib(init=False, repr=False)` so they're transient and never serialized; ensure the SOP plan's `_serialize_pause_state` explicitly resets them via `dataclasses.fields` filtering. Add a regression test (AC‑FS‑INBOX1 in the SOP plan). |
| R8 | `_active_async_task` is a single-slot attribute (CI:1049). If a single turn dispatches two async tools, the second `create_task(...)` overwrites the first — losing the handle, preventing cancellation on shutdown. | LOW | **v1.3 documents as known limitation.** Benign in v1: the CLI forces all tools synchronous (no concurrent dispatch path). Future server-side work that allows N concurrent async tools per turn must promote `_active_async_task` to a `set[asyncio.Task]` and ensure cancellation iterates the set. Out of scope for v1; called out here so the v2 plan knows where to look. |
| R9 | `_shutdown_requested` is NOT auto-reset when `run()` returns (v1.4 contract change). A caller that calls `request_shutdown()` then wants to call `run()` again must reset the flag explicitly. | LOW | **v1.4 documents as known contract.** Rationale: auto-reset would mask whether `run()` exited from caller-requested shutdown vs. natural completion (e.g. `auto_shutdown_on_sop_complete`). The test fixture AC-I3 and AC-I4 verify both pathways. CLI's usage pattern is "enable once, run once, exit" — does not encounter this. Server v2 must construct fresh CI per session or reset explicitly before reusing. |

---

## §8 Open questions (must be answered before Phase 4)

1. **Bounded vs unbounded inbox in production?** v1 unbounded; OpenStartup may want a soft cap (e.g. 100) with an alarm on overflow.
2. **Should `UserMessage.content` ever be empty?** v1 says no (filter‑drop in `_content_for_item`); decide if there's a legitimate "wake up but don't add content" use case.
3. **Per‑item timeout?** Should `run()` enforce a max per‑turn duration so a stuck LLM call doesn't block the inbox forever? v1: no — rely on `max_iterations` and the existing `run_agentic_loop` exit conditions.
4. **Multi‑SOP interaction:** when `multi_sop_focus_and_tool_concurrency_plan.md` v1.1 lands, do we need a per‑SOP inbox or one shared inbox routed by `sop_instance_id`? Recommendation: one shared inbox with `InboxItem` carrying an optional `sop_instance_id` field; routing happens in `_content_for_item`. Defer to that plan's v2.
5. **Re‑entrancy:** can `ci.run()` be safely called twice (e.g. inside a test setup that's already running it)? v1: no — `run()` checks a `self._running: bool` flag and raises if already true. **Resolved in v1.5: AC-I13 added to test this explicitly.**

---

## §9 File inventory

### §9.1 New files

| Path | Purpose |
|---|---|
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/inbox.py` | The 3 `InboxItem` value-types (`UserMessage`, `ToolCompletion`, `SyntheticContinue`) **only**. NEW module-level constant: `_SYNTHETIC_CONTINUE`. **Does NOT redefine `_CONTINUE_AFTER_TOOLS`** (v1.4 fix from v1.3 audit): that constant stays at CI:81 where it is already used at 3 sites inside `run_agentic_loop`; `inbox.py` IMPORTS it via `from .conversational_inferencer import _CONTINUE_AFTER_TOOLS` if needed (or duplicates the literal — both forms acceptable; verifier should confirm no circular-import). |
| `AgentFoundation/test/agent_foundation/common/inferencers/test_ci_inbox.py` | 12 ACs as RED tests |

### §9.2 Modified files

| Path | Change |
|---|---|
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py` | **+6 new fields** (v1.5 corrected count): `_inbox`, `_shutdown_requested`, `_default_interactive`, `_auto_shutdown_on_sop_complete`, `_turn_counter`, **`_running`** (re-entrancy guard for `run()`, AC-I12); **+7 new methods** (v1.5 corrected count): `enable_inbox`, `inbox_put`, `inbox_put_user`, `request_shutdown`, `run`, `_content_for_item`, `_next_turn_number`; **`run_agentic_loop` UNCHANGED** (v1.2 reverted v1.1's extraction); +2 lines at end of `_run_async` (CI:1035‑1049; v1.3 fix) for the inbox `put_nowait(ToolCompletion)` wake-up; +6 lines in `_check_phase_completion` for the auto-shutdown bridge (§3.3). v1.3: all 5 new fields declared with classic `attrib(default=...)` to match the existing `@attrs(slots=False)` decorator at CI:108 (not `attrs.define(slots=True)` — that would break the inheritance chain). |
| `AgentFoundation/src/agent_foundation/resources/tools/sop/cli.py` | Replace the two explicit `run_agentic_loop` calls at lines 170,187 with: `ci.enable_inbox(interactive, auto_shutdown_on_sop_complete=True)`; `ci.inbox_put_user(request)`; `await ci.run()`. |
| `OpenStartup/src/openteam/server/services/conversation_service.py` | **No change in v1.1** — server's per-turn `run_agentic_loop` path is correct (async tool results handled by `_run_async` back-compat branch). v2 plan iteration will introduce `OPENTEAM_CI_INBOX=1` env-gated persistent-CI + `await ci.run()` path. |
| `workflows_and_sop/sop_model_a_with_commands_and_resumability_plan.md` (this is companion plan v2.6) | Append a §0.1‑inbox row to its audit history noting the dependency. v1.3 fix: `_NON_SERIALIZABLE_PRIOR_CONTEXT_KEYS` was deleted; instead update `_serialize_pause_state` in the SOP plan §3.7 to explicitly exclude `_inbox` and `_default_interactive` from its serialized state (per `attrib(init=False, repr=False)` declaration in the inbox.py module). |

### §9.3 Deleted files

None. This is purely additive.

**Net file count: +2 new, ~3 modified, 0 deleted.**

---

## §10 Why this design (vs alternatives considered)

| Alternative | Why rejected |
|---|---|
| **Recursive `_run_async → run_agentic_loop`** | Rejected by user at 15:43. Re‑entrancy hazard; can stack; surprising for callers; hard to reason about exception propagation. |
| **Server‑layer callback ("server listens for task_completed event")** | Splits async‑completion logic across two repos. Every new caller (rankevolve, future server backends) reinvents the listener. Violates locality. |
| **Polling the existing `_messages` for new entries** | Inefficient (wasted CPU); doesn't compose with multiple concurrent tools; doesn't naturally support synthetic / user‑message sources. |
| **Per‑async‑tool dedicated Future that the next `run_agentic_loop` awaits** | Doesn't solve the "who calls the next `run_agentic_loop`" problem; just moves it. |
| **Make `run_agentic_loop` itself long‑lived** | Breaks every existing caller. The additive `run()` wrapper preserves both worlds. |

**This design wins because:** (a) one input path for everything — async results, user messages, synthetic continues; (b) zero polling (asyncio.Queue.get suspends until item arrives); (c) zero special‑casing (the event loop just calls the existing per‑turn primitive); (d) additive — old callers keep working unchanged; (e) composes cleanly with the InteractionSerializer + SOP Model A designs that already shipped earlier this week.

---

## §11 Comparison with companion plans

| Plan | Relationship |
|---|---|
| `workflows_and_sop/sop_model_a_with_commands_and_resumability_plan.md` v2.6 | This inbox plan is what unblocks SOP yolo end‑to‑end with async tools. The SOP plan assumes synchronous tools (which the CLI forces); the inbox lets the SERVER‑side SOP run with the real async pipeline. The SOP plan needs no rewrite, only a §0.1 audit‑row mention plus a v1.3-corrected hook: update `_serialize_pause_state` (§3.7) to explicitly exclude `_inbox` and `_default_interactive` from the serialized state. |
| `workflows_and_sop/multi_sop_focus_and_tool_concurrency_plan.md` v1.1 | Cleanly composes — one shared inbox per CI with `InboxItem.sop_instance_id` routing (deferred to that plan's next iteration; see Open Question 4). |
| `sop_runtime_enablement_plan.md` v3.1 §9.11 (InteractionSerializer) | Orthogonal — the inbox carries inputs into the CI; the InteractionSerializer carries interaction requests OUT to the UI. Both use `asyncio.Queue` semantics but for opposite directions. They are the two halves of a full event‑driven CI. |
| `conversational_inferencer_template_manager_migration_plan.md` v3.3 | Orthogonal — template rendering happens inside `run_agentic_loop`, which the inbox calls unchanged. |

---

## §12 Final recommendation

**If forced to pick one plan, I would pick this plan (v1.4).** It surfaces a real architectural fault (CI is request‑response only when the rest of the system became event‑driven) and fixes it with a minimal, additive, FIFO‑safe, zero‑polling change that composes with every adjacent plan already in flight. The SOP yolo end‑to‑end story has no robust answer without it. (v1.4: keeps the v1.0/v1.2 architecture, applies 4 rounds of empirical-fidelity corrections in-place.)

---

*End of plan v1.4. Empirical baselines verified 2026‑05‑28 15:52 (round 1, against live CI code) and 17:48 (round 3, fidelity pass). Round-1 integration (v1.1, 15:59) adopted 4 changes from Claude's parallel-authored plan; round-2 re-read (v1.2, 17:17) reverted 3 of those 4; round-3 (v1.3, 17:48) applied 10 of 12 reviewer-surfaced empirical-fidelity corrections plus Risk R8; round-4 (v1.4, 18:22) applied 7 of 7 reviewer-surfaced internal-consistency corrections plus Risk R9 — §3.1 now declares all 6 CI fields, §3.2 signature matches, §2.4 cli.py text honest, §9.1 inbox.py constant scope corrected. Zero architectural change since v1.2; the design is stable. v1.4 ships ready for implementation.*
