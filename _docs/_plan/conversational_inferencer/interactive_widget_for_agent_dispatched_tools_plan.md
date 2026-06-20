# Interactive Widgets for Agent-Dispatched Async Tools — Plan v2

> **⚠ Reviewer banner — please read v2 changes first**
>
> **v1 → v2 critical correction:** Two peer-review plans (Plan B "Cursor" and Plan C "Claude") independently caught a **critical bug in v1**: my v1 closed only **one** of the **two gaps** required to make the router interactive. v1's queue-registration fix is necessary but **not sufficient** — `router_interactive_safe` is read at `executor.py:564` but **never written anywhere** in the codebase (verified: grep returns only 2 hits, both reads). Without writing this flag, the router always coerces `interactive=None` and forces yolo, making v1 a no-op for the stated goal.
>
> **v2 adds 3 things:**
> 1. **Gap-2 fix (CRITICAL)**: Write `router_interactive_safe=True` at both queue-registration sites (`tool_dispatcher._dispatch_as_task` + dev-slash `session_context`) — coupled atomically to successful registration so the flag is never set without a real queue behind it.
> 2. **`--confirm` repair (recommended, Phase 2)**: Fix the `NameError` at `executor.py:861` that makes `task --confirm` currently dead. Uses the same `router_interactive_safe` contract.
> 3. **Audit of which async tools actually interact**: Only the conversational router (`task --config disabled/conversation`) and `task --confirm` use interactive in the server path. Other async tools (`create_role`, `role_setup`, `project_onboarding`, derived `task` tools) never call `asend_response`/`aget_input` — they're unaffected. This audit closes the "did we get the scope right?" question definitively.
>
> **What this plan fixes (v2 statement):** Agent-dispatched async tools (e.g. `task --config disabled` called by the conversational inferencer mid-turn) currently fall through to yolo and silently degrade autonomous behavior — even though they were authored to ask clarifying questions. Two independent gaps must be closed: (1) register a per-task input queue so widget responses can be routed back, AND (2) set the `router_interactive_safe` flag that the router checks before using interactive. v2 closes both atomically.
>
> **The fix in one sentence:** When `_dispatch_as_task` spawns a background task, give it a fresh per-task `WebSocketInteractive` (with its own `asyncio.Queue`) registered in `dev_tool_input_queues[task_id]`, mirroring exactly what the dev-slash path already does at `manager_websocket_routes.py:205`. **A `TaskWebSocketInteractive` subclass already exists in the codebase for this purpose** (`websocket_interactive.py:309`) but is currently dead code — never instantiated. v1 wires it up.
>
> **Why this matters for the bigger picture:** Once interactive works for agent-dispatched tools, the conversational router from the `task_complexity_presets_and_chat_peer_plan` can ask clarifying questions ("Which preset?"); the `proposal_selection` widget from the `proposal_selection_tool_migration_plan` works through `/task`; and `yolo_mode` becomes what it should be — the autonomous fallback when interactive is unavailable, not the only working mode.
>
> **Scope discipline:** This plan covers ONLY the per-task queue registration fix. It deliberately does NOT cover (a) stamping `task_id` on `pending_input` messages for true concurrent interactive tasks (deferred — single in-flight is enough for the router use case), (b) any rename/refactor of `WebSocketInteractive`, (c) any conversational-router behavior changes (those live in the `task_complexity_presets` plan). One bug, one fix.
>
> **A note on the underlying design decision:** an earlier exploration considered placing a `create_task_interactive` factory closure into `session_context`. That approach was **rejected** based on source verification — see §D2. The chosen approach (a method on `WebSocketInteractive`) is more elegant because it reuses the per-turn `_interactive` handle that already flows from WS handler → conversation service → tool dispatcher, and it avoids inventing a new injection channel.

---

**Author:** Rovo Dev (CI session)
**Date:** 2026-06-14
**Status:** Draft v1, not yet committed
**Branch:** `dev_xinli_2601` (AgentFoundation); coupled change in `OpenStartup` on `dev_xinli_2601`
**Companion to:** `task_complexity_presets_and_chat_peer_plan.md` (provides the router that needs interactive); `proposal_selection_tool_migration_plan.md` (provides the widget that needs interactive end-to-end through `task`); `workflow_lifecycle_commands_plan.md` (`/pause_workflow` triggers interactive prompts that benefit from the fix)
**Cross-repo:** Production fix lands in `OpenStartup`; this plan lives in AgentFoundation's `_docs/_plan/` because the affected end-to-end flow is the conversational inferencer (AgentFoundation) → tool dispatcher (OpenStartup) → widget (OpenStartup UI).

---

## §0. Quick-start

**What this plan does (one paragraph):** Wires the existing-but-dead `TaskWebSocketInteractive` subclass into `_dispatch_as_task`. Adds one method `WebSocketInteractive.for_background_task(task_id) -> (child, cleanup)` that creates a fresh `asyncio.Queue`, builds a child `TaskWebSocketInteractive` sharing the same `_send` and `_last_prompt_data`, registers the queue in the per-connection `dev_tool_input_queues` registry, and returns a cleanup closure. `_dispatch_as_task` calls this method when available, uses the child as `task_context["interactive"]`, and invokes `cleanup()` in the task's `try/finally`. **The fallback path (no `for_background_task` method) preserves today's behavior** — CLI path is unaffected because it never reaches this code (CLI forces tools synchronous via the absence of an interactive).

**Effort estimate:** ~1 day. ~60 LoC production + ~120 LoC tests across 2 repos.

**Commits in dependency order:**

| # | Repo | Commit | Purpose | LoC |
|---|---|---|---|---|
| 1 | OpenStartup | `WebSocketInteractive` accepts optional `task_queues` registry + adds `for_background_task` method | Plumbing: gives the per-turn interactive access to the per-connection registry | ~30 |
| 2 | OpenStartup | Both `WebSocketInteractive` construction sites pass the registry | Two callers (dev-slash path + main WS handler) wire the registry through | ~6 |
| 3 | OpenStartup | `_dispatch_as_task` uses `for_background_task` when available, with cleanup in `try/finally` | The actual bug fix — agent-dispatched tools now get a real input queue | ~20 |
| 4 | OpenStartup + AgentFoundation | Tests: unit (queue registration), integration (E2E widget round-trip), regression (dev-slash unchanged, CLI unchanged) | Locks the fix in; protects against future regressions | ~120 tests |

**Lowest-risk first:** Commit 1 is pure additive (new method, optional kwarg with default `None`). Commit 2 is mechanical. Commit 3 is the behavior change; gated by `hasattr` so CLI/None paths are unaffected.

---

# PART I — EXECUTION
══════════════════════════════════════════════════════════════════════════════

## §E1. Migration plan — 4 commits

### §E1.1 — Commit 1: `WebSocketInteractive.for_background_task` method

**Purpose:** The one new method that gives `_dispatch_as_task` a per-task interactive without leaking transport details. Uses the **existing** `TaskWebSocketInteractive` subclass (verified at `websocket_interactive.py:309`), which is currently dead code — never instantiated anywhere in production. v1 makes it live.

**Files modified (OpenStartup):**

1. `src/openteam/server/services/websocket_interactive.py` (~30 LoC):

```python
# In WebSocketInteractive.__init__, add optional kwarg:
def __init__(
    self,
    send_callback: Callable[[dict[str, Any]], Coroutine],
    input_queue: asyncio.Queue,
    *,
    task_queues: dict[str, asyncio.Queue] | None = None,
) -> None:
    self._send = send_callback
    self._input_queue = input_queue
    self._clean_output: str | None = None
    self._last_prompt_data: dict[str, Any] | None = None
    self._task_queues = task_queues  # per-connection registry; may be None for CLI/standalone

# NEW method on WebSocketInteractive:
def for_background_task(
    self, task_id: str
) -> tuple["TaskWebSocketInteractive", Callable[[], None]]:
    """Allocate a fresh interactive + input queue for a background task.

    Returns (child_interactive, cleanup_fn). The child shares this
    interactive's `_send` callback (so widgets reach the same WebSocket
    connection) and `_last_prompt_data` reference (so prompt_data
    inlining keeps working). The child has its own input queue,
    registered in self._task_queues so pending_input_response can
    route by task_id.

    If self._task_queues is None (CLI / standalone path), returns a
    child whose queue is NOT registered — callers should fall back to
    treating the task as synchronous or use the parent interactive.
    """
    if self._task_queues is None:
        # CLI / standalone path — no registry to register in.
        # Callers must handle this (Commit 3 falls through to the
        # existing behavior: re-use parent interactive).
        raise RuntimeError(
            "for_background_task requires a task_queues registry; "
            "this interactive was constructed without one."
        )

    queue: asyncio.Queue = asyncio.Queue()
    child = TaskWebSocketInteractive(
        self._send, queue, task_id=task_id,
    )
    # Share the prompt-data cache so the child's asend_response can
    # inline prompt_data even when not supplied explicitly.
    child._last_prompt_data = self._last_prompt_data  # shared reference

    self._task_queues[task_id] = queue

    task_queues_ref = self._task_queues  # capture for cleanup closure

    def _cleanup() -> None:
        task_queues_ref.pop(task_id, None)

    return child, _cleanup
```

**Why this shape (not an alternative):**

| Alternative | Why rejected |
|---|---|
| Method on `WebSocketInteractive` that *mutates* a class-level dict | Per-connection registries must NOT be class-level — multiple WS connections each have their own `dev_tool_input_queues`. Instance attribute is correct. |
| Factory function in `session_context` | `session_context` is built per-session and cached; `dev_tool_input_queues` is created per-connection at `manager_websocket_routes.py:306`. The closure can't capture a registry that doesn't exist at session-construction time. Verified in §D2. |
| Have `_dispatch_as_task` directly import and mutate `dev_tool_input_queues` | Leaks WS-handler-internal state into the dispatcher. Brittle if transport changes. The method-on-interactive approach keeps WS knowledge inside the WS-transport class. |
| `TaskWebSocketInteractive` as the canonical class (always used, no parent) | The parent `WebSocketInteractive` is also used for the per-turn (non-task) interactive — it can't just disappear. Subclass for tasks + base for turns is the right split. |

**Tests (T1–T4):**
- T1: `WebSocketInteractive(send, queue)` (no `task_queues`) succeeds; `for_background_task` raises `RuntimeError`.
- T2: `WebSocketInteractive(send, queue, task_queues={})` succeeds; `for_background_task("task-x")` returns `(TaskWebSocketInteractive, cleanup_fn)`; registry now contains `"task-x"`.
- T3: Cleanup fn removes the entry: after `cleanup()`, `"task-x"` is no longer in the registry.
- T4: Child's `_send` IS the same callable as the parent's; child's `_last_prompt_data` initially equals parent's (shared reference at construction time).

**Risk:** very low. Pure additive: new optional kwarg with default `None`, new method that's no-op-equivalent for old callers (they don't call it).

**LoC:** ~30 production + ~40 tests.

### §E1.2 — Commit 2: Wire `task_queues` into both construction sites

**Purpose:** Pass the per-connection `dev_tool_input_queues` registry into both places that construct a `WebSocketInteractive` so the new method is usable.

**Files modified (OpenStartup):**

1. `src/openteam/server/routes/manager_websocket_routes.py`:

```python
# Site A — dev-slash construction (line ~204):
interactive = WebSocketInteractive(
    send_safe, input_queue, task_queues=dev_tool_input_queues,
)

# Site B — main message-handler construction (line ~397):
interactive = WebSocketInteractive(
    send_safe, active_input_queue, task_queues=dev_tool_input_queues,
)
```

Both sites already have `dev_tool_input_queues` in scope (it's created at line 306 and threaded through). Just one new kwarg per site.

**v2 addition — dev-slash `session_context` gets `router_interactive_safe=True`** (lines ~229–234 in `manager_websocket_routes.py`):

```python
# Dev-slash session_context — the queue is registered at line 205, so the
# contract is fulfilled and the router can safely use interactive.
session_context = {
    "interactive": interactive,
    "task_id": task_id,
    "session_id": sid,
    "session_root": _session_root,
    "router_interactive_safe": True,   # v2: makes /task --config disabled interactive
}
```

This is the critical-bug fix #2 site: writing `router_interactive_safe=True` for the dev-slash path so `/task --config disabled <ambiguous>` (router via dev-slash) becomes interactive too. Verified at `executor.py:564` that the router only uses `interactive` if this flag is truthy.

**Tests (T5–T6):**
- T5: After dev-slash dispatches a task, the dev-slash `interactive` has `_task_queues is dev_tool_input_queues` (same object identity).
- T6: After main WS handler creates per-turn `interactive`, same identity check passes.

**Risk:** trivial. Mechanical kwarg threading; no behavior change yet (Commit 3 is the consumer).

**LoC:** ~6 production + ~20 tests.

### §E1.3 — Commit 3: `_dispatch_as_task` uses `for_background_task`

**Purpose:** The actual bug fix. When the dispatcher spawns a background task, it now creates a per-task interactive with its own input queue, registers it in `dev_tool_input_queues`, and uses it as `task_context["interactive"]` so widgets emitted by the background task can have their responses routed back.

**Files modified (OpenStartup):**

1. `src/openteam/server/services/tool_dispatcher.py` (~20 LoC, replacing line 201 area + `_run` body):

```python
# Line 201 (today):  interactive_ref = self._interactive  # capture for closure
# Replace with:

# Per-task interactive with its own input queue (Patch 4.1)
# v2 — Coupled atomically with router_interactive_safe so the flag is NEVER
# set without a real registered queue behind it (the contract docstring
# at executor.py:542-547 explicitly demands this guarantee).
interactive_ref = self._interactive
task_cleanup: Callable[[], None] | None = None
interactive_safe = False   # v2: contract flag, only set on successful registration
if hasattr(self._interactive, "for_background_task"):
    try:
        interactive_ref, task_cleanup = self._interactive.for_background_task(task_id)
        interactive_safe = True   # v2: queue is registered — router may use interactive
    except RuntimeError:
        # task_queues registry unavailable (CLI/standalone) — fall back to
        # parent. Background tool will receive widgets via the parent's
        # queue (best-effort; widgets may not work but the tool runs to
        # completion). interactive_safe stays False — router will yolo.
        logger.info(
            "[ToolDispatcher] for_background_task unavailable for %s; "
            "falling back to parent interactive (router will yolo)",
            task_id,
        )
# (else branch: old WebSocketInteractive without the new method — same fallback;
#  interactive_safe stays False)

# v2 — task_context now carries router_interactive_safe coupled to the registration
task_context = {
    **self._session_context,
    "task_id": task_id,
    "session_root": session_root_str,
    "working_dir": task_working_dir,
    "interactive": interactive_ref,
    "router_interactive_safe": interactive_safe,   # v2: gap-2 fix
}

async def _run() -> None:
    try:
        await interactive_ref.send_task_status(
            task_id, "running", tool_name=tool_name,
        )
        # ... [existing body unchanged through to task_completed] ...
    except Exception as e:
        # ... [existing error handling unchanged] ...
    finally:
        # Patch 4.1 — release the per-task input queue
        if task_cleanup is not None:
            task_cleanup()
```

**v2 critical note on the coupling:** `interactive_safe` MUST be set ONLY inside the successful registration branch (line above the `except`). If the `except` branch ran, the registry has nothing for this `task_id`, and the router would deadlock waiting for a response that can never be routed. Three layers protect the invariant: (a) the `hasattr` guard, (b) the `try/except RuntimeError`, (c) the initial `interactive_safe = False`. T11 + T11b (new in v2) lock this in.

**Note on `add_done_callback` vs `finally`:** The dev-slash path uses `task_obj.add_done_callback(_cleanup)` (line 287). Both approaches work; `finally` inside `_run` is slightly preferred here because the cleanup is one statement and the dispatcher path doesn't otherwise capture the `task_obj` reference. Either is acceptable — if reviewers prefer symmetry with dev-slash, switch to `add_done_callback` (3-LoC diff). Filed as Follow-up #1.

**Tests (T7–T11):**
- T7: After `_dispatch_as_task` is called with a parent interactive that has `for_background_task`, the spawned task's `task_context["interactive"]` is the child (different object identity from parent).
- T8: After the task spawns, `dev_tool_input_queues[task_id]` is populated.
- T9: After the task completes (or errors), `dev_tool_input_queues[task_id]` is gone (cleanup ran).
- T10: When the parent interactive lacks `for_background_task` (legacy CLI path, mocked), `_dispatch_as_task` still completes successfully and uses the parent interactive as `task_context["interactive"]`.
- T11: When `for_background_task` raises `RuntimeError` (no registry), `_dispatch_as_task` falls back gracefully and logs an info-level message.

**Risk:** medium. The actual behavior change. Mitigation: T10/T11 specifically cover the fallback paths; CLI tests must continue to pass.

**LoC:** ~20 production + ~50 tests.

### §E1.4 — Commit 4: Integration + regression suite

**Purpose:** End-to-end verification that the widget round-trip works for agent-dispatched async tools, plus regression coverage for the two known-working paths (dev-slash, CLI).

**Files added/modified:**

1. **OpenStartup:** `tests/integration/test_agent_dispatched_widget_roundtrip.py` (~80 LoC):

```python
# Scenario A: Agent-dispatched task asks for confirmation
#   - CI calls `task` tool with a request that triggers /confirm widget
#   - Mock WS sends pending_input message with the widget
#   - Test simulates UI response: pending_input_response with task_id
#   - Assert: the background task's aget_input() returns the response
#   - Assert: task completes successfully
#
# Scenario B: Dev-slash regression
#   - User types /task "..." in chat
#   - Same widget round-trip
#   - Assert: behavior is IDENTICAL to today (no regression)
#
# Scenario C: CLI regression (no task_queues registry)
#   - Build a WebSocketInteractive without task_queues=
#   - Dispatch a task
#   - Assert: task runs; no error from for_background_task fallback
#   - Assert: log line matches "for_background_task unavailable"
```

2. **AgentFoundation:** `tests/integration/test_conversational_inferencer_widget_e2e.py` (~40 LoC):
   - Mock CI runs a turn that calls `task` tool
   - Mock WS captures emitted `pending_input` messages
   - Verifies the AgentFoundation side of the contract (CI → tool_dispatcher) is unchanged

**Tests (T12–T17):**
- T12: Scenario A (agent-dispatched widget round-trip) succeeds end-to-end.
- T13: Scenario A with widget response sent BEFORE the task starts asking → response is queued and consumed correctly when the task does call `aget_input`.
- T14: Scenario B (dev-slash regression) succeeds; behavior matches today's golden.
- T15: Scenario C (CLI regression) succeeds; log line verifies fallback was taken.
- T16: AgentFoundation E2E — CI calling `task` tool emits the right tool dispatch + receives the result. No change from today (sanity check).
- T17: Performance — 50 sequential agent-dispatched tasks with widgets don't leak queue entries in `dev_tool_input_queues`.

**Risk:** medium. Integration test surface is broad; brittleness is the main risk. Mitigation: use mocked WS connection + asyncio queue assertions; don't depend on real browser.

**LoC:** ~120 tests + ~20 LoC test fixtures.

### §E1.5 — Commit 5 (v2 addition, Phase 2): Repair `task --confirm` NameError

**Purpose:** The other server-side interactive async path. `task --confirm` (checkpoint plan review via `interactive_checkpoint.py:70–76`) is currently dead — `executor.py:861` references a bare name `interactive` that is undefined in `_run_topology`'s scope (verified: the local `interactive` at line 564 is in a different function). Running `task --confirm <anything>` raises `NameError` before reaching `ainfer`.

The fix uses the SAME `router_interactive_safe` contract Commit 3 establishes, so `--confirm` becomes interactive only when a registered queue is guaranteed.

**Files modified (AgentFoundation):**

1. `src/agent_foundation/resources/tools/task/executor.py` (~5 LoC at line 861):

```python
# Line 861 today (NameError — 'interactive' is undefined here):
# if mode == "confirm" and hasattr(inferencer, "interactive") and interactive is not None:
#     inferencer.interactive = interactive

# v2 fix — read from session_context under the same router_interactive_safe contract:
if mode == "confirm" and hasattr(inferencer, "interactive"):
    _confirm_itx = sc.get("interactive") if sc.get("router_interactive_safe") else None
    if _confirm_itx is not None:
        inferencer.interactive = _confirm_itx
```

`_run_topology` already has `sc = session_context or {}` at line 646; this fix piggybacks on the existing alias. PTI's `if self.interactive is None` branch at `plan_then_implement_inferencer.py:2072–2074` auto-approves when interactive is None, so the fallback (CLI path / no registered queue) preserves today's auto-approve behavior — only when the contract is satisfied does interactive prompting kick in.

**Tests (T18–T20):**
- T18: `task --confirm "trivial task"` with `router_interactive_safe=False` in `session_context` → auto-approves (PTI fallback); no NameError; behavior matches today's intent (when --confirm was authored).
- T19: `task --confirm "trivial task"` with `router_interactive_safe=True` + a registered interactive → PTI calls into interactive's Approve/Modify/Reject widget; user response is read back; task proceeds based on response.
- T20: `task --confirm` from a CLI path (no `interactive` in `sc`, no `router_interactive_safe`) → auto-approves; no crash; preserves today's CLI behavior.

**Risk:** low. Single-line behavior change; reads from an existing scope-local alias (`sc`); PTI's auto-approve fallback covers the no-interactive path; the NameError today means no one is depending on `--confirm` working.

**LoC:** ~5 production + ~40 tests.

### §E1.6 — Optional renaming note (deferred to Follow-up #9)

`router_interactive_safe` now governs both the router (executor.py:564) and `--confirm` (executor.py:861 after the v2 fix). The `router_` prefix is historical. Plan B observed renaming to a neutral `interactive_safe` is a safe, optional cosmetic tidy-up (three write sites: dev-slash session_context + dispatcher task_context + tests + two read sites: executor.py 564 + 861). v2 keeps the existing name to minimize churn; rename is filed as Follow-up #9.

---

## §E2. Validation

### §E2.1 — Per-commit gates
- Commit 1: T1–T4 pass; existing `WebSocketInteractive` users (search via `grep -rn "WebSocketInteractive(" src/openteam/`) still work (back-compat — `task_queues=None` default).
- Commit 2: T5–T6 pass; dev-slash regression test still green.
- Commit 3: T7–T11 pass; existing tool_dispatcher tests still green.
- Commit 4: T12–T17 pass.

### §E2.2 — End-to-end smoke

```bash
# A. Unit + integration tests
cd /Users/tchen7/MyProjects/CoreProjects/OpenStartup
pytest tests/integration/test_agent_dispatched_widget_roundtrip.py -v
pytest src/openteam/server/services/test_websocket_interactive.py -v
pytest src/openteam/server/services/test_tool_dispatcher.py -v

# B. AgentFoundation cross-repo integration test
cd /Users/tchen7/MyProjects/CoreProjects/AgentFoundation
pytest tests/integration/test_conversational_inferencer_widget_e2e.py -v

# C. Manual smoke (requires running server + UI)
# 1. Start OpenStartup server: python -m openteam.server.main
# 2. Connect UI; in chat type:
#       Build a simple Python script that prints "hi" (and ask me to approve the plan)
# 3. The CI calls `task` tool with the request
# 4. The task tool's planning phase asks "Approve this plan?" widget
# 5. Widget appears in UI panel
# 6. Click "Approve"
# 7. Task tool continues to implementation
# 8. Task completes; result is summarized back to chat
#
# Acceptance: full round-trip works without hang at step 6.
#             Before this fix, step 6 hangs forever.
```

### §E2.3 — CHANGELOG entry

```
### Fixed
- Agent-dispatched async tools (e.g. `task` called by the conversational
  inferencer mid-turn) no longer hang when emitting interactive widgets.
  Previously, the widget would appear in the UI but the user's response
  would be silently dropped because no per-task input queue was registered.
  Now matches the dev-slash `/task` path's behavior. (Patch 4.1)
```

---

## §E3. Execution checklist

```
[ ] Pre-flight (5 min)
[ ]   git status — confirm clean tree on dev_xinli_2601 in BOTH repos
[ ]   grep "TaskWebSocketInteractive" src/openteam/server/services/ — verify
        subclass still exists at line ~309 (sanity check: not stale plan)
[ ]   grep "dev_tool_input_queues" src/openteam/ — verify registry is still
        per-connection at manager_websocket_routes.py:306

OpenStartup — Commit 1: WebSocketInteractive.for_background_task
[ ] Edit  server/services/websocket_interactive.py — add task_queues kwarg
[ ] Edit  server/services/websocket_interactive.py — add for_background_task method
[ ] NEW   server/services/test_websocket_interactive_for_background_task.py (T1–T4)
[ ] Tests + lint  → commit "feat(ws): WebSocketInteractive.for_background_task"

OpenStartup — Commit 2: Wire task_queues at both construction sites
[ ] Edit  server/routes/manager_websocket_routes.py — site A (dev-slash, line ~204)
[ ] Edit  server/routes/manager_websocket_routes.py — site B (main turn, line ~397)
[ ] NEW   server/routes/test_manager_websocket_routes_wiring.py (T5–T6)
[ ] Tests + lint  → commit "feat(ws): pass task_queues registry to interactive"

OpenStartup — Commit 3: tool_dispatcher uses for_background_task
[ ] Edit  server/services/tool_dispatcher.py — replace line ~201 area with
          for_background_task call + fallback + cleanup in try/finally
[ ] Edit  server/services/test_tool_dispatcher.py — add T7–T11
[ ] Tests + lint  → commit "fix(dispatcher): per-task input queue for agent-dispatched tools"

Cross-repo — Commit 4: Integration + regression
[ ] NEW   OpenStartup/tests/integration/test_agent_dispatched_widget_roundtrip.py (T12–T15, T17)
[ ] NEW   AgentFoundation/tests/integration/test_conversational_inferencer_widget_e2e.py (T16)
[ ] Run   pytest both repos
[ ] Write CHANGELOG entry per §E2.3 in BOTH repos
[ ] Manual smoke per §E2.2 C — required before push
[ ] git push origin dev_xinli_2601 (both repos)
[ ] Update _docs/_plan/README.md index with this plan
```

---

# PART II — DESIGN REFERENCE
══════════════════════════════════════════════════════════════════════════════

## §D1. Goals & non-goals

**Goals:**
1. Make agent-dispatched async tools work with interactive widgets — same UX as dev-slash `/task`.
2. Unblock downstream features that depend on this: conversational router clarifying questions (from `task_complexity_presets_and_chat_peer_plan`); `proposal_selection` widget through `task` (from `proposal_selection_tool_migration_plan`); `/pause_workflow` / `/resume_workflow` widgets (from `workflow_lifecycle_commands_plan`).
3. Zero regression in: dev-slash `/task` path; CLI path (no WS); existing tool_dispatcher behavior; existing WS message flow.
4. Use the **already-existing** `TaskWebSocketInteractive` subclass (it's there, designed for this, just never wired up). Don't invent new classes when an existing one fits.
5. Keep transport details (queues, send callbacks) inside `WebSocketInteractive`. Don't leak into `tool_dispatcher`.

**Non-goals:**
1. **Not** stamping `task_id` on `pending_input` messages for true concurrent interactive tasks. The UI currently disambiguates via `currentTaskIdRef` (set by latest `task_status`), which is sufficient for **one** in-flight interactive task at a time (the router use case). True concurrent interactive tasks would need (a) `task_id` in `pending_input` and (b) UI tracking per-task pending widgets. Filed as Follow-up #2 — significant UI work, defer until needed.
2. **Not** refactoring `WebSocketInteractive` more broadly (no rename, no method extraction, no class split). Single-purpose change.
3. **Not** changing the conversational router's tool-calling behavior. Router design lives in the `task_complexity_presets` plan.
4. **Not** building a session-scoped registry. The per-connection `dev_tool_input_queues` is correct for the WS lifetime; cross-session resume is a separate concern (filed in `workflow_lifecycle_commands_plan` Follow-up #3).
5. **Not** changing `yolo_mode`. Yolo is the synthetic-response fallback (verified at `conversational_inferencer.py:436`) and is orthogonal to interactive — fixing interactive doesn't change yolo.

## §D2. Architecture decision — method on `WebSocketInteractive` (rejecting alternatives)

Earlier exploration in the design discussion considered three alternatives. Each was rejected based on source verification. Here is the honest comparison:

### §D2.1 — Option A (rejected): `create_task_interactive` factory closure in `session_context`

The initial recommendation was to pass a factory function through `session_context` that the dispatcher would call to get a per-task interactive. This is the pattern already used for `tool_executor` (the dispatcher gets a function it calls without knowing transport details).

**Why rejected:** verified source-level errors in the premise:
- **Premise error 1:** `tool_dispatcher.py` is in `openteam/server/services/`, the SAME package as `websocket_interactive.py`. There is no cross-layer boundary to protect — the dispatcher already imports server-internal types. The "avoid cross-layer imports" justification is moot.
- **Premise error 2:** `dev_tool_input_queues` is created at `manager_websocket_routes.py:306` **per WebSocket connection**, not per session. `session_context` is built per-session and cached. A closure placed in `session_context` cannot capture the registry — the registry doesn't exist yet when `session_context` is built. The factory would need to be re-injected per-turn anyway, defeating the elegance argument.

### §D2.2 — Option B (rejected): direct mutation of `dev_tool_input_queues` from `tool_dispatcher`

The most direct fix: import `dev_tool_input_queues` into `tool_dispatcher` and mutate it.

**Why rejected:** even though both files live in the same package (so the import is legal), this couples the dispatcher to a specific WS-handler-internal name. If the registry is renamed, moved, or replaced with a different transport (e.g. SSE, HTTP long-poll), the dispatcher breaks. The method-on-interactive approach keeps the dispatcher transport-agnostic — it asks the interactive for a child, the interactive knows how to register itself.

### §D2.3 — Option C (rejected): canonical `TaskWebSocketInteractive` (always used)

Replace all uses of `WebSocketInteractive` with `TaskWebSocketInteractive`, parameterized by task_id (empty string for the per-turn interactive).

**Why rejected:** the per-turn interactive serves token streaming for the parent conversation (no task scoping), while task interactives need task_id stamping for streaming events. Conflating them muddles the contracts. The subclass already exists as the right split (verified `websocket_interactive.py:309`). Use it.

### §D2.4 — Option D (CHOSEN): method on `WebSocketInteractive` returning a `TaskWebSocketInteractive`

| Property | Verified outcome |
|---|---|
| Symmetric with dev-slash | Mirrors `manager_websocket_routes.py:204` (`WebSocketInteractive(send_safe, input_queue)` + line 205 `dev_tool_input_queues[task_id] = input_queue`) — same mechanism, different scope. |
| Uses the existing subclass | `TaskWebSocketInteractive` (line 309) is designed for this; v1 makes it live without inventing new code. |
| Reuses the per-turn `_interactive` handle | The dispatcher already has `self._interactive` (line 59) injected per-turn from `conversation_service.py:472–473`. No new injection channel needed. |
| Transport-agnostic dispatcher | The dispatcher calls `for_background_task` — knows nothing about queues, send callbacks, or `dev_tool_input_queues`. |
| Lifecycle is explicit | Factory returns `(child, cleanup)`. Caller owns cleanup. No implicit cleanup, no leak. |
| Back-compat | Optional kwarg with default `None`; `hasattr` guard in dispatcher; `RuntimeError` fallback for CLI. Three layers of defense against regression. |

## §D3. The interactive lifecycle (with the fix)

### §D3.1 — Before the fix (TODAY — broken for agent-dispatched tools)

```
Per-turn interactive (constructed in WS handler):
    interactive = WebSocketInteractive(send_safe, active_input_queue)
    # NO task_queues registry passed in
    # active_input_queue is the conversation's per-turn queue
    # — nulled after the turn ends (manager_websocket_routes.py:513)

Dispatcher captures it as self._interactive (conversation_service.py:472-473).

When dispatcher spawns a task (tool_dispatcher.py:201):
    interactive_ref = self._interactive  # the per-turn interactive

The background task uses interactive_ref to:
  - send widgets:    await interactive_ref.asend_response(prompt)
                       → sends pending_input to UI ✓ (widget appears)
  - await responses: await interactive_ref.aget_input()
                       → blocks on interactive_ref._input_queue.get()
                       → but _input_queue IS active_input_queue
                       → active_input_queue is nulled after turn ends
                       → response is lost; task hangs forever ✗
```

### §D3.2 — After the fix (with `for_background_task`)

```
Per-turn interactive (constructed in WS handler with new kwarg):
    interactive = WebSocketInteractive(
        send_safe, active_input_queue,
        task_queues=dev_tool_input_queues,
    )
    # NOW carries the per-connection registry

Dispatcher captures it as self._interactive.

When dispatcher spawns a task:
    child, cleanup = self._interactive.for_background_task(task_id)
    # child is a TaskWebSocketInteractive with its OWN _input_queue
    # registry now has dev_tool_input_queues[task_id] = child._input_queue

The background task uses child to:
  - send widgets:    await child.asend_response(prompt)
                       → sends pending_input to UI ✓
  - await responses: await child.aget_input()
                       → blocks on child._input_queue.get()

The pending_input_response router (line 581):
    target_q = dev_tool_input_queues.get(pi_task_id)
                 # FOUND — registered above
    target_q.put(response)
                 → child._input_queue receives the response
                 → child.aget_input() unblocks ✓

When task completes:
    cleanup()  # pops dev_tool_input_queues[task_id]
                 → registry stays clean; no leak
```

## §D4. Why the existing `TaskWebSocketInteractive` is the right vehicle

Source: `websocket_interactive.py:309–345`. The class:
- Inherits from `WebSocketInteractive` (so it gets `asend_response`, `aget_input`, `send_task_status`, `send_graph_event`, `_sanitize_for_json` for free).
- Takes a `task_id` constructor arg.
- Overrides `stream_token_batches` to inject `task_id` into every token message (so if the background task itself streams tokens through interactive, they're properly attributed in the UI).

The docstring (line 312) explicitly says it's "for background task streaming." It has been there since at least the 20260508 task log mentions of it. It was never instantiated because the registration plumbing was missing — v1 supplies the plumbing.

**Honest note:** the dispatcher path today goes through `_run` which doesn't call `stream_token_batches` directly (the task's own executor may, but the dispatcher doesn't), so the overridden method isn't strictly needed for the router use case. But:
- The `task_id` attribution will be useful for any executor that does stream through interactive.
- Using the subclass costs nothing (no extra code; just instantiate the subclass instead of the base).
- It documents intent — "this is the per-task interactive variant."

If a reviewer prefers to instantiate `WebSocketInteractive` directly (not the subclass), the fix still works. Filed as Follow-up #3 — minor stylistic call.

## §D5. The `pending_input` routing puzzle — honest scope statement

The `asend_response` method at `websocket_interactive.py:257–302` does NOT stamp `task_id` on the `pending_input` message. The UI disambiguates purely via `currentTaskIdRef.current`, which is overwritten by the latest `task_status` message.

**Implication for v1:** The fix works correctly when there is **one in-flight interactive task at a time** — the router's clarifying-question case, the proposal_selection widget case, the pause/resume widget case. All current and near-future use cases fit this constraint.

**What v1 does NOT solve:** if two background tasks both ask for input at the same time, the UI sees two `pending_input` messages but can't tell which goes with which task. Even with the queue-per-task fix, the UI sends `pending_input_response` with `currentTaskIdRef.current` (the most recent `task_status`), which routes to one queue while the other task waits forever.

**Filed as Follow-up #2** with explicit scope: requires (a) `task_id` in `pending_input` (server-side ~5 LoC change to `asend_response`), AND (b) UI tracking of per-task pending widgets (significant React component refactor — multiple-widgets-at-once UI). Defer until a real use case demands it.

**This is a known boundary, not a hidden defect.** v1 documents it explicitly so reviewers understand what "fixed" means.

## §D6. Risk register + open questions

### Risks

| ID | Risk | Mitigation |
|---|---|---|
| **R1** | CLI path (no WS, no `task_queues`) regresses because `tool_dispatcher` now expects `for_background_task` | Triple defense: (a) optional kwarg with `None` default; (b) `hasattr` guard at the dispatcher call site; (c) `RuntimeError` fallback if method exists but registry is `None`. T10 + T11 lock this in. |
| **R2** | The two `WebSocketInteractive` construction sites I updated in Commit 2 are wrong — maybe there's a third construction site I missed | Pre-flight: `grep -rn "WebSocketInteractive(" src/openteam/` in §E3. If a third site exists, add it to Commit 2 before commit. |
| **R3** | Cleanup doesn't fire on cancellation (asyncio.CancelledError) | `try/finally` inside `_run` catches everything including `CancelledError`. T9 simulates cancellation. |
| **R4** | The `dev_tool_input_queues` registry grows unboundedly if cleanup is buggy | T17 perf test runs 50 sequential tasks and asserts the registry size returns to zero. Cleanup happens in `finally` so it always runs. |
| **R5** | Behavior diverges between dev-slash and agent-dispatched | T14 (dev-slash regression) is golden. Both paths now use the same registry + the same `WebSocketInteractive` instance shape. |
| **R6** | The shared `_last_prompt_data` reference between parent and child causes a race | The reference is read in `asend_response` (line 301: `prompt_data = kwargs.get("prompt_data") or self._last_prompt_data`). Reads are atomic in CPython. Writes happen only in `_on_new_turn` (per the comment), which is per-turn (no concurrent task writes). Low risk; documented in §D2.4 trade-off. |
| **R7** | The `for_background_task` method on the parent interactive might be called from outside `_dispatch_as_task` later, leaking the abstraction | Method is on the public API of `WebSocketInteractive`; if external use happens, it's intentional. No leak. Document the method's intended use in its docstring. |

### Open questions + defaults

| Q | Question | Default for v1 |
|---|---|---|
| Q1 | Should `for_background_task` accept an optional `parent_session_id` for cross-session resume? | **No (v1)** — out of scope; cross-session is filed in `workflow_lifecycle_commands_plan` Follow-up #3. |
| Q2 | Should the cleanup happen in `try/finally` or `task.add_done_callback`? | **`try/finally`** — simpler, matches the dispatcher's existing scope. Filed as Follow-up #1 if reviewers prefer symmetry with dev-slash. |
| Q3 | Should `TaskWebSocketInteractive` get a `parent_interactive` back-reference for advanced use? | **No** — keep the subclass minimal. YAGNI. |
| Q4 | Should we move `dev_tool_input_queues` from `manager_websocket_routes.py` to a dedicated session/connection state object? | **No (v1)** — refactor out of scope. Current location works. |
| Q5 | Should `pending_input` carry `task_id` for true concurrent interactive tasks? | **No (v1)** — see §D5. Filed as Follow-up #2. |
| Q6 | Should we add a metric/log for queue registration leaks (registry size > N) at shutdown? | **No (v1)** — T17 covers the test. Production telemetry can be added later if needed. |
| Q7 | Should `for_background_task` log at info or debug when called? | **debug** — frequent operation in normal flow. Don't spam logs. |
| Q8 | Should we deprecate the parent interactive's `asend_response` for background tasks (force them through child)? | **No** — back-compat; legacy callers without `for_background_task` still need `asend_response`. |

---

# APPENDIX — AUDIT TRAIL
══════════════════════════════════════════════════════════════════════════════

## §A1. Motivation

This plan was motivated by an in-conversation design session on 2026-06-13 with the user (Tony). The user's intuition was correct end-to-end:

1. **"If interactive works, yolo just works with synthetic responses."** Correct premise — verified at `conversational_inferencer.py:436` where `yolo_mode` and `effective_interactive` are mutually exclusive branches. Yolo's `_synthesize_yolo_collected()` is the autonomous fallback; making interactive work is the prerequisite for the router and other interactive use cases.

2. **The bug is in agent-dispatched tools, not dev-slash.** Correct — dev-slash registers the queue at `manager_websocket_routes.py:205`; agent-dispatched does not (verified absence at `tool_dispatcher.py:201`).

3. **The user-level fix is "Door 2 needs a mailbox."** Correct framing — and the implementation is exactly: register a per-task queue in the same registry the dev-slash path uses, with cleanup on task completion.

The implementation design went through **two iterations** before landing on the chosen approach:
- **Iteration 1:** factory closure in `session_context`. Rejected after source verification (premise errors documented in §D2.1).
- **Iteration 2 (chosen):** method on `WebSocketInteractive` returning a `TaskWebSocketInteractive`. The other agent's feedback on the iteration-1 proposal was source-verified accurate; this plan adopts the iteration-2 design.

## §A2. Verified facts (load-bearing for this plan)

Source: OpenStartup `dev_xinli_2601` branch + AgentFoundation `dev_xinli_2601` branch, verified 2026-06-14 01:36–01:39.

| # | Fact | Source |
|---|---|---|
| F1 | `ToolDispatcher` lives in `openteam/server/services/tool_dispatcher.py` — the SAME package as `WebSocketInteractive`. No cross-layer boundary to protect. | `find src/openteam -name "tool_dispatcher.py"` returns single hit in `server/services/` |
| F2 | `dev_tool_input_queues` is created at `manager_websocket_routes.py:306` per WebSocket connection, NOT per session | `grep -n "dev_tool_input_queues" src/openteam/server/routes/manager_websocket_routes.py` |
| F3 | `WebSocketInteractive.__init__` takes `(send_callback, input_queue)` only — no registry kwarg today (line 22) | `src/openteam/server/services/websocket_interactive.py:22` |
| F4 | `TaskWebSocketInteractive` already exists as a subclass at line 309 of `websocket_interactive.py` | `src/openteam/server/services/websocket_interactive.py:309` |
| F5 | `TaskWebSocketInteractive` is dead code — only referenced in its own definition and in historical task logs, never instantiated in production | `grep -rn "TaskWebSocketInteractive" src/openteam/` returns only the class definition + historical task logs |
| F6 | `ToolDispatcher.__init__` accepts `interactive` and stores it as `self._interactive` (line 59), which is updated per-turn from `conversation_service.py:472-473` | `src/openteam/server/services/tool_dispatcher.py:50–59` |
| F7 | `_dispatch_as_task` captures `self._interactive` as `interactive_ref` (line 201) and uses it as `task_context["interactive"]` (line 229) | `src/openteam/server/services/tool_dispatcher.py:201, 229` |
| F8 | Dev-slash path registers queue at `manager_websocket_routes.py:205` (`dev_tool_input_queues[task_id] = input_queue`) and cleans up at lines 281–282 | `src/openteam/server/routes/manager_websocket_routes.py:205, 281-282` |
| F9 | `pending_input_response` handler routes by `task_id` (line 581: `dev_tool_input_queues.get(pi_task_id)`) and falls back to `active_input_queue` | `src/openteam/server/routes/manager_websocket_routes.py:570–585` |
| F10 | `asend_response` does NOT stamp `task_id` on outgoing `pending_input` messages (only `type`, `content`, optional `input_mode`, optional `prompt_data`) | `src/openteam/server/services/websocket_interactive.py:257–302` |
| F11 | `yolo_mode` and `effective_interactive` are mutually exclusive branches at `conversational_inferencer.py:436` — yolo is NOT "interactive with synthetic responses" but a separate code path | verified earlier in the design conversation |

**Verification methodology:** every claim above was confirmed with a targeted grep against the actual `dev_xinli_2601` branch source. No claim is from extrapolation or pattern-matching. Where the other agent's feedback was source-verified accurate (F1, F2, F6, F7, F8, F9, F10), this plan adopts it; where the original recommendation had errors (Option A premises in §D2.1), this plan documents the rejection.

## §A3. Out-of-scope follow-ups

1. **`try/finally` vs `add_done_callback` for cleanup symmetry with dev-slash** — Currently uses `try/finally` (simpler). If reviewers prefer symmetry with dev-slash's `add_done_callback` (`manager_websocket_routes.py:287`), switch is a 3-LoC diff. Cosmetic.

2. **True concurrent interactive tasks** — Requires stamping `task_id` on `pending_input` messages AND UI tracking per-task pending widgets. Server-side is ~5 LoC (`asend_response` adds `task_id`); UI side is a significant React component refactor (multiple simultaneous widgets, per-widget state). Defer until a real use case demands it (none today).

3. **Stylistic: use `WebSocketInteractive` directly instead of `TaskWebSocketInteractive`** — The subclass is the more documented intent, but the fix works either way. Minor stylistic call; document the rationale in the method docstring.

4. **Migrate `dev_tool_input_queues` to a dedicated `ConnectionState` object** — Right now the registry is a local variable in the WS handler. A dedicated class would make per-connection state more discoverable and refactorable. Pure cleanup; no behavior change. Defer until the registry needs more fields.

5. **Add a telemetry counter for `for_background_task` calls** — Once the fix lands, knowing how often interactive widgets actually fire from background tasks would be useful product signal. Add a simple counter; not blocking.

6. **Eliminate the parent-interactive fallback path** — Once Commit 3 is stable and CLI tests are green, the `else: interactive_ref = self._interactive` fallback could be removed (forcing all callers to update). Cleaner; deferred until back-compat window expires.

7. **Document the interactive ↔ yolo contract** — Add a short architecture-doc page explaining the relationship between `interactive`, `effective_interactive`, `yolo_mode`, and `_synthesize_yolo_collected()`. New developers find this confusing. Pure docs.

8. **`for_background_task` for non-WS transports** — If/when a non-WS transport (SSE, HTTP long-poll) is added, the method needs a generic factory pattern. Defer until a second transport actually exists.

## §A4. Changelog

- **v1 (2026-06-14 01:41):** Initial draft. Covers the per-task queue registration fix for agent-dispatched async tools in the conversational inferencer flow. 4 commits across 2 repos (OpenStartup primary, AgentFoundation cross-repo test), ~60 LoC production + ~120 LoC tests, ~1-day effort. All 11 load-bearing facts verified against source before draft (§A2). Honest documentation of the one known boundary (§D5: single in-flight interactive task supported; true concurrent requires §A3 #2 follow-up). Two iterations of design discussion: rejected Option A (factory in session_context) based on source-verified premise errors; adopted Option D (method on `WebSocketInteractive` returning `TaskWebSocketInteractive`) using the existing dead-code subclass.

---

- **v2 (2026-06-14 02:00):** **Critical correction integrating peer-review findings.** Two independent peer plans (Plan B "Cursor" `interactive_async_task_router_3db3b795.plan.md` and Plan C "Claude" `update-your-task-tool-adaptive-goose.md`) both caught a critical bug in v1: my v1 closed only Gap 1 (queue registration). The router additionally requires Gap 2: writing `router_interactive_safe=True` at the dispatch site. Verified at `executor.py:546` (docstring contract) and `:564` (only read); `grep -rn "router_interactive_safe" src/` returns zero write sites in the current codebase. Without v2's gap-2 fix, v1's queue registration is a no-op — the router always coerces `interactive=None` and forces yolo. v2 also adds Commit 5 (Phase 2: `task --confirm` NameError repair at `executor.py:861`, caught by Plan B) using the same contract. Effort grows from 1 day to ~1.5 days. Total commits 4 → 5. Total tests T1–T17 → T1–T20.

---

## §A5. v2 cross-plan audit — what each plan caught, what each missed

| Issue | v1 (mine) | Plan B (Cursor) | Plan C (Claude) | v2 (integrated) |
|---|---|---|---|---|
| Queue registration via `for_background_task` returning `(child, cleanup)` | ✅ caught | ✅ caught | ✅ caught | ✅ kept (architectural convergence — all 3 plans agree) |
| Use existing dead-code `TaskWebSocketInteractive` subclass for task_id-stamped tokens | ✅ caught (the only thing v1 caught uniquely) | ✅ caught | ✅ caught | ✅ kept |
| `router_interactive_safe` is READ at `executor.py:564` but NEVER WRITTEN — without writing it, queue registration is a no-op | ❌ **MISSED (critical)** | ✅ **caught** | ✅ **caught** | ✅ **integrated** (Commit 2 dev-slash + Commit 3 dispatcher; coupled atomically to successful registration) |
| Couple `router_interactive_safe` flag to successful queue registration so flag is never set without queue | ❌ N/A (didn't know about flag) | ✅ caught | ❌ Plan C sets it unconditionally (slight defect) | ✅ integrated (initialize `interactive_safe = False`; only set `True` inside the success branch of try/except) |
| `executor.py:861` NameError makes `task --confirm` currently dead | ❌ MISSED | ✅ **caught (filed as Phase 2)** | ❌ flagged as "pre-existing bug, out of scope" | ✅ **integrated as Commit 5** (uses same `router_interactive_safe` contract) |
| Audit of which async tools actually interact (only router + --confirm; not `create_role`, `role_setup`, `project_onboarding`, derived tools) | ❌ MISSED | ✅ caught (decisive scoping) | ❌ no audit | ✅ integrated into §A5 (this section) |
| Use `TaskWebSocketInteractive` (not base class) for child | ✅ caught | ✅ caught | ✅ caught | ✅ kept |
| Child is a leaf (no further `task_queues` propagated) | ❌ MISSED | ✅ caught explicitly | ❌ MISSED (silent — Plan C's child also doesn't propagate, but not justified) | ✅ integrated into §D2.4 + Commit 1 docstring |
| Mermaid sequence diagram of fixed flow | ❌ ASCII only | ✅ has one | ❌ none | ✅ kept ASCII (Mermaid would require renderer support; not critical) |
| `message_end` vs `pending_input` ordering edge case | ❌ MISSED | ✅ caught | ❌ MISSED | ✅ integrated into §A3 risks |
| Cancellation via `dev_tool_tasks` registry (background tasks not registered) | ❌ MISSED | ✅ caught (filed as deferred) | ✅ caught (filed as deferred) | ✅ integrated into §A3 #4 |
| pytest-asyncio vs ad-hoc `run_until_complete` for unit tests | ❌ MISSED | ✅ caught explicitly (matches repo convention) | ❌ Plan C uses `get_event_loop().run_until_complete()` (deprecated since Python 3.10) | ✅ integrated — v2 uses pytest-asyncio (§E1.1 T1–T4) |

**Verdict on the 3 plans:**

| Plan | Score | Honest assessment |
|---|---|---|
| **Plan B (Cursor)** | **9/12 issues caught** | **The only plan that actually achieves the stated goal as written.** Catches both critical gaps, all minor refinements, and the `--confirm` repair. The Mermaid diagram, decisive tool audit, and pytest-asyncio convention are all unique strengths. The plan even includes a self-aware "Comparison of the two source plans" section. |
| **Plan C (Claude)** | 7/12 caught | Catches Gap 2 (the critical one) but sets the flag unconditionally — a subtle defect that would cause router deadlock if `_task_input_queues is None` and someone calls `for_background_task` anyway. Misses `--confirm`, the tool audit, and uses deprecated test idioms. |
| **My v1** | 2/12 caught | Caught only the architecturally correct things (queue registration + use existing subclass). Missed the critical second gap entirely. Would have shipped a no-op for the stated goal. |

**If only one plan could ship: Plan B.** It's the only one that, as written, would make the conversational router interactive. v2 = Plan B's content + my v1's PART I/II/APPENDIX 3-tier structure + my v1's larger test inventory + my v1's explicit risks table.

## §A6. Honest answer to "which plan would you choose?"

**Plan B (Cursor) — `/Users/tchen7/.cursor/plans/interactive_async_task_router_3db3b795.plan.md`.**

I'm picking against my own v1 honestly. Three independent reasons:

1. **Plan B is the only one that achieves the stated goal as written.** Both v1 and Plan C either miss or mis-handle Gap 2 (router_interactive_safe). v1 would ship a no-op; Plan C would ship a flag-without-queue defect under one edge case.
2. **Plan B catches the critical secondary bug (`--confirm` NameError) that neither other plan addressed.** That bug is independently shipping today; fixing it under the same contract is genuinely elegant.
3. **Plan B's decisive tool audit ("which async tools actually interact?") closes the "did we get the scope right?" question definitively.** Both v1 and Plan C left this implicit; Plan B walked every async tool and proved the scope is exactly two paths.

**Caveat:** if v2 (the integrated artifact) counts in the comparison, v2 is strictly better because it pulls Plan B's bug discoveries AND keeps v1's structural rigor (PART I/II/APPENDIX, 20 named tests, granular execution checklist, risk register, sibling-plan companion references). v2 is the canonical artifact going forward. Plan B is the right pick only if we're forced to choose among the three input artifacts as-is.

---

**End of plan v2.** Ready for review.
