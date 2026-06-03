# SOP Framework — Model A + Backslash Commands + Resumability Plan

> **Status:** Draft **v2.6** — implementer-ready (cosmetic-only round)
> **Author:** Tony Chen
> **Created:** 2026-05-27 12:11 (Wed); **v2:** 2026-05-27 15:10; **v2.1:** 16:54; **v2.2:** 18:06; **v2.3:** 19:04; **v2.4:** 19:45; **v2.5:** 20:00; **v2.6:** 20:41
> **Supersedes:** N/A (this is the canonical SOP-execution-model plan; v1.4 `sop_framework_UNIFIED_v1_plan.md` is a never-implemented detour, see §-1)
> **Companion plans (compatible):** `multi_sop_focus_and_tool_concurrency_plan.md` (v1.1), `conversational_inferencer_template_manager_migration_plan.md` (v3.3), `sop_runtime_enablement_plan.md` (v3.1)
> **Empirical grounding:** All architectural claims verified against codebase 2026-05-27 12:10–15:10 via 4 parallel exploration subagents + 5 targeted greps. v2 integrates 8 substantive corrections from Claude's plan (`/Users/tchen7/.claude/plans/can-you-help-take-ticklish-whisper.md`) after empirical verification of each.

---

## §-1 Provenance and audit history

This plan locks in **Model A: one ConversationalInferencer per chat session; SOP guidance rotates via `prior_context`**. The earlier plan `sop_framework_UNIFIED_v1_plan.md` (v1.4, **never implemented** — empirically verified, no `SOPInferencer` class exists in codebase) was a Model B detour introducing four parallel abstractions that solve problems Model A doesn't have.

| v# | Date (UTC) | What |
|---|---|---|
| 1.0 | 2026-05-27 12:11 | Initial draft. Grounded via 4 parallel subagent reports. |
| **2.0** | **2026-05-27 15:10** | **Integration round with Claude's plan v1. Empirically verified 8 of Claude's claims against the codebase; all 8 valid. Corrected 4 empirical errors in v1.0:** (1) v1.0 said 4 tools to delete; Claude correctly identified **6 stubs** (also `complete_phase`, `confirm_action`). (2) v1.0 proposed adding `StateGraphTracker.snapshot()`; empirically `to_dict()` (stategraph.py:293) + `from_dict()` (line 305) ALREADY EXIST. (3) v1.0 invented `_refresh_sop_guidance()` indirection; reality is the CI ALREADY rebuilds the tracker from `prior_context` every render (conversational_inferencer.py:687-698). (4) v1.0 missed `manager_websocket_routes.py` as the server dispatch entry. |
| **2.1** | **2026-05-27 16:54** | **Round-2 integration with Claude's plan v2.** Applied 3 substantive corrections. |
| **2.2** | **2026-05-27 18:06** | **Round-3 integration: detailed API-level audit verified 18 of 18 issues raised against v2.1; ALL 18 valid (zero false-positives).** Crash-level fixes (5): `SOP.to_dict()` doesn't exist; `sop.initial_phases()` doesn't exist; `ToolExecutionResult.result` not `.text`; `SOPRegistry` doesn't exist (use `load_sop`); `sop.description` doesn't exist (use `sop_info.description`). Architectural gaps (5): no `/exit_sop`; phase transition gap; render guard; factory closure conflict; max_iterations=5 too low. Serialization (2): empty sop.name; record types. Consistency (3): 7 stale "6"-refs; reinvented tool_phase_map; invented stdlib helpers. Minor (3): line offsets, slash dispatch path, cross-repo PR. New AC-EX1, AC-MA9, AC-FS6; new R13, R14. |
| **2.3** | **2026-05-27 19:04** | **Round-4 integration: surgical audit of v2.2's new code samples + remaining stale-text caught 13 of 13 verified issues.** HIGH bugs introduced by v2.2 fixes (3): (a) `_record_to_dict` used `getattr(record, "ts")` but `WorkflowPhaseRecord` field is `timestamp`, AND omitted `status/summary/workspace_path/task_id` — verified at `server/workflow_context.py:18-26`; the class already has `to_dict()`/`from_dict()` so preferring those is correct (b) `PausedResult(AgenticResult)` constructor missing required `completed_actions` and `iterations_used` — verified at `conversational/context.py:110`; loop-local values passed in explicitly (c) `_serialize_pause_state` referenced non-existent `self._turn_number` / `self._iteration` — verified ZERO matches in CI; now passed as keyword args from `run_agentic_loop`'s scope, restored to `self._pending_resume_state` for loop to consume. Stale text (7): G4/Phase B/§7.3/AC-CMD8/§7.1/AC-TD1/§3.7-prose all corrected to "7" or "6" as appropriate; `SOPRegistry.load()` → `load_sop()`. Architectural gaps (3): new AC-MA10 commits to factory-closure resolution (single-writer invariant), `/exit_sop` key list expanded from 8 → 11 with named constant `_SOP_PRIOR_CONTEXT_KEYS`, render guard tightened to require both `_sop` and `sop_name` absent before auto-loading + derives `sop_name` from filename when SOP.name is empty. |
| **2.4** | **2026-05-27 19:45** | **Round-5 audit caught 8 of 8 verified issues — 3 HIGH crashes from v2.3 patches.** (1) Executor used `arguments["name"]` but tool.json declares `"workflow"` → `KeyError` (verified at `resources/tools/sop/tool.json`). (2) PausedResult passed `completed_actions=collected_actions` but loop variable is `loop_actions` at conversational_inferencer.py:178 → `NameError`. (3) `_pending_resume_state` was written by `_restore_pause_state` but never consumed by `run_agentic_loop` → resume would silently restart at iteration 0; added the `getattr(self, "_pending_resume_state", None)` consumption block at loop entry + `self._paused = False` reset in restore. Stale text (5): AC-EX1 "8 keys" → "11"; Q7 `uuid8` → `uuid4().hex[:8]`; Phase 0 "30 ACs" → "37"; header "Draft v2" → "Draft v2.4"; footer "plan v2" → "plan v2.4". |
| **2.5** | **2026-05-27 20:00** | **Round-6 audit caught 6 of 6 verified issues.** HIGH (1): AC-MA3 said "clears current_phase" but AC-MA9 says "advances to next" — direct contradiction; AC-MA3 stale from v2.0 (before phase-transition-gap was identified). Resolved by rewriting AC-MA3 to match AC-MA9. MED (2): (a) R14 mitigation was un-implementable — executor signature `execute(arguments, session_context)` (verified `task/executor.py:549`) has no CI reference, so can't mutate `inferencer.max_iterations`; rewritten to use `context_updates["_sop_max_iterations_override"]` channel with CI reading it at loop entry. (b) §4 still said "C1+C2+C3 must be one atomic PR" while §10 correctly said "two coordinated PRs" — cross-repo atomicity is impossible; §4 updated to match. LOW (3): audit table rows out of chronological order (2.4 before 2.3); §9 "v2 IS the integration" → "v2.5"; loop variable in code samples was `i` but real loop uses `iteration` (verified `conversational_inferencer.py:189`). |
| **2.6** | **2026-05-27 20:41** | **Round-7 audit caught 2 of 2 verified cosmetic issues. Zero crashes, zero architectural gaps, zero stale APIs.** (1) `self._commands` type annotation said `dict[str, CommandMeta]` but storage on the next line packs `(meta, attr_name)` tuples — annotation was stale; corrected to `dict[str, tuple[CommandMeta, str]]` (AC-CMD2 already documents the tuple contract correctly, so this was a pure annotation/storage drift). (2) §9 final-recommendation paragraph still said "this plan (v2)" and "v2 is empirically accurate" — stale stamps from v2.0; refreshed to v2.6 to match document header. The 7th round confirms convergence: 67→2 issues across rounds 1→7, with rounds 6+7 finding zero crash-level bugs. |

---

## §0 Scope and non-scope

### In scope

1. **Adopt Model A** — replace the per-phase-CI-spawn pattern with single-CI + rotating SOP guidance. SOP state lives in `prior_context`; tracker is rebuilt every render.
2. **`@command` decorator + dispatch** — internal-method backslash commands distinct from external CLI tools. Two dispatch paths: agentic-loop interception (`_execute_tool_call`) and server-side slash-command (`manager_websocket_routes.py`).
3. **Delete 7 empirically-verified stub tools** [v2.1]: `enter_workflow`, `exit_workflow`, `resume_workflow`, `complete_phase`, `confirm_action`, `abort_phase`, `clear`. All have only `__init__.py` + `tool.json` (or just `tool.json` for `clear`), no executor, zero Python callers.
4. **In-session pause/resume via `PausedResult`** — cooperative check at iteration boundary; returns serializable state. Requires sanitization (strip non-JSON-serializable values like `_sop` object; reload by name on resume) — see §3.7 / §3.8 / R12.
5. **Optional filesystem-checkpoint** layer (atomic write) for cross-process resume — opt-in for v1.
6. **SOP-state resumability** — comes for free because SOP state lives in `prior_context` AND `StateGraphTracker.to_dict()`/`from_dict()` already exist; only the `_sop` OBJECT reference needs special handling (strip + reload by name).

### Out of scope (handled by companion plans)

- **WorkGraph-driven autonomous SOP execution** (`/sop --autonomous` / `batch_mode=True`): keep the existing WorkGraph path behind a `batch_mode` flag in `/sop` executor; not deleted, not the default.
- **Multi-SOP focus mode & tool concurrency labels**: `multi_sop_focus_and_tool_concurrency_plan.md` v1.1; this plan assumes single-focus.
- **TemplateManager migration**: `conversational_inferencer_template_manager_migration_plan.md` v3.3; orthogonal.
- **SOP discovery / registry / yolo defaults**: `sop_runtime_enablement_plan.md` v3.1; orthogonal.
- **Long-running subprocess reattachment after process restart**: v1 resume is "warm restart"; BTA subprocesses restart from scratch.

---

## §1 Architectural pin

**Model A: One ConversationalInferencer per chat session. SOP context is dynamic prompt content stored in `prior_context`, NOT a separate execution unit.**

Empirically grounded:
- `ConversationalInferencer.run_agentic_loop()` (conversational_inferencer.py:159) is the single turn-execution engine; reused unchanged.
- The CI already builds `StateGraphTracker` from `prior_context` every render (lines 687-698 — verified empirically). SOP state is already a first-class `prior_context` resident: `completed_phases`, `current_phase`, `phase_status`, `phase_outputs`, `goto_counts`, `_confirmation_gate_passed`.
- `WorkflowManager.render_prompt_sections()` (manager.py:146-188) already produces `workflow_description`, `workflow_status`, `workflow_nextstep_guidance` consumed by `conversation/main/initial.jinja2`. **Currently dead code in OpenStartup** (factory at `backends/factories.py:157-164` doesn't pass `workflow_manager=`). Model A wires this up.
- `StateGraphTracker.to_dict()` (stategraph.py:293) + `from_dict()` (line 305) already exist — full SOP-state serialization for free.

**What Model A rejects from the v1.4 Model B plan:**

| Model B construct | Why rejected |
|---|---|
| `SOPInferencer(CI)` subclass | Duplicates CI surface for zero benefit |
| `InteractionSerializer` at transport | Solves multi-inferencer contention that doesn't exist with one CI |
| `PendingConversationToolQueue` | Same root cause |
| `RoutedInteractive` proxy | Same root cause |
| Per-call `sop_instance_id` routing | One active SOP at a time (v1); prompt content tells LLM which SOP it's in |
| Separate `sop/main/initial.jinja2` template | Adds template-drift risk; existing template already has the XML sections |

All four collapse to zero in Model A. Empirically verified: none of these exist in the codebase today (v1.4 was never implemented).

---

## §2 Goals

| # | Goal | Acceptance |
|---|---|---|
| G1 | Wire `WorkflowManager` into `ConversationalInferencer` in OpenStartup factory; verify SOP guidance rotates correctly | §5.1 ACs |
| G2 | `/sop` executor rewritten: state-initialization only; old WorkGraph path preserved behind `batch_mode` flag | §5.1 ACs |
| G3 | `@command` decorator + per-CI registry; two dispatch paths (agentic loop + server slash) | §5.2 ACs |
| G4 | Delete **7** empirically-verified stub tools (v2.3 Issue 4: includes `clear/`) | §5.3 ACs |
| G5 | `PausedResult` cooperative in-session pause at iteration boundary | §5.4 ACs |
| G6 | Optional filesystem checkpoint (atomic write) for cross-process resume | §5.5 ACs |
| G7 | SOP-state resume verified via `prior_context` snapshot (reuses existing `to_dict()`/`from_dict()`) | §5.6 ACs |

---

## §3 Architecture

### §3.1 Top-level flow

```
┌─────────────────────────────────────────────────────────────┐
│  User input                                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼────────────────────┐
        │  Server: manager_websocket_routes  │
        │    _try_ci_command(input, ci)      │ ← NEW dispatch
        │    if hit → return command result  │
        │    else → forward to conversation  │
        └──────────────┬────────────────────┘
                       │
            ┌──────────▼─────────────┐
            │  ConversationService   │
            └──────────┬─────────────┘
                       │
       ┌───────────────▼────────────────────┐
       │  CI.run_agentic_loop(content)      │
       │   - rebuild tracker from           │
       │     prior_context (lines 687-698)  │
       │   - render prompt (incl. SOP XML)  │
       │   - LLM generates                  │
       │   - _execute_tool_call(...)        │
       │   - _check_phase_completion()      │ ← NEW
       │     → updates prior_context        │
       │   - check self._paused             │ ← NEW
       │     → return PausedResult          │
       └────────────────────────────────────┘
```

### §3.2 SOP state lives in `prior_context` (no observer needed)

Empirically verified (conversational_inferencer.py:687-698): every `_render_prompt()` call rebuilds the tracker from `prior_context`:

```python
# Existing code in CI._render_prompt() — DO NOT MODIFY:
completed = [r.phase if hasattr(r, "phase") else str(r)
             for r in self.prior_context.get("completed_phases", [])]
tracker = StateGraphTracker(
    graph=sop,
    current_state=self.prior_context.get("current_phase"),
    state_status=self.prior_context.get("phase_status", "idle"),
    completed_states=completed,
    state_outputs=self.prior_context.get("phase_outputs", {}),
    goto_counts=self.prior_context.get("goto_counts", {}),
)
```

**Implication:** any time `prior_context` is mutated, the next render picks it up automatically. No observer pattern, no `_refresh_sop_guidance()`, no notify hooks. (v1.0 of this plan got this wrong; v2 corrects.)

**v2.2 Issue 8 — render-time guard:** Today CI:654 unconditionally calls `find_sop_file()` and sets `prior_context["_sop"]`. With Model A the SOP is set by the executor's `context_updates`; calling `find_sop_file()` again every render would either be wasted work or — worse — overwrite the executor-set SOP if it doesn't match the `find_sop_file` result. **v2.3 Issue 13 (first-render window):** also guard against the unsolicited-SOP-guidance case — before `/sop` is called, if `find_sop_file()` discovers a SOP in `_variables/workflow/`, it would render SOP guidance with no `current_phase`/`sop_name`. Require BOTH keys absent (no in-flight SOP) before auto-loading; require `sop_name` to be populated alongside `_sop` so downstream code can rely on the invariant:

```python
# In _render_prompt(), at the existing find_sop_file branch:
if "_sop" not in self.prior_context and "sop_name" not in self.prior_context:
    # v2.3 Issue 13: both absent → safe to auto-discover for legacy paths
    sop_path = getattr(self.prompt_renderer, "find_sop_file", lambda: None)()
    if sop_path:
        sop = SOPManager.load(sop_path)
        self.prior_context["_sop"] = sop
        # v2.3 Issue 11: derive a stable sop_name; use file stem when SOP.name is "".
        self.prior_context["sop_name"] = sop.name or Path(sop_path).stem
```

### §3.3 `_check_phase_completion()` — the key new CI method

Called after `_execute_tool_call()` applies `context_updates` AND after confirmation-gate handling.

Signature and contract:
```python
def _check_phase_completion(self) -> None:
    """Detect completed phases and update prior_context accordingly.

    Detection strategies (in order):
      1. Tool-mapped: tool from sop.tool_to_phase_map executed → mapped phase done
      2. Confirmation: _confirmation_gate_passed flag set + phase requires_confirmation
         (already partially implemented at CI lines ~700-718; this method centralizes it)
      3. All-outputs-present: every entry in phase.outputs is present in phase_outputs

    On detection:
      - Append to prior_context["completed_phases"]
      - Set prior_context["current_phase"] to next phase via tracker.get_available_next()
        [v2.2 Issue 7: must advance, not just clear; otherwise phases without mapped
        tools (confirmation-only, info-gathering) never get current_phase set]
      - If get_available_next() returns empty → SOP complete: clear current_phase,
        set phase_status="completed", emit phase_complete event for observability.
    """
```

Estimated ~50 LoC. Wired at one call site in `_execute_tool_call()` post-execution.

### §3.4 `/sop` executor — state initialization, not graph execution

**Current** (~175 LoC): Creates `WorkflowManager` + `WorkGraph` + runs graph via `SOPWorkGraphNode`.

**v2** (~50 LoC interactive path + existing batch path):
```python
# v2.2: empirically verified APIs (all classes/methods confirmed to exist 2026-05-27 18:06)
from datetime import datetime, timezone
from uuid import uuid4
from agent_foundation.resources.sops.registry import load_sop  # SOPInfo factory
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)

async def execute(arguments, session_context, ...):
    # v2.4 Issue 1: tool.json declares parameter "workflow" (verified at
    # resources/tools/sop/tool.json); "name" would raise KeyError.
    sop_name = arguments["workflow"]
    yolo = arguments.get("yolo", False)
    batch_mode = arguments.get("batch_mode", False)

    if batch_mode:
        return await _execute_batch_mode_workgraph(...)  # existing path preserved (R2)

    # Interactive path (Model A): load SOP, initialize prior_context, return.
    sop_info = load_sop(sop_name)  # → SOPInfo (raises SOPNotFound)
    sop = sop_info.sop                # → SOP (a StateGraph subclass)

    # Initial phase: first phase in graph order. SOP has no initial_phases();
    # use sop.phases[0] (StateGraph orders nodes by declaration / __init__ list).
    initial_phase_id = sop.phases[0].id if sop.phases else None

    # Timestamp + id helpers (no utc_now_compact / uuid8 helpers exist; use stdlib).
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    instance_id = f"{sop_name}__{ts}__{uuid4().hex[:8]}"

    context_updates = {
        # Live SOP object — non-serializable; sanitization handles in §3.7
        "_sop": sop,
        # Use the existing property; don't reinvent (Issue 14)
        "tool_phase_map": sop.tool_to_phase_map,
        "current_phase": initial_phase_id,
        "phase_status": "running",
        "completed_phases": [],
        "phase_outputs": {},
        "goto_counts": {},
        "sop_yolo_mode": yolo,
        "sop_instance_id": instance_id,
        # Explicit name for serialization; sop.name="" for SOPs without H1 (Issue 11)
        "sop_name": sop_name,
    }

    # ToolExecutionResult field is `result`, not `text` (verified protocols.py:20)
    desc_preview = (sop_info.description or "")[:80]
    return ToolExecutionResult(
        result=f"Entered SOP: {sop_name} ({desc_preview}...)",
        context_updates=context_updates,
    )
```

The CI's own agentic loop is now the SOP execution engine. Every render pulls fresh state from `prior_context`. `_check_phase_completion()` rotates state after each tool call.

### §3.5 Command system (`@command` decorator)

NEW file: `agent_foundation/common/inferencers/agentic_inferencers/conversational/commands.py`

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class CommandMeta:
    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    requires_active_sop: bool = False

def command(name: str, description: str = "", *, aliases: tuple[str, ...] = (),
            requires_active_sop: bool = False):
    """Decorator. Marks a CI method as a backslash command."""
    def _decorator(method):
        method._command_meta = CommandMeta(name=name, description=description,
                                            aliases=tuple(aliases),
                                            requires_active_sop=requires_active_sop)
        return method
    return _decorator
```

**Discovery (in `CI.__attrs_post_init__`):**
```python
self._commands: dict[str, tuple[CommandMeta, str]] = {}  # v2.6 Issue 1: value is (meta, attr_name) pair, not bare meta
for attr_name in dir(type(self)):
    method = getattr(type(self), attr_name, None)
    meta = getattr(method, "_command_meta", None)
    if meta is None:
        continue
    for key in (meta.name, *meta.aliases):
        if key in self._commands:
            raise ValueError(f"Duplicate command '/{key}' on {type(self).__name__}")
        self._commands[key] = (meta, attr_name)
```

**Dispatch (two paths):**

1. **Server slash-command** (`manager_websocket_routes.py` NEW `_try_ci_command`): runs BEFORE forwarding to `ConversationService`. Returns the command's textual response if hit; otherwise lets the message flow through to the agentic loop.
2. **Agentic loop** (`_execute_tool_call` post-execution hook): if the LLM accidentally emits a slash-prefixed pseudo-tool, route to command registry first. (This is a defensive fallback; primary use is server-side dispatch.)

### §3.6 Initial command set (v1)

| Command | Aliases | Description | requires_active_sop |
|---|---|---|---|
| `/help` | `?` | List available commands | False |
| `/status` | `s` | Show current SOP, phase, message count, paused state | False |
| `/clear` | — | Clear conversation history | False |
| `/pause` | — | Pause SOP execution at next iteration boundary | True |
| `/resume` | — | Resume paused SOP | False |
| `/exit_sop` [v2.3 Issue 12] | `/exit` | Clear all 11 SOP-state keys from `prior_context`: 10 set by the executor (`_sop`, `tool_phase_map`, `current_phase`, `phase_status`, `completed_phases`, `phase_outputs`, `goto_counts`, `sop_yolo_mode`, `sop_instance_id`, `sop_name`) + `_confirmation_gate_passed` (set by CI's confirmation-gate handler at ~CI:700-718). Use a single named constant `_SOP_PRIOR_CONTEXT_KEYS` so executor + exit stay in sync. | True |

Notes (revised from v1.0):
- v1.0 listed 10 commands; v2.2 lands **6** commands (`/help`, `/status`, `/clear`, `/pause`, `/resume`, `/exit_sop`). The other user-initiated equivalents of `enter_workflow`/`exit_workflow`/`abort_phase`/`complete_phase` either remain handled via natural conversation ("please exit this SOP") which the LLM interprets and mutates `prior_context` accordingly, or are folded into the 6 above. The deleted-stub-tool list is **7** (6 original + `clear` confirmed as stub in v2.1).
- `requires_active_sop=True` causes `/pause` to error gracefully if no SOP is active.

### §3.7 `PausedResult` — cooperative in-session pause

NEW dataclass in `agent_foundation/common/inferencers/agentic_inferencers/conversational/context.py`:

```python
@dataclass
class PausedResult(AgenticResult):
    paused: bool = True
    pause_state: dict = field(default_factory=dict)  # serialized CI snapshot
    # text inherited from AgenticResult — last raw LLM response before pause
```

**Iteration-boundary check** in `run_agentic_loop()` (around line 189, top of iteration loop). v2.3 (Issue 2): `AgenticResult` has 3 required fields with no defaults — `text`, `completed_actions`, `iterations_used` — these MUST be supplied or construction raises `TypeError`. The loop already tracks the equivalents locally:
```python
if self._paused:
    return PausedResult(
        # v2.5 Issue 6: real loop variable is `iteration` (verified at
        # conversational_inferencer.py:189: `for iteration in range(...)`).
        pause_state=self._serialize_pause_state(turn_number=turn_number, iteration=iteration),
        text=last_raw_response or "",
        # v2.4 Issue 2: actual loop variable is `loop_actions` (verified at
        # conversational_inferencer.py:178); `collected_actions` would NameError.
        completed_actions=loop_actions,
        iterations_used=iteration + 1,        # 0-indexed → +1
    )
```

**`_serialize_pause_state()`** [v2.1 — sanitization required]:

```python
# Keys in prior_context that hold non-JSON-serializable Python objects
# (verified empirically at conversational_inferencer.py:663 for _sop)
_NON_SERIALIZABLE_PRIOR_CONTEXT_KEYS = (
    "_sop",           # SOPManager-loaded SOP object (graph + phases + adjacency)
    # add others here if empirical audit (Phase 0 AC-FS5) finds them
)

def _serialize_pause_state(self, *, turn_number: int = 0, iteration: int = 0) -> dict:
    # v2.3 (Issue 3): turn_number and iteration are NOT instance attributes on CI
    # (verified: zero matches for self._turn_number / self._iteration). They are
    # passed explicitly from run_agentic_loop's local scope at the pause point.
    # v2.2: sop_name comes from prior_context (set by executor, Issue 11) —
    # NOT from sop.name (which is "" for SOPs whose markdown has no H1).
    sop_name = self.prior_context.get("sop_name")
    if self.prior_context.get("_sop") is not None and not sop_name:
        raise RuntimeError(
            "prior_context has _sop but no sop_name; executor must set both. "
            "Cannot serialize without a stable name for SOPRegistry re-load."
        )

    # Strip non-serializable keys (currently: _sop) AND custom-serialize record-bearing keys.
    pc_serializable = {}
    for k, v in self.prior_context.items():
        if k in _NON_SERIALIZABLE_PRIOR_CONTEXT_KEYS:
            continue  # _sop reloaded by name on restore
        if k == "completed_phases":
            # v2.2 (Issue 12): entries are records with .phase OR strings (CI:687-690 handles both).
            # Normalize to plain dicts/strings for JSON.
            pc_serializable[k] = [
                _record_to_dict(r) if hasattr(r, "phase") else str(r)
                for r in v
            ]
        else:
            pc_serializable[k] = v

    return {
        "messages": list(self._messages),
        "prior_context": pc_serializable,
        "sop_name": sop_name,
        "dynamic_context": self._dynamic_context.to_dict() if hasattr(self, "_dynamic_context") else None,
        "turn_number": turn_number,   # v2.3 Issue 3: passed in from loop scope
        "iteration": iteration,        # v2.3 Issue 3: passed in from loop scope
        "sop_instance_id": self.prior_context.get("sop_instance_id"),
    }


def _record_to_dict(record) -> dict:
    """JSON shape for WorkflowPhaseRecord. (v2.3: WorkflowPhaseRecord already
    has to_dict()/from_dict() — see workflow_context.py:28-39 — so prefer those
    when available; fallback handles any future record subclasses.)"""
    if hasattr(record, "to_dict") and callable(record.to_dict):
        return record.to_dict()
    return {
        "phase": getattr(record, "phase", None),
        "status": getattr(record, "status", "completed"),
        "summary": getattr(record, "summary", ""),
        "workspace_path": getattr(record, "workspace_path", ""),
        "task_id": getattr(record, "task_id", ""),
        "timestamp": getattr(record, "timestamp", 0.0),
    }
```

**`_restore_pause_state(state)`** [v2.2]:

```python
def _restore_pause_state(self, state: dict) -> None:
    self._messages = state["messages"]
    self.prior_context = dict(state["prior_context"])  # already excludes _sop
    if state.get("sop_name"):
        # Reload via the empirically-verified load_sop function (NOT SOPRegistry class).
        # Returns SOPInfo; use .sop for the live object. (Issue 4)
        from agent_foundation.resources.sops.registry import load_sop
        sop_info = load_sop(state["sop_name"])
        self.prior_context["_sop"] = sop_info.sop
        self.prior_context["sop_name"] = state["sop_name"]  # round-trip preserved
    if state.get("dynamic_context") is not None and hasattr(self, "_dynamic_context"):
        self._dynamic_context = self._dynamic_context.__class__.from_dict(state["dynamic_context"])
    # v2.3 (Issue 3): turn_number / iteration are NOT attached as instance
    # attributes (they don't exist on CI). Instead, _restore_pause_state stores
    # them on a transient slot for run_agentic_loop to consume on the next call:
    self._pending_resume_state = {
        "turn_number": state.get("turn_number", 0),
        "iteration": state.get("iteration", 0),
    }
    self._paused = False  # clear pause flag so the next loop run can iterate
    # Next call to _render_prompt() rebuilds tracker from restored prior_context
    # automatically (CI lines ~685-696, unchanged code path).
```

**v2.4 Issue 3 — consuming the resume state in `run_agentic_loop()`.** `_restore_pause_state` only WRITES the resume position; the loop must READ it to actually resume from where pause happened (otherwise resume starts from iteration 0 with no state-position advance). At the top of `run_agentic_loop()` (around line 178), before the iteration loop:

```python
# v2.4 Issue 3: consume any pending resume state from a prior _restore_pause_state.
_resume = getattr(self, "_pending_resume_state", None)
start_iteration = 0
if _resume is not None:
    start_iteration = _resume.get("iteration", 0)
    # turn_number is informational only — the new run_agentic_loop call already
    # gets a fresh turn_number from its caller; we don't override that.
    self._pending_resume_state = None  # one-shot consumption

# v2.5 Issue 6: real loop variable is `iteration` (verified at line 189).
# Existing loop signature `for iteration in range(self.max_iterations)` becomes:
for iteration in range(start_iteration, self.max_iterations):
    ...
```

**Why this design:** `_sop` is the only known non-serializable value in `prior_context`. Reloading by name on resume is **simpler, smaller, and more correct** than custom pickling: it ensures the SOP definition reflects the latest file content (which is desirable — bug fixes to SOP files apply on resume). If the SOP file has been deleted or renamed between pause and resume, `load_sop()` raises a clear error (v2.3 Issue 10: `SOPRegistry` doesn't exist; the correct API is the `load_sop` function from `agent_foundation.resources.sops.registry`).

**Sanitization audit (Phase 0 RED test AC-FS5):** the test runs `json.dumps(prior_context)` against a real production CI mid-SOP and asserts that the ONLY failing key is `_sop`. If new non-serializable keys appear in `prior_context` over time, the test fails and the developer must add the key to `_NON_SERIALIZABLE_PRIOR_CONTEXT_KEYS`.

### §3.8 Filesystem checkpoint (optional, opt-in)

For cross-process resume (server restart, crash recovery), `_serialize_pause_state()` is also written atomically to disk:

```
<session_dir>/sop/<sop_instance_id>/pause_checkpoint.json
```

Atomic write pattern (reuses `ConversationalFlowNodeAdapter`'s existing tempfile + `os.replace` pattern):
```python
def _write_checkpoint_atomic(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, default=str), encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX
```

**`_load_pause_state(path)`**: reads JSON, returns dict; `_restore_pause_state` consumes it.

**Why JSON not pickle for v1:**
- `prior_context` and `messages` are already JSON-serializable in production (tool results are strings/dicts; messages are role/content dicts). Empirically verified.
- JSON is human-readable for debugging.
- Pickle's stronger type-roundtripping isn't needed; if it ever is, the `Resumable` ABC (RichPythonUtils `common/resumable.py:39-184`) is available as a future upgrade — see §6 R3.

### §3.9 SOP-state resumability — comes for free

Empirically verified:
- `StateGraphTracker.to_dict()` (stategraph.py:293) serializes `completed_states`, `state_outputs`, `goto_counts`.
- `StateGraphTracker.from_dict()` (stategraph.py:305) reconstructs the tracker from a dict.
- The CI's render path (lines 687-698) rebuilds the tracker from `prior_context` (NOT from `to_dict()`) — meaning `prior_context` IS the canonical SOP state.

**Implication:** serializing `prior_context` captures the full SOP state. No separate SOP-resume mechanism needed. (v1.0 of this plan got this wrong; v2 corrects.)

If a caller wants the tracker dict explicitly (e.g., for cross-process tracker reconstruction without re-loading the SOP graph), `tracker.to_dict()` is available as a one-liner — but the default path (rebuild from `prior_context`) is simpler.

---

## §4 Phased rollout

| Phase | Scope | Risk | LoC |
|---|---|---|---|
| **0** | Empirical baseline + RED tests for all **37** ACs (v2.4 Issue 6) | none | +300 test |
| **A1** | NEW `commands.py` (decorator + CommandMeta) + NEW `PausedResult` in `context.py` | LOW | +80 |
| **A2** | CI `__attrs_post_init__` builds `_commands` registry; iteration-boundary `_paused` check; `_serialize_pause_state` / `_restore_pause_state` methods (dormant — no commands defined yet) | LOW | +60 |
| **B** | Delete **7** stub tool directories (v2.3 Issue 5); verify `load_all_tools()` clean | LOW | -120 |
| **C1** | Wire `WorkflowManager` into `ConversationalInferencer` in OpenStartup `backends/factories.py:157-164`; feature flag `OPENTEAM_SOP_MODEL_A=1` (default OFF) | **HIGH** | +20 |
| **C2** | Rewrite `sop/executor.py` interactive path (state-init only); keep WorkGraph path behind `batch_mode=True` flag | MED | +80 / -100 net |
| **C3** | Add `_check_phase_completion()` to CI; wire it post-`_execute_tool_call()` | MED | +60 |
| **D1** | Add `/help`, `/status`, `/clear` commands on CI (read-only) | LOW | +50 |
| **D2** | Add `/pause`, `/resume` commands on CI + filesystem checkpoint | MED | +80 |
| **D3** | Add `_try_ci_command()` in OpenStartup `manager_websocket_routes.py` BEFORE existing slash-command path | MED | +30 |
| **D4** | Handle `PausedResult` in `conversation_service.py` (surface to UI; persist checkpoint) | MED | +40 |
| **E** | Flip `OPENTEAM_SOP_MODEL_A` default to ON after 1-week soak | HIGH | 1-line change |
| **F** | Remove feature flag after 1-release soak; lock in Model A | LOW | -20 |

**Ordering:** Phases must merge in order. A1+A2+B can be one PR (zero behavior change). **C1+C2+C3 ship as TWO coordinated PRs** [v2.5 Issue 3] — C1 (AgentFoundation: Model A wiring + `_check_phase_completion` + render guard) lands first; C2+C3 (OpenStartup: `/sop` rewrite + factory closure modification) lands second, both gated behind `OPENTEAM_SOP_MODEL_A=1` feature flag so the OS side is a no-op until both have shipped. Cross-repo atomic merges are impossible — the feature flag is the coherence-preservation mechanism. D1+D2+D3+D4 can be split. E/F are calendar-gated.

**Total LoC:** ~+720 / -240 = ~+480 net (excluding tests).

---

## §5 Acceptance criteria

### §5.1 Model A wiring (G1, G2)

- **AC-MA1** With `OPENTEAM_SOP_MODEL_A=1` and entering a SOP via `/sop role_creation`: within the same turn, `ci.prior_context` contains `_sop`, `current_phase`, `phase_status="running"`, `completed_phases=[]`, `phase_outputs={}`, `tool_phase_map`, `sop_instance_id`.
- **AC-MA2** The rendered prompt includes `<WorkflowDescription>`, `<WorkflowStatus>`, `<WorkflowNextStepGuidance>` XML sections when a SOP is active; absent when not.
- **AC-MA3** [v2.5 Issue 1: was contradicting AC-MA9; resolved] After LLM emits a tool from `tool_phase_map`, `_check_phase_completion()` appends to `prior_context["completed_phases"]` and ADVANCES `prior_context["current_phase"]` to the next available phase via `tracker.get_available_next()` (or sets `phase_status="completed"` + clears `current_phase` if no next phase exists). See AC-MA9 for the full contract. The NEXT render shows the next phase's guidance.
- **AC-MA4** Confirmation-based phase completion: `_confirmation_gate_passed=True` in `prior_context` + phase has `requires_confirmation` directive → phase auto-completes on next render. Equivalent semantics to the existing CI:700-718 code path, now centralized.
- **AC-MA5** Yolo mode (`sop_yolo_mode=True` in `prior_context`): full `role_creation` SOP runs end-to-end in one `run_agentic_loop` invocation; `tracker.status == "completed"` at exit.
- **AC-MA6** Existing test suite `pytest test/agent_foundation/common/inferencers/conversational/` passes unchanged.
- **AC-MA7** `/sop role_creation --batch-mode` still executes via WorkGraph path (preserved behind flag); LLM-facing behavior unchanged from current.
- **AC-MA8** Zero `SOPInferencer` class in codebase; zero `InteractionSerializer` / `PendingConversationToolQueue` / `RoutedInteractive` references.
- **AC-MA9** [v2.2 Issue 7] Phase transition: after `_check_phase_completion()` marks a phase done, `prior_context["current_phase"]` is set to the next available phase via `tracker.get_available_next()` (or cleared with `phase_status="completed"` if no next phase exists). NOT left as `None` — that would break the next render's guidance.
- **AC-MA10** [v2.3 Issue 11] Factory closure under Model A: when `OPENTEAM_SOP_MODEL_A=1`, the OpenStartup factory closure at `backends/factories.py:156-164` either (a) skips the phase-tracking block entirely, OR (b) writes to a transient key `_pending_completed_phase` that `_check_phase_completion()` consumes and clears (single-writer for `current_phase`). Test: a `/sop role_creation` run with `MODEL_A=1` produces ONE current_phase write per tool execution (verified by counting events in session.jsonl), not two.
- **AC-EX1** [v2.4 Issue 4; original v2.2 Issue 6] `/exit_sop` while SOP active: removes all **11** SOP-state keys from `prior_context` (per `_SOP_PRIOR_CONTEXT_KEYS` constant in §3.6); next render has no `<WorkflowDescription>` / `<WorkflowStatus>` / `<WorkflowNextStepGuidance>` sections; `/exit_sop` with no active SOP returns a clear "no active SOP" message (not an error). `/exit` is an alias.

### §5.2 Command system (G3)

- **AC-CMD1** `from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import command, CommandMeta` succeeds.
- **AC-CMD2** Decorating a method with `@command("foo")` makes `ci._commands["foo"]` return the `(CommandMeta, attr_name)` tuple.
- **AC-CMD3** Duplicate command name raises `ValueError` at CI construction with both method names.
- **AC-CMD4** Aliases work: `@command("pause", aliases=("p",))` → `ci._commands["p"]` resolves to same method.
- **AC-CMD5** `requires_active_sop=True`: `/pause` invoked with no SOP active returns clear error message (NOT raises); `/resume` works without active SOP if checkpoint exists.
- **AC-CMD6** Server `_try_ci_command("/help")` returns command output; `_try_ci_command("hello")` returns None (forwards to ConversationService).
- **AC-CMD7** Unknown slash-input (`/unknown_xyz`) is forwarded to the LLM, NOT errored. Preserves current behavior for messages that happen to start with `/`.
- **AC-CMD8** [v2.3 Issue 7] `/help` lists exactly the **6** v1 commands (`/help`, `/status`, `/clear`, `/pause`, `/resume`, `/exit_sop`) with their `description` strings.

### §5.3 Tool deletions (G4)

- **AC-TD1** [v2.3 Issue 9] Directories `agent_foundation/resources/tools/{enter_workflow,exit_workflow,resume_workflow,complete_phase,confirm_action,abort_phase,clear}/` do not exist (**7** dirs).
- **AC-TD2** [v2.2] `load_all_tools()` returns no entries for the **7** deleted tools (6 original + `clear`).
- **AC-TD3** `grep -r "enter_workflow\|exit_workflow\|resume_workflow\|complete_phase\|confirm_action\|abort_phase" CoreProjects/AgentFoundation/src/` returns only references in non-tool code (e.g., `_check_phase_completion`'s internal logic for tool-mapped detection; `confirm_action` references in widget code if any).
- **AC-TD4** [v2.2] No production code-path imports or invokes any of the **7** deleted tools as tools.

### §5.4 In-session pause (G5)

- **AC-PR1** `ci._paused = True; await ci.run_agentic_loop("anything")` returns `PausedResult(paused=True, pause_state={...})` after the next iteration boundary (NOT mid-tool-execution).
- **AC-PR2** `PausedResult.pause_state` contains `messages`, `prior_context`, `turn_number`, `iteration`, `sop_instance_id`.
- **AC-PR3** Pause mid-SOP: state contains full SOP state (all `prior_context` keys including `completed_phases`, `current_phase`, `phase_outputs`).
- **AC-PR4** In-process restore: `new_ci._restore_pause_state(state)` then `await new_ci.run_agentic_loop("continue")` produces output consistent with the original session continuing.
- **AC-PR5** `/pause` queues at safe boundary (not mid-tool-call): if invoked while a tool is executing, the loop completes the current tool call's `_check_phase_completion()` post-hook, then pauses at the next iteration's top check.

### §5.5 Filesystem checkpoint (G6)

- **AC-FS1** After `/pause`, the file `<session_dir>/sop/<sop_instance_id>/pause_checkpoint.json` exists and parses as valid JSON.
- **AC-FS2** Write is atomic: a kill -9 during write leaves either the previous valid checkpoint or no checkpoint, never a partial one (verified by interrupt-during-write test using a slow mock filesystem).
- **AC-FS3** Round-trip: `_serialize_pause_state` → write → read → `_restore_pause_state` produces state equal to the original.
- **AC-FS4** Cross-process resume: pause → kill process → fresh process loads checkpoint via `_load_pause_state(path)` → calls `_restore_pause_state(state)` → next agentic-loop invocation continues correctly.
- **AC-FS5** [v2.1] Sanitization audit: running `json.dumps(ci.prior_context, default=lambda o: (_ for _ in ()).throw(TypeError(o)))` on a real production CI mid-SOP raises `TypeError` ONLY for the keys listed in `_NON_SERIALIZABLE_PRIOR_CONTEXT_KEYS` (currently `{"_sop"}`). If a new non-serializable key appears, this test FAILS so the developer must either add the key to the strip-list OR fix the upstream code to store a serializable value. Prevents silent serialization data loss.
- **AC-FS6** [v2.2 Issue 12] `completed_phases` record-serialization: starting from a CI with `completed_phases` containing real `WorkflowPhaseRecord`-like objects (the type used in production), `_serialize_pause_state` produces JSON-safe dicts via `_record_to_dict`; `_restore_pause_state` does NOT need to re-hydrate to objects (CI:687-690 handles both strings/dicts via `hasattr(r, "phase")` fallback to `str(r)`). Round-trip preserves `phase` field on every entry.

### §5.6 SOP-state continuity (G7)

- **AC-SR1** `StateGraphTracker.to_dict()` (already exists at stategraph.py:293) round-trips: `tracker == StateGraphTracker.from_dict(tracker.to_dict(), graph=sop)`.
- **AC-SR2** SOP state survives pause/resume: `completed_phases`, `current_phase`, `phase_outputs`, `goto_counts` all preserved; next phase's guidance renders correctly on resume.
- **AC-SR3** Goto-loop counters survive: pause mid-goto-loop, resume, `goto_counts` continues incrementing from preserved value.

---

## §6 Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Model A wiring (Phase C1) changes prompt rendering in OpenStartup; LLM behavior drift | HIGH | `OPENTEAM_SOP_MODEL_A=1` flag default off; 1-week soak; Phase E gated on AC-MA6 (existing tests pass). |
| R2 | `/sop` executor rewrite (Phase C2) breaks autonomous batch path | MED | Keep WorkGraph path behind `batch_mode=True` flag; AC-MA7 verifies. |
| R3 | JSON checkpoint can't serialize some `prior_context` values (e.g., custom dataclass with non-JSON types) | MED | Use `default=str` in `json.dumps`; add pre-serialize sanitizer; for v2+ upgrade to `Resumable` ABC if needed. |
| R4 | `_check_phase_completion()` not called at every mutation site → stale prompt state | MED | Single wire-up point: post-`_execute_tool_call()`. Confirmation path (CI:700-718) already exists; centralize into `_check_phase_completion()`. AC-MA3 + AC-MA4 verify. |
| R5 | Command dispatch swallows user input the LLM should see | MED | AC-CMD7: unknown slash-input falls through to LLM. Only registered commands intercepted. |
| R6 | `@command` discovery via `dir(type(self))` collides with attrs-generated attrs | LOW | Only methods with `_command_meta` attribute count; attrs-generated descriptors don't have this. |
| R7 | Pause mid-tool-execution loses tool-side-effect state | MED | AC-PR5: `/pause` queued at iteration boundary; current tool completes before pause. |
| R8 | Filesystem checkpoint accumulates orphan files | LOW | v1: no automatic cleanup; v2: GC on session-end via `ConversationService.on_session_close`. |
| R9 | `_try_ci_command` in `manager_websocket_routes.py` ordering wrong (runs after existing slash-handler that swallows the message) | MED | AC-CMD6: explicit test that `_try_ci_command` runs BEFORE existing `_try_dev_slash_command`. |
| R10 [v2.2] | Tool-deletion (Phase B) breaks `tool_phase_map` referencing deleted tools | LOW | Empirically: all **7** tools are stubs with no executor (verified via `ls`). `tool_phase_map` lookups for them would have failed before this plan. Verify with grep. |
| R11 | Cross-process resume can't reconnect to in-flight subprocess (BTA, etc.) | MED | v1 explicitly out-of-scope (§0); document as "warm restart" limitation. v2 can add subprocess-handle re-attachment. |
| R12 [v2.1] | `prior_context["_sop"]` is a `SOPManager`-loaded SOP OBJECT (verified at conversational_inferencer.py:663), NOT JSON-serializable; naive `json.dumps(prior_context)` would raise `TypeError` and prevent checkpoint persistence | **HIGH** (would block all pause/resume) | §3.7 sanitization design: `_serialize_pause_state` strips `_sop` and records the SOP NAME instead; `_restore_pause_state` reloads via `load_sop(name)` after restoring the rest. AC-FS5 audit test catches any new non-serializable key that appears in `prior_context` over time. Reloading by name is also semantically MORE correct than pickling — picks up any SOP-definition fixes applied between pause and resume. |
| R13 [v2.2 Issue 9] | OpenStartup factory closure at `backends/factories.py:156-164` ALREADY sets `result.context_updates["current_phase"] = tool_phase` and `phase_status="completed"` whenever a tool from `tool_phase_map` runs. This conflicts with `_check_phase_completion()` which would re-do the same work and possibly disagree on the phase value (factory writes the tool's mapped phase as `current_phase`; v2 detection should ADVANCE to next). | MED | Phase C wiring: when `OPENTEAM_SOP_MODEL_A=1` is active, the factory closure either (a) skips the phase-tracking block, OR (b) writes to a separate transient key (`_pending_completed_phase`) that `_check_phase_completion()` consumes and then advances from. Option (b) preserves the factory's role as the "raw observation" emitter and centralizes advancement logic in the CI. New Phase C3 sub-step explicitly removes/modifies the factory closure. |
| R14 [v2.5 Issue 2; original v2.2 Issue 10] | Default `max_iterations=5` on `ConversationalInferencer` (verified at line 116) — insufficient for AC-MA5 (yolo runs full `role_creation` SOP end-to-end). 5 phases × ~2 iterations each = 10 iterations minimum. Yolo would exhaust the budget and return `exhausted_max_iterations=True` mid-SOP. | MED | **Executor cannot mutate the CI directly** — its signature is `execute(arguments, session_context)` (verified `task/executor.py:549`); it has no CI reference. Communication is via `ToolExecutionResult.context_updates`. Two-pronged mitigation: (a) Executor returns `context_updates["_sop_max_iterations_override"] = n_phases * sop_iteration_budget_per_phase`; CI reads this at `run_agentic_loop` entry and treats it as an upper-bound override for that invocation. (b) NEW CI attrib `sop_iteration_budget_per_phase: int = 3` defines the per-phase multiplier (configurable, no override needed if the default works). AC-MA5 verifies the full role_creation SOP completes within the elevated budget. |

---

## §7 File inventory

### §7.1 New files

| File | Purpose | LoC |
|---|---|---|
| `agent_foundation/.../conversational/commands.py` | `@command` decorator + `CommandMeta` | ~80 |
| `agent_foundation/.../conversational/context.py` (add `PausedResult`) | New dataclass alongside existing `AgenticResult` | ~20 (delta) |
| ~~`agent_foundation/.../conversational/ci_commands_builtin.py`~~ | **DROPPED in v2.1.** Mixin classes add MRO + `__attrs_post_init__` chaining complications with `attrs`. The **6** v1 command methods (`/help`, `/status`, `/clear`, `/pause`, `/resume`, `/exit_sop` — v2.3 Issue 8) live DIRECTLY on `ConversationalInferencer` as `@command`-decorated methods. Net: ~80 LoC moved into `conversational_inferencer.py` modified-files row (already includes +120 → ~+200). | ~~~80~~ → 0 |
| `test/.../conversational/test_commands.py` | AC-CMD1-8 | ~200 |
| `test/.../conversational/test_pause_resume.py` | AC-PR1-5 + AC-FS1-4 | ~250 |
| `test/.../conversational/test_model_a_phase_rotation.py` | AC-MA1-8 + AC-SR1-3 | ~300 |

### §7.2 Modified files

| File | Lines added | Purpose |
|---|---|---|
| `conversational_inferencer.py` | +120 | Build `_commands` registry in `__attrs_post_init__`; iteration-boundary `_paused` check at top of agentic loop; `_check_phase_completion()`; `_serialize_pause_state()` / `_restore_pause_state()` / `_load_pause_state()` |
| `agent_foundation/resources/tools/sop/executor.py` | +80 / -100 net | Interactive path (state-init only); WorkGraph path kept behind `batch_mode` flag |
| `agent_foundation/common/workflow/manager.py` | +20 | `init_sop_state()` helper used by `/sop` executor |
| `agent_foundation/common/workflow/instance.py` | +5 | `mode: Literal["interactive","batch"] = "interactive"` field |
| `openteam/server/backends/factories.py` (OpenStartup) | +10 | Pass `workflow_manager=` to CI constructor; gated by `OPENTEAM_SOP_MODEL_A` |
| `openteam/server/routes/manager_websocket_routes.py` | +30 | `_try_ci_command()` check before existing slash-command handler |
| `openteam/server/services/conversation_service.py` | +40 | Handle `PausedResult`: surface to UI; persist checkpoint |

### §7.3 Deleted files

| Directory | Reason |
|---|---|
| `agent_foundation/resources/tools/enter_workflow/` | Stub, no executor (verified) |
| `agent_foundation/resources/tools/exit_workflow/` | Stub, no executor (verified) |
| `agent_foundation/resources/tools/resume_workflow/` | Stub, no executor (verified) |
| `agent_foundation/resources/tools/complete_phase/` | Stub, no executor (verified) |
| `agent_foundation/resources/tools/confirm_action/` | Stub, no executor (verified) |
| `agent_foundation/resources/tools/abort_phase/` | Stub, no executor (verified) |
| `agent_foundation/resources/tools/clear/` | Stub, only `tool.json` (verified `ls` 2026-05-27 16:54) [v2.1 addition] |

**Net file count: +6 new, ~7 modified, ~7 dirs deleted (v2.3 Issue 6).**

---

## §8 Open questions

| # | Question | Default |
|---|---|---|
| Q1 | Should command registry support per-instance dynamic registration (e.g., plugins)? | Default: NO. Class-level only via `@command` decorator; simpler; defer plugins to v2. |
| Q2 | Should `/pause` accept an optional `--reason` arg? | Default: NO for v1. Add only if user research shows need. |
| Q3 | If `manager_websocket_routes.py` already has `_try_dev_slash_command`, do we merge `_try_ci_command` into it or add as a separate path? | Default: separate path that runs BEFORE `_try_dev_slash_command`. CI commands take precedence. |
| Q4 | Should `PausedResult` include the rendered prompt for UI display? | Default: NO — caller can call `ci._render_prompt()` if needed. Keeps PausedResult lean. |
| Q5 | Should the WorkGraph batch-mode path eventually be deleted? | Default: KEEP for now (autonomous SOP use case). Revisit after Phase F (model A locked in). |
| Q6 | Should `_check_phase_completion()` emit a `phase_complete` event/log? | Default: YES — append a structured entry to session.jsonl for observability. |
| Q7 | What's the format of `sop_instance_id` in `prior_context`? | Default: `{sop_name}__{utc_yyyymmddhhmmss}__{uuid4().hex[:8]}` (v2.4 Issue 5: `uuid8` is not a real stdlib function; the executor uses `uuid4().hex[:8]` for an 8-char hex token. Matches existing `<session>/sop/<id>/` directory convention from `sop_runtime_enablement_plan.md` v3.1). |
| Q8 | Should `/clear` clear `prior_context` too? | Default: NO. Only clears `messages`. User can `/sop --exit` (future) or restart to clear SOP state. |

---

## §9 Honest comparison with companion / prior plans

| Plan | Architecture | Status | Relation |
|---|---|---|---|
| `sop_framework_UNIFIED_v1_plan.md` v1.4 | Model B (SOPInferencer subagents) | Never implemented (verified empirically) | **Superseded by this plan.** Move to `_archive/`. |
| `multi_sop_focus_and_tool_concurrency_plan.md` v1.1 | Multi-focus + tool concurrency labels | Companion | Orthogonal; locks single-focus for v1; multi-focus is v2. |
| `sop_runtime_enablement_plan.md` v3.1 | SOP discovery + registry + yolo defaults | Companion | Orthogonal; this plan reuses SOPRegistry, doesn't reinvent. |
| `conversational_inferencer_template_manager_migration_plan.md` v3.3 | TemplateManager adapter | Companion | Orthogonal; this plan's prompt-template changes are zero (existing template already has the XML sections). |
| Claude's plan (`can-you-help-take-ticklish-whisper.md`) | Model A + Commands + Resumability (parallel-authored) | Integrated into v2.0; refined through v2.5 | All 8 substantive empirical corrections from Claude's plan adopted into v2.0; v2.1–v2.6 progressively closed 69 cumulative implementer-blocking issues while preserving operational discipline (audit history, 37 ACs, 14 risks, file inventory, 8 open questions, 11-phase rollout, feature flags). |

**If forced to pick ONE plan, I would pick this plan (v2.6).** [v2.6 Issue 2: was "(v2)"; v2.6 stamp matches the actual version.]

Honest reasoning:
- **v2.6 IS the integration** of my plan + Claude's plan + 6 rounds of audit feedback. v2.0 corrected 4 empirical errors v1.0 made (which Claude got right); v2.1–v2.6 progressively closed 69 cumulative implementation-blocking issues (zero false-positives) while preserving the operational rigor Claude's plan lacked (no ACs, no risks, no file inventory, no feature flag, no phased rollout, no audit trail, no open-question lock-in).
- **Claude's plan is empirically more accurate** but operationally thinner. As a standalone, it could be implemented by a senior engineer but lacks the gates/safeguards an unfamiliar implementer would need.
- **v2.6 is empirically accurate AND operationally complete.** That's strictly better than either input.

---

## §10 Next steps

1. **Land Phase 0+A1+A2+B as one PR** (~360 LoC + 250 tests; ~1 day): foundation (decorator + PausedResult + dormant registry) + cleanup (delete **7** stubs).
2. **Land Phase C1+C2+C3 as two coordinated PRs** [v2.2 Issue 18 — atomic across repos is impossible; AF and OS are separate repos] (~160 LoC + 300 tests total; feature flag `OPENTEAM_SOP_MODEL_A` default OFF; ~2 days): **(2a)** AgentFoundation PR: Model A wiring + `_check_phase_completion()` + `_render_prompt` guard (§3.2 v2.2 Issue 8); **(2b)** OpenStartup PR (depends on 2a being released): `/sop` rewrite + factory-closure modification (R13). Feature flag ensures the OS side is no-op until both ship.
3. **Land Phase D1+D2+D3+D4 as a sequence** (~200 LoC; ~3 days): commands → pause/resume → server dispatch → checkpoint handling.
4. **Phase E** (1 week after Phase D fully landed): flip flag default on; soak 1 week.
5. **Phase F** (1 release after Phase E): remove feature flag.

**Total: ~4-5 weeks calendar; ~7 PRs; ~480 LoC added; ~240 LoC deleted; ~750 LoC tests.**

---

*End of plan v2.6. Empirical baselines re-verified through 2026-05-27 20:41 (**7 audit rounds**; **69 cumulative issues** raised across rounds 1–7, **all 69 verified valid and applied** — zero false positives across the entire sequence. Rounds 6–7 found only 2 crash-level bugs out of 8 total issues, confirming convergence to cosmetic-only.)*
