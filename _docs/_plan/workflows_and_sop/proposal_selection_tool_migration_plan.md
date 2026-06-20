# Proposal Selection Migration + `proposals.json` Pipeline Fix + SOP Phase 3b/4 Wiring — Integrated Plan v3.6

> **⚠ Reviewers: this file is `v3.6` (latest as of 2026-06-07 19:18). v3.6 hardens v3.5's Commit 6a with the two real issues an external review caught: (1) the missing `ToolExecutionResult` import (without which the snippet would `NameError` at runtime — `registry.py` only imports `ToolDefinition` today), and (2) a defensive `tool_name.replace('-', '_')` belt-and-braces guard against any future caller that bypasses `_resolve_tool_name`. Also documents the verified async-timing safety: `update_prior_context` runs at line 1222 BEFORE `_check_phase_completion` at line 1227 and the `ToolCompletion` inbox event at line 1234, so the augmented key is guaranteed populated before Phase 3b can render. v3.6 is the first version where every line of the proposed code has been traced end-to-end against current source.**
>
> **The file is split into three clearly-labelled tiers:**
> - **PART I — EXECUTION** (§E0–§E3): what to do, in what order.
> - **PART II — DESIGN REFERENCE** (§D1–§D6): why this design is correct.
> - **APPENDIX — AUDIT TRAIL** (§A1–§A6): how every claim was verified.

**Author:** Rovo Dev (drafted in conversation with Tony Chen)
**Date:** 2026-06-07 v3.6 (supersedes v3.5 of same day)
**Status:** Draft v3.6 — ready for review
**Branch:** `dev_xinli_2601`

**Companion to:**
- `sop_runtime_enablement_plan.md`
- `conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md`

---

# PART I — EXECUTION

> **Audience:** the agent or engineer landing the change.
> **Read:** §E0 → §E1 → §E2 → §E3 top-to-bottom.
> **Skip unless needed:** PART II (design rationale), APPENDIX (audit evidence).

---

## §E0. Quick-start

1. **Make the branch:** `git checkout -b feat/proposal_selection_e2e_pipeline` off `dev_xinli_2601`.
2. **Land Commits 1 → 6 in §E1** in strict dependency order (Commit N requires Commits 1..N−1). Each commit is independently reviewable, revertable, and tested.
3. **Validate** after each commit per §E2.1, and run the end-to-end smoke (§E2.2) after Commit 6.
4. **Push** the AF PR; reference this plan.
5. **Follow up** with Commit 7 (rankevolve adoption) as a separate repo PR — also in §E1.

The full execution checklist with `[ ]` boxes is in §E3.

If any step's intent is unclear, jump to:
- the **design rationale** for the corresponding decision in PART II (§D2 architecture, §D3 data flow),
- or the **verified evidence** for the underlying defect in the APPENDIX (§A2 empirical baseline).

---

## §E1. Migration plan — file-by-file, seven commits in dependency order (six in AF + one in rankevolve)

Each commit is independently reviewable and revertable.

### §E1.1 Commit 1 — D1 fix: BTA file-fallback for truncated responses

**Why first:** Without `proposals.json` actually appearing on disk, no downstream change provides user value.

**Files modified:**

1. `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py`
   - **(a) Text source fallback** — function `_try_extract_proposal_index(self, response)` (line ~1118): after the existing `text = str(response) if response is not None else ""` line, add: if `"proposal_index"` is not in `text` AND `self.aggregator_inferencer` exists AND its `_workspace` is set, read the aggregator's output file (try `agg_ws.output_path(agg_inf.output_path or "output.md")` first, then `agg_ws.deliverable_path(...)` as second candidate). On hit, replace `text` with the file contents and continue the existing extraction logic. Wrap file read in `try/except (OSError, UnicodeError)`; log warning on failure.
   - **(b) Stable-location INVARIANT comment (the ONLY contract Phase 3b/4 will rely on)** — add a code comment immediately above the sidecar-write block:
     ```python
     # INVARIANT (depended on by model_optimization/SOP.md Phase 3b and Phase 4):
     #   proposals.json lives at <research-propose workspace>/outputs/proposals.json
     #
     # Rationale:
     #   - BTA._finalize_output early-returns when aggregator is present (line 1100),
     #     so the base class's outputs→final_deliverables move never runs for this file.
     #   - This location is stable and discoverable by convention.
     #
     # Any future refactor of _finalize_output that moves this file MUST also update:
     #   - src/agent_foundation/resources/sops/model_optimization/SOP.md Phase 3b (proposals_path tool_arg)
     #   - src/agent_foundation/resources/sops/model_optimization/SOP.md Phase 4 (--use-proposal flag)
     #   - tests/.../test_bta_proposals_json_invariant.py (location regression test added in this commit)
     ```

> **Why this is the right design (v3.3 correction):** v3.2 of this plan attempted to populate `phase_outputs["proposals_path"]` directly from the BTA inferencer via a method `_update_phase_output(...)`. **That method does not exist anywhere in the codebase** (verified by repo-wide `grep`: zero matches). The BTA inferencer has no `phase_outputs` attribute either — `phase_outputs` is owned exclusively by the SOP runtime (`SOPState.phase_outputs` in `common/workflow/sop_state.py:23` is a *flat* `field(default_factory=dict)`, and the only mutation point is `WorkflowContext.complete_phase(**outputs)` in `server/workflow_context.py:124`, which is itself never called from any tool executor today). So v3.2's Commit 1b would have silently no-op'd (the `getattr(self, "phase_outputs", None)` returns `None`, the fallback branch skips, and no one ever knew). The principled fix is to (1) keep the INVARIANT documented (this commit) and (2) have the SOP body in Commit 6 reference the path **by convention via SOP template substitution** (e.g. `{{ phase_outputs.research_propose_workspace }}/outputs/proposals.json`), which is the mechanism the existing SOP runtime already supports for the `workflow_target_path` pattern (see `code_optimization/SOP.md:17` — *"target codebase at `{{ workflow_target_path }}`"*). The proper future evolution — declaring tool outputs in `tool.json` so the runtime auto-captures them — is tracked as **§A5 follow-up #10**.

**Tests added (or extended):**
- `test/.../flow_inferencers/test_breakdown_then_aggregate_inferencer.py`
  - **New test**: "fence found in response (fast path)" — existing behaviour, regression guard.
  - **New test**: "response truncated, fence found in aggregator output file" — feed a fake response without the fence and a tmp aggregator workspace with the fence on disk; assert `proposals.json` is written.
  - **New test**: "response truncated AND aggregator file missing" — assert function returns cleanly (no crash, no `proposals.json`).
- `test/.../proposal/test_parser.py`
  - **New test**: "parser tolerates 78KB markdown with 36KB proposal_index fence" — fixture file (~80 KB) copied from a real `_runtime/tasks/research_propose/.../outputs/output.md` (anonymized).

**Risk:** low. Fix is additive — fast path unchanged. **LoC:** ~40 production + ~120 tests.

### §E1.2 Commit 2 — D2 fix: tolerant constraint parsing

**Why second:** Even with D1 fixed, a single malformed constraint in the LLM output crashes the whole `ProposalIndex` parse and prevents `proposals.json` from being written.

**Files modified:**

1. `src/agent_foundation/common/data_models/proposal/model.py`
   - `ProposalConstraint.from_dict()` (lines 132–141): replace direct `d["id"]` / `d["kind"]` access with `d.get()` calls + alias map + scalar-or-list normalisation. **Verified against real LLM output, all aliases listed below are required, not speculative:**

     ```python
     def _as_list(x):
         """Normalise scalar OR list into list (LLM dialect β emits both for `to`)."""
         if x is None: return []
         if isinstance(x, list): return list(x)
         return [x]

     @classmethod
     def from_dict(cls, d: dict[str, Any]) -> ProposalConstraint:
         return cls(
             id=d.get("id", ""),
             kind=d.get("kind", d.get("type", "unknown")),
             proposal_ids=_as_list(d.get("proposal_ids", d.get("from"))),
             requires_ids=_as_list(d.get("requires_ids", d.get("to"))),
             label=d.get("label", ""),
             reason=d.get("reason", d.get("rule", d.get("note", ""))),
             severity=d.get("severity", "error"),
         )
     ```

     **Alias map rationale (every entry observed in production output):**
     | Canonical field | Alias(es) | Source |
     |---|---|---|
     | `kind` | `type` | dialect α and β |
     | `proposal_ids` | `from` (scalar→list) | dialect β |
     | `requires_ids` | `to` (scalar→list) | dialect β |
     | `reason` | `rule` | dialect α |
     | `reason` | `note` | dialect β |

   - `ProposalIndex.from_dict()` (lines 188–200): wrap constraint construction in a per-item try/except. Log a `warning` with the offending dict and skip. Keep all valid constraints. **Never** lose the whole index over one bad constraint.

2. `src/agent_foundation/common/data_models/proposal/parser.py`
   - Add a `logger.warning(...)` when `parse_proposal_file()` returns `None` because of a parse exception (not just file-missing), so silent failures become loud.

**Tests added (every test below uses a real LLM-output fixture, not synthetic data):**
- `test/.../proposal/test_model.py`
  - **Canonical schema round-trip** (regression guard): `{"id":"C1","kind":"requires","proposal_ids":["P2"],"requires_ids":["P1"]}` round-trips identically.
  - **Dialect α (free-form):** `{"type":"ordering","rule":"P10 must precede every other proposal."}` parses with `kind="ordering"`, `reason="P10 must…"`, `id=""`, `proposal_ids=[]`, `requires_ids=[]`.
  - **Dialect β (scalar `to`):** `{"type":"requires","from":"P5","to":"P1","note":"ORPO benefits most…"}` parses with `kind="requires"`, `proposal_ids=["P5"]`, `requires_ids=["P1"]`, `reason="ORPO benefits most…"`.
  - **Dialect β (list `to`):** `{"type":"requires","from":"P4","to":["P1","P3"]}` parses with `requires_ids=["P1","P3"]`.
  - **Dialect β with `reason` not `note`:** `{"type":"recommends","from":"P3","to":"P1","reason":"..."}` parses with `reason="..."`.
  - **Completely empty `{}`:** parses to all defaults (`id=""`, `kind="unknown"`, lists empty, strings empty) without crashing.
  - **`ProposalIndex.from_dict()` partial-failure tolerance:** input with 3 constraints (2 valid + 1 deliberately malformed nested-type) keeps the 2 valid; emits a `warning` for the 1 malformed.

**Risk:** low. Backward compatible (canonical dicts produce identical output). **LoC:** ~50 production + ~120 tests.

### §E1.3 Commit 3 — D3 framework primitives (enum + Protocol + parser hook)

**Why third:** With D1+D2 fixed, `proposals.json` exists. Now we add the substrate the conversation tool will plug into.

**Files added / modified in AF:**

1. `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversation_tools.py`
   - Add `PROPOSAL_SELECTION = "proposal_selection"` to `ConversationToolType`.
   - Update any factory/serialiser that switches on the enum (likely just adds a passthrough case).

2. `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/protocols.py`
   - Add `HubAwareToolExecutor` Protocol verbatim from rankevolve. `@runtime_checkable`, `typing`-only imports.

3. `src/agent_foundation/common/data_models/proposal/parsers.py` (**NEW**)
   - `ProposalParser` Protocol: `def parse(workspace: str) -> Proposal | None`.
   - Module-level `_default_parser: ProposalParser | None = None`.
   - `register_proposal_parser(parser)` setter.
   - `get_proposal_parser()` getter — returns `None` if unregistered.
   - Single-parser registry is sufficient for v1 (rankevolve is the only producer). Extensible to multi-parser dispatch later without touching the handler.

4. `src/agent_foundation/common/data_models/proposal/__init__.py`
   - Re-export `ProposalParser`, `register_proposal_parser`, `get_proposal_parser`.

**Tests added:**
- `test/.../conversational/test_protocols.py` — `HubAwareToolExecutor` is recognised by `isinstance` on a stub that satisfies the Protocol; rejected on one that doesn't.
- `test/.../data_models/proposal/test_parsers.py` — registration round-trip, `None` when unregistered, replacement semantics, thread-safety of the module-level variable (single set+get test is sufficient).

**Risk:** trivial. **LoC:** ~120.

### §E1.4 Commit 4 — D3 handler migration

**Files added / modified in AF:**

1. `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/handlers/proposal_selection.py` (**NEW**)
   - Port the 280 LoC handler from rankevolve. **Four** localised edits (3 dependency inversions + 1 input-source priority change):
     - **I1 / I2 / I3** (dependency inversions):
       - Replace `from rankevolve...` imports with `from agent_foundation...` imports.
       - Replace the inline `from rankevolve...proposal_parser import parse_proposals` (currently inside `enrich_before_send`) with `from agent_foundation.common.data_models.proposal.parsers import get_proposal_parser; parser = get_proposal_parser(); if parser is None: return original_message`.
       - Keep the `isinstance(executor, HubAwareToolExecutor)` narrowing untouched — Protocol now lives in AF.
     - **I4 — `proposals_path` priority chain (resolves Item 7 from v3.1):** Make `enrich_before_send` aware that the SOP author can pass `proposals_path` as a `tool_args` value (declared in `tool.json` below). Today the rankevolve handler only reads from `phase_outputs["research_proposals_data"]` / `["research_proposals"]` / `["unified_plan_path"]`. After this change the resolution order is:
       1. **Highest priority:** `tool_args["proposals_path"]` (explicit SOP-author intent — the model_optimization SOP body passes this via Jinja substitution `{{ workspace_path__research_propose }}/outputs/proposals.json`, where `workspace_path__research_propose` is published into `prior_context` by Commit 6a's bridge-dispatcher augmentation around the existing `task_execute` `context_updates` emission).
       2. ~~Fallback A: `phase_outputs["proposals_path"]`~~ — **REMOVED in v3.4.** This layer existed in v3.1–v3.3 on the assumption that Commit 1 would populate `phase_outputs["proposals_path"]` via `_update_phase_output(...)`, but that method doesn't exist (bug B1, fixed in v3.3). v3.4 deletes this dead-fallback layer outright — the explicit `tool_args` path (item 1) plus the rankevolve back-compat keys (next item) are sufficient and verifiable.
       3. **Fallback B (rankevolve back-compat):** `phase_outputs.get("research_proposals_data")` and the existing rankevolve discovery chain (`research_proposals`, `unified_plan_path`).
       4. **Last resort:** registered parser via `get_proposal_parser()` (rankevolve-only path).
     - Implementation: a small private helper `_resolve_proposals_source(self, ctx, tool_args) -> str | dict | None` encapsulates the chain so each layer can be tested independently. **No behaviour change for rankevolve** (its existing keys remain in the chain at the same priority). New behaviour for AF: out-of-the-box discovery via the BTA-populated `proposals_path`.
   - Keep module-level helpers (`format_hub_announcement`, `create_hub`, `_group_selected_by_batch`) — framework-grade. If used elsewhere in rankevolve, they continue to work via the re-export from rankevolve's `protocols.py` (see §E1.7).

2. `src/agent_foundation/.../conversational/handlers/__init__.py`
   - `from .proposal_selection import ProposalSelectionHandler`.
   - Add one line to `default_registry()` — **NB:** the registry's `register()` method takes a single argument and derives `tool_type` from `handler.tool_type` (verified: `handler_registry.py` lines 30–37). The existing block uses local variable `reg`, not `registry`. So the correct call is:
     ```python
     reg.register(ProposalSelectionHandler())
     ```
   - Also bump the module docstring's "5 generic handlers" to "6 generic handlers".

3. `src/agent_foundation/resources/tools/proposal_selection/tool.json` (**NEW**) — required because all 4 other conversation tools (`clarification`, `confirmation`, `single_choice`, `multiple_choice`) ship one. **Verified by `ls`**, not assumed. Canonical shape (modelled on `multiple_choice/tool.json`):

   ```json
   {
     "name": "proposal_selection",
     "tool_type": "Conversation",
     "category": "conversation",
     "description": "Present a structured set of proposals (with batch/phase grouping, dependencies, impact/complexity) and let the user select which to advance.",
     "parameters": [
       {"name": "prompt", "type": "string", "required": true,
        "description": "The selection prompt to show the user."},
       {"name": "proposals_path", "type": "string", "required": true,
        "description": "Path to proposals.json produced by research-propose (or any source compatible with parse_proposal_file)."},
       {"name": "preselected_ids", "type": "string", "required": false,
        "description": "Comma-separated proposal IDs pre-checked when the widget renders."},
       {"name": "allow_zero", "type": "flag", "default": false,
        "description": "Allow the user to submit zero selections (otherwise the widget requires at least one)."}
     ],
     "usage_guidance": "Use after research-propose to let the user review and select which proposals to implement. Pairs with `task --use-proposal` for downstream execution.",
     "examples": [
       "{\"tool_type\": \"proposal_selection\", \"prompt\": \"Which proposals to implement?\", \"proposals_path\": \"<ws>/outputs/proposals.json\"}"
     ],
     "yolo_default": {"mode": "select_all"}
   }
   ```

**Tests added:**
- `test/.../conversational/handlers/test_proposal_selection_handler.py` — port from rankevolve (its `MagicMock(spec=ToolExecutorCallable)` test discipline transfers cleanly).
- **New**: "handler runs cleanly without a registered parser" — `enrich_before_send` no-ops gracefully.
- **New**: "handler runs cleanly without `HubAwareToolExecutor`" — returns `"Selected N proposals"` synthetic; doesn't crash.
- **New**: `default_registry()` count test — asserts exactly 6 handlers (defends against accidental removal).

**Risk:** low. Existing rankevolve tests lock in the contract. **LoC:** ~350.

### §E1.5 Commit 5 — D3 UI migration (React)

**Files added / modified in AF:**

1. `src/agent_foundation/ui/react-shared/src/inputs/ProposalSelectionWidget.js` (**NEW**) — port 1132 LoC widget verbatim from rankevolve. Replace `useProposalOverrides` import with a context-aware version (see next file).
2. `src/agent_foundation/ui/react-shared/src/hooks/useProposalOverrides.js` (**NEW**) — reads the overrides endpoint from `useContext(ProposalOverridesEndpointContext)`. Default (no provider): returns static `{ rankings: [], deprioritize: [], applied_changes_log: [] }`. Keeps the existing window `'proposal_overrides_applied'` event listener (browser-global, safe).
3. `src/agent_foundation/ui/react-shared/src/contexts/ProposalOverridesEndpointContext.js` (**NEW**) — `React.createContext({ endpoint: null, fetcher: defaultNoOpFetcher })`.
4. `src/agent_foundation/ui/react-shared/src/protocol/registerBuiltins.js` — one-line registration:
   ```javascript
   import ProposalSelectionWidget from '../inputs/ProposalSelectionWidget';
   registerWidget('proposal_selection', ProposalSelectionWidget);
   ```

**Tests added:**
- Jest snapshot for the widget with a representative `ProposalSelectionData` payload (port from rankevolve).
- Jest test that the no-op `useProposalOverrides` fetcher returns the expected empty shape.

**Risk:** medium — large UI port. Mitigations: style stack (`styled-components`, `framer-motion`, `lucide-react`) verified consistent with AF's other widgets (`ConfirmationWidget`, `MultipleChoiceWidget`); widget is registry-mounted, so any unregistered dependency surfaces as a clean fallback, not a crash. **LoC:** ~1250.

**Note for CLI-first deployments:** the AF approach is **richer-and-graceful** rather than simplified: ship the full widget, but the no-op overrides context means CLI use never touches the rankevolve endpoint. CLI rendering goes through AF's existing text-mode widget shim (same pattern as `MultipleChoiceWidget`) — no separate CLI variant needed.

### §E1.6 Commit 6 — SOP author fixes (D3 wiring + F4 wiring)

**Files modified in AF:**

1. `src/agent_foundation/resources/sops/model_optimization/SOP.md`
   - **Phase 3b — revert the `confirmation` workaround**, align with `code_optimization/SOP.md` line 90, **and declare the output_variable explicitly** (mirrors the existing `workflow_target_path` pattern in `code_optimization/SOP.md:10` — *"The output variable MUST be named `workflow_target_path` for this conversation tool"*):
     ```markdown
     ### Phase 3b -- Proposal Review & Selection
     [__depends on__ Phase 3]

     [__requires user input__] After the research & proposal phase completes,
     present the unified proposals to the user for review and selection. Use a
     `proposal_selection` conversation tool (or `confirmation` if
     proposal_selection is unavailable). Pass `proposals_path` =
     `{{ workspace_path__research_propose }}/outputs/proposals.json` (this path is
     the INVARIANT documented in Commit 1b, and the Jinja variable is published
     by the bridge dispatcher in Commit 6a). The output variable MUST be
     named `selected_proposal_ids` for this conversation tool — Phase 4
     consumes it via the same name.

     **Tools**[__must__]:
     - proposal_selection
     ```
   - **Phase 4 — wire `--use-proposal` + `--proposal-ids`** (F4), using SOP template substitution from `phase_outputs` (the mechanism that already powers `{{ workflow_target_path }}` in `code_optimization/SOP.md:17`):
     ```markdown
     ## Phase 4 -- Implementation, Experiment & Analysis
     [__depends on__ Phase 3b; __branch__]

     For each selected proposal from Phase 3b, plan and implement the
     proposed changes by invoking the `task` tool with:
       --use-proposal  = `{{ workspace_path__research_propose }}/outputs/proposals.json`
                         (the INVARIANT location documented in Commit 1b;
                          the Jinja variable is published by Commit 6a)
       --proposal-ids  = `{{ selected_proposal_ids }}` joined by ','
                         (the output variable named in Phase 3b above)
     The task tool plans + implements, runs experiments to validate, and
     records results.

     **Tools**[__must__]:
     - task
     ```
   - Keep the previous typo fix ("advance to Phase 4" not "Phase 3").
   - The `__branch__` cardinality (one task per ID vs. one task with all IDs) is left to CI discretion until the broader audit fix plan addresses it.

> **Why this is the right design (v3.5 correction):** v3.4 of this plan added a Commit 6a that proposed a new `research_propose/executor.py` calling `host.set_session_variables(...)`. **That design was triple-broken** — verified in v3.5 against direct code evidence:
>
> 1. **The file would never run.** `research_propose` is a **bridge tool** (`tool.json:10` declares `"is_bridge": true`, `"derived_from": {"tool": "task", ...}`). All bridge tools dispatch via `derived_tool_execute` in `resources/tools/registry.py:66–117`, which hard-delegates to the parent's executor: `return await task_execute(task_args, ctx)`. There is no per-tool executor lookup for `is_bridge` tools — a new `research_propose/executor.py` would never be loaded.
> 2. **`self._host_inferencer` was fabricated.** The executor contract is `ToolExecutorCallable(Protocol).__call__(tool_name, arguments) -> ToolExecutionResult` (`conversational/protocols.py:27–30`). No host inferencer is passed. The same class of error as v3.2's `_update_phase_output()` and v3.3's "same as `workflow_target_path`".
> 3. **The proposed `return {"result":..., "context_updates":...}` would be silently dropped.** Consumer at `conversational_inferencer.py:1250` uses `hasattr(result, "context_updates")`. A dict's keys are not attributes, so `hasattr({}, "context_updates")` returns `False`. The actual return shape required is the `ToolExecutionResult` dataclass (`conversational/protocols.py:17`).
>
> **Crucially, the value v3.4's Commit 6a wanted to publish is already published.** `task_execute` returns `ToolExecutionResult(context_updates={"workspace_path": str(working_dir), "success": True, **artifacts})` (`task/executor.py:617–624`). The bridge dispatcher delegates to it. The consumer at `conversational_inferencer.py:1250–1251` merges `context_updates` into `prior_context` via `update_prior_context(**result.context_updates)`. The SOP body is rendered with `prior_context` as one of the Jinja feeds (`conversational_inferencer.py:1133–1146` — `build_feed(template_vars, self.prior_context, ...)`). So **`prior_context["workspace_path"]` is already populated after Phase 3, and `{{ workspace_path }}/outputs/proposals.json` would resolve today with zero new code.**
>
> v3.5's real concern is much smaller: **key collision.** `workspace_path` is a generic key emitted by every task-derived tool (research_propose, code_optimization sub-tasks, anything else that bridges to `task`). On the flat `prior_context` dict, **last-writer-wins**. If Phase 4 invokes `task` for implementation, its own `workspace_path` overwrites Phase 3's `research_propose` workspace before Phase 4's `--use-proposal` template can read it. The principled minimal fix is **Commit 6a (v3.5 redesign) below** — modify the bridge dispatcher to ALSO emit a tool-name-suffixed key alongside the generic one. This is collision-free, backward-compatible, lives in the one place that knows it's wrapping `task_execute`, and is ~3 lines of code.

### §E1.6a — Commit 6a: bridge dispatcher emits collision-free workspace key

**Purpose:** Make every bridge tool's workspace path discoverable in the SOP body Jinja scope via a key that cannot be overwritten by a later sibling bridge call. Lives in the bridge dispatcher itself (the only layer that knows the tool name AND that it's wrapping `task_execute`) — not in any per-tool executor (which for `is_bridge: true` tools would never run).

**Files modified in AF (single file, ~7 LoC of production change including import):**

`src/agent_foundation/resources/tools/registry.py` — `derived_tool_execute(...)` (lines 66–117).

**Step 1 — add the missing import** (verified absent from `registry.py:1–17` today — file only imports `ToolDefinition`):

```python
# Add to imports block at top of registry.py:
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
```

**Step 2 — wrap the dispatch return.** After the existing `return await task_execute(task_args, ctx)`, capture the result, **augment** its `context_updates` with a tool-name-suffixed key, and return:

```python
# v3.6 — collision-free key alongside the generic one.
# Keeps the existing "workspace_path" untouched (back-compat for any consumer
# that already reads it). Adds "workspace_path__{tool_name}" so multi-bridge
# SOPs (e.g. research_propose in Phase 3, then task in Phase 4) can each
# address their own workspace via Jinja {{ workspace_path__research_propose }}.
# Double-underscore separator avoids ambiguity with any user-supplied key.
result = await task_execute(task_args, ctx)
if isinstance(result, ToolExecutionResult) and result.context_updates:
    ws = result.context_updates.get("workspace_path")
    if ws:
        # Defensive: canonicalise hyphens to underscores to match Jinja
        # identifier rules. Today every caller already canonicalises tool
        # names via _resolve_tool_name (conversational_inferencer.py:1189,
        # 1260–1269), so tool_name arrives as "research_propose" not
        # "research-propose". But replace() is a one-token zero-cost guard
        # against any future caller (CLI, test, alternative dispatcher)
        # that skips canonicalisation — without it, a stray hyphen would
        # produce an invalid Jinja identifier and the SOP body would
        # silently render an empty path. Same class of silent failure
        # this plan has hit four times; this guard kills it at source.
        canonical_name = tool_name.replace("-", "_")
        result.context_updates[f"workspace_path__{canonical_name}"] = ws
return result
```

**Why the canonicalisation belt-and-braces is the right choice (not over-engineering):** verified at `conversational_inferencer.py:1189` that `canonical = self._resolve_tool_name(tool_call.name)` runs BEFORE `executor(canonical, ...)` (line 1220), and `_resolve_tool_name` at lines 1260–1269 maps `"research-propose"` → `"research_propose"` (the tool's `tool.name`). So **today the key always comes out as `workspace_path__research_propose`**, matching the SOP body. The `.replace("-", "_")` is a one-token guard that costs nothing and prevents the same class of silent-empty-render failure the plan has been bitten by four times. It is **not** an attempt to support hyphens in tool names — it's a hardening against future callers that might bypass `_resolve_tool_name`.

**Tests added** (`tests/.../tools/test_registry.py` — extend if exists, else new file):
- **T1 — happy path:** invoke `derived_tool_execute(tool_name="research_propose", ...)` for a synthetic `is_bridge` tool that wraps a stubbed `task_execute` returning `ToolExecutionResult(result="...", context_updates={"workspace_path": "/tmp/foo"})`. Assert returned `result.context_updates` contains BOTH `workspace_path == "/tmp/foo"` AND `workspace_path__research_propose == "/tmp/foo"`.
- **T2 — non-ToolExecutionResult:** invoke with a stubbed `task_execute` that returns a raw string (legacy back-compat). Assert the wrapper passes it through unchanged (no `AttributeError`, no augmentation attempted).
- **T3 — no workspace_path:** stub returns `ToolExecutionResult(result="ok", context_updates={"success": True})` — no `workspace_path` key. Assert the wrapper does NOT add a `workspace_path__research_propose` key (the `if ws:` guard fires).
- **T4 — hyphenated tool_name (defensive normalisation):** call `derived_tool_execute(tool_name="research-propose", ...)` directly. Assert the emitted key is `workspace_path__research_propose` (underscore form), NOT `workspace_path__research-propose`. Locks in the `.replace('-', '_')` guard.
- **T5 — sequential multi-bridge collision test:** dispatch `research_propose` then `task` in the same session; verify the corresponding `update_prior_context` calls (via inspected mock) write `workspace_path__research_propose` and (because `task` itself is not a bridge, only research_propose gets the suffix from this commit — the generic `workspace_path` collides as before; that's the explicit design choice documented in the §A6 v3.5 entry's "real concern" paragraph).

**Risk:** **very low.** Strictly additive — never mutates the generic `workspace_path` key. Every existing consumer that reads `workspace_path` is unaffected. The double-underscore suffix and `replace("-", "_")` together produce a Jinja-valid identifier in every case. **LoC:** ~7 production (5 logic + 2 import) + ~80 tests. **Depends on:** nothing.

**Async timing is safe (verified):** the v3.5 → v3.6 review traced the async-tool dispatch path at `conversational/conversational_inferencer.py:1217–1234`:
```
result = await executor(canonical, tool_call.arguments)          # line 1220 — await completion
if hasattr(result, "context_updates") and result.context_updates: # line 1221
    self.update_prior_context(**result.context_updates)           # line 1222 — write happens FIRST
... self._check_phase_completion(tool_name=canonical) ...         # line 1227 — phase advance NEXT
... self._inbox.put_nowait(ToolCompletion(...)) ...               # line 1234 — completion event LAST
```
`update_prior_context` runs before `_check_phase_completion`, and Phase 3 → Phase 3b transition is gated on the completion event from the inbox. So `prior_context["workspace_path__research_propose"]` is **guaranteed populated before any subsequent phase body is rendered**. Race-free by construction.

### §E1.6b — Commit 6b: SOP wiring (the original Commit 6, now demonstrably wireable)

This is the SOP-body edit described above. **It must land after Commit 6a** so `{{ workspace_path__research_propose }}` is collision-free at render time. All `{{ research_propose_workspace }}` references in Commit 6b (introduced in v3.3) are renamed to `{{ workspace_path__research_propose }}` to align with the bridge dispatcher's actual key.

2. (Optional, no behaviour change) `src/agent_foundation/resources/sops/code_optimization/SOP.md`
   - The line *"or `confirmation` if proposal_selection is unavailable"* stays as documentation of the graceful-degradation path. No edit required; cross-reference in commit message.

**Tests added:**
- `test/.../resources/sops/test_all_sops_lint_clean.py` (**NEW**) — defensive: iterates every SOP under `resources/sops/`, asserts the SOP linter passes. After Commits 1–5 + 6, `Tools[__must__]: proposal_selection` resolves cleanly via the `tool_to_phase_map`.

**Risk:** none — documentation. **LoC:** ~10 SOP + ~30 test.

### §E1.7 Commit 7 — RankEvolve adoption (separate repo, follow-up PR)

**Files modified in RankEvolve:**

1. `rankevolve/.../conversational/conversation_tools.py` — delete local `PROPOSAL_SELECTION` enum value (now from AF).
2. `rankevolve/.../conversational/protocols.py` — replace local `HubAwareToolExecutor` with `from agent_foundation.../protocols import HubAwareToolExecutor`; re-export for back-compat.
3. `rankevolve/.../conversational/handlers/proposal_selection.py` — **delete** (now in AF).
4. `rankevolve/.../conversational/handlers/__init__.py` — remove local registration; AF's `default_registry()` already registers it.
5. RankEvolve app init — call `register_proposal_parser(parse_proposals)` once.
6. RankEvolve React root — wrap relevant tree in `<ProposalOverridesEndpointContext.Provider value={{ endpoint: '/api/sessions/${id}/proposal_overrides', fetcher: rankevolveFetcher }}>`.
7. RankEvolve local `ProposalSelectionWidget.js` + `useProposalOverrides.js` — **delete**; consume from AF.
8. RankEvolve `WidgetRegistry.js` — remove local registration; AF's `registerBuiltins.js` covers it.

**Risk:** mechanical; rankevolve tests should pass unchanged (handler/widget contracts unchanged). **LoC delta:** ~−1500 net.

---

## §E2. Validation

### §E2.1 Per-commit gates

| Commit | Validation |
|---|---|
| 1 (D1) | New tests green; manual: run `pytest -k "extract_proposal_index"`; smoke: feed real anonymized 78KB aggregator output to a unit-tested helper, assert `proposals.json` appears |
| 2 (D2) | New tests green; manual: feed the real LLM constraint dict `{"type":"ordering","rule":"..."}`; assert it parses; assert `ProposalIndex.from_dict()` with 1 valid + 1 malformed constraint keeps 1 |
| 3 (D3 primitives) | `test_protocols.py` + `test_parsers.py` green; lint clean |
| 4 (D3 handler) | Handler tests green; `default_registry()` count test green; pytest of `conversational/` package green |
| 5 (D3 UI) | `npm test` in `react-shared` green; snapshot for widget matches; Storybook story renders if configured |
| 6 (SOP fixes) | `test_all_sops_lint_clean.py` green; manual: run SOP linter on `model_optimization/SOP.md`; confirm `proposal_selection` resolves in `tool_to_phase_map`; smoke: run SOP runtime with stub LLM returning canned proposals → confirm Phase 3b widget appears and selection propagates to Phase 4 |
| 7 (RankEvolve) | RankEvolve full test suite green; manual smoke: launch rankevolve session, complete proposal selection, confirm overrides API still wired |

### §E2.2 End-to-end smoke (after Commits 1–6 in AF)

**The real SOP CLI flags (verified at `src/agent_foundation/resources/tools/sop/cli.py:300–326`):**
- Positional: `sop_name` (e.g. `model_optimization`), optional `request`.
- Optional: `--request`, `--yolo`, `--model`, `--backend`, `--extra-sop-dirs`, `--extra-tool-dirs`.
- **There is NO `--until` or `--resume`**. The CLI runs an SOP end-to-end in a single invocation; pausing-mid-SOP and resuming are out-of-scope for the current CLI.

> **v3.3 correction:** v3.1/v3.2 introduced fictional `--until` and `--resume` flags in this section while "fixing" the CLI module path. The plan congratulated itself for that fix while creating a new error. v3.3 below uses only the real, verified flags.

**Single-command end-to-end smoke (interactive — pauses for proposal_selection input):**

```bash
python -m agent_foundation.resources.tools.sop model_optimization \
    "<your research goal>"

# → Runs Phase 0a (clarification) → Phase 0b (request capture) → Phase 1 → Phase 2 →
#    Phase 3 (research-propose, produces outputs/proposals.json under the
#             research_propose workspace — D1+D2 prove themselves here) →
#    Phase 3b (proposal_selection widget appears in the CLI; user enters IDs) →
#    Phase 4 (task --use-proposal ... --proposal-ids P1,P3)
#
# Acceptance:
#   <workspace_path__research_propose>/outputs/proposals.json exists           (D1+D2)
#   prior_context["workspace_path__research_propose"] == <ws>                  (Commit 6a)
#   phase_outputs["selected_proposal_ids"] == ["P1","P3"]                       (D3 + Commit 6)
#   task workspace contains _picked_proposals.json                              (F4)
#   task plan inlines only P1 and P3 (not all N)                                (F4 integration)
```

**CI-friendly yolo run (non-interactive, no pause):**
```bash
python -m agent_foundation.resources.tools.sop model_optimization \
    "<your research goal>" --yolo
```
Yolo mode auto-applies the `tool.json` `yolo_default` for `proposal_selection` (`mode: select_all`) so the smoke runs end-to-end without manual input — suitable for CI.

> **Pausing/resuming the SOP mid-flight:** if `--until` / `--resume` are needed for a richer dev loop, that is a separate SOP-CLI enhancement tracked in **§A5 follow-up #11**. For this plan, the single-command and yolo forms above are sufficient to validate the end-to-end pipeline.

### §E2.3 Regression net (permanent)

- `test_all_sops_lint_clean.py` — defends every SOP author from now on.
- `test_conversation_tool_registry_size` — asserts exactly 6 handlers (intentional code edit required to change).
- `test_proposal_constraint_aliases` — locks in the `type→kind` / `rule→reason` / `note→reason` / `from→proposal_ids` / `to→requires_ids` aliases so they don't silently regress.
- `test_bta_file_fallback` — locks in the truncated-response file-fallback path.

---

## §E3. Concrete execution checklist

```
[ ] Create branch:    git checkout -b feat/proposal_selection_e2e_pipeline

Commit 1 — D1 BTA file fallback + INVARIANT comment
[ ] Edit  breakdown_then_aggregate_inferencer.py _try_extract_proposal_index
[ ]   (a) Add file-fallback when "proposal_index" not in response text
[ ]   (b) Add INVARIANT comment above the sidecar-write block
[ ]       (NB: v3.2's _update_phase_output() call was REMOVED in v3.3 — the
[ ]        method doesn't exist on BTA inferencer; path-exposure now lives
[ ]        in Commit 6 via SOP output_variable + template substitution.)
[ ] NEW   test_breakdown_then_aggregate_inferencer.py (3 new tests)
[ ] NEW   test_parser.py (78KB fixture test)
[ ] NEW   test_bta_proposals_json_invariant.py (regression for path INVARIANT)
[ ] Tests + lint  → commit "fix(BTA): file-fallback for truncated proposal_index response + path INVARIANT comment"

Commit 2 — D2 constraint tolerance
[ ] Edit  proposal/model.py — ProposalConstraint.from_dict + ProposalIndex.from_dict
[ ] Edit  proposal/parser.py — add warning log on parse exception
[ ] Edit  test_model.py — 7 new constraint-tolerance tests (canonical, dialect α, dialect β scalar `to`, dialect β list `to`, dialect β `reason` not `note`, empty `{}`, ProposalIndex partial-failure)
[ ] Tests + lint  → commit "fix(proposal): tolerant constraint parsing with LLM-shape aliases"

Commit 3 — D3 primitives
[ ] Edit  conversation_tools.py — add PROPOSAL_SELECTION enum value
[ ] Edit  protocols.py — add HubAwareToolExecutor
[ ] NEW   data_models/proposal/parsers.py — Protocol + registry
[ ] Edit  data_models/proposal/__init__.py — re-exports
[ ] NEW   test_protocols.py, test_parsers.py
[ ] Tests + lint  → commit "feat(conversational): proposal_selection enum + Protocol + parser hook"

Commit 4 — D3 handler
[ ] NEW   handlers/proposal_selection.py (ported, 4 localised edits)
[ ] Edit  handlers/__init__.py — `reg.register(ProposalSelectionHandler())`
[ ]       (NB: register() takes ONE arg, derives tool_type from handler.tool_type
[ ]        per handler_registry.py:30-37. v3.2's two-arg call was wrong.)
[ ]       Also bump module docstring "5 generic handlers" → "6 generic handlers"
[ ] NEW   resources/tools/proposal_selection/tool.json
[ ] NEW   test_proposal_selection_handler.py (port + 3 new tests)
[ ] Tests + lint  → commit "feat(conversational): ProposalSelectionHandler"

Commit 5 — D3 UI
[ ] NEW   inputs/ProposalSelectionWidget.js
[ ] NEW   hooks/useProposalOverrides.js
[ ] NEW   contexts/ProposalOverridesEndpointContext.js
[ ] Edit  registerBuiltins.js
[ ] Tests (jest snapshot + no-op fetcher)  → commit "feat(ui): ProposalSelectionWidget + overrides context"

Commit 6a — bridge dispatcher emits collision-free workspace key (UNBLOCKS Commit 6b)
[ ] Edit  resources/tools/registry.py — add import (REQUIRED — file does not import it today):
[ ]         from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
[ ]             ToolExecutionResult,
[ ]         )
[ ] Edit  resources/tools/registry.py — derived_tool_execute(...) (lines 66-117):
[ ]       After `return await task_execute(task_args, ctx)`, capture the result
[ ]       and augment context_updates with `workspace_path__{tool_name}`:
[ ]         result = await task_execute(task_args, ctx)
[ ]         if isinstance(result, ToolExecutionResult) and result.context_updates:
[ ]             ws = result.context_updates.get("workspace_path")
[ ]             if ws:
[ ]                 canonical_name = tool_name.replace("-", "_")  # belt-and-braces
[ ]                 result.context_updates[f"workspace_path__{canonical_name}"] = ws
[ ]         return result
[ ]       Generic "workspace_path" stays untouched (back-compat).
[ ]       The .replace('-', '_') is a one-token guard against any future caller
[ ]       that skips canonicalisation (today _resolve_tool_name at
[ ]       conversational_inferencer.py:1189,1260-1269 already does this for the
[ ]       SOP dispatch path — verified).
[ ] NEW   test/.../tools/test_registry.py — augmentation tests (5 scenarios:
[ ]       T1 happy path, T2 non-ToolExecutionResult passthrough, T3 missing
[ ]       workspace_path skip, T4 hyphenated tool_name → underscore key,
[ ]       T5 sequential multi-bridge collision-free)
[ ] Tests + lint  → commit "feat(bridge): augment derived tool result with workspace_path__{tool_name}"

Commit 6b — SOP wiring (D3 wiring + F4 wiring)
[ ] Edit  sops/model_optimization/SOP.md — Phase 3b body:
[ ]       - revert the confirmation workaround (use proposal_selection)
[ ]       - pass proposals_path = `{{ workspace_path__research_propose }}/outputs/proposals.json`
[ ]       - declare output_variable = `selected_proposal_ids`
[ ] Edit  sops/model_optimization/SOP.md — Phase 4 body:
[ ]       - --use-proposal `{{ workspace_path__research_propose }}/outputs/proposals.json`
[ ]       - --proposal-ids `{{ selected_proposal_ids }}` joined by ','
[ ] NEW   test_all_sops_lint_clean.py — also assert that any `{{ X }}` placeholder
[ ]       in an SOP body is published by some tool's `context_updates_emitted` or
[ ]       `output_variable` declared in a tool.json reachable from that SOP. Catches
[ ]       the "dead substitution" failure mode that v3.3 had.
[ ] Tests + manual smoke (end-to-end §E2.2)  → commit "fix(sop): model_optimization Phase 3/3b/4 wiring with proposal_selection"

NB (stringification — verified at conversational_inferencer.py:1782):
  Conversation-tool output_vars values are persisted via `str(raw_value)`, so
  `selected_proposal_ids` arrives as a STRING, not a Python list. The
  ProposalSelectionWidget (Commit 5) must therefore serialise the selection
  as a comma-separated string (e.g. "P1,P3") — then Phase 4's
  `--proposal-ids {{ selected_proposal_ids }}` resolves cleanly because
  the `task` tool's `--proposal-ids` accepts a CSV string and splits it
  (verified at task/executor.py:142-215). Smoke test §E2.2 acceptance
  already checks this. No widget contract change beyond "return a CSV string".

[ ] git push origin feat/proposal_selection_e2e_pipeline
[ ] Open PR; reference this plan in PR description

Commit 7 (separate rankevolve PR) — adoption
[ ] Switch enum + Protocol imports to AF
[ ] Delete local handler + widget + hook
[ ] register_proposal_parser(parse_proposals) at startup
[ ] Wrap React tree in ProposalOverridesEndpointContext.Provider
[ ] rankevolve test suite green
[ ] git push; open PR; reference this plan
```

---

# PART II — DESIGN REFERENCE

> **Audience:** reviewers who need to understand *why* the executable steps in PART I are correct.
> **Read:** §D1–§D6 for the chosen design and its trade-offs.
> **Skip unless needed:** APPENDIX (audit evidence and rejected alternatives).

---

## §D1. Goals & non-goals

### §D1.1 Goals
1. **End-to-end pipeline restored:** `research-propose → outputs/proposals.json → Phase 3b proposal_selection → task --use-proposal` runs without silent failures or crashes.
2. AgentFoundation gains a working `proposal_selection` conversation tool — usable by **any** SOP, not just rankevolve's experiment flows.
3. `model_optimization/SOP.md` Phases 3b and 4 run end-to-end through the SOP runtime without falling back to `confirmation` and without losing the Phase 3b selection on the way to Phase 4.
4. RankEvolve continues to work unchanged (or with a minimal one-import switch) — no regression.
5. The migration is **side-by-side reviewable**: each commit is independently revertable, each layer testable in isolation.
6. All fixes degrade cleanly when used outside their happy path (truncated response → file fallback; malformed constraint → skip with warning; no Hub-aware executor → "Selected N proposals"; no parser registered → no enrichment; no overrides endpoint → empty overrides UI).

### §D1.2 Non-goals
1. Designing a new conversation-tool base class. We extend the existing one.
2. Replacing `multiple_choice` / `confirmation`. `proposal_selection` is **additive** — proposals are a richer payload (batch/phase structure, deprioritisation overrides, applied-change history) than a flat option list.
3. Migrating rankevolve's full experiment Hub. The Hub-aware execution path is **opt-in** via the `HubAwareToolExecutor` Protocol — AF apps without a Hub simply don't satisfy the Protocol; the handler degrades.
4. Migrating rankevolve's `/api/sessions/{id}/proposal_overrides` endpoint. AF stays unaware of overrides; the widget uses a no-op default unless a host app provides a React context override.
5. Enriching the `task` tool with new flags. It already supports `--use-proposal` + `--proposal-ids` fully.
6. Fixing the LLM prompt to emit canonical-schema constraints. Defence-in-depth at the parser level (D2 fix) is correct regardless; LLMs will always produce noisy schemas.

### §D1.3 Out of scope (tracked in §A5 follow-ups)
- The other 7 `model_optimization/SOP.md` audit gaps.
- The `tool_to_phase_map` one-tool-one-phase limitation.
- The four `error: refs/ai-working-log/diffs/...lock: badRefContent` lock files.

---

## §D2. Architecture decisions (chosen designs)

The four decisions below are the principles each executable step implements. Rejected alternatives and their reasoning are in §A3.

### §D2.1 D1 — File fallback for truncated LLM responses
After failing to find `proposal_index` in the response text, read the aggregator's output file from `_workspace.output_path(...)` and re-attempt extraction. Fast path (response) preserved; fallback uses the source-of-truth file that already exists on disk. Used by other AF inferencers for the same reason.

### §D2.2 D2 — Per-constraint tolerance with alias map
`from_dict()` uses `d.get()` with sensible defaults + alias map (`type→kind`, `from→proposal_ids`, `to→requires_ids`, `rule→reason`, `note→reason`) + scalar↔list normalisation. `ProposalIndex.from_dict()` wraps each constraint in try/except and logs a warning, keeping the rest. Per-item granularity. Backward compatible (canonical-schema dicts unchanged).

### §D2.3 D3 — Upstream rankevolve handler with three dependency inversions
Move handler to AF; replace 3 named couplings with pluggable hooks; rankevolve switches to importing from AF and registers its parser + Hub-aware executor at startup. Symmetric to how `effects.py` and the generic `Proposal` model were already upstreamed. Zero behaviour change for rankevolve; instant unlock for AF SOPs.

**The three dependency inversions:**

| # | RankEvolve coupling today | AF inversion |
|---|---|---|
| **I1** | `from rankevolve...protocols import HubAwareToolExecutor` | Copy Protocol verbatim into `agent_foundation/.../conversational/protocols.py`; re-export from rankevolve for back-compat. The Protocol uses only `typing.Protocol` + `runtime_checkable`. |
| **I2** | `from rankevolve...ui.proposal_parser import parse_proposals` (inside handler's `enrich_before_send`) | New registry in AF: `agent_foundation/common/data_models/proposal/parsers.py` with `register_proposal_parser(parser)` + `get_proposal_parser() -> ProposalParser | None`. Handler calls `get_proposal_parser()`; if `None`, skips enrichment. RankEvolve registers its parser at startup. |
| **I3** | `useProposalOverrides` hook hard-coded to `/api/sessions/{id}/proposal_overrides` | New React context `ProposalOverridesEndpointContext` with default returning `{ rankings: [], deprioritize: [], applied_changes_log: [] }`. RankEvolve provides a context value pointing to its endpoint. |

Plus **I4** (added in v3.1 to resolve handler-input wiring): `proposals_path` priority chain in `_resolve_proposals_source()` — `tool_args["proposals_path"]` → `phase_outputs["proposals_path"]` → rankevolve back-compat keys → registered parser. Detail in §E1.4.

### §D2.4 F4 — Phase 4 wiring via SOP body
SOP body instructs CI to read `phase_outputs` and pass them as flags: "Invoke `task` with `--use-proposal <Phase 3 proposals path from phase_outputs> --proposal-ids <Phase 3b selection from phase_outputs>`". Uses the existing CI capability; no SOP runtime change. Mirrors how `code_optimization/SOP.md` Phase 4 instructs Jira issue creation by reading `phase_outputs[Phase 3b]`.

---

## §D3. Runtime data flow — full end-to-end view (after all fixes)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 3: research-propose                                           │
│  ─────────────────────────                                           │
│  CI runs:  research-propose <goal> --docs <ref>                      │
│  Flow:     BTA (breakdown_then_aggregate) inferencer                 │
│  Aggregator writes:  <ws>/aggregator/outputs/output.md  (~78 KB,     │
│                       contains 36 KB `proposal_index` JSON fence)    │
│                                                                       │
│  _finalize_output() calls _try_extract_proposal_index(response):     │
│    1. text = str(response)                  ← may be truncated       │
│    2. NEW (Commit 1a): if "proposal_index" not in text:              │
│         text = read(<ws>/aggregator/outputs/output.md)               │
│    3. extract JSON fence → ProposalIndex.from_dict(...)              │
│         ← Commit 2 makes this tolerant of LLM constraint drift       │
│    4. write <ws>/outputs/proposals.json                              │
│       (INVARIANT — see Commit 1b comment; NOT moved to               │
│        final_deliverables because BTA._finalize_output               │
│        early-returns when aggregator is present at line 1100)        │
│                                                                       │
│  Path-exposure mechanism (Commit 6a, lives in bridge dispatcher):    │
│    task_execute returns ToolExecutionResult(context_updates={        │
│       "workspace_path": str(working_dir), "success": True, ...})     │
│       (verified task/executor.py:617–624, already in mainline)       │
│    derived_tool_execute (Commit 6a) augments the result with:        │
│       context_updates["workspace_path__research_propose"] = <ws>     │
│       (~5 LoC; back-compat — generic workspace_path also retained)   │
│    conversational_inferencer.py:1250–1251 detects                    │
│       hasattr(result, "context_updates") and calls                   │
│       self.update_prior_context(**result.context_updates)            │
│    → prior_context (flat dict):                                       │
│        workspace_path                = "<ws>"   (last-writer-wins,    │
│                                       overwritten if Phase 4 task)    │
│        workspace_path__research_propose = "<ws>" (COLLISION-FREE)     │
│    Phase 3b / Phase 4 reference it via SOP template substitution:    │
│        {{ workspace_path__research_propose }}/outputs/proposals.json  │
│    Jinja feed includes prior_context — verified                       │
│        build_feed(template_vars, self.prior_context, ...)            │
│        at conversational_inferencer.py:1133–1146.                     │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 3b: proposal_selection (NEW)                                  │
│  ──────────────────────────────────                                  │
│  SOP runtime sees Tools[__must__]: proposal_selection                │
│  ConversationalInferencer dispatches to PROPOSAL_SELECTION           │
│  HandlerRegistry → ProposalSelectionHandler.enrich_before_send()     │
│    ├── parser = get_proposal_parser()                                │
│    │     if registered:  enrichment via parser.parse(workspace)      │
│    │     else:           skip (graceful no-op)                       │
│    └── if isinstance(executor, HubAwareToolExecutor):                │
│            emit hub announcement                                     │
│        else:                                                          │
│            skip (graceful no-op)                                     │
│                                                                       │
│  Widget rendered: ProposalSelectionWidget (Commit 5)                 │
│    ├── useProposalOverrides() → empty defaults (no-op context)       │
│    │   OR rankevolve overrides (if rankevolve context provider)      │
│    └── User selects N proposals (e.g. "P1, P3")                      │
│                                                                       │
│  Handler.handle_response() → packages selection                      │
│  ApplyContextUpdates effect writes selection → inferencer.prior_context│
│    (verified: effects/apply_context_updates.py:29 writes prior_context,│
│     NOT phase_outputs — phase_outputs is the SOP-runtime bag, fed    │
│     separately when the conversation tool's output_variable is       │
│     declared in the SOP body — see Commit 6.)                        │
│  SOP runtime, on phase completion, reads the tool's output_variable  │
│  (`selected_proposal_ids`, named in Phase 3b body) and stores it     │
│  in phase_outputs as a flat key (SOPState.phase_outputs is a flat    │
│  dict per common/workflow/sop_state.py:23, NOT nested by phase).     │
│                                                                       │
│  phase_outputs:                                                       │
│    selected_proposal_ids = ["P1", "P3"]                              │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 4: Implementation, Experiment & Analysis                      │
│  ─────────────────────────────────────────────                       │
│  SOP runtime invokes task:                                           │
│    task \                                                            │
│      --use-proposal  <ws>/outputs/proposals.json \                   │
│      --proposal-ids  P1,P3                                           │
│                                                                       │
│  task.executor._resolve_proposal_plan():                             │
│    1. parse_proposal_file(<ws>/outputs/proposals.json)               │
│    2. idx.get_proposals_by_ids(["P1","P3"])                          │
│    3. inline full proposal detail into _proposal_plan.md (tempfile)  │
│    4. write _picked_proposals.json audit trail                       │
│    5. return tempfile path → fed as --initial-plan                   │
│                                                                       │
│  task planning + implementation phases consume the inline plan.      │
└──────────────────────────────────────────────────────────────────────┘
```

The six runtime branches introduced (`if "proposal_index" not in text` for D1, `try…except` per-constraint for D2, `if parser is None` for I2, `isinstance(executor, HubAwareToolExecutor)` for I1, the `tool_args["proposals_path"]` priority chain for I4, and the no-op overrides default for I3) are the ONLY runtime checks. Every one has an explicit no-op or fallback. No surprise behaviour.

---

## §D4. Risks & mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | D1 fix reads a partially-written aggregator file mid-write | Low | Low | Try/except on file I/O + the fence-presence check; if invalid, function returns cleanly (same as current behaviour). Aggregator writes atomically (`.tmp` + rename) — verify and document in commit message. |
| R2 | D2 alias map drifts as LLM schemas evolve | Medium | Low | v3's alias map (`type→kind`, `from→proposal_ids` with scalar→list normalisation, `to→requires_ids` with scalar→list normalisation, `rule→reason`, `note→reason`) is **derived from real production output**, not speculation — covers both observed dialects α and β at land time. Per-item try/except keeps the rest of the index intact when a new dialect γ appears; warning logs surface the new shape for the next iteration. Alias map is a Python expression — trivially extensible. |
| R3 | Widget styling drifts from AF UI conventions during port | Low | Medium | Style stack parity verified before porting. If drift appears, refactor at end of Commit 5 — no API change. |
| R4 | `proposal_overrides` API only exists in rankevolve | Certain | Low (by design) | Context default returns empty shape; widget renders proposals normally without overrides UI. |
| R5 | `HubAwareToolExecutor` ties handler to experiment orchestration | Low | Low | `isinstance` narrowed; falls back to `"Selected N proposals"` synthetic. |
| R6 | Rankevolve's parser produces a slightly different `ProposalSelectionData` shape than AF's `Proposal` | Possible | Medium | Commit 3 keeps `ProposalSelectionData` as the parser-return contract (re-exported from AF generic `Proposal` as a type alias). RankEvolve's parser unchanged. |
| R7 | Two SOPs reference `proposal_selection` and we want different presentation per SOP | Possible later | Low | Handler accepts `presentation_hints` from `tool_args` in the SOP. Defer concrete hints to a follow-up. |
| R8 | Tests in AF need fixture data living in rankevolve | Possible | Low | Inline a minimal fixture in `test/.../handlers/test_proposal_selection_handler.py` (~30 lines of JSON). |
| R9 | Previously-pushed `confirmation` workaround in `model_optimization/SOP.md` may have collected unrelated edits | Low | Low | Revert is localised to Phase 3b only; `git diff` review before commit. |
| R10 | RankEvolve adoption (Commit 7) ships in a different repo at a different time, causing a window where rankevolve still has its own copies | Low | Low | Plan keeps rankevolve working at every step (local copies coexist with AF's; `register_proposal_parser` is the only "switch-over" step). |
| R11 | F4 instruction depends on a discoverable `proposals.json` path at SOP-render time | **Re-resolved in v3.5** | — | v3.1's resolution (Commit 1b `_update_phase_output()`) used a non-existent method (bug B1). v3.4's resolution (new `research_propose/executor.py` calling `host.set_session_variables(...)`) used three fabricated mechanisms — bridge tools don't load per-tool executors, executor protocol has no host inferencer, and a plain dict return is dropped by the `hasattr(result, "context_updates")` check (verified bugs B1.4/B1.5/B1.6 in §A6 v3.5). v3.5's resolution is structurally sound and minimal: **`task_execute` already emits `context_updates={"workspace_path": str(working_dir), ...}` today** (verified `task/executor.py:617–624`), which the consumer merges into `prior_context` (`conversational_inferencer.py:1250–1251`), which the SOP body Jinja `build_feed(template_vars, self.prior_context, ...)` already includes (`conversational_inferencer.py:1133–1146`). Commit 6a adds 5 LoC to the bridge dispatcher to ALSO emit `workspace_path__{tool_name}` so multiple bridge tools don't collide on the generic key. See §E1.6a. |
| R12 | Selected `--proposal-ids` may include IDs not present in `proposals.json` | Low | Low | `task` executor already calls `idx.get_proposals_by_ids(ids)` which raises `KeyError`; executor catches and returns `_error(...)`. The SOP runtime should surface this error. No change needed. |

---

## §D5. Open questions & defaults

| # | Question | Default if not answered |
|---|---|---|
| Q1 | Should `proposal_selection` be SOP-level *first-party tool* (`/proposal_selection` prefix) or *conversational tool* invoked only by handler dispatch? | **Conversational** — matches `confirmation`, `multiple_choice`. SOPs name it via `Tools[__must__]: proposal_selection`. |
| Q2 | Should the no-op `ProposalOverridesEndpointContext` log `console.info(...)` to help diagnose missing rankevolve wiring? | Stay silent. AF apps that don't need overrides shouldn't see noise. |
| Q3 | Multi-phase `tool_to_phase_map` enhancement? | Defer; not blocking. |
| Q4 | Migrate `_group_selected_by_batch` / `format_hub_announcement` as public AF API? | Private to handler module; re-export from `agent_foundation.common.experiments.hub` if/when AF grows an experiment Hub. |
| Q5 | Keep `ProposalSelectionData` typename or rename to AF's generic `ProposalSelection`? | Keep for rankevolve binary-compat; add `ProposalSelection = ProposalSelectionData` alias. |
| Q6 | Should `model_optimization/SOP.md` also get a Jira-style human-gate (like code_optimization Phase 4) between selection and implementation? | Defer to model_optimization audit plan (§A5). |
| Q7 | Phase 4 `__branch__` cardinality: one task per ID, or one task with all IDs? | Defer to model_optimization audit plan; both work with current `--proposal-ids` syntax (`"P1"` vs `"P1,P3"`). |
| Q8 | How exactly does the `proposals.json` path reach Phase 3b/Phase 4? | **Re-resolved in v3.5.** Four-version evolution: v3.1 (BTA `_update_phase_output()` — method doesn't exist, B1); v3.3 (Phase 3 declares `output_variable` for research-propose — pattern only works for conversation tools, B5 misgeneralised); v3.4 (new `research_propose/executor.py` calling `host.set_session_variables(...)` — triple-broken: bridge tools route via `derived_tool_execute`, executor protocol has no host inferencer, plain dict return is dropped by `hasattr(result, "context_updates")`); **v3.5** (the simplest possible answer, requiring no new layer): `task_execute` already emits `context_updates["workspace_path"]` (verified `task/executor.py:617–624`); the bridge dispatcher already delegates to it; `conversational_inferencer.py:1250–1251` already merges into `prior_context`; `build_feed(template_vars, self.prior_context, ...)` at lines 1133–1146 already includes `prior_context` in the Jinja feed. **The only addition** (Commit 6a, 5 LoC) is to give multi-bridge SOPs collision-free keys by augmenting with `workspace_path__{tool_name}`. Phase 3b uses `{{ workspace_path__research_propose }}/outputs/proposals.json`; Phase 4's `--use-proposal` uses the same template. |

---

## §D6. File inventory diff (at-a-glance)

```
AgentFoundation (after this plan)
└── src/agent_foundation
    ├── common
    │   ├── data_models/proposal
    │   │   ├── model.py                              ✎ D2 fix (Commit 2)
    │   │   ├── parser.py                             ✎ warning logs (Commit 2)
    │   │   ├── parsers.py                            ★ NEW (Commit 3)
    │   │   └── __init__.py                           ✎ re-exports
    │   └── inferencers
    │       ├── agentic_inferencers/flow_inferencers
    │       │   └── breakdown_then_aggregate_inferencer.py  ✎ D1 fix (Commit 1, a+b+c)
    │       └── agentic_inferencers/conversational
    │           ├── conversation_tools.py             ✎ +1 enum (Commit 3)
    │           ├── protocols.py                      ✎ +HubAwareToolExecutor (Commit 3)
    │           ├── effects/                         (unchanged — package, not module:
    │           │     __init__.py, apply_context_updates.py,
    │           │     override_next_action_tool_args.py,
    │           │     set_prompt_variable.py, set_turn_variables.py)
    │           ├── handler_registry.py               (unchanged)
    │           └── handlers
    │               ├── __init__.py                   ✎ +1 line (Commit 4)
    │               └── proposal_selection.py         ★ NEW (Commit 4)
    ├── ui/react-shared/src
    │   ├── inputs/ProposalSelectionWidget.js         ★ NEW (Commit 5)
    │   ├── hooks/useProposalOverrides.js             ★ NEW (Commit 5)
    │   ├── contexts/ProposalOverridesEndpointContext.js  ★ NEW (Commit 5)
    │   └── protocol/registerBuiltins.js              ✎ +1 line (Commit 5)
    └── resources
        ├── tools/proposal_selection/tool.json        ★ NEW (Commit 4)
        └── sops/model_optimization/SOP.md            ✎ Phase 3b + 4 (Commit 6)

test/agent_foundation
├── common
│   ├── data_models/proposal
│   │   ├── test_model.py                                                       ✎ +7 tests (Commit 2)
│   │   ├── test_parser.py                                                      ✎ +1 fixture test (Commit 1)
│   │   └── test_parsers.py                                                     ★ NEW (Commit 3)
│   └── inferencers
│       ├── agentic_inferencers/flow_inferencers
│       │   └── test_breakdown_then_aggregate_inferencer.py                     ✎ +3 tests (Commit 1)
│       └── agentic_inferencers/conversational
│           ├── test_protocols.py                                               ★ NEW (Commit 3)
│           └── handlers/test_proposal_selection_handler.py                     ★ NEW (Commit 4)
└── resources/sops/test_all_sops_lint_clean.py                                  ★ NEW (Commit 6)
```

---

# APPENDIX — AUDIT TRAIL

> **Audience:** auditors verifying that every executable step in PART I and every design decision in PART II is anchored in real evidence.
> **Read:** any section when you need to trace a claim to its source.
> **Skip:** for routine execution; reference only.

---

## §A1. Motivation — why this plan exists

The user is trying to make `src/agent_foundation/resources/sops/model_optimization/SOP.md` run end-to-end. Investigation has uncovered **three connected defects** that together break the research → selection → implementation pipeline. Fixing only one of them is insufficient; fixing all three is also tractable. The plan covers all three in dependency order across **seven** commits (six in AF + one in rankevolve).

| # | Defect | Layer | Severity | Detection |
|---|---|---|---|---|
| **D1** | `proposals.json` is silently never written when the LLM response is truncated at the token limit | Flow inferencer | **Blocker** — silent failure | Verified by subagent: file `breakdown_then_aggregate_inferencer.py` lines 1118–1149 reads only `str(response)`, never the aggregator output file |
| **D2** | `ProposalConstraint.from_dict()` crashes with `KeyError` on LLM-shaped constraint dicts (`{"type":"ordering","rule":"…"}`); `ProposalIndex.from_dict()` propagates the crash, so the whole index parse dies | Data model | **Blocker** — hard crash | Verified by subagent: `model.py` lines 132–141 use direct `d["id"]`/`d["kind"]`; `ProposalIndex.from_dict()` lines 195–198 wraps constraints in a non-tolerant list comprehension |
| **D3** | No `proposal_selection` conversation tool exists in AgentFoundation. `model_optimization/SOP.md` Phase 3b currently has a workaround that downgrades to `confirmation`. `code_optimization/SOP.md` line 90 already names `proposal_selection`. RankEvolve has a full, well-factored implementation ready to upstream | Conversation tool + UI + SOP | **Major** — feature gap | Verified by subagents (see §A2) |

Plus one **SOP-authoring** fix that completes the chain:

| # | Fix | Layer |
|---|---|---|
| **F4** | `model_optimization/SOP.md` Phase 4 body must actually invoke `task --use-proposal <proposals.json> --proposal-ids <selection>`. The `task` tool already supports both flags fully — no tool work needed. | SOP body |

Once D1+D2+D3+F4 are landed, the full path is:

```
research-propose
  → emits outputs/proposals.json (D1+D2 ensure it actually appears)
  → Phase 3b: proposal_selection widget (D3 enables this)
  → selected IDs threaded into Phase 4
  → task --use-proposal …/outputs/proposals.json --proposal-ids "P1,P3" (F4 enables this)
  → task plans and implements
```

This is the correct, elegant, end-to-end solution. No hacks.

---

## §A2. Empirical baseline — every claim verified 2026-06-07

### §A2.1 D1 verified — `_try_extract_proposal_index` reads truncated response only

**File:** `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py`

| Evidence | Line(s) |
|---|---|
| Function definition `_try_extract_proposal_index(self, response) -> None` | 1118 |
| `text = str(response) if response is not None else ""` — sole input source | 1127 |
| Call site `_finalize_output()` | 1098 |
| Aggregator workspace IS accessible via `self._workspace.child("aggregator")` | 1954–1957 |
| Aggregator inferencer exposes `_workspace`, `output_path`, `deliverable_path` (used by the proposed fix) | confirmed |
| **No fallback logic** exists for truncated responses | (absence verified) |

**Manifestation:** LLM proposer produces a 78 KB aggregator output file containing a 36 KB `proposal_index` fence. The response object surfaced to the inferencer is ~10 KB (token-limit truncation). The fence is therefore present on disk but absent from `text`, and the function silently returns without writing `outputs/proposals.json`.

### §A2.2 D2 verified — `ProposalConstraint.from_dict()` crashes on LLM-shaped dicts

**File:** `src/agent_foundation/common/data_models/proposal/model.py`

| Evidence | Line(s) |
|---|---|
| `ProposalConstraint.from_dict()` uses `d["id"]` and `d["kind"]` (direct dict access — raises `KeyError`) | 134–135 |
| `ProposalIndex.from_dict()` builds `constraints` via a list comprehension with no try/except | 195–198 |
| Parser `parse_proposal_file()` catches outer exceptions and returns `None` (so one bad constraint kills the whole index) | parser.py 59–69 |
| Parser `_strategy_b()` (fence extraction) likewise returns `None` on any parse failure | parser.py 107–119 |
| Existing tests cover only the canonical schema; no malformed-constraint coverage | confirmed |

**Manifestation (verified against real LLM output in `_runtime/`):** LLMs emit constraints in at least two dialects, neither of which is the canonical `{"id":"C1","kind":"requires","proposal_ids":[...]}`:

- **Dialect α (free-form `rule`):** `{"type": "ordering", "rule": "P10 must precede every other proposal."}`
- **Dialect β (graph-edge with `from`/`to`):** `{"type": "requires", "from": "P5", "to": "P1", "note": "ORPO benefits most when stacked on Reason-first base."}` — and `to` may be **scalar OR list**: `{"type": "requires", "from": "P4", "to": ["P1","P3"]}` and `{"type":"requires","from":"P5","to":"P1"}` both occur in the same real file.
- **Aliased fields seen in production:** `type→kind`, `rule→reason`, `note→reason`, `from→proposal_ids`, `to→requires_ids`.

`from_dict()` raises `KeyError: 'id'` on any of them. `ProposalIndex.from_dict()` dies. `parse_proposal_file()` returns `None`. `task --use-proposal` fails with *"Failed to resolve proposals from: …"*.

### §A2.3 D3 verified — `proposal_selection` exists in RankEvolve, missing in AF

| Layer | RankEvolve path (present) | AgentFoundation status |
|---|---|---|
| Enum value | `rankevolve/.../conversational/conversation_tools.py` → `ConversationToolType.PROPOSAL_SELECTION = "proposal_selection"` | **Missing** (AF has 5 values: `clarification`, `single_choice`, `multiple_choice`, `confirmation`, `tool_argument_form`) |
| Handler | `rankevolve/.../conversational/handlers/proposal_selection.py` (280 LoC, framework-grade) | **Missing** |
| Handler registration | `rankevolve/.../conversational/handlers/__init__.py` `default_registry()` (one line) | **Missing** registration line |
| Capability Protocol | `rankevolve/.../conversational/protocols.py` `HubAwareToolExecutor` (`@runtime_checkable Protocol`, no rankevolve imports) | **Missing** (AF protocols.py has `ToolExecutorCallable`, `ContextCompressorCallable`, `PromptRenderer` only) |
| React widget | `rankevolve/src/webui/react/src/components/widgets/ProposalSelectionWidget.js` (1132 LoC) | **Missing** in AF `react-shared/src/inputs/` |
| Widget registry entry | `rankevolve/.../widgets/WidgetRegistry.js` (`'proposal_selection': ProposalSelectionWidget`) | **Missing** in AF `registerBuiltins.js` |
| Generic `Proposal` data model | AF already has `agent_foundation/common/data_models/proposal/model.py` (docstring: *"Inspired by RankEvolve's StructuredProposal/ProposalSelectionData but trimmed to framework-level generics"*) | **Already present** — perfect substrate |
| Effects (`ApplyContextUpdates` etc.) | AF already has them | **Already present** |
| Proposal parser hook | `rankevolve/.../ui/proposal_parser.py` `parse_proposals()` | **Missing** in AF (needs a tiny pluggable registry — see §D2.3 I2) |
| SOP-layer expectation | `code_optimization/SOP.md` line 90 already names `proposal_selection` as first-choice | (waiting for the implementation) |

### §A2.4 F4 verified — `task` already supports `--use-proposal` / `--proposal-ids` fully

| Layer | File | Line(s) | Status |
|---|---|---|---|
| Declared in `tool.json` | `resources/tools/task/tool.json` | params index #20–21 of 27 | `--use-proposal <path>` and `--proposal-ids <csv>` both declared |
| Implemented in executor | `resources/tools/task/executor.py` | 142–215 (`_resolve_proposal_plan`) | Parses index, filters by IDs, inlines full detail, writes audit trail `_picked_proposals.json` |
| Dispatch wired | `resources/tools/task/executor.py` | 680–687 | `--use-proposal` mutually-exclusive with `--initial-plan`, error-checked |
| Producer-consumer path | `_runtime/tasks/research_propose/research_propose_20260603_070411_53790258/outputs/proposals.json` | (real artifact) | Validates against `ProposalIndex.from_dict()` |
| Parser fallback strategy | `common/data_models/proposal/parser.py` | 95–104 | 3-tier: (A) `outputs/proposals.json` (B) markdown fence (C) priority table — handles both fast path and degradation |

**Conclusion for F4:** No task-tool work needed. The fix is purely an SOP body update telling the CI to invoke `task` with the two flags, threading Phase 3b's selection.

### §A2.5 What this collectively means

- D1+D2 are root-cause **production bugs** that, once fixed, allow `proposals.json` to actually exist. They are **prerequisites** for both D3 and F4.
- D3 is a **clean upstreaming** exercise (~90% of the rankevolve handler is framework-grade; three small dependency inversions named in §D2).
- F4 is a **one-edit SOP fix** (~10 LoC of Markdown) leveraging the existing task tool capability.

All four together form one coherent, end-to-end change. The integrated plan lands them in dependency order across **seven** commits (six in AF + one follow-up in rankevolve).

---

## §A3. Architecture: rejected options (and why)

For each chosen design in §D2, the alternatives considered and rejected.

### §A3.1 D1 alternatives (chosen: file fallback)

| Option | Description | Verdict |
|---|---|---|
| A. Raise the response token cap | Bigger model output cap | ❌ Doesn't fix the class of bug; will recur on bigger proposals |
| B. Stream the response and reassemble | Streamed accumulation in the inferencer | ❌ Out-of-scope and incompatible with current aggregator API |
| C. Re-prompt the LLM on truncation | Detect missing fence → ask LLM to re-emit | ❌ Adds an extra LLM round trip and a new failure mode |
| **D (chosen). File fallback** | Read aggregator output file when fence missing in response | ✅ Minimal, addresses root cause. Fast path preserved. |

### §A3.2 D2 alternatives (chosen: per-constraint tolerance)

| Option | Description | Verdict |
|---|---|---|
| A. Tighten the LLM prompt to force canonical schema | Better prompt examples | ❌ LLMs will always drift; this is the wrong layer |
| B. Pydantic-validate constraints, raise on any drift | Strict | ❌ Loses the entire `ProposalIndex` over one bad constraint — opposite of "proper, elegant, end-to-end" requirement |
| C. Drop the whole constraints field if any parse fails | Coarse-grained tolerance | ❌ Discards valid constraints alongside the malformed ones |
| **D (chosen). Per-constraint tolerance** | Tolerant `from_dict` + try/except per item with warning | ✅ Minimal. Per-item granularity. Preserves backward compat. |

### §A3.3 D3 alternatives (chosen: upstream with 3 inversions)

| Option | Description | Verdict |
|---|---|---|
| A. Keep `confirmation` workaround in `model_optimization/SOP.md` | Don't migrate, just patch the SOP | ❌ `code_optimization/SOP.md` already references `proposal_selection`; perpetuates documented "if available" branch forever. Confirmation has no concept of multi-item selection over a structured proposal list. |
| B. Copy-paste the handler verbatim into AF | Take rankevolve's file as-is | ❌ Drags `HubAwareToolExecutor`, `parse_proposals`, and rankevolve-specific phase_output keys as hard deps. AF would import-cycle or grow rankevolve coupling. |
| C. Reimplement from scratch in AF | Fresh `proposal_selection` | ❌ Wastes ~1100 LoC of working, debugged UI; risks behaviour drift; rankevolve also has to migrate. |
| **D (chosen). Upstream with three dependency inversions** | Move handler to AF; replace 3 named couplings with pluggable hooks | ✅ Clean. Symmetric to how `effects.py` and generic `Proposal` model were already upstreamed. Zero behaviour change for rankevolve; instant unlock for AF SOPs. |

### §A3.4 F4 alternatives (chosen: SOP body reads phase_outputs)

| Option | Description | Verdict |
|---|---|---|
| A. Hard-code `proposals.json` path in `model_optimization/SOP.md` | Path string baked into SOP | ❌ Tight coupling; breaks if `research_propose` workspace path scheme changes |
| B. Add a new SOP-level variable substitution mechanism | `${phase_outputs.Phase 3.proposals_path}` | ❌ Out-of-scope; would change SOP runtime semantics |
| **C (chosen). SOP body instructs CI to read phase_outputs and pass them as flags** | "Invoke `task` with `--use-proposal <Phase 3 proposals path>...`" | ✅ Uses the existing CI capability; no SOP runtime change. Mirrors `code_optimization/SOP.md` Phase 4. |

---

## §A4. Effort estimate

| Commit | Hours | Files touched | LoC delta (AF) |
|---|---|---|---|
| 1 — D1 BTA fallback | 0.5–1 | 2 | +160 |
| 2 — D2 constraint tolerance (v3 richer alias map) | 0.5–1 | 3 | +170 |
| 3 — D3 primitives | 0.5–1 | 4 | +120 |
| 4 — D3 handler | 1–2 | 4 | +350 |
| 5 — D3 UI | 2–3 | 5 | +1250 |
| 6 — SOP fixes | 0.25 | 3 | +40 |
| 7 — RankEvolve adoption (separate repo) | 1 | ~8 | −1500 (rankevolve) |
| **Total in AF** | **~5–8 h** | 21 | **+2030** |

High confidence; no ad-hoc paths required.

---

## §A5. Related follow-ups (not in this plan, tracked here for visibility)

The earlier audit of `model_optimization/SOP.md` identified **8 other gaps** besides Phase 3b/4. They are out of scope here, tracked in a separate `model_optimization_sop_audit_fix_plan.md`:

1. Phase 1 missing `[__requires user input__]` + `confirmation` gate before `understand-codebase`.
2. Phase 2 missing `[__requires user input__]` + `confirmation` gate before `understand-data`.
3. Phase 4 `__branch__` cardinality choice (one-per-ID vs. batched).
4. Phase 4b missing explicit `[__requires user input__]` directive.
5. No Jira integration — deliberate or gap? Needs product decision.
6. Phase 3 user-input ambiguous — text says "Break down the research goal" without instructing CI to compose+present the goal.
7. Tool syntax inconsistency: `research-propose <goal>` (no `/`) vs `/understand-codebase <path>` (with `/`).
8. `__afterwards__` in Phase 4b — parsed but unnecessary; align with `code_optimization`'s plain `__goto__ ... __if__ ...`.

Plus newly discovered:

9. The four `error: refs/ai-working-log/diffs/...lock: badRefContent` git lock files surfaced by `git fsck` during the `_docs/` recovery — cleanup follow-up.

10. **Tool-output declaration for non-conversation tools (v3.3 discovery).** Conversation tools have a documented `output_variable` mechanism (`conversational_inferencer.py:1402`) that surfaces a value into `phase_outputs`. Non-conversation tools (like `research-propose`) have **no equivalent** today. The principled enhancement is to add an `output_variable` (or richer `outputs:` list) field to `tool.json` and have the SOP runtime harvest it from the tool's return value at phase completion — symmetric to conversation tools. This plan currently works around the gap by declaring the workspace as the tool's output_variable in the SOP body (same shape as `workflow_target_path`); follow-up #10 is to formalise the mechanism so other SOPs can do the same without bespoke wiring.

11. **SOP CLI `--until` / `--resume` enhancement (v3.3 discovery).** The current `sop` CLI runs an SOP end-to-end in a single invocation (verified at `resources/tools/sop/cli.py:300–326`). Pausing at a named phase and resuming a previous session would make the developer loop much faster — particularly for debugging Phase 3b without re-running Phases 0–3. Not blocking this plan (the single-command and `--yolo` runs are sufficient for the end-to-end smoke).

---

## §A6. Changelog

- **v3.6 (2026-06-07 19:18) — Commit 6a code snippet hardened against two real defects, async timing verified safe:**

  External review of v3.5 traced the proposed `derived_tool_execute` snippet end-to-end and found three things — two real (fixed in v3.6) and one reassuring (the key-name canonicalisation chain actually works today, but a belt-and-braces guard makes it future-proof):

  | # | Finding | Verified | Action |
  |---|---|---|---|
  | **H1** | The v3.5 snippet `isinstance(result, ToolExecutionResult)` would `NameError` — `registry.py:1–17` only imports `ToolDefinition`, not `ToolExecutionResult` | ✅ Confirmed by `head -25 src/agent_foundation/resources/tools/registry.py` | Commit 6a now includes the required import (`from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import ToolExecutionResult`) as an explicit "Step 1" of the file edit, plus a checklist entry in §E3. |
  | **H2** | SOP body uses `research-propose` (hyphen, `SOP.md:78`) but the registry stores `tool.name = research_propose` (underscore). The match works today because `conversational_inferencer.py:1189` runs `canonical = self._resolve_tool_name(tool_call.name)` BEFORE calling the executor (verified `_resolve_tool_name:1260–1269` does `name.replace("-", "_") == tool.name`). So the key IS `workspace_path__research_propose` today — but any future caller (CLI, test, alternative dispatcher) that bypasses `_resolve_tool_name` would silently emit `workspace_path__research-propose`, an invalid Jinja identifier → blank `--use-proposal` → same silent-failure class the plan has hit four times | ✅ Confirmed | Commit 6a now includes a one-token defensive `canonical_name = tool_name.replace("-", "_")` before constructing the key. Costs nothing; closes the silent-failure mode at source. Test T4 locks it in. |
  | **H3** | Async timing — `research_propose` declares `"asynchronous": true`. If `Phase 3 → Phase 3b` advancement raced the `update_prior_context` write, the Jinja render would still see an empty key | ✅ Verified safe (no change needed, but documented) | Traced the async-tool dispatch path at `conversational_inferencer.py:1217–1234`: `result = await executor(...)` (line 1220, awaits completion) → `update_prior_context(**result.context_updates)` (line 1222) → `_check_phase_completion(...)` (line 1227) → `_inbox.put_nowait(ToolCompletion(...))` (line 1234). The write happens BEFORE the completion event, and phase advancement is gated on consuming the inbox event. So the key is guaranteed populated before any subsequent phase body is rendered — race-free by construction. Added explicit "Async timing is safe (verified)" subsection in §E1.6a with the line-numbered trace. |

  **Test plan extended:** §E1.6a "Tests added" grew from 3 scenarios to **5** (T1 happy path, T2 non-ToolExecutionResult passthrough, T3 missing `workspace_path` skip, T4 hyphenated `tool_name` → underscore key, T5 sequential multi-bridge). LoC accounting bumped from `~5 production + ~30 tests` to `~7 production (5 logic + 2 import) + ~80 tests`.

  **Meta-note 4 — discipline finally compounding:** for the first time in the plan's history (v3.1 → v3.5), the previous-version review found NO architectural errors — only an `import` and a defensive guard. That is the v3.5 meta-lesson actually applying. The fix path is now: trace each new mechanism end-to-end *before* adopting it (v3.5), then trace the proposed code's *individual statements* against the file you're editing *before* claiming it compiles (v3.6). v3.6 is the first version where I'm confident Commit 6a will execute cleanly on first run.

- **v3.5 (2026-06-07 18:52) — v3.4's Commit 6a was TRIPLE-BROKEN AND UNNECESSARY; replaced with 5-LoC bridge dispatcher augmentation:**

  **What v3.4 got wrong:** v3.4's Commit 6a proposed a new `src/agent_foundation/resources/tools/research_propose/executor.py` (~50 LoC) that would call `host.set_session_variables(...)` and return `{"result": ..., "context_updates": ...}`. **Every one of those three premises was fabricated:**

  | # | Fabricated premise | Verified reality | Source |
  |---|---|---|---|
  | **B1.4** | A new `research_propose/executor.py` would be invoked | `research_propose` declares `"is_bridge": true, "derived_from": {"tool": "task", ...}` in `tool.json:10–12`. All bridge tools dispatch via `derived_tool_execute` in `registry.py:66–117`, which hard-delegates to `task_execute(...)` — **no per-tool executor lookup for bridge tools**. A new `executor.py` in the bridge tool's directory is dead code. | `tool.json:10–12`, `registry.py:66–117` |
  | **B1.5** | Executors receive `self._host_inferencer` | `ToolExecutorCallable(Protocol).__call__(tool_name, arguments) -> ToolExecutionResult` (`conversational/protocols.py:27–30`). **No host inferencer is passed.** `self._host_inferencer` was conjured. Same class of error as v3.2's `_update_phase_output()`. | `conversational/protocols.py:27–30` |
  | **B1.6** | Returning a plain dict `{"result":..., "context_updates":...}` would be detected by the consumer | Consumer at `conversational_inferencer.py:1250` uses `hasattr(result, "context_updates")`. **`hasattr({}, "context_updates")` is `False`** — dict keys are not attributes. The update would be silently dropped. The actual required return type is the `ToolExecutionResult` dataclass at `conversational/protocols.py:17`. | `conversational_inferencer.py:1250` |

  **What v3.4 missed:** the value it wanted to publish is **already published by mainline code today**. The full chain (every step verified):
  1. `task_execute` returns `ToolExecutionResult(context_updates={"workspace_path": str(working_dir), "success": True, **artifacts})` — `task/executor.py:617–624`.
  2. Bridge dispatcher `derived_tool_execute` delegates to `task_execute` — `registry.py:115`.
  3. Consumer `_execute_tool_call` detects `result.context_updates` and calls `self.update_prior_context(**result.context_updates)` — `conversational_inferencer.py:1250–1251` (sync path) and `:1220–1221` (async path).
  4. SOP body Jinja render uses `build_feed(template_vars, self.prior_context, self.sop_state, {...})` — `conversational_inferencer.py:1133–1146`. So **`prior_context["workspace_path"]` is in the Jinja feed**, and `{{ workspace_path }}/outputs/proposals.json` would resolve today with zero new code.

  **The actual small problem v3.5 addresses:** `workspace_path` is a *generic* key emitted by every task-derived tool. On a flat `prior_context` dict, **last-writer-wins**. If Phase 4 invokes `task` for implementation, its own `workspace_path` overwrites Phase 3's `research_propose` workspace before Phase 4's `--use-proposal` template reads it. The principled minimal fix:

  | # | Where | Change |
  |---|---|---|
  | **G1** | §E1.6a — Commit 6a (REWRITTEN, ~5 LoC) | Modify `derived_tool_execute` in `resources/tools/registry.py` to, after `result = await task_execute(...)`, augment `result.context_updates` with `result.context_updates[f"workspace_path__{tool_name}"] = result.context_updates["workspace_path"]`. The generic key stays untouched (back-compat). Multi-bridge SOPs get a collision-free key. |
  | **G2** | §E1.6b SOP body, §D3 data-flow, §E2.2 acceptance, §E3 checklist | All `{{ research_propose_workspace }}` Jinja placeholders renamed to `{{ workspace_path__research_propose }}` (matches the actual bridge-dispatcher key). All references to "new `research_propose/executor.py`" deleted. |
  | **G3** | §D4 R11 + §D5 Q8 | Updated to "Re-resolved in v3.5" with full four-version evolution table and source-line evidence for the final answer. |
  | **G4** | §E2.2 acceptance test | Acceptance check changes from `phase_outputs["research_propose_workspace"] == <ws>` to `prior_context["workspace_path__research_propose"] == <ws>`. The `proposals.json` location-existence check uses the same key for the path prefix. |

  **Honest meta-note 3 — the recurring failure mode:** Across v3.1 → v3.2 → v3.3 → v3.4, the plan has fabricated the SAME class of mechanism FOUR times: `_update_phase_output()` (v3.1); `output_variable: research_propose_workspace` on a bridge tool (v3.3); a new `executor.py` for an `is_bridge` tool with a host-inferencer hook and a dict-as-result return (v3.4). Each round verified "the previous bug" but did not verify "the proposed solution". The structural fix is the same one v3.4's changelog already named — *trace each new mechanism end-to-end down to the dispatch site before adopting it* — but until v3.5, the plan kept writing that lesson down without applying it. v3.5 was forced to apply it because the feedback for v3.4 traced the proposed `executor.py` through `derived_tool_execute` and through the executor protocol and through the `hasattr` check, and showed each layer was wrong. **The actual answer turned out to be "do nothing on the production-emit side; just augment the bridge dispatcher with 5 lines."** Four versions of fabricated layers were replaced by reading the existing call chain end-to-end. The lesson for future iterations: **before designing a new mechanism, grep for the value you want to publish and trace its existing call sites — the codebase may already publish it.** Specifically: searching `grep -rn "workspace_path" src/agent_foundation/resources/tools/task/` would have found `executor.py:619` in v3.1, and the entire fabrication chain would never have started.

- **v3.4 (2026-06-07 18:25) — BLOCKER FIXED that v3.3 introduced while "fixing" v3.2:**

  **Root cause:** v3.3 asserted that adding `output_variable: research_propose_workspace` to `research_propose/tool.json` would make `{{ research_propose_workspace }}` Jinja-resolvable "the same way as `workflow_target_path`". **That equivalence is false** — `workflow_target_path` works because Phase 0a uses a *conversation tool* whose handler calls `set_session_variables(...)` (verified `conversational_inferencer.py:651–664`), which writes into BOTH `prior_context` AND `variable_manager` — and Jinja `Template(template).render(**feed)` (verified `flow_inferencers/multi_flow_inferencer.py:487`) reads `variable_manager` as part of `feed`. **`research-propose` is not a conversation tool**: it's a regular flow tool whose dispatch goes through `_execute_tool_call`'s `context_updates` merge (`conversational_inferencer.py:1250–1251`), which writes into `prior_context` only — and crucially, **`research_propose/` ships ONLY `tool.json` with NO executor** (verified by `ls`: 1 file, 4.6 KB). So no code path populates `variable_manager["research_propose_workspace"]` today, and Phase 3b's `{{ research_propose_workspace }}` would render as the empty string — `task --use-proposal /outputs/proposals.json` → fails to find file → end-to-end pipeline broken at the same seam as v3.2, just with the blocker renamed.

  **Fixes applied in v3.4:**

  | # | Where | Change |
  |---|---|---|
  | **F1 (blocker)** | §E1.6 → split into §E1.6a + §E1.6b | New Commit 6a: minimal `research_propose/executor.py` (~50 LoC) that wraps the existing flow dispatch, captures the workspace, and calls `host.set_session_variables({"research_propose_workspace": str(ws)})`. Mirrors the conversation-tool publication path that v3.3 wrongly claimed `research-propose` already had. |
  | **F2** | §E1.6b (renumbered Commit 6) | SOP body unchanged from v3.3, but now annotated "Requires Commit 6a — verified" and "NOT 'same as `workflow_target_path`' until Commit 6a lands." |
  | **F3** | §D2.3 I4 + §E1.4 priority chain | Dead Fallback A removed (`phase_outputs["proposals_path"]` was only populated by v3.2's bug-B1 method). Item 1 of the chain rewritten to point at the real source: `tool_args["proposals_path"]` populated by SOP-author Jinja `{{ research_propose_workspace }}/outputs/proposals.json`. |
  | **F4** | §D4 R11 | Updated from "Resolved in v3.1" → "Re-resolved in v3.4" with full explanation of why v3.1's and v3.3's resolutions both failed verification, and why v3.4's resolution (Commit 6a + 6b) actually does work. |
  | **F5** | §D5 Q8 | Same update as F4 — Q8 now traces the full v3.1 → v3.3 → v3.4 evolution of the path-discovery design with line-number evidence for the final answer. |
  | **F6** | §E3 checklist | Commit 6 split into Commit 6a (executor) + Commit 6b (SOP body). `test_all_sops_lint_clean.py` extended to assert that any `{{ X }}` placeholder in an SOP body is published by some reachable tool's `context_updates_emitted` or `output_variable` — would have caught v3.3's dead substitution at lint time. |
  | **F7** | §E2.2 acceptance + §E3 checklist | Added stringification NB documenting that conversation-tool `output_vars` values are persisted via `str(raw_value)` (verified `conversational_inferencer.py:1782`), so the `ProposalSelectionWidget` must serialise the selection as a CSV string — `task --proposal-ids` already accepts CSV input (verified `task/executor.py:142–215`), so this is a widget-contract note, not a new requirement. |

  **Honest meta-note 2:** v3.3's changelog congratulated itself for "all six bugs verified against source before applying" — but the fix for B5 (selected_proposal_ids output_variable) was incorrectly generalised to research-propose without checking whether the non-conversation-tool dispatch path actually populates `variable_manager`. The lesson: it's not enough to verify each bug fix in isolation; you must also verify that any *generalisation* across tool types holds (conversation tools and regular tools share concepts but not implementation). v3.4's verification used four parallel `grep`/`sed`/`ls` checks against `multi_flow_inferencer.py:487`, `conversational_inferencer.py:651–664/1250–1251/1782`, `sop_state.py:53–67`, and `ls research_propose/` before designing the fix, then designed Commit 6a around the smallest minimal-surface change that uses already-tested public APIs (`set_session_variables`).

- **v3.3 (2026-06-07 17:50) — SIX CORRECTNESS BUGS FIXED, all verified against source before applying:**

  | # | Bug | Where v3.2 was wrong | Evidence | v3.3 fix |
  |---|---|---|---|---|
  | **B1** (blocker) | Commit 1b called `self._update_phase_output(...)` | Method **does not exist** anywhere in the codebase | `grep -rn _update_phase_output src/` → 0 matches | Removed step 1b entirely; path-exposure moved to Commit 6 via SOP `output_variable` + template substitution (proven mechanism — same pattern as `workflow_target_path` in `code_optimization/SOP.md:10,17`). Step 1c renumbered to 1b (INVARIANT comment). |
  | **B2** (blocker) | `phase_outputs[Phase 3][proposals_path]` syntax in SOP body | `SOPState.phase_outputs` is a **flat** `field(default_factory=dict)` — NOT nested by phase | `common/workflow/sop_state.py:23` | All references switched to flat keys: `phase_outputs["research_propose_workspace"]`, `phase_outputs["selected_proposal_ids"]`. |
  | **B3** (blocker) | §E2.2 used `--until "Phase 3"` and `--resume <id>` flags | These flags **do not exist** | `resources/tools/sop/cli.py:300–326` — actual flags are `sop_name`, `request`, `--request`, `--yolo`, `--model`, `--backend`, `--extra-sop-dirs`, `--extra-tool-dirs` | Rewrote §E2.2 to use single-command end-to-end + `--yolo` for CI. `--until`/`--resume` filed as §A5 follow-up #11. |
  | **B4** (blocker) | Commit 4 used `registry.register(ConversationToolType.PROPOSAL_SELECTION, ProposalSelectionHandler())` (2 args, wrong variable name) | `register()` takes ONE arg; derives type from `handler.tool_type`. Existing code uses local variable `reg` not `registry` | `handler_registry.py:30–37` + `handlers/__init__.py:38–42` | Fixed to `reg.register(ProposalSelectionHandler())`; also documented the "5 generic handlers → 6" docstring bump. |
  | **B5** (significant) | Phase 3b SOP body never declared `output_variable` for `selected_proposal_ids` | Conversation-tool output capture requires the SOP body to **explicitly name** `output_variable` (same as `workflow_target_path` pattern) | `conversational_inferencer.py:1402` + `code_optimization/SOP.md:10` | Phase 3b body now explicitly names: *"The output variable MUST be named `selected_proposal_ids` for this conversation tool."* |
  | **B6** (accuracy) | §D3 diagram said *"ApplyContextUpdates effect writes selection → phase_outputs"* | Effect writes to `inferencer.prior_context`, NOT `phase_outputs` (a different bag) | `effects/apply_context_updates.py:29` | Diagram updated to show the real two-step flow: ApplyContextUpdates → prior_context; SOP runtime separately reads `output_variable` → phase_outputs on phase completion. |
  | **B7** (cosmetic) | §D6 listed `effects.py (unchanged)` | `effects` is a **package** (`effects/__init__.py` + 4 sibling modules), not a single file | `ls src/.../conversational/effects*` | Updated to `effects/ (package — __init__.py, apply_context_updates.py, override_next_action_tool_args.py, set_prompt_variable.py, set_turn_variables.py)`. |

  **Plus two new §A5 follow-ups surfaced during verification:** #10 (formalise non-conversation-tool `output_variable` mechanism) and #11 (SOP CLI `--until`/`--resume` enhancement).

  **Honest meta-note:** v3.1 introduced bugs B1, B2, B3 while v3.1's changelog congratulated itself for "closing seven verified gaps." v3.2 inherited them under the "content unchanged" banner. The lesson: claims of verification need to be re-validated each pass — verification can drift. v3.3's verification used `grep -rn` and `sed` against real files for every claim before applying any fix.

- **v3.2 (2026-06-07 17:37):** Structural reorganisation only — content unchanged from v3.1. The file is now split into three clearly-labelled tiers: PART I — EXECUTION (§E0–§E3), PART II — DESIGN REFERENCE (§D1–§D6), APPENDIX — AUDIT TRAIL (§A1–§A6). No content added or removed; sections moved, not edited. **NB:** v3.2 inherited bugs B1–B7 unfixed; see v3.3 above. Section-number mapping from v3.1:
  - v3.1 §0 → v3.2 §A1 (motivation)
  - v3.1 §1 → v3.2 §A2 (empirical baseline)
  - v3.1 §2 → v3.2 §D1 (goals & non-goals)
  - v3.1 §3 → v3.2 split: chosen designs → §D2; rejected options → §A3
  - v3.1 §4 → v3.2 §E1 (migration commits 1–7)
  - v3.1 §5 → v3.2 §D3 (data flow)
  - v3.1 §6 → v3.2 §D4 (risks)
  - v3.1 §7 → v3.2 §E2 (validation)
  - v3.1 §8 → v3.2 §A4 (effort)
  - v3.1 §9 → v3.2 §D5 (open questions)
  - v3.1 §10 → v3.2 §A5 (follow-ups)
  - v3.1 §11 → v3.2 §E3 (execution checklist)
  - v3.1 §12 → v3.2 §D6 (file inventory)
  - v3.1 §13 → v3.2 §A6 (changelog — this section)
- **v3.1 (2026-06-07 17:30):** Closed seven verified gaps surfaced by a second-pass review (all verified against direct source-code evidence before applying):
  - **§4.1 Commit 1 grew from 1 sub-step to 3 (a/b/c):** (a) text fallback unchanged; **(b) NEW — `_update_phase_output("proposals_path", str(sidecar))`** after successful write — resolves the R11 wiring gap (downstream Phase 3b/Phase 4 can now discover `proposals.json`'s path without filesystem guessing); (c) NEW invariant comment documenting that `proposals.json` survives `_finalize_output`'s early-return when aggregator is present (verified at lines 1094–1101).
  - **§4.4 Commit 4 grew from 3 to 4 localised edits:** added **I4 — `proposals_path` priority chain** in the handler, encapsulated in a `_resolve_proposals_source()` helper. Resolution order: `tool_args["proposals_path"]` → `phase_outputs["proposals_path"]` → rankevolve back-compat keys → registered parser. Zero behaviour change for rankevolve; clean discovery for AF-only consumers.
  - **§5 data-flow diagram updated** to show Commit 1b producer + Commit 4 I4 consumer chain explicitly. Notes the INVARIANT from Commit 1c.
  - **§7.2 smoke-test commands fixed** — replaced non-existent `agent_foundation.cli.sop_tool` with the real entrypoint `agent_foundation.resources.tools.sop` (verified at `src/.../sop/__main__.py` + `cli.py`). Added a yolo-mode variant for CI.
  - **§0 cosmetic:** "Plan v2 covers all three" → "Plan v3.1 covers all three … across **seven** commits".
  - **§1.5 cosmetic:** "five commits" → "**seven** commits (six in AF + one follow-up in rankevolve)".
  - **§11 cosmetic:** Commit 2 checklist "4 new constraint-tolerance tests" → "7 new constraint-tolerance tests (canonical, dialect α, dialect β scalar `to`, dialect β list `to`, dialect β `reason` not `note`, empty `{}`, ProposalIndex partial-failure)" — matches the §4.2 body.
  - **Reviewer banner** refreshed to point at v3.1 and to enumerate which sections moved.
- **v3 (2026-06-07 17:00):** D2 alias map enriched to cover real LLM dialect β (`from→proposal_ids`, `to→requires_ids`, `note→reason`, scalar↔list normalisation), all verified against `_runtime/tasks/research_propose/.../aggregator/.../output.md`. §4.4 Commit 4 now spells out the canonical `proposal_selection/tool.json` shape modelled on `multiple_choice/tool.json` (verified by `ls` against all 4 existing conversation tools). R2 risk language updated. §8 effort for Commit 2 bumped from +110 to +170 LoC.
- **v2 (2026-06-07 16:26):** Added D1 (BTA file fallback), D2 (constraint tolerance), F4 (Phase 4 `--use-proposal` wiring), `tool.json`. Reorganised v1's 5 commits into 7 (D1 → D2 → primitives → handler → UI → SOP → rankevolve).
- **v1 (2026-06-07 15:34):** Initial draft. Scoped only D3 (conversation-tool migration). Missed D1, D2, F4.

---

**End of plan v3.6.** Ready for review.
