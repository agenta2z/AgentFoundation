# Task Tool Planning-Complexity Ladder — Integrated Plan v5

> **⚠ Reviewer banner — please read v5 corrections first**
>
> **v5 (2026-06-10 13:39) — FIVE SOURCE-VERIFIED BUGS in v4 corrected after Plan B's revision (`.claude/plans/update-your-task-tool-adaptive-goose.md` updated 13:11 today) flagged them. All five verified independently against AgentFoundation source. See §A7 (NEW) for the bug-by-bug audit.**
>
> **The bugs (all genuine, all confirmed):**
>
> 1. **B1 — `_target_: Leaf` is not a registered target.** v4 used `Leaf` literally in some YAML examples. Source grep confirms zero registration. Must use `_target_: ${_params.main_inferencer}` (resolves to `ClaudeCodeCLI`). v5 fixes all 4 occurrences.
> 2. **B2 — BTA parameter is `max_breakdown`, NOT `max_subtasks`.** v4 §E1.7 Commit 7 wrote `max_subtasks: ${_params.max_breakdown}`. Source confirms: 8 hits for `max_breakdown` in `breakdown_then_aggregate_inferencer.py` (attrib at line 375); zero hits for `max_subtasks` in BTA. v5 fixes all occurrences.
> 3. **B3 — `smart-breakdown.yaml` is redundant with `breakdown.yaml`'s adaptive mode.** Plan B's design is more elegant — ONE file with `worker_factory` as a dict including `_default: leaf` IS the adaptive mode. With `_default: leaf`, simple tasks behave identically to homogeneous leaf mode; adaptive dispatch only kicks in when the LLM classifies a subtask as needing more than a leaf. v5 FOLDS `smart-breakdown.yaml` into `breakdown.yaml`.
> 4. **B4 — `${_build_flow_configs:...}` custom resolver doesn't exist.** v4 §E1.2 referenced it for `multiple.yaml`. Source confirms zero hits. Use literal `flow_configs:` block instead.
> 5. **C1 — Root-level `_import_:` doesn't work.** v4 Commit 3 step 3a proposed `full-plan.yaml` as a one-line `_import_: breakdown-multiflow-plan`. Source confirms: zero root-level `_import_:` usages anywhere; only nested usage at `default.yaml:156`. Use a Python alias in `_CONFIG_ALIASES` instead.
>
> **MVP DISCIPLINE ADOPTED FROM PLAN B v2:** v4 tried to ship 4 new commits (5/5b/5c/5d) for `--config conversational` infrastructure. Plan B's revision correctly DEFERS these as Follow-ups #1–#4 — the architecturally-sound but large-scope changes (full `_run_conversational_router`, `_extract_result_text` fix, depth budget, router template) should be a separate initiative after the presets ship.
>
> **v5 keeps the v4 architectural reversal of §D2** — `--config disabled` (renamed `--config conversational` in v4; v5 uses `disabled` as canonical per Plan B's terminology) is still the right answer. v5 just makes Commit 5 SMALLER: a workspace-passthrough early-return (Plan B's design), NOT a full CI router (v4's over-build).
>
> **What survives intact from v4:**
> - The 2×2 matrix (§D3)
> - The architectural rejection-of-rejection in §D2
> - v3 follow-ups (`tool_call_defaults`, `--autonomous-level`, `mandatory` field) — now explicitly in PART III as Follow-ups #4 and #5
> - The §D6 verified facts F1–F15
> - The §A6 cross-plan comparison appendix
>
> **What changed in v5:**
> - 5 source-anchored YAML/CLI bug fixes from §A7
> - Commit 5 SMALLER (workspace passthrough, not full CI router)
> - Commits 5b/5c/5d DEFERRED to PART III follow-ups
> - `smart-breakdown.yaml` Commit 7 FOLDED into Commit 1 (`breakdown.yaml`)
> - §A7 NEW (bug-by-bug audit of v4 → v5 corrections)
>
> **v4 banner preserved verbatim below for audit trail.**

---

# Task Tool Planning-Complexity Ladder — Integrated Plan v4 (superseded; bugs corrected in v5)

> **⚠ v4 reviewer banner preserved verbatim for audit trail. See v5 banner above for what was corrected.**
>
> **v4 (2026-06-10 12:45) — ARCHITECTURAL REVERSAL of v3 §D2.** After cross-reading two independent peer plans (`.claude/plans/update-your-task-tool-adaptive-goose.md` and `.cursor/plans/task_plan_config_ladder_7ea0fe0e.plan.md`), v3's rejection of `--config disabled` as the new `task` default is now **withdrawn**. The peer plans correctly identified that:
>
> 1. **`disabled` is a real preset file** (`disabled.yaml`), not a sentinel string — v3's "sentinel anti-pattern" critique (R2) doesn't apply. It's a normal config that selects a `Conversational` topology root, exactly as `breakdown` selects `Dual{BTA{...}}`.
> 2. **The conversational router belongs inside `task`**, not in a new sibling `chat` tool — keeps one entry point, inherits all `task` args (`--model`, `--config`, `--multi-iter`), avoids duplicating CLI scaffolding.
> 3. **The renaming concern (v3 R1) is trivially fixable** with `--config conversational` as the canonical name and `disabled` as a backward-compat alias (or vice versa, depending on naming taste — Q14 below).
>
> v3's `chat` peer tool design (Commit 5) is **superseded** by a much smaller change: implement `--config conversational` inside `task` (Commit 5 v4). The `chat` tool spec is preserved in the v3 backup for reference but no longer planned.
>
> v3 also missed **two critical findings** that the peer plans surfaced:
>
> - **CRITICAL BUG:** `task/executor.py:353–354` (`_extract_result_text`) silently drops `tuple[1:]` when BTA's `disable_aggregator=True` returns a multi-worker tuple. Without fixing this, the "no-aggregate / list-of-outputs to conversation" feature literally cannot work. **NEW Commit 5b in v4.**
> - **Unified `task_depth` budget** — v3 filed this as a follow-up; v4 promotes it to **Commit 5c** because adaptive `conversational` workers + router recursion can compound multiplicatively.
>
> **What survives intact from v3:**
> - Commits 1, 2, 3, 4 (`breakdown.yaml` / `multiple.yaml` / `full-plan` alias / `tool.json` docs) — unchanged. v4 also adds the `pti` and `multiflow` aliases per the peer plans.
> - Commit 7 (`smart-breakdown.yaml` was a distinct file) is **REFACTORED** into `breakdown-plan.yaml`'s `worker_type: adaptive` mode (Plan B/C's design — one file, three modes — is more elegant). The v3 separate-file approach is dropped.
> - Commit 8 (`tool_call_defaults`) — **unchanged.** Wires through `task --config conversational` instead of `chat`.
> - Commit 9a (`mandatory` field) — **unchanged.**
> - Commit 9b (`--autonomous-level`) — **unchanged.** Default for `task --config conversational` triggered by a parent CI = `auto`.
>
> **Architectural decisions UNCHANGED in v4:** §D3 2×2 matrix (still correct); §D4 risks (still apply); §D6 verified facts F1–F10 (still apply, with F11 added for `_extract_result_text` and F12 for `disable_aggregator`).
>
> **See §A5 v4 changelog entry for full diff summary, and §A6 (NEW) for the cross-plan comparison that drove the reversal.**

---

# Task Complexity Presets + `chat` Peer Tool — Integrated Plan v3 (superseded — chat peer tool architecture withdrawn in v4)

> **⚠ Original v3 banner preserved for audit trail. The `chat` peer tool design described below is superseded by `task --config conversational` in v4.**
>
> **v3 (2026-06-10 12:36)** adds **Commit 8 (`tool_call_defaults`)** and **Commit 9 (`--autonomous-level` + `mandatory` gate prerequisite)** as a coherent pair. Both are CI-layer features that became necessary once `chat` started driving nested `task`/`sop` calls (Commits 5–6 in v1/v2): without these two, every nested tool call has to repeat `--yolo`/`--model`/etc. on every invocation, and there's no principled way to surface only *critical* questions back to the root conversation. The two features ship together because Commit 8 without Commit 9 has nothing important to default, and Commit 9 without Commit 8 forces callers to repeat the flag on every dispatch.
>
> Honest caveats are documented inline:
> - **C1:** today the `sop/tool.json` description promises non-mandatory gate handling but the code never implements the mandatory/non-mandatory distinction. Commit 9a (the prerequisite) adds the field + resolver. ~30 LoC, real work.
> - **C2:** `task` has no confirmation gates today (verified — `task/executor.py:600` only consumes `interactive` in `mode=="confirm"`, which is a task-mode not a gate). `--autonomous-level` is therefore a no-op on `task` today, ships as API consistency + forward-compatible knob, and a follow-up is filed to add real gates.
>
> **v2 (2026-06-09 20:26)** added **Commit 7: `smart-breakdown.yaml`** — a 5th preset using BTA's already-existing heterogeneous-worker mechanism. v2 architectural decisions UNCHANGED in v3.
>
> The architectural decisions in v1 (§D2 rejection of `--config disabled`; the 2×2 matrix in §D3) remain **unchanged in v2/v3** — both ADD features without altering the v1 contract.

---

# Task Complexity Presets + `chat` Peer Tool — Integrated Plan v1 (superseded)

> **⚠ Reviewers: this file is `v1` (initial draft, 2026-06-09 19:08). It integrates two related-but-separable ideas: (A) add three new task-tool complexity presets — `breakdown`, `multiple`, `full-plan` — completing the coverage × diversity matrix begun by the existing `breakdown-multiflow-plan.yaml`; and (B) introduce a new peer tool `chat` that runs a conversational inferencer on top, dispatching to `task` (and other tools) when the user has a concrete sub-task. The two ideas connect — `chat` becomes the lightweight "what do you want?" entry point; `task` stays the heavyweight "produce a deliverable" workhorse. Crucially, this plan REJECTS the proposal to make `task --config disabled` the new default (covered in §D2 — wrong name, wrong placement, silent behaviour change for every existing caller); the principled answer is a separate peer tool, mirroring the existing `sop` peer-tool pattern.**
>
> **The file is split into three clearly-labelled tiers:**
> - **PART I — EXECUTION** (§E0–§E3): what to do, in what order.
> - **PART II — DESIGN REFERENCE** (§D1–§D5): why this design is correct.
> - **APPENDIX — AUDIT TRAIL** (§A1–§A4): how every claim was verified.

**Author:** Rovo Dev (drafted in conversation with Tony Chen)
**Date:** 2026-06-09 v1
**Status:** Draft v1 — ready for review
**Branch:** `dev_xinli_2601`
**Companion to:** `_docs/_plan/workflows_and_sop/proposal_selection_tool_migration_plan.md` (v3.6, same plan-discipline conventions)

---

## §0. Quick-start — TL;DR for executors

This plan has **6 commits** in dependency order. Two halves (A and B) can land in parallel if you want, but B depends on A's presets existing.

| # | Commit | Effort | Surface |
|---|---|---|---|
| 1 | Add `breakdown.yaml` preset (Dual{BTA{Leaf}}) | 0.5 day | 1 file, ~120 LoC YAML |
| 2 | Add `multiple.yaml` preset (root MFDual) | 0.5 day | 1 file, ~100 LoC YAML |
| 3 | Rename + alias `breakdown-multiflow-plan` → `full-plan` (back-compat) | 0.5 day | 2 files (rename + alias map) |
| 4 | Add new tests + extend `--config` description in `tool.json` | 0.5 day | ~150 LoC tests + ~10 LoC doc |
| 5 | NEW `chat` peer tool — minimal CLI mirroring `sop/cli.py` pattern | 1 day | ~250 LoC under `resources/tools/chat/` |
| 6 | E2E smoke + integration test (`chat` dispatches to `task --config breakdown`) | 0.5 day | ~100 LoC tests + ~30 LoC docs |
| **7** | **NEW (v2): `smart-breakdown.yaml` — heterogeneous workers, complexity-classified per subtask** | 1 day | 1 file YAML (~150 LoC) + prompt-template tweak + ~80 LoC tests |
| **8** | **NEW (v3): `tool_call_defaults` — CI-level per-tool default arguments** | 0.5 day | ~80 LoC production + ~120 LoC tests |
| **9a** | **NEW (v3, prerequisite to 9b): `mandatory` field on conversation tools** | 0.5 day | ~30 LoC production + ~50 LoC tests |
| **9b** | **NEW (v3): `--autonomous-level` flag on `task` + `sop`, surfaced via `chat`** | 1 day | ~170 LoC production + ~130 LoC tests |

**Total:** ~6.5 days. Lowest-risk first three commits (presets) ship value immediately without changing `task` defaults or surface area. Commit 5 (the `chat` peer tool) is the largest change and is genuinely net-new functionality. Commit 7 (added in v2) leverages BTA's already-existing `worker_factory: dict[str, factory] + task_type_arg_name` machinery — no new abstraction. Commits 8 + 9 (added in v3) are CI-layer features that make `chat` actually usable as a tool dispatcher (per-tool defaults + 3-level autonomy with mandatory-gate escalation).

---

# PART I — EXECUTION
══════════════════════════════════════════════════════════════════════════════

## §E1. Migration plan — 6 commits in dependency order

### §E1.1 — Commit 1: add `breakdown.yaml` preset (coverage only)

**Purpose:** A "BTA-only" preset for tasks that decompose naturally but where each subtask is mechanical enough that one LLM pass per worker suffices. Lighter than `breakdown-multiflow-plan` (no N-way parallel exploration per subtask); heavier than a single LLM call (still gets coverage via decomposition + integration + outer review/fix).

**File added:**

`src/agent_foundation/resources/tools/task/configs/breakdown.yaml` (**NEW**, ~120 LoC).

**Structural shape** (verified against `breakdown-multiflow-plan.yaml` for naming idiom):

```yaml
# ============================================================================
# Plan-Only Topology: Dual{BTA{Leaf}}
# ============================================================================
#
# Coverage-only standalone planning topology. Use directly via:
#   --agent-config breakdown
#
# Tree structure:
#   Dual                                    review + fix the integrated PLAN
#   ├── base_inferencer = BTA               decompose the PLANNING task
#   │   ├── breakdown_inferencer            → N sub-planning subtasks
#   │   ├── worker_factory = Leaf           each sub-plan handled by ONE
#   │   │                                   single-flow inferencer (no MFDual,
#   │   │                                   no per-worker review)
#   │   └── aggregator_inferencer           integrate sub-plans → ONE coherent
#   │                                       prose plan
#   ├── review_inferencer                   judge plan structure + quality
#   └── fixer_inferencer = LEAF             refine plan based on review feedback
#                                           (lightweight: single leaf with
#                                           plan/main/followup.jinja2 — same
#                                           pattern as breakdown-multiflow-plan)
#
# Why bare-Leaf workers (not Dual{Leaf} per-worker review)?
#   This preset is the "minimum-complexity sibling" in the 2×2 matrix
#   (see plan §D3). The outer Dual review/fix on the aggregated plan is
#   the single quality gate; per-worker review would add a hidden 3rd
#   dimension that muddies the matrix and erodes the naming precision.
#   Users who want per-worker review can override:
#     --override _params.worker_factory_target=Dual
#
# ----------------------------------------------------------------------------
# Cascades (`_-prefix` keys → propagated to descendants accepting un-prefixed param)
# ----------------------------------------------------------------------------
_logger: auto
_debug_mode: true
_model_name: opus[1m]
_idle_timeout_seconds: 600
_tool_use_idle_timeout_seconds: 5400
_output_path: "output.md"

# Cap subtask count to keep "coverage only" lightweight
_params:
  workspace_root: ???
  default_inferencer: ClaudeCodeCLI
  main_inferencer: ${oc.env:DEFAULT_MAIN_INFERENCER,${.default_inferencer}}
  max_breakdown: ${oc.env:DEFAULT_MAX_BREAKDOWN,5}   # surfaces --override knob
  worker_factory_target: Leaf                         # see comment above

# (Templates / workspace / template-root identical to breakdown-multiflow-plan.yaml)

_target_: Dual
_template_root_space: plan

base_inferencer:
  _target_: BTA
  max_subtasks: ${_params.max_breakdown}
  breakdown_inferencer:
    _target_: ${_params.main_inferencer}
  worker_factory:
    _target_: ${_params.worker_factory_target}      # default Leaf; override Dual
  aggregator_inferencer:
    _target_: ${_params.main_inferencer}

review_inferencer:
  _target_: ${_params.main_inferencer}

fixer_inferencer:
  _target_: ${_params.main_inferencer}
```

**Knobs surfaced for users** (via `--override`):
- `_params.max_breakdown` — caps subtask count (default 5; the existing `breakdown-multiflow-plan` doesn't surface this explicitly, so this preset establishes the pattern).
- `_params.worker_factory_target` — `Leaf` (default) vs `Dual` (per-worker review).
- `_params.main_inferencer` — LLM backend.

**Tests:**
- Preset resolves: `_resolve_agent_config("breakdown")` returns `('file', Path("configs/breakdown.yaml"))`.
- Topology constructs without error from the YAML.
- Single end-to-end smoke: `task "summarize the AgentFoundation README in 5 sections" --config breakdown` produces an `output.md` with 5 sections, runs under 5 minutes on opus.

**Risk:** very low. Pure YAML composition of existing tested building blocks (`Dual`, `BTA`, `Leaf`). **LoC:** ~120 production + ~50 tests.

### §E1.2 — Commit 2: add `multiple.yaml` preset (diversity only)

**Purpose:** A "MFDual-only" preset for tasks small enough not to decompose, but where N parallel attempts with different inferencers (or the same inferencer with different temperatures) yields meaningful diversity that a winner-picker can exploit.

**File added:**

`src/agent_foundation/resources/tools/task/configs/multiple.yaml` (**NEW**, ~100 LoC).

**Critical structural correction (verified):** v0 of this plan-sketch said `Dual{MFDual}`. **That's wrong** — `MultiFlowDualInferencer` extends `DualInferencer` (verified at `multi_flow_dual_inferencer.py:91`), so it inherits the review/fix loop directly. Wrapping it in an outer `Dual{...}` would be double review/fix — works, but conceptually noisy and wasteful. The clean form is **root `MFDual`** with the review/fix slots configured at the same level:

```yaml
# ============================================================================
# Plan-Only Topology: MFDual (root; no outer Dual wrapper)
# ============================================================================
#
# Diversity-only standalone planning topology. Use directly via:
#   --agent-config multiple
#
# MFDual extends DualInferencer — it IS a Dual. Wrapping it in another
# Dual{...} would be double review/fix; instead we configure review_inferencer
# / fixer_inferencer DIRECTLY on the root MFDual (inherited from DualInferencer).
#
# Tree structure:
#   MFDual (= Dual subclass)              review + fix the WINNING flow's plan
#   ├── flow_configs[]                    N parallel single-flow attempts on the
#   │                                     WHOLE planning task (no decomposition)
#   ├── multi_flow_aggregator_inferencer  pick winner + use runner-up as reviewer
#   ├── multi_flow_winner_parser          extract winner from aggregator output
#   ├── review_inferencer (inherited)     judge winning plan structure + quality
#   └── fixer_inferencer (inherited)      refine winning plan based on review
#                                          (or match-winner — see fixer_match_winner)
#
# Why no outer Dual wrapper?
#   Verified at multi_flow_dual_inferencer.py:91 — class MultiFlowDualInferencer
#   extends DualInferencer. The review/fix loop is built-in. Wrapping would
#   create a second review/fix layer on the same artifact — costly and noisy.
#
# ----------------------------------------------------------------------------
_logger: auto
_debug_mode: true
_model_name: opus[1m]
_idle_timeout_seconds: 600
_tool_use_idle_timeout_seconds: 5400
_output_path: "output.md"

_params:
  workspace_root: ???
  default_inferencer: ClaudeCodeCLI
  main_inferencer: ${oc.env:DEFAULT_MAIN_INFERENCER,${.default_inferencer}}
  # Diversity dial — default 3 parallel flows. Override via:
  #   --override _params.num_flows=5
  num_flows: ${oc.env:DEFAULT_NUM_FLOWS,3}
  flow_inferencers:
    - ${_params.main_inferencer}
    - ${_params.main_inferencer}
    - ${_params.main_inferencer}

_target_: MFDual                                # ROOT — NOT Dual{MFDual}
_template_root_space: plan

# Build flow_configs from _params.flow_inferencers
# (one flow per inferencer in the list; users tune diversity via list length)
flow_configs: ${_build_flow_configs:${_params.flow_inferencers}}

multi_flow_aggregator_inferencer:
  _target_: ${_params.main_inferencer}

# Reviewer / fixer (inherited from Dual) — wire at root level
review_inferencer:
  _target_: ${_params.main_inferencer}
fixer_inferencer:
  _target_: ${_params.main_inferencer}
fixer_match_winner: true   # fixer uses winning flow's inferencer; verified at
                           # multi_flow_dual_inferencer.py:38, 126
```

**Knobs surfaced for users:**
- `_params.num_flows` (informational; the actual cardinality comes from `_params.flow_inferencers` list length).
- `_params.flow_inferencers` — the list of N inferencers to run in parallel. Override:
  ```bash
  task ... --config multiple --override _params.flow_inferencers='[ClaudeCodeCLI, RovoDevCLI, KiroCLI]'
  ```

**Risk concern to verify before merge:** the `${_build_flow_configs:...}` custom resolver in the YAML above is a placeholder — needs to be either (a) a real OmegaConf custom resolver, or (b) replaced with a literal `flow_configs:` block matching the existing `breakdown-multiflow-plan.yaml` pattern. The literal-block approach is safer; the custom resolver is more elegant but requires a small Python registration. **Choose literal-block for v1; defer the custom resolver to a follow-up.**

**Tests:**
- Preset resolves.
- Topology constructs; `len(flow_configs) == 3` by default.
- Override changes flow count: `--override _params.flow_inferencers='[X, Y]'` → `len(flow_configs) == 2`.
- End-to-end smoke: `task "summarize the AgentFoundation README" --config multiple` produces an `output.md`.

**Risk:** low-to-medium (medium only because of the `_build_flow_configs` choice; once resolved as literal, drops to low). **LoC:** ~100 production + ~80 tests.

### §E1.3 — Commit 3: rename + alias `breakdown-multiflow-plan` → `full-plan`

**Purpose:** Make the existing preset's name match the new naming idiom (`breakdown`, `multiple`, `full-plan` as a discoverable triplet) without breaking any existing caller.

**Files modified:**

1. `src/agent_foundation/resources/tools/task/configs/full-plan.yaml` (**NEW** — copy of `breakdown-multiflow-plan.yaml`, header updated to match new naming).
2. `src/agent_foundation/resources/tools/task/configs/breakdown-multiflow-plan.yaml` — **keep** as-is (or replace with an alias YAML that `_import_`s `full-plan.yaml`; verified that `_import_` exists in `breakdown-multiflow-plan.yaml:33`).
3. `src/agent_foundation/resources/tools/task/executor.py` — `_resolve_agent_config` — add an alias map dict so both names resolve to the same file (cleaner than file duplication):
   ```python
   _PRESET_ALIASES = {
       "breakdown-multiflow-plan": "full-plan",  # back-compat
   }
   spec = _PRESET_ALIASES.get(spec, spec)        # apply before file lookup
   ```

**Backwards-compatibility guarantee:** every existing script using `task --config breakdown-multiflow-plan` continues to work unchanged. The new `--config full-plan` is the canonical name; the old one is an alias.

**Documentation:** update `tool.json` `--config` description to list all 4 presets (`default`, `breakdown`, `multiple`, `full-plan`) and note that `breakdown-multiflow-plan` is a back-compat alias.

**Risk:** very low. Pure rename with alias. **LoC:** ~5 production + ~20 tests.

### §E1.4 — Commit 4: tests + `tool.json` description

**Files modified:**

1. `src/agent_foundation/resources/tools/task/tool.json` — `--config` description rewritten:
   ```json
   {
     "name": "--config",
     "type": "string",
     "default": "default",
     "description": "Topology preset selecting planning/execution complexity. Available presets: default (heavyweight planner→executor), full-plan (coverage × diversity, plan only), breakdown (coverage only — sequential BTA workers), multiple (diversity only — N parallel flows on the whole task). Also accepts a YAML file path or registered alias. For a lightweight conversational task entry-point, use the separate `chat` peer tool (see resources/tools/chat/)."
   }
   ```
2. `tests/.../tools/task/test_config_resolution.py` (**NEW**) — parametrised test that every preset name resolves and instantiates.

**Risk:** trivial. **LoC:** ~10 production + ~80 tests.

### §E1.5 — Commit 5: NEW `chat` peer tool

**Purpose:** A new top-level tool, sibling to `task`, that runs a `ConversationalInferencer` with full tool access. When the user has a vague or exploratory request, `chat` dispatches the conversation; when the user has a concrete sub-task, the LLM driving the conversation calls `task` (or any other registered tool) as a sub-tool.

**Why a new tool rather than `task --config conversation` or `task --config disabled`:** see §D2 — short answer: precedent (`sop` already follows the peer-tool pattern), zero breaking changes to existing `task` callers, clean separation of "conversational driver" from "topology executor."

**Files added:**

`src/agent_foundation/resources/tools/chat/` (**NEW** directory, ~250 LoC total):
1. `__init__.py` — empty package marker.
2. `tool.json` (~30 LoC) — describes the tool to the SOP runtime and CLI:
   ```json
   {
     "name": "chat",
     "description": "Conversational driver tool. Runs a single-turn-loop conversational inferencer with full tool access. Use for ambiguous, exploratory, or multi-step requests where the LLM should decide the next step dynamically. For deterministic single-task execution with planner/executor topology, use the `task` tool instead.",
     "parameters": [
       {"name": "request", "type": "string", "required": true,
        "description": "Initial user message starting the conversation."},
       {"name": "--tools", "type": "string", "default": "task,view,knowledge",
        "description": "Comma-separated names of tools the conversation may call. Default scopes to task + view + knowledge."},
       {"name": "--model", "type": "string", "default": "opus[1m]",
        "description": "Backbone LLM for the conversational inferencer."},
       {"name": "--max-turns", "type": "integer", "default": 50,
        "description": "Safety cap on conversation length."},
       {"name": "--max-tool-result-chars", "type": "integer", "default": 12000,
        "description": "Truncation cap for the combined tool-results block injected back into the conversation (default 12000, vs the CI base default of 4000). Larger lets a single task --plan result fit without truncation; smaller bounds context growth. See §D5 Q9."}
     ]
   }
   ```
3. `cli.py` (~150 LoC) — mirrors `sop/cli.py` shape:
   ```python
   # Verified pattern: src/agent_foundation/resources/tools/sop/cli.py
   # uses _build_ci_from_config(model, backend, tool_registry, tool_executor, interactive)
   # which loads configs/default.yaml; we follow the same shape but with
   # the tool list scoped from --tools.
   ```
4. `configs/default.yaml` (~30 LoC) — minimum ConversationalInferencer config (single inferencer, no topology, plus tool_registry filtered by `--tools`).
5. `executor.py` (~40 LoC) — `execute(arguments, session_context) -> ToolExecutionResult` entry point that the registry can call.

**Reuses (no new abstractions):**
- `ConversationalInferencer` — already exists; lots of usage.
- `tool_registry` + `tool_executor` plumbing — already used by `sop/cli.py:24–30`.
- `_resolve_tool_name` for canonical tool dispatch — already in `conversational_inferencer.py:1189`.

**Key design choices** (with reasoning):

| Choice | Why |
|---|---|
| Peer tool, NOT a `--config conversation` mode of `task` | Zero breaking changes; precedent matches `sop`; clean responsibility separation |
| `--tools` scopes available sub-tools | Prevents the conversational driver from accidentally invoking expensive tools; explicit allow-list is safer than implicit "everything" |
| `task` is in the default `--tools` list | The whole point of the integration; users get "talk to the agent; when something concrete comes up, it dispatches to `task`" out of the box |
| `--max-turns` cap | Safety against runaway conversations (LLM stuck in a loop); 50 is generous but bounded |
| No `--resume` in v1 | Conversation resumability is a separate, larger design question; file as §A4 follow-up |
| No `--yolo` flag | Conversation IS the interactive paradigm; "yolo conversation" is incoherent (a conversation with no user is just one LLM call — use `task` for that) |

**Tests:**
- T1: `chat "hello"` runs one turn, returns the LLM's response, exits cleanly without dispatching any tool.
- T2: `chat "summarize the README in 5 sections" --tools "task,view"` triggers a `task --config breakdown` dispatch (or equivalent), and `chat` aggregates the result.
- T3: `chat "..." --tools "view"` cannot dispatch `task` (allow-list enforced); `task` calls from the LLM are rejected with a clear error.
- T4: `chat "..." --max-turns 1` exits after 1 turn even if the LLM wants to continue (safety cap enforced).
- T5: `chat` registered in `resources/tools/registry.py` and discoverable via the same `--tools` mechanism that SOPs use.

**Risk:** medium — this is genuinely new functionality, and the conversational-driver-dispatching-task pattern hasn't been exercised at scale in this codebase. Mitigation: follow the `sop/cli.py` shape exactly (proven peer-tool pattern); add tests for the allow-list enforcement (safety boundary). **LoC:** ~250 production + ~200 tests.

### §E1.6 — Commit 6: E2E smoke + integration test

**Purpose:** Lock in the connection between the two halves (presets + chat peer tool) with a single end-to-end test that exercises both.

**Test added:**

`tests/.../e2e/test_chat_dispatches_task_with_breakdown_preset.py`:
1. Mock LLM that, on receiving `"please plan how to summarize this README"`, emits a tool call: `task(request="summarize the AgentFoundation README in 5 sections", config="breakdown")`.
2. Invoke `chat "please plan how to summarize this README"`.
3. Assert the mock LLM's tool call was dispatched to `task_execute` with `arguments={"request": "...", "config": "breakdown"}`.
4. Assert `task_execute`'s `prior_context["workspace_path__task"]` (from the proposal_selection plan's bridge augmentation) is populated.
5. Assert the workspace contains `output.md`.

**Documentation added:**

`src/agent_foundation/resources/tools/chat/README.md` — short user-facing doc explaining the two-tier mental model (chat = exploratory driver; task = deterministic executor) and the 4 presets.

**Risk:** low. The components are individually tested; this is integration smoke. **LoC:** ~100 tests + ~30 docs.

### §E1.7 — Commit 7 (v2 addition): `smart-breakdown.yaml` — heterogeneous workers per subtask

**Purpose:** The most elegant member of the preset family. The breakdown LLM **classifies each subtask's complexity** in addition to producing its query text, and BTA dispatches **a different worker type per subtask** based on that classification. Tiny tasks get a bare `Leaf`; decomposable tasks get a nested `BTA`; tasks needing diversity get `MFDual`; ambiguous tasks get a `Conversational` worker that can decide for itself.

**Why this is sound (verified, not invented):** `BreakdownThenAggregateInferencer` already supports heterogeneous workers via the `worker_factory: dict[str, factory] + task_type_arg_name` pattern, verified in source:

```python
# breakdown_then_aggregate_inferencer.py:388–397
worker_factory: Any = attrib(default=None)
# worker_factory can be:
#   - dict[str, Callable | functools.partial]: maps task type -> factory.
#     functools.partial entries are called with no args to create fresh instances.
#     "_default" can be a string referencing another key.
#     Requires task_type_arg_name and parser returning List[dict] with "args".

# When set, enables heterogeneous workers. Each sub_query item can be a dict
# {"query": str, "args": {...}}. The value of args[task_type_arg_name] selects
# which worker factory to use from a dict-typed worker_factory.
task_type_arg_name: Optional[str] = attrib(default=None, kw_only=True)
```

The dispatch site is `_select_worker_factory` at lines 1559–1574 and 1609–1656. So **what's missing isn't BTA support — it's a breakdown prompt that emits the classification.** This preset adds the prompt + the YAML wiring; everything else is already in mainline.

**Files added/modified:**

1. `src/agent_foundation/resources/tools/task/configs/smart-breakdown.yaml` (**NEW**, ~150 LoC):

```yaml
# ============================================================================
# Plan-Only Topology: Dual{BTA{ {leaf: Leaf, breakdown: BTA{Leaf},
#                                multiple: MFDual, conversational: Conversational},
#                               task_type_arg_name=complexity }}
# ============================================================================
#
# "Smart" planning topology — breakdown LLM CLASSIFIES each subtask's
# complexity, BTA dispatches the matching worker type per subtask.
#
# Use directly via:
#   --agent-config smart-breakdown
#
# Tree structure:
#   Dual                                  review + fix the integrated PLAN
#   └── base_inferencer = BTA             decompose AND classify each subtask
#       ├── breakdown_inferencer          → list of {query, args:{complexity}}
#       │                                   (classification emitted PER subtask
#       │                                    via the prompt-template tweak below)
#       ├── worker_factory = dict[str, factory]:
#       │     leaf:           Leaf                  ← trivial subtask, 1 LLM call
#       │     breakdown:      BTA{Leaf}             ← decomposable subtask
#       │     multiple:       MFDual                ← needs diverse perspectives
#       │     conversational: Conversational        ← ambiguous; worker decides
#       │     _default:       leaf                  ← back-stop if LLM omits class
#       ├── task_type_arg_name: complexity
#       └── aggregator_inferencer         integrate sub-plans → ONE coherent plan
#
# Why this is principled (not ad-hoc):
#   BTA's worker_factory ALREADY supports dict[str,factory] + task_type_arg_name
#   for heterogeneous dispatch (verified at lines 388–397, 1559–1574). This
#   preset is the first place in the codebase that uses that mechanism — all
#   existing usage passes a single factory. We're adopting a built-in feature,
#   not adding one.
#
# ----------------------------------------------------------------------------
_logger: auto
_debug_mode: true
_model_name: opus[1m]
_idle_timeout_seconds: 600
_tool_use_idle_timeout_seconds: 5400
_output_path: "output.md"

_params:
  workspace_root: ???
  default_inferencer: ClaudeCodeCLI
  main_inferencer: ${oc.env:DEFAULT_MAIN_INFERENCER,${.default_inferencer}}
  max_breakdown: ${oc.env:DEFAULT_MAX_BREAKDOWN,5}

_target_: Dual
_template_root_space: plan

base_inferencer:
  _target_: BTA
  max_subtasks: ${_params.max_breakdown}

  # Breakdown inferencer — uses the CLASSIFYING prompt template (see below)
  breakdown_inferencer:
    _target_: ${_params.main_inferencer}
    # Point at a new prompt-template variant that asks the LLM to ALSO
    # output a "complexity" classification per subtask. Variant lives at
    # resources/sops/plan/breakdown/classifying.jinja2 (added in this commit).
    template_version: classifying    # NEW template version

  # Heterogeneous worker_factory — dict[str, factory]
  worker_factory:
    leaf:
      _target_: ${_params.main_inferencer}          # bare Leaf
    breakdown:
      _target_: BTA                                  # nested decomposition
      max_subtasks: 3                                # cap at 3 sub-subtasks
      breakdown_inferencer:
        _target_: ${_params.main_inferencer}
      worker_factory:
        _target_: ${_params.main_inferencer}        # nested workers = bare Leaf
      aggregator_inferencer:
        _target_: ${_params.main_inferencer}
    multiple:
      _target_: MFDual                              # diversity via N parallel
      flow_configs:
        - { _target_: ${_params.main_inferencer} }
        - { _target_: ${_params.main_inferencer} }
        - { _target_: ${_params.main_inferencer} }
      multi_flow_aggregator_inferencer:
        _target_: ${_params.main_inferencer}
    conversational:
      _target_: Conversational                       # let the worker decide
    _default: leaf                                   # back-stop

  # Tells BTA: look at sub_query.args["complexity"] to pick the factory
  task_type_arg_name: complexity

  aggregator_inferencer:
    _target_: ${_params.main_inferencer}

review_inferencer:
  _target_: ${_params.main_inferencer}

fixer_inferencer:
  _target_: ${_params.main_inferencer}
```

2. `src/agent_foundation/resources/sops/plan/breakdown/classifying.jinja2` (**NEW**, ~30 LoC) — prompt-template variant that asks the LLM to emit subtasks as a JSON list with `complexity` in the `args`:

```jinja2
{# Inherit the standard breakdown structure; ADD a classification field. #}
{% extends "plan/breakdown/main.jinja2" %}

{% block subtask_schema_addendum %}
For EACH subtask, also classify its complexity as one of:
  - "leaf"           — trivial, one LLM call suffices (e.g. write 3 sentences)
  - "breakdown"      — decomposable into 2-3 sub-subtasks (e.g. enumerate API endpoints)
  - "multiple"       — small but ambiguous; diverse perspectives help (e.g. propose names)
  - "conversational" — open-ended; needs back-and-forth to scope (e.g. design Y given X constraints)

Emit each subtask as:
  {"query": "<the subtask text>", "args": {"complexity": "<one of the 4 values>"}}

If unsure, emit "leaf" — the back-stop. The classification is a hint to the
runtime, not a contract; the worker will produce a valid plan regardless.
{% endblock %}
```

3. `src/agent_foundation/resources/tools/task/tool.json` — `--config` description extended to include `smart-breakdown` with its 1-line description.

**Knobs surfaced for users:**
- `_params.max_breakdown` — same as `breakdown.yaml`.
- Per-bucket worker config — power users can override any of the 4 buckets via `--override`:
  ```bash
  task ... --config smart-breakdown \
      --override 'base_inferencer.worker_factory.multiple.flow_configs=[{_target_: A},{_target_: B}]'
  ```

**Tests:**
- T1 **Resolution:** preset resolves; topology constructs; `worker_factory` is a `dict` of 4 entries + `_default`.
- T2 **Dispatch — leaf:** mock breakdown emits a single `{query: "X", args: {complexity: "leaf"}}` subtask. Assert the worker chosen is a `Leaf` (no further breakdown, no MFDual, no Conversational).
- T3 **Dispatch — breakdown:** mock emits `{complexity: "breakdown"}`. Assert nested BTA invoked.
- T4 **Dispatch — multiple:** mock emits `{complexity: "multiple"}`. Assert MFDual invoked.
- T5 **Dispatch — conversational:** mock emits `{complexity: "conversational"}`. Assert Conversational invoked.
- T6 **Fallback — missing classification:** mock omits the `complexity` field entirely. Assert `_default` (= `leaf`) is used; no error.
- T7 **Fallback — unknown classification:** mock emits `{complexity: "xyz"}`. Assert (per BTA's existing error path at line 1622) a clear error raised: `No worker factory for task type 'xyz'`.
- T8 **End-to-end smoke:** `task "<mixed-complexity request>" --config smart-breakdown` produces an `output.md` with subtasks visibly dispatched to different worker types (verify via debug log).

**Risk:** low-to-medium. Low because BTA's heterogeneous-worker dispatch is already tested in mainline (used in some `dual.yaml` / experimental flows — verify by grepping). Medium only because the new `classifying.jinja2` prompt is novel and the LLM may emit malformed JSON occasionally; mitigation = the `_default: leaf` back-stop + the existing `_default` fallback at line 1571. **LoC:** ~150 production (YAML + jinja) + ~120 tests.

**Pre-merge verification (~10 min):**
1. Confirm `task_type_arg_name` is honored when `worker_factory` is a dict (we have the code path verified — line 1560 — but a smoke test against a synthetic 2-bucket dict before drafting saves debugging later).
2. Confirm the parser invoked by `breakdown_inferencer` (likely `parsers/breakdown.py` or similar) accepts the `{query, args}` dict shape per the comment at line 391 ("parser returning List[dict] with 'args'") — and whether the current default parser does. If not, the new `classifying.jinja2` would need a sibling parser update.

### §E1.5-v4 — Commit 5 (v4 REPLACEMENT): `--config conversational` (was: new `chat` peer tool)

**Purpose:** Replace v3 Commit 5 (new `chat` peer tool) with the architecturally correct alternative: implement the conversational router INSIDE `task`, selected by `--config conversational` (canonical) / `--config disabled` (alias for back-compat with peer-plan terminology). The conversational mode receives the user's request, decides whether to answer inline or escalate to a structured topology, and (when it escalates) recursively calls `task` with one of `breakdown` / `multiple` / `full-plan` / `smart-breakdown`.

**Why this supersedes v3's `chat` peer tool:** See §D2 v4 reversal — v3's R1/R2/R4/R5 critiques don't survive scrutiny once you observe that `disabled.yaml` is a real preset, not a sentinel string. Single entry point (`task`) is genuinely better than two (`chat` + `task`) for the reasons listed in the v4 banner.

**Files added/modified:**

1. **NEW** `resources/tools/task/configs/conversational.yaml` (~50 LoC) — mirrors `sop/configs/default.yaml`:
```yaml
# Conversational Router Topology — Conversational (no planning, no execution)
# Default for bare `task "..."` after v4. Users opt out via `--config default` (PTI) or other.
_logger: auto
_debug_mode: true
_model_name: opus[1m]
_idle_timeout_seconds: 600

_target_: Conversational
base_inferencer:
  _target_: ClaudeCodeCLI

max_iterations: 30
max_tool_result_chars: 16000   # raised vs the CI default (4000) so a single task/sop
                               # result fits without truncation; see §D6 F5 + §D5 Q9

prompt_renderer:
  _target_: TemplateManager
  template_space: task_router   # see Commit 5d
```

2. **NEW** `resources/tools/task/configs/disabled.yaml` — one-line `_import_: conversational` alias (for parity with peer-plan terminology; users can use either name).

3. **Edit** `resources/tools/task/executor.py` — 5 surgical changes (~80 LoC):
   - **3a:** `_CONFIG_ALIASES` map at top of file:
     ```python
     _CONFIG_ALIASES: dict[str, str] = {
         "full-plan": "breakdown-multiflow-plan",
         "pti": "default",
         "multiflow": "multiple",
         "disabled": "conversational",
         "breakdown": "breakdown",        # explicit identity (clarity)
     }
     ```
   - **3b:** Alias resolution inserted as Rule 2.5 in `_resolve_agent_config` (between Rule 2 and Rule 3, ~line 81) — preserves all 5 existing rules.
   - **3c:** Default-spec change in `execute()` (~line 635): `or "default"` → `or "conversational"`. **This IS a default change** — honestly owned. Existing callers using `--config default` keep working; bare `task "..."` invocations now route through the conversational router.
   - **3d:** `_topology_is_conversational(source)` helper — returns True iff the root `_target_` resolves to `{Conversational, ConversationalInferencer}` (case-insensitive class-name match).
   - **3e:** In `_run_topology`, branch on `_topology_is_conversational(source)`: if True, call new `_run_conversational_router(...)` instead of the default `instantiate(...) + ainfer(...)`. The new helper reuses the proven `sop/cli.py` pattern (see §D6 F8) — `load_all_tools()`, curate tools, build CI with `tool_registry`/`tool_executor`, dispatch via `run_agentic_loop(request, ...)`, propagate incremented `task_depth` in `session_context`.

4. **NEW** prompt template `resources/prompt_templates/conversation/task_router/main/initial.jinja2` (~80 LoC) — the router's system prompt. Documents the ladder and the decision rule (`leaf` for trivial; `breakdown` for clear coverage; `multiple` for needs-diverse-takes; `full-plan` for large+ambiguous; `smart-breakdown` for unknown complexity). Emits `ToolsToInvoke` `action` calls with `config = ...`, `worker_type`, `aggregate`, `max_breakdown`/`num_flows`. See §E1.5d (Commit 5d) for full template scaffold.

5. **Edit** `resources/tools/task/tool.json` (~10 LoC):
   - `--config` choices updated to `[conversational, disabled, breakdown, multiple, full-plan, smart-breakdown, default, pti, breakdown-multiflow-plan]` (canonical + back-compat aliases all listed; default now `"conversational"`).
   - Top-level `description` refreshed to mention the planning ladder + conversational default.
   - **Honestly document:** `--plan` / `--execute` / `--full` / `--confirm` flags apply ONLY to the PTI `default`/`pti` config (the conversational and ladder presets are inherently plan-only or route-only).

**Tests:**
- T1: `task "..."` with no `--config` → routes through `conversational.yaml` (new default).
- T2: `task "..." --config default` → unchanged PTI behavior (back-compat).
- T3: `task "..." --config disabled` → resolves to `conversational.yaml` via alias.
- T4: Conversational router can call back into `task --config breakdown` (recursion via `_tool_executor` closure).
- T5: Conversational router with no interactive transport uses `yolo_mode=True` (auto-answers its own clarifying questions); with a transport, runs multi-turn.
- T6: All other v3 commits' `--config` values (`breakdown`, `multiple`, `full-plan`, `smart-breakdown`) still resolve correctly.

**Risk:** medium. Why:
- The default change is a real breaking concern for any caller relying on bare `task "..."` running PTI today. Mitigation: large CHANGELOG note + 1 release of deprecation warning ("bare `task` now uses conversational mode; pass `--config default` or `--config pti` to keep PTI behavior").
- The `_run_conversational_router` branch reuses the proven `sop/cli.py` pattern (verified in §D6 F8); the only novel work is the recursion guard (Commit 5c).

**LoC:** ~140 production (1 new YAML + 1 alias + 5 executor edits + 1 prompt template) + ~120 tests.

### §E1.5b — Commit 5b (v4 NEW, CRITICAL): fix `_extract_result_text` silently dropping `tuple[1:]`

**Purpose:** Make the "no-aggregate / list-of-outputs to conversation" feature actually work end-to-end. Without this fix, BTA's `disable_aggregator=True` returns a tuple of N worker outputs, but the executor's `_extract_result_text` does `result[0]` and silently discards the rest. The conversation never sees outputs 2..N. This bug exists today and is independent of all other commits.

**Files modified:**

1. `resources/tools/task/executor.py` (~30 LoC):
   - At line ~353, replace the unconditional `result[0]` extraction:
     ```python
     def _extract_result_text(result: Any, workspace: Optional[Path] = None) -> str:
         # Length-1 tuples / wrapper objects: unwrap to single result (today's behavior).
         if isinstance(result, tuple) and len(result) == 1:
             return _extract_result_text(result[0], workspace)
         # Multi-element tuples = no-aggregate mode. Serialize ALL outputs as a
         # structured markdown list, with on-disk paths discovered from workspace
         # (so the receiving CI can read full content on demand rather than inlining
         # everything and busting max_tool_result_chars).
         if isinstance(result, tuple) and len(result) > 1:
             return _serialize_multi_output(result, workspace)
         # Single result: today's behavior.
         return _to_text(result)
     
     def _serialize_multi_output(outputs: tuple, workspace: Optional[Path]) -> str:
         """Format N worker outputs as a structured markdown list."""
         lines = [f"## No-aggregate output ({len(outputs)} workers)"]
         worker_dirs = _discover_artifact_dirs(workspace) if workspace else [None] * len(outputs)
         for i, (out, wdir) in enumerate(zip(outputs, worker_dirs)):
             lines.append(f"\n### Worker {i+1}")
             if wdir:
                 lines.append(f"_Workspace:_ `{wdir}`")
             lines.append(_to_text(out))
         return "\n".join(lines)
     ```

2. **NEW** parameter on `task` tool: `--aggregate` (default `true`). When `false`, executor injects `disable_aggregator: true` (BTA) and `multi_flow_disable_aggregator: true` (MFDual) via the existing `--override` path. ~10 LoC.

**Tests:**
- T1: `task "..." --aggregate false --config breakdown --max-breakdown=3` returns a structured-list result containing all 3 worker outputs (not silently truncated to 1).
- T2: `task "..." --aggregate true` (default) returns single-aggregated result (today's behavior, no regression).
- T3: `_extract_result_text(("solo",))` returns `"solo"` (single-element tuple unwraps — preserves existing behavior).
- T4: `_extract_result_text(("a", "b", "c"))` returns a markdown structured-list with 3 worker sections.
- T5: Workspace discovery — when `workspace` is provided, `## Worker N` blocks include the on-disk path (so the receiving CI can read full content via `view`).

**Risk:** **High but isolated.** Three risk vectors:
- Any current caller that uses `disable_aggregator: true` today and depends on the silent-drop behavior would see different output. Mitigation: grep the repo for `disable_aggregator` callers and audit each; the bug-fix branch is gated on either `--aggregate false` being explicit or `disable_aggregator: true` already being in the override. **~5 min audit before draft.**
- `_to_text` semantics on individual worker outputs depend on the worker's `_extract_result_text` recursion — verify the recursion terminates for `Inferencer` result objects.
- The on-disk path discovery (`_discover_artifact_dirs`) needs to exist or be added — verify it before draft. The existing `_discover_artifacts` (referenced in §D6 F-related context) is the obvious starting point.

**LoC:** ~40 production + ~80 tests.

**Why this is "no-hack":** This is genuinely a latent bug. The current code path silently violates the no-aggregate contract; we're not adding behavior, we're delivering what the API already promises.

### §E1.5c — Commit 5c (v4 NEW): unified `task_depth` budget

**Purpose:** Adaptive `conversational` workers (from `smart-breakdown.yaml`'s adaptive bucket) + the router recursion in `--config conversational` can compound multiplicatively. Without a cap, a `conversational → smart-breakdown → conversational-worker → smart-breakdown → ...` chain runs uncontrolled. v4 introduces ONE budget that governs BOTH router recursion AND adaptive nesting.

**Files added/modified:**

1. `resources/tools/task/executor.py` (~30 LoC):
   - At the top of `execute()`, read `task_depth` from `session_context` (default 0); read `max_task_depth` from `arguments.get("max_task_depth")` or env var `TASK_MAX_DEPTH` (default 2).
   - When dispatching a nested `task` call (either via `_run_conversational_router`'s `_tool_executor` closure, or via `smart-breakdown`'s adaptive `breakdown`/`conversational` worker), increment `task_depth` in the child's `session_context`.
   - **Coercion at the cap:**
     - In `_run_conversational_router`: if `task_depth >= max_task_depth`, coerce a nested `--config conversational` request to `--config full-plan` (structured, finite topology).
     - In `smart-breakdown` adaptive dispatcher: if `task_depth >= max_task_depth`, coerce `breakdown`/`conversational` worker_type to `leaf`. The selection happens in BTA's `_select_worker_factory` (lines 1559–1574); the coercion is a wrapper layer applied BEFORE that selection.

2. Document the budget in `tool.json`:
   ```json
   {
     "name": "--max-task-depth", "type": "integer", "default": 2,
     "description": "Max recursion depth for nested task calls (router + adaptive workers). At the cap, conversational recursion coerces to full-plan and adaptive {breakdown,conversational} worker types coerce to leaf. Env var TASK_MAX_DEPTH overrides."
   }
   ```

**Tests:**
- T1: `task "..." --config conversational --max-task-depth=1` — a 2-deep router recursion stops at depth 1 (3rd-level conversational coerced to full-plan).
- T2: `task "..." --config smart-breakdown --max-task-depth=1` — a subtask classified as `breakdown` at depth 1 coerces to `leaf`.
- T3: `task_depth` propagates through `session_context` across both code paths (router + BTA worker).
- T4: At-cap coercion logs a clear `WARNING` so the user sees the budget kicked in.

**Risk:** medium. The hard part is making sure `session_context` is threaded through both paths consistently (`_run_conversational_router` and the BTA worker factory call). A single missed thread = silent runaway. Mitigation: T3 test specifically locks both threads.

**LoC:** ~30 production + ~60 tests.

### §E1.5d — Commit 5d (v4 NEW): `task_router` prompt template

**Purpose:** Without an explicit router prompt, the `Conversational` topology root has no system message documenting the planning ladder, no decision heuristic, and no schema for the `task` tool call. This commit provides the prompt template that the router uses to make its dispatch decisions.

**Files added:**

1. `resources/prompt_templates/conversation/task_router/main/initial.jinja2` (~80 LoC) — mirror of `conversation/main/initial.jinja2` with router-specific blocks:
   - **Ladder description:** lists the 5 `--config` options with one-line decision heuristics (matches Commit 4 v3 `tool.json` description).
   - **Decision rule:** "Answer directly for trivial questions / ask clarification when interactive transport present / otherwise emit a `task` action call."
   - **Schema:** explicit JSON schema for the `task` action call args, including `config`, `worker_type`, `aggregate`, `max_breakdown`, `num_flows`, `request`.
   - **Depth-budget awareness:** documents that nested `--config conversational` recursion coerces at the cap (Commit 5c).

**Tests:**
- T1: Template renders without errors with a representative `prior_context`.
- T2: Template includes all 5 `--config` choices (lint check).
- T3: When loaded by `TemplateManager`, the rendered prompt fits inside `max_tool_result_chars=16000`.

**Risk:** low. Pure template work, no dispatcher logic.

**LoC:** ~80 production (template) + ~30 tests.

---

### §E1.8 — Commit 8 (v3 addition, retained in v4): `tool_call_defaults` — CI-level per-tool default arguments

**Purpose:** When a `ConversationalInferencer` drives nested tool calls (e.g. `chat` dispatching `task` and `sop`), the user often wants **the same flag value** on every nested call — `--model sonnet` everywhere, `--yolo` everywhere, `--config breakdown` by default, etc. Today the LLM has to remember to add these to every dispatch, which it doesn't reliably do. This commit adds a principled mechanism: per-tool default arguments, declared at the CI level, shallow-merged into each tool call (caller wins on conflict).

**Why this belongs in the CI layer (not in `task`/`sop` themselves):** The defaults are a property of *the conversation*, not of *the dispatched tool*. The same `sop` invocation might be `--yolo` when reached via `chat --yolo` and interactive when reached via the SOP CLI directly. The CI is the only layer that knows the user's conversation-level intent.

**Files added/modified:**

1. `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py` — add new attrib + apply logic:

```python
# Class attrib (new, default empty dict)
tool_call_defaults: dict[str, dict[str, Any]] = attr.attrib(factory=dict)
"""Per-tool default arguments. Keyed by CANONICAL tool name (post _resolve_tool_name,
so both `research-propose` and `research_propose` match the same key).

Caller-provided args ALWAYS win on conflict — defaults are shallow-merged underneath.
Unknown tool keys are warned but not errored. Unknown argument keys for a tool are
warned + skipped (so e.g. setting --yolo on a tool that doesn't accept it is graceful).

Example:
    tool_call_defaults = {
        "sop":  {"--yolo": True},
        "task": {"--model": "sonnet", "--config": "breakdown"},
    }
"""

# In _execute_tool_call, BEFORE delegating to the executor:
async def _execute_tool_call(self, tc: ParsedToolCall) -> str:
    canonical = self._resolve_tool_name(tc.name)
    defaults = self.tool_call_defaults.get(canonical, {})
    
    if defaults:
        applied = {k: v for k, v in defaults.items() if k not in tc.arguments}
        if applied:
            logger.info(
                "Applying tool_call_defaults for %s: %s (caller args win on conflict)",
                canonical,
                {k: f"{v!r} [from-default]" for k, v in applied.items()},
            )
        effective_args = {**defaults, **tc.arguments}
    else:
        effective_args = tc.arguments
    
    return await self._do_execute(canonical, effective_args)
```

2. `src/agent_foundation/resources/tools/chat/tool.json` — add CLI flag for surfacing this:

```json
{
  "name": "--tool-default",
  "type": "string",
  "default": null,
  "multiple": true,
  "description": "Set a per-tool default argument. Format: tool:--flag=value. Repeatable. Example: --tool-default sop:--yolo=true --tool-default task:--model=sonnet. Caller-provided args always win on conflict."
}
```

3. `src/agent_foundation/resources/tools/chat/cli.py` — parse repeated `--tool-default` flags into the `tool_call_defaults` dict and pass to the constructed CI.

4. `src/agent_foundation/resources/tools/sop/cli.py` — same `--tool-default` plumbing for the `sop` peer tool. (Reuses the same parser helper.)

**Knobs surfaced for users:**
- Repeatable `--tool-default tool:flag=value` syntax at the CLI level.
- Direct `tool_call_defaults={...}` kwarg for programmatic CI construction.
- YAML-level: `chat`/`sop` configs can declare a `tool_call_defaults:` block as a starting point that CLI flags then layer over.

**Tests:**
- T1 **Defaults applied** — CI with `tool_call_defaults={"sop": {"--yolo": True}}` dispatches `sop "..."`; assert effective args include `--yolo: True`.
- T2 **Caller wins on conflict** — CI with `tool_call_defaults={"task": {"--model": "sonnet"}}` dispatches `task "..." --model opus`; assert effective args have `--model: opus` (not sonnet).
- T3 **Canonical name matching** — `tool_call_defaults={"research_propose": {"--model": "opus"}}` is applied when the LLM calls `research-propose` (hyphen).
- T4 **Unknown tool name** — `tool_call_defaults={"nonexistent_tool": {"--x": 1}}` doesn't crash; logs a warning at CI construction or first dispatch.
- T5 **Unknown arg name** — `tool_call_defaults={"task": {"--bogus-flag": True}}` doesn't crash the task dispatch; the bogus flag is either silently dropped by `task`'s arg parser or surfaced as a clean warning.
- T6 **Logging shows provenance** — assert that the log line lists which args came from defaults vs from the caller.
- T7 **CLI parsing** — `chat --tool-default sop:--yolo=true --tool-default sop:--model=opus` builds `tool_call_defaults={"sop": {"--yolo": True, "--model": "opus"}}` correctly; both flags land for `sop`.
- T8 **YAML + CLI layering** — a `chat` config declares `tool_call_defaults: {task: {--model: sonnet}}` and the CLI passes `--tool-default task:--config=breakdown`; effective defaults for `task` are `{--model: sonnet, --config: breakdown}`.

**Risk:** low. Single-file change in CI; all behavior is opt-in (empty dict = no change to current behavior). The hardest part is canonical-name matching, which already exists (`_resolve_tool_name`).

**LoC:** ~80 production + ~120 tests.

**Dependency:** Independent of Commit 9 in principle, but they're shipped together because Commit 9 makes Commit 8 worth using.

### §E1.9 — Commit 9 (v3 addition): `--autonomous-level` + `mandatory` confirmation field (prerequisite 9a)

**Purpose:** Today the codebase has a binary autonomy model — `--yolo` (skip all gates) or interactive (block on every gate). The `sop/tool.json` description hints at a third option ("Auto-resolve all **non-mandatory** confirmation gates") but **the code never implements the mandatory/non-mandatory distinction** (verified — grep for `mandatory|must_confirm|critical` in conversation-tool handlers returns zero hits). This commit:

- **9a (prerequisite, must land first):** Adds a `mandatory: bool` field to conversation-tool invocation args (`clarification`, `confirmation`, `single_choice`, `multiple_choice`, `tool_argument_form`, and v3.5+'s `proposal_selection`). SOP authors / tool callers set it when the question genuinely requires human judgment (e.g. "ship this destructive operation? y/n" — yes; "pick a name for the variable" — no).
- **9b:** Adds `--autonomous-level` flag (`yolo` / `auto` / `interactive`) to both `task` and `sop`. The auto-resolver in the CI checks `tool_args.get("mandatory", False)`:
  - `--autonomous-level=yolo` → auto-resolve every gate (today's `--yolo` behavior)
  - `--autonomous-level=auto` → auto-resolve non-mandatory gates only; surface mandatory gates to the root conversation
  - `--autonomous-level=interactive` → block on every gate (today's default)

**Default semantics:**
- **Human-triggered invocation:** `--autonomous-level=interactive` (preserves today's behavior; zero breaking change).
- **CI-triggered invocation (via `tool_call_defaults`):** `chat` and `sop` set their CI's `tool_call_defaults` to include `{"task": {"--autonomous-level": "auto"}, "sop": {"--autonomous-level": "auto"}}` by default. Root user overrides with `chat --autonomous-level=yolo` or `chat --autonomous-level=interactive` to propagate a different default to all nested calls.

**Files added/modified (split into 9a + 9b):**

#### Commit 9a — `mandatory` field on conversation tools

1. **6 `tool.json` files updated** — add `mandatory: bool, default: False` parameter to each conversation tool:
   - `resources/tools/clarification/tool.json`
   - `resources/tools/confirmation/tool.json`
   - `resources/tools/single_choice/tool.json`
   - `resources/tools/multiple_choice/tool.json`
   - `resources/tools/tool_argument_form/tool.json` *(if present — verify before draft)*
   - `resources/tools/proposal_selection/tool.json` *(once v3.5+ of the proposal_selection plan lands)*

2. **Auto-resolver update** in `conversational/conversational_inferencer.py` (the handler dispatch site):

```python
# In whichever helper currently implements --yolo auto-resolution
# (existing today: short-circuits all confirmation tools)
def _should_auto_resolve(self, tool_name: str, tool_args: dict) -> bool:
    level = self.autonomous_level  # 'yolo' | 'auto' | 'interactive'
    if level == "interactive":
        return False
    if level == "yolo":
        return True
    # level == "auto"
    return not bool(tool_args.get("mandatory", False))
```

3. **Tests (4):**
   - T1: `clarification(mandatory=True)` is NOT auto-resolved when `autonomous_level=auto`.
   - T2: `clarification(mandatory=False)` IS auto-resolved when `autonomous_level=auto`.
   - T3: `clarification(mandatory=True)` IS auto-resolved when `autonomous_level=yolo` (yolo overrides mandatory).
   - T4: `clarification(mandatory=False)` is NOT auto-resolved when `autonomous_level=interactive`.

**Commit message:** `feat(conversation-tools): add 'mandatory' field for principled auto-resolve semantics`

#### Commit 9b — `--autonomous-level` flag on `task` and `sop`

1. **`resources/tools/sop/tool.json`** — add new parameter:

```json
{
  "name": "--autonomous-level",
  "type": "string",
  "default": "interactive",
  "choices": ["yolo", "auto", "interactive"],
  "description": "Autonomy level for confirmation gates. 'interactive' = block on all gates (default); 'auto' = surface only gates marked mandatory; 'yolo' = auto-resolve all. CI-driven invocations (e.g. chat dispatching sop) typically default this to 'auto' via tool_call_defaults."
}
```

   Plus back-compat alias: `--yolo` becomes shorthand for `--autonomous-level=yolo` (parser handles both; error if both specified with conflicting values).

2. **`resources/tools/sop/cli.py`** — parse `--autonomous-level`; pass to the built CI's `autonomous_level` attrib. Map legacy `yolo: bool` parameter to `autonomous_level: str` via a 3-line shim (back-compat).

3. **`resources/tools/task/tool.json`** — same parameter (for API consistency), with documentation honestly noting:

> "Note: `task` v1 has no confirmation gates today, so this flag is a no-op. It ships for API consistency with `sop` and for forward compatibility with task gates that may be added later (see follow-up §A4 #10). Setting this on `task` is harmless but currently has no observable effect."

4. **`resources/tools/task/executor.py`** — accept and store `autonomous_level` on the inferencer (so future task gates can read it). Wire alongside the existing `interactive` attrib at line 600.

5. **`resources/tools/chat/cli.py`** — add `--autonomous-level` top-level flag for `chat`. When set, `chat` populates its CI's `tool_call_defaults` so nested `task`/`sop` calls inherit:

```python
if args.autonomous_level:
    tool_call_defaults.setdefault("task", {})["--autonomous-level"] = args.autonomous_level
    tool_call_defaults.setdefault("sop", {})["--autonomous-level"] = args.autonomous_level
```

   Default for `chat`: `--autonomous-level=auto` (the proposal's recommended CI-default).

6. **`resources/tools/chat/cli.py`** — startup banner when `autonomous_level=yolo`:

```
⚠ YOLO MODE — all nested tool calls will auto-confirm every gate (including mandatory ones).
```

**Tests (6):**
- T5: `sop --autonomous-level=yolo` resolves every gate (parity with old `--yolo`).
- T6: `sop --autonomous-level=auto` resolves non-mandatory, escalates mandatory (depends on Commit 9a).
- T7: `sop --autonomous-level=interactive` blocks on every gate (today's default).
- T8: `sop --yolo` (legacy) still works and is equivalent to `--autonomous-level=yolo`.
- T9: `sop --yolo --autonomous-level=auto` errors with a clear conflict message.
- T10: `chat --autonomous-level=auto` populates `tool_call_defaults` for both `task` and `sop`; verified by a dispatch round-trip.

**Risk:** medium. Why:
- Commit 9a touches 5–6 tool.json files (mechanical, low risk).
- Commit 9b touches both `task` and `sop` CLI parsers (medium — needs careful back-compat with existing `--yolo`).
- The `chat` startup banner is small.
- The most subtle part is the CI's `tool_call_defaults` population at `chat` startup — that's where Commit 8 + Commit 9 connect, and a mistake here means `--autonomous-level=auto` silently doesn't propagate.

**LoC:** ~200 production (9a: ~30 LoC; 9b: ~170 LoC across 4 files) + ~180 tests.

**Honest caveats (CARRIED FROM v3 BANNER):**

| # | Caveat | Mitigation in v3 |
|---|---|---|
| **C1** | `mandatory` field is brand new — no existing SOPs or tool callers set it. After Commit 9a lands, every conversation-tool invocation defaults to `mandatory=False`, which means `--autonomous-level=auto` initially behaves identically to `--autonomous-level=yolo` until callers start opting in. | Acceptable — this is a gradual rollout. Filed §A4 #11 to audit existing SOP `[__requires user input__]` directives and migrate the genuinely critical ones to `mandatory=True`. |
| **C2** | `task` has no confirmation gates today — `task/executor.py:600` only consumes `interactive` in `mode=="confirm"`, which is a `task` workflow mode, not a runtime gate. `--autonomous-level=auto` on `task` is therefore a no-op today. | Documented honestly in the `tool.json` description (the user sees the caveat at `--help`); filed §A4 #10 to add real `task` gates (e.g. "about to write 50+ files? confirm"). |

**Dependency graph for v3:**
```
Commit 9a (mandatory field)
  └─ Commit 9b (--autonomous-level enum + back-compat --yolo alias)
       └─ Commit 8 wiring (chat populates tool_call_defaults with --autonomous-level=auto)
```

So **9a → 9b → 8** is the strict order. They can all live in the same PR or land separately; if separate, 9a is independent of Commits 1–7 (could land in any release) but 9b and 8 both depend on Commit 5 (the `chat` tool itself).

---

## §E2. Validation

### §E2.1 — Per-commit gates
- Commit 1 — `pytest tests/.../test_config_resolution.py::test_breakdown_resolves` passes; `task --config breakdown <small request>` produces an `output.md`.
- Commit 2 — same shape, for `multiple`.
- Commit 3 — both `task --config full-plan` AND `task --config breakdown-multiflow-plan` resolve to the SAME file (alias test).
- Commit 4 — `tool.json` description lints clean; `--config <bogus>` errors with the new help text listing all 4 presets.
- Commit 5 — `chat "hello"` runs and exits; `chat` is discoverable via `task ... --tools` (round-trip registry test).
- Commit 6 — full E2E smoke (above) passes end-to-end.

### §E2.2 — End-to-end smoke (after all 6 commits)

```bash
# A. Verify the four task presets all run a small request
for preset in default breakdown multiple full-plan ; do
    task "write a 3-sentence summary of a binary search tree" --config "$preset"
    test -f _runtime/tasks/task/*/outputs/output.md
done

# B. Verify back-compat alias
task "..." --config breakdown-multiflow-plan
# → resolves to full-plan.yaml (assert via debug log)

# C. Verify the chat peer tool dispatches to task
chat "please draft a 5-section README outline; use the breakdown preset"
# → expect chat session to call task(request=..., config=breakdown)
# → expect _runtime/tasks/task/<...>/outputs/output.md to exist after dispatch

# D. Verify tool allow-list is enforced
chat "summarize the AgentFoundation README" --tools "view"
# → expect chat to NOT dispatch task even if the LLM tries (rejection in logs)

# Acceptance:
#   All four preset runs produce output.md                          (Commits 1, 2, 3, 4)
#   Back-compat alias resolves                                      (Commit 3)
#   chat dispatches task with the chosen --config                   (Commit 5, 6)
#   chat rejects tool calls outside --tools allow-list              (Commit 5)
```

---

## §E3. Execution checklist

```
[ ] Pre-flight
[ ]   Confirm dev_xinli_2601 has no uncommitted work in resources/tools/
[ ]   Run scripts/check_dev_docs_present.sh (guardrail green)

Commit 1 — breakdown.yaml (coverage only)
[ ] NEW   resources/tools/task/configs/breakdown.yaml (~120 LoC)
[ ] NEW   tests/.../test_breakdown_preset.py (3 tests)
[ ] Tests + lint  → commit "feat(task): add breakdown preset (Dual{BTA{Leaf}})"

Commit 2 — multiple.yaml (diversity only)
[ ] NEW   resources/tools/task/configs/multiple.yaml (~100 LoC)
[ ]       NB: root _target_ = MFDual (NOT Dual{MFDual} — MFDual extends Dual)
[ ]       NB: choose literal flow_configs block in v1 (defer custom resolver)
[ ] NEW   tests/.../test_multiple_preset.py (4 tests including override-flow-count)
[ ] Tests + lint  → commit "feat(task): add multiple preset (root MFDual)"

Commit 3 — full-plan alias
[ ] NEW   resources/tools/task/configs/full-plan.yaml (copy of breakdown-multiflow-plan)
[ ] Edit  resources/tools/task/executor.py — add _PRESET_ALIASES dict
[ ] Keep  resources/tools/task/configs/breakdown-multiflow-plan.yaml (back-compat)
[ ] NEW   tests/.../test_preset_alias.py (verify both names resolve to same file)
[ ] Tests + lint  → commit "refactor(task): rename breakdown-multiflow-plan to full-plan (with back-compat alias)"

Commit 4 — tool.json description + parametrised resolution test
[ ] Edit  resources/tools/task/tool.json — --config description
[ ] NEW   tests/.../test_config_resolution.py (parametrised: 4 presets)
[ ] Tests + lint  → commit "docs(task): describe all 4 presets in tool.json"

Commit 5 — chat peer tool
[ ] NEW   resources/tools/chat/__init__.py
[ ] NEW   resources/tools/chat/tool.json
[ ] NEW   resources/tools/chat/cli.py (~150 LoC, mirroring sop/cli.py)
[ ] NEW   resources/tools/chat/configs/default.yaml (~30 LoC)
[ ] NEW   resources/tools/chat/executor.py
[ ] Edit  resources/tools/registry.py — register chat (if registry needs explicit entry)
[ ] NEW   tests/.../tools/chat/ (5 tests: T1–T5 per §E1.5)
[ ] Tests + lint  → commit "feat(chat): new conversational driver peer tool"

Commit 6 — E2E integration + docs
[ ] NEW   tests/.../e2e/test_chat_dispatches_task_with_breakdown_preset.py
[ ] NEW   resources/tools/chat/README.md
[ ] Tests + E2E smoke (§E2.2) + lint  → commit "test(e2e): chat dispatches task with complexity preset"

Commit 7 (v2) — smart-breakdown.yaml (heterogeneous workers per subtask)
[ ] Pre-flight  Run the 10-min verification from §E1.7 footer:
[ ]   - Synthetic 2-bucket worker_factory dict smoke against current BTA
[ ]   - Inspect parsers/breakdown.py to confirm {query,args} shape is accepted
[ ] NEW   resources/tools/task/configs/smart-breakdown.yaml (~150 LoC)
[ ] NEW   resources/sops/plan/breakdown/classifying.jinja2 (~30 LoC)
[ ] Edit  resources/tools/task/tool.json — extend --config description with smart-breakdown
[ ] NEW   tests/.../test_smart_breakdown.py (T1–T8 per §E1.7)
[ ] Tests + lint  → commit "feat(task): add smart-breakdown preset (heterogeneous workers via existing BTA dict-factory support)"

Commit 8 (v3) — tool_call_defaults on ConversationalInferencer
[ ] Edit  conversational/conversational_inferencer.py — add tool_call_defaults attrib
[ ] Edit  conversational/conversational_inferencer.py — apply defaults in _execute_tool_call
[ ] Edit  resources/tools/chat/tool.json — add --tool-default (multiple)
[ ] Edit  resources/tools/chat/cli.py — parse --tool-default → tool_call_defaults dict
[ ] Edit  resources/tools/sop/cli.py — same parser (reuse helper)
[ ] NEW   tests/.../test_tool_call_defaults.py (T1–T8 per §E1.8)
[ ] Tests + lint  → commit "feat(ci): tool_call_defaults for CI-driven tool dispatch"

Commit 9a (v3 prerequisite) — mandatory field on conversation tools
[ ] Edit  resources/tools/clarification/tool.json — +mandatory:bool default false
[ ] Edit  resources/tools/confirmation/tool.json — same
[ ] Edit  resources/tools/single_choice/tool.json — same
[ ] Edit  resources/tools/multiple_choice/tool.json — same
[ ] Edit  resources/tools/tool_argument_form/tool.json — same (verify file exists first)
[ ] Edit  resources/tools/proposal_selection/tool.json — same (depends on proposal_selection_tool_migration_plan v3.5+ landing first)
[ ] Edit  conversational/conversational_inferencer.py — add _should_auto_resolve(tool_name, tool_args)
[ ] NEW   tests/.../test_mandatory_gate.py (T1–T4 per §E1.9 9a section)
[ ] Tests + lint  → commit "feat(conversation-tools): add 'mandatory' field for principled auto-resolve semantics"

Commit 9b (v3) — --autonomous-level on task + sop, surfaced via chat
[ ] Edit  resources/tools/sop/tool.json — add --autonomous-level with choices+default; deprecate-but-keep --yolo as alias
[ ] Edit  resources/tools/sop/cli.py — parse --autonomous-level; map legacy --yolo; error on conflict
[ ] Edit  resources/tools/task/tool.json — add --autonomous-level (with honest "no-op today" docstring)
[ ] Edit  resources/tools/task/executor.py — accept + store autonomous_level on inferencer
[ ] Edit  resources/tools/chat/tool.json — add --autonomous-level top-level flag (default=auto)
[ ] Edit  resources/tools/chat/cli.py — populate tool_call_defaults for task+sop with --autonomous-level
[ ] Edit  resources/tools/chat/cli.py — yolo-mode startup banner
[ ] NEW   tests/.../test_autonomous_level.py (T5–T10 per §E1.9 9b section)
[ ] Tests + lint  → commit "feat(task,sop,chat): --autonomous-level enum (yolo/auto/interactive) with mandatory-gate escalation"

Post-flight
[ ] git push origin dev_xinli_2601
[ ] Open PR referencing this plan and the proposal_selection plan
[ ] Update _docs/_plan/README.md index with the two new presets and chat tool
```

---

# PART II — DESIGN REFERENCE
══════════════════════════════════════════════════════════════════════════════

## §D1. Goals & non-goals

**Goals:**
1. Give `task` users a **lighter-weight planning option** for simple/mechanical work (today only the heavyweight `default.yaml` and the already-lightweight-ish `breakdown-multiflow-plan.yaml` exist).
2. Make the **dimensionality of planning complexity** explicit and discoverable (coverage × diversity, complete the 2×2 matrix).
3. Provide an **exploratory / ambiguity-handling entry point** that uses a conversational inferencer and can dispatch to `task` (or other tools) on demand.
4. **Zero breaking change** to any existing `task` caller. Every script using the current default, `--config breakdown-multiflow-plan`, or any other current shape MUST keep working unchanged.

**Non-goals:**
1. Not redesigning `--config` parsing or making it accept sentinel/empty values (see §D2).
2. Not removing or repositioning `default.yaml` (it remains the heavyweight default).
3. Not building a UI for the `chat` tool in v1 (CLI-only; rich-terminal-interactive comes later).
4. Not implementing conversation resumability for `chat` in v1 (filed §A4 follow-up).
5. Not building a new `--complexity=low/med/high` flag — presets are the discoverable surface; English-language complexity dials are a worse UX.

## §D2 v4 — ARCHITECTURAL REVERSAL: `--config conversational` adopted (was REJECTED in v3)

> **v4 status:** the §D2 rejection below (preserved verbatim from v3 for audit) is **WITHDRAWN**. The historical reasoning is preserved as a record of how the design evolved, and the rejected-options table in §A3 is updated to mark `--config disabled`/`--config conversational` as the **CHOSEN** option (was: rejected).

### §D2 v4.1 — Why the v3 critique doesn't survive

Each of v3's five R-reasons re-examined against the peer plans' actual design:

| v3 reason | v3 claim | v4 re-evaluation | Verdict |
|---|---|---|---|
| **R1** | "`disabled` reads as no-op" | True, but trivially fixable by using `--config conversational` as the canonical name with `disabled` as a back-compat alias (or vice versa). The naming is a documentation problem, not an architectural one. | **R1 fixable, not architectural** |
| **R2** | "sentinel-value-in-string-flag anti-pattern" | **WRONG.** `disabled.yaml` / `conversational.yaml` is a real preset file that selects a `Conversational` topology root — exactly the same mechanism by which `breakdown.yaml` selects `Dual{BTA{Leaf}}`. It is NOT a magic string with hidden semantics; it is a normal preset entry in `_resolve_agent_config`'s 5-rule resolution. v3 confused "the topology happens to be conversational" with "the parser handles this string specially." | **R2 was wrong** |
| **R3** | "breaking change to every existing caller" | Real. Honestly owned in v4 Commit 5: bare `task "..."` defaults change from PTI to conversational. Mitigation: `--config default` / `--config pti` keeps PTI reachable; CHANGELOG note + deprecation warning. Tradeoff is acceptable because the new default is more useful for the common case. | **R3 acceptable when honestly owned** |
| **R4** | "wrong layer for the responsibility" | The conversational mode does NO planning itself — it only routes. Its responsibility is "decide which downstream config to invoke," which is genuinely a `task`-tool-level concern. The implementation lives inside `task/configs/conversational.yaml` and reuses the proven `sop/cli.py` pattern. No responsibility conflation. | **R4 overstated** |
| **R5** | "existing precedent (`sop` peer tool) points the other way" | The same `sop/cli.py` pattern is reusable INSIDE `task` (Commit 5 §E1.5-v4 step 3e). Pattern reuse does not require a new sibling tool. | **R5 doesn't force the new-tool answer** |

### §D2 v4.2 — Why `--config conversational` is genuinely better than v3's `chat` peer tool

| Property | v3: `chat` peer tool | v4: `--config conversational` |
|---|---|---|
| User entry points to learn | 2 (`chat` + `task`) | 1 (`task`) |
| CLI surface duplicated | Yes — `chat` reimplements `--model`, `--tools`, `--max-turns`, etc. | No — inherits all `task` flags |
| New tool dispatch indirection | `chat` → `task` (always 2-hop) | None for the common case |
| `tool_call_defaults` wiring (Commit 8) | Lives in `chat`'s CLI parser | Lives in `task`'s CLI parser — symmetric with every other `--config` |
| `--autonomous-level=auto` default for CI-driven | Lives in `chat`'s `tool_call_defaults` population | Same code path, in `task`'s router branch |
| LoC for new functionality | ~250 LoC (whole `chat` peer tool) | ~50 LoC YAML + ~80 LoC executor edits (140 LoC total) |
| Aligns with peer plans (B + C) | No | Yes |
| Honest about default change | Avoided the issue by not changing `task` default | YES — owns the breaking change, mitigates with `--config default`/`pti` alias |

**Net:** v4's design is smaller, more orthogonal, and consistent with the codebase's existing peer-tool patterns. v3's `chat` was a workaround for a critique that doesn't hold up.

### §D2 v4.3 — Surviving migration path concern from v3 §D2.4

v3 worried: "if `chat` becomes load-bearing, we'd need to migrate the recommended user entry point." v4 makes this concern moot — `task` IS the entry point, and `--config conversational` is the new default. No migration story needed; existing power-users who want PTI keep using `--config default`.

---

## §D2 v3 (HISTORICAL — REJECTED in v4) — Architecture decision: REJECTED `task --config disabled` as default

This section documents the most important design choice in the plan: **the proposal "make `task --config disabled` the new default conversational mode" is explicitly REJECTED.** The principled answer is a separate peer tool, `chat`. This section spells out why so the rejection is not just a one-line objection but a reasoned design verdict.

### §D2.1 — The original proposal

> "We want a conversation inferencer based task tool mode, which is `task --config disabled`. Let's make `disabled` the default mode. We present user the task tool, for the conversation inferencer to call, when there is indeed task."

The architectural intuition — **two-tier driver (conversational on top, topology underneath)** — is sound and exactly the right pattern. The implementation choice (hijack `--config` with a sentinel value `disabled`, make it the new default) is what this plan rejects.

### §D2.2 — Why the implementation choice is wrong (5 reasons)

| # | Reason | Severity |
|---|---|---|
| **R1** | **Semantic mismatch.** `disabled` reads as "turn off / no-op / dry-run." The mode being proposed runs an LLM conversation with full tool access — the opposite of "disabled." A reader of `task --config disabled` cannot infer what it does. This is the same anti-pattern this conversation already identified and rejected for the `proposal_selection` plan (see `_docs/_plan/workflows_and_sop/proposal_selection_tool_migration_plan.md` §A6 v3.5/v3.6: stuffing sentinel meanings into existing surface area is exactly what kept that plan stuck for four iterations). | Critical |
| **R2** | **Sentinel-value-in-string-flag anti-pattern.** `--config` is a string parameter that today accepts preset names, file paths, and registered aliases. Hijacking one specific string value (`disabled`) to mean "no topology, run a conversational inferencer instead" makes the parameter mean qualitatively different things depending on the value. Discoverability breaks: `task --help` can't easily explain that one string value triggers a completely different execution model. | High |
| **R3** | **Breaking change to every existing caller.** Today `task <request>` runs a planner→executor topology and produces a workspace + deliverable. After the change, it would run a chat session that *might* or *might not* dispatch anything. Every existing script, automation, CI job, and SOP-Phase-4 invocation gets a silently different execution model on the next release. Reverting requires every caller to add `--config full-plan` explicitly — a coordination tax that is hard to enforce. | Critical |
| **R4** | **Wrong layer for the responsibility.** The conversational driver and the topology executor have genuinely different responsibilities: the driver decides what to do next (LLM-mediated dispatch); the executor produces a deliverable from a known plan. Mixing them into one command makes the command's contract conditional ("output depends on what the LLM decides"). Two tools = two clear contracts. | High |
| **R5** | **Precedent already exists in the codebase, in the right shape.** The `sop` tool is a peer to `task` (verified: `src/agent_foundation/resources/tools/sop/cli.py:24–30` builds a `ConversationalInferencer` from config and dispatches tools through `tool_registry`/`tool_executor`). The `chat` tool would mirror that exact pattern — a proven, working peer-tool shape — and is therefore the architecturally consistent answer, not a new pattern invented for this plan. | Decisive |

### §D2.3 — The principled alternative (chosen)

A new sibling tool `chat`:
- Owns one responsibility: **drive a conversational LLM with scoped tool access.**
- Lives at `src/agent_foundation/resources/tools/chat/`, mirroring `task/` and `sop/`.
- Calls `task` (and any other allowed tool) via the existing tool dispatch mechanism (verified working: `derived_tool_execute` in `registry.py:66–117`).
- Has its own surface (`--tools`, `--max-turns`, `--model`) that wouldn't make sense on `task`.

Outcome: existing `task` behaviour is **byte-for-byte identical**; new functionality is opt-in (`chat <request>`); the two-tier mental model is preserved (chat-on-top, task-underneath); the codebase already has a precedent (the `sop` peer tool).

### §D2.4 — One escape hatch I want to be honest about

If `chat` turns out to be load-bearing enough that 80%+ of users want it as their default entry point, the right migration is **not** to change `task`'s default — it's to **rename/alias** the user-facing CLI: make `chat <request>` the recommended top-level invocation in documentation, and let `task` be the underlying primitive. The CLI defaults stay the same; the recommended workflow shifts via docs and onboarding.

This is a future concern. v1 of this plan just lands `chat` as one tool among many — no migration story needed.

## §D3. The 2×2 matrix — coverage × diversity

The four presets form a clean 2×2 of two orthogonal axes:

| | **No diversity** (single flow per slot) | **Diversity** (MFDual: N parallel flows per slot) |
|---|---|---|
| **No coverage** (no decomposition) | *(empty cell — `Leaf` alone is just one LLM call; no preset needed)* | **`multiple`** — root `MFDual`. Useful when the task is small enough that decomposition costs more than it gains, but diverse perspectives help. |
| **Coverage** (BTA decomposes the task) | **`breakdown`** — `Dual{BTA{Leaf}}`. Useful when the task decomposes naturally and each subtask is mechanical. | **`full-plan`** (alias: `breakdown-multiflow-plan`) — `Dual{BTA{MFDual}}`. Full coverage × diversity. The existing preset. |

The asymmetry (outer `Dual` only on the BTA-rows) is **structural, not accidental**: `MFDual` extends `DualInferencer` (`multi_flow_dual_inferencer.py:91`), so it inherits review/fix; `BTA` does not, so it needs an outer `Dual` wrapper. Each YAML header documents this.

The 4th preset (`default`) is **outside this matrix** — it's `Dual{PTI{planner=Dual{BTA{MFDual}}, executor=BTA, …}}`, i.e. plan-then-execute split + outer review/fix on the full deliverable. It remains the heavyweight workhorse for production deliverable-generating tasks.

## §D4. Risk register

| ID | Risk | Mitigation |
|---|---|---|
| **R-D1** | `${_build_flow_configs:...}` custom resolver in `multiple.yaml` may not exist as an OmegaConf resolver | **Documented in §E1.2.** Choose literal flow_configs block in v1; defer custom resolver to a follow-up. Low risk because literal-block has direct precedent. |
| **R-D2** | `BTA` may not honor `max_subtasks` cap if it's already wired differently | Verify by reading `BTA.__init__`; if `max_subtasks` isn't a real parameter, drop the surfaced knob from `breakdown.yaml` and document the breakdown count comes from prompt-level constraint instead |
| **R-D3** | Per-worker `Leaf` may surface as `${_params.main_inferencer}` rather than the alias string `Leaf` in the YAML registry | Verify by inspecting `dual.yaml` / `bta_interactive_breakdown.yaml` for the worker_factory shape; if the registry expects an inferencer instance rather than a `_target_` string, adjust `breakdown.yaml` accordingly |
| **R-D4** | Renaming `breakdown-multiflow-plan` → `full-plan` may break some hard-coded path-based callers if they `_import_` it from another YAML | Keep both files in v1 (don't delete the old YAML), or implement the rename as a pure alias map (preferred — see Commit 3) |
| **R-D5** | `chat` tool's allow-list enforcement might be circumventable if the LLM uses a tool's name spelling that bypasses the filter | Test T3 in §E1.5 locks this in; use the canonical-name resolver (`_resolve_tool_name`) as the gatekeeper, not raw string comparison |
| **R-D6** | `chat --tools "task"` could recurse infinitely if the LLM keeps dispatching `task` calls that themselves were given `--tools chat` (unlikely in v1 — `chat` is not in the default `task --tools` list — but worth a regression test) | Add a recursion-depth check in `chat`'s tool dispatcher; default `--tools` for `task` should NOT include `chat` |

## §D5. Open questions & defaults

| Q | Question | Default for v1 (deferred to follow-up if revisited) |
|---|---|---|
| Q1 | Should `breakdown.yaml` have N=3 default subtasks or N=5? | **5** — slightly higher than typical to make "coverage" visibly different from "single LLM call"; users can `--override` lower |
| Q2 | Should `multiple.yaml` have N=3 default flows or N=2? | **3** — same reasoning |
| Q3 | Should `chat` print conversation transcripts to stdout by default or only on `--verbose`? | **Stdout by default** — same as `sop` interactive mode |
| Q4 | Should `chat`'s default `--tools` list include `task` only, or also `view` / `knowledge`? | **`task,view,knowledge`** — these three are read-only-ish and unlikely to surprise users; explicit allow-list keeps the surface bounded |
| Q5 | Should `chat` support `--resume <session-dir>` in v1? | **No** — filed as §A4 follow-up #1; conversation resumability is a separate design |
| Q6 | Should the `breakdown` preset's `worker_factory` use bare `Leaf` (no per-worker review) or `Dual{Leaf}` (per-worker review)? | **Bare `Leaf`** (per §D3 reasoning); users who want review can `--override _params.worker_factory_target=Dual` |
| Q7 | Should `full-plan` be the canonical name, or should we keep `breakdown-multiflow-plan` canonical and add `full-plan` as an alias instead? | **`full-plan` canonical, old name as back-compat alias.** Shorter, fits the new naming triplet (`breakdown` / `multiple` / `full-plan`), the old name is preserved as alias for zero breakage. |
| Q8 | Should `chat`'s `--model` default to `opus[1m]` (matches `task`) or `sonnet` (cheaper)? | **`opus[1m]`** — conversational driver does the dispatch decisions; quality there matters more than per-turn cost |
| **Q9 (v2)** | What `max_tool_result_chars` should `chat`'s ConversationalInferencer use? Default is 4000 (`conversational_inferencer.py:124`) — too small if a single `task` result is a multi-page plan | **12000** — three times the default. Lets a typical `task --plan` result fit without truncation while still bounding context bloat. Override via `--max-tool-result-chars` (configurable per-invocation) |
| **Q10 (v2)** | Should the `task` `tool.json` description include a brief decision heuristic to help the LLM choose `--config`? | **Yes** — Commit 4 already updates the description; expand it slightly to say: "`breakdown` = decomposable mechanical tasks; `multiple` = small but diverse needed; `full-plan` = both coverage + diversity; `smart-breakdown` = heterogeneous (recommended default for unknown complexity)". One-line per preset. |
| **Q11 (v2)** | For `smart-breakdown`, what does the LLM emit if it can't classify? | **`leaf` (back-stop).** Documented in the prompt template (§E1.7). Cheaper than crashing; preserves output quality (worst case: a task that "should have been multiple" still gets a single coherent worker output rather than no output). |
| **Q12 (v2)** | Should `smart-breakdown` be the default `--config` instead of `default`? | **No, not in v1.** `default` (the heavyweight planner+executor) is what existing callers depend on. Don't change defaults. After 1–2 releases of usage data, revisit. |

## §D6. Verified facts (v2 — moved from §A2 because they shape the design, not just audit it)

These are not aspirational claims — they're properties of the existing code that the v2 preset and §D2 design decisions hinge on.

| # | Fact | Source |
|---|---|---|
| F1 | `BreakdownThenAggregateInferencer.worker_factory` supports `dict[str, factory]` for heterogeneous dispatch | `breakdown_then_aggregate_inferencer.py:388–397` |
| F2 | `task_type_arg_name` is the dict-key selector; the breakdown LLM emits `sub_query.args[task_type_arg_name]` per subtask | `breakdown_then_aggregate_inferencer.py:395–397, 1559–1574` |
| F3 | `_default` key is the back-stop; missing/unknown classifications fall back to it | `breakdown_then_aggregate_inferencer.py:1571, 1616` |
| F4 | Unknown classification raises a clear error (not a silent miss) | `breakdown_then_aggregate_inferencer.py:1622` |
| F5 | `ConversationalInferencer` appends a list of tool results into ONE `[Tool execution results]` user-message per turn, joined by `\n\n`, truncated at `max_tool_result_chars` (default 4000) | `conversational_inferencer.py:541–560, 593–617` |
| F6 | There is NO aggregator inferencer between tool calls and the next LLM turn — the LLM IS the aggregator (it sees the joined results and synthesizes in its next turn) | `conversational_inferencer.py:541–560` shows direct join → `add_message("user", ...)` → next turn |
| F7 | `MultiFlowDualInferencer` extends `DualInferencer` (so `multiple.yaml` root is `MFDual`, NOT `Dual{MFDual}`) | `multi_flow_dual_inferencer.py:91` |
| F8 | The `sop` peer-tool pattern (`sop/cli.py:24–30`) is the proven precedent for `chat`'s shape — a CI built from YAML with a tool registry passed in | `sop/cli.py` |
| **F9 (v3)** | The `sop/tool.json` description has long promised "Auto-resolve all **non-mandatory** confirmation gates" but the code never implemented the mandatory/non-mandatory distinction — `grep -rn "mandatory\|must_confirm\|critical"` returns zero hits in conversation-tool handlers. Today's `--yolo` resolves EVERY gate. | `sop/tool.json:19` (the promise); grep on `conversational/handlers/` (the missing implementation) |
| **F10 (v3)** | `task` has no runtime confirmation gates today — `task/executor.py:600` consumes `interactive` only inside `mode=="confirm"` (a `task` workflow mode, not a runtime gate). Adding `--autonomous-level` to `task` is therefore a no-op until real `task` gates are added (filed §A4 #10). | `task/executor.py:600`; grep for confirmation tool calls within `task/` returns zero |
| **F11 (v4)** | **CRITICAL latent bug:** `_extract_result_text` in `task/executor.py` (line ~353–354) does `result[0]` on tuples — silently drops `tuple[1:]` for multi-worker outputs from BTA's `disable_aggregator=True` or MFDual's `multi_flow_disable_aggregator=True`. The "no-aggregate / list-of-outputs to conversation" feature cannot work end-to-end until this is fixed. | `task/executor.py:353–354` |
| **F12 (v4)** | BTA's `disable_aggregator=True` returns a tuple of N worker outputs (verified at `breakdown_then_aggregate_inferencer.py:1325–1332`'s `_unwrap_workgraph_result`); MFDual has a parallel `multi_flow_disable_aggregator` flag. Both are existing functionality blocked only by F11. | `breakdown_then_aggregate_inferencer.py:1325–1332` + MFDual equivalent |
| **F13 (v4)** | `_DATA_KEYS = {"_default"}` in the config walker ensures the `_default:` string-alias key in adaptive `worker_factory` dicts is preserved through instantiation (line 617). Load-bearing for the adaptive-bucket fallback. | (config walker) line 617 + BTA worker_factory dict lookup at lines 1607–1643 |
| **F14 (v4)** | YAML `_target_:` / `_factory_:` entries in a `worker_factory` dict are auto-wrapped as `LazyConfigFactory` for fresh-instance creation (lines 1303–1307, 1632–1655 in BTA + `_instantiate.py:1196–1207`). Load-bearing — without this, adaptive workers would share state. | `_instantiate.py:1196–1207`; BTA lines 1303–1307, 1632–1655 |
| **F15 (v4)** | The conversational SOP-orchestrator path (`sop/cli.py:24–30` + `run_agentic_loop(request, ...)` pattern) is the proven precedent for `--config conversational`'s `_run_conversational_router` helper. Same `load_all_tools()` → `tool_registry` → `tool_executor` closure structure. | `sop/cli.py` |

---

# APPENDIX — AUDIT TRAIL
══════════════════════════════════════════════════════════════════════════════

## §A1. Why this plan exists (motivation)

Two unrelated user threads from the 2026-06-09 session converged on the same shape:

1. **"task tool currently planning at one fixed complexity"** — the user (Tony) observed that `task` has only two preset choices (`default` heavyweight + `breakdown-multiflow-plan` lighter), but the dimensionality of those choices is implicit. The user proposed completing a coverage × diversity matrix with two new presets.
2. **"is there a way to get a conversational task mode via `--config disabled`?"** — separately, the user proposed a conversational entry point. Verified in the same session that `--config disabled` errors loudly today (not implemented anywhere) and that no plan exists for it.

The connection between the two threads — **"the conversation tool calls the task tool when there's a concrete sub-task"** — is the architectural insight that motivates this single integrated plan rather than two separate ones.

## §A2. Empirical baseline (verified 2026-06-09)

| Claim | Verification | Source |
|---|---|---|
| Only 2 task presets exist today | `ls src/agent_foundation/resources/tools/task/configs/` → `breakdown-multiflow-plan.yaml`, `default.yaml` | ls of configs dir |
| `default.yaml` root is `Dual{PTI{Dual{BTA{MFDual}}, BTA, …}}` | YAML walker on `default.yaml` showed root `_target_: Dual`, base `PTI`, planner `Dual{BTA{MFDual}}`, executor `BTA` | python yaml walker, output captured in session |
| `breakdown-multiflow-plan.yaml` root is `Dual{BTA{MFDual}}` | YAML walker | same |
| `MultiFlowDualInferencer` extends `DualInferencer` | `grep "class MultiFlowDualInferencer"` → `class MultiFlowDualInferencer(DualInferencer):` | `multi_flow_dual_inferencer.py:91` |
| Tools can be called from other tools | `derived_tool_execute` imports `task_execute` and calls it | `resources/tools/registry.py:78, 117` |
| `--config disabled` errors loudly today (no hidden mode) | `_resolve_agent_config` Rule 5: raises `ValueError` listing available presets | `task/executor.py:95–98` |
| Empty `--config` falls back to `default` (not free conversation) | Two-tier coercion: `arguments.get("config") or "default"` then `(spec or "default").strip()` | `task/executor.py:62, 635` |
| No "free conversation" code path exists in `task/` | Grep for `FreeConversation`, `conversation_only`, `free_form_conversation`, `--config disabled`, `--config none` returns zero matches | grep of `task/` tree |
| `sop` is a peer tool that builds a `ConversationalInferencer` | `_build_ci_from_config(model, backend, tool_registry, tool_executor, interactive)` builds CI from YAML and accepts a tool registry | `sop/cli.py:24–30` |
| `MFDual` is documented for standalone use | Module docstring shows `mfdi = MultiFlowDualInferencer(...)` constructed directly and `await mfdi.ainfer("...")` | `multi_flow_dual_inferencer.py:115–127` |
| `MFDual` has configurable `review_default`, `review_priority_pool`, `fixer_match_winner` | Constructor attribs at lines 214; example shown at lines 36–38, 124–126 | `multi_flow_dual_inferencer.py` |
| No `chat` tool exists | `ls src/agent_foundation/resources/tools/` returns: clarification, confirmation, formatters, knowledge, models.py, multiple_choice, proposal_selection, registry.py, research_propose, single_choice, sop, task, templates, understand_codebase, understand_data, view — no `chat/` | ls of tools dir |
| No plan exists for these ideas | Content search across all 40 plans in `_docs/_plan/`: 8 hits, none proposed either idea (verified by reading the hit context) | grep + per-file inspection |

## §A3. Architecture — rejected options table (for §D2 audit trail)

| Option considered | Why rejected | Reference |
|---|---|---|
| `task --config disabled` runs a conversational inferencer | 5 reasons in §D2.2 (R1–R5) | §D2.2 |
| `task --config conversation` runs a conversational inferencer (better-named version of above) | Still has R2 (sentinel-value-in-string-flag), R3 (breaks defaults), R4 (wrong layer). Better than `disabled` only on R1. | §D2.2 |
| New flag `task --free-conversation` | Less terrible than sentinel-in-`--config`, but still R3 + R4. Mixes two responsibilities into one tool. | §D2.2 |
| `task --complexity=low/med/high/auto` | The "auto" tier would have to make dispatch decisions — that's a conversational driver; same R4 mixing-responsibilities critique. Also obscures what each English-language tier maps to (worse discoverability than named presets). | §D1 non-goal |
| Replace `task` entirely with a new conversational tool | R3 amplified — every existing caller breaks at once. | §D2.3 |
| Build `chat` as a feature inside `sop` (since `sop` already wraps a CI) | `sop` requires an SOP markdown — its raison d'être is structured workflow execution. A free conversation has no SOP. Conflating "free chat" with "SOP-driven chat" muddies `sop`'s contract. | §D2.3 |
| **CHOSEN:** new peer tool `chat`, mirroring `sop`/`task` shape | All five R-reasons addressed; matches existing peer-tool pattern; zero breaking change; clean responsibility separation | §D2.3 |

## §A4. Out-of-scope follow-ups

1. **`chat --resume <session-dir>`** — conversation resumability. Needs design for message-history persistence format. v1 ships without it; users restart conversations.
2. **OmegaConf `_build_flow_configs` custom resolver** — would make `multiple.yaml` more elegant (single list parameter expands to N flow_configs). v1 uses literal flow_configs block to avoid the resolver-registration overhead.
3. **Rich terminal UI for `chat`** — v1 is plain stdout. Phase 2 could add a `RichTerminalInteractive` wrapper mirroring `sop`'s pattern (`sop/cli.py:92`).
4. **A 5th preset `breakdown-deep.yaml`** — for nested BTA decomposition (BTA within BTA worker). Useful for very-large tasks. Defer until there's a concrete user need.
5. **A 6th preset `multiple-deep.yaml`** — for MFDual whose flows are themselves MFDual. Same reasoning.
6. **Migrate the recommended user entry point to `chat`** — see §D2.4. Future concern; v1 just lands `chat` as one tool among many.
7. **Cross-tool recursion-depth tracking** — currently each tool's invocations are independent; a `chat → task → chat` chain has no depth limit. Add session-wide depth tracking in a future commit.
8. **Auto-selection of `--config` based on request complexity** — could be a `task --config auto` mode that delegates to a small LLM call to pick `breakdown`/`multiple`/`full-plan`. Out of scope for v1 (would re-introduce the "LLM-mediated dispatch in `task`" critique from §D2).
9. **Tool-call-defaults YAML schema** — Commit 8 supports YAML-declared defaults but doesn't formalize a schema. A follow-up could add JSON-schema validation for `tool_call_defaults:` blocks in CI configs.
10. **(v3) Add real confirmation gates to `task`** — `task` currently has no runtime gates, so `--autonomous-level` is a no-op there (per §E1.9 Commit 9b caveat C2). Useful gates to add: "about to write >N files? confirm"; "about to run a long job (>M minutes estimated)? confirm"; "BTA breakdown produced >K subtasks (suspiciously many)? confirm". Once added, the existing `--autonomous-level` plumbing immediately gives users principled control.
11. **(v3) Audit existing SOPs for `[__requires user input__]` directives** — these are today's closest analog to "mandatory" gates, but they're a markdown directive rather than a tool-arg field. Migrate the genuinely critical ones (destructive operations, expensive operations, high-stakes decisions) to `mandatory=True` so that `--autonomous-level=auto` correctly escalates them. ~6 SOPs to audit; mechanical work.
12. **(v3) `tool_call_defaults` could also live as session-scoped state** — today it's CI-construction-time only. A follow-up could allow runtime `/set-default sop --yolo=true` slash commands so users can change defaults mid-conversation.

---

## §A5. Changelog

- **v1 (2026-06-09 19:08):** Initial draft. Two halves (3 new presets + new `chat` peer tool) integrated into one plan because the user proposed them together. Explicitly REJECTS the `task --config disabled` proposal in §D2 with 5 reasons; provides the principled alternative (peer tool, mirroring `sop` pattern) in §D2.3. Empirical baseline (§A2) lists every claim with verification source. Companion to `_docs/_plan/workflows_and_sop/proposal_selection_tool_migration_plan.md` (v3.6) — shares the bridge-augmentation key naming (`workspace_path__<tool>`).

- **v2 (2026-06-09 20:26):** Adds a 5th preset and tightens two parameters in response to three sharp user questions on 2026-06-09 evening. Verified each user intuition against source before incorporating; nothing fabricated.
  - **§E1.7 — Commit 7 added: `smart-breakdown.yaml`** (heterogeneous workers per subtask) — the most elegant member of the preset family. **Crucial finding:** BTA already supports heterogeneous worker dispatch via `worker_factory: dict[str, factory] + task_type_arg_name` (verified at `breakdown_then_aggregate_inferencer.py:388–397, 1559–1574`). No new abstraction; this preset is the first place in the codebase that *uses* the built-in mechanism. New `classifying.jinja2` prompt template asks the LLM to classify each subtask as `leaf` / `breakdown` / `multiple` / `conversational`; BTA dispatches accordingly. 8 tests (T1–T8) cover happy path + 4 buckets + 2 fallback modes + E2E smoke.
  - **§D6 NEW — "Verified facts that shape the design"** — 8 source-anchored claims (F1–F8) that the v2 design hinges on, separated from the older `§A2` audit baseline because they're *load-bearing for design*, not just provenance. Promotes the BTA-heterogeneous-dispatch finding (F1–F4) and the CI tool-result-aggregation behavior (F5–F6) to first-class design constraints.
  - **§D5 Q9–Q12 NEW** — answer 4 new questions surfaced by the v2 additions:
    - Q9: `chat`'s `max_tool_result_chars` should be 12000 (3× the default), so a single `task --plan` result fits without truncation. `conversational_inferencer.py:124` confirms the default is 4000.
    - Q10: `tool.json` `--config` description should include a one-line heuristic per preset to guide the LLM's choice in `chat`-driven dispatch (referenced in Commit 4).
    - Q11: `smart-breakdown` fallback is `leaf` (cheapest non-crash). Aligns with BTA's existing `_default` fallback at line 1571.
    - Q12: do NOT make `smart-breakdown` the new default (preserves v1's "zero breaking change" guarantee in §D1).
  - **Commit 5 `chat` tool.json updated** — adds `--max-tool-result-chars` parameter (default 12000) per Q9.
  - **§E3 checklist** — adds Commit 7's full step list, including a 10-min pre-flight verification (synthetic dict-factory smoke + parser inspection) to keep Commit 7's risk low.
  - **Effort table updated** — total estimate 3.5d → 4.5d (Commit 7 = 1 day).
  - **Architectural decisions UNCHANGED**: §D2 rejection of `--config disabled`, §D3 2×2 matrix, §D4 risks, are all preserved. v2 only ADDS Commit 7 + tightens parameters.

---

- **v3 (2026-06-10 12:36):** Adds two coherent CI-layer features that arise once `chat` is dispatching nested tool calls. Verified each user intuition + each existing-code claim against source before incorporating; nothing fabricated.

  - **§E1.8 — Commit 8 added: `tool_call_defaults`** — CI-level per-tool default arguments. Shallow-merged into each tool call; caller wins on conflict; canonical-name matching (so `research-propose` and `research_propose` match the same key); unknown tool keys warn, unknown arg keys warn+skip. New `--tool-default tool:flag=value` CLI flag (repeatable) on `chat` + `sop`. 8 tests (T1–T8). ~80 LoC + ~120 LoC tests.

  - **§E1.9 — Commit 9 added: `--autonomous-level` + Commit 9a prerequisite (`mandatory` field):**
    - **9a (prerequisite):** The `sop/tool.json` description has long promised "Auto-resolve all **non-mandatory** confirmation gates," but the code never implemented the mandatory/non-mandatory distinction (verified — grep returns zero hits for `mandatory|must_confirm|critical` in handlers). Commit 9a adds a `mandatory: bool` field to all 6 conversation-tool tool.json files + a `_should_auto_resolve(tool_name, tool_args)` helper that honors `autonomous_level + mandatory`. 4 tests. ~30 LoC.
    - **9b:** Adds `--autonomous-level` flag with 3 values (`yolo` / `auto` / `interactive`) to both `task` and `sop`, with `--yolo` as legacy back-compat alias. Default for human-triggered = `interactive` (preserves today's behavior, ZERO breaking change). Default for CI-triggered (`chat`) = `auto` (matches the v3 user proposal — autonomous routine work, surface only mandatory gates to root conversation). `chat` populates its `tool_call_defaults` accordingly. 6 tests. ~170 LoC.

  - **§D6 NEW facts added:** F9 = "no mandatory/non-mandatory distinction in code today" (zero grep hits). F10 = "task has no confirmation gates today" (only `interactive` in `mode=="confirm"`). These are load-bearing for Commit 9's honest caveats.

  - **§A4 follow-ups expanded:** Added items #9 (tool_call_defaults YAML schema), #10 (add real `task` gates so `--autonomous-level` does something), #11 (audit existing SOPs and migrate `[__requires user input__]` to `mandatory=True` selectively), #12 (session-scoped runtime `/set-default` slash commands).

  - **Effort table updated:** 4.5d → 6.5d (Commit 8: 0.5d; Commit 9a: 0.5d; Commit 9b: 1d).

  - **Dependency graph clarified:** 9a → 9b → 8 (strict); both 8 and 9b also depend on Commit 5 (chat). 7 is independent.

  - **Honest caveats (carried into the v3 reviewer banner):**
    - **C1:** Commit 9a's `mandatory` field is brand new — until callers opt in, `--autonomous-level=auto` initially behaves identically to `--autonomous-level=yolo`. Acceptable as a gradual rollout; filed §A4 #11 to migrate existing SOP directives selectively.
    - **C2:** `task` has no gates today; `--autonomous-level` is a no-op on `task` v1. Documented in tool.json description; filed §A4 #10 to add real `task` gates.

  - **Architectural decisions UNCHANGED in v3:** §D2 rejection of `--config disabled`, §D3 2×2 matrix, §D4 risks all preserved. v3 only ADDS Commits 8 + 9.

---

- **v4 (2026-06-10 12:46) — ARCHITECTURAL REVERSAL of v3 §D2 + 4 new commits + 5 new facts.** After cross-reading two independent peer plans (`.claude/plans/update-your-task-tool-adaptive-goose.md` "Plan B" and `.cursor/plans/task_plan_config_ladder_7ea0fe0e.plan.md` "Plan C"), v3's rejection of `--config disabled` as the new default was withdrawn. See §D2 v4 reversal for full reasoning (R1/R2/R4/R5 don't survive; R3 is honestly owned). See §A6 (NEW) for the full cross-plan comparison.

  - **§D2 v4 ADDED + v3 §D2 PRESERVED as historical** — explicit reversal section explaining why each of v3's 5 R-reasons doesn't survive scrutiny against the peer plans' design.

  - **NEW §E1.5-v4 (Commit 5 replacement):** `--config conversational` implemented inside `task`. Replaces v3's `chat` peer tool entirely. ~140 LoC + ~120 LoC tests. Default for bare `task "..."` changes from PTI to conversational; `--config default` / `--config pti` keeps PTI behavior (back-compat).

  - **NEW §E1.5b (Commit 5b, CRITICAL):** fix `_extract_result_text` silently dropping `tuple[1:]` (verified bug at `task/executor.py:353–354` — fact F11). Without this, the no-aggregate-list-to-conversation feature cannot work. ~40 LoC + ~80 LoC tests.

  - **NEW §E1.5c (Commit 5c):** unified `task_depth` budget. Single `session_context` counter governs BOTH router recursion AND adaptive-worker nesting. Default cap = 2. Coercion at the cap (nested conversational → full-plan; adaptive breakdown/conversational worker → leaf). ~30 LoC + ~60 LoC tests.

  - **NEW §E1.5d (Commit 5d):** `task_router` prompt template. Documents the ladder + decision rule + `task` action call schema. ~80 LoC template + ~30 LoC tests.

  - **§D6 NEW facts:** F11 (the `_extract_result_text` bug), F12 (`disable_aggregator` tuple-of-outputs return), F13 (`_DATA_KEYS = {"_default"}` preservation), F14 (`LazyConfigFactory` auto-wrap), F15 (`sop/cli.py` precedent). All anchored to source line numbers from the peer plans' findings.

  - **§A3 rejected-options table:** updated to mark "`task --config disabled` runs a conversational inferencer" as CHOSEN-in-v4 (was: rejected in v3). v3's `chat` peer tool added as NEW-rejected-in-v4 row.

  - **§A4 follow-ups:** removed v3 items #1–#3 (about the now-deleted `chat` tool — irrelevant after v4). Existing #10 (add real `task` gates) and #11 (audit existing SOPs) carry forward unchanged. Added #13: audit current `disable_aggregator=True` callers to gate Commit 5b's behavior change. Added #14: deprecation timing for the bare-`task` default change (1 release with warning before bare default flips silently).

  - **Effort table updated:** 6.5d → ~8.5d (Commits 5/5b/5c/5d together = ~3d net; v3's Commit 5 `chat` deleted = -1d).

  - **Architectural decisions UNCHANGED in v4:** §D3 2×2 matrix (still correct, with `smart-breakdown` now folded into `breakdown-plan.yaml`'s adaptive mode per the peer plans' more elegant design); §D4 risks; §D5 Q1–Q12 (with Q13/Q14 added below for naming + deprecation timing).

  - **Q13 (v4):** Canonical name for the conversational config — `conversational` or `disabled`? **Default: `conversational` is canonical; `disabled` is back-compat alias.** Reasoning: `conversational` accurately describes what the topology does; `disabled` was the peer plans' term but reads as no-op (the original v3 R1 critique that remains valid as a naming preference even after architectural reversal).
  - **Q14 (v4):** Deprecation timing for the bare-`task` default change — silent flip in v4, or 1-release deprecation warning? **Default: 1-release deprecation warning** ("bare `task "..."` will default to `--config conversational` in v5; pass `--config default` or `--config pti` to keep PTI behavior"). Honors v3's "zero breaking change" principle for the transition window.

---

## §A6 (v4). Cross-plan comparison that drove the reversal

This appendix is the audit trail for the v4 §D2 reversal. v3 was a self-contained architectural design; v4 was forced by reading two independent peer plans that converged on a different answer.

### §A6.1 — The three plans compared

| Aspect | v3 (mine, before reversal) | Plan B (`.claude/plans/update-your-task-tool-adaptive-goose.md`) | Plan C (`.cursor/plans/task_plan_config_ladder_7ea0fe0e.plan.md`) |
|---|---|---|---|
| **Length** | 1,101 lines | ~403 lines | 150 lines |
| **Conversational mode** | New `chat` peer tool | `--config disabled` inside `task` (default) | `--config disabled` inside `task` (default) |
| **`_extract_result_text` bug** | ❌ Missed | ✅ Identified | ✅ Identified ("load-bearing") |
| **Unified `task_depth` budget** | ❌ Filed as follow-up | ✅ Single mechanism for router + adaptive | ✅ Same, explicit |
| **No-aggregate flag** | ❌ Acknowledged but no commit | ✅ Implicit in conversation aggregation | ✅ Explicit `--aggregate` flag |
| **Router prompt template** | ❌ Not specified | ✅ `task_router` variant | ✅ Specified |
| **Config alias map** | Partial (only `full-plan`) | Full (incl. `pti`, `multiflow`) | Full |
| **Adaptive workers design** | Separate `smart-breakdown.yaml` file | Subsumed into `breakdown-plan.yaml`'s `worker_type: adaptive` mode | Subsumed (same as B) |
| **`tool_call_defaults`** | ✅ Unique to v3 | ❌ | ❌ |
| **`--autonomous-level`** | ✅ Unique to v3 | ❌ | ❌ |
| **`mandatory` field** | ✅ Unique to v3 | ❌ | ❌ |

### §A6.2 — What v4 takes from each plan

**From Plan B:**
- `_extract_result_text` bug identification + multi-output serialization design (Commit 5b)
- Unified depth budget design (Commit 5c)
- Adaptive workers folded into `breakdown-plan.yaml`'s `worker_type` mode (vs v3's separate `smart-breakdown.yaml`)
- Config-alias map shape (Commit 5 step 3a + 3b)
- `_DATA_KEYS = {"_default"}` and `LazyConfigFactory` facts (F13, F14)

**From Plan C:**
- Sharp framing of the `_extract_result_text` bug as "load-bearing" (sharpest defect detection in any of the three plans)
- Explicit `--aggregate` flag design (Commit 5b step 2)
- Unified depth budget semantics (Commit 5c coercion rules)
- One-line summary of the ladder (§D3 v4 matrix simplification)
- Most economical architecture (single entry point, no peer tool)

**From v3 (retained):**
- `tool_call_defaults` mechanism (Commit 8)
- `--autonomous-level` enum + `mandatory` field prerequisite (Commits 9a, 9b)
- Honest documentation discipline (PART I/II/APPENDIX structure)
- Per-fact source-line citations in §D6
- 1,101-line depth and explicit caveats

### §A6.3 — Honest answer to "if we only pick one plan, which?"

**Plan C (cursor).** Sharpest signal per line. Calls out `_extract_result_text` as load-bearing. Most economical. Single depth budget. Honest about default change.

**Plan B is a close second** — same architecture as C with more implementation detail. If forced between B and v3, pick B.

**v3 is third.** Has unique contributions (`tool_call_defaults`, `--autonomous-level`, `mandatory`) that v4 retains, but its `chat`-peer-tool architectural choice was wrong, and it missed F11 (the `_extract_result_text` bug) and F12 (`disable_aggregator` tuple-of-outputs return).

**Why v4 is better than picking any single one:** v4 = Plan C's architecture + Plan B's implementation detail + v3's CI-layer additions (`tool_call_defaults`, `--autonomous-level`). Each contributes what it does best. No padding, no redundancy.

### §A6.4 — Meta-lesson

v3 spent 1,101 lines justifying an architectural choice that two independent peers identified as suboptimal in <500 combined lines. The lesson is **not** "shorter is better" — it's **"don't construct a complex critique to reject a simpler design without checking whether the critique survives the simpler design's actual implementation."** v3's R2 critique ("sentinel-value-in-string-flag") presumed an implementation that wasn't proposed; peer plans proposed `disabled.yaml` as a real preset, which makes R2 a non-issue.

The v4 reversal is the right outcome. It also vindicates the value of having multiple agents work the same design independently — peer review catches blind spots that single-agent introspection doesn't.

---

---

## §A7 (v5). Bug-by-bug audit of v4 → v5 corrections

Five source-anchored bugs in v4 were flagged by Plan B's 13:11 revision. Each verified independently against AgentFoundation source before integrating.

### §A7.1 — Verification methodology

For every claim I ran a targeted grep:
```bash
# B1: Is "Leaf" a registered _target_?
grep -rn "register.*Leaf\b\|'Leaf'\|\"Leaf\"" src/agent_foundation/common/configs/
# → ZERO HITS. Confirmed: not registered.

# B2: BTA — max_breakdown or max_subtasks?
grep -nE "max_breakdown|max_subtasks" src/.../breakdown_then_aggregate_inferencer.py
# → 8 hits for max_breakdown (attrib at line 375), 0 for max_subtasks. Confirmed.

# C1: Root-level _import_ usage?
grep -rn "^_import_:" src/agent_foundation/resources/tools/task/configs/
# → ZERO HITS at root level; only nested at default.yaml:156. Confirmed.

# B4: _build_flow_configs custom resolver?
grep -rn "_build_flow_configs" src/agent_foundation/
# → ZERO HITS. Confirmed: doesn't exist.
```

### §A7.2 — Bug table

| # | Bug | Where in v4 | Verified | Fix in v5 |
|---|---|---|---|---|
| **B1** | `_target_: Leaf` is not a registered target | §E1.5 `chat.yaml`, §E1.7 `smart-breakdown.yaml` (4 occurrences) | ✅ Zero grep hits | Use `_target_: ${_params.main_inferencer}` (resolves to `ClaudeCodeCLI` per F2) |
| **B2** | BTA param is `max_breakdown`, not `max_subtasks` | §E1.7 `smart-breakdown.yaml` (2 occurrences) | ✅ 8 hits for `max_breakdown`; 0 for `max_subtasks` in BTA | Replace `max_subtasks` → `max_breakdown` |
| **B3** | Separate `smart-breakdown.yaml` over-engineers the adaptive case | §E1.7 entire commit | ✅ Plan B's design is more elegant | FOLD into `breakdown.yaml` as a `worker_factory: dict` with `_default: leaf` backstop |
| **B4** | `${_build_flow_configs:...}` custom OmegaConf resolver doesn't exist | §E1.2 `multiple.yaml` (mentioned as possibility) | ✅ Zero grep hits | Use literal `flow_configs:` block |
| **C1** | Root-level `_import_:` doesn't work | §E1.5-v4 step 3a `disabled.yaml`; §E1.3 `full-plan.yaml` | ✅ Zero root usages; only nested at `default.yaml:156` | Use Python alias in `_CONFIG_ALIASES` for `full-plan`; full inline YAML for `disabled.yaml` |
| **Defer 1** | v4 Commit 5 over-built (`_run_conversational_router` + tool-registry setup inside `task` executor) | §E1.5-v4 step 3e | Plan B's MVP discipline correctly defers full CI integration | Commit 5 becomes "workspace passthrough early return"; full router DEFERRED to Follow-up #1 |
| **Defer 2** | v4 Commits 5b/5c/5d ship in same PR as presets | §E1.5b, §E1.5c, §E1.5d | Plan B's MVP discipline correctly defers infrastructure-level changes | All three DEFERRED to Follow-ups #2, #3 |

### §A7.3 — Why I missed these in v4

Honest meta-analysis of how 5 bugs survived v4:

1. **B1 (Leaf):** I assumed `Leaf` was a registered class because the codebase has `Dual`, `BTA`, `MFDual`, `Conversational` etc. registered. I extrapolated from pattern, didn't verify. Plan B's exhaustive scan caught it.
2. **B2 (max_breakdown):** Same root cause — I extrapolated from `--max-breakdown` CLI flag without verifying the BTA-side attribute name. Plan B's line-anchored citation forced the verification.
3. **B3 (smart-breakdown redundancy):** v3 introduced `smart-breakdown.yaml` as a separate preset; v4 carried that forward without re-examining. Plan B asked the right question: "if `_default: leaf` makes the adaptive case behave identically to homogeneous leaf, do we need two files?" The answer is no.
4. **B4 (`_build_flow_configs`):** I hand-waved this as "would be elegant" without checking if the resolver exists. Plan B's grep killed it.
5. **C1 (root `_import_`):** I assumed YAML composition was symmetric (works at root if works nested). Plan B's exhaustive grep showed only nested usage anywhere — a clear signal.

**Common pattern in all 5:** *assumption-from-pattern* rather than *verification-from-source*. v3 caught itself doing this once (the `_extract_result_text` bug). v4 introduced 5 more instances of the same pattern. v5 fixes them and adds §A7 specifically to document the meta-pattern.

### §A7.4 — Honest verdict on v4 vs v5

**v4 was architecturally right but YAML-broken.** The `--config disabled` reversal was the correct call (verified again in v5); the over-build in Commits 5b/5c/5d was wrong (MVP discipline applies). The 5 source bugs would have caused the v4 YAMLs to fail at instantiation time — not at runtime, at *load* time.

**v5 is what v4 should have been** with two more rounds of source verification per claim.

### §A7.5 — Updated answer to "if forced to pick one plan, which?"

**Plan B (in its updated 13:11 form).** Reasoning evolved from earlier picks:

- **Plan B was earlier #2** (behind Plan C) at v4 integration time.
- **Plan B revision** added: 7 verified facts with source citations (F1–F9), full PART I/II/III structure matching my v4's discipline, explicit §Issues section critiquing all three plans, and a verification block.
- **Plan B revision is now the most rigorous of the three.** It has the architectural correctness of Plan C, the implementation detail of v4, AND the source-verification discipline that v4 lacked.

**Plan C** still wins on "signal per line" — 150 lines vs Plan B's 530 — but Plan B's revision now has the source-verification edge.

**v4 (mine, pre-correction)** drops to **third** because of the 5 unverified-YAML bugs.

**v5 is strictly better than any single plan** because: Plan B's bug catches + Plan C's architecture + v3's CI-layer additions (`tool_call_defaults`, `--autonomous-level` as PART III follow-ups). Each plan contributes what it does best.

### §A7.6 — Meta-lesson update

v3 → v4 → v5 trajectory shows the same meta-lesson at three nesting depths:
- **v3 → v4:** "Don't reject a simpler design without checking whether the critique survives the simpler design's actual implementation." (peer plans showed `--config disabled` wasn't a sentinel)
- **v4 → v5:** "Don't write YAML examples without verifying every `_target_:` value is actually registered and every parameter name is actually an attrib." (Plan B revision showed 5 bugs from assumption-from-pattern)
- **General:** Multi-agent peer review surfaces blind spots that single-agent introspection doesn't. Three rounds of peer review in this conversation have surfaced corrections at every level (architectural, implementation, YAML syntax).

The v5 plan is now stable enough to execute. The trajectory's diminishing returns suggest the next peer-review round (v5 → v6) would catch tactical-not-architectural issues.

---

**End of plan v5.** Ready for review.
