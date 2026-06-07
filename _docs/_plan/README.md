# AgentFoundation Plans Index

Reorganized: 2026-05-24. All plans now categorized by theme. The flat `_plan/` directory previously held 22 mixed-topic plans; they are now grouped into 6 thematic subfolders + the pre-existing `_archive/`.

> **Note:** The parallel `_docs/_plans/` (plural) directory has been emptied — its 5 files were consolidated here. You can `rmdir _docs/_plans` to remove the empty husk (the `delete_file` tool cannot remove directories).

---

## Folder map

### `inferencer_architecture/` (5 files)
The inheritance-axes refactor and the terminal-inferencer rename effort. Subject: `InferencerBase`, `TerminalInferencerBase`, `TerminalSessionInferencerBase`, `StreamingInferencerBase`, `TemplatedInferencerBase` relationships.

| File | Role |
|---|---|
| `inferencer_axes_INTEGRATED_v5_plan.md` | **Current canonical** (v5 — delta on top of v4) |
| `inferencer_axes_INTEGRATED_v4_plan.md` | Foundation for v5 |
| `inferencer_axes_INTEGRATED_v3_plan.md` | Historical |
| `terminal_inferencer_axes_and_streaming_rename_plan.md` | Companion: terminal-axes rename effort |
| `terminal_inferencer_axes_AUDIT_FINDINGS.md` | Audit appendix |

### `mfdual_bug_fixes/` (6 files)
`MultiFlowDualInferencer` (MFDual) bug investigations and fixes. Workspace anomalies, peer visibility, self-promotion gap, hygiene.

| File | Role |
|---|---|
| `mfdual_hygiene_INTEGRATED_plan.md` | Consolidated hygiene fixes |
| `mfdual_self_promotion_gap_INTEGRATED_plan.md` | Self-promotion gap fix |
| `mfdual_peer_visibility_path_aware_fix_plan.md` | Peer visibility |
| `mfdual_hollow_workspace_anomaly_7_fix_plan.md` | Anomaly #7 (hollow workspace) |
| `mfdual_workspace_layout_anomalies_fix_plan.md` | Largest layout-anomaly plan |
| `mfdual_workspace_anomalies_IMPLEMENTATION_REF.md` | Implementation reference doc |

### `orchestration_path_aware/` (7 files)
Path-aware orchestration fixes spanning `DualInferencer`, `LWI` (LeafWorkflowInferencer), `Aggregator`, `Orchestrator`, `PTI` preflight, and the prompt-echo defense port.

| File | Role |
|---|---|
| `orchestrator_path_aware_INTEGRATED_plan.md` | Orchestrator: integrated plan |
| `orchestrator_path_aware_outcome_passing_plan.md` | Outcome-passing addendum |
| `dual_inferencer_path_aware_followup_INTEGRATED_plan.md` | DualInferencer followup |
| `aggregator_dual_bug_unified_fix_plan.md` | Aggregator unified fix |
| `lwi_reflective_path_aware_fix_plan.md` | LWI reflective fix |
| `pti_full_mode_preflight_fix_plan.md` | PTI full-mode preflight |
| `prompt_echo_defense_port_plan.md` | Prompt-echo defense (DualInferencer ← RankEvolve port) |

### `templates_and_variables/` (1 file + 2 subfolders)
Prompt-template rendering, variable population, multi-dot variable resolution.

| Path | Role |
|---|---|
| `leaf_owned_template_rendering_INTEGRATED_plan.md` | Leaf-owned rendering refactor |
| `template_and_variable_versioning_formalization/` | Versioning formalization (8 files: current + 7 `.bak` snapshots) |
| `load_variables_multidot/` | `load_variables` multi-dot enhancement (4 files: v1 + v2/v3/v4 INTEGRATED iterations) |

### `ai_employee_and_agents/` (2 files)
AI Employee framework + ConversationalInferencer skill architecture.

| File | Role |
|---|---|
| `ai_employee_framework.md` | AI Employee framework design |
| `multi_workflow_as_skill_plan.md` | Multi-workflow-as-skill for ConversationalInferencer |

### `ui/` (1 subfolder)
UI components.

| Path | Role |
|---|---|
| `ui_components_formalization/` | UI components formalization (5 files: current + 4 `.bak` snapshots) |

### `workflows_and_sop/` (1 file — added 2026-05-24)
First-class workflows in `ConversationalInferencer` + SOP grammar v2 + `task` tool relocation + new `sop` tool + new `agent_foundation/ui/` CLI library. This is a cross-cutting plan that spans ai-employee, inferencer, templates, and ui themes — placed in its own folder rather than splitting across four locations.

| File | Role |
|---|---|
| `conversational_workflows_and_sop_framework_plan.md` | v1 draft — 14 sections covering SOP grammar v2 (EBNF), workflow runtime types (Definition / Instance / Thread), two-layer prompt rendering, task tool relocation with OpenStartup bridge shim, new `sop` tool, new `ui/` module (rich + prompt_toolkit), 10-phase rollout, 10 risks, 5 open questions |

### `_archive/` (8 files — pre-existing convention)
Older or superseded full plans. **Different semantics from the category folders:** `_archive/` holds plans that have been entirely replaced; the category folders hold active and historical revisions that may still be referenced.

Includes `_archive/README.md` documenting its own conventions.

---

## Naming conventions observed

- `*_INTEGRATED_plan.md` → consolidated/integrated version
- `*_fix_plan.md` → bug-fix plan
- `*_AUDIT_FINDINGS.md` → audit appendix to a plan
- `*_IMPLEMENTATION_REF.md` → reference doc, not a plan itself
- `*.md.bak` → older version snapshot (pattern used in the two existing subfolders)
- `_alt_plan_*.md` → alternative draft (only in `_archive/`)

---

## Disambiguation decisions made during reorganization

A few files crossed categories. Final placement and rationale:

| File | Could be | Placed in | Rationale |
|---|---|---|---|
| `prompt_echo_defense_port_plan.md` | templates_and_variables / orchestration | **orchestration_path_aware** | Primary scope is `DualInferencer` / `MultiFlowDualInferencer` (per its own §Scope); prompt templates are touched but not the primary subject |
| `multi_workflow_as_skill_plan.md` | inferencer_architecture / ai_employee_and_agents | **ai_employee_and_agents** | High-level skill architecture for `ConversationalInferencer`, not a base-class change |
| `dual_inferencer_path_aware_followup_INTEGRATED_plan.md` | inferencer_architecture / orchestration | **orchestration_path_aware** | Subject is path-aware orchestration behavior, not inferencer-base shape |
| `leaf_owned_template_rendering_INTEGRATED_plan.md` | orchestration / templates | **templates_and_variables** | Subject is template-rendering ownership boundary |
| `conversational_workflows_and_sop_framework_plan.md` | ai_employee_and_agents / inferencer_architecture / templates_and_variables / ui (genuinely cross-cutting) | **workflows_and_sop** (new folder) | Spans 4 existing themes; splitting would fragment a cohesive design. Folder is named after the dominant theme (workflows + SOP) so future v2/v3/audit iterations accumulate together, matching the `load_variables_multidot/` pattern. |

---

## What changed in this reorganization

- 21 of 22 flat files moved into 6 thematic folders (1 was already a subfolder — preserved)
- 2 existing nested subfolders (`template_and_variable_versioning_formalization/`, `ui_components_formalization/`) relocated under the new themes
- 4 `load_variables_multidot_*` plans consolidated from the parallel `_docs/_plans/` directory into `templates_and_variables/load_variables_multidot/`
- 1 byte-identical duplicate (`ai_employee_framework.md`) eliminated from `_docs/_plans/`
- `_archive/` untouched (different semantic role: superseded entire plans, distinct from active/historical revisions)
- This `README.md` index added

**Untouched intentionally:** every plan file's content, every `.bak` snapshot, the `_archive/` directory and its `README.md`.

**Manual cleanup remaining:** `rmdir CoreProjects/AgentFoundation/_docs/_plans` to remove the now-empty parallel directory (the `delete_file` tool only handles files, not directories).
