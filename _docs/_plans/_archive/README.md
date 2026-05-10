# Archived Plans

These plans are **superseded** and kept here for historical reference only.
Do NOT consult them for implementation guidance — they are obsolete.

## Active Plans

The currently active plans live in the parent directory:

- `../leaf_owned_template_rendering_INTEGRATED_plan.md` — **active** plan for the refactor that moves all template rendering to leaf inferencers and removes orchestrator-side `<role>_prompt` fields. 6 rounds of cross-agent audit (2026-05-09).
- `../dual_inferencer_path_aware_followup_INTEGRATED_plan.md` — **shipped** plan for the path-aware followup fix (Phase 0 of the leaf-rendering refactor). Implementation complete and tested.
- `../ai_employee_framework.md` — separate framework doc, unrelated to template rendering.

## What's Archived Here

| File | Why Archived |
|---|---|
| `leaf_owned_template_rendering_refactor_plan.md` | First draft of the leaf-rendering plan. Superseded by INTEGRATED version (which incorporated all valid feedback from cross-agent review). |
| `_alt_plan_leaf_rendering_splendid_lantern.md` | Alternate plan from a parallel agent. Eventually rewritten as a comparison-only document. All valid insights folded into the INTEGRATED plan. |
| `dual_inferencer_path_aware_followup_fix_plan.md` | First draft of the path-aware followup fix plan. Superseded by INTEGRATED version. |
| `_alt_plan_splendid_lantern.md` | Alternate plan from parallel agent for the path-aware fix. Insights folded into the INTEGRATED version. |
| `COMPARISON_Plan_A_vs_Plan_B.md` | Side-by-side comparison artifact from one of the integration rounds. Documented decision rationale; no longer needed once the INTEGRATED plan absorbed the conclusions. |

## Audit Trail Preserved

Each INTEGRATED plan has a `§9 Provenance` section that documents:
- Which prior plan(s) it draws from
- Each round of cross-agent audit (with dates and agent verdicts)
- What was changed in each round and why

If you need details on the journey from these archived plans to the active ones, read the Provenance section of the corresponding INTEGRATED plan.
