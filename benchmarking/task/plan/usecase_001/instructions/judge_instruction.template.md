# Judge Instruction Template — plan-quality evaluation

> **Purpose:** Reusable template for instructing an LLM judge to compare and score
> multiple candidate plans against a weighted rubric. Render with `render_instruction.py`
> against a `manifest.yaml` to produce a verbatim, ready-to-paste judge instruction.
>
> **Schema version:** `1.0`
>
> **Placeholders (Mustache-style `{{...}}`):**
> - `{{N_PLANS}}` — integer count of plans (e.g., `4`)
> - `{{N_PLANS_WORD}}` — same as `N_PLANS` but as an English word (`three`, `four`, …)
> - `{{PLAN_LIST_BULLETS}}` — multi-line bullet list of plan paths, one per line, indented
> - `{{TOP_WEIGHTED_AXES_CSV}}` — comma-separated top-weighted axes (e.g., `plan depth, comprehensiveness, correctness, elegance (design quality)`)
> - `{{USECASE_ID}}` — e.g., `usecase_001` (for provenance footer; not in the prompt body)
> - `{{RENDERED_AT}}` — ISO-8601 timestamp the rendering happened (for provenance footer)
>
> **Authoring notes:**
> - The wording is preserved as faithfully as possible to the original 2026-06-06 instruction
>   used in usecase_001, with typos corrected (`assessing`, `assessment`) so future usecases
>   don't inherit them. The verbatim original is preserved at
>   `instructions/judge_instruction.3way.md` for historical fidelity.
> - "Spawn as many agents as possible" is intentionally preserved as a *prompt intent* signal
>   even though some judges (e.g., Rovo Dev in the 2026-06-06 reference run) chose direct
>   read over delegation. The judgment file should record whether subagents were actually
>   used so the {prompt intent vs. judge behavior} delta is auditable.

---

## RENDERED INSTRUCTION STARTS BELOW THIS LINE — paste from here to the judge

Here are a few plans, they might have been updated so re-read the plans

{{PLAN_LIST_BULLETS}}

Carefully compare the quality of {{N_PLANS_WORD}} plans, identify any issues or problems from each. So compare from various perspectives, and scoring them, {{TOP_WEIGHTED_AXES_CSV}} is of top weighted perspective.

I might be wrong, therefore YOU MUST make carefully, thoroughly double check with critical-thinking and honest assessment, make really deep, thorough and accurate investigation; ultrathink. Fulfil my ask properly and elegantly, no ad-hoc, no hacky.

Please make carefully, thoroughly double check with critical-thinking, with really deep, thorough and accurate investigation; ultrathink. Spawn as many agents as possible, do as many iterations as needed, and work on user's ask end to end. DO NOT stop until you get your job done assessing all plans.

## RENDERED INSTRUCTION ENDS ABOVE THIS LINE — do not paste this section

---

<!-- provenance footer (not part of the prompt to the judge):
     usecase:     {{USECASE_ID}}
     rendered_at: {{RENDERED_AT}}
     template:    instructions/judge_instruction.template.md (v1.0)
-->
