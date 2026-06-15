# Verbatim 3-way Judge Instruction (as given on 2026-06-06 ~10:08 PDT)

> **Status:** HISTORICAL — frozen, never edit. Preserves the exact prompt the
> reference 3-way judge (Rovo Dev) was given, including typos. The corresponding
> reference judgment is `evaluation/2026-06-06_rovo-dev_judgment.md`.
>
> **Typos preserved verbatim:** `assesment` (→ should be `assessment`),
> `assesing` (→ should be `assessing`). The reusable template at
> `instructions/judge_instruction.template.md` corrects these.

---

## RENDERED INSTRUCTION (verbatim — paste this exact text to reproduce)

Here are a few plans , they might have been updated so re-read the plans

  /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_runtime/tasks/task/task_20260606_012220_87b5ffb3/children/propose/outputs/final_deliverables/output.md

  /Users/tchen7/MyProjects/atlassian-agi/data/src/_docs/_plan/00_PLAN_data_builder.md

  /Users/tchen7/.claude/plans/take-a-look-into-fluffy-wand.md


  Carefully compare the quality of three plans, identify any issues or problems from each. So compare from various perspectives, and scoring them, plan depth, comprehensiveness, correctness, elegance (design quality) is of top weighted perspective.

  I might be wrong, therefore YOU MUST make carefully, thoroughly double check with critical-thinking and honest assesment, make really deep, thorough and accurate investigation; ultrathink. Fulfil my ask properly and elegantly, no ad-hoc, no hacky.

  Please make carefully, thoroughly double check with critical-thinking, with really deep, thorough and accurate investigation; ultrathink. Spawn as many agents as possible, do as many iterations as needed, and work on user's ask end to end. DO NOT stop until you get your job done assesing all plans.

---

## Reproduction caveats specific to this instruction

1. The 3 plan paths above point to **original on-disk locations** (`~/.claude/plans/`,
   atlassian-agi repo, AgentFoundation `_runtime/`). Plans A and B at those paths have
   since been mutated (B → v2.x at 874 lines; A → 33 KB). Use the benchmark copies in
   `plans/` instead for byte-faithful replay:
   - `plans/plan_A-Claude_Code-fluffy_wand.md`
   - `plans/plan_B-Rovo_Dev-master_plan_data_builder.md`
   - `plans/plan_C-AgentFoundation-aggregator_output.md`
2. Plan D was not part of this instruction. See `judge_instruction.4way.md` for the
   superseding 4-plan instruction.
3. The rubric weights {depth: 1.5, comprehensiveness: 1.5, correctness: 1.5,
   elegance: 1.5, operationalizability: 1.0} were not in the prompt — the judge inferred
   them from "of top weighted perspective." A new judge should be given the weights
   explicitly to reduce inference variance.
