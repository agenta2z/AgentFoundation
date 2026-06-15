# Verbatim 4-way Judge Instruction (as used for the 4-way reference judgment)

> **Status:** CURRENT CANONICAL — the prompt the 4-way reference judge (Rovo Dev)
> was given on 2026-06-06 ~12:25 PDT. Corresponds to
> `evaluation/2026-06-06_rovo-dev_judgment_4way.md`.
>
> **Source-of-truth note:** Unlike the 3-way instruction, this 4-way instruction
> points at the **benchmark copies in `plans/`** (not the volatile on-disk
> originals), so it is byte-faithful by construction.

---

## RENDERED INSTRUCTION (paste this exact text to reproduce a 4-way judgment)

Here are four plans, they might have been updated so re-read the plans

  - benchmarking/task/plan/usecase_001/plans/plan_A-Claude_Code-fluffy_wand.md
  - benchmarking/task/plan/usecase_001/plans/plan_B-Rovo_Dev-master_plan_data_builder.md
  - benchmarking/task/plan/usecase_001/plans/plan_C-AgentFoundation-aggregator_output.md
  - benchmarking/task/plan/usecase_001/plans/plan_D-Cursor-data_builder_corpus_pipeline.md

Carefully compare the quality of four plans, identify any issues or problems from each. So compare from various perspectives, and scoring them, plan depth, comprehensiveness, correctness, elegance (design quality) is of top weighted perspective.

I might be wrong, therefore YOU MUST make carefully, thoroughly double check with critical-thinking and honest assessment, make really deep, thorough and accurate investigation; ultrathink. Fulfil my ask properly and elegantly, no ad-hoc, no hacky.

Please make carefully, thoroughly double check with critical-thinking, with really deep, thorough and accurate investigation; ultrathink. Spawn as many agents as possible, do as many iterations as needed, and work on user's ask end to end. DO NOT stop until you get your job done assessing all plans.

---

## Reproduction caveats specific to this instruction

1. The 4 plan paths above point to **benchmark copies in `plans/`** — byte-faithful
   to the original archive (md5s verified). Plans A/B at the *original* locations
   (`~/.claude/plans/`, atlassian-agi repo) have since been mutated; do NOT use the
   originals for replay.
2. The rubric weights {depth: 1.5, comprehensiveness: 1.5, correctness: 1.5,
   elegance: 1.5, operationalizability: 1.0} were not in the prompt — the judge inferred
   them. For a new judge run, provide the weights explicitly if you want to reduce
   inference variance, or leave them implicit if you want to *test* inference variance.
3. The original upstream prompt (the one that produced the 4 plans) is at `request.md`.
   It is NOT part of the judge instruction — but a judge that wants to score
   "fidelity to the upstream prompt" should be given `request.md` as context.
