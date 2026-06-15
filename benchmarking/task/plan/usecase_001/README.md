# usecase_001 — Plan-Quality Evaluation Benchmark

A reproducible benchmark for evaluating LLM judges on the task of *comparing and
scoring multiple candidate plans that all answer the same upstream prompt*.

This usecase captures a real **4-way plan comparison** from 2026-06-06. Each plan
was produced by a *different* AI coding tool from the same upstream prompt:

| ID | Tool | Slug | Lines | Stance |
|---|---|---|---|---|
| **A** | **Claude Code**     | `fluffy-wand`                          | ~434 | IC implementer's tactical build plan |
| **B** | **Rovo Dev**        | `master_plan_data_builder` v1.0        | ~412 | Master strategy plan |
| **C** | **AgentFoundation** | `aggregator_output`                    | ~734 | 6-worker propose-aggregate integration plan |
| **D** | **Cursor**          | `data_builder_corpus_pipeline` v2.0    | ~118 | Cursor plan-mode todo-graph artifact (proposes superseding Plan B) |

All four answer the same upstream request (build an `atlassian-agi-data-builder`
package + post-training corpus strategy for an enterprise-intelligence LLM).
This means the benchmark also doubles as a **cross-tool comparison** of how
different agent harnesses approach the same multi-faceted planning task.

> ✅ **Current canonical judgment:** `evaluation/2026-06-06_rovo-dev_judgment_4way.md`
> (4-way; A/B/C/D; ranking **C > D > B > A**; scores 62.5 / 58.0 / 51.0 / 47.5 out of 70).
> The 3-way predecessor (`2026-06-06_rovo-dev_judgment.md`) is preserved for
> judge-stability analysis — A/B/C scores were NOT revised between the runs.

---

## Why this benchmark exists

Plan-quality evaluation is notoriously hard to evaluate because:
1. The "right" answer is not a single artifact but a *layered relationship* between
   strategy, contract, and tactics.
2. Judges drift across LLM versions, prompts, and even reading order.
3. Surface signals (length, headers, jargon density) are easy to game.

By preserving the *exact* input the judge saw (3 plans verbatim + the original
request + the repo state at evaluation time), we can:
- Re-run the same evaluation with a new judge model and measure agreement.
- A/B-test scoring rubrics against a fixed plan set.
- Detect when a judge over-weights surface features.

---

## Directory layout

```
usecase_001/
├── README.md                          # you are here
├── manifest.yaml                      # machine-readable index
├── request.md                         # the original upstream prompt (verbatim)
├── plan_sources.yaml                  # where each plan came from + md5 hashes
├── repos.yaml                         # repo pins (commit, branch, dirty state)
├── plans/
│   ├── plan_A-Claude_Code-fluffy_wand.md                # 434 L, IC build plan (Claude Code)
│   ├── plan_B-Rovo_Dev-master_plan_data_builder.md      # 412 L, v1.0 master plan (Rovo Dev; was untracked!)
│   ├── plan_C-AgentFoundation-aggregator_output.md      # 734 L, aggregator over 6 workers (AgentFoundation)
│   └── plan_D-Cursor-data_builder_corpus_pipeline.md    # 118 L, Cursor plan-mode artifact (added 2026-06-06 12:20)
├── evaluation/
│   ├── 2026-06-06_rovo-dev_judgment.md       # 3-way (A,B,C) — predecessor / judge-stability baseline
│   └── 2026-06-06_rovo-dev_judgment_4way.md  # 4-way (A,B,C,D) — current canonical ranking
├── instructions/
│   ├── judge_instruction.template.md         # reusable, placeholder-based; rendered via render_instruction.py
│   ├── judge_instruction.3way.md             # frozen historical — verbatim prompt for the 3-way judgment
│   ├── judge_instruction.4way.md             # current canonical — verbatim prompt for the 4-way judgment
│   └── render_instruction.py                 # Python renderer (auto-detects N from plan_sources.yaml)
└── scripts/
    ├── capture.sh                            # record repo state into repos.yaml
    └── restore.sh                            # fetch + checkout pinned commits
```

---

## How to reproduce the evaluation

### 1. Restore repo state (optional — plans are self-contained in `plans/`)

```bash
cd <this dir>
bash scripts/restore.sh           # detached-HEAD checkout of pinned commits
# add --dry-run to preview, --force to override dirty trees
```

Repos pinned (see `repos.yaml`):
- `~/MyProjects/atlassian-agi` @ `f56cbbff` (main)
- `~/MyProjects/CoreProjects/AgentFoundation` @ `e22a35e7` (dev_xinli_2601)

> ⚠️ **CRITICAL**: Plan B (`00_PLAN_data_builder.md`) was `untracked` in atlassian-agi
> at capture time. Restoring the commit alone WILL NOT recreate it. The verbatim
> copy in `plans/plan_B-Rovo_Dev-master_plan_data_builder.md` is the source-of-truth.
> Plan C lives under `_runtime/` (gitignored by convention) — same story.

### 2. Provide the inputs to a new judge

Give the judge:
- The verbatim judge instruction (`instructions/judge_instruction.4way.md` for the
  current canonical run, `instructions/judge_instruction.3way.md` for the historical
  3-way baseline, or render a fresh one via `python3 instructions/render_instruction.py`).
- The four files in `plans/`.
- *Optionally* `request.md` (the upstream prompt the plans were answering) — useful
  if the judge should score fidelity-to-upstream-prompt; omit if not.

### 3. Capture the new judgment

Save under `evaluation/<YYYY-MM-DD>_<judge>_judgment.md`.

### 4. Compare against the reference

Canonical 4-way ranking: **C > D > B > A** with weighted scores
62.5 / 58.0 / 51.0 / 47.5 (out of 70).
See `evaluation/2026-06-06_rovo-dev_judgment_4way.md` for the full critique,
or `evaluation/2026-06-06_rovo-dev_judgment.md` for the 3-way predecessor
(preserved for judge-stability analysis).

---

## How to add a new repo pin to this usecase

```bash
bash scripts/capture.sh /path/to/some/repo --note "why this repo matters"
```

The script appends-or-updates a marker-bracketed YAML block in `repos.yaml`
(idempotent — safe to re-run).

---

## How to spawn a new usecase

```bash
mkdir -p ../usecase_002/{plans,evaluation,scripts}
cp scripts/*.sh ../usecase_002/scripts/      # scripts are generic, reusable
# Then author request.md, manifest.yaml, plan_sources.yaml; run capture.sh per repo.
```

---

## Integrity checks (before trusting a replay)

The following invariants must hold:

| Check | Command | Expected |
|---|---|---|
| Plan md5s unchanged | `md5 plans/*.md` | matches `plan_sources.yaml::plans[].md5` |
| Pinned commits reachable | `bash scripts/restore.sh --dry-run` | exit 0, no `WARN: commit not found` |
| Manifest schema_version | `grep schema_version manifest.yaml` | `"1.0"` |

If any of the above fails, the replay is **not** a faithful reproduction.

---

## Honest limitations of this benchmark

1. **The judge LLM itself is not pinned.** Re-running through a different model
   family (or even a different temperature) produces a different judgment by
   design. This benchmark measures *judgment stability across judges*, not
   absolute correctness.
2. **The 6 upstream worker outputs that Plan C aggregates are NOT included.**
   They are large, ephemeral (live under `_runtime/`), and the aggregator already
   absorbs their substance. If a future evaluation needs worker-level granularity,
   create `usecase_001_extended/` and snapshot those files separately.
3. **Plan B drift after capture.** The live `00_PLAN_data_builder.md` had already
   been edited to v1.1+ (791 lines) by the time this benchmark was authored. The
   `.v1.0.backup.md` (412 lines) — preserved in `plans/` — is what was scored.
4. **No code is executed.** This is a documentation-judgment benchmark; no Python,
   no API calls, no LLM tool use beyond reading text.

---

## Schema version

`1.0` — minor schema changes (additive YAML fields, new evaluation/ files) do
NOT bump this. Breaking changes (renamed top-level keys, changed scripts/ CLI)
do. See `manifest.yaml::schema_version`.
