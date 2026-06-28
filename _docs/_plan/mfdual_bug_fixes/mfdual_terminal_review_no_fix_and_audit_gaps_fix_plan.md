# MFDual: Terminal-Review-Without-Fix + Audit/Logging Gaps — Fix Plan v1

> **Status:** Draft v1 (2026-06-27). Source-verified against `dev_xinli_2601`.
> **Run analyzed:** `_runtime/tasks/multimodal_plan_3flow/multimodal_plan_3flow_20260627_140601_8d62955c`
> **Trigger:** A "successful" task-tool run shipped a worker whose final plan was its
> **un-fixed propose output**, despite the reviewer raising a **CRITICAL** issue. The success
> banner was wrong because verification only checked structural existence, not whether the
> review→fix cycle actually contributed.

---

## §0 — Honest executive summary

The run had **one true correctness bug** + **two real observability gaps** + **two
benign-but-misleading artifacts**. I initially mis-scored several as "green" and I corrected
each by reading source + workspace. The headline:

| # | Symptom (user-reported) | Verdict | Severity |
|---|---|---|---|
| **B1** | worker_00 final output == propose output; reviewer said CRITICAL; **fix never ran** | **REAL BUG** — terminal-review-without-fix | **Critical** |
| **B2** | `round_log.jsonl` only contains `{"round": N}` (winner_idx/ranking/reviewer/fixer all absent) | **REAL GAP** — audit under-populated | High (it's *the* dispatch oracle) |
| **B3** | Per-step `logs/session/` empty for Codex + Claude-Code (CLI & SDK); only RovoDev populates | **REAL GAP** — inconsistent session logging | Medium |
| **B4** | Panelist count varies per worker (worker_00=2, worker_01/02=1) | **NOT A BUG** — intended `ALL_NON_WINNERS` dynamic panel | Doc/clarity only |
| **B5** | Empty initial (pre-round) `review/`, `fix/` scaffold dirs | **NOT A BUG** — scaffold created, consensus short-circuits | Cosmetic |
| **META** | Verification claimed success without catching B1–B3 | **REAL PROCESS GAP** — see §6 | High |

**The one thing that must not ship again:** B1. A worker can deliver an **un-reviewed-as-fixed**
plan that a CRITICAL review explicitly rejected, and the pipeline reports success.

---

## §1 — Ground truth (workspace forensics, this run)

| Worker | rounds | output vs propose | last-round fix output | review verdict (last round) |
|---|---|---|---|---|
| worker_00 | **1** | **IDENTICAL** (29 827 B) | **NONE** | both panelists **CRITICAL** |
| worker_01 | 2 | differs (43 965 vs 41 953 B) | present (43 965 B) | round_02 review: `issues: []` |
| worker_02 | 2 | differs (46 400 vs 41 270 B) | present (45 231→46 400 B) | round_02 review: pass |

Corrections to the earlier audit (which I owe you honestly):
- The earlier "all `fix/` dirs empty" was a **measurement artifact** of `find -maxdepth 2`;
  fix output lives at `…/fix/outputs/final_deliverables/output.md` (depth 4). worker_01/02 fix
  **did** run and write. Only worker_00 genuinely has no fix output.
- The earlier "worker_00 propose is a degenerate stub / progress note" was a **mis-read**: that
  quote came from the reviewer of the **initial** (pre-round) propose. worker_00's *round_01*
  propose is a **full substantive plan**. So worker_00 is not a "degenerate flow" case — it is a
  genuine **fix-skipped-on-terminal-review** case, which is worse (the plan was good enough to
  review, the review found CRITICAL, and the fix that would have addressed it never ran).

---

## §2 — Root causes (source-verified)

### B1 (Critical) — Terminal review without fix
**File:** `…/flow_inferencers/dual_inferencer.py`

The consensus loop is built as three steps (`:1094-1110`):
```
propose
review
fix   (loop_back_to="review", loop_condition=_check_loop_condition,
       max_loop_iterations = config.max_iterations - 1)
```
- `_check_loop_condition` (`:1080-1092`) = `not state.get("consensus_reached", False)`.
- `_step_review_impl` (`:1206-1547`): parses review, computes `reached =
  consensus_checker(...)` (`:1509`), records the **review** audit (`:1535`), and — **if
  `reached` is True** — appends the iteration, sets `final_output = base_output_str`
  (the *current/propose* output), and `raise WorkflowAborted()` (`:1540-1545`).
- `_step_fix_impl` (`:1549-1733`) only runs when review did **not** raise, and it is the only
  place `base_output_str` is replaced (`:1722`) and `_last_output_child_ws` is re-pointed to the
  fix child.

The consensus checker itself is **correct** (`_default_check_consensus`, `:2093-2110`): it
returns `False` if **any** issue exceeds threshold, *even when `approved: true`*. So worker_00's
CRITICAL → `reached=False` → it did **not** abort at review. That means the loop *should* have
proceeded to `fix`. **Yet worker_00 ran only 1 round and produced no fix output.**

**The actual defect is the loop-budget / terminal-iteration interaction.** With
`max_loop_iterations = max_iterations - 1`, when the **last permitted iteration's `review`
returns not-consensus**, the workflow has two ways to end *after a review and before the matching
fix*:
1. The fix step's loop budget is exhausted, so the engine treats the post-`fix` loop-back as
   terminal — but the failing case is when the run terminates while the **last executed step was
   `review`** (not `fix`), leaving `base_output_str` = propose and `_last_output_child_ws` =
   propose child. `_finalize_output` (`:613-645`) then symlinks the **propose** child →
   `output.md == propose`.
2. `final_output` is set to `base_output_str` (propose) **only** on the consensus-abort path
   (`:1543`); on the non-consensus terminal path there is **no guarantee a fix ran**, so the
   delivered output is silently the un-fixed propose.

**Net:** there exists a terminal path where **a not-consensus review is the last action and no
fix is dispatched**, so a CRITICAL-rejected propose is shipped verbatim. (worker_00 hit it;
worker_01/02 had enough budget that their last action was a `fix`.)

> **Why worker_00 specifically?** It ran fewer effective iterations than 01/02 (1 vs 2). The
> per-worker iteration/budget asymmetry is the same dynamic-config surface as B4
> (`ALL_NON_WINNERS`): worker_00's flow ranking/round outcome left it on the terminal-review
> boundary. The fix must make the boundary **safe regardless of budget**, not depend on luck.

### B2 (High) — `round_log.jsonl` under-populated
**File:** `dual_inferencer.py:781-838` (`_record_round_audit`).
It writes `{round, phase, inferencer_class, inferencer_workspace, timestamp, **extra}`. The
dispatch-decision fields the user expects — `winner_idx`, `ranking`, `reviewer_idx`,
`fixer_idx`, `consensus_reached`, `severity` — are only present if passed via `extra`, and the
review/fix callers (`:1535`, `:1729`) pass **no `extra`**. So the audit degrades to `{"round":
N}` and is useless as the dispatch oracle. (This is the same class of "audit not populated" gap
seen in prior MFDual sessions.)

### B3 (Medium) — Per-step session logs only for RovoDev
**File:** `streaming_inferencer_base.py:216-235` (`_get_clean_output_for_cache`) +
`:843-862` pipeline. RovoDev overrides `_get_clean_output_for_cache()` (reads `--output-file`)
so its clean transcript is persisted per step; Codex CLI, Claude-Code CLI, and both SDK
inferencers **do not override it**, so the per-step `logs/session/` stays empty. Their transcript
*is* captured in the parent Dual's `Round*` parts files, but the **per-step workspace
`logs/session/` is empty**, which breaks per-node debuggability and any verification that keys
on it.

### B4 (Not a bug) — Variable panelist count
**Files:** `dual_inferencer.py:265` (`num_reviewers=1`), panel-dir creation `:1430-1446`
(`_panel_k = 1 + len(_panel_extra)` when `num_reviewers<=1`); `multi_flow_dual_inferencer.py`
populates `reviewers` from **non-winner flows** at runtime (`ALL_NON_WINNERS`, ~`:901-912`). So a
worker with more non-winner flows that round gets more panelists. **Intended** — but undocumented
and surprising, and it interacts with B1 (ranking outcome → budget → terminal boundary).

### B5 (Cosmetic) — Empty initial review/fix scaffold dirs
`_reassign_role_workspace` / round-scaffold creates `review/`, `fix/` at the initial
(pre-round) level; when initial consensus short-circuits, they're never populated. Misleading
but harmless.

---

## §3 — Fix design (proper & elegant, no hacks)

### F1 — **Guarantee no CRITICAL-rejected output ships** (the core fix)
Two complementary, defense-in-depth changes:

1. **Terminal-fix guarantee.** Restructure the loop end so that whenever the **last review is
   not consensus** and the iteration budget is exhausted, a **final fix is still dispatched**
   for that review before termination. Concretely: make the loop boundary "review→fix" atomic —
   the budget should count *review+fix pairs*, never end on a bare not-consensus review. The
   cleanest expression: keep `max_iterations` as the number of *review→fix pairs*, and ensure the
   step graph cannot terminate immediately after a non-consensus `review` (only after a `fix`, or
   after a consensus `review`).
2. **Finalization invariant (fail-loud safety net).** In `_finalize_output` (`:613`) /
   `_finalize_response`, assert: *if the last recorded review was not-consensus (a blocking
   issue remained) and no subsequent fix ran, the run must NOT silently symlink the propose
   output.* Instead either (a) dispatch a final remedial fix, or (b) mark the attempt
   `degraded=True` in the audit + emit a loud `log_warning` (and surface it to the aggregator so
   it can down-weight/skip the worker). This guarantees the failure is **visible**, never silent.

> Decision (Q1, §5): prefer **(F1.1) terminal-fix guarantee** as the primary fix; keep **(F1.2)
> invariant** as the permanent safety net so any future regression is loud, not silent.

### F2 — **Populate the dispatch audit** (`round_log.jsonl`)
At both audit call sites (`:1535` review, `:1729` fix) pass an `extra` dict with the
decision fields already in scope: `winner_idx`, `ranking`, `reviewer_idx`/`reviewer_alias`,
`fixer_idx`/`fixer_alias`, `consensus_reached`, `severity`, `iteration`, `attempt`. For MFDual,
thread the resolver's dispatch result (winner/ranking) into the audit. Result: each round_log
line fully describes the dispatch decision — restoring it as the oracle the user expects.

### F3 — **Uniform per-step session logging across CLI/SDK inferencers**
Give Codex CLI, Claude-Code CLI, and both SDK inferencers a working
`_get_clean_output_for_cache()` (or hoist the persistence so it does not depend on a
provider-specific override). Two options:
- **F3a (preferred):** move the per-step session-log write into the base pipeline so it persists
  the already-captured clean stream for **every** inferencer (RovoDev's `--output-file` becomes
  an optimization, not a prerequisite).
- **F3b:** each CLI/SDK inferencer overrides `_get_clean_output_for_cache()` to return its
  already-parsed final text.
Decision (Q2): **F3a** — one base-level guarantee beats four per-provider overrides.

### F4 — **Document B4** (variable panel) and **suppress B5** (empty scaffold)
- B4: add a docstring + one `round_log` field (`panel_size`, `reviewer_strategy`) so the panel
  count is explained in the audit rather than surprising.
- B5: create the initial `review/`/`fix/` scaffold **lazily** (only when actually used), or
  prune empty scaffold dirs in finalization, so the tree isn't misleading.

---

## §4 — VERIFICATION.md enhancements (catch this class early)

Both catalogs check **structural existence** (dirs exist, files non-empty) but not **semantic
contribution**. Add these audit rows to **both**
`OpenStartup/test/openteam/resources/tools/task/VERIFICATION.md` and
`OpenStartup/test/openteam/resources/tools/VERIFICATION.md`:

| New check | What it asserts | Catches |
|---|---|---|
| **V-FIX-CONTRIB** | For every Dual/MFDual worker whose last review was **not consensus**, the worker's `output.md` MUST differ from its `propose/outputs/output.md` (a fix contributed). | **B1** |
| **V-NO-CRITICAL-SHIP** | No delivered worker output may correspond to a last-round review containing an unresolved issue at/above the consensus threshold (parse the panelist reviews). | **B1** (semantic) |
| **V-ROUND-LOG-COMPLETE** | Every `round_log.jsonl` line for a review/fix phase MUST contain non-null `consensus_reached` + (for MFDual) `winner_idx`/`ranking`. | **B2** |
| **V-SESSION-LOG-PRESENT** | Every leaf inference invocation (any CLI/SDK type) MUST have a non-empty `logs/session/`. | **B3** |
| **V-PANEL-EXPLAINED** | Each review step's panelist count MUST equal the `panel_size` recorded in its round_log (i.e. the count is explained, not arbitrary). | **B4** |
| **V-NO-EMPTY-SCAFFOLD** | No empty `review/`/`fix/` scaffold dirs remain after finalization. | **B5** |

> **Principle:** existence ≠ contribution. The catalog must verify that review/fix **changed the
> output** and that **no CRITICAL-rejected artifact is delivered** — the two checks whose absence
> let the false "success" through.

---

## §5 — Open questions
- **Q1:** Terminal-review-without-fix — fix by (a) guaranteeing a final fix dispatch, or (b)
  only marking degraded + failing loud? **Default:** both — (a) primary, (b) permanent net.
- **Q2:** Session logging — base-level guarantee (F3a) vs per-provider override (F3b)?
  **Default:** F3a.
- **Q3:** When a final remedial fix still cannot reach consensus (genuinely hard issue), should
  the worker be **excluded** from aggregation or **included with a degraded flag**? **Default:**
  included + degraded flag + aggregator down-weight (don't silently drop work).
- **Q4:** Should `max_iterations` semantics be redefined as "review→fix pairs" (clearer) vs the
  current "loop-backs"? **Default:** redefine to pairs + migrate configs; document in the audit.

---

## §6 — Process post-mortem (why "success" was claimed)
The success banner checked: deliverable exists ✓, 0 errors ✓, store.json valid ✓, panelist dir
count, symlinks, vestigial dirs. It did **not** check: did fix contribute? do per-step logs
exist? is any CRITICAL-rejected output being shipped? The new §4 rows close exactly those gaps.
**Commitment:** no run is "successful" until V-FIX-CONTRIB + V-NO-CRITICAL-SHIP +
V-ROUND-LOG-COMPLETE pass.

---

## §7 — Execution (ordered, atomic commits; each ends green)
- **C1 — Audit population (F2).** Pass `extra` dicts at `:1535`/`:1729`; thread MFDual
  winner/ranking. Lowest-risk, highest-diagnostic-value; do first so subsequent fixes are
  observable. Test: round_log lines contain the decision fields.
- **C2 — Finalization invariant (F1.2).** Add the fail-loud safety net in
  `_finalize_output`/`_finalize_response`: detect "last review not-consensus & no fix after it",
  emit `degraded=True` + `log_warning`. Test: a synthetic 1-iteration not-consensus run is
  flagged (reproduces worker_00).
- **C3 — Terminal-fix guarantee (F1.1).** Restructure the loop so termination cannot occur on a
  bare non-consensus review. Test: a max_iterations=1 not-consensus case still dispatches one
  fix; output differs from propose.
- **C4 — Uniform session logging (F3a).** Base-pipeline per-step session-log persistence. Test:
  Codex CLI + Claude-Code CLI + both SDK leaves produce non-empty `logs/session/`.
- **C5 — Panel doc + scaffold cleanup (F4).** Docstring + `panel_size`/`reviewer_strategy` in
  round_log; lazy/cleaned scaffold. Test: V-PANEL-EXPLAINED + V-NO-EMPTY-SCAFFOLD pass.
- **C6 — VERIFICATION.md rows (§4).** Add all 6 checks to both catalogs + wire into the audit
  harness. Test: re-running the harness on the analyzed run FAILS V-FIX-CONTRIB +
  V-NO-CRITICAL-SHIP for worker_00 (proving the catalog now catches it).

> **Sequencing rationale:** C1 (see) → C2 (fail-loud) → C3 (actually fix) → C4/C5 (hygiene) →
> C6 (lock it in the catalog). C1–C3 are the safety-critical core; C6 guarantees regression
> can't recur silently.

---

## §8 — Definition of Done
1. A `max_iterations=1` run whose only review is **not consensus** still dispatches a fix; the
   worker output **differs** from propose. (B1 fixed.)
2. No run can deliver a worker output that corresponds to an unresolved at/above-threshold
   review — either a fix resolves it, or the worker is flagged `degraded` and the aggregator is
   informed. (B1 semantic safety.)
3. Every review/fix `round_log.jsonl` line carries `consensus_reached` + (MFDual)
   `winner_idx`/`ranking`/`panel_size`. (B2.)
4. Every leaf invocation (Codex/Claude-Code CLI+SDK, RovoDev) has a non-empty `logs/session/`.
   (B3.)
5. Variable panel size is documented and **explained in the audit**; no empty scaffold dirs
   remain. (B4/B5.)
6. Re-running the verification harness on
   `multimodal_plan_3flow_20260627_140601_8d62955c` **fails** the new V-FIX-CONTRIB /
   V-NO-CRITICAL-SHIP / V-ROUND-LOG-COMPLETE rows for worker_00 (catalog proven to catch it),
   and **passes** after C1–C5 land.

---

## §9 — Changelog
- **v1 (2026-06-27):** Initial plan. Source-verified all 5 symptom clusters. Corrected two
  earlier mis-reads (fix-dir "empty" was a depth artifact; worker_00 propose was substantive, not
  a stub). Identified the true core bug as **terminal-review-without-fix** (a non-consensus
  review can be the last action, shipping the un-fixed, CRITICAL-rejected propose). Designed a
  defense-in-depth fix (terminal-fix guarantee + fail-loud finalization invariant), audit
  population, uniform session logging, and 6 new VERIFICATION rows whose absence let the false
  "success" through.
