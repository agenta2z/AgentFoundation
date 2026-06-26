# MFDual / MFI Full Definition–State Separation (finishing "AF1")

> **Status:** v2 (2026-06-21) — integrated with Claude peer plan + source-verified
> against the *already-implemented* `run_context/` package. Completes the deferred
> "AF1" item from `runcontext_explicit_state_separation_plan.md`.
>
> **v2 correction (honest):** v1 assumed the RunContext infra had to be built from
> scratch (`state_factory`/`InferencerStateBase` invented here). **That was wrong** —
> verified: the `run_context/` package is **already implemented** on `dev_xinli_2601`
> (`context.py`, `store.py`, `bridge.py`, `state.py`, `purity.py`, `lint.py`), and the
> typed states **`MultiFlowState`/`DualState`/`MFDualState` already exist** at
> `run_context/state.py:133-187` (currently **dormant** — defined, registered, but not
> yet wired into MFI/MFDual). v2 rewrites Part 1 to **wire the existing dormant infra**
> rather than reinvent it. v1 also **missed two real fields** (`_last_reviewer_alias`,
> `_last_fixer_alias` `:305-306`) — corrected below.
>
> **Scope:** eliminate ALL runtime mutation of `self` in MFI and MFDual so that
> (a) per-run dispatch state lives in the per-call ctx node (`ctx.node.call`,
> typed `MultiFlowState`/`MFDualState`), and (b) the effective reviewer/fixer/panel is
> **resolved on-read** from the winner index/aliases rather than by mutating
> `self.review_inferencer`/`self.fixer_inferencer`/`self.reviewers`. Proven complete by
> the existing **purity gate** (`run_context/purity.py`). **Part 2** then wires BTA to
> consume a single declared `worker_inferencers` over dynamic-N (the user's concrete
> goal; the payoff a pure MFDual unlocks).

---

## §0 — Quick-start (TL;DR for the implementer)

The user's complaint is correct and this plan owns it: the prior plan **deferred** the
hardest, most important case (MFDual/MFI) under the label "AF1", which made the rest of
the state/definition separation *cosmetic* for the one topology that needed it most.

**What must change (one sentence):** dispatch results move off `self` into a typed
per-run state object, and the Dual review/fix loop stops reading mutated
`self.review_inferencer`/`self.fixer_inferencer` and instead calls a **resolver**
(`_resolve_review_target(state)` / `_resolve_fix_target(state)`) that computes the
effective role inferencer from the winner index **at the moment of use**.

**Three pillars:**
1. **State off self** — wire the **already-existing dormant** typed states
   `MultiFlowState` / `DualState` / `MFDualState` (`run_context/state.py:133-187`) onto
   `ctx.node.call` via **compat-properties** (the proven `active_session_id` pattern),
   holding `winner_idx`, `reviewer_alias`, `fixer_alias`, `ranking`, `chosen_role`,
   `runner_up`, `panel_idxs`.
2. **Dynamic resolution, not mutation** — delete every `self.review_inferencer = …` /
   `self.fixer_inferencer = …`; the Dual base reads role targets through resolver
   methods that consult the per-run state (default resolver returns the static
   configured value → byte-identical legacy behavior).
3. **Post-call readers** — `get_winner_flow_idx()` / `get_winner_inferencer()` /
   the MFDual audit (`mfi._last_winner_idx`) must read from the **last run's state**,
   preserved via an explicit, opt-in "last completed run" handle (NOT live mutable
   `self`), so concurrent runs don't clobber each other.

---

## §1 — Source-verified problem statement (the exact mutation surface)

All line numbers verified against the current `dev_xinli_2601` branch on 2026-06-21.

### 1.1 MFI — dispatch state written to `self` (4 fields, single writer, 8 getters)
| Field | Decl | Reset | Write | Read |
|---|---|---|---|---|
| `_last_winner_idx` | `:304` `attrib(default=None, init=False)` | `:916` | `:980,1006-1007` | getters `:1014,1024`; MFDual handoff |
| `_last_reviewer_alias` | `:305` | `:917` | `:987` | `get_chosen_reviewer_alias`; MFDual |
| `_last_fixer_alias` | `:306` | `:918` | `:994` | `get_chosen_fixer_alias`; MFDual |
| `_last_ranking` | `:307` | `:919` | `:1005` | runner-up getters; MFDual |
| `_all_judgments` (per-attempt) | `:303` `factory=list` | `:908 .clear()` | `:768` append | aggregator closure `:673` (in-call) |
| `_latest_per_flow` (per-attempt) | `:302` | — | `:751,871` | dynamic-input/aggregator closures `:664,752,793` (in-call) |

Single in-call writer: `_extract_dispatch_state` (`:976-1009`, called `:1298` async / `:1315` sync).
Per-call reset: `_reset_dispatch_state_for_call` (`:910-919`). **8 post-call getters**
(`get_winner_flow_idx:1014` … `get_non_winner_inferencers:1069`).

### 1.2 MFDual — role inferencers mutated on `self`
- `self._fixer_inferencer_original = …` / `self._review_inferencer_original = …`
  (`:279-280`) — snapshot machinery that **only exists because** the live fields get
  mutated.
- `_select_reviewer_and_fixer()` (`:522`) mutates `self.review_inferencer`
  (`:584,592,607,631,637,639`) and `self.fixer_inferencer` (`:655,657`).
- `_step_propose_impl` (`:728`) saves `prev_reviewer/prev_fixer` (`:731-732`), calls
  `_select_reviewer_and_fixer()` (`:734`), then `_reassign_role_workspace(...)`
  (`:739-740`) and panel reassignment (`:743+`).
- Dual base **consumes** `self.review_inferencer`/`self.fixer_inferencer`/`self.reviewers`
  at ~20 sites (`dual_inferencer.py:292-293,342-343,464-465,566,1104-1286,1382,1411-1561,2086-2102`).

### 1.3 Why this defeats the separation
A single shared MFDual run across N concurrent subtasks → `_last_winner_idx` and
`self.review_inferencer` are clobbered between runs → **silent cross-run corruption**.
This is precisely why dynamic-N currently *requires* `worker_factory` to mint fresh
instances. Finishing this plan removes that *correctness* reason (ergonomic factory may
still remain by choice, but it is no longer a safety crutch).

---

## §2 — Design (wire the EXISTING dormant infra; do not reinvent)

### 2.0 Proven in-tree patterns to copy (verified — do NOT invent)
- **Compat read/write** — `streaming_inferencer_base.py` `_tier3_get`/`_tier3_set`
  (`:416,437`) + the `active_session_id` property (`:472,494`): read the ctx-branch if a
  ctx is active, else the instance backing; **never write the backing under a real ctx**
  (isolation). This is the exact template for MFI's dispatch compat-properties.
- **Read-flip resolution** — `templated_inferencer_base.py` `switch_role`/`_effective_role`,
  already wired and **tested** (`test_m7_role_flip.py`). Template for MFDual roles.
- **Dormant typed states** — `run_context/state.py:133-187`: `MultiFlowState`
  (`winner_idx, reviewer_alias, fixer_alias, ranking, cached_sub_queries`), `DualState`
  (`chosen_role, runner_up, review_workspace, fix_workspace`), `MFDualState` (composition:
  `dual: DualState`, `multiflow: MultiFlowState`), `BTAState` (`latest_per_flow`).
  **Built for exactly this — wire them.**
- **Purity gate** — `run_context/purity.py` (`snapshot_vars` before/after diff) +
  `test_m7_purity_gate.py`: the **completeness proof** for "no residual `self` mutation".

### 2.1 Why the original "AF1" deferral was wrong (verified)
The deferral claimed routing dispatch through a "finally-cleared `_active_ctx`" would
return `None` post-call. **Disproven:** `bridge.exit_run` resets only the **ContextVar**
(`bridge.py:63`) — the per-turn **`RunStateStore` node persists** (shared by reference
down the tree; `store.py`). So a child node written during `mfi.ainfer(run_context=child)`
**survives for the parent to read after the call.** This is what makes Part A-routing and
the MFDual→MFI handoff work without any `self` write.

### Part A — MFI dispatch state → `ctx.node.call` (typed `MultiFlowState`)
The 4 fields (`:304-307`) have a single in-call writer `_extract_dispatch_state`
(`:976-1009`) and 8 post-call getters. **Convert the 4 to compat-properties** (read
`ctx.node.call.<field>` if a ctx is active, else the instance backing) — following
`active_session_id` verbatim. `_extract_dispatch_state` writes via the property (→ ctx
under a real ctx; → backing on legacy). `_reset_dispatch_state_for_call` stays for the
legacy backing; under a ctx a fresh node *is* the reset.

**Part A-routing (the one subtlety):** a **bare** `mfi.infer()` (no `run_context`)
legacy-mints a root discarded on exit → post-call `_active_ctx` is `None` → getters fall
back to the **instance backing** (byte-identical legacy). Under a **real caller ctx**
(MFDual child / `parallel_i`), the write goes to the ctx node (never the backing) → no
cross-talk between concurrent shared-instance workers. The dispatch store is per-turn and
discarded (unlike connection-scoped Tier-3 handles), so this routing is correct and
leak-free.

### Part B — MFI per-attempt working state → `ctx.node.attempt`
`_latest_per_flow` (`:302`; w `:751,871`; r `:664,752,793`) and `_all_judgments` (`:303`;
w `:768`; r `:673`) are per-**attempt** working state read inside the aggregator/dynamic
closures (which run under the active ctx during the call). Move to `ctx.node.attempt`
(typed or dict); closures read via `active_run_context().node.attempt`; `reset_attempt()`
clears. **In-call reads → simpler than Part A** (no post-call survival needed).

### Part C — MFDual reviewer/fixer/reviewers → resolve-on-read (NO `self` mutation)
**Verified:** every branch of `_select_reviewer_and_fixer` (`:522-658`) resolves to a
**stable serializable id** — an alias via `self._resolve_id(...)` or an MFI **flow index**
(winner/runner-up/non-winner). `_collect_candidate_inferencers` (`:659`) is the candidate
basis; **no branch picks an object outside the pre-constructed candidate set** — including
the **panel branch** (`:584-599`), where `self.reviewers = _non_winners[1:]` is fully
derivable from `winner_idx` + the static flow set (it's "all flows except the winner").
*(Honest note: an earlier audit claimed the panel branch was not resolve-on-read because
`DualState` lacks a `panel_ids` field — that conflated "no field today" with "not
derivable". Non-winners = all-flows-minus-winner, derivable from `winner_idx` alone; add a
`panel_idxs` field to `DualState` for explicitness/serialization.)*

- **Replace** `_select_reviewer_and_fixer` (mutate-self) with **on-read resolvers**
  `_effective_reviewer(ctx)`/`_effective_fixer(ctx)`/`_effective_reviewers(ctx)` that re-run
  the same precedence logic against the now-virtualized dispatch state + definition,
  returning the object on demand. Write the chosen **id(s)** into `ctx.node.call`
  (`DualState.chosen_role`/`runner_up` + new `panel_idxs`) under a ctx; legacy → mutate
  `self` (byte-identical).
- **Make `review_inferencer`/`fixer_inferencer`/`reviewers` compat-properties**
  (resolver-or-`self`), so Dual's **~12 point-of-use reads** (`_step_review_impl`
  `:1104,1143,1156,1220,1229,1281`; `_step_fix_impl` `:1411,1434,1446,1494,1502,1510`) and
  the `assertIs(mfdi.review_inferencer, …)` tests keep working unchanged. Construction-time
  default falls back to the existing snapshots `_review_inferencer_original`/
  `_fixer_inferencer_original` (`:279-280`, set once → they **stay** as the static-config
  baseline, NOT per-run mutation).
- **Chosen reviewer's workspace + role (trickiest):** today `_reassign_role_workspace`
  (`:441`) sets the chosen instance's `_workspace` + `switch_role`. Under resolve-on-read,
  the chosen reviewer runs with `run_context=self._rc_child("review")` (Dual already
  threads this), so its **workspace** comes from `ctx.workspace` and its **role/template**
  from `RoleState` in that ctx (the already-wired M7 path) — **no `self` mutation**. The
  identity-guard snapshot becomes a pure baseline, not a mutation target.

### 2.5 What this enables (the payoff — and Part 2's prerequisite)
With zero `self` mutation (purity-gate-proven), one MFDual instance is safe to run
concurrently across N ctxs (`ctx.child(parallel_i)`, each its own `MFDualState`). This is
exactly what makes **Part 2** (single declared `worker_inferencers` over dynamic-N) safe.

---

## §3 — Execution (ordered, atomic commits; each ends green)

> Build on the **existing** `run_context/` infra; no new state framework. Each commit
> ends green and is byte-identical on the legacy (`run_context=None`) path.

- **C0 — Confirm `attrs` mechanics (preflight):** verify `slots=False` on MFI + MFDual
  (compat-properties need a name-mangled backing for the `attrib`s `:304-307`,
  `review_inferencer`/`fixer_inferencer`). Lock this before C2/C4.
- **C1 — Wire `panel_idxs` into `DualState` (additive):** add the one missing field to the
  *existing* `DualState` (`run_context/state.py:145`). No behavior change. Unit test:
  `MFDualState.to_json` round-trips with the nested field (N-S5 discriminator survives).
- **C2 — Part A: MFI dispatch compat-properties:** convert `_last_winner_idx`/
  `_last_reviewer_alias`/`_last_fixer_alias`/`_last_ranking` to compat-properties
  (`_tier3`-style read-ctx-or-backing). `_extract_dispatch_state` writes via the property.
  Gate: every MFI test passes (bare `infer()`→`get_winner_flow_idx()` byte-identical);
  NEW test: two interleaved ctx runs don't clobber each other's `winner_idx`.
- **C3 — Part B: MFI per-attempt state → `ctx.node.attempt`:** move `_latest_per_flow`/
  `_all_judgments`; closures read via `active_run_context().node.attempt`. Gate: aggregator
  + dynamic-input paths unchanged.
- **C4 — Part C resolvers (additive, shadow):** add `_effective_reviewer/fixer/reviewers`
  (pure on-read; port all 6 branches verbatim incl. warning fallbacks). Do NOT yet remove
  mutation — compute resolvers alongside and **assert equal** to the mutated values
  (parity shadow across all dispatch-flag combos: rule-based, reviewer_match_second,
  fixer_match_winner, panel/all-non-winners).
- **C5 — Part C cutover: compat-properties + DELETE mutation:** make
  `review_inferencer`/`fixer_inferencer`/`reviewers` compat-properties (resolver-or-`self`);
  remove the `self.* =` writes in `_select_reviewer_and_fixer`; keep `_*_original`
  snapshots as the **static baseline** (not mutation). Rewire `_reassign_role_workspace`
  to resolve workspace/role from the child ctx (M7 path). Gate: parity shadow (C4) proves
  resolver == legacy on all branches; Dual's ~12 point-of-use reads + `assertIs` tests pass.
- **C6 — Purity gate (completeness proof):** run `run_context/purity.py` `snapshot_vars`
  before/after a full `ainfer` under an active ctx for **both** MFI and MFDual → assert
  **zero `__dict__` delta** (mirrors `test_m7_purity_gate.py`). This catches *any* residual
  mutation, not just the enumerated fields.
- **C7 — Concurrency proof (the AF1 acceptance test):** ONE shared MFDual across N inputs
  concurrently, each under `ctx.child(f"parallel_{i}")`; assert each branch gets its own
  winner/reviewer/fixer in its own `ctx.node.call` with no cross-talk.
- **C8 — Docs/headers:** update MFDual module docstring (`:41-43`); flip the RunContext
  plan's AF1 deferral note (`swift-launching-backus.md` §2.2 / §6 D6) to "completed".

---

## §4 — Risks & honest open questions

- **R1 (resolution-seam scope):** the ~20 Dual base reads must ALL go through the
  resolver; a missed one silently reverts to static config. Mitigation: C5 parity
  shadow test exercises every dispatch-flag combination; add the AST child-call/role-read
  lint (from the RunContext plan §9.5) extended to flag direct `self.review_inferencer`
  reads inside the review/fix loop.
- **R2 (`_reassign_role_workspace` identity guard):** today it compares against
  `_original` snapshots. After deletion, workspace isolation must key off
  "is `state.review_target` a different instance than the static config?". Verify the
  panel case (`reviewers` list) is covered.
- **R3 (post-call getter semantics change):** `get_winner_*` becomes last-run-wins with
  a concurrency guard. Audit all external callers (verified internal: `:541,753-754,1059,1069`);
  confirm no external repo consumer depends on mid-run reads.
- **R4 (workers `_repeat_` / round-robin reuse):** once MFDual is pure, the
  `workers[i % len]` round-robin reuse (`bta:1661`) becomes safe for stateful MFDual —
  worth a follow-up note, but out of scope here.
- **Q1:** Resume serialization — the typed states store **ids/indices** (winner_idx,
  aliases, panel_idxs), all trivially serializable; role objects are re-resolved on read
  from those ids. **Default: store ids, re-resolve (already the design).**
- **Q2:** Keep `worker_factory` or add `[worker]*N` ergonomic now? **Default: keep
  factory; defer ergonomic** — this plan's job is correctness, not config sugar.

---

## §4.5 — PART 2: end-to-end `worker_inferencers` single-declaration → dynamic-N (gated behind Part 1)

> **This is the concrete user-facing goal of the plan.** Make this YAML work
> *inside BTA* for `breakdown-multiflow-plan.yaml`:
>
> ```yaml
> worker_inferencers:          # ONE declared worker, replicated to runtime-N
>   _target_: MultiFlowDual
>   ...
>   flow_configs:
>     - _repeat_: ${_params.num_flows}   # static-K flows inside (unchanged)
>       ...
> ```
>
> **Hard gate:** Part 2 is UNSAFE without Part 1. A single declared worker reused
> across N runtime subtasks is exactly the shared-stateful-instance hazard Part 1
> eliminates. Part 2 commits MUST NOT merge until §5 DoD items 1–4 are green.

### 4.5.1 Field-design decision (honest recommendation)
Three options for the BTA surface:

| Option | Shape | Verdict |
|---|---|---|
| **A. New field `worker_inferencers`** | `worker_inferencers: {single _target_}` → replicate to N | Clear intent, but adds a 3rd worker field (`worker_factory`, `workers`, `worker_inferencers`) → surface bloat + 3-way precedence. |
| **B. Extend existing `workers`** to accept a **single mapping** (not just a list) meaning "this one worker, replicated to dynamic-N" | `workers: {single _target_}` (mapping) vs `workers: [list]` (static-K) | Reuses the existing field + the existing `:1660-1661` consumption seam. Type-switch (list vs mapping) is the only new logic. |
| **C. Keep `worker_factory` only** | no new surface | Rejected — it's the factory the user explicitly wants to drop. |

**Recommendation: Option B (extend `workers`).** Rationale: (1) the existing
`workers` path already round-robins at `:1661` (`workers[i % len(self.workers)]`) —
after Part 1 makes MFDual pure, reusing one instance across N is *safe*, so a single
declared worker is just `len==1` round-robin (every subtask → the same pure instance,
each under its own `ctx.child(parallel_i)`); (2) no new precedence rules — `workers`
already wins over `worker_factory` (`:1660` checks `self.workers` first); (3) smallest
diff. The YAML key can still be spelled `worker_inferencers` via a thin alias if the
user prefers that name, mapping to the `workers` field.

> **Decision needed from user:** keep the *field* as `workers` (recommended) and allow
> the YAML *key* `worker_inferencers` as an alias? Or introduce a genuinely separate
> `worker_inferencers` field? Default if unanswered: **extend `workers`, add
> `worker_inferencers` as a YAML alias.**

### 4.5.2 The two replication semantics (and which to use)
Once MFDual is pure (Part 1), both become safe; pick per the §2.5 cost/robustness call:

1. **Reuse-one-instance (recommended):** `workers=[the_one_pure_mfdual]`; the loop's
   round-robin (`:1661`) serves the *same* instance for every subtask, each run isolated
   by `ctx.child(parallel_{i})` carrying its own `MultiFlowDualState`. Zero extra
   instances; correctness comes entirely from Part 1's state separation. **This is the
   payoff that justifies Part 1.**
2. **Replicate-N (alternative):** deep-copy/re-instantiate the declared worker N times.
   Safe but redundant once (1) is correct; only needed if any residual non-virtualized
   instance field is discovered (Part 1's lint/parity tests must prove there is none).

**Recommendation: ship (1).** It is the literal embodiment of "state separated → one
definition, many runs." Keep (2) unimplemented unless C-tests surface residual impurity.

### 4.5.3 Execution (Part 2 commits — all gated behind Part 1)
- **C9 — BTA accepts single-worker `workers`:** allow `workers` to be a single
  `InferencerBase` (or 1-element list); the `:1660-1661` round-robin then serves that one
  pure instance for all N subtasks. Add the `worker_inferencers` YAML alias → `workers`.
  **Precondition assert (fail-loud):** if `workers` resolves to a single shared *stateful*
  inferencer, require Part 1's purity (assert MFDual is **purity-gate-clean** and has no
  live dispatch mutation — reuse the `run_context/lint.py` child-call lint). Gate: existing `workers`-list tests
  unchanged (byte-identical).
- **C10 — wire `breakdown-multiflow-plan.yaml`:** convert the `worker_factory: {_factory_:
  MultiFlowDual}` block to the single-declaration `worker_inferencers:` form. Acceptance:
  on a **fixed/mocked breakdown** (N pinned), the single-declaration run produces
  **byte-identical** deliverables to the current `worker_factory` version. (Same N, same
  per-subtask inputs, same aggregation.)
- **C11 — dynamic-N concurrency acceptance:** with a breakdown returning N>1 distinct
  subtasks, assert the one reused MFDual instance produces **per-subtask-correct** results
  with **no winner/role/workspace bleed** across subtasks (the real end-to-end proof that
  Part 1 + Part 2 compose). Include the `use_async=True` parallel path (workers may run
  concurrently — `:1640 use_async`), which is the strictest test of Part 1's purity.

### 4.5.4 Part 2 risks
- **R5 (the round-robin was unsafe before):** `workers[i % len]` reuse is exactly why
  `workers` was reserved for "static-K homogeneous" — Part 1 is what makes single-instance
  reuse safe. C8's fail-loud precondition prevents anyone enabling it for a still-impure
  worker.
- **R6 (`use_async` parallel workers):** if BTA runs the N reused-instance workers
  *concurrently* (`:1640`), Part 1's ContextVar/`ctx.child` isolation (from the RunContext
  plan) must hold under `asyncio.gather` — C10 must exercise this explicitly. If the
  RunContext per-call isolation isn't yet landed for the async-parallel BTA path, C10 gates
  on it (cross-plan dependency — call it out, don't paper over it).
- **R7 (workspace isolation per reused run):** each reused-instance run must get a distinct
  child workspace (`flow_N`/subtask_i) — verify the `:1716+` workspace-assignment block
  keys off the per-run ctx, not the (now-shared) instance.

---

## §5 — Definition of Done
**Part 1 (C0–C8 — purity):**
1. The mutation writes are gone, but **NOT** by grep alone — the authoritative proof is
   the **purity gate** (C6): `vars()` before/after a full ctx-run shows **zero `__dict__`
   delta** for MFI and MFDual. (Grep `self.review_inferencer\s*=` / `self._last_winner_idx\s*=`
   outside construction → zero is a fast sanity check, not the proof.)
2. Parity shadow (C4) green: `_effective_reviewer/fixer/reviewers` == legacy mutated values
   across **all 6 branches** (rule-based, reviewer_match_second, fixer_match_winner,
   panel/all-non-winners, both alias paths, warning fallbacks).
3. C7 concurrency test: one shared MFDual, N concurrent ctx-isolated inputs, each with its
   own winner/reviewer/fixer in its own `ctx.node.call`, no cross-talk.
4. Byte-identical legacy: existing `test_multi_flow_inferencer.py` (bare `infer()` →
   getters) + `test_multi_flow_dual_inferencer.py` (`assertIs(mfdi.review_inferencer, …)`)
   pass unchanged.
5. Full AF inferencer + `test/agent_foundation/run_context/` M7 suites green (Buck runner;
   no bare-shell pytest).
6. RunContext ledger AF1 deferral note (`swift-launching-backus.md` §2.2/§6 D6) flipped to
   "completed".

**Part 2 (C9–C11 — `worker_inferencers`):**
7. `breakdown-multiflow-plan.yaml` uses the single `worker_inferencers` declaration (no
   `worker_factory` for the MFDual worker); byte-identical vs `worker_factory` on fixed N.
8. Dynamic-N acceptance green — one reused MFDual, N>1 distinct subtasks, `use_async=True`,
   no winner/role/workspace bleed.
9. Fail-loud precondition present: a single shared **stateful** worker is rejected unless
   Part 1 purity holds (purity-gate-clean MFDual + no live dispatch mutation).

> **Sequencing:** Part 1 (C0–C8) lands first and fully — **the purity gate (DoD 1) is the
> hard gate.** Part 2 (C9–C11, renumbered from the §4.5 C8–C10 to follow Part 1) is the
> user-facing goal but is **safety-gated** on Part 1: DoD 1–4 must be green before any
> Part-2 commit.

---

## §6 — Changelog
- **v2 (2026-06-21):** Integrated Claude peer plan + source verification. **Corrected v1's
  foundational error** (it reinvented `state_factory`/`InferencerStateBase`; the
  `run_context/` package + dormant `MultiFlowState`/`DualState`/`MFDualState` already
  exist — wire them). Adopted the 3-pillar architecture (Part A compat-properties /
  Part B `ctx.node.attempt` / Part C resolve-on-read), the proven in-tree patterns
  (`active_session_id`, `switch_role`, purity gate), the disproven-deferral note (store
  node persists post-`exit_run`), and the corrected field set (added `_last_reviewer_alias`/
  `_last_fixer_alias`). **Rejected one peer-audit false-negative** (panel branch *is*
  resolve-on-read — non-winners derive from `winner_idx`). Preserved v1's unique **Part 2**
  (BTA `worker_inferencers` end-to-end), now safety-gated on the purity gate. v1 saved as
  `.v1.bak`.

<!-- session: see get_session_metadata -->

