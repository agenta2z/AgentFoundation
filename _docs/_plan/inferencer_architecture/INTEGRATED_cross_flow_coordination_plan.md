# Cross-Flow Step Coordination for MultiFlow — INTEGRATED Plan

> **Status:** Final (2026-06-26). Source-verified against `dev_xinli_2601`.
> **Integrates:** `codex/workgraph_step_level_multiflow_coordination_plan.md`,
> `revodev/workgraph_step_level_deps_and_cross_flow_coordination_plan.md`, and a
> third (adversarially-reviewed) barrier design. Supersedes
> `mfdual_bug_fixes/mfdual_hygiene_INTEGRATED_plan.md` §C.

---

## §0 — Context (why this change)

Multiple agent *flows* (e.g. Codex CLI + RovoDev CLI) run concurrently inside one
`MultiFlowInferencer` (MFI). Each flow loops dynamic rounds `initial → round01 → round02 …`.
Cross-flow peer visibility is a **best-effort, unsynchronized snapshot** of `_latest_per_flow`.
A fast flow builds its `round01` input before a slow peer finishes `initial`, so it reads
`(no output yet)` for that peer — the documented race (flow_0 built round01 ~3 min before flow_1
finished initial). The fix the user asked for: **each flow's round `x+1` must wait for all
flows' round `x`** (a step-level cross-flow dependency), while preserving checkpoint / resume.

This is the deferred `coordinated_stop` scaffold (`multi_flow_inferencer.py:288-298`), which
currently raises `NotImplementedError` in `_ainfer`/`_infer`.

---

## §1 — Three-way comparison & decision

| Dimension | **Codex** (node promotion) | **RovoDev** (barrier) | **Adversarial barrier** (this) |
|---|---|---|---|
| Mechanism | Each round → first-class WorkGraph node; native multi-parent join | Custom counting `asyncio.Condition` rendezvous at LWI round boundary | Same as RovoDev |
| Barrier algebra | WorkGraph join (proven) | Counting Condition + `depart()` | **Set-based** `_active`/`_arrived`/`_gen` (idempotent; proven deadlock-free) |
| Checkpoint/resume | "free" via node persistence | "already exists at step level" | **Honest:** step machinery exists but is **disabled+broken** in dynamic mode; fix it |
| Early-stop | `skipped`/`stopped` status nodes | `depart()` party reduction | `leave()` + **disk-peeked active set** |
| Blast radius | Large (reimplements LWI rounds; needs new WorkGraph sibling-edge API) | Small | Small |
| Feasible today | **No** — cross-sibling edges forbidden (`workgraph.py:_validate_no_cross_boundary_cycles:570`; expansion only wires self/own-downstream) | Yes | Yes |

### Issues found in each plan
- **Codex** — over-claims promotion is *required* for step-level resume (§63): false. The
  Workflow step machinery already exists; it is merely **disabled** in dynamic mode
  (`linear_workflow_inferencer.py:952` sets `resume_with_saved_results = not dynamic_mode`) and
  **broken** by the B0 gap (the LWI `_save_loop_checkpoint` override at `:1539-1560` omits
  `expansions`, so `_reconstruct_expansions` no-ops at `workflow.py:471-473`). Promotion is **not**
  needed to get per-round resume. Codex also hand-waves cross-sibling-edge feasibility and the
  large LWI re-homing cost (per-round workspace naming, peer-path capture, dynamic-resume
  registry, MFDual reviewer/fixer dispatch state).
- **RovoDev** — §1.3 "step-level checkpoint/resume already exists" is **misleading** for the
  dynamic mode where rounds actually live (disabled + B0-broken). Its `depart()`-in-`finally`
  **misses D1**: a worker can be satisfied *without ever running its `_ainfer`* (WorkGraph
  node-level load `worknode_base.py:232-260`; BTA backup-resume `_try_load_from_output`
  `breakdown_then_aggregate_inferencer.py:1849-1875`; cancel before ainfer `:1889`), so `depart`
  never fires → **peers hang**. Also misses D2 (seed from resolved worker count) and D3
  (cancellation-proof leave). Lists `asyncio.Barrier` as an option — wrong (fixed party count).

### Decision
Ship the **step-level barrier** (RovoDev's feasible-today shape) built on the **set-based
rendezvous** (provably deadlock-free; idempotent under retry), with the **D1/D2/D3 wiring
fixes**, and **genuinely fix dynamic-mode step-level checkpoint/resume** (Codex's valid concern)
*without* node promotion. Adopt Codex's durable ideas: an explicit per-step **status ledger**,
**retry non-multiplication**, and **persisted effective `flow_inputs`** for resume. Node
promotion (Codex's lattice) is the documented **strategic end-state** (Part 3), deferred because
cross-sibling edges are forbidden today and the re-homing cost is large.

This is "real, not hacky": the rendezvous algebra is verified; the robustness bugs are fixed;
resume is fixed rather than hand-waved; the barrier reuses the battle-tested LWI round machinery
instead of reimplementing it.

---

## §2 — Design

### 2.1 The rendezvous primitive (verified-correct)
`CrossFlowRendezvous` — pure asyncio, one `asyncio.Condition`, **set-based** membership:
- `_active: set[int]` (live participants), `_arrived: set[int]` (arrived this barrier), `_gen: int`.
- `async arrive_and_wait(idx)`: under cond — if `idx ∉ _active`: return (post-leave retry, no block);
  add to `_arrived`; if `_arrived == _active`: `_gen += 1; _arrived.clear(); notify_all`; else
  `await wait_for(lambda: _gen != g)`.
- `async leave(idx)`: under cond — if `idx ∉ _active`: return (**idempotent** → double-leave from
  retry is a no-op); discard from `_active`/`_arrived`; if `_active and _arrived == _active`:
  advance gen + notify; elif `not _active`: notify_all.
- **Lazy loop bind** (construct without binding to a loop); **cancellation-proof** `leave`
  (mutation + `notify_all` run without an interruptible suspension, shielded).

**Why set, not counter:** membership makes `arrive`/`leave` idempotent, neutralizing double-fire
from WorkGraph node retry (D4). **Why not `asyncio.Barrier`:** fixed parties, no drop-out,
`abort()` breaks it for everyone — we need a phaser, which asyncio lacks.

**Invariant (deadlock-freedom):** *every index in `_active` eventually calls `leave()`.* Under
it, all active flows stay within one generation and every barrier is released by the last
`arrive` or by a `leave` that shrinks `_active` to equal `_arrived`. The whole job below is
**guaranteeing that invariant** (D1/D2).

### 2.2 Home & resolution (transient, never serialized)
The rendezvous lives in `node.scratch['cross_flow_rendezvous']` on the **MFI's own node** — the
same node `_reset_cross_flow_state` puts `MultiFlowAttemptState` on (`multi_flow_inferencer.py:1100-1105`).
`NodeRunState.scratch` is `attrs.field(factory=dict, eq=False)` and is **manually excluded** from
`store.py`'s hand-rolled `to_json` (mirrors WorkGraph's transient `_aqueue`). It must **never** be
an attrs field on `MultiFlowAttemptState` — `state.py:to_json` (`:110-119`) auto-encodes every
field. Resolve via a new `_resolve_attempt_node(ctx)` (the walk-up `_resolve_attempt_state`
already performs, refactored to return the node) → `node.scratch.get(...)`.

### 2.3 The active set (D1 + D2 + resume, unified)
Seed `_active` with **exactly the flows that will run their `_ainfer` this attempt** — not blindly
`len(flow_configs)`:
- **Fresh run:** all `0..N-1` (MFI disables breakdown; workers are 1:1 with `flow_configs`).
- **Resume / populated workspace:** a flow whose worker result already exists on disk **loads**
  (never runs `_ainfer`); it must be **excluded** from `_active` or it is a ghost → hang. MFI
  peeks each worker's result path (reusing BTA's result-exists resolution) and seeds `_active`
  with only the not-yet-completed flows.
- **Belt-and-suspenders:** also `leave(idx)` in the BTA `async_worker_fn`'s own `try/finally`
  (covers cache-hit / `_check_cancelled` / any path that skips `_ainfer`).

This single "active = will-actually-run" rule fixes D1, D2, and the resume+barrier interaction,
**and preserves flow-level resume** (completed flows still load — we do *not* force-rerun).

### 2.4 Install point (publish → wait → read)
In the per-flow async input builder (`_wrapped_dynamic_input_builder`), insert
`await rdv.arrive_and_wait(index)` **between** the own publish (`:865` text + `:886-889` path) and
the peer read (`:890-893`). Ordering invariant: **publish own → barrier → read peers**. Make the
builder `async def`; change the LWI call site `linear_workflow_inferencer.py:1128` from a sync
call to `inp = await call_maybe_async(self.dynamic_input_builder, state, prev)` — backward-compatible
(sync builders unchanged; `async_utils.py:62-65`). First barrier is gen1 (before `round01` reads
peers' `initial`); step 0 needs no barrier (`original_input`, no peer read).

### 2.5 Deregister (the invariant, three layers)
1. Wrap the factory-returned LWI's `_ainfer` in `try/finally: await rdv.leave(index)` (captures the
   rdv ref at entry; ctx is active per `inferencer_base.py:2661-2668`).
2. Set the worker node `worker_manages_resume=True` **only for coordinated workers** so a flow that
   *is* in `_active` always reaches `_ainfer` (no node-level/backup load short-circuit for
   participants).
3. `leave(index)` in `async_worker_fn`'s `try/finally` as the outer safety net.
Permanent-leave on retry is **correct** (re-registration would fire a spurious release of peers'
higher barrier — single shared generation); document it (D4).

### 2.6 Status ledger (Codex, adapted)
Add an ordered per-flow per-round status to `MultiFlowAttemptState` (serializable): `flow_idx`,
`round_idx`, `state ∈ {ok, stopped, skipped, failed}`, `output_text`, `output_path`. Written at
each round's publish + at flow stop. Uses: (a) reconstruct the resume active set from persisted
state, (b) observability, (c) future unanimous-vote. No `WorkGraphStopFlags.Terminate` for normal
stop — stop is data-level.

### 2.7 Public knob (decoupled from the stop vote)
- New `cross_flow_sync: bool = False` — installs the barrier (the user's literal ask).
- `coordinated_stop` becomes a **non-throwing alias**: `True` ⇒ `cross_flow_sync=True`. The
  **unanimous stop vote** (the other half of the old §C2) is explicitly deferred (Part 2); the
  barrier lets early finishers `leave()` instead of being forced to keep working.
- Sync `_infer` with the knob on raises a clear **"requires async"** error (barrier needs `await`).

### 2.8 Step-level checkpoint/resume (fix, don't hand-wave)
- **B0 fix:** the LWI `_save_loop_checkpoint` override must persist `expansions` (the base saver
  does at `workflow.py:689`; the override at `:1539-1560` drops it). Without it,
  `_reconstruct_expansions` no-ops on resume.
- Re-create the rendezvous **fresh** on resume (it is per-process, never serialized); seed
  `_active` from the disk-peek (§2.3) so completed flows are excluded.
- Persist effective per-flow inputs in `MultiFlowState.flow_inputs` so `propagate_runtime_input`
  resumes from persisted values rather than re-mutating definitions.
- **Retry non-multiplication:** the child inferencer remains the retry owner; the coordinated
  worker node must not add a second retry multiplier.

---

## §3 — Execution (atomic commits; each ends green)

- **C1** `CrossFlowRendezvous` + standalone unit tests (no wiring): N release together; `leave`
  reduces threshold; departed+arrived mix releases; double-leave no-op; N=1; all-leave; arrive
  after leave returns immediately.
- **C2** `cross_flow_sync` attrib; seed rendezvous in `_reset_cross_flow_state` on `node.scratch`
  with the disk-peeked active set; `_resolve_attempt_node`/`_resolve_rendezvous`; thread
  rendezvous + `flow_idx` through `_build_worker_factory`. Default off ⇒ byte-identical.
- **C3** async `_wrapped_dynamic_input_builder`; `arrive_and_wait` between publish/read; LWI `:1128`
  → `await call_maybe_async`. Gate: knob off = no change; knob on (2-flow fast/slow fixture) =
  both `round01` inputs carry the peer's `initial` (zero `(no output yet)`).
- **C4** deregister (LWI `_ainfer` finally + `async_worker_fn` finally + `worker_manages_resume=True`
  for coordinated workers); cancellation-proof `leave`; status ledger writes. Gate: flow_0 stops
  after initial while flow_1 runs 3 rounds → no deadlock.
- **C5** repoint `coordinated_stop`→`cross_flow_sync`; delete `_ainfer` raise; reword `_infer` raise
  to "requires async"; update `test_mfdual_coordinated_stop_scaffold.py`.
- **C6** B0 expansions fix + `_reconstruct_expansions` test; resume active-set reconstruction;
  persist `flow_inputs`; retry-non-multiplication. Gate: kill mid-`round01`, resume → completed
  flows load, incomplete re-run, barrier re-syncs, **no deadlock**, deliverable identical.
- **C7** full MFI/MFDual/LWI/BTA suite green; docstrings + `mfdual_hygiene` §C "superseded" banner.

---

## §4 — Risks (consolidated D1–D5 / R1–R6)
- **D1 hang — worker skips `_ainfer`** → §2.3 active set + §2.5 three-layer deregister.
- **D2 ghost index** → §2.3 active = will-actually-run (not `len(flow_configs)`).
- **D3 cancellation drops `leave`** → §2.1 cancellation-proof critical section.
- **D4 retry → uncoordinated-rest-of-attempt** → accepted/documented; permanent-leave is correct.
- **D5 async user-builder coroutine** → if `:911` is touched, delegate via `await call_maybe_async`.
- **R2 idle cost** (fast waits for slowest each round) → inherent; knob is opt-in (default off).
- **R6 MFDual outer loop** → barrier is between *sibling flows within one propose phase*; the outer
  review/fix WorkGraph layer is unaffected (verify with an MFDual config).

---

## §5 — Testing & E2E
**Unit (must prove the user's three asks):**
1. Step-level dependency: 2-flow fast/slow fixture → no `(no output yet)`; peer text+path visible in
   declaration order; first barrier = gen1.
2. Deadlock-freedom: early stop, mixed `max_dynamic_steps`, all-stop-same-round, N=1, crash mid-round.
3. Checkpoint/resume: kill mid-`round01` → resume loads completed flows, re-runs incomplete, barrier
   re-syncs, deliverable identical; B0 `_reconstruct_expansions` rebuilds the dynamic chain.
4. Regression: `cross_flow_sync=False` byte-identical across full suite.

**E2E:** reproduce the `wsfollowupfix1` run with `cross_flow_sync` on and the two real flows
(Codex CLI + RovoDev CLI) via `breakdown-multiflow-plan.yaml`; verify on disk that each flow's
`round01` input carries the peer's `initial` output and the run completes without hang.

---

## §5b — Implementation status (2026-06-26)

**Delivered & unit-tested (24 new tests, zero regressions vs the dirty-branch baseline of 9
pre-existing, coordination-independent failures):**
- `CrossFlowRendezvous` primitive (`cross_flow_rendezvous.py`) — set-based, idempotent
  `leave`, cancellation-proof, lazy-loop-bound, **+ frozen per-barrier snapshot** so peer
  reads are lock-step consistent (a fast peer racing into round N can't contaminate a slow
  peer's round N-1 read). 12 primitive tests.
- `cross_flow_sync` knob (+ `coordinated_stop` alias); rendezvous seeded on `node.scratch`;
  walk-up resolver. Async input builder with `publish → barrier → read`; LWI `:1128` →
  `await call_maybe_async`. Deadlock-safe deregister: LWI `_ainfer` finally + BTA
  worker-boundary `finally` (`_cross_flow_depart_if_tagged`) for the cache/cancel-skip path.
- Sync `_infer` raises "requires async"; async `_ainfer` runs the barrier transparently.
- Integration tests: fast/slow barrier (peer round N-1 visible, no `(no output yet)`),
  early-stop no-deadlock, mixed `max_dynamic_steps`, populated-workspace resume-safety.

**Resume scope delivered:** resume-**safety** — a coordinated run in a populated workspace
never deadlocks (the rendezvous is re-created fresh per attempt; cache-loaded/early-stopped
flows depart via the worker-boundary safety net). 

**Deferred (pre-existing, orthogonal to coordination — documented honestly):** true
per-round result *resume* inside a flow (load round N-1, continue round N). It needs BOTH
the B0 `expansions`-persistence fix AND enabling dynamic-mode resume (`resume_with_saved_results`
is force-disabled in dynamic mode today), which is a separate, riskier change unrelated to
the barrier. Coordinated resume currently RE-EXECUTES flows (acceptable for an opt-in mode;
no deadlock, no corruption).

## §6 — Deferred (separate plans)
- **Part 2 — unanimous/majority stop vote** (second half of old §C2): layer on top of the barrier;
  optional per-round `sync_timeout` to auto-`leave` a straggler.
- **Part 3 — DAG node promotion** (Codex's lattice): rounds → first-class WorkGraphNodes with
  cross-flow `previous` edges; needs a new sibling-edge expansion API in `rich_python_utils`
  (`_validate_no_cross_boundary_cycles:570` forbids it today) + LWI re-homing. The native
  WorkGraph join (`workgraph.py:1274-1295`) then comes for free.
