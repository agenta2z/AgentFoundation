# WorkGraph Step-Level Dependencies + Cross-Flow Coordination — Plan v1

> **Status:** Draft v1 (2026-06-26). Source-verified against `dev_xinli_2601`.
> **Goal:** Make MultiFlow support **cross-flow round coordination** (each flow's round `x+1`
> waits for *all* flows' round `x` to complete), via a **step-level dependency** primitive —
> **without** promoting rounds to full WorkGraph nodes. Also formalize that step-level
> **checkpoint / retry / resume already exists** at the Workflow layer and is preserved.
>
> **Supersedes** the deferred `mfdual_bug_fixes/mfdual_hygiene_INTEGRATED_plan.md` **§C**
> (the `_ainfer_coordinated` bypass sketch). This plan implements the same user feature with a
> non-bypass, checkpoint-preserving design.

---

## §0 — Quick start (TL;DR)

- **User's 3 claims — all CONFIRMED against source:**
  1. *"A flow's followup takes in the latest artifacts from all flows"* — ✅ true, but it's an
     **unsynchronized snapshot** (`_latest_per_flow` buffer), hence the `(no output yet)` race.
  2. *"Make each flow's round x+1 wait for all flows' round x"* — ✅ **not supported today**;
     it is the deferred `coordinated_stop` scaffold (raises `NotImplementedError`,
     `multi_flow_inferencer.py:1673-1680, 1694-1699`).
  3. *"We can achieve checkpoint/retry/resume at the step level without promoting rounds to
     nodes"* — ✅ **already true** at the Workflow layer (see §1.3). Rounds are Workflow
     **steps**; steps already checkpoint, mark-in-progress, retry, and resume.
- **The chosen design (this plan):** a **step-level barrier** primitive — a per-round
  rendezvous keyed by `(round_index)` shared across the N sibling flows, installed at the LWI
  round boundary — gated by a new `cross_flow_sync` knob. **No WorkGraph node promotion**, **no
  `super()._ainfer()` bypass**, **all existing per-step checkpoint/resume preserved**.
- **Effort:** ~Part 1 (barrier primitive + wiring) is the shippable core; Part 2 (abstain/stop
  protocol) and Part 3 (DAG-promotion, optional future) are sequenced after.

---

## §1 — Ground truth (source-verified this session)

### 1.1 The topology
`MultiFlowInferencer(BreakdownThenAggregateInferencer)`; `BTA(InferencerBase, WorkGraph)`.
MFI `_ainfer` (`multi_flow_inferencer.py:1668-1690`) delegates to `BTA._ainfer`, which fans out
`worker_i` nodes via the WorkGraph executor. **Each `worker_i` is a
`LinearWorkflowInferencer(InferencerBase, Workflow)`** — a *sequential* Workflow, not a graph.

### 1.2 Where rounds live (NOT WorkGraph nodes)
Each flow's rounds (`initial`, `round01`, `round02`, …) are **Workflow steps** inside one
opaque `worker_i` WorkGraph node. They are `StepWrapper` objects appended dynamically via
`_DynamicStepRegistry` + `ExpansionResult` (`linear_workflow_inferencer.py:82-109, 683`), run
as a sequential chain by `Workflow._arun` (`workflow.py:1395-...`, the `while i < len(self._steps)`
loop at `:1534`). Naming via `_dynamic_child_name` (`lwi:1077-1091`): step 0→`initial`,
step N→`round{N:02d}`.

### 1.3 Step-level checkpoint / retry / resume — ALREADY EXISTS (key finding)
`Workflow` (`workflow.py:46`) already implements, **per step**:
- **Per-step result save** — `_save_result(result_id=step_name)` gated by `enable_result_save`.
- **Checkpoint** — `__wf_checkpoint__` carrying `exec_seq, step_index, next_step_index,
  loop_counts, state, expansions` (`_save_checkpoint:590`, `_try_load_checkpoint:595`,
  `CheckpointState:1733`).
- **In-progress marker** — `__wf_step_in_progress__.json` with `step_index, step_name,
  started_at, attempt` (`_save_step_in_progress_marker:725`, `_load_...:772`) → crash
  detection + per-step attempt counts (`_step_attempt_counts`).
- **Resume** — checkpoint-based for loops, else backward scan over saved step results
  (`_arun:1416-1488`), with `___seqN` glob for loop iterations (`_glob_seq_results:796`).
> **Consequence:** the user is right — **we do NOT need to promote rounds to WorkGraph nodes
> to get step-level checkpoint/retry/resume.** It is already step-granular. Promotion would
> only be needed for *cross-flow DAG edges* (Part 3), and even that is optional given the
> barrier primitive (Part 1).

### 1.4 The cross-flow channel + the race
Peer visibility is a **best-effort snapshot** of `_latest_per_flow` /
`_latest_per_flow_path` (`mfi:1257-1290`), read when a flow builds its followup prompt
(`_format_followup_input:624`). Because flows run under one `asyncio.gather` with **no barrier**,
a fast flow's `round01` can build its prompt before a slow peer finishes `initial` →
`(no output yet)` (`_FOLLOWUP_NO_PEERS`, `mfi:101`). Confirmed by run timestamps
(flow_0 built round01 input ~3 min before flow_1 finished initial).

### 1.5 The native barrier that DOES exist (but at the wrong layer)
`WorkGraphNode._arun` (`workgraph.py:1274-1295`) implements a true multi-parent join:
`num_real_parents = sum(1 for p in previous if p is not self)`, each parent enqueues on a
per-node `asyncio.Queue`, and `if qsize() < num_real_parents: return` — body fires only when
the last parent arrives, then `_merge_upstream_inputs`. **This is exactly the barrier we want
— but it only applies between WorkGraph nodes, and rounds are not nodes.** The BTA
worker→aggregator diamond is the live proof it works.

### 1.6 The deferred scaffold (the user's exact feature)
`coordinated_stop: bool = attrib(default=False)` (`mfi:296`) + `NotImplementedError` guards in
both `_ainfer` (`:1673`) and `_infer` (`:1694`); scaffold test
`test/agent_foundation/common/inferencers/test_mfdual_coordinated_stop_scaffold.py`. The
`mfdual_hygiene` plan §C2 sketches `_ainfer_coordinated` as a `for step: gather(...)` loop that
**bypasses `super()._ainfer()`** — which would lose checkpoint/retry/aggregator/reporter and
regress the just-landed peer-path capture. **This plan rejects the bypass.**

---

## §2 — Design

### 2.1 The decision: step-level barrier, not node promotion, not bypass
We add a **cross-flow step barrier**: a rendezvous object shared by the N sibling LWI flows,
on which each flow waits at the **end of round `x`** before starting round `x+1`. Because all
N flows already run concurrently under the BTA's `gather`, an `asyncio.Barrier(N)` (or an
equivalent counting rendezvous) keyed per round index gives the exact semantics the user asked
for, while:
- staying **inside** the existing LWI/Workflow execution (no `super()._ainfer()` bypass);
- preserving **all** per-step checkpoint/retry/resume (§1.3) — the barrier wraps the step
  boundary, it does not replace the step machinery;
- preserving aggregator fan-in, graph-viz, workspace dispatch, and peer-path capture.

### 2.2 Why a barrier (Part 1) rather than DAG promotion (Part 3) first
The native WorkGraph barrier (§1.5) is architecturally the "purest" answer, but promotion is a
real restructure (rounds → first-class nodes, re-homing per-round workspace naming, peer-path
capture, dynamic-mode resume registry, MFDual review/fix dispatch state) **and** the dynamic
cross-sibling edge it needs is *forbidden today* (`_validate_no_cross_boundary_cycles`,
`workgraph.py:570`; `_handle_graph_expansion` only wires to self/own-downstream). So:
- **Part 1 (barrier)** ships the feature with minimal blast radius.
- **Part 3 (DAG promotion)** is the optional strategic end-state, requiring a new
  sibling-edge expansion API in `rich_python_utils` — deferred, scoped honestly.

### 2.3 The barrier primitive (shared rendezvous)
Introduce a small `CrossFlowRendezvous` owned by the MFI (the common parent of the N flows),
created once per MFI run, sized to `N = len(flow_configs)`:
- `await rendezvous.wait(round_index, flow_idx)` — blocks until all **active** flows have
  reached `round_index`, then releases all together. Implementation: an `asyncio.Barrier(N)`
  per round index (created lazily), OR a counting `asyncio.Condition` (preferred — supports
  dynamic party-count decrement when a flow stops early; see §2.5).
- The rendezvous is passed into each flow's LWI via the existing per-flow build closure
  (`_build_worker_factory` → `_factory(sub_query, index)`, `mfi:839-...`), so each flow knows
  its `flow_idx` and the shared rendezvous.

### 2.4 The install point (LWI round boundary — no bypass)
Install the wait at the LWI **dynamic-step boundary**, where round `x` finishes and round
`x+1` is about to be decided/appended — `_build_dynamic_step_wrapper` /
`_next_inferencer_and_result` (`lwi:1038-1060`) + the dynamic registry
(`_DynamicStepRegistry`, `lwi:82-109`). Concretely, the followup step wrapper, **before** it
builds its input (so peer artifacts are fresh), calls
`await rendezvous.wait(round_index=this_step_index, flow_idx=self._flow_idx)`. This guarantees
every peer has **published** its `round x` output to `_latest_per_flow[_path]` before any flow
reads peers for `round x+1`.

> **Ordering invariant (critical):** the per-flow publish to `_latest_per_flow[_path]` must
> happen *before* the barrier release, and the peer-read (`_format_followup_input`) must happen
> *after*. So: **publish → barrier.wait → read peers → run round x+1**. This is the whole fix.

### 2.5 Early-stop / abstain protocol (the make-or-break detail)
A strict `Barrier(N)` **deadlocks** if a flow stops early (finishes in fewer rounds) or errors
mid-round — it never arrives, and peers wait forever. So the rendezvous must support
**dynamic party reduction**:
- When a flow's LWI decides to **stop** (no further round), it calls
  `rendezvous.depart(flow_idx)` — permanently decrementing the expected party count so the
  remaining flows' next `wait` releases at the reduced count.
- When a flow **errors**, the same `depart` is invoked in a `finally` so a crash can't hang
  peers (mirrors the WorkGraph `AbstainResult` sentinel idea, but in-band).
- This is why a **counting `asyncio.Condition`** is preferred over `asyncio.Barrier` (the
  latter's party count is fixed at construction).

### 2.6 Decouple "barrier" from "unanimous stop vote"
The original §C2 bundled two concerns: (a) the **visibility barrier** (wait so peers are
fresh) and (b) a **unanimous stop vote** (all flows stop together). The user's ask is **only
(a)**. This plan ships **(a)** as `cross_flow_sync` and leaves **(b)** as an *optional* future
mode (`coordinated_stop` can later layer the vote on top of the barrier). Barrier-without-vote
lets early finishers `depart` (§2.5) instead of being forced to keep working.

### 2.7 The public knob
Replace the dead `coordinated_stop` scaffold semantics with an explicit, non-throwing knob:
- `cross_flow_sync: bool = attrib(default=False)` — when True, install the rendezvous barrier.
- Keep `coordinated_stop` as a **future** superset (barrier + unanimous vote); for now, if set
  True, it implies `cross_flow_sync=True` plus the vote (or continues to raise until Part 2
  lands the vote). Decision recorded in §5 Q1.

---

## §3 — Execution (ordered, atomic commits; each ends green)

- **C1 — `CrossFlowRendezvous` primitive (additive, unit-tested in isolation).** New small
  class (counting `asyncio.Condition`): `wait(round_index, flow_idx)`, `depart(flow_idx)`,
  `active_count`. Pure asyncio, no inferencer deps. Unit tests: N flows release together;
  `depart` reduces release threshold; a departed+all-arrived mix releases; error-path `depart`
  in `finally` never hangs. **No wiring yet.**
- **C2 — `cross_flow_sync` knob + rendezvous creation in MFI.** Add the attrib; create one
  rendezvous per MFI run sized to `len(flow_configs)`; thread it + `flow_idx` through
  `_build_worker_factory._factory` into each flow's LWI (new optional LWI attribs
  `_cross_flow_rendezvous`, `_flow_idx`, both default `None` → byte-identical when unset).
  Gate: all MFI tests pass with `cross_flow_sync=False` (default) unchanged.
- **C3 — install the wait at the LWI round boundary.** In the followup step wrapper: enforce
  **publish → `await rendezvous.wait(...)` → read peers** (§2.4 invariant). Only active when
  `_cross_flow_rendezvous is not None`. Gate: with the knob off, zero behavior change; with it
  on, a 2-flow test (fast Codex + slow RovoDev fixture) shows **both** flows' `round01` inputs
  carry the peer's `initial` output (no `(no output yet)`).
- **C4 — early-stop / error `depart` wiring.** Call `depart(flow_idx)` when a flow's LWI
  terminates its loop (stop decision) and in a `finally` around the flow run. Gate: a test
  where flow_0 stops after `initial` while flow_1 runs 3 rounds → no deadlock; flow_1
  completes; flow_0's early departure is logged.
- **C5 — replace the `coordinated_stop` NotImplementedError scaffold.** Re-point
  `coordinated_stop=True` to enable `cross_flow_sync` (barrier) and document that the unanimous
  **vote** is the only deferred piece (Part 2). Update the scaffold test to assert the new
  semantics. Gate: setting the knob no longer raises; barrier runs.
- **C6 — checkpoint/resume coexistence proof (the user's 3rd claim, locked by test).** A test
  that (a) runs a 2-flow `cross_flow_sync` run to `round01`, (b) kills it, (c) resumes — and
  asserts each flow resumes from its **per-step** checkpoint (`__wf_checkpoint__` /
  in-progress marker) AND the barrier re-synchronizes correctly on resume (the rendezvous is
  re-created fresh; departed flows re-derive from persisted stop state). Gate: resume produces
  the same deliverable as an uninterrupted run.
- **C7 — docs/headers.** Update MFI docstring (`:287-298`), the `mfdual_hygiene` §C banner
  ("superseded by this plan; barrier shipped, vote deferred"), and add a short
  "step-level checkpoint already exists" note to the LWI module docstring.

> **Sequencing:** C1→C2→C3 is the shippable barrier (the user's core ask). C4 is required for
> safety (no deadlock). C5–C7 are cleanup/proof. **Part 2 (unanimous vote)** and **Part 3
> (DAG node promotion)** are separate, later plans.

---

## §4 — Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Deadlock** if a flow never arrives (early stop / error). | §2.5 `depart` + `finally`; C4 test with an early-stopping flow. The counting Condition (not fixed Barrier) makes party reduction first-class. |
| R2 | **Idle cost** — fast flow waits for slowest each round (intrinsic to the semantics; ~3 min/round in the observed run). | Document as inherent; the knob is **opt-in** (default off). Optionally add a per-round `sync_timeout` that `depart`s a straggler (Part 2). |
| R3 | **Resume + live barrier** — the rendezvous is an in-memory asyncio object, not serialized. | C6: re-create the rendezvous fresh on resume; reconstruct active set from each flow's persisted stop state (the step checkpoint already records whether a flow finished). Barrier is per-process-run, not persisted — correct by construction. |
| R4 | **Ordering bug** — if publish happens *after* the barrier, peers still read stale. | §2.4 invariant enforced + asserted in C3 (peer output present, not `(no output yet)`). |
| R5 | **Nested gather context** — flows run under BTA's `gather`; the barrier must use the *same* event loop. | The rendezvous is created in MFI `_ainfer` (same loop as the BTA gather); pure asyncio primitives; no thread hop (unlike the sync `parallel_infer` ThreadPool path — `cross_flow_sync` is async-only; for sync `_infer`, either raise a clear "async-only" error or fall back to independent, recorded in §5 Q2). |
| R6 | **MFDual review/fix loop above the flows** — does the barrier interact with the outer Dual rounds? | The barrier is *between sibling flows within one MFI propose phase*; the outer MFDual review/fix is a separate WorkGraph layer and is unaffected. Verify in C3 with an MFDual config. |

---

## §5 — Open questions
- **Q1:** Should `coordinated_stop` (the old scaffold) become a strict superset
  (`cross_flow_sync` + unanimous vote), or should we deprecate it in favor of two orthogonal
  knobs (`cross_flow_sync`, `cross_flow_stop_vote`)? **Default:** two orthogonal knobs; keep
  `coordinated_stop` as a deprecated alias = both-on. (Cleaner; matches §2.6.)
- **Q2:** Sync path (`_infer`) — barrier is async-only. Raise a clear "cross_flow_sync requires
  async" error, or silently fall back to independent? **Default:** raise (no silent
  divergence — consistent with the current loud scaffold philosophy).
- **Q3:** Per-round `sync_timeout` to auto-`depart` a straggler vs. waiting unboundedly?
  **Default:** none in Part 1 (unbounded wait + explicit `depart` only); add timeout in Part 2.
- **Q4:** Should the barrier release pass *merged* peer artifacts (like WorkGraph's
  `_merge_upstream_inputs`) or just unblock and let each flow read the shared buffer?
  **Default:** unblock + read shared buffer (reuses the existing `_latest_per_flow[_path]`
  channel; minimal change). Merge-on-release is a Part 3 nicety.

---

## §6 — Out of scope (separate/later plans)
- **Part 2 — unanimous/majority stop vote** (the second half of the old §C2).
- **Part 3 — DAG node promotion** (rounds → first-class WorkGraphNodes with cross-flow
  `previous` edges; needs a new sibling-edge expansion API in `rich_python_utils` since
  `_validate_no_cross_boundary_cycles:570` forbids it today). The native barrier (§1.5) would
  then come for free, but the re-homing cost (workspace naming, peer-path capture,
  dynamic-resume registry, MFDual dispatch state) is substantial.

---

## §7 — Definition of Done (Part 1)
1. `cross_flow_sync=False` (default): **byte-identical** to today (full MFI/MFDual suite green).
2. `cross_flow_sync=True`, 2-flow heterogeneous run: **both** flows' `round01` inputs carry the
   peer's prior-round output (zero `(no output yet)`), verified on disk.
3. Early-stop test: one flow departs after `initial`, peers complete, **no deadlock**.
4. Resume test (C6): kill mid-`round01`, resume → each flow resumes from its per-step
   checkpoint AND the barrier re-synchronizes; deliverable identical to uninterrupted run.
5. `coordinated_stop=True` no longer raises (re-pointed to the barrier; vote documented as the
   lone deferred piece).
6. `mfdual_hygiene` §C banner updated to "superseded — barrier shipped here".

---

## §8 — Changelog
- **v1 (2026-06-26):** Initial plan. Source-verified all three user claims (peer-visibility
  snapshot race; cross-flow barrier unsupported = deferred `coordinated_stop` scaffold;
  step-level checkpoint/retry/resume **already exists** at the Workflow layer). Chose the
  **step-level barrier** (counting rendezvous at the LWI round boundary) over both the §C2
  `super()._ainfer()` bypass (rejected — loses checkpoint/aggregator) and full WorkGraph node
  promotion (deferred to Part 3 — forbidden cross-sibling edge + heavy re-homing). Decoupled
  the visibility barrier from the unanimous stop vote (user asked only for the barrier).
