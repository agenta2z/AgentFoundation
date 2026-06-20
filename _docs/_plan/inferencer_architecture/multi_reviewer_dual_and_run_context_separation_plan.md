# Multi-Reviewer DualInferencer + Definition/State Separation (instance-held two-tier RunState) — Integrated Plan

> **Status:** Draft **v4** (2026-06-19) — reconciled with peer plan; instance-held two-tier `self._run` (NOT ContextVar — AF1); workspace/session/roles stay on instance
> **Author:** Tony Chen (synthesized with Rovo Dev, grounded in live-source verification)
> **Home:** `_docs/_plan/inferencer_architecture/` — next chapter after the `inferencer_axes_INTEGRATED_v*` lineage; depends on / relates to `mfdual_bug_fixes/*`
> **Scope:** TWO sequenced initiatives — (1) a **multi-reviewer** capability for `DualInferencer`/`MFDual` (ship first, feature), and (2) an **orchestration-state consolidation** into an **instance-held two-tier `self._run: RunState`** (`call` + `attempt` buckets) behind compat properties, with **workspace, session, and role instances deliberately left as plain instance fields** (foundational, second). The carrier is the instance, NOT a ContextVar (AF1). The feature is designed to *benefit from* but **not depend on** the refactor.
>
> **⚠️ v4 RECONCILIATION (2026-06-19):** This plan went through four passes. **v1** proposed "move everything (incl. workspace) into a ContextVar + index-based roles + immutable definition" → rejected (breaks AF2/AF3). **v2** over-corrected to "keep ALL state on the instance, single bucket" → missed the two-tier retry invariant (AF4). My **v3-draft** then over-reversed to a **ContextVar** carrier after a subagent wrongly claimed all dispatch reads are in-call. **That was wrong (AF1, re-verified line-by-line):** 15+ unit tests set `mfi._last_winner_idx` with **no `ainfer`**, and a real test reads `get_winner_flow_idx()` **post-call** — a ContextVar reset inside `ainfer` returns `None` there. **v4 is the verified answer (matching the peer plan's corrected v3):** an **instance-held `self._run`** carrier + **two-tier (`call`/`attempt`) lifetime** + **compat properties**; workspace (143 reads, AF2), session (AF2/C-2), and role instances (AF3, accessor seam, no relocation) all stay as plain instance fields. All ContextVar/`enter_node_scope`/`copy_context` machinery from the v3-draft is dropped as accidental complexity (siblings are fresh instances, F5). See §2.0. *(Section headings still labelled "v3" are historical; the design is v4.)*

---

## 🚨 Reviewer banner — read first

This plan answers two distinct user asks:

1. **Feature:** "Our DualInferencer should support **multiple reviewers**, and the fixer takes in all reviews to fix." Plus the two trigger forms the user proposed: (a) `review_inferencer = list[...]` → auto-build a review-BTA; (b) `review_inferencer = single + num_reviewers + review_aggregator`; (c) "just hand a custom BTA as the reviewer."
2. **Refactor (the user's deeper instinct):** "the current inferencer design mixes **definition and state** — is there a proper, elegant way to separate them?" — with the specific sub-question: *"when you initialize a BTA, how do you create, manage, trace, and correctly assign states for child inferencers during inference?"*

**The honest through-line:** the `num_reviewers`-with-a-bare-instance dilemma (can you safely reuse one inferencer N times?) is itself a **symptom** of the definition/state conflation. So the two asks are related — but they are **different sizes and risk profiles**, and this plan **sequences** them rather than entangling them.

**Every load-bearing source claim below was verified live** (file:line cited). Where a claim is a design proposal rather than current code, it is marked **[DESIGN]**.

---

## §0 — Quick-start (TL;DR for the implementer)

- **Initiative A (feature, ship first):** Add multi-reviewer support to `DualInferencer` with **three trigger forms**, all backward-compatible (singular path = N=1 special case):
  - **Form 1 — `review_inferencer: list[...]`** → DualInferencer auto-builds a **review-BTA** with the list as workers.
  - **Form 2 — `review_inferencer: single` + `num_reviewers: N` + `review_aggregator: <inf|None>`** → auto-build a review-BTA repeating the reviewer N times (via factory); `review_aggregator` set → real BTA aggregator; unset → **rule-based deterministic merge**.
  - **Form 3 — `review_inferencer: <a user-built BTA>`** → just works (no new code), as long as it emits the review JSON schema.
  - **Merge default = rule-based union** (never drops a blocking issue → existing `_default_check_consensus` stays correct for free); **LLM aggregator = opt-in**.
  - **MFDual add-on:** `reviewer_match_all_non_winners` (panel mode) — turn the otherwise-idle losing flows into the reviewer panel; winner stays fixer.
  - **Make-or-break wiring:** the proposal under review must reach each review worker. BTA workers receive **only their `sub_query`**, not the BTA's `inference_input`/`extra_feed` — so add a **gated `forward_extra_feed_to_workers`** flag and feed the proposal (already built into Dual's `_review_extra_feed`) to each worker via per-call `extra_feed`.
- **Initiative B (orchestration-state consolidation, second):** Introduce an **instance-held** `self._run: RunState` with **two buckets** — `call` (whole top-level `ainfer`) and `attempt` (reset per retry) — and move ONLY in-memory orchestration state into it behind **compat properties** (`_cached_sub_queries`, `_graph_topology_emitted`, `_latest_per_flow`/`_all_judgments`, dispatch `_last_*`). **Workspace stays on the instance** (143 reads, AF2); **session stays on the instance** (AF2/C-2); **role instances stay on `self`** and are read via an accessor seam (`_get_review_inferencer`/`_get_fixer_inferencer`), with only the chosen alias/index *recorded* in `self._run.call` (AF3 — no index dispatch, no relocation). The carrier is the **instance, NOT a ContextVar**, because dispatch state is read post-call and outside any call by 15+ tests (AF1). The **two-tier lifetime is mandatory** to preserve the winner-survives-malformed-retry invariant (AF4). Concurrency needs nothing special — BTA workers are **fresh instances** (F5), each with its own `self._run`.

---

## §1 — Context (what the user observed, verified)

Three asks emerged from understanding MFDual's propose→(review→fix) loop:

- **Today's MFDual review dispatch (verified):** exactly **one** reviewer = the **runner-up flow** (`ranking[1]`); the **winner** flow's agent becomes the **fixer**; the 3rd-place flow is **discarded**. Review→fix loops 1–`consensus_max_iterations` rounds by that **single** reviewer, early-exit on approval. (Source: `multi_flow_dual_inferencer.py:505-624`; empirically confirmed across 6 workers in run `task_20260609_212629_caee3a33`.)
- **Ask 1 (feature):** support **multiple reviewers**; fixer consumes all (merged) reviews.
- **Ask 2 (refactor):** the inferencer design **mixes definition and per-run state**; separate them elegantly. The `num_reviewers`-reuse dilemma is the trigger that exposed it.

### §1.1 — Verified source facts (the design must respect these)

| # | Fact | Source:line |
|---|---|---|
| F1 | DualInferencer contract is **one review in**: `review_inferencer.ainfer()` → `review_parser` → `consensus_checker(parsed_review, threshold)` → `_build_followup_feed(proposal, parsed_review)`. A list/tuple of reviews does **not** fit without change. | `dual_inferencer.py` (review step + `_default_check_consensus`) |
| F2 | DualInferencer accepts **any `InferencerBase`** as `review_inferencer` — so a BTA drops in. | `dual_inferencer.py` role wiring |
| F3 | **BTA string `predefined_sub_queries` auto-repeats N times** (N = `max_breakdown` or `max_concurrency` or 1); **list form** passes N distinct sub-queries. So "N identical" and "N different (lens)" reviewers are **already supported, zero new code**. | `breakdown_then_aggregate_inferencer.py:993-1009` |
| F4 | **BTA workers receive only their `sub_query`**, NOT the BTA's `inference_input`/`extra_feed` (`_make_worker_fn` calls `w.ainfer(q, ...)`). ⇒ the proposal under review does **not** auto-reach review workers. **This is the make-or-break wiring blocker.** | BTA `_make_worker_fn` (~`:1714`) |
| F5 | BTA workers are created **fresh per call** via `worker_factory()` (must be `LazyConfigFactory`/callable to avoid shared instances). | BTA `:1608-1657` |
| F6 | BTA per-worker workspace is assigned by **mutating** `worker._workspace = self._workspace.child(name)`; the `_workspace` **property setter** cascades `_configure_for_workspace` + `_propagate_workspace_to_children`. | BTA `:1660-1677`; `inferencer_base.py:269-281` |
| F7 | **`worker_isolation_check`** exists — a runtime detector that warns when two workers share an `id()` (the smoking gun for the conflation). | BTA `:1281-1312` |
| F8 | **MFDual mutates shared config mid-call**: `self.review_inferencer = …` / `self.fixer_inferencer = …` at 5 sites; plus identity-guard snapshots (`_fixer_inferencer_original`) and `_reassign_role_workspace`. | `multi_flow_dual_inferencer.py:363,378,567,574,598; 269-272,424-464` |
| F9 | **`ContextVar` already used** for per-call isolation — but for **exactly one** thing: `_current_fallback_state` (module-level, `:23`; set/reset per task at `:2096-2111`). The pattern exists; it was never generalized. | `inferencer_base.py:23, 2096-2111` |
| F10 | **`workspace` (declarative attrib) vs `_workspace` (runtime property)** is an existing **definition/state split** — synced once in `__attrs_post_init__` (`self._workspace = self.workspace`). The seed of the refactor is already in the tree. | `inferencer_base.py:262-281, 659` |
| F11 | Per-call reset methods already exist and are scattered: `_reset_cross_flow_state` (`multi_flow_inferencer.py:895-907`), `_reset_dispatch_state_for_call` (`:910-917`), `_cached_sub_queries=None`, `_graph_topology_emitted=False`. | as cited |

---

## §2 — HONEST RECOMMENDATION on Definition/State separation (the user's explicit ask)

> *"When you initialize a BTA, how do you create, manage, trace, and correctly assign states for child inferencers during inference? Is there a proper, elegant way to separate state from definition?"*

**Verdict (v4, reconciled — see §2.0): yes, separating definition from per-run state is worth doing, and the elegant answer is an INSTANCE-HELD `self._run: RunState` object — NOT a ContextVar — holding ONLY in-memory ORCHESTRATION state, with a TWO-TIER lifetime (`call` + `attempt` buckets), and with THREE things deliberately left as plain instance fields: (1) `workspace`/`_workspace` (AF2 — 143 reads, construction+post-call lifetime), (2) `session` (post-call + cross-round), and (3) role-instance resolution (AF3 — routed via an accessor seam, the resolved instance kept as a live reference in `self._run.call`). The carrier is the INSTANCE because dispatch state is read post-call AND outside any call by 15+ tests (AF1) — a ContextVar reset inside `ainfer` would return `None` there. This is NOT the v1 "move everything incl. workspace + index roles into a ContextVar" design (breaks logic), NOR the v2 "single-bucket instance" design (misses the two-tier retry invariant AF4), NOR a ContextVar carrier (breaks AF1). It is the verified answer: instance-held `self._run`, two-tier, compat-properties.**

### §2.0 — DEEP-AUDIT FINDINGS (two audit passes; v3 reconciles them) ⚠️

This section records **two** audit passes and their reconciliation. The v1 draft proposed "move ALL per-run state (incl. workspace) into a ContextVar + index-based roles + immutable definition." A first audit (v2) rejected that. A **second, deeper pass — reconciling against a peer plan ("swift-launching-backus") and re-verified live (subagents, 2026-06-19)** — found v2 had **over-corrected** and **missed a real invariant**. v3 is the reconciled, correct design.

| Finding | Status after reconciliation (verified) | Consequence for the design |
|---|---|---|
| **AF2 — workspace must STAY on the instance** | ✅ **CONFIRMED** (both passes agree). `_workspace` has construction-time (logger `inferencer_base.py:975`), mid-call mutation (BTA `:1678`, PTI `:2692`), and post-call (`pti:1562`) lifetimes; **~450 `_workspace` occurrences / "143 reads across 6 files."** Moving it is a NO-GO. | **Workspace is NOT moved.** Per-run sibling isolation is already provided by child-naming (`worker_0` vs `worker_1`) — we don't need to relocate workspace to get it. |
| **AF3 — role dispatch needs the INSTANCE, not an index** | ✅ **CONFIRMED.** Base loop reads `self.review_inferencer`/`self.fixer_inferencer` at **~14 sites** (`dual_inferencer.py:276,326,448,550,1084,1088,1114,1126,1191,1199,1263,1291,1294,1310,1321`); identity guard `if inferencer is original` (`multi_flow_dual_inferencer.py:449`). | **No index-based roles.** Instead: an **accessor seam** (`_get_review_inferencer()`/`_get_fixer_inferencer()`) routes all reads; Part 2 stores the *resolved instance* as a **live reference** in call-scoped state (transient, never serialized → identity guard stays valid). |
| **AF1 — dispatch state (`_last_winner_idx`/`_last_ranking`) read post-call AND outside any call** | ✅ **RE-CONFIRMED (my v2 was right; my v3-draft ContextVar reversal was WRONG).** Decisive evidence: **15+ unit tests set `mfi._last_winner_idx = …` directly and then call `_select_reviewer_and_fixer(...)` with NO `ainfer` at all** (`test_multi_flow_dual_inferencer.py:934,940,946,1016,1235,1265,1414,1486,1493`; `test_mfdual_workspace_anomalies_integration.py:201-202,266-267`); and a real-integration test reads `get_winner_flow_idx()` **after the full `ainfer` returns** (`test_multi_flow_dual_real.py:305`). The dispatch read path is `mfi._last_winner_idx` (`multi_flow_dual_inferencer.py:714-715`). | **A ContextVar set/reset inside `ainfer` returns `None` in all these cases ⇒ breaks 15+ tests + the post-call getter.** Therefore the carrier MUST be **instance-held** (`self._run`), exactly like today's fields — NOT a ContextVar. |
| **AF4 — TWO-TIER lifetime (verified, and orthogonal to the carrier)** | 🔴 **CONFIRMED.** `_reset_cross_flow_state` runs **per attempt** (`multi_flow_inferencer.py:895-908`, clears `_latest_per_flow`/`_all_judgments`); `_reset_dispatch_state_for_call` runs **once per call** (`:910-919`) **so a winner parsed in an early attempt survives a malformed retry** — documented invariant `:901-903`; retry loop `inferencer_base.py:1496-1509`. (A real past bug: `test_real_integration/_docs/debugging_analogue…:61`.) | **`self._run` MUST hold TWO buckets:** `call` (whole `ainfer`: dispatch winner/ranking) reset in the public `ainfer`/`infer` override, and `attempt` (reset each retry at `_ainfer`/`_infer` top: cross-flow outputs, cached sub-queries). A single bucket would silently break the invariant. |

**Honest reconciliation (three passes + a reversal I own):** v1 said "move everything incl. workspace into a ContextVar" → wrong (AF2/AF3). v2 said "keep ALL state on the instance" → too conservative AND missed AF4. My v3-*draft* then over-reversed to "two-tier **ContextVar**" after a subagent wrongly claimed all dispatch reads are in-call. **That subagent was wrong:** the tests set `_last_winner_idx` with no `ainfer`, and a real test reads the getter post-call (AF1, just re-verified line-by-line). **The reconciled answer (this v4, matching the peer plan's corrected v3): instance-held `self._run` carrier + two-tier (call/attempt) lifetime + compat properties.** Workspace, session, and role-instances stay on the instance. No ContextVar — it would add only single-instance concurrency isolation, which never occurs because workers are fresh instances (F5).

### §2.1 — RECONCILED answer to the four verbs for BTA (v3)

| Verb | v4 recommendation | Why it's safe / verified |
|---|---|---|
| **CREATE** | At the **outermost** `ainfer`, lazily create `self._run = RunState(run_id, parent_run_id, call={}, attempt={})` on the **instance** (mirrors how `_last_*` fields live today). A child BTA worker is a **fresh instance** (F5) and gets its **own** `self._run`; thread `parent_run_id` for provenance. | Dispatch state is read post-call & outside any call (AF1) — an instance field persists there; a ContextVar would not. Isolation already comes from fresh-instance-per-worker (F5), so no ContextVar is needed. |
| **MANAGE (two-tier!)** | `self._run.call` holds dispatch winner/ranking/resolved-roles (survives malformed retries — **AF4 invariant**); reset **once per call** in the public `ainfer`/`infer` override (= today's `_reset_dispatch_state_for_call`). `self._run.attempt` holds cross-flow outputs + cached sub-queries; reset **per attempt** at the top of `_ainfer`/`_infer` (= today's `_reset_cross_flow_state`). Re-point the EXISTING two reset methods to clear their matching bucket **before deleting them**, gated by an explicit retry test. | Preserves the documented winner-survives-retry invariant (`multi_flow_inferencer.py:901-903`) that a single bucket would silently break. Two reset SITES already exist — we keep both. |
| **CONCURRENCY** | **Nothing special required.** BTA spawns **fresh instances per worker** (F5), each with its own `self._run` — siblings cannot collide. No ContextVar, no `copy_context`, no per-node scope CM. | The only thing a ContextVar would add is isolation when *one* instance runs concurrently — which never happens here (every worker is a distinct instance). Confirmed: the prior ContextVar machinery is **accidental complexity** and is dropped. |
| **TRACE** | `self._run` carries `run_id`/`parent_run_id`; child run-states link to the parent ⇒ a provenance chain. Record the chosen reviewer/fixer alias+index in `self._run.call` for logging. Replaces the `_role_history` mutable-list hack. | Additive metadata on the instance; no new addressing layer. |
| **ASSIGN role (MFDual)** | Route ALL reviewer/fixer reads through an **accessor seam** `_get_review_inferencer()`/`_get_fixer_inferencer()` (today they just return `self.review_inferencer` — zero behavior change). MFDual **still assigns the instance** to `self.review_inferencer`/`self.fixer_inferencer` (F8/AF3); Part 2 additionally **records** the chosen alias/index in `self._run.call` for provenance. The `_*_original` snapshots + `is original` identity guard (`:449`) stay intact. | **No index-based dispatch** and **no relocation of the role instance** (AF3) — eliminating that mutation needs a base redesign (deferred). We make the smell *traceable*, not removed. |

### §2.2 — Why this v4 design is the elegant, non-breaking fit

1. **Carrier is the instance (AF1).** `self._run` persists exactly like today's `_last_*` fields, so the 15+ tests that set `_last_winner_idx` with no `ainfer`, and the post-call `get_winner_flow_idx()` read, all keep working. A ContextVar would return `None` there.
2. **Workspace stays put (AF2).** The 143-read / construction+post-call surface is untouched; sibling isolation already comes from child-naming + fresh instances. Biggest de-risking vs v1.
3. **The two-tier lifetime is honored (AF4).** `call` vs `attempt` buckets map 1:1 onto the existing `_reset_dispatch_state_for_call` vs `_reset_cross_flow_state` SITES, so the winner-survives-malformed-retry invariant is preserved by construction.
4. **Roles stay instances (AF3).** The accessor seam centralizes reads; the role instance still lives on `self`; Part 2 only *records* the choice in `self._run.call`. No index indirection, no base redesign.
5. **Compat properties = non-breaking mechanism.** `_last_winner_idx` etc. become properties delegating to `self._run.call[...]`, lazily creating `self._run`. Every existing read/set site is byte-for-byte preserved.
6. **Strict gate ordering:** re-point each old reset to clear the matching bucket and prove green BEFORE deleting it; land the MultiFlow step only after the retry/two-tier test passes.
7. **Drops accidental complexity:** all ContextVar/`enter_node_scope`/`copy_context`/closure-scope machinery from the prior draft is deleted — each instance reads its own `self._run`; there is nothing to scope.

### §2.3 — The honest caveats (where it's genuinely tricky — the user was right to flag it)

- **C-1: `workspace` stays on the instance (AF2).** This is the load-bearing scope decision: the `workspace` attrib / `_workspace` backing split already works and is the node-addressing layer; sibling isolation comes from child-naming + fresh instances, not workspace relocation. `self._run` holds **no workspace object**.
- **C-2: `session` stays on the instance for this plan.** It carries conversation history the Dual **review→fix loop mutates across rounds** and is read post-call. Only an `active_session_id` *string* (if needed for resume) would ever go into the persisted bucket — the live session handle never does. Relocating the live session is explicitly **dropped from scope** (marginal win, high risk of live-handle leaks).
- **C-3: two-tier lifetime is mandatory (AF4), and gate-ordered.** `self._run.call` (survives malformed retries) vs `self._run.attempt` (reset per attempt). Re-point the existing two resets to clear the matching bucket and prove green **before** deleting them; land the MultiFlow step only after an explicit retry/two-tier test passes — otherwise the winner-survives-retry bug returns **silently**.
- **C-4: compat-property coverage is the real risk (P3).** A missed `self._last_*` / `self.review_inferencer` read site that still touches a stale plain attribute would create a silent dual-source split. Mitigation: grep-enforce that no raw reads of the migrated fields remain outside the property/accessor.
- **C-5: don't block the feature on the refactor.** Part 1 (multi-reviewer) ships today with the `worker_factory` idiom; Part 2 (`self._run`) is sequenced after and does not change Part 1's behavior. The factory remains the supported isolation idiom even after Part 2.

### §2.4 — What I would NOT do (rejected alternatives, honestly)

- ❌ **Move `workspace` into the run-state object** (the v1 idea) — 143 reads / construction+post-call lifetime ⇒ NO-GO (AF2).
- ❌ **A `ContextVar` carrier for orchestration state** (my v3-draft over-reversal) — dispatch state is read post-call AND outside any call by 15+ tests (AF1); a ContextVar reset inside `ainfer` returns `None` there. It would also buy nothing: isolation already comes from fresh-instance-per-worker (F5). **Carrier = the instance (`self._run`).**
- ❌ **Single-bucket consolidation** (my v2 over-correction) — misses the two-tier retry invariant (AF4); would clobber an early-attempt winner on a malformed retry.
- ❌ **Index-based role dispatch** replacing `self.review_inferencer = instance` — breaks the base-loop contract + identity guard (AF3). Use the accessor seam + keep the instance on `self`, recording only the choice in `self._run.call`.
- ❌ **Reuse one bare instance N times** for `num_reviewers` — trips `worker_isolation_check` (F7). **Require a factory** for N>1 (permanent supported idiom).
- ❌ **Feed N raw review transcripts to the fixer** — redundancy/contradictions/bloat. **Merge first** (rule-based union default).
- ❌ **Make multi-reviewer default-on** — N× cost & slower convergence; opt-in, default single.
- ❌ **Relocate leaf `active_session_id`/`cache_folder`** — marginal win, high live-handle-leak risk; dropped.

---

## PART I — EXECUTION

### §E1 — Initiative A: Multi-Reviewer (ship first)

**Commit A1 — BTA `forward_extra_feed_to_workers` flag (the wiring foundation; F4 fix).**
- Add a **gated** attrib `forward_extra_feed_to_workers: bool = attrib(default=False, kw_only=True)` to BTA. In `_build_subgraph_spec` (~`:1533`) pull `worker_extra_feed` out of `kwargs["_inference_args"]` when the flag is set; thread it into `_make_worker_fn` (add `extra_feed=None` param) so each worker call becomes `w.ainfer(q, extra_feed=worker_extra_feed, …)` (both async+sync branches; wire at node construction ~`:1796`). Default off ⇒ **zero behavior change**.
- **Critical kwarg-name detail (verified):** Dual must pass the proposal feed through a kwarg named **`worker_extra_feed`**, NOT `extra_feed` — because `extra_feed` is popped by `_infer_single` (`inferencer_base.py:1293`) and a BTA's no-op `_render_prompt` drops it. `worker_extra_feed` survives into BTA `_ainfer` `**kwargs` → `_inference_args` → `_build_subgraph_spec`. The proposal already lives in Dual's review feed (`_build_review_feed`, `dual_inferencer.py:1575-1615`, under `placeholder_proposal`/`main_response`).
- Tests: (a) off → workers get only `sub_query` (unchanged); (b) on → a `TemplatedInferencerBase` review worker **renders the proposal text** from `worker_extra_feed` into its review prompt (guards the silent-drop risk).

**Commit A2 — `merge_reviews()` deterministic union (rule-based default).**
- New pure function in `flow_parsers.py`: `merge_reviews(parsed_reviews: list[dict]) -> dict`. Union all reviewers' issues; **cluster/dedup** by normalized issue key; **never downgrade severity** (max across reviewers); tag each issue with **`agreement_count`** (how many reviewers raised it). Emit in the **existing review JSON schema** so `review_parser`/`consensus_checker` work unchanged.
- Tests: union preserves a lone CRITICAL; dedup collapses duplicates; agreement_count correct; output validates against review schema.

**Commit A3 — DualInferencer trigger forms (sugar; backward-compatible).**
- In DualInferencer base (so MFDual inherits): in `__attrs_post_init__`/role-resolution, normalize `review_inferencer`:
  - `list` → build a **review-BTA** with the list as `worker_factory` outputs + `forward_extra_feed_to_workers=True` + (aggregator if `review_aggregator` set, else rule-merge via A2).
  - `single` + `num_reviewers>1` → build review-BTA repeating the reviewer **via factory** (fresh instances; never bare reuse — see §2.4).
  - `single` + no `num_reviewers` → **unchanged** (today's path).
  - already a BTA → pass through (Form 3).
- The review step consumes **one** merged review either way (BTA aggregator output, or A2 applied to the tuple via `_coerce_review_result`). `consensus_checker`/`_build_followup_feed` **unchanged**.
- **Accessor seam (also forward-compat for Part 2):** add `_get_review_inferencer()`/`_get_fixer_inferencer()` and route **every** reviewer/fixer read through them (verified sites: `_step_review_impl` ~`:1084,1088,1114,1191,1199`; `_step_fix_impl` ~`:1291,1294,1310,1370,1378`; MFDual `_step_propose_impl` `:706-707`). Today they just return `self.review_inferencer`/`self.fixer_inferencer` → **zero behavior change**; Part 2 keeps the instance on `self` and only *records* the chosen alias/index in `self._run.call`.
- **Lifecycle:** `_iter_child_inferencers` (`:1954`) must also yield the pre-built review workers (stash as `review_bta._static_review_workers`) so `aconnect`/`adisconnect` reach them.
- **Guard:** raise `ValueError` if multi-reviewer is combined with an incompatible existing option; require a **factory** (not a bare instance) for the `num_reviewers` repeat form (else it trips `worker_isolation_check`).
- Tests: each form produces a single merged review; N=1 path byte-identical to today.

**Commit A4 — MFDual `reviewer_match_all_non_winners` panel mode.**
- New flag: when set, the reviewer panel = **all non-winner flows' agents** (not just runner-up); winner stays fixer. Wire via the A3 list form (panel members as the review-BTA workers).
- Tests: 3-flow run → 2 non-winner flows become panel; winner is fixer; merged review drives one consensus check.

**Commit A5 — Review-merge aggregator prompt (LLM opt-in).**
- A `review_aggregator` template whose **output obeys the review JSON schema** (not free prose): cluster issues, **never downgrade CRITICAL/MAJOR**, emit `agreement_count`. (Guards the §A-risk: default aggregator template summarizes away severities.)
- Tests: feed N reviews with one CRITICAL raised once → survives; output parses with `review_parser`.

### §E2 — Initiative B: orchestration-state consolidation via instance-held two-tier `self._run` (second, sequenced)

> **Scope (verified):** consolidate ONLY in-memory **orchestration** state into one **instance-held** `self._run: RunState` with **two buckets**, behind **compat properties** so every existing read/set is byte-for-byte preserved. **Workspace, session, role-instances, and the graph-reporter callback stay as plain instance fields.** No ContextVar, no `enter_node_scope`, no `copy_context` — siblings are fresh instances (F5), so there is nothing to scope. Each step is independently testable; **the full existing inferencer suite is the regression gate after every step** (realistic precisely because the 143-read workspace surface does NOT move).

**Shapes & API (B-shapes).**
- `RunState` dataclass on the instance: `run_id`, `parent_run_id`, `call: dict`, `attempt: dict`. `self._run: RunState | None`, lazily created on first access (so a standalone leaf works unchanged).
- **No ContextVar.** Each inferencer instance reads its own `self._run`. A child BTA worker is a separate instance (F5) with its own `self._run`; link `parent_run_id` for provenance.
- **Compat properties** delegate the migrated fields to the buckets: e.g. `_last_winner_idx` → `self._run.call["winner_idx"]`, `_cached_sub_queries` → `self._run.attempt["cached_sub_queries"]`. Public getters (`get_winner_flow_idx`, …) keep signatures, reading through the properties. A test that does `mfi._last_winner_idx = 0` outside any call hits the setter → persists on the instance (AF1-safe).

**Commit B1 — Container (pure plumbing).** Add the `RunState` dataclass + `self._run` (+ lazy creator) to `inferencer_base.py`; finalize the `_get_review_inferencer()`/`_get_fixer_inferencer()` hook (call sites added in Part 1 A3). **No fields moved yet.** Test: additive — nothing consumes it; existing suite green.

**Commit B2 — InferencerBase.** Move `_output_finalized` into `self._run.call` behind a compat property; lazy-create `self._run` on first access so `_complete_inference` works for a standalone leaf unchanged. Test: leaf-standalone finalizes; nested leaf uses `call`.

**Commit B3 — BTA (lowest risk, highest payoff).** Move `_cached_sub_queries`/`_graph_topology_emitted`/`_last_aggregation_guidance` into `self._run.attempt` (compat properties); re-point the top-of-`_infer`/`_ainfer` resets to clear the attempt bucket. Replace `_inject_aggregator_extra_feed`'s `template_extra_feed` mutation (`:696-748`) with the aggregator's per-call feed via Part 1's `worker_extra_feed` channel. **Leave workspace assignment + aggregator-drift correction + graph-reporter callback exactly as-is.** Test: BTA behavior byte-identical; cache still hit within an attempt; sequential runs don't leak.

**Commit B4 — MultiFlow (the two-tier core; gated by the retry test — AF4).** Convert dispatch `_last_*` (→ `self._run.call`) and `_latest_per_flow`/`_all_judgments` (→ `self._run.attempt`) to compat properties. Re-point `_reset_dispatch_state_for_call` → clear `call` (stays in the public `ainfer`/`infer` override, per-call) and `_reset_cross_flow_state` → clear `attempt` (stays at `_ainfer`/`_infer` top, per-attempt), **preserving the winner-survives-malformed-retry invariant** (`:901-903`). **Do not delete the old reset bodies until the replacement is green.** Test: the **retry/two-tier test** (attempt 1 parses winner, attempt 2 malformed → winner in `call` survives AND `attempt` bucket reset), run before+after re-pointing; plus the 15+ direct-set unit tests pass unchanged.

**Commit B5 — MFDual (provenance only).** **Record** the chosen reviewer/fixer alias+index in `self._run.call` for logging/tracing, and thread `parent_run_id` into worker run-states. **Still assign the instance** to `self.review_inferencer`/`self.fixer_inferencer` (F8/AF3); keep `_*_original` snapshots + the `is original` identity guard + `_reassign_role_workspace` intact. This makes the worst-smell mutation *traceable*, not eliminated (eliminating it needs a base redesign — deferred). Test: role assignment + workspace identical to pre-refactor on the recorded run `task_20260609_212629_caee3a33`; `parent_run_id`/alias appear in `round_log`.

**Resume contract.** `self._run` fields split **persisted** (JSON-serializable: cached sub-queries, dispatch ints/strings, `output_finalized`, any `active_session_id` *string*) vs **transient** (live references, reporters, lambdas — never serialized). The chosen role *instances* are NOT in `self._run` (they stay on `self`), and workspace is not in `self._run` either, so no live handle ever enters serialization. On resume, buckets reconstruct via the existing small-JSON checkpoint path (`_save_breakdown_checkpoint` `:1044-1068`); transient state re-established by `ainfer` as today.

**Dropped from scope (honest):** relocating leaf `active_session_id`/`cache_folder`, index-based roles, ContextVar carrier, physical `frozen=True`, and bare-instance `num_reviewers` — marginal win and/or breaks current logic (AF1/AF2/AF3). The **factory stays the supported isolation idiom.**

### §E3 — Pre-flight checks (run before coding)
- **P1:** Confirm exact BTA `_make_worker_fn` signature + where `extra_feed` would thread (A1). *(Verified ~`:1714`.)*
- **P2:** Confirm the **review JSON schema** + `review_parser`/`consensus_checker` field names (A2/A5 must emit them).
- **P3:** Enumerate **every** reader of the dispatch fields/getters so the compat-property shim (B4) covers them all — a missed read = silent dual-source split. *(Found: `multi_flow_inferencer.py:999-1042`, `multi_flow_dual_inferencer.py:714-715`; **15+ tests set `_last_winner_idx` directly with NO `ainfer`** — `test_multi_flow_dual_inferencer.py:934,940,946,1016,1235,…`; `test_mfdual_workspace_anomalies_integration.py:201-202,266-267`; and `test_multi_flow_dual_real.py:305` reads `get_winner_flow_idx()` post-call. This is exactly why the carrier is the instance, AF1.)*
- **P4:** Confirm `_default_check_consensus` treats any blocking issue as blocking (so rule-merge union is correct for free).
- **P5:** Re-confirm the MFDual `self.*_inferencer =` mutation sites (F8: `:363,378,567,574,598`) are **left intact** by B5 — we only *record* the chosen alias/index in `self._run.call`; the instance assignment + `_*_original` snapshots + identity guard stay (AF3).
- **P6:** Grep-enforce that after each migration step **no raw read of a migrated field** (`self._last_*`, `self._cached_sub_queries`, …) remains outside its compat property/accessor — a missed site = silent dual-source split (C-4). *(No ContextVar/`copy_context` audit needed: carrier is the instance.)*
- **P7:** Capture the **retry/two-tier baseline** on current code (attempt-1 winner survives attempt-2 malformed) before B4 re-points the resets (AF4 gate).

---

## PART II — DESIGN & ARCHITECTURE

### §D1 — Why BTA-as-reviewer (composition) over a native reviewer-list+merge loop
Reusing `review_inferencer = BTA{N workers}` gives parallel fan-out, `max_concurrency`, workspace isolation, checkpoint/resume, and graph reporting **for free**, and DualInferencer already accepts any `InferencerBase` (F2). "Compose proven infra, write no orchestration." The native deterministic merge (A2) is kept as the **default merger and the fallback** if LLM-merge drops issues.

### §D2 — Aggregate-or-not: ENABLE aggregation (decisive)
DualInferencer's contract is **one review in** (F1). BTA-with-aggregation returns **one** consolidated review → drops into the existing single-review path with **zero** DualInferencer changes. BTA-without-aggregation returns **N** reviews → forces `review_parser`/`consensus_checker`/feed to handle N → reintroduces exactly the machinery the composition was meant to avoid. **So aggregation ON is what makes the idea clean.** Default merger = rule-based union (A2); LLM aggregator = opt-in (A5).

### §D3 — Diversity beats count
N identical workers (string-repeat, F3) reduce sampling variance but **share blind spots**. Prefer the **list-of-lenses** form (correctness / completeness / risk / internal-consistency), natively supported by BTA's list form (F3). Different models per worker is even better. The **`agreement_count`** lever (A2/A5) turns the panel into an adversarial filter: lone-reviewer issues = low-confidence (precision ↑, convergence ↑); ≥K-agreement issues = fix-first.

### §D4 — The two failure modes to actively guard
1. **Aggregator silently softens severities** → consensus mis-fires. Mitigate: structured review schema + "never downgrade CRITICAL/MAJOR" (A5) and rule-merge default (A2) which is immune.
2. **Identical-repeat illusion** of robustness while sharing blind spots. Mitigate: lens diversity (§D3).

### §D5 — Instance-held two-tier `self._run` architecture (v4 detail of §2)
- **Definition (conceptually immutable; not physically frozen):** model_id, max_retry, fallback_inferencer, output_path, template_key, role wiring (`review_inferencer`, `fixer_inferencer`, `flow_configs`). (Classified from `inferencer_base.py:80-310`.)
- **MOVED into `self._run` (instance-held), two buckets (compat properties preserve every existing read/set):**
  - `self._run.call` (whole top-level `ainfer`): dispatch `_last_winner_idx`/`_last_ranking`, chosen reviewer/fixer **alias+index** (for provenance), `_output_finalized`. *Survives malformed retries (AF4).*
  - `self._run.attempt` (reset per retry): `_latest_per_flow`/`_all_judgments`, `_cached_sub_queries`, `_graph_topology_emitted`, `_last_aggregation_guidance`. *Wiped each attempt (AF4).*
- **NOT moved (left on instance, by audit):** `_workspace`/`workspace` (143 reads, construction+post-call, AF2), `session` (post-call + cross-round mutation, AF2/C-2), the graph-reporter callback (transient), **and the role instances `self.review_inferencer`/`self.fixer_inferencer`** (AF3) — read via the accessor seam; `self._run.call` only *records* which was chosen.
- **Carrier:** an **instance field** `self._run: RunState` (mirrors today's `_last_*` fields; NOT a ContextVar — AF1: dispatch is read post-call & outside any call). Created lazily at the outermost `ainfer`; each BTA worker is a fresh instance (F5) with its own `self._run` linked via `parent_run_id`. Concurrency needs nothing special — there is no shared instance to isolate.
- **Provenance:** `run_id`/`parent_run_id` on `self._run` + recorded role alias/index replace the `_role_history` mutable-list hack.
- **Resume:** buckets split persisted (JSON-able) vs transient (live handles/lambdas never serialized); workspace and role instances excluded (not in `self._run`).

---

## APPENDIX

### §A1 — Sequencing & risk
- **Ship A first** (feature, low surgery, immediate value). **B second** (foundational, cross-cutting). A uses the existing factory idiom so it does **not** depend on B; after B lands, `num_reviewers` with a bare instance becomes safe and factories become optional.
- **Risk (A):** LLM-merge dropping issues → mitigated by rule-merge default + "never downgrade." Proposal-propagation (F4) → mitigated by gated A1 + a real wiring test (P1).
- **Risk (B):** MEDIUM, contained. Workspace/session/role-instances untouched (AF2/AF3) and carrier is the instance (AF1) → the 143-read surface doesn't move, so the **existing inferencer suite is a realistic regression gate after every step**. The two real risks: (1) **AF4 two-tier** — mitigated by gate-ordering (re-point resets, prove the retry test green, only then delete the old bodies) and the P7 baseline; (2) **compat-property coverage (C-4)** — a missed raw read of a migrated field → silent dual-source split; mitigated by the P6 grep-enforcement after each step.

### §A2 — Verification log
All F1–F11 verified live this session (subagent traces + direct reads): `inferencer_base.py:23,80-310,262-281,659,2096-2111`; BTA `breakdown_then_aggregate_inferencer.py:993-1009,1281-1312,1608-1677,~1714`; `multi_flow_dual_inferencer.py:269-272,363,378,424-464,505-624,567,574,598`; `multi_flow_inferencer.py:895-917`. Run evidence: `task_20260609_212629_caee3a33` (6 MFDual workers; dispatch logged once each; reviewer=ranking[1], fixer=ranking[0]).

### §A3 — Open questions (each with a default if unanswered)
- **Q1:** Agreement threshold **K** value? *Default:* K=1 for v1 (union), expose K as a consensus knob in a follow-up.
- **Q2:** Should panel reviewers get **distinct lens templates** in MFDual (vs same template)? *Default:* same template v1 (the flows already differ by draw); add lens assignment as a fast-follow (§D3).
- **Q3:** Physical `frozen=True` definitions ever? *Default:* **no** — out of scope; RunContext already gives the separation without freezing the definition attribs. A debug-mode write-guard is an optional fast-follow.
- **Q4:** Pursue run-scoped **session** (relocate the live handle) later? *Default:* **no** — keep session on the instance; only an `active_session_id` *string* may enter the persisted bucket if resume needs it (live-handle relocation = high leak risk, marginal win).
- **Q5:** Keep `worker_isolation_check` (F7)? *Default:* **keep it** as a debug guard — it stays valid (RunContext isolates in-memory state; the factory still produces fresh instances).

### §A4 — Changelog
- **v4 (2026-06-19):** **Carrier REVERSED to instance-held `self._run` after re-verifying AF1 line-by-line.** The v3-draft moved orchestration state into a **ContextVar** on a subagent's (wrong) claim that all dispatch reads are in-call. Direct check disproved it: **15+ unit tests set `mfi._last_winner_idx` with NO `ainfer`** (`test_multi_flow_dual_inferencer.py:934,940,946,1016,1235,…`; `test_mfdual_workspace_anomalies_integration.py:201-202,266-267`) and `test_multi_flow_dual_real.py:305` reads `get_winner_flow_idx()` **post-call** — a ContextVar reset in `ainfer` returns `None` there. So Initiative B now uses **instance-held `self._run` + two-tier (`call`/`attempt`) + compat properties** (matching the peer plan's corrected v3); roles **stay on `self`** (B5 only *records* the choice); all ContextVar/`enter_node_scope`/`copy_context`/closure-scope machinery DROPPED as accidental complexity (siblings are fresh instances, F5). Rewrote §0, §2-verdict, §2.0 (AF1 re-confirmed), §2.1–§2.4, §E2 (B1–B5), §D5, P3/P5/P6, §A1 risk-B. Section headings still labelled "v3" are historical. Regression gate unchanged: full existing suite after each step + AF4 retry test.
- **v3 (2026-06-19):** **Reconciliation with peer plan (*swift-launching-backus*), re-verified live.** Found v2 had **over-corrected** (kept ALL state on the instance) and **missed the two-tier retry invariant (AF4)**, and that my AF1 "post-call read break" claim was **wrong** (all orchestration-state reads are in-call). **Initiative B rewritten** to a `ContextVar`-backed **two-tier `RunContext`** (`call_state`/`attempt_state`) moving ONLY orchestration state; **workspace + session + role-instance-resolution stay on the instance** (AF2/AF3, accessor seam + live reference, not index). Added AF4 + corrected AF1 in §2.0; rewrote §2.1–§2.4, §E2 (6-step B1–B5 + resume contract + WorkGraph closure-scope), §D5; added the verified `worker_extra_feed` kwarg detail and accessor seam to §E1 (A1/A3); added P5–P7. Regression gate = full existing inferencer suite after each step + AF4 retry test.
- **v2 (2026-06-19):** Deep-audit correction (AF1–AF3) → proposed instance-held `RunState` consolidation. *(Superseded by v3: too conservative; missed AF4.)*
- **v1 (2026-06-19):** Initial integrated plan; honest def/state recommendation up front; two-initiative sequencing; F1–F11 verified live. *(Superseded.)*

