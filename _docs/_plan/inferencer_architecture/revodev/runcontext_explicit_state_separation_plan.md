# Explicit `RunContext`: Immutable Definition vs. Mutable Run-State — Integrated Plan

> **Status:** Draft **v6** (2026-06-19) — supersedes the *instance-held `self._run`* approach in
> `../multi_reviewer_dual_and_run_context_separation_plan.md`. v6 folds in the peer `swift-launching-backus.md`
> **v11** corrections (all re-verified against live source): (1) the BTA guard `_validate_worker_isolation`
> **only warns, never raises** ⇒ fail-fast is NEW work; (2) the `vars()` purity test would **false-fail** ⇒
> valid only post-retirement with an empty allow-list; (3) `_last_winner_idx`/`_cached_sub_queries` need
> name-mangled backing + a `slots=False` precondition (verified satisfied); (4) measurement honesty —
> **OpenStartup = 73 test files** (not 112); `_runtime/tasks` reads are gitignored artifacts, not coupling.
> Carries the **three-tier state model** + V7/V8/V9 from v5. Part 1 (multi-reviewer) inherited unchanged.
> **Carrier:** an *explicit, caller-threaded* `RunContext` with a **root-owned path-keyed `RunStateStore`**
> + a **transient/serializable split** (`RuntimeBindings` vs `NodeRunState`) — integrated from the Codex peer plan.
> **v2 enhancement (§2.4):** typed per-class `InferencerStateBase` via a `state_factory` attribute
> (generalizing the shipped `initial_state_factory`) carried **as the value type inside** the path-keyed store.
> **v3 integration:** path-keyed store (R3), `RuntimeBindings`/`NodeRunState` split (R6), effective-query
> accessor to stop `flow_configs` mutation (§E5b), `_legacy_bind_context` bridge (§E6), kwarg-leak guard (R5).

> **⚠️ WHY A NEW PLAN (honest framing):** The prior lineage (v1→v7) converged on an **instance-held**
> `self._run` because it was the smallest non-breaking change. A follow-up design discussion (recorded in
> §A1) surfaced a sharper question: *if factories are mandatory for safe parallelism anyway, and an
> inferencer instance already carries `_workspace`/`session`/`graph_reporter`/role-mutation, then the
> instance is NOT a pure definition — so where should run-state actually live?* The conclusion: the
> **reusable definition is the YAML/config/factory; each instantiated inferencer is a stateful runtime
> actor; the cleanest separation is an explicit `RunContext` minted by the application host (e.g. the
> `task` tool) and threaded through `infer`/`ainfer`, with `context.child(path)` for descendants.** This
> plan adopts that — with a **mandatory legacy-default fallback** so nothing breaks if no context is passed.

---

## §0 — Quick-start (TL;DR for an implementer)

- **Definition layer:** YAML config + `LazyConfigFactory` (already how the system works). An inferencer
  *instance* is NOT the reusable definition.
- **Run-state layer:** a new `RunContext` object — `run_id`, `parent_run_id`, `workspace`, `call: dict`,
  `attempt: dict`, plus transient handles (`graph_reporter`, `interactive`, `checkpoint_store`).
- **Threading:** `ainfer(input, *, run_context=None)`. If `None`, the inferencer **mints a legacy context
  from its own `self._workspace`/attrs** (exact current behavior — zero break). Children call
  `run_context.child("<node_path>")` to get an isolated sub-context.
- **Host:** the `task` executor (`resources/tools/task/executor.py:854,875`) mints the **root** context
  right after `instantiate(cfg)` and passes it into `inferencer.ainfer(request, run_context=root)`.
- **Isolation contract (unchanged from prior plans):** parallel siblings MUST be **fresh instances via
  factory**; `RunContext` does NOT make a shared instance safe (it still mutates `_workspace`/`session`).
  A fail-fast guard rejects duplicate object identity in worker/reviewer panels.
- **Part 1 (multi-reviewer) is independent** and ships with factory isolation regardless of Part 2.
- **This is a broad refactor (it touches the base class).** Measured surface: **167 inferencer test files,
  ~1,379 state references, 25 real-LLM tests, 108 tests that write `self._workspace=`**. The strategy
  (§E-TEST) is **"refactor, not rewrite": existing tests pass UNCHANGED** via legacy-default + read/write
  compat properties; net-new tests cover the new capabilities (concurrency isolation, resume, provenance,
  definition-immutability). **OpenStartup needs ZERO changes** under the legacy default (verified) — its **73**
  tests are themselves the "legacy unchanged" acceptance gate.

---

## §1 — Honest assessment: does the discussion make sense?

**Yes — and it correctly identifies the real architectural truth the prior plans worked *around*.** Critical evaluation of each claim (full transcript in §A1):

| Discussion claim | My verdict | Evidence |
|---|---|---|
| "Instance-held `self._run` is unsafe if the instance is shared concurrently" | ✅ **TRUE** | Even with `self._run`, the same object has one mutable `_workspace`/`session`/role; `MFDual` mutates `self.review_inferencer` in place (`multi_flow_dual_inferencer.py:1084-1088`). Sharing breaks regardless of where dispatch ints live. |
| "Factory-as-definition makes most sharing problems disappear" | ✅ **TRUE, already the idiom** | BTA already *requires* `LazyConfigFactory`/fresh-per-worker and raises on non-fresh factories (`breakdown_then_aggregate_inferencer.py:1300-1307`). |
| "If factory is safest, maybe no separation is needed; let YAML be the definition" | ⚠️ **PARTLY** | Factories solve *parallel-sibling* isolation. They do NOT solve: (a) scattered `_last_*`/`_cached_*`/`_latest_*` leaking across **sequential** calls on a reused instance; (b) clean **resume/serialization**; (c) provenance. So *some* state object still has value. |
| "State must be passed in through the inference method; app layer owns it" | ✅ **RIGHT, cleanest** | Public `ainfer`/`infer` already take `**_inference_args` (`inferencer_base.py:2180,1643`) → adding `run_context=` is non-breaking. The `task` executor already mints `working_dir` then calls `ainfer` (`executor.py:672,875`) → natural host. |
| "Adding `run_context` alone gives full separation" | ❌ **FALSE (key honest caveat)** | The instance still owns/mutates `_workspace`, `session`, `graph_reporter`, `switch_role`, `flow_configs[i]["input"]`, PTI per-iteration state. A `run_context` kwarg is *another carrier* unless those fields are **virtualized behind the context**. Full separation = real work, not a kwarg. |
| "Children should allocate `context.child(path)`, not the app pre-creating every entry" | ✅ **ESSENTIAL** | Otherwise the host must know BTA/MFDual internals (leaky). `context.child()` is the load-bearing non-leaky API. |
| "Do it now, not phase-1/phase-2, testing is slow" | ✅ **LEGIT, with caveat** | Getting the *contract* right up front is correct (a shipped kwarg is hard to change later). But "all at once" must NOT mean "break 69 call sites at once." Resolution: **lock the final contract now; roll out additively behind the legacy default** so the framework runs at every commit. |

**Net:** the destination (explicit `RunContext`, host-minted, `context.child()`, definition=YAML/factory) is the **right design**. Reject only the implicit "a kwarg = done" — true separation requires migrating instance-mutated fields behind the context, sequenced carefully **without** a flag-day break.

---

## §2 — The design (PART II)

### §2.1 — Two layers, one contract

```
DEFINITION (immutable, reusable)              RUN-STATE (mutable, per-invocation)
────────────────────────────────             ──────────────────────────────────
YAML config / LazyConfigFactory     ──▶       RunContext(run_id, parent_run_id,
  _target_, model, template_keys,                        workspace, call={}, attempt={},
  num_reviewers, worker_factory, …                       graph_reporter, interactive,
                                                          checkpoint_store)
        │ instantiate()                                │ minted by host (task executor)
        ▼                                              ▼
  inferencer instance  ◀── ainfer(input, *, run_context=ctx) ──  threaded in
        │                                              │
        │ children via factory (fresh instances)       │ children via ctx.child("round_01/review/worker_0")
        ▼                                              ▼
  worker instance      ◀── ainfer(query, *, run_context=child_ctx) ──
```

**The contract (final — lock now):**
1. `infer`/`ainfer` gain an explicit keyword `run_context: RunContext | None = None`.
2. **If `run_context is None`** → the inferencer mints a **legacy context** from its own attrs
   (`workspace=self._workspace`, fresh `run_id`, empty buckets) → **byte-identical to today**.
3. A descendant receives `run_context.child(node_path)` — own `call`/`attempt` buckets + child workspace;
   `parent_run_id` links provenance.
4. **Definition objects never hold run-state**; run-state never holds the *definition*. The instance is the
   bridge that executes a definition against a context.

### §2.2 — `RunContext` shape (path-keyed store + transient/serializable split)

> **Integrated from the Codex peer plan (R3/R6):** the **root** owns a **path-keyed `RunStateStore`**; each
> `RunContext` is a lightweight **node-scoped view** over one path. Serializable node-state is structurally
> separated from transient live handles (`RuntimeBindings`). This is strictly better than v2's per-context
> buckets for **resume** (one store to serialize) and **provenance** (stable path keys), and it directly
> answers the original question *"is state keyed by child inferencer paths?"* → **yes, the store is; each
> inferencer only ever touches its own node view via `ctx.child(path)`.**

```python
@dataclass
class NodeRunState:                       # PERSISTABLE — one per node path
    call: dict = field(default_factory=dict)        # whole-call (dispatch winner/ranking, typed state)
    attempt: dict = field(default_factory=dict)     # per-retry (cross-flow outputs, cached sub-queries)
    conversation: dict = field(default_factory=dict)  # V9: _messages/prior_context/sop_state/_suspended_sops/
                                                      #     dynamic_context — EXACTLY _serialize_pause_state's form
    checkpoints: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)  # parent path, chosen reviewer/fixer alias+index, …

@dataclass
class RunStateStore:                      # owned by the ROOT context; serialized as one object
    nodes: dict[str, NodeRunState] = field(default_factory=dict)
    def node(self, path: str) -> NodeRunState:
        return self.nodes.setdefault(path, NodeRunState())

@dataclass
class RuntimeBindings:                     # TIER 2 — TRANSIENT, shared-by-reference; CONCURRENCY-SAFE SINKS ONLY
    graph_reporter: Any = None             # fan-out sink: each child gets child_reporter(node) — safe (V11)
    stream_observer: Any = None            # fan-out sink
    interactive: Any = None                # UI bridge (single root consumer)
    checkpoint_store: Any = None           # path-keyed writes (admit only if writes are genuinely path-keyed)
    cancellation_token: Any = None         # read-only flag
    # NOTE (v5, integrated from Claude v10 V8): a live SESSION/CLIENT is NOT a concurrency-safe sink and is
    # therefore BARRED from Tier 2 by definition — it lives in Tier 3 (LiveHandles). v4 wrongly placed a
    # shared `session_store` here; that was the same latent bug Claude's plan flagged in Codex. Removed.

@dataclass
class LiveHandles:                         # TIER 3 — WORKER-LOCAL / CONNECTION-SCOPED; never serialized
    # Verified (V7/V8): leaf handles are created once via aconnect, REUSED across many ainfer calls in a
    # worker branch, and torn down at adisconnect/__aexit__ — they are connection-scoped, NOT per-call.
    sdk_client: Any = None                 # ClaudeCode self._client (one, reused) — collides if shared
    subprocess: Any = None                 # RovoDevServe self._server_process (one) — collides if shared
    http_session: Any = None               # RovoDevServe self._http_client
    live_session_id: Any = None            # carries multi-turn continuity (fed into the next call)
    logger: Any = None                     # workspace-scoped
    cache_folder: Any = None
    # Each concurrent worker BRANCH owns its own LiveHandles ⇒ fixes the BTA gather() collision WITHOUT
    # relaunching the subprocess or losing session_id continuity (the bug a naive call-local model causes).

@dataclass
class RunContext:                          # lightweight NODE-SCOPED view
    run_id: str
    parent_run_id: str | None
    path: str                              # e.g. "/round_01/review/worker_0"
    workspace: Workspace                   # per-run output tree (was self._workspace)
    state_store: RunStateStore             # shared root-owned store (same object across the tree)
    runtime: RuntimeBindings               # Tier 2: shared transient SINKS (by reference)
    handles: LiveHandles                   # Tier 3: worker-local live connection (per branch)

    @property
    def state(self) -> NodeRunState: return self.state_store.node(self.path)
    @property
    def call(self) -> dict: return self.state.call
    @property
    def attempt(self) -> dict: return self.state.attempt

    def child(self, node_path: str, *, workspace: Workspace | None = None,
              fresh_handles: bool = False) -> "RunContext":
        child_path = f"{self.path.rstrip('/')}/{node_path}"
        return RunContext(
            run_id=self.run_id, parent_run_id=self.path, path=child_path,
            workspace=workspace or self.workspace.child(node_path),
            state_store=self.state_store,        # Tier 1: SAME store (path-keyed) — not copied
            runtime=self.runtime,                # Tier 2: SAME safe sinks — inherited by reference
            handles=LiveHandles() if fresh_handles else self.handles,  # Tier 3: own handles per worker branch
        )
```

**Three tiers — the boundary rule is dictated by concurrency, not taste** (integrated from Claude v10, V7/V8/V11):
- **Tier 1 — `RunStateStore`/`NodeRunState` (serializable, persisted).** Path-keyed; survives restart.
- **Tier 2 — `RuntimeBindings` (transient, shared-by-reference) — CONCURRENCY-SAFE SINKS ONLY.** A handle may
  be shared by reference **iff** it is read-only or a fan-out sink designed for concurrent writers (`graph_reporter`
  via `child_reporter`, V11). **Verified fix:** a live session is *not* such a sink ⇒ it is structurally
  excluded here (removes v4's `session_store`-shared hazard by construction).
- **Tier 3 — `LiveHandles` (worker-local / connection-scoped) — NEW in v5.** SDK client / subprocess / httpx
  session / `live_session_id`. **Verified (V7/V8):** reused across calls, torn down at `adisconnect` — so each
  concurrent worker branch gets its **own** via `child(..., fresh_handles=True)`, which is the *only* model that
  fixes the BTA `gather()` collision without per-call subprocess relaunch / continuity loss.

- **Serializable core** = `run_id` + the whole `RunStateStore.nodes` map + each `workspace.root` path string.
- **Transient** = `RuntimeBindings` **and** `LiveHandles` (both re-attached/re-`aconnect`'d on resume).
- **Resume:** persist the one `RunStateStore`; on resume, re-mint the root with the loaded store, re-attach
  fresh `RuntimeBindings`, and re-establish `LiveHandles` via the normal `aconnect` (+ `live_session_id` if
  continuity is wanted) — same setup path as a fresh run, no split-object rehydration.
- **The `NodeRunState.call`/`attempt` values hold the typed `InferencerStateBase` objects from §2.4** —
  i.e. `ctx.call["state"]` is a typed `MultiFlowState`, not a bare dict. (Codex used untyped dicts; this plan
  keeps the typed-state enhancement as the *value type* inside Codex's path-keyed store. Best of both.)
- **Conversational state is Tier 1, in its EXISTING serialized form (v5, V9):** `NodeRunState.conversation`
  holds `_messages` / `prior_context` / `sop_state.to_dict()` / `_suspended_sops` / `dynamic_context.to_dict()`
  exactly as `_serialize_pause_state` produces them (verified `conversational_inferencer.py:956-967`). This is
  **independent in-memory state, NOT workspace-derivable** — a state-separation that moved only
  `_last_*`/`_cached_*`/`_workspace` would silently break conversational resume. Crucially, `SOPState` and
  `AgenticDynamicContext` **already** have `to_dict()/from_dict()` — so we **keep their dict form** and do
  **NOT** force them into the typed `InferencerStateBase` hierarchy (that would be churn for no gain).

### §2.3 — Why this beats instance-held `self._run` (and where it costs more)

| Dimension | instance-held `self._run` (prior) | explicit `RunContext` (this plan) |
|---|---|---|
| Sequential-call leakage | reset each call (works) | structural — new context per call (cleaner) |
| Concurrent shared instance | unsafe | **still unsafe for `_workspace`/`session`** until migrated; safe after |
| Resume/serialization | smeared across fields | **one object to serialize** ✅ |
| Provenance tree | ids on instance | native via `child()` path ✅ |
| Blast radius | small (compat props) | **larger** (69 signatures; field virtualization) ⚠️ |
| Correctness risk | low | **medium** — must preserve `_workspace` cascade + AF4 two-tier + role identity |

**Honest trade:** right long-term shape, genuinely more invasive. De-risked by (a) the **legacy default**
(no-context path byte-identical at every step), (b) **compat properties** delegating `self._last_*`/
`self._cached_*` to the active context, (c) a strict **migration order** (contract first, field
virtualization last).

### §2.4 — Typed per-class state via `state_factory` (the enhancement)

> **Origin:** a follow-up design idea (§A1.b) — *"have an `InferencerStateBase`; every inferencer has a
> corresponding state class; every inferencer has a `state_factory` attribute that creates its state object,
> and YAML can supply `state_factory` with nested children as the factory's params."* **Honest verdict:
> sound and largely already supported — adopt it, with three corrections.**

**Why it's idiomatic (not new infra) — VERIFIED:**
- The pattern **already ships**: `LinearWorkflowInferencer.initial_state_factory: Optional[Callable] =
  attrib(default=None)` (`linear_workflow_inferencer.py:156`), invoked as
  `self._pending_state = self.initial_state_factory(inference_input)` (`:920-921`), and already wired from a
  lambda/YAML (`tool_as_inferencer.py:629`). So `state_factory` **generalizes a proven field**, it doesn't
  invent one.
- YAML supports `state_factory: {_target_: …, <params>}` natively via the Hydra `_target_`-with-children →
  kwargs walker, incl. `_partial_: true` (`_instantiate.py:614,619+`). It would "just work".
- Per-class state nests 1:1 with the class hierarchy (state inventory, §A1.c): `BaseState` →
  `BTAState(_cached_sub_queries, _graph_topology_emitted)` → `MultiFlowState(_last_winner_idx,
  _latest_per_flow, _all_judgments)`; `DualState(_pending_state, _current_attempt)`.

**Correction 1 — `state_factory` is the *typed populator of the `RunContext`*, NOT a replacement for it.**
Two readings exist and they differ sharply in safety:
- *Reading A (reject):* `state_factory()` builds a `MultiFlowState` and the inferencer stores it as
  `self._state` → this is just the **instance-held `self._run`** design again (unsafe for a shared
  instance — the whole reason this plan exists).
- *Reading B (adopt):* the typed state object is created **into the threaded `RunContext`** for that node
  (`run_context.call`/`attempt` hold a typed `XxxState` instead of a bare dict), never parked on the
  instance as the durable carrier.
⇒ **`RunContext` stays the carrier + transient-handle holder; `state_factory` builds each class's typed
slice of it.** Best of both: typing/cleanliness **and** threaded concurrency-safety.

**Correction 2 — "after full separation, no factory is needed" is only ~70% true.** It becomes true **only
when the state object captures `workspace` AND `role` too** (E6). Plain data fields (dispatch ints, caches)
are isolated by a fresh `state_factory()` per call — but `switch_role` mutates `template_key` /
`_pending_role_changes` on the **instance** (`templated_inferencer_base.py:383`) and MFDual swaps
`self.review_inferencer` in place; a data-only state object does **not** isolate those. **So the factory
(fresh-instance-per-sibling) retires only after E6 moves workspace+role into the context/state — not the
moment a state class is introduced.** Until then, factories remain mandatory for concurrent siblings.

**Correction 3 — use COMPOSITION, not multiple inheritance, for the state hierarchy.** `MultiFlowDual`
derives from both `Dual` and (effectively) `MultiFlow` → a `MFDualState(DualState, MultiFlowState)` would be
a **diamond**. Instead: `MFDualState` *has-a* `dual: DualState` + `flow: MultiFlowState`. And
**transient handles** (`_workspace`, `session`, `graph_reporter`, `interactive`) are NOT serializable state
— they live in the `RunContext`'s transient slots, never in a `XxxState` (else `to_json` breaks).

**Resulting shape:**
```python
@dataclass
class InferencerStateBase:                       # serializable run-state only (no live handles)
    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, d: dict) -> "InferencerStateBase": ...

@dataclass
class BTAState(InferencerStateBase):
    cached_sub_queries: dict = field(default_factory=dict)
    graph_topology_emitted: bool = False

@dataclass
class MultiFlowState(BTAState):                   # mirrors class MRO
    last_winner_idx: int | None = None
    latest_per_flow: dict = field(default_factory=dict)
    all_judgments: list = field(default_factory=list)

@dataclass
class MFDualState(InferencerStateBase):           # COMPOSITION (no diamond)
    flow: MultiFlowState = field(default_factory=MultiFlowState)
    dual: "DualState" = field(default_factory=lambda: DualState())

# on each inferencer (generalizes initial_state_factory):
state_factory: Callable[[Any], InferencerStateBase] | None = attrib(default=None, kw_only=True)
```
`RunContext.call["state"]` (per node) holds the typed object the node's `state_factory` built; compat
properties (`self._last_winner_idx` → `self._active_ctx.call["state"].last_winner_idx`) preserve every
existing read/set and AF1.

---

## §E — Execution (PART I)

> **Invariant for EVERY commit:** the **no-`run_context` path is byte-identical to today** (legacy mint).
> The full existing inferencer suite is the regression gate after each commit. Land the *contract* (E1–E3)
> first so it's locked; virtualize fields (E4–E7) incrementally; each is independently revertible.

### §E1 — `RunContext` container + legacy mint (pure addition)
- Add `RunContext` dataclass (`§2.2`) in a new `inferencers/run_context.py`.
- Add `_active_ctx: RunContext | None = None` instance field + `_ensure_ctx(run_context)` helper on
  `InferencerBase`: returns the passed context, else mints a legacy one from `self._workspace`.
- **No signatures changed yet.** Test: construct, `child()` composes paths, legacy mint mirrors
  `self._workspace`.

### §E2 — Thread `run_context` through the public API (additive kwarg)
- Add `run_context: RunContext | None = None` to public `infer`/`ainfer` (`inferencer_base.py:1643,2180`)
  and the `_infer`/`_ainfer` internals (`:943,1837`). Default `None` → `_ensure_ctx` → legacy.
- Store the resolved context as `self._active_ctx` for the duration of the call (set at entry, cleared in
  `finally`) so existing internal reads can find it during migration.
- **CRITICAL (kwarg-leak guard, integrated from Codex R-leak):** the base layer MUST **pop/consume**
  `run_context` before any `**_inference_args` is forwarded to `_inference_api(...)` — today unknown kwargs
  are passed through to the model/provider call, so a leaked `run_context` would crash or corrupt the API
  request. Add a failing test with a dummy API inferencer that rejects unknown kwargs.
- **Behavior unchanged** (nothing reads `call`/`attempt` yet). Test: pass an explicit context → same result
  as no context; `self._active_ctx` is `None` after the call; `run_context` never reaches the API call.

### §E3 — Host mints the root context (`task` executor)
- In `executor.py` after `instantiate(cfg)` (`:854`) and `working_dir` resolution (`:672`), mint
  `root = RunContext(run_id=task_id, parent_run_id=None, workspace=Workspace(working_dir),
  graph_reporter=…, interactive=…)` and call `inferencer.ainfer(request, run_context=root)` (`:875`).
- Mirror in the `sop` host and the conversational host (so all three top-level entry points seed a root).
- Test: a `task` run threads one root context end-to-end; provenance `run_id == task_id`.

### §E4 — Children allocate `ctx.child(path)` (the load-bearing API)
- BTA `_make_worker_fn`/`_make_agg_fn` (`breakdown_then_aggregate_inferencer.py:1715,1937`): each worker/
  aggregator closure calls `child_ctx = run_context.child(node_id)` and passes it into `w.ainfer(q, …,
  run_context=child_ctx)`. (Pairs with Part-1 A1's `worker_extra_feed`.)
- MFDual/Dual/PTI: each sub-step threads `run_context.child(<step>)` into the child `ainfer`.
- **Workspace still assigned the old way too** (E4 only *adds* the context; E6 makes it authoritative).
- Test: nested run produces a provenance tree; child contexts have isolated buckets.

### §E5 — Typed per-class state via `state_factory`, populated into the context (the §2.4 enhancement)
- Add `InferencerStateBase` + per-class typed states (`BTAState`/`MultiFlowState`/`DualState`;
  `MFDualState` via **composition**, not MI — Correction 3) in `inferencers/states.py`.
- Add `state_factory: Callable[[Any], InferencerStateBase] | None = attrib(default=None, kw_only=True)` on
  the relevant inferencers, **generalizing the shipped `initial_state_factory`** (`linear_workflow_inferencer.py:156`).
  Default `None` → a built-in default factory returns the class's empty typed state (legacy-equivalent).
- At call entry, build the node's typed state via `state_factory(input)` and store it **into the context**:
  `run_context.call["state"] = <typed>` (Reading B — Correction 1; **never** parked on the instance as the
  durable carrier).
- Convert `_last_winner_idx`/`_last_ranking`/`_latest_per_flow`/`_all_judgments`/`_cached_sub_queries`/
  `_graph_topology_emitted` to **compat properties** delegating to `self._active_ctx.call["state"]`
  (per-call) / `self._active_ctx.attempt["state"]` (per-attempt).
- Re-point `_reset_dispatch_state_for_call` → reset the call-state; `_reset_cross_flow_state` → reset the
  attempt-state (each stays at its existing site), **gated by the AF4 retry test** (winner survives
  malformed retry).
- **AF1 preserved:** the compat property reads/writes through `self._active_ctx`, which is the legacy mint
  when no context is threaded — so the 15+ direct-set unit tests and the post-call `get_winner_flow_idx()`
  still work (the legacy context persists on the instance until the next call).
- **YAML:** `state_factory: {_target_: …, <params>}` flows through the existing walker (`_instantiate.py:619+`).
- Test: AF4 retry baseline (before) == (after); all dispatch unit tests green; a YAML-supplied
  `state_factory` builds the expected typed state; `MFDualState` composition round-trips `to_json`/`from_json`.

### §E5b — Stop mutating DEFINITION fields: effective-query accessor (integrated from Codex)
- **Problem (verified):** MultiFlow mutates `flow_configs[i]["input"]` and `predefined_sub_queries` at runtime
  so BTA sees runtime inputs — i.e. it **mutates the definition**, which breaks the separation and is unsafe
  under shared instances.
- **Fix:** add `_get_effective_predefined_sub_queries(run_context)` returning
  `run_context.attempt.get("predefined_sub_queries", self.predefined_sub_queries)`; write the runtime value
  into `attempt` (never into `flow_configs`/`predefined_sub_queries`). BTA reads via the accessor.
- Test: under an explicit context, `flow_configs`/`predefined_sub_queries` are **provably not mutated**;
  BTA still sees the effective runtime sub-queries.

### §E6 — Make `workspace` authoritative from the context (the big one — last, most careful)
- `_workspace` becomes a **property** reading `self._active_ctx.workspace` (falling back to the legacy
  field when no active context). The `_workspace` *setter* still triggers `_configure_for_workspace`
  cascade — now it writes into the active context's workspace slot.
- **Transitional bridge (integrated from Codex):** for modules not yet fully context-aware, allow a
  **guarded** `with self._legacy_bind_context(run_context): …` that temporarily syncs
  `self._workspace = run_context.workspace`. **Explicitly marked as a bridge, NOT concurrency-safe** — any
  code using it MUST retain factory/fresh-instance isolation. Each migrated method either reads
  `run_context.workspace` directly (preferred) or uses the bridge (temporary).
- MFDual `_reassign_role_workspace` (`:424-464`) and `switch_role` (`inferencer_base.py:398`) write the
  child workspace into the child context instead of mutating the shared instance.
- **This is the step that finally makes a shared instance safe** (workspace no longer lives on the shared
  object). Gate: the 143-read workspace surface (AF2) — full suite + the MFDual workspace-anomaly
  integration tests must stay green; remove each `_legacy_bind_context` use only when its module reads the
  context directly.

### §E6b — Tier-3 worker-local live handles + concurrency stress (NEW in v5, from V7/V8)
- Make ClaudeCode `_client`/`_session_id` and RovoDevServe `_server_process`/`_http_client` resolve from the
  context's `LiveHandles` (Tier 3), established via the **existing `aconnect` lifecycle** and **reused across
  calls** in a worker branch (NOT per-call — V7). Each concurrent worker gets its own via
  `child(..., fresh_handles=True)`.
- **Canary first:** convert ONE leaf with BOTH a concurrency stress test (N gather'd workers, no
  `_session_id`/subprocess cross-talk) AND a multi-turn continuity test (session_id survives across calls,
  no subprocess relaunch) before generalizing. This is the single highest-risk correctness step.

### §E6c — Conversational state → Tier-1 `conversation` (NEW in v5, from V9)
- Map `_serialize_pause_state`/`_restore_pause_state` to read/write `ctx.state.conversation` (the existing
  dict form — `_messages`/`prior_context`/`sop_state.to_dict()`/`_suspended_sops`/`dynamic_context.to_dict()`).
  **Keep `SOPState`/`AgenticDynamicContext` as dicts** (they already have `to_dict/from_dict`) — do NOT typed-ify.
- Test: pause→resume a conversational run rehydrates from Tier-1 `conversation` (NOT workspace); SOP phase
  position + suspended SOPs + dynamic vars all survive.

### §E7 — Resume via the context (simplification payoff)
- Replace the scattered checkpoint fields with a single `RunStateStore.to_json()`/`from_json()`; BTA's
  `resume_with_saved_results` reconstructs child contexts by path; conversation rehydrates from Tier-1
  `conversation`; Tier-3 handles re-established via `aconnect`; Tier-2 supplied fresh by the app.
- Test: kill-and-resume a BTA mid-fan-out → identical final output; resumed contexts carry the same
  `run_id` tree.

### §E-purity — Per-class factory retirement gate (v6: corrected per verified V4 + purity-flaw)
- **Factory STAYS** (reverses any "factory dies now"): role still mutates `self` (V1 `switch_role`
  `setattr(self,...)` at `templated_inferencer_base.py:384,404,407`; V2 Dual/MFDual
  `self.review_inferencer._workspace=` at `dual_inferencer.py:1088,1294`, `multi_flow_dual_inferencer.py:464`),
  so a shared instance is not concurrency-safe yet.
- **⚠️ Fail-fast guard is NEW WORK, not a flag flip (v6 correction, verified):** the BTA method is
  **`_validate_worker_isolation`** and it **only warns, never raises** — `if not self.worker_isolation_check:
  return` (`breakdown_then_aggregate_inferencer.py:1293-1294`); the duplicate-`id()` branch calls
  `_logger.warning(...)` (`:1302`); there is **no `raise`** in `:1281-1312` (field default
  `worker_isolation_check=True`, `:524`). v5 wrongly described this as "promote warning→raise" as if a raise
  existed — it must be **added** (per-class, gated behind the §E-guard), the exact error this plan flagged in
  the peer plans. **Recommended scope:** per-class *warn-until-converted* (not a global raise that could break
  out-of-tree consumers on the legacy path).
- **Per-class retirement:** a class drops `worker_factory`-for-isolation + `_validate_worker_isolation` ONLY
  after it passes a **purity test**. Role virtualization (`switch_role` → write a `RoleState` into
  `ctx.call.state`) is the gating work.
- **⚠️ The purity test must be designed to avoid a FALSE-FAIL (v6 correction, verified):** a naive
  "snapshot `vars(inferencer)` before/after a context-driven `ainfer` and assert empty diff" **will false-fail
  during the compat window** — the `_workspace` **property setter** writes the name-mangled
  `_InferencerBase__workspace` into `__dict__` and **cascades** via `_configure_for_workspace` (writes
  `cache_folder`, normalises/redirects loggers) on every call (`inferencer_base.py` setter +
  `_configure_for_workspace`). Therefore the purity assertion is only valid **(a) AFTER per-class retirement
  of the legacy-mint backing-write, run with an EMPTY allow-list** — i.e. once the class no longer writes any
  runtime field to `self` under a context. Until then, use an explicit allow-list of the known legacy-mint
  writes (`_InferencerBase__workspace`, `cache_folder`, logger fields, `_active_ctx`) and shrink it to empty
  as virtualization completes. The empty-allow-list pass is the actual certification.

### §E-Part1 — Multi-reviewer feature (inherited UNCHANGED from `swift-launching-backus.md` v7)
Ships independently with **factory isolation** (A1 `worker_extra_feed` with `pop` semantics; A2 deterministic
`merge_reviews`; A3 accessor seam as thin getters; A4 panel `switch_role`; A5 `review_inferencer_factory`;
m3 non-templated-worker fail-fast). It does **not** depend on E4–E7; when E4 lands, review workers simply
also receive a `child_ctx`. **Recommended sequencing: ship Part 1 first (feature value, low risk), then E1→E7.**

### §E-TEST — Testing & migration strategy (this is a broad refactor — verified surface)

> **Scope reality (measured, not guessed):** the refactor touches the **base class every inferencer
> inherits**, so the blast radius is the whole subsystem. Verified counts:
> **167 inferencer test files**; **~1,379 instance-state references** in tests; **179 direct constructor
> calls** + **62 YAML `instantiate()`**; **25 real-LLM integration tests** (56 files assert on
> workspace/output paths); and critically **108 tests do `self._workspace = …` directly** + **12 set
> `_last_winner_idx`**, **14 touch `_cached_sub_queries`**, **10 read `get_winner_flow_idx()`/`get_ranking()`
> post-call**. These numbers drive the strategy below.

**Guiding principle — "refactor, not rewrite": the vast majority of existing tests MUST pass UNCHANGED.**
The legacy-default (`run_context=None`) + **compat properties with working setters** are precisely what
makes that possible. A test that breaks is a signal the compat shim is incomplete, not that the test is wrong.

**T0 — Pre-flight baseline (before any code).** Snapshot the green suite + record the AF4 retry baseline +
capture the 25 real-LLM tests' asserted workspace/output paths. This is the regression oracle.

**T1 — Compat properties MUST be read/write (the ~107-writes finding).** `_workspace`, `_last_winner_idx`,
`_cached_sub_queries`, … become properties with **both getter and setter** delegating to the active context
(legacy mint when none). The ~107 `self._workspace = X` test writes (83 in the inferencers subdir, 107 whole
tree) and 12 `_last_winner_idx = N` + 6 `_cached_sub_queries =` writes keep working verbatim.
**⚠️ Precondition (v6, verified):** `_workspace` is **already** a property+setter, but `_last_winner_idx`
(attrs `attrib`, `multi_flow_inferencer.py:304`) and `_cached_sub_queries` (plain attribute,
`breakdown_then_aggregate_inferencer.py:612`) are **not** — converting them needs a **name-mangled backing
field + setter**, which requires the owning `@attrs` class to be **`slots=False`**. Verified safe: all owners
are `slots=False` (`BreakdownThenAggregateInferencer` `@attrs(slots=False)` `:292`; `MultiFlowInferencer`
`@attrs(slots=False)` `:143`; `DualInferencer`/`InferencerBase`/`TemplatedInferencerBase` default `slots=False`).
A future class added with `slots=True` would break this — add a one-line `slots` assertion to the conversion.
**New test:** a parametrized "compat-field round-trips through the property" test for every migrated field
(set→get equality, with and without an explicit context).

**T2 — Per-commit gate = full existing suite + targeted additions.** After E1–E7 each: run the **entire**
inferencer suite (167 files). Plus the **new** tests each step introduces:
- E1: `RunContext`/`RunStateStore`/`RuntimeBindings` unit tests (`child()` path composition; `node()`
  idempotency; `to_json` excludes `RuntimeBindings`).
- E2: kwarg-leak test (dummy API rejects unknown kwargs ⇒ `run_context` must be popped); explicit-context ==
  no-context equivalence.
- E3: host seeds a root (task/sop/conversational) — one test each asserting `run_id`/provenance.
- E4: nested run builds a provenance **tree**; child contexts have isolated `call`/`attempt`.
- E5: AF4 two-tier retry (winner survives malformed retry) before==after; all 12 `_last_winner_idx` + 10
  post-call getter tests green; YAML `state_factory` builds the typed state; `MFDualState` composition
  `to_json`/`from_json` round-trip.
- E5b: under explicit context, `flow_configs`/`predefined_sub_queries` **provably not mutated**.
- E6: the 108 `_workspace=` writes still work; **56 real-LLM path-asserting tests green**; MFDual
  workspace-anomaly integration tests green; each `_legacy_bind_context` removed only when its module reads
  the context directly.
- E7: kill-and-resume a BTA mid-fan-out ⇒ identical final output; resumed store rebuilds by path.

**T3 — NEW tests the refactor warrants (net-new capability, not just regression).**
- **Concurrency-isolation test:** two `asyncio.gather`'d runs through fresh-instance workers do NOT cross
  state (the property the design promises). Add a deliberately-shared-instance variant that the §E-guard
  **rejects**.
- **Resume golden test:** serialize `RunStateStore` mid-run, reconstruct, finish ⇒ byte-identical output.
- **Provenance test:** assert the path-keyed tree (`/round_01/review/worker_0`) matches the workspace tree.
- **Definition-immutability test:** after a full BTA-multiflow run under explicit context, the YAML-built
  definition objects are unchanged (no `flow_configs`/`predefined_sub_queries`/role mutation persisted).

**T4 — Examples (58 files; ~27 call `ainfer`/`infer`; none are in CI today).**
- **Must keep working unchanged under legacy-default** (they pass bare positional args — no `run_context`
  collision; verified). No edits required for correctness.
- **Add 3–5 as END-TO-END acceptance** (promote to CI): `example_rovodev_non_legacy_streaming.py`,
  `example_ag_streaming.py`, a BTA/DAG example (`example_diamond_dag_with_variable_passing.py`), and
  `example_05_run_yaml_configs.py` (YAML `state_factory` path). Update **one** new example
  `example_runcontext_explicit.py` showing host-minted context + provenance (documents the new contract).

**T5 — OpenStartup (measured: near-zero blast radius).** Verified: OpenStartup builds CIs via
`factories.py:_wrap_in_conversational → build_ci_from_config` and calls `ainfer(input)` at tool executors
(`role_setup`, `create_role`, `project_onboarding`); it **never reads** `_workspace`/`session`/
`get_final_output`/`_last_clean_output` in tracked production code (calls pass a single positional input +
keyword args only). **⚠️ v6 correction:** OpenStartup has **73 `test_*.py` files** (v5 wrongly said "112"); and
the only apparent "internal reads" are in `_runtime/tasks/…`, which are **gitignored runtime artifacts, not
real coupling**. **With the legacy-default + keyword-only `run_context`, OpenStartup needs ZERO changes** and
its 73 tests (incl. `test_real_session_with_rovodev_cli.py`) must pass untouched — that is itself an
**acceptance gate** for "legacy path unchanged."
- **Ideal (optional, separate commit):** mint a **root `RunContext` at the server session/turn boundary**
  and thread it through `build_ci_from_config` → `ainfer(run_context=root)`, so cross-tool provenance +
  resume work end-to-end. Gated behind the same legacy fallback; add **one** OpenStartup test asserting the
  session root `run_id` flows into a tool's child context.

**Definition of done (testing):** (a) all 167 AF inferencer tests + 25 real-LLM tests green with **no
behavioral edits** (only compat-property mechanics); (b) the T3 net-new tests green; (c) 3–5 examples in CI;
(d) OpenStartup's 73 tests green untouched (legacy proof) + 1 optional root-context test if the ideal path
is taken.

### §E-guard — Shared-instance fail-fast (closes the discussion's sharp edge)
At panel/worker assembly, detect duplicate **object identity** among active sibling slots and **raise**
unless explicitly opted-in; the supported path is **factory → fresh instances** (F5/F7). This guard is
valuable **before** E6 (when sharing is still unsafe) and harmless after.

---

## §A — Appendix

### §A1 — Source-verified facts
- **F-ctx1:** public `ainfer`/`infer` already accept `**_inference_args` → adding `run_context=` is
  non-breaking. `inferencer_base.py:2180,1643`.
- **F-ctx2:** internals `_infer`/`_ainfer` at `:943,1837`; 69 `*infer*` signatures total (bounded blast radius).
- **F-ctx3:** task host instantiates topology (`executor.py:854`) then `inferencer.ainfer(request)` (`:875`)
  after `working_dir` allocation (`:672`) — the natural root-context mint site.
- **F-ctx4:** the instance genuinely mutates run-state today: `_workspace` cascade, `switch_role`
  (`inferencer_base.py:398`), MFDual role mutation (`multi_flow_dual_inferencer.py:1084-1088`), MultiFlow
  `flow_configs[i]["input"]`. ⇒ the "kwarg ≠ full separation" caveat is real (E6 is required).
- **F-ctx5:** BTA already mandates fresh-per-worker factories and raises otherwise
  (`breakdown_then_aggregate_inferencer.py:1300-1307`) ⇒ "factory = definition" is already the idiom.
- **F-ctx6 (state_factory precedent — VERIFIED):** `LinearWorkflowInferencer.initial_state_factory:
  Optional[Callable] = attrib(default=None)` (`linear_workflow_inferencer.py:156`), called
  `self._pending_state = self.initial_state_factory(inference_input)` (`:920-921`), already supplied via
  lambda/YAML (`tool_as_inferencer.py:629`). ⇒ §2.4's `state_factory` **generalizes a shipped field**.
- **F-ctx7 (YAML factory shape — VERIFIED):** the `_target_`-with-nested-children → kwargs walker + `_partial_`
  (`_instantiate.py:614,619+`) makes `state_factory: {_target_: …, <params>}` work natively; `worker_factory`
  in `breakdown.yaml:113-153` already uses exactly this shape (auto-wrapped `LazyConfigFactory`).
- **F-ctx8 (per-class state inventory — VERIFIED):** state nests with the class MRO; `MFDual` derives from
  both `Dual` and `MultiFlow` ⇒ **composition required** (Correction 3); `_workspace`/`session`/
  `graph_reporter`/`interactive` are transient handles, **not** serializable state.
- **F-ctx9 (V7/V8 — live handles are CONNECTION-scoped, re-verified this round):** ClaudeCode lazily
  creates one `self._client` and reuses it across calls, writes `self._session_id` from each result and feeds
  it into the next call, tears down at `adisconnect` (`claude_code_inferencer.py:306,308,189,312-319`);
  RovoDevServe keeps one `self._server_process` + `self._http_client` created at `aconnect`, reused
  (`rovodev_serve_inferencer.py:106-137,263,178-199,217-220`). ⇒ Tier-3 handles must be **worker-local /
  connection-scoped, NOT per-call** — the basis for §2.1 Tier 3 + §E6b. Tearing down per call would relaunch
  the subprocess and break `session_id` continuity.
- **F-ctx10 (V9 — conversational state is serialized INSTANCE state, re-verified):** `_serialize_pause_state`
  serializes `_messages`, `prior_context`, `sop_state.to_dict()`, `_suspended_sops`, `dynamic_context.to_dict()`
  (`conversational_inferencer.py:956-967`). It is **independent in-memory state, NOT workspace-derivable** ⇒ a
  separation moving only `_last_*/_cached_*/_workspace` would MISS it (basis for §E6c). `SOPState` &
  `AgenticDynamicContext` already have `to_dict/from_dict` (`sop_state.py:82`, `context.py:74,84`) ⇒ keep
  their dict form; do NOT force typed `InferencerStateBase` (refines §2.4 enthusiasm).
- **(inherited)** AF1/AF2/AF3/AF4 + F1–F11 from the prior plans (re-verified there).

### §A2 — Risks
- **R1 (E6 workspace virtualization):** **highest risk — now quantified: 143 production reads + 108 TEST
  write-sites (`self._workspace = …`) + the `_configure_for_workspace` cascade.** The 108 writes mean the
  property MUST keep a **working setter** (T1); a read-only property would break 108 tests. Mitigation:
  read/write property with legacy fallback + `_legacy_bind_context` bridge; land last; full suite + the 56
  path-asserting real-LLM tests + MFDual workspace-anomaly tests gate it (T2/E6).
- **R2 (AF4 two-tier in the context):** mitigated by re-point-before-delete + the retry baseline test (E5).
- **R3 (blast radius, 69 signatures):** mitigated by the additive kwarg + legacy default (E2) — no flag day.
- **R4 (host coverage):** all three hosts (task/sop/conversational) must seed a root, or nested children
  silently fall back to legacy mints (still correct, just no provenance). Mitigation: E3 covers all three.
- **R5 (kwarg leak to model API — Codex R1):** `run_context` must be popped before `**_inference_args`
  reaches `_inference_api`. Mitigation: base-layer pop + a dummy-API-rejects-unknown-kwargs test (E2).
- **R6 (god-object — Codex R2):** keep `NodeRunState` (serializable) and `RuntimeBindings` (transient)
  structurally separate; never store live inferencer handles in persisted state. Mitigation: the §2.2 split
  + a `to_json` test that fails if a non-serializable handle is present.
- **R7 (half-migration two-source-of-truth — Codex R3):** each migrated field gets exactly one compat
  accessor; grep for raw reads after each step; assert definition fields (`flow_configs`) are not mutated
  under explicit context.
- **R8 (Tier-3 lifetime correctness — NEW v5, V7/V8):** live handles MUST be connection-scoped/worker-local,
  not per-call; a naive call-local model relaunches subprocesses and breaks `session_id` continuity.
  Mitigation: §E6b canary — convert ONE leaf with BOTH a concurrency stress test AND a multi-turn-continuity
  test before generalizing.
- **R9 (conversational resume — NEW v5, V9):** `_messages`/`sop_state`/`dynamic_context`/`_suspended_sops`
  must map into Tier-1 `conversation`; a separation that touches only `_workspace`/`_last_*`/`_cached_*`
  silently breaks pause/resume. Mitigation: §E6c pause→resume test rehydrating from Tier-1, not workspace.

### §A3 — Open questions (each with a default)
- **Q1** Should `_active_ctx` be a ContextVar instead of an instance field, for one-instance concurrency?
  *Default:* **no** — isolation is fresh-instance-per-worker (F5); an instance field + legacy fallback
  preserves AF1 (post-call/out-of-call reads). (This is exactly the carrier lesson from the prior lineage.)
- **Q2** Migrate `session` into the context too? *Default:* **not in v1** — cross-round live handle, high
  leak risk; only an `active_session_id` string if resume needs it.
- **Q3** Ship Part 1 before or after E1–E3? *Default:* **before** — feature value now; contract lands in
  parallel without blocking.

### §A5 — If we only pick ONE plan (honest, updated for Claude v10)
**If forced to pick ONE document today: Claude's `swift-launching-backus.md` (v10).** This is a change from my
prior answer, and I own the reason: between rounds, that plan absorbed *all three* plans, and — verified
against live source this round — it got **two things right that my v4 got wrong or omitted**:
- **The three-tier model with worker-local `LiveHandles` (V7/V8).** A live SDK client / subprocess / session
  is **connection-scoped, reused across calls** — so it must be a distinct worker-local tier, not lumped with
  shared sinks. My v4 had only two tiers and **even placed a shared `session_store` in the transient
  bindings** — the exact latent concurrency bug Claude flagged. (Now fixed here in v5 §2.1/§2.2.)
- **Conversational state is serializable instance state (V9)** that a `_workspace`/`_last_*`/`_cached_*`-only
  separation would silently drop — and `SOPState`/`AgenticDynamicContext` should stay dicts, *not* be forced
  into the typed hierarchy. My v4 missed this entirely. (Now fixed here in v5 §E6c.)
- It also correctly **keeps the factory until role is virtualized** with a **per-class purity gate** — more
  rigorous than my v4's hand-wave. (Now incorporated here in v5 §E-purity.)

**With this v5 update, this `revodev/` plan is now at parity** — it carries Claude's three-tier model + V7/V8/V9
corrections **plus** two things Claude's plan still lacks: (a) the measured **§E-TEST testing/migration
strategy** (167 test files, the 108 `_workspace=` write finding, OpenStartup-needs-zero-changes), and (b) the
**typed `state_factory`** enhancement (the user's latest idea) reconciled honestly (typed for *new*
orchestration state; dict for conversational). Credit remains shared: **Codex** = path-keyed `RunStateStore` +
`RuntimeBindings`/`NodeRunState` split + effective-query accessor + kwarg-leak guard; **Claude v10** = three-tier
resource model + V7/V8/V9 verification + factory/role purity gate; **this plan** = `state_factory` + §1
adjudication + §E-TEST + legacy-default-every-commit discipline.

**Bottom line:** pick **this v5** to *execute from* (it's the superset incl. the test/migration strategy);
pick **Claude v10** if you want the single leanest already-adversarially-reviewed design narrative. They now
describe the *same architecture* — the remaining difference is breadth (this) vs. narrative economy (Claude).

### §A4 — Changelog
- **v6 (2026-06-19):** **Folded in Claude `swift-launching-backus.md` v11's four corrections, each
  re-verified against live source by parallel agents** (per the "don't blindly trust" rule). (1) **BTA guard
  is warn-only:** `_validate_worker_isolation` has no `raise` in `:1281-1312` (only `_logger.warning` at
  `:1302`; `worker_isolation_check=True` `:524`) ⇒ §E-purity now states fail-fast is **new work**, scoped
  per-class warn-until-converted. (2) **Purity test false-fail:** the `_workspace` setter writes
  `_InferencerBase__workspace` + cascades (`_configure_for_workspace` → cache_folder/loggers) into `__dict__`
  every call ⇒ §E-purity now requires the assertion run **post-retirement with an empty allow-list** (shrinking
  allow-list during the window). (3) **Compat-property precondition:** `_last_winner_idx`
  (`multi_flow_inferencer.py:304`) + `_cached_sub_queries` (`bta:612`) are plain attrs, not properties ⇒ need
  name-mangled backing + setter + a **`slots=False`** precondition; verified all owners are `slots=False`
  (`bta:292`, `mfi:143`, others default). (4) **Measurement honesty:** OpenStartup is **73 test files** (v5's
  "112" was wrong) and `_runtime/tasks` reads are gitignored artifacts, not coupling — fixed in §0/T5/DoD.
  Added these as verified notes; bumped to v6. (Net effect: this plan and Claude v11 are now full parity.)
- **v5 (2026-06-19):** **Integrated Claude `swift-launching-backus.md` v10 — upgraded to a THREE-tier state
  model** after re-verifying its V7/V8/V9 claims against live source (both CONFIRMED). (1) Added **Tier 3
  `LiveHandles`** (worker-local, connection-scoped SDK client/subprocess/httpx/`live_session_id`) — verified
  `claude_code_inferencer.py:306,308,189,312-319` + `rovodev_serve_inferencer.py:106-137,263,178-199`; this is
  the only model that fixes the BTA `gather()` collision without per-call subprocess relaunch. (2) **Removed
  the shared `session_store` from `RuntimeBindings`** (Tier 2 = concurrency-safe sinks ONLY) — that was a real
  latent bug in v4. (3) Added **`NodeRunState.conversation`** + §E6c: conversational state (`_messages`/
  `sop_state`/`dynamic_context`/`_suspended_sops`) is serializable instance state, NOT workspace-derivable
  (verified `conversational_inferencer.py:956-967`); **keep `SOPState`/`AgenticDynamicContext` as dicts** (do
  not typed-ify — refines §2.4). (4) Added **§E-purity** (factory stays until role virtualized; per-class
  `vars()` snapshot purity gate; promote `worker_isolation_check` warning→raise). (5) Added facts F-ctx9/F-ctx10
  + risks R8/R9; **updated §A5 pick-one** to honestly name Claude v10 as the single-doc pick while bringing
  this plan to architectural parity (superset via §E-TEST + `state_factory`). Bumped to v5.
- **v4 (2026-06-19):** **Added §E-TEST — full testing & migration strategy**, after measuring the real
  surface (the user asked whether tests/examples are covered and whether AF + OpenStartup get fully tested).
  Verified counts (167 inferencer test files, ~1,379 state refs, 179 direct ctors, 62 YAML instantiate, 25
  real-LLM tests, **108 `self._workspace=` test writes**, 12 `_last_winner_idx` writes, 10 post-call getter
  reads). Key finding folded in: **compat properties must be read/WRITE** (108 writes) — a read-only property
  would break 108 tests; R1 re-quantified accordingly. Strategy = "refactor, not rewrite" (existing tests
  unchanged via legacy-default + r/w compat props) + T3 net-new tests (concurrency/resume/provenance/
  definition-immutability) + T4 examples (3–5 promoted to CI) + **T5 OpenStartup needs ZERO changes**
  (verified; its tests are the legacy-unchanged gate [count corrected to 73 in v6]; optional session-boundary root context as a
  separate commit). Added §0 surface note. Bumped to v4.
- **v3 (2026-06-19):** **Integrated the Codex peer plan** (`codex/run_context_definition_state_separation_plan.md`).
  Adopted its **path-keyed `RunStateStore` + node-scoped `RunContext` view** (R3), the **`RuntimeBindings`
  (transient) vs `NodeRunState` (serializable) split** (R6) — rewrote §2.2; the **effective-query accessor**
  to stop `flow_configs`/`predefined_sub_queries` definition-mutation (new §E5b); the **`_legacy_bind_context`
  guarded bridge** for the 143-read `_workspace` migration (§E6); and the **kwarg-leak guard** (E2 + R5).
  Kept this plan's **typed `state_factory` states (§2.4)** as the *value type inside* the path-keyed store
  (Codex used bare dicts). Added R5/R6/R7 risks + §A5 pick-one. Bumped to v3.
- **v2 (2026-06-19):** Added §2.4 — typed per-class `InferencerStateBase` + `state_factory` enhancement, in
  response to the user idea "each inferencer has a state class + `state_factory` field with YAML-supplied
  params." Honest verdict recorded: **sound and largely already supported** (verified `initial_state_factory`
  precedent F-ctx6, YAML factory shape F-ctx7, state inventory F-ctx8) — adopted with **three corrections**:
  (1) `state_factory` populates the **threaded `RunContext`**, not a parked-on-instance `self._state`
  (Reading B); (2) "no factory needed after separation" is only true once workspace+role also move (E6) —
  factories stay mandatory for concurrent siblings until then; (3) `MFDualState` via **composition**, not
  multiple inheritance; transient handles never go in serializable state. Rewrote §E5 to build typed state
  via `state_factory` into the context; bumped status to v2.
- **v1 (2026-06-19):** New plan in `revodev/`. Adopts explicit host-minted `RunContext` (definition=YAML/
  factory; run-state=context; `context.child()` for descendants) after the §A1 design discussion. Inherits
  Part-1 multi-reviewer unchanged from `swift-launching-backus.md` v7. Honest caveat recorded: a
  `run_context` kwarg alone is NOT full separation (E6 workspace virtualization is required); sequenced
  behind a legacy default so the framework runs at every commit. Supersedes the instance-held `self._run`
  carrier for the state-separation half only.

---

> *Note: the full design-discussion transcript is the user-provided context for this plan; key claims are
> distilled and individually adjudicated in §1, and load-bearing code facts are verified in §A1.*
