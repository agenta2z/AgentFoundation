# Integrated Plan v2: Definition-Stable Inferencers and Invocation-Owned Runtime

## 0. Decision

The requested direction is correct, but the precise target is **definition stability**, not literal object immutability:

> Under a host-owned `RunContext`, an inferencer call must not write call-specific semantic or live execution state into the inferencer definition or into another inferencer definition. Definition-derived caches and deliberately connection-scoped handles remain valid, but their ownership must be explicit and testable.

The implementation should use five state homes plus one transport layer:

| State kind | Lifetime | Correct owner |
|---|---|---|
| Definition and pure definition cache | construction to destruction | inferencer instance and canonical construction recipe |
| Invocation and retry-attempt runtime | one logical call or one attempt | unique `InvocationFrame` and typed runtime components |
| Durable semantic result | run, save, and resume | typed `NodeRunState` fields and orchestrator state |
| Connection/session continuity | `connect` to `disconnect`, possibly across calls | `LiveHandleStore`, keyed by context path, with explicit bare-call compatibility |
| Restart journal and external artifacts | workspace generation | versioned checkpoint files and a BTA execution manifest |
| Shared services | run | existing `RuntimeBindings`; this transports reporters, cancellation, and interaction but is not a semantic state store |

This replaces the overly broad rule "move mutable state to `RunContext`." A `NodeRunState` is keyed by path and can be revisited by retry, round, or resume. It is not a safe owner for a live graph or two simultaneous calls at the same path. Live graph objects, queues, closures, child ledgers, and transient caches belong to a unique invocation or attempt.

The target shape is:

```text
Inferencer definition
  config + canonical recipe + pure caches
  no call-derived state under a host context

InvocationFrame                                  one per single-input call
  call_id, owner, exact RunContext, runtime registry, cleanup registry
  AttemptFrame                                   one per retry attempt
    BtaAttemptRuntime                            fresh graph and live stages

NodeRunState                                     serializable facts
  call, attempt, role_state, task_contract, provenance, checkpoints

LiveHandleStore                                  connection-scoped resources
  provider clients, conversations, subprocesses

Workspace generation                             durable restart truth
  manifest, breakdown, expansion, worker and aggregator checkpoints
```

## 1. Verified Current State

This plan was reconciled against the current executed fan-out stack and the live source, not against older plan claims.

### 1.1 Correct behavior already present and preserved

1. A host with `bta_inferencer` renders once and creates a fresh self-fan-out BTA representative per call.
2. Self-fan-out workers are parent-derived fresh instances and receive prepared shards.
3. Indexed worker outcomes prevent asynchronous completion order from mis-pairing worker results.
4. Child workspace reads use the child's context slot.
5. Breakdown, worker, and aggregator checkpoints support partial resume.
6. Context-scoped roles influence rendering and fan-out role mapping.
7. Aggregator feed publication is context-scoped rather than written into the runtime aggregator feed.
8. The MetaMate research-propose rollout remains default-off.
9. Paid MetaMate gates G1-G6 remain outside this refactor unless separately approved.

### 1.2 Confirmed correctness and purity defects

| ID | Current defect | Consequence |
|---|---|---|
| C1 | `RoleState` is written to `node.call` even though `NodeRunState.role_state` exists | It can overwrite `BTAState`, `MultiFlowState`, `MFDualState`, or another typed call state |
| C2 | `switch_role()` appends `_role_history` and uses `_pending_role_changes` on `self` under a host context | Shared definitions accumulate and interleave run audit state |
| C3 | `_last_rendered_task_instructions` and BTA `_worker_task_instructions` are object-held "last call" values | Parents can read another concurrent or previous call's contract; current BTA selection is completion-order-sensitive |
| C4 | BTA remains both reusable definition and mutable `WorkGraph` | Overlapping calls can read another call's graph and silently skip expansion or produce wrong output |
| C5 | `build_aggregator()` writes a built instance back into `aggregator_inferencer` | A factory/config slot stops being a definition after the first call |
| C6 | BTA retains current query, promoted breakdown memo, workers, topology flags, guidance, and graph configuration on `self` | Reuse leaks state, disconnect can target only the last call, and concurrency is unsafe |
| C7 | The guardrail fingerprint window and `_last_inference_input` live on the inferencer | Retry behavior can leak across calls |
| C8 | Guardrail setup assigns `judge.template_manager = None` | One call permanently mutates a potentially shared judge definition |
| C9 | `_propagate_to_children()` merges parent feed and modes into child instance fields | Two parents sharing a child can contaminate each other's definitions |
| C10 | BTA writes reporter/interactive/runtime configuration into stage objects | A definition is modified for one execution |
| C11 | `WorkGraph._run()` uses `ThreadPoolExecutor.submit(asyncio.run, ...)` without copying `ContextVar` state | Sync execution invoked from a running event loop can lose `RunContext` and the future invocation frame |
| C12 | The purity helper ignores removed keys and `assert_pure()` compares two immediate snapshots | The enforcement tool can miss mutation or report success without spanning a call |
| C13 | Some streaming/session leaves bypass context-aware handle properties or retain per-call accumulators on `self` | Branch session leakage and ambiguous post-call state remain possible |
| C14 | PTI, LWI, Dual, and no-context MultiFlow retain smaller per-call workspace/current-value writes | Definition purity is incomplete outside BTA |

The old claim that `_last_aggregation_guidance` is never reset is not fully current: parsing resets it. The remaining defect is that resume can bypass parsing and a reused BTA still has an object-held value. It remains call-owned state and must move.

### 1.3 Confirmed test-tool defects

The current purity tool has three material limitations:

1. `diff_vars()` iterates only the after snapshot and therefore cannot report removed keys.
2. `assert_pure()` snapshots the same object twice without a call between them and is always vacuously green.
3. Best-effort deep copies can produce false changes for identity-based objects and do not give a stable recursive fingerprint for nested mutable fields.

No migration should claim definition purity until this tool is repaired and a measured baseline is recorded.

## 2. Scope, Compatibility, and Non-Goals

### 2.1 In scope

- Make host-context calls definition-stable across `InferencerBase`, templated and streaming bases, BTA, MultiFlow, Dual, LWI, PTI, and representative provider leaves.
- Add one generic invocation lifetime primitive used by sync, async, and streaming entrypoints.
- Keep serializable facts separate from live runtime objects.
- Make BTA sequential reuse correct and distinct-context/distinct-workspace concurrency correct.
- Make same-path or same-workspace concurrent execution fail before checkpoint or model I/O.
- Keep BTA stage slots as definitions and make factory ownership explicit.
- Compose a fresh WorkGraph per BTA attempt and later remove BTA's WorkGraph inheritance.
- Add strict resume identity and preserve valid checkpoint paths.
- Preserve explicit bare-call compatibility APIs while preventing them from influencing host-context calls.
- Add a purity ratchet, concurrency tests, checkpoint/event goldens, cleanup tests, and fresh-process resume tests.

### 2.2 Compatibility contract

Compatibility means observable behavior, not preserving accidental `__dict__` mutation:

- Public returned values, expected files, checkpoint relative paths, graph events, role selection, render count, fan-out mapping, and resume behavior remain stable unless a numbered defect is intentionally fixed.
- Bare calls retain documented post-call getters through an explicit compatibility sink.
- Host-context reads never fall back to a bare-call instance mirror.
- Constructor/YAML names remain stable through the composition migration.
- Existing workspaces resume only when identity can be proven. Silent unsafe legacy resume is not compatibility.

### 2.3 Non-goals

- No separate Template/Runner class hierarchy rewrite for every inferencer.
- No blanket clone-per-call rule for every configured child.
- No live graph, queue, client, lock, or closure serialization.
- No full prompt duplication into `RunStateStore` or the BTA manifest.
- No multi-writer checkpoint semantics for one workspace generation.
- No paid MetaMate G1-G6 execution without separate approval.
- No removal of all public `_last_*` getters in the first stack; ambiguous getters are deprecated after internal callers migrate.

## 3. Invariants

| ID | Invariant | Certification |
|---|---|---|
| I1 | A host-context call leaves the definition and every borrowed child definition unchanged except audited pure caches and handle-store internals | fixed purity snapshot plus per-class measured-debt ratchet |
| I2 | Invocation runtime is unique by call, not by context path | overlapping same-instance tests with distinct contexts; no live objects in `NodeRunState` JSON |
| I3 | Two simultaneous calls cannot mutate one `NodeRunState` path or one physical workspace | active path claim and advisory filesystem lease tests, including different owner instances |
| I4 | Parents read child outcomes from the exact child context, never from "last call on object" | concurrent role/task-contract tests |
| I5 | Definition slots are never overwritten by resolved runtime stages | before/after identity assertions for factory, recipe, and object slots |
| I6 | Every runtime-owned resource is cleaned exactly once on all exits | ownership-ledger tests for success, retry, failure, timeout, and cancellation |
| I7 | Resume reconstructs live objects from current input plus verified durable facts | fresh-process resume and mismatch tests |
| I8 | Sync, async, and streaming paths preserve the same context, role, output, and cleanup semantics | parity tests and explicit ContextVar propagation tests |
| I9 | Existing valid checkpoint paths and graph event ordering remain stable | checked-in workspace and event goldens |
| I10 | Known definition-mutation debt only shrinks | measured debt entries name their removal diff; a stale or new entry fails |

## 4. State Ownership Architecture

### 4.1 Definition state

The inferencer object may hold:

- attrs constructor fields and child definitions;
- `_init_recipe`, canonical identity, and factory metadata;
- template resources and immutable prompt assets;
- caches that are pure functions of definition inputs;
- explicit no-context reconfiguration such as a permanent role change;
- the `LiveHandleStore` container, whose branch values are not definition semantics;
- a bare-call compatibility sink used only by documented post-call getters.

A host-context call may not write:

- current input, rendered prompt, task contract, role history, or retry fingerprints;
- current graph, queues, workers, controllers, topology flags, or guidance;
- built stages into definition slots;
- current output/result metadata;
- runtime workspace/feed/modes/observer values into borrowed child definitions.

### 4.2 `InvocationFrame`: unique live call owner

Add a generic non-serialized primitive under the run-context package:

```python
@attrs.define(slots=True)
class InvocationFrame:
    call_id: str
    owner: "InferencerBase"
    context: RunContext
    parent: "InvocationFrame | None"
    mode: Literal["host", "legacy"]
    components: RuntimeComponentRegistry
    cleanup: RuntimeResourceRegistry

@attrs.define(slots=True)
class AttemptFrame:
    attempt_id: str
    number: int
    components: RuntimeComponentRegistry
    cleanup: RuntimeResourceRegistry
```

Required semantics:

1. One frame is minted at each public single-input sync/async call boundary.
2. Batch inputs each receive their own frame.
3. A direct public streaming call receives one frame spanning the generator lifetime.
4. Internal streaming consumption calls a private streaming pipeline and does not re-enter the public wrapper.
5. Nested child inference creates a child frame; a `ContextVar` stack preserves ancestry.
6. Private `super()._ainfer()` and internal hook calls stay in the current frame.
7. The frame spans preprocessing, render, retry, guardrail, provider/orchestrator work, finalization, state-graph update, response postprocessing, and cleanup.
8. Each retry gets an `AttemptFrame`. Failed-attempt resources are cleaned before the next attempt.
9. Runtime components use typed keys or explicit typed accessors; callers do not pass around an unstructured global `dict[str, Any]`.
10. Frames are destroyed in `finally` and are never encoded by `RunStateStore`.

Do not infer public re-entry from owner identity. Refactor the current streaming self-reentry to a private pipeline first. A second public call is a new invocation even if `ContextVar` ancestry was copied into a child task.

### 4.3 Active path claim

Add an in-process `ActiveInvocationRegistry` to shared runtime bindings. It claims the exact `(RunStateStore identity, context path)` for the lifetime of a host invocation.

- A second non-nested invocation at the same path fails with `ConcurrentInvocationError` before node or model I/O.
- The claim is path-based, not owner-based, so different instances of the same class cannot silently share one node.
- Sequential reuse of a path is allowed after release.
- Orchestrator rounds that intentionally reuse a path do so sequentially.
- A legitimate nested operation must use a child slot, not recursively open the same public entrypoint at the same path.
- Legacy-minted roots have independent stores, but physical workspace safety is still enforced separately.

### 4.4 Typed runtime component registry

Use a small typed registry rather than descriptors that silently redirect arbitrary attrs fields:

```python
T = TypeVar("T")

@attrs.define(frozen=True)
class RuntimeKey(Generic[T]):
    name: str
    factory: Callable[[], T]

class RuntimeComponentRegistry:
    def get_or_create(self, key: RuntimeKey[T]) -> T: ...
    def require(self, key: RuntimeKey[T]) -> T: ...
```

Reasons:

- Runtime ownership remains visible at each read/write site.
- attrs construction, deepcopy, pickling, and introspection are not changed by descriptor magic.
- A component has a declared type and lifecycle.
- Host calls never fall through to stale instance backing.
- Bare compatibility is handled at publication boundaries, not by making every attribute context-sensitive.

### 4.5 Durable node state

Add and use dedicated typed fields:

```python
@attrs.define(frozen=True)
class RenderedTaskContractState(InferencerStateBase):
    text: str
    sha256: str
    role: str | None
    source_path: str

@attrs.define
class NodeRunState:
    ...
    role_state: RoleState | None = None
    task_contract: RenderedTaskContractState | None = None
```

Rules:

- Keep orchestrator-specific semantic state in the existing typed `node.call` channel.
- Never store `RoleState` in `node.call`.
- Do not add a generic unbounded `outputs` dictionary when a typed field or the orchestrator's typed call state is sufficient.
- `node.provenance` contains bounded JSON-serializable audit events.
- `node.checkpoints` contains references/hashes, never live objects.
- Parent readers receive the exact child `RunContext` and use a non-claiming `peek`.

### 4.6 Bare-call compatibility

A legacy-minted call still gets an `InvocationFrame`, so its internal behavior is isolated. At successful publication boundaries only, documented compatibility getters may mirror a small result into a dedicated instance-backed compatibility sink.

- Host mode never reads or writes the sink.
- Failed calls do not publish a successful last result.
- Internal parent/orchestrator readers never use the sink.
- Every compatibility field has an owner, public getter, deprecation status, and removal test.
- Do not preserve byte-identical incidental `__dict__` mutation as a design goal.

### 4.7 Connection-scoped live handles

Sessions, clients, subprocesses, and conversations remain in `LiveHandleStore` when their meaning spans calls.

- Host-context lookup is exact-path and never falls back to another branch's backing value.
- Reset uses an explicit tombstone so `None` cannot accidentally mean "fall back to shared backing."
- A true no-context reset clears the backing and any branch slots that its getter would otherwise discover.
- Provider metrics and terminal response metadata belong to the invocation outcome, not the connection handle.
- `adisconnect()` closes connection-owned handles; invocation cleanup closes invocation-owned stages.

## 5. Generic Base and Templated Fixes

### 5.1 Repair purity tooling first

Replace the current snapshot comparison with a reliable test-only structural fingerprint:

- report added, changed, and removed keys;
- preserve original identity for uncopyable objects and compare identity before value comparison;
- recursively fingerprint primitive containers and attrs values with cycle detection;
- allow class-specific normalizers for loggers/locks/handles;
- snapshot the inferencer and each direct borrowed child;
- make `assert_pure()` accept a caller-provided before snapshot or remove it;
- import all runtime annotation symbols explicitly;
- add a negative test that mutates and restores neither an existing nested list nor a removed field;
- record measured `KNOWN_DEBT` by class and owning removal diff;
- fail when a debt entry is no longer observed, so the allowlist cannot rot.

Warm pure definition caches before the measured call when the cache is an allowed resident. The test must distinguish first-use definition cache creation from call-derived semantic state.

### 5.2 Fix context propagation before adding a new ContextVar

The `WorkGraph` sync-from-async thread hop must copy the current context before `InvocationFrame` lands:

```python
ctx = contextvars.copy_context()
pool.submit(ctx.run, asyncio.run, self._arun(...))
```

Audit every thread/process hop on inferencer call paths. A process boundary cannot inherit live context and must reconstruct an explicit `RunContext` from serialized inputs.

### 5.3 Fix role state and role audit

1. `_record_role_state()` writes `node.role_state`.
2. `_active_role_state()` reads `node.role_state`.
3. If legacy state has `RoleState` in `node.call` and `role_state` is empty, migrate it without touching a non-RoleState call value.
4. Replace `_pending_role_changes` with a local typed `RoleTransition` passed to the base implementation.
5. Under a host context, append the audit event to bounded `node.provenance`; do not write `_role_history` or `_applied_role`.
6. Outside a host context, preserve explicit permanent role reconfiguration.
7. Resolve the full effective role state, including key, root, version, variables, modes, feed changes, and other currently accepted role fields.

### 5.4 Publish task contracts at exact nodes

- A templated leaf publishes its rendered task contract at its own node.
- A BTA worker outcome carries the worker index and exact child context path.
- BTA chooses the lowest successful worker index, not first completion.
- BTA publishes the selected contract at the BTA node.
- LWI, MultiFlow, Dual, and MFDual publish their own representative contract according to their semantic winner/proposer rules.
- Parents read only from the exact child context they invoked.
- The bare compatibility getter mirrors only the final published contract.

### 5.5 Stop parent-to-child definition writes

Replace `_propagate_to_children()` mutation with context publication:

- publish effective feed, modes, observer, interaction, and workspace overrides into the intended child context before the child call;
- resolve precedence as explicit child-call override, nearest context publication, role state, then child definition;
- preserve the self-fan-out feed barrier so peer bundles do not leak back into parent rendering;
- never write a parent feed/mode into a borrowed child instance;
- keep no-context permanent configuration as an explicit setup API rather than an implicit call prologue side effect.

### 5.6 Move retry and guardrail state into the attempt

- Move the empty-output fingerprint window into the active attempt component.
- Use the already captured rendered input from the invocation/attempt; remove `_last_inference_input` from normal call flow.
- Call a guardrail judge with `prepared_input=True` rather than clearing its template manager.
- If a configured judge requires a preprocessor, perform the explicit agreed preprocessing step before the prepared call; first inventory configured judges.
- Ensure retry/fallback state resets at attempt and call boundaries as intended.
- Delete confirmed dead `_output_finalized`/`_complete_inference` surface only after a repository-wide reader check.

## 6. BTA Definition and Runtime Separation

### 6.1 Runtime objects

Use two private typed components:

```python
@attrs.define(slots=True)
class BtaCallRuntime:
    original_query: str
    request_sha256: str
    final_summary: BtaCallSummary | None = None

@attrs.define(slots=True)
class BtaAttemptRuntime:
    graph: WorkGraph
    breakdown: ResolvedStage | None
    aggregator: ResolvedStage | None
    workers: list[ResolvedStage]
    promoted_breakdown: PromotedBreakdown | None = None
    topology_emitted: bool = False
    pending_topology: Any = None
    aggregation_guidance: str | None = None
    worker_contracts: dict[int, RenderedTaskContractState] = attrs.field(factory=dict)
```

The call runtime spans retries and exposes only a serializable/minimal final summary to finalization. Each retry attempt receives a fresh graph, caches, and owned-stage ledger.

Move all current per-call BTA state:

| Current holder | Target |
|---|---|
| `start_nodes`, queues, graph registry, async mode, expansion limits | `BtaAttemptRuntime.graph` |
| `_cached_original_query` | `BtaCallRuntime.original_query`, captured by per-attempt closures |
| `_promoted_breakdown_cache` | `BtaAttemptRuntime.promoted_breakdown` |
| `_worker_instances` | `BtaAttemptRuntime.workers` and final summary |
| built aggregator/breakdown | `ResolvedStage` entries in the attempt |
| `_graph_topology_emitted`, `_pending_topology` | attempt fields |
| `_last_aggregation_guidance` | attempt field plus verified durable plan |
| `_worker_task_instructions` | indexed worker outcomes, then typed node contract |

Helpers receive the runtime explicitly. Avoid a broad ambient `current_bta_run` property; the invocation registry is only the carrier across framework seams such as finalization.

### 6.2 Definition slots and stage ownership

`breakdown_inferencer`, `worker_inferencers`, and `aggregator_inferencer` remain definitions. Do not apply a blanket clone rule. Use an explicit resolver:

```python
@attrs.define(frozen=True)
class ResolvedStage:
    inferencer: InferencerBase
    ownership: Literal["borrowed_definition", "invocation_owned"]
    definition_identity: str

class IndependentInferencerFactory(Protocol):
    definition_fingerprint: str
    def create(self) -> InferencerBase: ...
```

Resolution rules:

1. An `InferencerBase` slot is a borrowed reusable definition by default. It is invoked under an exact child context and must remain unchanged.
2. An `IndependentInferencerFactory` creates an invocation-owned stage. It is cleaned by the attempt/call that created it.
3. A canonical recipe can be wrapped as an independent factory.
4. Self-fan-out keeps its executed fresh-parent-worker semantics through the existing recipe/factory path.
5. An explicit fresh-prototype binding may request `fresh_instance()` when independent identity is semantically required.
6. Arbitrary inference callables are not guessed to be factories.
7. A factory returning the same live identity twice in one attempt is rejected.
8. Static worker definitions may be borrowed concurrently only after their class passes the host-purity and distinct-context isolation gates.
9. Until a class passes those gates, its binding must use an independent factory or be rejected for concurrent dispatch.

This preserves intentional connection continuity for borrowed definitions while providing strict independence where the API requires it. It avoids both unsafe sharing and unnecessary blanket cloning.

### 6.3 Remove aggregator write-back atomically

`build_aggregator()` becomes a pure resolver returning `ResolvedStage`; it never assigns `self.aggregator_inferencer`.

Redirect every reader in the same diff:

- aggregator validation and seeding;
- feed/context publication;
- output path and no-aggregator fallback;
- finalization and response promotion;
- workspace and reporting;
- disconnect/cleanup;
- self-fan-out setup.

The definition slot identity must be unchanged after success, retry, failure, and resume.

### 6.4 Fresh composed WorkGraph per attempt

A reusable BTA must no longer execute inherited graph state.

```text
BTA definition
  policies + stage definitions + graph configuration intent
            |
            v one attempt
BtaAttemptRuntime
  BtaExecutionGraph(WorkGraph)
    fresh breakdown node
    call-local expansion registry
    fresh node queues
    exact reporter/log/workspace bindings
```

Requirements:

- Build registry callbacks per attempt; closures capture the attempt and original query.
- Sync and async create the same graph specification; only the runner differs.
- Never flip `self.use_async`.
- Resume checks and topology operate on the attempt graph.
- Project every used WorkGraph configuration field explicitly.
- Preserve node names, child slot names, relative checkpoint paths, expansion records, event sequence, result-path callbacks, logging identity, retry behavior, and parentage order.
- Use a small `BtaExecutionGraph` adapter only for explicit log/result-path delegation; do not expose it after the call.

### 6.5 Inheritance migration

Use two proven steps:

1. Composition step: BTA temporarily remains a `WorkGraph` subclass for constructor/config compatibility, but calls neither mutate nor execute inherited graph state.
2. Hierarchy cleanup: after repository-wide type/config/introspection audit, remove `WorkGraph` from BTA and MultiFlow bases, redeclare only stable graph policy fields, and keep YAML field names stable.

The temporary inheritance state is not the final architecture. Do not leave permanent `start_nodes` compatibility properties that expose the last execution.

### 6.6 Cleanup ownership

Every `ResolvedStage` has explicit ownership.

- Borrowed definitions are not disconnected by an invocation.
- Invocation-owned stages are registered immediately after creation.
- Failed attempts cancel tasks, await cancellation, and close attempt-owned stages before retry.
- On terminal success/failure, close remaining invocation-owned resources exactly once.
- Cleanup errors do not replace an active inference exception; if inference succeeded, cleanup failure is surfaced.
- Sync cleanup uses supported sync hooks and does not create an untracked event loop.
- `BTA.adisconnect()` closes only definition-owned long-lived handles and is idempotent.

## 7. Resume Identity and Workspace Concurrency

### 7.1 Workspace lease

Use an advisory filesystem lock held by an open file descriptor at a fixed checkpoint path, for example `checkpoints/.bta_execution.lock`.

- Acquire before reading or writing manifest/checkpoints or calling a model.
- Use kernel-released advisory locking so a crashed process does not leave a stale semantic lock.
- The lock covers the resolved physical workspace generation, not inferencer identity.
- Two processes or threads targeting the same workspace fail immediately with `BtaWorkspaceBusyError`.
- Distinct workspaces may run concurrently on the same BTA definition.
- Validate the resolved path and never lock a broad directory such as repository root or home.

The in-process path claim protects `NodeRunState`; the filesystem lease protects external artifacts. Neither substitutes for the other.

### 7.2 Versioned BTA execution manifest

Write an atomic, secret-free manifest before consuming resume artifacts:

```json
{
  "schema_version": 1,
  "request_sha256": "...",
  "request_type": "str",
  "request_bytes": 12780,
  "bta_definition_sha256": "...",
  "relevant_arguments_sha256": "...",
  "checkpoint_mode": "jsonfy",
  "decomposition_schema": 1,
  "effective_worker_plan": [
    {"index": 0, "query_sha256": "...", "stage_definition_sha256": "..."}
  ],
  "aggregation_guidance_sha256": "...",
  "topology_sha256": "...",
  "worker_outcome_schema": 1
}
```

Identity rules:

- The caller supplies the full original input again; persist only its type, byte length, and digest.
- Definition identity comes from canonical serialized constructor intent/factory recipes, excluding secrets and live handles.
- Opaque factories must expose a stable definition fingerprint when resume is enabled.
- Include fields that change topology or output semantics; exclude scheduling-only concurrency settings.
- Persist the effective worker plan after parsing, truncation, selection, todo expansion, and heterogeneous dispatch resolution.
- Write with temp file, fsync as required by the existing checkpoint convention, and atomic rename.
- Validate header identity before loading breakdown/expansion/worker/aggregator artifacts.
- Validate finalized effective-plan identity before reusing worker results.

### 7.3 Legacy resume policy

Fail closed by default:

- No artifacts and no manifest: fresh execution.
- Matching manifest: reconstruct fresh live runtime and resume.
- Input/definition/argument mismatch: `BtaResumeIdentityMismatch` before stage creation.
- Corrupt or internally inconsistent manifest/checkpoints: `BtaResumeCorruptionError`.
- Legacy artifacts without a manifest: `UnverifiedLegacyBtaResumeError` by default.
- A separate migration command/helper may create a manifest only if exact request, definition, and effective worker mapping can be proven.
- An explicit unsafe override may exist for emergency recovery, but it is never the default, is prominently logged, and writes no "verified" manifest.

Self-fan-out and MetaMate use strict mode from the first new manifest. Generic BTA also defaults to strict for new workspaces.

## 8. Remaining Orchestrator and Provider Migration

### 8.1 Dual and MultiFlow

- Move in-call `_last_iteration_record` and equivalent live data into typed invocation components.
- Keep durable consensus/dispatch facts in existing typed call state.
- Publish representative task contracts at the orchestrator's own node.
- Eliminate no-context definition drift from runtime input propagation; use the invocation component.
- Keep public winner/ranking getters through explicit compatibility publication until callers migrate.

### 8.2 LWI and PTI

- Move `_current_*` and previous-attempt live fields into typed invocation components.
- Capture the owner node before entering child contexts when a child closure must publish back to the owner.
- Publish runtime workspaces through exact child contexts under host mode.
- Preserve legacy setter behavior only for true no-context setup paths.
- Treat per-run child save/resume flags and result-root overrides as a separate atomic subphase with location goldens; do not partially virtualize them.

### 8.3 Streaming and provider leaves

- Split public streaming wrappers from private streaming pipelines so one logical call has one frame.
- Move usage/token/tool counters and terminal stream metadata to invocation outcomes.
- Preserve direct public streaming result getters through the bare compatibility sink.
- Route session/conversation writes through exact-path live-handle fields with reset tombstones.
- Replace process-wide or instance-wide "session initialized" flags with handle values keyed to the actual session.
- Convert one provider family per commit and run its integration mocks before proceeding.

## 9. Executable Landing Stack

### Diff 0: Certification baseline

Deliver:

- repaired purity tool;
- measured per-class mutation debt;
- direct-child mutation snapshots;
- checked-in BTA workspace/event/location goldens for fresh sync, fresh async, partial resume, no aggregator, and MultiFlow;
- regression tests for C1-C14, marked strict expected-failure only where needed;
- inventory of all context-losing thread/process hops;
- inventory of all post-call instance readers and WorkGraph/BTA type/introspection users.

Stop gate: do not begin runtime migration until the baseline is deterministic and catches nested mutation plus removed fields.

### Diff 1: Foundational correctness

Deliver:

- WorkGraph ContextVar propagation fix;
- `RoleState` dedicated-field migration and coexistence tests;
- deterministic indexed BTA contract selection in both sync and async paths;
- guardrail fingerprint isolation using existing attempt state as an interim correction;
- session reset through context-aware properties where currently bypassed;
- delete only verified dead fields/helpers.

This diff fixes known defects without introducing the new runtime framework.

### Diff 2: Generic invocation lifetime

Deliver:

- `InvocationFrame`, `AttemptFrame`, typed runtime registry, and resource registry;
- frame integration at sync/async single-input boundaries;
- public/private streaming split and direct streaming frame;
- active `(store, path)` invocation claim;
- lifecycle tests covering render through finalization and cleanup;
- no migrated fields yet except minimal proof components.

Stop gate: nested child calls, batch calls, direct streaming, cancellation, sync-from-async, and copied-context sibling tasks must all demonstrate unique frames.

### Diff 3: Typed outcomes, roles, and propagation

Deliver:

- `RenderedTaskContractState` and exact-context readers;
- leaf/BTA/LWI/MultiFlow/Dual task-contract publication;
- local `RoleTransition` and provenance audit;
- full effective-role resolution;
- context-published feed/modes/observer/interactive overrides;
- removal of host-context `_last_rendered_task_instructions`, `_pending_role_changes`, `_role_history`, and child feed/mode writes;
- guardrail prepared-input call without judge mutation.

Stop gate: no internal parent reader may use a child instance's last-call getter.

### Diff 4: BTA call/attempt runtime and stage ownership

Deliver:

- `BtaCallRuntime`, `BtaAttemptRuntime`, `BtaCallSummary`, and `ResolvedStage`;
- move workers, guidance, promoted-breakdown memo, query mailbox, topology, and task contracts off `self`;
- pure aggregator resolution with atomic reader migration;
- explicit borrowed versus invocation-owned cleanup ledger;
- definition-slot identity tests;
- sequential reuse tests green.

At this point inherited WorkGraph execution remains, so distinct-workspace concurrent reuse is not yet enabled.

### Diff 5: Per-attempt WorkGraph composition

Deliver:

- fresh `BtaExecutionGraph` per attempt;
- call-local registry closures;
- identical sync/async graph specification;
- no inherited graph mutation or execution;
- exact checkpoint/event/location goldens;
- distinct-context/distinct-workspace concurrency tests green;
- old silent wrong-output reproducer green.

Stop gate: if any checkpoint relative path, expansion replay, topology ordering, logger parentage, or MultiFlow behavior changes without an explicit compatibility decision, stop and repair before continuing.

### Diff 6: Workspace lease and strict resume identity

Deliver:

- advisory filesystem lease;
- atomic versioned manifest;
- canonical definition identity;
- effective worker-plan identity;
- strict mismatch/corruption/legacy errors;
- provable legacy migration helper;
- fresh-process resume and same-workspace cross-process tests.

### Diff 7: Remaining orchestrators

Deliver:

- Dual, MultiFlow, LWI, and PTI invocation components;
- host-context workspace publication for remaining safe paths;
- atomic handling of resume-sensitive child flags/root overrides;
- removal of confirmed dead runtime fields;
- location and resume goldens unchanged.

### Diff 8: Streaming/provider handles and outcomes

Deliver one provider family per commit:

- streaming result/usage outcome migration;
- exact-path session/conversation handles;
- reset tombstones;
- no raw backing writes under host context;
- direct-stream getter compatibility;
- cleanup and branch-isolation tests.

### Diff 9: Hierarchy cleanup and enforcement

Deliver:

- remove BTA/MultiFlow WorkGraph inheritance after repository-wide audit;
- explicitly declare stable graph-policy fields;
- remove temporary compatibility shims and migrated instance fields;
- shrink measured debt to documented legitimate residents;
- update run-context and inferencer authoring documentation;
- add static/source checks for prohibited host-call writes and runtime objects in serialized state.

## 10. Test Matrix

### 10.1 Purity and fixed point

- Snapshot host, every borrowed direct child, and relevant factories before/after calls.
- Detect added, changed, removed, and nested-mutated state.
- Run a second call with a fresh context; compare output and normalized node JSON.
- Assert definitions, definition slots, and canonical identity are unchanged after success, retry, failure, timeout, cancellation, and resume.
- Assert only documented pure caches/handle containers change after warm-up.

### 10.2 Invocation lifetime

- Sync, async, batch, direct async streaming, sync streaming, and sync-from-running-event-loop each get the correct frame.
- Private super/internal calls remain in one frame.
- Nested child inference gets a child frame.
- Copied-context sibling tasks get distinct frames.
- Same store/path overlap fails even for different instances of the same class.
- Sequential same-path rounds work after release.

### 10.3 Role and task contract

- `RoleState` coexists with `BTAState`, `MultiFlowState`, `MFDualState`, and `LinearWorkflowState`.
- Legacy `RoleState` in `call` migrates without discarding other call state.
- Host role changes do not mutate definition fields/history.
- Full variables/modes/feed/version role changes affect rendering.
- Worker completion order `[2, 0, 1]` still selects successful index `0`.
- Failed lowest worker selects the next successful index.
- Concurrent roles on one leaf publish isolated contracts.
- Dual/MultiFlow/MFDual read the exact intended child contract.

### 10.4 Propagation and guardrail

- Two parents sharing one child do not alter the child definition or cross-contaminate feed/modes.
- Parent, role, child, and explicit-call precedence matches the declared rule.
- The fan-out barrier preserves render-once semantics.
- Guardrail judge definition remains unchanged and receives the exact prepared prompt.
- A failed call's fingerprint history cannot affect the next call.

### 10.5 BTA reuse and concurrency

- Sequential calls do not reuse graph, query, cache, topology, guidance, outcomes, or invocation-owned stages.
- Definition stage slots remain byte-for-byte/identity unchanged.
- Borrowed pure stages work concurrently under distinct child contexts.
- Factory stages are distinct, invocation-owned, and cleaned exactly once.
- A reused factory identity is rejected.
- Distinct-context/distinct-workspace calls produce independent graphs and results.
- Slow A breakdown plus fast B breakdown reproducer returns both correct aggregates.
- Same path fails before node/model I/O.
- Same workspace fails before checkpoint/model I/O in one or two processes.

### 10.6 WorkGraph and resume

- Fresh sync/async graph specs are equivalent.
- Old/new checkpoint relative paths and formats match.
- Partial crash resumes only incomplete workers and aggregator.
- Fresh process reconstructs from current input plus manifest/checkpoints.
- Changed request, stage recipe, semantic BTA setting, effective ordering, or topology fails before reuse.
- Scheduling-only concurrency change does not invalidate identity.
- Missing/corrupt manifest follows strict policy.
- Breakdown-only, no aggregator, predefined queries, custom parsing, interactive selection, todo expansion, nested BTA, and MultiFlow all resume the exact effective plan.

### 10.7 Cleanup and handles

- Success, validation failure, retry, timeout, cancellation, partial worker failure, finalizer failure, and cleanup failure exercise exact ownership.
- One call cannot cancel or disconnect another call's stages.
- Borrowed stages are never invocation-disconnected.
- Invocation-owned stages are disconnected once in reverse creation order where required.
- No pending asyncio tasks or held file leases remain.
- Host session reset cannot fall back to a sibling/backing session.
- Intentional same-branch conversation continuity remains.

## 11. Validation Gates

Every diff must pass:

1. Focused red/green regression tests for the exact defect.
2. AgentFoundation owning BUCK targets for edited files.
3. BTA, BTA resume, outcome pairing, self-fan-out, fresh-instance, fan-out primitive, task-contract, unified-finalization, LWI, PTI, Dual, MultiFlow, MFDual, streaming, and run-context suites as applicable.
4. RichPythonUtils WorkGraph tests for graph/context changes.
5. `arc f`, `arc lint -a`, and `arc pyre check-owning-targets`.
6. Purity debt does not grow; entries assigned to the diff disappear.
7. Checked-in checkpoint/event/location goldens remain stable.
8. Failure IDs are compared with the recorded baseline, not only aggregate counts.
9. No code edits occur while long-running Buck tests are reading the tree.

Before hierarchy removal, run a repository-wide audit for BTA `isinstance`/`issubclass`, inherited WorkGraph methods, constructor fields, `start_nodes`, `subgraph_registry`, direct graph execution, config aliases, and tests inspecting graph internals.

## 12. Risk Table

| Area | Change risk | Main failure mode | Required mitigation |
|---|---:|---|---|
| Purity tool | Medium | false positives or missed mutation | negative nested/removal tests; class normalizers; measured baseline |
| Invocation frame | High | wrong lifetime, copied-context aliasing, finalize outside frame | private streaming pipeline; exact boundary tests; unique call IDs |
| Active path claim | Medium | legitimate re-entry rejected | remove public self-reentry; require explicit child slot; sequential-round tests |
| Role state | Medium-high | rendering or typed call state regression | legacy migration and coexistence matrix |
| Task contract | Medium-high | reviewer/fixer receives wrong contract | exact-child context, deterministic indexed selection, full chain tests |
| Child propagation | High | precedence or render-once changes | explicit precedence tests and fan-out barrier tests |
| Guardrail | Medium | skipped preprocessing or retry change | judge inventory, prepared-input exact-prompt tests, retry goldens |
| BTA runtime | High | stale finalizer reads or cleanup leaks | typed final summary, explicit helper parameters, ownership ledger |
| WorkGraph composition | Very high | checkpoint/event/resume drift | staged composition, checked-in goldens, old-workspace resume fixture |
| Stage resolution | High | lost session continuity or unsafe shared object | explicit borrowed/owned policy and per-class purity gate |
| Resume manifest | Medium-high | valid legacy workspace rejection or false reuse | fail-closed errors, provable migration helper, canonical identity tests |
| Filesystem lease | Medium | stale/broad lock or unsupported filesystem behavior | advisory FD lock, explicit resolved file, process tests, clear error |
| PTI/LWI workspace | Very high | checkpoint path drift | separate atomic subphase and location goldens |
| Provider handles | High | lost conversation continuity | one provider per commit, exact-path and bare continuity tests |

## 13. Comparison of the Five Current Plans

| Dimension | `/home/zgchen/.llms/plans/idempotent-enchanting-lake.md` | `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/inferencer_template_runtime_separation.plan.md` before this integration | `/home/zgchen/.llms/plans/inferencer_template_runtime_decoupling.plan.md` | `/home/zgchen/.claude/plans/snappy-doodling-star.md` | `/home/zgchen/.claude/plans/here-is-some-context-harmonic-liskov.md` |
|---|---|---|---|---|---|
| Core state model | Clear five-owner summary | Strong four-home model | Strong broad six-home inventory | Strong four-home model with `CallFrame` | Clearest narrow placement rule and BTA root cause |
| Invocation identity | Stack-owned runtime, lightly specified | Unique `InferenceCallRuntime`, but generic `live` dict underspecified | Uses call run objects but also proposes path scratch for generic access | Most detailed frame ancestry/reentry design | Prefers locals; uses node scratch only across seams |
| Main weakness in call primitive | Lacks entry/reentry/streaming detail | No explicit copied-context/reentry contract | `_call_scratch()` is path-keyed and not safe for simultaneous same-path calls | Owner-based guard misses different instances; `RunScoped` descriptors add hidden semantics | Scratch publication can collide; no full generic lifecycle |
| Purity certification | Good acceptance list, no deep tool repair | Basic snapshots, misses current tool defects | Excellent certification-first matrix and goldens | Strongest measured debt ratchet and purity-tool diagnosis | Identifies broken helper but narrower certification |
| Breadth beyond BTA | Moderate | Moderate | Broadest inventory across base, orchestrators, providers | Broad and highly detailed | Intentionally BTA-focused |
| Role-state fix | Correct | Correct | Incomplete/ambiguous about dedicated field in some passages | Correct with migration | Focuses mostly on role audit, not full collision |
| Task-contract design | Typed outcome concept | Dedicated typed field | Generic `outputs` dictionary | Strong typed contract and exact-child read | Treats result/API cleanup as partly optional |
| Child definition mutation | Not deeply covered | Covered only generally | Strong feed/modes/observer analysis | Identifies it but phases it late | Mostly omitted |
| Guardrail/base defects | Includes fingerprint migration | Mostly omitted | Strong judge/fingerprint analysis | Strong bug inventory | Mostly omitted |
| BTA graph composition | Correct target | Detailed staged composition and hierarchy removal | Detailed `_BtaRun`/`_BtaGraph` mechanics | Defers inheritance removal | Strong root-cause explanation and direct composition |
| Stage materialization | Blanket fresh stages | Blanket fresh stages | Reuses static object stages; fresh factory stages | Defers blanket materialization | Reuses existing instances in common cases |
| Stage policy weakness | Can unnecessarily break connection continuity | Same blanket-clone risk | Does not explicitly purity-gate borrowed concurrency | Leaves ambiguous unsafe definitions during migration | Ownership/cleanup policy is incomplete |
| Cleanup | Strong per-attempt ledger | Strong per-execution cleanup | Mixed: some lifecycle remains until `adisconnect` | Defers call-end cleanup | Call cleanup present but less ownership detail |
| Same-path/workspace concurrency | Filesystem lease, clear | In-memory coordinator only; cross-process unsupported | Documents same-path limitation and defers generic guard | Owner-keyed host guard; defers workspace lease | Temporary object marker and later workspace guard |
| Resume identity | Strongest fail-closed manifest | Detailed manifest but allowed unsafe `legacy_warn` default | Preserves old resume but lacks strict identity | Explicitly defers manifest | Adds only a request contract hash |
| Logic-preservation mechanics | Good but concise | Strong golden matrix | Excellent line-level mechanics and phase runbook | Excellent frame/entry analysis | Excellent BTA checkpoint/path explanation |
| Sequencing | Clean eight-step order | Six diffs, but misses certification-first | Excellent Phase 0 then phased migration | Most comprehensive P0-P10 staging | Simple S0-S4, but S0 contains temporary object guard |
| Elegant end state | Good BTA-centric target | Strong, but incomplete framework breadth | Broad, but path scratch and no strict resume remain | Powerful, but descriptors and deferred correctness add complexity | Elegant BTA core, incomplete framework target |
| Best contribution adopted here | strict resume, cleanup, concise acceptance criteria | four-home ownership, typed contract, staged graph composition | broad inventory, Phase 0 goldens, propagation/provider findings | entrypoint analysis, measured debt ratchet, bug inventory | root-cause narrative, local closure capture, checkpoint preservation |
| Main proposal rejected here | blanket fresh materialization | generic live dict and legacy-warn default | canonical live state in node scratch | `RunScoped` descriptor, owner-only guard, deferred lease/manifest/cleanup | object in-flight bit, optional typed result cleanup, partial fingerprint |

## 14. Single-Plan Choice

If only one of the five current files could be selected **before integration**, choose:

`/home/zgchen/.llms/plans/inferencer_template_runtime_decoupling.plan.md`

Reason: it has the broadest verified inventory, the strongest certification-first execution order, detailed compatibility mechanics, and the best coverage of non-BTA state. It is the most useful implementation guide for the whole class family.

It should not be executed unchanged. Its path-keyed generic scratch cannot be the canonical owner of simultaneous call runtime; it lacks strict resume identity and a cross-process workspace lease; it does not define a sufficiently explicit borrowed-versus-owned stage contract; and its generic `outputs` dictionary is weaker than typed outcome fields.

After this integration, the recommended plan is this file:

`/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/inferencer_template_runtime_separation.plan.md`

It combines the decoupling plan's breadth and certification, the snappy plan's entrypoint and bug analysis, the harmonic plan's BTA root-cause clarity, the idempotent plan's strict resume/cleanup discipline, and the prior canonical plan's concurrency, WorkGraph, and typed-state architecture while removing their conflicting or unsafe mechanisms.

## 15. Rejected Approaches

- Do not store live call state canonically in `NodeRunState.scratch`; the node is path-keyed, not invocation-keyed.
- Do not use a `RunScoped` descriptor to make arbitrary object fields silently change backing by context; explicit runtime components are easier to audit and type-check.
- Do not use an owner-only concurrency guard; two different instances can still share one path/state.
- Do not put an in-flight bit on the inferencer; synchronization belongs to runtime coordination, not definition state.
- Do not preserve accidental bare-call `__dict__` writes for byte identity; preserve documented API behavior through explicit publication.
- Do not blanket-clone every stage; use explicit borrowed-definition versus invocation-owned resolution.
- Do not share a borrowed stage until its class passes purity/isolation gates.
- Do not remove aggregator write-back without atomically migrating all readers.
- Do not serialize the full original prompt or live graph for resume.
- Do not accept unverified legacy checkpoints by warning and continuing.
- Do not remove WorkGraph inheritance before composition is proven.
- Do not expose a "last graph" compatibility property on the reusable definition.
- Do not combine provider/session migration with the core WorkGraph composition diff.

## 16. Rollback

- Diffs 0-3 are independently reversible while typed-state readers retain legacy decoding.
- Diff 4 is reversible before new manifest use, provided the aggregator reader migration reverts atomically.
- Diff 5 must retain readers for existing expansion/checkpoint records; revert composition as one unit if goldens fail.
- Diff 6 writes a versioned manifest. Rollback must tolerate the new file and preserve the advisory-lock path; never delete user checkpoints automatically.
- Diffs 7-8 revert provider/orchestrator families independently because each keeps an explicit compatibility boundary.
- Diff 9 is hierarchy cleanup only after behavior is proven; reverting it may restore inheritance without changing checkpoint data.
- Never roll back indexed worker outcomes or other already-executed fan-out correctness fixes.

## 17. Final Acceptance Criteria

The refactor is complete only when all statements are true:

1. A host-context call produces no unapproved instance mutation on the inferencer or borrowed children after definition-cache warm-up.
2. The purity gate reliably catches added, changed, removed, and nested-mutated fields.
3. Every public single-input path has exactly one unique invocation frame and every retry has a separate attempt frame.
4. No live runtime object is serialized into `NodeRunState`.
5. `RoleState` and typed orchestrator call state coexist.
6. Parent task-contract reads are exact-context and deterministic.
7. Parent feed/modes/workspace/observer propagation does not mutate child definitions.
8. Guardrail and retry state cannot leak across calls and the judge remains unchanged.
9. BTA definition slots remain unchanged after all outcomes.
10. BTA creates and executes a fresh WorkGraph per attempt.
11. Distinct-context/distinct-workspace concurrent BTA calls are correct.
12. Same-path and same-workspace overlap fail before semantic or external I/O.
13. Every invocation-owned child is cleaned exactly once; borrowed definitions are not invocation-disconnected.
14. Resume succeeds in a fresh process only when manifest identity matches.
15. Invalid, corrupt, or unverified legacy resume fails loudly by default.
16. Existing valid checkpoint paths, event ordering, role behavior, render-once behavior, and fan-out outcomes match goldens.
17. Bare compatibility getters still work without becoming a host-context source of truth.
18. BTA no longer inherits WorkGraph in the final hierarchy.
19. The measured mutation-debt table is empty except documented definition caches, handle containers, and explicit no-context compatibility residents.
20. All owning tests, lint, format, and type-check gates pass, with no unexplained change from the recorded failure baseline.
