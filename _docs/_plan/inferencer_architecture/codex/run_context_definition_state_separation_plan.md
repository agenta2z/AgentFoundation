# Explicit RunContext, Definition/State Separation, and Multi-Reviewer Dual Plan

Status: integrated Codex v2, 2026-06-19

This plan integrates the strongest parts of:

- `/Users/tchen7/.claude/plans/swift-launching-backus.md` (Swift v11)
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plan/inferencer_architecture/revodev/runcontext_explicit_state_separation_plan.md` (Revodev v5)
- this Codex plan's prior v1

It is intended to be the self-contained implementation plan. The key design decision is explicit: inferencer YAML and object instances are reusable definitions; all mutable per-run execution state belongs to an application-owned `RunContext` tree passed through inference calls.

## 1. Honest Comparison and Choice

### Best ideas from Swift v11

Swift has the strongest architectural model:

- App-minted `RunContext`, not inferencer-owned hidden state.
- A three-tier separation:
  - serializable run state,
  - concurrency-safe runtime bindings,
  - worker-local live handles.
- No `_active_ctx` as a target design.
- Factory remains until role/workspace/live-resource mutation is virtualized and proven safe.
- Full AgentFoundation + OpenStartup testing inventory.
- Corrected details: BTA worker isolation currently warns rather than raises; `_last_winner_idx` and `_cached_sub_queries` are attrs fields; OpenStartup `_runtime` task reads are ignored artifacts.

Remaining gap in Swift as a standalone plan: it still reads like a design ledger and says to execute from Revodev plus corrections. It also underspecifies `state_factory`, OpenStartup as a non-optional integration target, and multi-reviewer milestones.

### Best ideas from Revodev v5

Revodev has the most concrete migration/test spine:

- Explicit milestone sequencing.
- Stronger test inventory discipline.
- Good attention to task tool and OpenStartup host responsibilities.
- Useful compatibility gates for legacy callers.

Remaining gaps in Revodev as-is:

- It still contains stale claims: OpenStartup test count, BTA "raise" behavior, and some `_active_ctx` language.
- Its `RunContext(...)` construction examples are inconsistent with a `RuntimeBindings(...)` container.
- It overstates "shared instance safe" after workspace changes while role/session/live-handle issues remain.
- Its Tier-3 live-resource story is less clean than Swift's worker-local, connection-scoped model.

### Best ideas from Codex v1

Codex v1 had the right host boundary:

- Task tool and OpenStartup conversation service pair root inferencers with root run state.
- `InferencerBase` should provide generic child-context routing.
- State should be path-keyed, not instance-keyed.
- Legacy behavior must remain available without forcing every caller to migrate at once.

Remaining gap in Codex v1: it did not yet absorb the three-tier resource model, typed state factories, latest test inventory, and multi-reviewer migration details.

### If We Pick Only One Existing Plan

If forced to pick one existing file as-is, pick:

`/Users/tchen7/.claude/plans/swift-launching-backus.md`

Reason: it has the most correct architecture and the fewest dangerous design assumptions. However, it should not be executed literally until its "execute from Revodev" ambiguity is removed. This integrated Codex plan is the cleaner execution artifact because it folds Swift's architecture, Revodev's migration/testing discipline, and Codex's host-boundary model into one place.

## 2. Target Mental Model

Today, inferencers mix three things:

1. Definition: topology, child inferencers, template config, role config, static parameters.
2. Per-run state: current attempt, winner index, cached subqueries, conversation pause/resume data, aggregation artifacts.
3. Live resources: SDK sessions, subprocess handles, streaming files, connection-scoped interactive transport, graph reporters.

The target model separates them:

- Inferencer instances are reusable definitions.
- Applications create a root `RunContext` per invocation/session/turn.
- Composite inferencers create child contexts by stable child paths.
- Per-run state is stored in `RunStateStore`, keyed by child path.
- Safe shared runtime bindings are explicit in `RunContext.runtime`.
- Unsafe live resources are created and owned by worker-local `LiveHandles`, not stored as reusable definition state.

The application layer is responsible for pairing a root inferencer with a root `RunContext`:

- Task tool: one root context per task execution or resumed task run.
- OpenStartup conversation service: one root context per conversation turn, or one resumable root context per session if/when session-level resumability is required.
- SOP/CLI/test hosts: one root context per command/run.

`InferencerBase` is responsible for generic child routing:

- It receives `run_context`.
- It derives child contexts with stable paths.
- It passes the right child context to the right child inferencer.
- It never requires the caller to manually build a state dict for every child.

## 3. Core API

### Inference Signature

Add a keyword-only `run_context` argument to public inference entrypoints:

```python
async def ainfer(
    self,
    input_data: Any,
    *args: Any,
    run_context: RunContext | None = None,
    **kwargs: Any,
) -> Any:
    ...
```

Synchronous `infer(...)` mirrors the same keyword-only argument.

Rules:

- `run_context` is keyword-only.
- Existing callers keep working while compatibility is enabled.
- Missing `run_context` creates a local ephemeral context only in legacy-compatible mode.
- New internal calls must pass `run_context` explicitly.
- Tests must fail if `run_context` leaks into prompt template variables or backend model kwargs.

### RunContext

```python
@attrs.define
class RunContext:
    run_id: str
    path: tuple[str, ...]
    state_store: RunStateStore
    runtime: RuntimeBindings = attrs.Factory(RuntimeBindings)
    live: LiveHandles | None = None
    metadata: dict[str, Any] = attrs.Factory(dict)

    @classmethod
    def root(
        cls,
        *,
        run_id: str | None = None,
        state_store: RunStateStore | None = None,
        runtime: RuntimeBindings | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RunContext": ...

    def child(self, segment: str | int, *, role: str | None = None) -> "RunContext": ...

    @property
    def node_state(self) -> NodeRunState: ...
```

Path identity is logical, not object identity. Examples:

- `("root", "planner")`
- `("root", "implementation")`
- `("root", "bta", "worker", "query_003")`
- `("root", "mfd", "flow", 1, "reviewer", 0)`

Stable paths are the backbone of traceability, resumability, and shared-instance safety.

### RunStateStore

```python
@attrs.define
class RunStateStore:
    nodes: dict[tuple[str, ...], NodeRunState] = attrs.Factory(dict)
    schema_version: int = 1

    def get_or_create(self, path: tuple[str, ...]) -> NodeRunState: ...
    def snapshot(self) -> dict[str, Any]: ...
    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "RunStateStore": ...
```

Requirements:

- Serializable to JSON-compatible data.
- Versioned.
- No live SDK handles, subprocesses, file objects, or interactive objects.
- Safe to persist as task/session checkpoint payload.

### NodeRunState

```python
@attrs.define
class NodeRunState:
    typed: dict[str, Any] = attrs.Factory(dict)
    call: dict[str, Any] = attrs.Factory(dict)
    attempt: dict[str, Any] = attrs.Factory(dict)
    conversation: dict[str, Any] = attrs.Factory(dict)
    provenance: dict[str, Any] = attrs.Factory(dict)
    checkpoints: dict[str, Any] = attrs.Factory(dict)
```

Use `typed` for state produced by an inferencer's `state_factory`.

Use `conversation` for existing conversational pause/resume state first. Do not prematurely force conversational state into a typed class until the resume semantics are proven.

Use `provenance` for trace metadata: source subquery, flow id, reviewer id, role, prompt version, and parent path.

### RuntimeBindings

```python
@attrs.define
class RuntimeBindings:
    graph_reporter: Any | None = None
    interactive: Any | None = None
    cancellation_token: Any | None = None
    checkpoint_store: Any | None = None
```

Only concurrency-safe shared sinks belong here. `checkpoint_store` is provisional until verified as safe for concurrent child writes. If it is not safe, wrap it in an async lock or move it into per-run state persistence instead.

Interactive transport is allowed here only when it is explicitly connection/session scoped and safe for multiple child messages. Per-task receive queues still belong to the transport layer, not inferencer definitions.

### LiveHandles

```python
@attrs.define
class LiveHandles:
    sdk_client: Any | None = None
    subprocess: Any | None = None
    http_session: Any | None = None
    stream_writer: Any | None = None
    logger: Any | None = None
    cache: dict[str, Any] = attrs.Factory(dict)
```

Tier-3 live handles are worker-local and connection-scoped:

- Reuse them across repeated calls in the same branch when beneficial.
- Do not share them across concurrent branches unless the handle is explicitly concurrency-safe.
- Tear them down with `adisconnect()`, `__aexit__`, or equivalent worker cleanup.
- Do not serialize them.
- Do not store them in reusable inferencer definitions.

This avoids both bad extremes: one shared live session for all flows, and reconnecting on every single call.

## 4. State Factory Design

Each inferencer may declare a `state_factory`.

```python
@attrs.define
class InferencerStateBase:
    schema_version: int = 1

    def to_jsonable(self) -> dict[str, Any]: ...
    @classmethod
    def from_jsonable(cls, payload: Mapping[str, Any]) -> "InferencerStateBase": ...
```

Expected factory signature:

```python
def state_factory(*, context: RunContext, inferencer: InferencerBase) -> InferencerStateBase:
    ...
```

YAML shape:

```yaml
_target_: SomeInferencer
state_factory:
  _target_: SomeInferencerState
  schema_version: 1
  other_default: value
```

Rules:

- If no `state_factory` is configured, use generic `NodeRunState` dict fields.
- If configured, create or restore the typed state under `context.node_state.typed[inferencer_state_key]`.
- Do not store the typed state on the inferencer instance.
- Multiple typed states may coexist under distinct keys when a composite owns several state domains.
- State classes must be serializable and versioned.

This gives YAML a definition role without turning YAML into a mutable runtime object.

## 5. Compatibility Policy

The refactor is large, so the compatibility policy must be deliberate.

Legacy-compatible mode:

- Public `infer`/`ainfer` may create an ephemeral root context if no `run_context` is provided.
- Existing tests and examples should keep passing while migration proceeds.
- Legacy properties may read/write the active node state for a transition period, but only when a context is available.

Strict mode:

- Internal composite calls must pass `run_context`.
- New examples should pass `run_context`.
- New tests should assert no hidden instance mutation.
- After migration, strict mode can become the default for framework internals.

Avoid `_active_ctx` as the target architecture. If a temporary context variable is unavoidable for a compatibility property, it must be:

- local to a call scope,
- never used for shared-instance safety claims,
- covered by concurrency tests,
- removed before factory retirement for that inferencer.

## 6. Migration Plan

### M0: Baseline and Golden Traces

Before edits:

- Run and record core AgentFoundation test suites.
- Run targeted inferencer tests.
- Run task tool preflight/CLI tests.
- Run OpenStartup server tests.
- Capture golden traces for representative topologies:
  - leaf inferencer,
  - `DualInferencer`,
  - `BreakdownThenAggregate`,
  - `MultiFlowDual`,
  - `PlanThenImplement`,
  - `ConversationalInferencer`,
  - task tool default topology,
  - task tool conversational/router topology if present.

Golden traces should capture topology shape, child path names, final outputs, and checkpoint summaries without depending on exact LLM prose.

### M1: Add RunContext Primitives

Implement:

- `RunContext`
- `RunStateStore`
- `NodeRunState`
- `RuntimeBindings`
- `LiveHandles`
- `InferencerStateBase`

Add unit tests for:

- root creation,
- child path stability,
- serialization/restore,
- runtime bindings not serialized,
- live handles not serialized,
- metadata/provenance preservation.

### M2: Add Keyword-Only API Support

Thread `run_context` through base inference entrypoints.

Requirements:

- Existing callers keep working in compatibility mode.
- `run_context` is stripped before prompt rendering and backend SDK calls.
- Tests prove no `run_context` kwarg leaks into templates or model clients.

### M3: Host Roots

Update application hosts to mint root contexts:

- Task tool under `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/task`
  - root context per tool execution,
  - restore from task checkpoint when resuming,
  - persist state snapshot with task artifacts.
- SOP/CLI hosts
  - root context per command.
- OpenStartup server under `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server`
  - root context at the conversation-turn boundary initially,
  - preserve option to promote to session-scoped state once resume requirements are explicit,
  - verify the context reaches the backend leaf inferencer.

This is not optional for OpenStartup. Compatibility tests may prove legacy behavior, but integration tests must prove the real host path can supply a root context.

### M4: Generic Child Routing in InferencerBase

Add helper methods:

```python
def _child_context(
    self,
    run_context: RunContext,
    segment: str | int,
    *,
    role: str | None = None,
) -> RunContext: ...
```

Composite inferencers use this for children. Examples:

- `DualInferencer`: `main`, `reviewer`, `fixer`
- `BreakdownThenAggregate`: `breakdown`, `worker/{subtask_id}`, `aggregate`
- `MultiFlowDual`: `flow/{i}`, `reviewer/{i}`, `winner`
- `PlanThenImplement`: `plan`, `implement`, `review`
- `ConversationalInferencer`: `base`

Tests:

- each child gets a distinct path,
- repeated calls reuse the same logical path when intended,
- concurrent calls do not overwrite each other's state,
- shared child instance used in two places receives different child contexts.

### M5: State Factories and Typed State

Implement `state_factory` support.

Start with low-risk orchestration fields:

- `DualInferencer`: winner/retry state.
- `MultiFlowDual`: per-flow output, reviewer output, winner index.
- `BreakdownThenAggregate`: cached subqueries, worker outputs, aggregation summary.

Keep conversational pause/resume state in `NodeRunState.conversation` first.

Tests:

- YAML-instantiated state factory creates typed state.
- Restore uses serialized state rather than creating fresh state.
- Explicit multiple typed states do not collide.
- Version mismatch has a clear migration/error path.

### M6: Stop Definition Mutation

Convert mutable definition fields into context-backed state or effective accessors.

Known targets:

- `_last_winner_idx`
- `_cached_sub_queries`
- runtime role/workspace mutation
- temporary flow/subquery overrides
- retry/attempt counters

Important attrs/slotted detail:

- `_last_winner_idx` and `_cached_sub_queries` are attrs fields today. Replacing them requires a deliberate backing-field/property/setter strategy and slots compatibility check.
- Compatibility setters may write through to current `NodeRunState`, but only during the transition.

Use effective accessors for config-like values:

```python
def _effective_flow_configs(self, run_context: RunContext) -> list[FlowConfig]: ...
def _effective_predefined_sub_queries(self, run_context: RunContext) -> list[str]: ...
```

Do not mutate reusable definition objects to inject one run's values.

### M7: Tier-3 Live Handle Canary

Pick one backend leaf inferencer as the canary, preferably the currently active CLI backend used by OpenStartup.

Goal:

- live connection/session/subprocess state moves out of reusable definition fields,
- live handles are created in branch-local context,
- handles can be reused inside one branch,
- parallel branches do not share unsafe handles,
- cleanup runs on success, error, and cancellation.

Tests:

- two concurrent branches get separate handles,
- repeated calls in one branch can reuse a handle,
- cleanup is invoked exactly once,
- serialized state does not include the handle.

Only after this can broader leaf-inferencer conversion proceed.

### M8: Factory Retirement Gate

Keep factories until the specific inferencer passes all purity gates.

Factory retirement is per class, not global.

Purity gate must check:

- no new instance attributes added during inference,
- no mutation of existing mutable definition fields,
- slotted attrs behavior,
- no class-level/shared mutable state mutation,
- no role/workspace mutation on shared children,
- no unsafe live handle sharing,
- no hidden `_active_ctx` dependency.

`vars(inferencer)` alone is not enough. Use deep snapshots of known definition fields plus class-specific mutation probes.

BTA worker isolation:

- current behavior warns when a worker is not a fresh factory result,
- do not globally convert this warning to an exception at the start,
- after a worker type is proven definition-pure, promote stricter enforcement for that type or configuration only.

### M9: Resume and Checkpoint Integration

Persist `RunStateStore.snapshot()` beside existing checkpoints.

Rules:

- New checkpoints include schema version.
- Old flat checkpoints are handled by best-effort migration where practical.
- If migration is impossible, fail with a precise message rather than silently resuming incorrectly.
- Conversation pause/resume data stays under `NodeRunState.conversation`.
- Task tool resume restores root context before instantiating or invoking the topology.

### M10: Multi-Reviewer Dual

Add multi-reviewer support as a first-class milestone, not just a side note.

Target:

- Review phase supports a panel of reviewers.
- Reviewer outputs are stored under stable reviewer child paths.
- Aggregation/merge logic is explicit and traceable.
- `MultiFlowDual` can optionally use a reviewer panel per flow or for final winner review.

Definition sketch:

```yaml
reviewers:
  - _target_: ClaudeCodeCLI
    role: correctness_reviewer
  - _target_: ClaudeCodeCLI
    role: risk_reviewer
  - _target_: ClaudeCodeCLI
    role: simplicity_reviewer
review_merge:
  _target_: ReviewMergePolicy
  strategy: weighted_consensus
```

Implementation rules:

- Reviewer instances remain definitions.
- Reviewer run outputs live in `RunContext.child("reviewer").child(i)`.
- Merge state lives in the parent node state.
- Reviewer failures are explicit:
  - fail-fast,
  - quorum,
  - best-effort with diagnostic output.
- Config must state the failure policy.

Tests:

- multiple reviewers run without state collision,
- reviewer paths are stable,
- merge result is reproducible from state snapshot,
- failed reviewer follows configured failure policy,
- existing single-reviewer `DualInferencer` behavior remains unchanged.

### M11: Examples and Documentation

Update examples under AgentFoundation:

- Add root `RunContext.root(...)` to framework examples.
- Keep legacy examples only where deliberately testing compatibility.
- Promote at least four mock/no-network examples into regular CI.
- Put real LLM/CLI examples behind nightly or opt-in integration gates.

Document:

- app responsibilities,
- inferencer responsibilities,
- state factory YAML,
- resume format,
- live handle rules,
- multi-reviewer configuration.

### M12: Final Rebaseline

Only rebaseline golden traces for intentional behavioral changes.

Do not use "rebaseline" to hide accidental drift. Any trace delta must be classified:

- expected topology/path naming change,
- expected serialization format change,
- expected multi-reviewer behavior,
- bug/regression.

## 7. Application Responsibilities

### Task Tool

The task tool owns:

- root context creation,
- loading prior `RunStateStore` on resume,
- saving snapshots,
- passing `run_context` into the root topology,
- exposing useful state/provenance in task artifacts.

It should not:

- manually assign child states,
- mutate YAML definitions for one run,
- store live handles in checkpoint payloads.

### OpenStartup Server

OpenStartup owns:

- root context at session/turn boundary,
- runtime bindings for interactive transport and graph reporting,
- state persistence policy for conversations,
- stable run ids tied to server/session/turn ids.

It should not:

- rely on inferencer instance fields to distinguish concurrent turns,
- store WebSocket queues in serializable state,
- assume compatibility mode is enough to validate integration.

### InferencerBase

`InferencerBase` owns:

- receiving and forwarding `run_context`,
- deriving child contexts,
- invoking `state_factory`,
- providing compatibility shims during migration,
- protecting backend calls from framework-only kwargs.

It should not:

- own the root state store,
- store per-run mutable state on reusable definitions,
- globally share unsafe live handles.

## 8. Test Strategy

### AgentFoundation Required Gates

Run before/after relevant milestones:

- full existing AgentFoundation unit suite,
- inferencer-specific tests,
- task tool tests,
- config instantiation tests,
- examples that are mock/no-network safe,
- lint/type checks where already used.

Add new suites:

- `test_run_context.py`
- `test_run_state_store.py`
- `test_context_threading.py`
- `test_definition_immutability.py`
- `test_state_factory.py`
- `test_live_handles.py`
- `test_multi_reviewer_dual.py`
- task-tool resume/state tests.

Use fast local gates per milestone and full suite/nightly gates at integration checkpoints. Do not make every small local edit require the entire slow integration matrix, but do require the full matrix before declaring the refactor complete.

### OpenStartup Required Gates

Run OpenStartup server tests after host integration.

Add focused tests:

- conversation service creates root context,
- context reaches backend leaf,
- two concurrent turns do not collide,
- runtime bindings carry interactive/graph reporter,
- no serializable state contains WebSocket queues or live handles.

Ignore `_runtime`, `.venv`, and other generated artifacts when auditing production reads.

### Golden/Trace Tests

Trace tests should assert:

- stable logical paths,
- no duplicate child paths,
- no state overwrite across parallel branches,
- resume loads expected node states,
- multi-reviewer state can reconstruct merge result.

### Purity Tests

For each inferencer class before factory retirement:

- deep-snapshot definition fields before/after inference,
- include attrs/slotted fields,
- include mutable containers,
- include class-level fields,
- run concurrent shared-instance test,
- verify all changes are in `RunStateStore` or `LiveHandles`.

## 9. Risk Register

### R1: Shared Instance Is Declared Safe Too Early

Mitigation: factory stays until that inferencer passes the purity gate. Do not make broad claims from one field conversion.

### R2: Hidden Live Handles Leak into State

Mitigation: serialization tests reject known live handle types; `LiveHandles` is explicitly non-serializable.

### R3: `_active_ctx` Becomes Permanent

Mitigation: avoid it as a target. If a short compatibility shim is needed, tag it with removal criteria and cover concurrent shared-instance tests.

### R4: State Factory Becomes Too Magical

Mitigation: strict signature, explicit YAML shape, versioned state, and clear errors for invalid factories.

### R5: OpenStartup Only Tests Legacy Compatibility

Mitigation: require at least one real host-path test that passes a root context into the conversation inferencer/backend.

### R6: Multi-Reviewer Changes Single-Reviewer Behavior

Mitigation: preserve default single-reviewer behavior and add regression tests around existing `DualInferencer`.

### R7: Checkpoint Migration Silently Corrupts Resume

Mitigation: versioned snapshots and explicit migration/failure behavior.

## 10. Acceptance Criteria

The work is complete only when:

- Existing AgentFoundation tests pass or any deltas are explicitly justified.
- Existing OpenStartup server tests pass or unrelated failures are documented with proof.
- Task tool can run and resume with root `RunContext`.
- OpenStartup conversation path can run with root `RunContext`.
- Child inferencer states are path-keyed and distinct.
- Shared instance concurrency tests pass for every class whose factory is retired.
- No backend receives `run_context` as a model/template kwarg.
- No serializable state contains live handles.
- Multi-reviewer config has tests for success, failure policy, stable paths, and merge reproducibility.
- Documentation explains app responsibilities versus inferencer responsibilities.

## 11. Practical Execution Order

Recommended order:

1. M0 baseline and golden traces.
2. M1 primitives.
3. M2 keyword-only API.
4. M3 host roots for task tool and OpenStartup.
5. M4 child context routing.
6. M5 state factories.
7. M6 stop definition mutation for orchestration fields.
8. M7 live-handle canary.
9. M10 multi-reviewer dual.
10. M9 resume/checkpoint integration.
11. M8 factory retirement per class.
12. M11 examples/docs.
13. M12 final trace audit.

This order intentionally puts host integration early. If applications do not own root context creation, the rest of the design can look correct in unit tests while failing in the real task/conversation paths.

## 12. Final Recommendation

Use this integrated Codex plan as the execution plan.

If only one of the three original files can be kept, keep Swift v11 because it has the most correct architecture. If only one plan should be executed by engineers, execute this integrated Codex plan because it is self-contained, incorporates Swift's corrections, keeps Revodev's migration discipline, and makes task/OpenStartup responsibilities explicit.
