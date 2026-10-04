# Integrated Plan: Generic `bta_inferencer` Self-Fan-Out

## 1. Decision

Add a generic `InferencerBase.bta_inferencer` option. When configured, the owning inferencer `P` remains the stable caller-facing and graph-facing boundary, but its normal provider execution is replaced for that call by a fresh `BreakdownThenAggregateInferencer` subtree:

```text
caller -> P prepare/render once
              |
              v
       fresh BTA execution
       /       |       \
 breakdown  fresh P workers  aggregator
              |
              v
       P publishes/finalizes aggregate once
```
caller -> P prepare/render once
This is the safe implementation of “the BTA represents the parent.” A raw construction-time replacement is rejected because current code attaches role switching, `render_only`, prompt capability, task-instruction snapshots, session handling, extraction, postprocessing, and output ownership to the original object.
       /       |       \
The plan is generic. No framework branch may name MetaMate, `flow_03`, or `research_propose`. MetaMate is only the first rollout configuration.
14. Serialized definitions or audited independent factories are the strict path. No unrestricted deepcopy or live-attrs reconstruction is a correctness primitive.
## 2. Non-Negotiable Invariants
- `MultiFlowDualInferencer` passes template-role arguments only to `TemplatedInferencerBase` instances; a raw BTA in that slot changes behavior.
1. `bta_inferencer=None` is behaviorally unchanged.
2. `bta_inferencer` is a declarative definition, never a live child.
3. The BTA worker slot is reserved for fresh instances of the owning parent definition.
4. A foreign `worker_inferencers` value is rejected before any model call; it is never silently overwritten.
5. Breakdown and aggregation may be any compatible inferencer.
6. A blank controller slot means a fresh parent-derived role using canonical BTA slot defaults.
7. The parent preprocesses and renders its effective current-role contract exactly once.
8. Workers receive only self-contained prepared subqueries; they do not rerender or rediscover the parent context.
9. Every BTA, controller, and worker is fresh per logical call, with distinct runtime state.
10. Parent-level retry/guardrail/resume never wraps the whole BTA topology.
11. Worker identity travels with checkpointed results; arrival order is never treated as declaration order.
12. Direct `infer()`, `ainfer()`, `__call__()`, iterator, and parallel-item paths agree.
13. Unsupported streaming, raw-provider-response, or session-continuation modes fail before model execution.
14. Serialized definitions or audited independent factories are the strict path. No unrestricted deepcopy or live-attrs reconstruction is a correctness primitive.
15. The first production stage is initial-turn-only, default-off, and reversible without removing the generic mechanism.
- `_is_orchestrator()` is class-structural. Pretending a guardrailed leaf is an orchestrator would trigger existing construction/recovery behavior intended for real orchestrator classes.
## 3. Why the Parent Remains the Boundary
Therefore:
A raw BTA cannot safely replace `P` in the configured graph:
- `P` remains the static node and outward contract.
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_dual_inferencer.py` aliases flow leaves and later calls `switch_role()` on those exact objects.
- `MultiFlowDualInferencer` passes template-role arguments only to `TemplatedInferencerBase` instances; a raw BTA in that slot changes behavior.
- `render_only`, `_effective_role()`, `_proposer_task_instructions()`, response postprocessing, extraction, state graphs, and canonical output are parent contracts.
- `_is_orchestrator()` is class-structural. Pretending a guardrailed leaf is an orchestrator would trigger existing construction/recovery behavior intended for real orchestrator classes.

Therefore:

- `P` remains the static node and outward contract.
- The runtime graph reporter shows `P/bta_inferencer/{breakdown,worker_N,aggregator}`.
- `P` emits a structured `delegated_to_bta` state/event.
- The BTA subtree is the actual execution representative; `P` does not call its own backend for that call.

## 4. Definition and Runtime Separation

### 4.1 Exact Factory Contract

Add a reusable definition-factory protocol:

- The runtime graph reporter shows `P/bta_inferencer/{breakdown,worker_N,aggregator}`.
- `P` emits a structured `delegated_to_bta` state/event.
    @property
    def definition_fingerprint(self) -> str: ...

## 4. Definition and Runtime Separation
### 4.1 Exact Factory Contract
    def derive(
        self,
        overrides: Mapping[str, Any],
    ) -> IndependentInferencerFactory: ...

    def create(self) -> InferencerBase: ...
Required semantics:
    def derive(
- The recipe and injectables are copied/frozen at capture time.
- `create()` runs normal construction and every `__attrs_post_init__` on a fresh unbound object tree.
- `derive()` deep-merges serialized definition values, not live instance fields.
- Every creation gets fresh inferencer IDs, loggers, caches, workspaces, clients, and sessions.
- The fingerprint is computed from the normalized serialized definition plus relevant injectables.
- No live inferencer, parent pointer, workspace, logger, RunContext, observer, transport, process, or session identifier is stored in the recipe.
- Every creation gets fresh inferencer IDs, loggers, caches, workspaces, clients, and sessions.
Extend `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_lazy_config_factory.py` to satisfy this protocol.
- No live inferencer, parent pointer, workspace, logger, RunContext, observer, transport, process, or session identifier is stored in the recipe.
Strict accepted values:
Extend `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_lazy_config_factory.py` to satisfy this protocol.
- a YAML/dict definition captured as `LazyConfigFactory`;
- an explicit `IndependentInferencerFactory`;
- an audited subclass implementation of `clone_for_independent_inference()` wrapped as a factory.

Strictly rejected values:

- a bare live inferencer without an audited factory;
- an arbitrary callable that may return a cached singleton;
- a `functools.partial` that closes over live runtime objects;
- `_PrototypeCloneFactory`/`deepcopy_with_fresh_id()` as proof of isolation.

### 4.2 Attach Definition Provenance Without Rewriting Targets

Do not add a construction-time target-replacement hook. Instead add a dependency-neutral post-materialization provenance hook to `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py`:

- an explicit `IndependentInferencerFactory`;
class DefinitionBacked(Protocol):
    def __bind_definition_factory__(
        self,
        factory: IndependentInferencerFactory,
    ) -> None: ...
- `_PrototypeCloneFactory`/`deepcopy_with_fresh_id()` as proof of isolation.

After imports, inheritance, environment overrides, `_repeat_`, aliases, `SLOT_DEFAULTS`, lazy-field capture, and Hydra construction are complete, `instantiate()` builds an immutable factory from that exact effective mapping and offers it to objects implementing the hook.
class DefinitionBacked(Protocol):
Properties:
        self,
- RichPythonUtils does not import AgentFoundation.
- The parent target remains unchanged.
- The recipe reflects canonical config semantics and exists before runtime mutation.
- A parent-derived worker factory is `P._definition_factory.derive({"bta_inferencer": None, ...})`.
- Nested BTA/controller factories continue using the existing `lazy_config_factory` mechanism.
- Direct Python construction must pass/bind an explicit parent factory before fan-out can be enabled.
    ) -> None: ...
Tests must prove definition binding happens exactly once and after effective config resolution for `_import_shared_`, `_inherits_`, environment overrides, `_repeat_`, aliases, and `SLOT_DEFAULTS`.
- The parent target remains unchanged.
### 4.3 Public Fields
- A parent-derived worker factory is `P._definition_factory.derive({"bta_inferencer": None, ...})`.
Add to `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/inferencer_base.py`:
Tests must prove definition binding happens exactly once and after effective config resolution for `_import_shared_`, `_inherits_`, environment overrides, `_repeat_`, aliases, and `SLOT_DEFAULTS`.
```python
bta_inferencer: Optional[Any] = attrib(
    default=None,
    kw_only=True,

### 4.3 Public Fields

Add to `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/inferencer_base.py`:

        "lazy_config_factory": True,


Legacy checkpoints:
bta_activation_policy: Literal[
    "always",
    "unswitched_only",
    "role_allowlist",
] = attrib(default="always", kw_only=True)

bta_allowed_roles: FrozenSet[str] = attrib(factory=frozenset, kw_only=True)
- If the checkpointed value itself contains a valid marked outcome, resume normally.

Rules:

- `inferencer_template` makes all live child walkers skip the dormant BTA definition.
- `strict_independent_factory` prevents `_wrap_bare_factory_inferencers()` from silently converting a bare object to `_PrototypeCloneFactory`.
- `unswitched_only` checks both RunContext `RoleState` and no-context role history.
- `role_allowlist` uses the same normalized effective role used by rendering.
- Every parent-derived object is created with `bta_inferencer=None` and `bta_enabled=False`.
- Explicit nested fan-out on separately configured controller definitions remains legal, with cycle and depth checks.

## 5. BTA Correctness Prerequisite

Land this independently before self-fan-out.

### 5.1 Indexed Worker Outcomes

Modify `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py`.

Every worker result is a versioned checkpoint-safe marked mapping:

- outstanding task handles;
{
    "__bta_worker_outcome__": 1,
    "worker_index": 2,
    "node_name": "worker_02",
    "query_id": "...",
    "value": "...",
    "failure_type": None,
    "failure_message": None,
}
```

Requirements:

- Wrap success, contained failure, output-file cache, and checkpoint-resume paths at the worker node.
- Sort by carried `worker_index`, never completion/arrival order.
- Preserve original indices after quorum filtering.
- Derive value, query, path, worker instance, deliverable directory, label, custom-builder arguments, synthetic output, and no-aggregator output from the same ordered outcome list.
- Share one ordering/quorum implementation between sync and async aggregation.
- Keep the public custom aggregator builder contract as aligned declaration-ordered `worker_results` and `worker_output_paths` lists.
- Remove `_last_worker_output_paths` as a second source of truth.

Legacy checkpoints:

- If the checkpointed value itself contains a valid marked outcome, resume normally.
- If a legacy value can be associated with a trustworthy worker-node identity before fan-in, migrate it once and write the new envelope.
- If identity is absent or ambiguous, reject the multi-worker checkpoint. Never derive identity from fan-in position.
- A single-worker legacy checkpoint may migrate deterministically.

### 5.2 BTA Attempt State and Cleanup

Add a RunContext-scoped `BtaAttemptState` containing:

- ordered runtime worker registry;
- controller registry;
- worker outcome map keyed by declared index;
- outstanding task handles;
- call/topology fingerprints;
- size counters.

Register each runtime child when materialized. In `finally`:

1. cancel outstanding tasks;
2. disconnect children in reverse creation order;
3. clear the registry;
4. release the parent single-flight guard.

Do not store per-call workers on reusable BTA instance fields. This fixes the current dead `_worker_instances` read without adding cross-call mutable state.

### 5.3 Aggregator Feed Ownership

Move the reusable child-context publication logic out of `MultiFlowInferencer` into a neutral helper used by both MFI and BTA.

Define exact per-key projection:

1. Walk from the target child context toward the root.
2. For each explicitly allowlisted key, take the nearest value.
3. Never inherit reserved BTA-owned keys `upstream_artifacts` or `aggregation_guidance`.
4. Publish BTA-owned values at the exact aggregator context.
5. Publish `aggregation_guidance=None` when absent.
6. Do not mutate a shared aggregator instance under a RunContext.

`aggregator_inherited_feed_keys` defaults to empty. A future follow-up rollout may explicitly allow `include_iteration_judgment`; peer artifacts remain denied.

Also add a fan-out context boundary so breakdown and prepared workers cannot rediscover ancestor task feeds already rendered into the parent contract.

### 5.4 Sync/Async and Concurrency Parity

- Sync aggregation gains the same failure filtering and quorum semantics as async.
- Existing direct-BTA argument behavior is made symmetric without changing its default policy.
- Correct the stale BTA deadlock warning. `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/RichPythonUtils/src/rich_python_utils/common_objects/workflow/workgraph.py` releases the semaphore before downstream propagation.
- Test an enabled aggregator with `max_concurrency` 1, 2, and 3.

## 6. Render Once, Execute Prepared

### 6.1 Parent Seam

Refactor the matched sync/async pipelines in `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/inferencer_base.py` into explicit phases:

1. establish RunContext/workspace/logger;
2. preprocess raw input;
3. resolve effective role/feed and render once;
4. capture task instructions;
5. honor `render_only`;
6. choose native execution or BTA delegation;
7. complete the appropriate epilogue.

The fan-out branch is after render and `render_only`, but before parent resume/retry/provider/guardrail/fallback.

Why:

- The rendered contract contains the actual initial/follow-up/reviewer/fixer role and call-scoped feed.
- Workers must not reconstruct the broad parent or peer context.
- A parent guardrail retry must not rerun decomposition and all successful workers.
- Resume belongs to the BTA subtree, not to a stale parent-level provider cache.

### 6.2 Prepared Execution API

Add matched internal APIs:

```python
- call/topology fingerprints;
- size counters.
    prepared: PreparedInferenceInput,
    inference_config: Optional[dict] = None,
    *,
    run_context: Optional[RunContext] = None,
    projected_args: Mapping[str, Any],

1. cancel outstanding tasks;
2. disconnect children in reverse creation order;
3. clear the registry;
    prepared: PreparedInferenceInput,
    inference_config: Optional[dict] = None,
    *,
    run_context: Optional[RunContext] = None,
    projected_args: Mapping[str, Any],
Do not store per-call workers on reusable BTA instance fields. This fixes the current dead `_worker_instances` read without adding cross-call mutable state.


`PreparedInferenceInput` contains:

- self-contained query text;
- declared worker index and stable subtask ID;
- explicit projected context/artifact manifest;
- byte/hash metadata.

4. Publish BTA-owned values at the exact aggregator context.
5. Publish `aggregation_guidance=None` when absent.
- input preprocessing;

- ancestor feed lookup;
- the same `bta_inferencer`.
Also add a fan-out context boundary so breakdown and prepared workers cannot rediscover ancestor task feeds already rendered into the parent contract.


- RunContext/workspace/logger setup;
- Sync aggregation gains the same failure filtering and quorum semantics as async.
- retry, attempt timeout, fallback, guardrail, and terminal `UPDATE` behavior;
- the worker clone's normal finalizer.
1. establish RunContext/workspace/logger;
This supports orchestrator parents correctly: a fresh Dual/LWI/BTA worker runs its own inner topology on the prepared shard and uses its own orchestrator finalizer. Only the outer delegated parent uses the non-polymorphic delegated epilogue because its normal topology did not run.

The parent-derived worker factory clears:

- `bta_inferencer` and activation fields;
- workspace/runtime placement;
- input preprocessor;
- response postprocessor;
- parent-level `expected_extraction` and state graphs.

It preserves model/provider, retry, timeout, fallback, guardrail, and scope-judge configuration.

### 6.3 Exact Capability and Argument Contracts

2. preprocess raw input;
3. resolve effective role/feed and render once;
4. capture task instructions;
5. honor `render_only`;
6. choose native execution or BTA delegation;
7. complete the appropriate epilogue.

    provider_request_measurement: Literal[
        "exact", "estimate", "unknown"
    ] = "unknown"

Why:

- The rendered contract contains the actual initial/follow-up/reviewer/fixer role and call-scoped feed.
- Workers must not reconstruct the broad parent or peer context.
```python
def _infer_prepared_single(
    self,
    prepared: PreparedInferenceInput,
    inference_config: Optional[dict] = None,
    *,
    run_context: Optional[RunContext] = None,
    projected_args: Mapping[str, Any],
) -> Any: ...

async def _ainfer_prepared_single(
    self,
    prepared: PreparedInferenceInput,
    inference_config: Optional[dict] = None,
    *,
    run_context: Optional[RunContext] = None,
Rules:
) -> Any: ...
- The four projection sets are pairwise disjoint.
- `common` goes to all stages; each `*_only` key goes only to that stage.
- An unknown, ambiguous, denylisted, or target-unsupported key is an error.
- There is no silent drop or implicit rename.
- `inference_config` uses its dedicated parameter.
- Framework-owned context/workspace/reporter/session/raw-response arguments cannot be projected.
- Validation for every stage finishes before rendering or model execution.

Inventory every registered concrete `InferencerBase` and every override of public/single/retry dispatch. Each must route through the common fan-out/prepared contract and declare capabilities. Integrate OpenClaw and Conversational inferencers rather than silently bypassing them; if a backend truly cannot satisfy the contract, setting `bta_inferencer` on it fails at construction with a typed capability error.
It retains:
MetaMate conformance must prove prepared execution still reaches its scope judge, `_ainfer`, streaming accumulator, final request measurement, and fresh-session behavior.

## 7. Runtime BTA Materialization

For every delegated call:

1. Acquire a single-flight guard for the parent instance/workspace.
2. Validate activation, capabilities, factories, BTA type, worker slot, paths, arguments, limits, cycles, and depth before model execution.
3. Create a fresh BTA from `bta_inferencer`.
4. Derive a fresh parent-worker factory from the bound parent definition.
5. Materialize fresh explicit controllers or parent-derived blank roles.
6. Bind the BTA to `P_workspace/children/bta_inferencer` and a child RunContext.
7. Persist the rendered contract and call manifest.
8. Call `BTA.run_topology_once()` / `BTA.arun_topology_once()`.
9. Promote the aggregate into P's canonical output, replacing only that canonical file.
10. Run P's delegated-output epilogue once.
11. Cleanup through `BtaAttemptState` in `finally`.
- An explicit self marker bound by the config loader to the same definition fingerprint is valid.
### 7.1 Reserved Worker Slot

After materialization:

- `worker_inferencers is None` is valid and is filled with the parent-derived factory.
- An explicit self marker bound by the config loader to the same definition fingerprint is valid.
- Any other worker definition is a configuration error.

The worker source yields one fresh P instance per subquery. Recursive identity validation ensures no runtime inferencer object is shared with P, the BTA definition, controllers, or another worker.

### 7.2 Blank Controller Roles

Blank controllers derive from the serialized parent definition, never a live parent.

Breakdown overlay:

1. clear parent template selectors including `template_root_space`;
2. clear input/response postprocessors and parent guardrail/fallback;
3. apply canonical `BREAKDOWN_TEMPLATE_DEFAULTS` last so `task_breakdown` wins;
4. preserve its `decomposed_subtasks` extraction;
5. clear BTA/workspace/runtime state.

Aggregation overlay:

1. clear version/mode selectors;
2. inherit the parent's effective template root;
3. clear input/response postprocessors and parent extraction/state graphs;
4. apply canonical `AGGREGATION_DEFAULTS` last;
5. keep a fresh parent guardrail so the final generated content is judged once;
6. clear BTA/workspace/runtime state.

Explicit controller definitions remain authoritative and are only freshly constructed through their factories. Blank roles require prompt-rendering capability; otherwise preflight requires an explicit controller.

### 7.3 Single-Shot Topology Ownership

A represented BTA must not enter its inherited whole-inferencer retry/fallback/guardrail wrapper.

Add explicit `run_topology_once()` and `arun_topology_once()` methods that:

- execute one breakdown/fan-out/aggregation topology;
- enforce one total topology deadline;
- allow stage-local retry policies;
- use BTA checkpoint/resume;
- reject explicit conflicting BTA-level retry/fallback/guardrail settings.

This prevents a terminal aggregate `UPDATE` or failure from rerunning completed workers.

### 7.4 Delegated Output Epilogue

Extract the duplicated sync/async normal/resume epilogue into a shared helper and separate base leaf finalization from orchestrator-specific child promotion.

Delegated order:

1. atomically replace P's canonical output link/file with the BTA aggregate;
2. apply base leaf extraction/manifest behavior;
3. update P's state graphs;
4. run P's response postprocessor;
5. return normalized text.

Do not call P's orchestrator override because P's normal children did not execute. The BTA and worker clones already ran the correct finalizers for their actual topologies.

## 8. Input and Memory Safety

### 8.1 Structured Shard Schema

Add a versioned self-fan-out breakdown prompt/schema under:

### 7.2 Blank Controller Roles

Every shard must contain:

Breakdown overlay:
- focused question;
- repository/path or resource scope;
2. clear input/response postprocessors and parent guardrail/fallback;
3. apply canonical `BREAKDOWN_TEMPLATE_DEFAULTS` last so `task_breakdown` wins;
4. preserve its `decomposed_subtasks` extraction;
5. clear BTA/workspace/runtime state.

- stop conditions;
The original rendered contract is sent only to the non-MetaMate breakdown controller. It is never concatenated into worker queries.
- Workers inherit no ancestor feed by default.
### 8.2 Deny-by-Default Context Projection

- Workers inherit no ancestor feed by default.
- Only shard-selected context/artifacts may be attached.
- References use a versioned manifest: path, hash, size, media type, and allowed reader budget.
- Reserved peer bundles and BTA-owned keys cannot cross into workers.
- Scope directives and files/tools touched are recorded for empirical partition checks.

### 8.3 Enforceable Limits

Add configurable limits for:

- full breakdown input bytes;
- structured worker-query bytes;
- projected context/attachment count and bytes;
- each final provider request;
- cumulative worker request bytes per attempt;
- auto-continuation requests;
- worker output bytes;
- aggregation input and artifact-read bytes.

A provider request-size hook returns `exact`, `estimate`, or `unknown`. A configured hard ceiling requires exact measurement. MetaMate enforces immediately before every `engine_start_v2` call, after scope injection and all request fields are known.

All oversize cases fail explicitly. Nothing truncates silently.

### 8.4 Bounded Aggregation

Add backward-compatible `aggregation_original_query_mode`:

- `inline`: current behavior;
- `artifact_reference`: persist the full contract once, give the aggregator a bounded task/response-contract summary, aggregation guidance, and budgeted artifact path.

The rollout uses `artifact_reference` and a local aggregator. A path is not counted as zero bytes: the artifact reader enforces its manifest/read budget.

Byte limits are regression controls, not proof of solving the 512 MiB remote failure. Acceptance also requires narrower observed scopes and zero memory-limit terminations.

## 9. Resume and Checkpoint Identity

Before any breakdown, expansion, worker-output, or provider-cache reuse, validate:

- Reserved peer bundles and BTA-owned keys cannot cross into workers.
- represented-parent definition fingerprint;
- Scope directives and files/tools touched are recorded for empirical partition checks.

- hash of the rendered parent contract;
- effective role;
- normalized relevant call arguments;
- structured decomposition hash;
- ordered expanded worker-query hashes;
- ordered context/artifact manifest hashes.

A mismatch fails before cache consumption and requests a fresh versioned workspace. Fresh Python objects may resume the same logical workspace only when all identities match.

New rollout workspaces include the outcome/decomposition schema version in their namespace. Once indexed checkpoints exist, rollback must retain the new reader; use a forward hotfix to disable fan-out rather than reverting to an old reader that could misinterpret marked mappings.

## 10. Streaming, Sessions, Depth, and Errors

- `infer()`, `ainfer()`, `__call__()`, iterator items, and parallel-item paths use the same decision seam.
- `render_only=True` returns P's rendered contract and does not materialize BTA.
- Direct `infer_streaming()`/`ainfer_streaming()` raises a typed unsupported-mode error in v1 when fan-out is selected.
- Explicit session continuation and raw SDK response modes raise before model execution.
- Every MetaMate worker begins with empty `_conversation_uuid` and `_conversation_fbid`; `reset_session()` runs defensively after construction.
- Nested explicit fan-outs are bounded by RunContext path depth, maximum 8.
- Cyclic definition fingerprints fail preflight.
- Concurrent fan-out on the same parent instance/workspace fails clearly; separate instances/workspaces and workers within one call remain concurrent.

## 11. MetaMate Rollout

### 11.1 Pin the Real Topology First

Do not assume that MetaMate is always index 3. Before creating the rollout preset:

1. Capture the exact production launch at source revision `5b801928eb1a`.
2. Record the resolved base config, ordered flow targets, and complete non-secret `TASK__*` / `RESEARCH_PROPOSE__*` override manifest.
3. Check the immutable snapshot into:

`/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/fixtures/research_propose_metamate_bta_source.yaml`

4. Record and assert its SHA-256 in the config test.

### 11.2 Named, Default-Off Preset

- cumulative worker request bytes per attempt;

- worker output bytes;
A provider request-size hook returns `exact`, `estimate`, or `unknown`. A configured hard ceiling requires exact measurement. MetaMate enforces immediately before every `engine_start_v2` call, after scope injection and all request fields are known.
The preset:
All oversize cases fail explicitly. Nothing truncates silently.
- imports the base topology;
- replaces the entire `flow_configs` list with explicit mappings derived from the pinned snapshot;
- keeps each target and its BTA stanza in the same mapping;
- does not use a parallel positional BTA list;
- is selected explicitly through `--agent-config` or equivalent existing config selection;
- does not change the default config in `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/research_propose/tool.json`.

Stage-1 MetaMate initial leaf:


Add backward-compatible `aggregation_original_query_mode`:

- `inline`: current behavior;
  bta_activation_policy: unswitched_only
- `artifact_reference`: persist the full contract once, give the aggregator a bounded task/response-contract summary, aggregation guidance, and budgeted artifact path.

The rollout uses `artifact_reference` and a local aggregator. A path is not counted as zero bytes: the artifact reader enforces its manifest/read budget.

Byte limits are regression controls, not proof of solving the 512 MiB remote failure. Acceptance also requires narrower observed scopes and zero memory-limit terminations.

## 9. Resume and Checkpoint Identity

Before any breakdown, expansion, worker-output, or provider-cache reuse, validate:
      - stop_conditions
- BTA definition fingerprint;
- decomposition schema version;
- hash of the rendered parent contract;
- effective role;
- normalized relevant call arguments;
    aggregation_original_query_mode: artifact_reference
- ordered expanded worker-query hashes;
      _target_: ClaudeCodeCLI

      _target_: ClaudeCodeCLI


The full checked-in mapping must also include every inherited timeout, retry, guardrail, template, and follow-up field from the pinned source; the fragment above is not a substitute for the full mapping.

Stage 1 rules:

- `enable_metamate_initial_bta` defaults false.
- The follow-up mapping is byte-equivalent to the pinned source and contains no BTA field.
- Reviewer/fixer reuse of the initial object is blocked by `unswitched_only`.
- Both controllers are pinned to the registered local `ClaudeCodeCLI`, not an environment-overridable main inferencer.
- Preflight verifies local execution, prompt rendering, exact/bounded artifact reads, and required templates.
- A separate follow-up preset/diff is designed only after stage-1 acceptance; there is no inert stage-1 follow-up switch.

Update `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/scripts/run_research_propose.py` only as needed to select/log the preset and initial kill switch. Preserve the existing default tool behavior.

### 11.3 Reproducer

Preserve the current private `_ainfer()` arm in `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/scripts/reproduce_flow03.py` as the raw one-turn control.

Add separate production-shaped public `ainfer()` arms for:

- initial fan-out;
- partial-workspace resume.

Use the captured raw plan subtask, not a doubly rendered prompt. Map timeout controls to the actual MetaMate streaming timeout. Keep the current hard-OOM and graceful-give-up classifiers.

## 12. Implementation Stack

### Diff 1: Canonical BTA Correctness

Files:

### 11.1 Pin the Real Topology First

- `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/inferencer_base.py`
- BTA, resume, checkpoint, feed, and concurrency tests

Deliver indexed outcomes, sync/async quorum parity, general aggregator feed ownership, `BtaAttemptState`, cleanup, and corrected concurrency documentation. No self-fan-out API yet.
2. Record the resolved base config, ordered flow targets, and complete non-secret `TASK__*` / `RESEARCH_PROPOSE__*` override manifest.
### Diff 2: Definition Factories

Files:
`/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/task/configs/breakdown-multiflow-plan-metamate-bta.yaml`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_lazy_config_factory.py`
- replaces the entire `flow_configs` list with explicit mappings derived from the pinned snapshot;
- does not use a parallel positional BTA list;
- is selected explicitly through `--agent-config` or equivalent existing config selection;
- canonical BTA definition fields
- does not change the default config in `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/resources/tools/research_propose/tool.json`.
Deliver immutable factory provenance, `derive()`, fingerprints, strict field handling, public slot-default collection, and parent-derived role overlays. Behavior remains disabled.
Stage-1 MetaMate initial leaf:
### Diff 3: Prepared Execution and Delegation
  bta_inferencer:
Files:
    breakdown_format: json_subtasks
      - exclusions
      - deliverable_format
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/streaming_inferencer_base.py`
      - stop_conditions
    max_concurrency: 3
- every registered public-entrypoint override identified by the inventory test

Deliver the render-once seam, typed prepared execution, capabilities/projection, fresh BTA materialization, single-shot topology execution, delegated epilogue, role activation, graph events, and lifecycle cleanup.

### Diff 4: Memory Boundaries

- `enable_metamate_initial_bta` defaults false.

- Reviewer/fixer reuse of the initial object is blocked by `unswitched_only`.
- base/templated inferencer boundary helpers;
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/external/metamate/metamate_sdk_inferencer.py`;
- versioned breakdown schema/prompt assets.
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_inferencer.py`
Deliver all byte/read ceilings, exact provider-boundary enforcement, artifact manifests, `artifact_reference` aggregation, and telemetry.
- BTA, resume, checkpoint, feed, and concurrency tests
### Diff 5: Default-Off MetaMate Rollout


- `test_bta_worker_outcome`;
- pinned resolved-topology fixture;
- named preset;
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/scripts/run_research_propose.py`;
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/scripts/reproduce_flow03.py`;
- config and harness tests.
- `test_inferencer_self_fanout`;
Dependencies are strict: Diff 2 depends on Diff 1; Diff 3 on Diff 2; Diff 4 on Diff 3; Diff 5 on Diff 4. Every stack point must construct and test cleanly.

## 13. Test Matrix

### 13.1 BTA Correctness

- completion order `[2,0,1]` retains declaration indices and paths;
- middle-worker failure retains every survivor's original path;
- local, non-local inline, custom builder, synthetic fallback, and no-aggregator paths;
4. Run the full AgentFoundation inferencer suite before any live experiment.
- indexed pickle/`jsonfy` round-trip;
- trustworthy node-identity migration and ambiguous legacy rejection;
- exact child-context feed ownership and no-context compatibility;
- concurrency 1/2/3 with aggregator;
- cleanup on success, error, cancellation, and timeout.
7. Follow-up design/review only after stage 1 passes.
### 13.2 Definition and Isolation
- worker/result/path indices;
- source mapping mutation cannot change a captured factory;
- parent runtime mutation cannot change the serialized definition;
- template fields are skipped by live child walkers and prototype wrappers;
- each factory call yields disjoint root and descendant identities;
- no parent pointer, workspace, logger, cache, observer, client, or session is shared;
- MetaMate conversation fields start empty;
- blank controller overlay order is exact;
- foreign workers and unaudited bare objects fail before model calls;
- every registered alias has factory/capability coverage.
- aggregate schema, citations, and proposal index;
### 13.3 Delegation
## 15. Rollback
- disabled path is unchanged;
- one parent preprocess/render and one outward epilogue;
- zero worker preprocess/render/ancestor-feed passes;
- parent provider/retry/guardrail never wraps BTA;
- worker and aggregator policies execute only at their stages;
- leaf and orchestrator parents work, including Dual, LWI, and BTA-as-parent;
- exact argument projection rejects unknown/ambiguous/session/raw keys;
- OpenClaw, Conversational, CLI, MetaMate, iterator, and parallel entrypoints conform;
- stale canonical output is atomically replaced;
- same-parent/workspace concurrency rejects; separate instances isolate;
- direct streaming fails before execution.
- Disable the initial BTA switch and drain active runs first.
### 13.4 Resume and Memory

- every call/decomposition/query/context fingerprint is validated before cache reuse;
- completed workers resume with fresh Python objects and do not rerun;
- original-contract sentinel reaches breakdown only;
- worker provider requests contain only their shards;
- all per-stage and cumulative ceilings fail without truncation;
- auto-continuations are measured;
- artifact reads obey manifests and budgets;
- scope/file telemetry demonstrates partitioning.

### 13.5 Rollout Configuration
| Untyped `prepared_input=True` kwarg | Can leak through provider kwargs and is not a capability contract. |
- pinned source fixture hash matches;
- explicit full flow order matches the captured production topology;
- MetaMate and BTA are in the same mapping;
- only the initial unswitched MetaMate turn is eligible;
- follow-up and reviewer/fixer calls are unchanged;
- local controllers satisfy capabilities;
- kill switch restores the previous topology;
## 17. Side-by-Side Comparison

Register focused targets in:

`/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/BUCK`
| General aggregator feed ownership | Strong intent | Strong intent | Strong: per-key child projection + boundary | Partial/barrier-centric | Strong: child-context publication |
Use names:
| Leaf and orchestrator parent semantics | Partial | Strong intent | Strong and explicit | Partial | Partial |
- `test_bta_worker_outcome`;
- `test_inferencer_definition_factory`;
- `test_inferencer_self_fanout`;
- `test_metamate_bta_preset`.

Retain and run existing targets for BTA, BTA resume, worker resume, unified finalization, task-instruction snapshots, output guardrails, RunContext feed isolation, and MetaMate timeout/session behavior.

## 14. Verification and Empirical Gates

Baseline at `5b801928eb1a`: last known 321 passed, 0 failed. Record the exact baseline command and output before implementation.

After each diff:

1. Run the owning focused BUCK targets, including:
   - `fbcode//_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers:test_breakdown_then_aggregate`
   - `fbcode//_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers:test_bta_worker_resume`
   - `fbcode//_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers:test_bta_resume_original_query`
   - `fbcode//_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers:test_bta_resume_workspace_binding`
   - `fbcode//_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers:test_unified_finalize_output`
   - the four new targets above.
2. Resolve every other changed-file owner with `buck uquery` and add it to the same test run.
3. Run `arc f`, `arc lint -a`, and `arc pyre check-owning-targets`.
4. Run the full AgentFoundation inferencer suite before any live experiment.

Live gates:

1. Raw MetaMate turn, scope judge off.
2. Raw MetaMate turn, scope judge on.
3. Initial fan-out, scope judge on.
4. Partial-workspace fan-out resume.
5. At least eight fresh fan-out runs.
6. Full initial-only `research_propose` run after the eight-run gate.
7. Follow-up design/review only after stage 1 passes.

Record per stage:

- query/rendered/final-provider/output/aggregate bytes and hashes;
- cumulative fan-out bytes;
| Streaming semantics | Unsafe: buffered chunk called streaming | Strong: fail loud | Strong: fail loud | Unsafe: buffered chunk | Unsafe: buffered chunk |
| Provider/session argument projection | Partial hooks | Strong intent | Strong: exact typed projection | Partial | Partial |
- hard-OOM and graceful-give-up markers;
- retries, timeouts, and elapsed time;
| Artifact-reference aggregation | Missing/deferred | Strong | Strong | Missing/deferred | Strong |
- substantive/guardrail verdicts;
- aggregate schema, citations, and proposal index;
- checkpoint/output topology.

### Single-Plan Choice

- zero memory-limit terminations across at least eight runs;
- distinct conversation for every worker;
- every enforced size/read ceiling passes;
- original contract and peer bundle absent from worker provider requests;
It is the only plan that simultaneously provides:
- at least 2 of 3 workers substantive;
- render-once prepared execution;
- strict serialized-definition isolation without live reconstruction;
- prerequisite BTA identity/feed/lifecycle fixes;
- flow_03 evidence reaches the final aggregator, improving on 0 of 197 unique sentences;
- no regression in non-MetaMate flows.
- full checkpoint call identity;
## 15. Rollback
- a pinned, non-positional, initial-only rollout;
- Disable the initial BTA switch and drain active runs first.
- New rollout workspaces use a schema-versioned namespace.
- Once indexed checkpoints exist, do not revert to the old reader. Use a forward hotfix that disables fan-out while retaining indexed-outcome parsing.
- Quarantine new-schema workspaces from old binaries.
- Revert rollout independently from the generic feature.
- Revert generic behavior independently from BTA correctness; retain correctness fixes unless they regress independently.
- Test legacy read, indexed resume, disabled-feature resume rejection, and attempted downgrade handling before rollout.
## 16. Rejected Approaches
| Approach | Reason |
|---|---|
| Replace P with a raw BTA during graph construction | Breaks parent identity, role/type checks, render-only behavior, task snapshots, and outward finalization. |
| Rebuild from live attrs fields | Definition/runtime provenance is already lost; workspace loggers and derived orchestrator fields leak. |
| Global `__new__` or `__attrs_pre_init__` recipe capture | Broad invisible behavior for every inferencer; unnecessary once the loader records canonical definition provenance. |
| Unrestricted deepcopy / `_PrototypeCloneFactory` | MetaMate contains lock-bearing state; fresh IDs do not imply fresh runtime resources. |
| Workers rerender the parent template | Recreates the large follow-up/peer context in every shard. |
| Untyped `prepared_input=True` kwarg | Can leak through provider kwargs and is not a capability contract. |
| Warn and overwrite foreign workers | Hides configuration errors and weakens the hard parent-worker invariant. |
| Positional per-flow BTA list | Reordering and `_repeat_` behavior can target the wrong flow. |
| Single-chunk “streaming” | It is buffering, not streaming; fail explicitly in v1. |
| Parent or BTA whole-topology retry | Can repeat successful expensive workers and recreate budget failures. |
| Positional legacy checkpoint migration | Arrival order cannot prove declaration identity. |
| Only prompt-char limits | The failure is dominated by in-turn tool accumulation; exact request and empirical scope checks are also required. |
| Initial and follow-up rollout together | Doubles blast radius before initial-turn behavior is proven. |
## 17. Side-by-Side Comparison
Legend: **Strong** = sound and implementable direction; **Partial** = valuable but incomplete; **Unsafe** = violates a load-bearing invariant.
| Criterion | Snappy v3 | Idempotent | Updated Canonical | Self-Fan-Out v2 | Harmonic |
| Parent remains stable boundary | Strong | Strong | Strong | Strong | Strong |
| Render once; workers receive only shards | Strong | Strong | Strong | Strong | Unsafe: workers rerender parent |
| Strict definition/runtime separation | Unsafe: hidden live constructor recipe | Strong intent, incomplete mechanics | Strong: loader-bound immutable definition | Unsafe: global attrs recipe/live leaves | Unsafe: live-field reconstruction |
| Foreign worker handling | Unsafe: warns/overwrites | Strong: rejects | Strong: rejects | Unsafe: warns/overwrites | Strong: rejects |
| BTA arrival/quorum identity | Partial: indexed, permissive legacy shim | Strong | Strong | Partial: permissive legacy shim | Partial: fallback index shim |
| Resume request identity | Missing | Partial | Strong: full call/decomposition/query fingerprints | Missing/known limit | Missing |
| General aggregator feed ownership | Strong intent | Strong intent | Strong: per-key child projection + boundary | Partial/barrier-centric | Strong: child-context publication |
| Runtime worker cleanup | Partial: instance list, sync leak | Partial | Strong: RunContext attempt registry | Partial | Partial |
| Leaf and orchestrator parent semantics | Partial | Strong intent | Strong and explicit | Partial | Partial |
| Direct entrypoint coverage | Partial: explicit exclusions | Strong intent | Strong: capability inventory and conformance | Partial: OpenClaw excluded | Partial |
| Streaming semantics | Unsafe: buffered chunk called streaming | Strong: fail loud | Strong: fail loud | Unsafe: buffered chunk | Unsafe: buffered chunk |
| Provider/session argument projection | Partial hooks | Strong intent | Strong: exact typed projection | Partial | Partial |
| Exact provider-boundary budgets | Partial telemetry/char cap | Strong | Strong | Missing/deferred | Partial/deferred |
| Artifact-reference aggregation | Missing/deferred | Strong | Strong | Missing/deferred | Strong |
| Rollout selection safety | Strong named preset, but both turns | Strong named preset | Strong: pinned explicit initial-only preset | Unsafe positional list | Strong named preset |
| Empirical rollout gates | Strong | Strong | Strongest and staged | Strong | Strong |
| Implementation detail | Highest | Medium | Highest | Highest | High |
| Safe to implement verbatim | No | No | Yes after approval | No | No |
### Single-Plan Choice
If only one of the five current files may be selected, choose this updated canonical plan:

`/data/users/zgchen/fbsource/fbcode/_tony_dev/CoreProjects/AgentFoundation/bta_inferencer.plan.md`

It is the only plan that simultaneously provides:

- render-once prepared execution;
- strict serialized-definition isolation without live reconstruction;
- prerequisite BTA identity/feed/lifecycle fixes;
- exact argument and capability contracts;
- full checkpoint call identity;
- exact provider-boundary and artifact-read budgets;
- a pinned, non-positional, initial-only rollout;
- downgrade-safe checkpoint handling;
- a dependency-ordered implementation and validation stack.

If the canonical plan were excluded and one of the other four had to be chosen unchanged, choose `/home/zgchen/.llms/plans/idempotent-enchanting-lake.md`. It has the safest core architecture, but still needs the concrete factory provenance, cleanup, call-fingerprint, argument-projection, and rollout mechanics supplied here.
