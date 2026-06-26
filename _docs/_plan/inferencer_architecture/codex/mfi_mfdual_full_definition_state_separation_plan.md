# Integrated MFI/MFDual State Separation + BTA `worker_inferencers`

Status: integrated Codex plan, 2026-06-22

This plan integrates the strongest parts of:

- `/Users/tchen7/.claude/plans/mfi-mfdual-state-separation.md`
- `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plan/inferencer_architecture/revodev/mfdual_mfi_full_state_definition_separation_plan.md`

It also records the main rejected idea from the Revodev plan: aliasing
`worker_inferencers` to `workers` and reusing one eager shared MFDual instance.
That may become viable later, but it is not the correct first implementation
because BTA still performs per-worker runtime setup and the current purity gate
explicitly permits MFI/MFDual mutation carve-outs.

## Summary

Use `/Users/tchen7/.claude/plans/mfi-mfdual-state-separation.md` as the base
plan. It is closest to current source truth. Pull in the Revodev plan's concrete
user-facing goal: allow `breakdown-multiflow-plan.yaml` to declare a BTA worker
with:

```yaml
worker_inferencers:
  _target_: MultiFlowDual
  propagate_runtime_input: true
  winner_pick: true
  reviewer_match_second: true
  fixer_match_winner: true
  visible_flows: all
  max_retry: 3
  flow_configs:
    ...
  multi_flow_aggregator_inferencer:
    ...
```

Do not adopt the unsafe shortcut of aliasing `worker_inferencers` to `workers`
and reusing one eager shared instance in the first implementation. Current BTA
still performs per-worker runtime setup, and current MFI/MFDual still have
runtime mutation carve-outs. The correct first implementation is:

- make MFI/MFDual runtime state context-backed;
- make BTA `worker_inferencers` a lazy worker definition;
- instantiate a fresh worker per expanded subtask, matching today's
  `worker_factory` safety model while giving the cleaner YAML shape.
- keep arbitrary shared-singleton MFDual reuse out of scope until inherited
  Dual/LWI runtime fields are audited and either moved into context or proven
  immutable. The concrete BTA goal does not need shared-singleton reuse because
  `worker_inferencers` creates fresh workers.

If only one plan is chosen after this integration, choose this Codex plan file.
It preserves Claude's accurate MFI/MFDual mechanics, adds Revodev's
`worker_inferencers` goal, and avoids Revodev's risky shared-instance shortcut.

## Source-Verified Current State

- BTA currently exposes `worker_factory` and `workers`, but no
  `worker_inferencers`.
- `worker_factory` is currently the lazy fresh-subtree path because
  RichPythonUtils wraps attrs fields ending in `_factory` into
  `LazyConfigFactory`.
- `workers` is a prebuilt list served round-robin. It is not a lazy definition
  mechanism.
- `breakdown-multiflow-plan.yaml` currently uses:

```yaml
worker_factory:
  _factory_: MultiFlowDual
  ...
```

- MFI still stores runtime dispatch/attempt state on `self`, including
  `_latest_per_flow`, `_all_judgments`, `_last_winner_idx`,
  `_last_reviewer_alias`, `_last_fixer_alias`, and `_last_ranking`.
- MFDual still mutates `self.review_inferencer`, `self.fixer_inferencer`, and
  `self.reviewers` during runtime dispatch.
- The current run-context API is `ctx.node().call` and `ctx.node().attempt`, not
  `ctx.node.state`.
- Current typed states already include `MultiFlowState`, `DualState`, and
  `MFDualState` by composition.
- `LinearWorkflowInferencer` already uses `ctx.node().call` as its workflow
  state dictionary through `_pending_state`; a typed `MFDualState` cannot simply
  replace that slot without colliding with LWI workflow state.
- Current `test_m7_purity_gate.py` intentionally allows the exact mutations this
  plan removes. Updating that gate is part of the work, not a test cleanup.
- MFI also mutates child prompt feeds at runtime by writing
  `template_extra_feed["upstream_artifacts"]` on follow-up and aggregator
  inferencers. This is runtime state and must be made per-call/context-scoped.
- Dual direct role reads are broader than the review/fix step bodies. Marker
  writing, active proposer resolution, child iteration, audit, and panel logic
  must be considered when adding role accessor seams.

## Key Design Decisions

### 1. `worker_inferencers` is lazy, not shared eager state

Add `worker_inferencers` as a BTA field whose config is preserved as a lazy
definition and invoked once per expanded subtask.

Do not implement it as a direct alias to `workers`. A single shared instance can
be reconsidered only after the full purity proof is complete and BTA's own
per-worker writes are either virtualized or proven harmless.

Implementation rule: `worker_inferencers` must produce a `LazyConfigFactory` (or
equivalent lazy config wrapper) and BTA must call it for each expanded subtask.
It must not be eagerly instantiated into a single `InferencerBase` object during
Hydra/config construction.

### 2. Preserve existing worker surfaces

Keep the existing meanings:

- `worker_factory`: existing dynamic fresh-worker mechanism; backward-compatible.
- `workers`: existing static prebuilt list; round-robin; unchanged.
- `worker_inferencers`: new declarative lazy worker definition; fresh worker per
  expanded subtask.

If more than one of these is configured, fail fast with a clear error. Do not
guess precedence.

### 3. Use current run-context state shape

Extend current `MultiFlowState` and `MFDualState`. Do not introduce a parallel
`MultiFlowDualState` class name unless the existing `MFDualState` is renamed in
a separate migration.

Use:

- `ctx.node().call` for per-call dispatch state;
- `ctx.node().attempt` for per-retry-attempt working state.

Do not use stale names such as `ctx.node.state`, `MultiFlowDualState`, or raw
`review_target` object fields in serialized state. The existing state model is
`MFDualState(dual=DualState(), multiflow=MultiFlowState())`.

MFI can use `MultiFlowState` directly as its call state. MFDual is different
because it inherits LWI, and LWI already expects its call state to be a workflow
dictionary. Therefore MFDual must not blindly set `state_factory = MFDualState`
unless the LWI workflow state is first moved to a separate slot. In this plan,
preserve the LWI dictionary and store MFDual state under a reserved key, for
example:

```python
MFDUAL_STATE_KEY = "__mfdual_state__"

def _mfdual_state(ctx: RunContext) -> MFDualState:
    call = ctx.node().call
    if call is None:
        call = {}
        ctx.node().call = call
    if not isinstance(call, dict):
        raise TypeError("MFDual requires dict workflow call state")
    return call.setdefault(MFDUAL_STATE_KEY, MFDualState())
```

This keeps LWI workflow state and MFDual dispatch state co-resident but not
confused. A later broader LWI migration can move workflow state into a typed
`LinearWorkflowState`; that is not required for the concrete BTA
`worker_inferencers` goal.

### 4. Runtime role selection resolves from state

MFDual should stop mutating role-definition attributes during dispatch. Instead,
it should compute selected reviewer/fixer/panel roles from current call state
and resolve those roles at point of use.

The inner MultiFlow propose step runs under Dual's `"propose"` child context, so
MFDual should read the inner MultiFlow dispatch state from that child context
instead of reading `mfi._last_winner_idx` or related instance fields.

The selected role state should store serializable refs, not raw inferencer
objects. Runtime code resolves those refs back to objects from the definition.

### 5. Prompt-feed writes are runtime state

Do not mutate a shared child inferencer's `template_extra_feed` to pass
per-flow artifacts. Move upstream artifacts into per-call render/feed data.

If the existing LWI or BTA call sites only accept a string input today, add the
smallest explicit seam needed for a dynamic input builder or aggregator call to
return both:

- the prompt/input text;
- call-scoped `extra_feed` or render kwargs.

The child inferencer definition remains unchanged across flows and runs.

## Implementation Plan

### Commit 0: Preflight and guardrails

Before implementation, add or update focused failing tests that prove the exact
target behavior:

- BTA config parsing currently strips unknown `worker_inferencers`; the test
  should fail before Commit 6 and pass after.
- `worker_inferencers` with `_target_: MultiFlowDual` must become a lazy
  definition, not an eagerly constructed shared object.
- Current purity gate carve-outs for MFI dispatch and MFDual role references
  must be removed by the end of the plan.
- Current legacy post-call getter behavior must be captured before changing MFI
  dispatch storage.

This preflight prevents the implementation from accidentally solving only the
YAML syntax while leaving the state separation incomplete.

### Commit 1: Complete MFI call and attempt state

Update `run_context/state.py`:

- Add fields to `MultiFlowState` for:
  - `winner_idx`
  - `reviewer_alias`
  - `fixer_alias`
  - `ranking`
  - `effective_sub_queries`
  - `flow_inputs`
- Add `MultiFlowAttemptState` for:
  - `latest_per_flow`
  - `judgments`

Update `MultiFlowInferencer`:

- Set `state_factory` to produce `MultiFlowState`.
- Route `_extract_dispatch_state()` writes to `ctx.node().call`.
- Route getter reads through the active call state when available.
- Move `_latest_per_flow` and `_all_judgments` into `ctx.node().attempt`.
- Capture the parent MFI run context before building dynamic worker closures.
  Worker callbacks run under worker child contexts, but `_latest_per_flow` and
  judgments are parent-MFI attempt state. Closures must write through the
  captured parent context, not `active_run_context()` from inside the worker.
- Reset typed call and attempt fields at call/attempt start. Do not rely only
  on `state_factory` freshness, because `_init_call_state` intentionally does
  not overwrite an existing node.
- Replace follow-up/aggregator `template_extra_feed["upstream_artifacts"]`
  mutation with per-call `extra_feed` or render kwargs.
- Preserve legacy direct-call behavior through explicit compatibility backing.

Important compatibility point: current `propagate_runtime_input=True` writes only
when `ctx.node().call` is `None` or a `dict`. That must be updated so typed
`MultiFlowState` receives `effective_sub_queries` and `flow_inputs`, because BTA
already reads `effective_sub_queries` from typed call state.

The old instance attributes may remain only as private compatibility backings for
legacy direct calls. They must no longer be the source of truth under an explicit
application or child run context.

### Commit 2: Add explicit legacy-mint tracking

Add a `legacy_mint` marker to `RunContext` or an equivalent explicit flag in the
bridge.

Purpose:

- direct `mfi.infer()` without a supplied context can preserve the historic
  post-call getter pattern;
- explicit application contexts and nested child contexts must not write
  runtime state back to shared instance fields.

The compatibility setter rule:

- active non-legacy context: write only to context state;
- legacy-minted root: may mirror to legacy backing for post-call getters;
- children of a legacy-minted root must preserve the marker unless there is a
  deliberate child boundary that opts out. This keeps existing bare direct-call
  behavior intact through nested flows;
- explicit application contexts remain isolated and must never be treated as
  legacy-minted merely because they have no parent;
- concurrent use of one shared definition requires explicit non-legacy sibling
  contexts. Bare direct calls remain legacy-compatible, not concurrency-safe.

### Commit 3: Replace MFDual mutation with role resolution

Replace `_select_reviewer_and_fixer()` with a pure resolver that returns selected
role refs.

Role refs should be serializable when possible:

- flow index refs for winner/runner-up/non-winner selections;
- pool alias refs for `inferencer_pool` selections;
- static slot refs for configured defaults.

Store selected refs in the reserved MFDual state object, using the current
composed state shape:

- `MFDualState.multiflow` for MultiFlow dispatch details;
- `MFDualState.dual` for selected reviewer/fixer/panel refs and workspaces.

Do not store raw inferencer objects as durable state. Resolve objects at point of
use from refs and the current definition.

Extend `DualState` only with serializable ref fields, for example:

- `reviewer_ref`
- `fixer_ref`
- `panel_refs`
- `review_workspace`
- `fix_workspace`

The exact ref encoding should distinguish flow-index refs from `inferencer_pool`
alias refs and static slot refs.

Because MFDual inherits LWI, access this `MFDualState` through the reserved-key
helper described in Decision 3, not by replacing `ctx.node().call` wholesale.

Before declaring MFDual shared-definition safe, audit inherited Dual/LWI runtime
fields touched by MFDual runs, including `_state`, `_pending_state`,
`_current_attempt`, `_current_inference_config`,
`_current_extra_inference_args`, `_current_round_ws`, `step_configs`, and
`_last_output_child_ws`. Either move each runtime value into context/local
scope, or keep shared-singleton reuse explicitly out of the acceptance claim.

### Commit 4: Add Dual role accessor seams

Add accessor methods in Dual for:

- effective review inferencer;
- effective fixer inferencer;
- effective reviewer panel;
- effective role workspaces.

Default implementation returns current static attributes, preserving normal
Dual behavior.

MFDual overrides these accessors to use the selected refs stored in context
state.

Bind selected role objects to local variables inside review/fix steps so each
round uses one stable role resolution.

Add or update the run-context lint/purity gate so direct review/fix loop reads
of `self.review_inferencer`, `self.fixer_inferencer`, and `self.reviewers` are
only allowed inside the approved accessor methods. This catches missed call
sites in Dual's long review/fix implementation.

The lint/gate should cover all runtime-sensitive role reads, not only the main
review/fix loops:

- marker writing;
- active proposer/reviewer/fixer resolution;
- child iteration when it reflects active runtime roles;
- audit/output metadata;
- panel logic.

Static definition introspection may still read static attrs, but runtime
decision points must go through the accessors.

### Commit 5: Make MFDual role workspace/template changes context-scoped

Under an active non-legacy context, do not call `switch_role()` in a way that
mutates the selected shared reviewer/fixer instance.

Instead:

- publish review/fix workspace into the child context;
- publish role/template info into `RoleState`;
- keep selected leaf session/handle state isolated by child context path.

Legacy no-context behavior may keep the old mutation path where needed for
compatibility.

Do not assume `RoleState` alone solves this. `TemplatedInferencerBase.switch_role`
virtualizes template role fields, but base `InferencerBase.switch_role` still
mutates workspace and deliverable flags by design. MFDual's active-context path
must avoid those writes for selected shared runtime roles.

### Commit 6: Add BTA `worker_inferencers`

Add a BTA attr:

```python
worker_inferencers: Any = attrib(default=None, kw_only=True, metadata={"lazy_config_factory": True})
```

Update RichPythonUtils config instantiation so attrs fields with
`metadata={"lazy_config_factory": True}` receive the same raw-config capture and
`LazyConfigFactory` replacement currently applied to `*_factory` fields.

This is intentionally metadata-based instead of naming the field
`worker_inferencers_factory`; the public YAML key is the user-facing goal, while
the config-loader metadata preserves the existing lazy-instantiation semantics.

Update BTA worker resolution:

- validate at construction or graph-build time that only one worker mechanism is
  configured;
- if `worker_inferencers` is set, call the lazy definition per expanded subtask
  to create a fresh worker;
- call the lazy definition with no arguments; `LazyConfigFactory` has a no-arg
  contract. The subtask text/index are still passed to the resulting worker via
  the existing `worker.infer/ainfer(query_str, run_context=...)` path, not into
  the factory itself;
- keep `workers` and `worker_factory` behavior unchanged.

Do not add a `workers: {_target_: ...}` single-mapping mode in this plan.
`workers` remains the static prebuilt list path.

### Commit 7: Migrate `breakdown-multiflow-plan.yaml`

After Commit 6 is tested, migrate:

```yaml
worker_factory:
  _factory_: MultiFlowDual
  ...
```

to:

```yaml
worker_inferencers:
  _target_: MultiFlowDual
  ...
```

The behavior must remain: one fresh `MultiFlowDual` worker per BTA subtask.

### Commit 8: Update documentation and plan ledger

Update the relevant RunContext/AF1 notes and MFDual module comments so they no
longer describe MFI/MFDual mutation as a deliberate unresolved carve-out.

Do not update docs before tests prove the carve-outs are actually removed.

## Tests

### MFI state tests

- Dispatch tags populate `MultiFlowState`.
- Reviewer/fixer aliases are isolated per run.
- Ranking and winner index are isolated per run.
- `_latest_per_flow` and judgments are isolated per attempt.
- Dynamic worker closures write attempt state to the captured parent MFI
  context, not to the worker child context.
- Upstream artifacts passed to follow-up and aggregator prompts do not mutate
  child `template_extra_feed` and do not bleed across sibling flows.
- Update existing tests that assert `_all_judgments` on `self`; after this plan,
  those assertions should inspect attempt state under an explicit context.
- `propagate_runtime_input=True` still feeds BTA through typed
  `effective_sub_queries`.
- Two concurrent runs on one MFI instance with sibling contexts do not bleed
  state.
- Legacy direct call followed by `get_winner_flow_idx()` remains covered.

### MFDual role-resolution tests

- `reviewer_match_second` matches old selection behavior.
- `reviewer_match_all_non_winners` populates a runtime panel without mutating
  `self.reviewers`.
- `fixer_match_winner` resolves winner as fixer without mutating
  `self.fixer_inferencer`.
- Alias dispatch through `inferencer_pool` still works.
- Warning/fallback behavior remains equivalent when winner or aliases are
  missing.
- Audit metadata reads from context state, not raw `_last_*` instance fields.
- Review/fix/panel accessors bind local role objects per round so a later state
  change cannot affect an in-flight round.
- LWI workflow call state remains a dictionary while MFDual state lives under
  the reserved MFDual key. Existing `_pending_state` behavior remains intact.
- Include real review/fix runs under explicit `RunContext`, not only direct
  `_select_reviewer_and_fixer()` unit tests.
- Update existing tests that assert role/workspace/output-flag mutation on
  selected reviewer/fixer instances; active-context expectations should inspect
  context-published workspace/role state instead.

### Purity tests

Update `test_m7_purity_gate.py`:

- remove carve-outs for MFI `_last_*`;
- remove carve-outs for MFDual reviewer/fixer reference mutation;
- add coverage for `_latest_per_flow`, `_all_judgments`, and panel roles.
- add the approved-accessor lint/gate for Dual role reads.
- add a purity check for child `template_extra_feed` so upstream artifacts are
  not persisted on shared follow-up or aggregator definitions.
- if inherited Dual/LWI runtime fields remain on `self`, the acceptance wording
  must not claim arbitrary shared-singleton MFDual safety.

### BTA worker tests

- `worker_inferencers: {_target_: MultiFlowDual, ...}` becomes a lazy factory.
- A fresh worker is created for each expanded subtask.
- Existing `worker_factory` configs still work.
- Existing `workers: [...]` configs still work.
- Mixing `worker_factory`, `workers`, and `worker_inferencers` fails clearly.
- `breakdown-multiflow-plan.yaml` old and new forms produce equivalent mocked
  results for fixed subtasks.
- RichPythonUtils tests prove metadata-marked lazy fields are wrapped in
  `LazyConfigFactory`, including single `_target_` blocks and invalid-key
  filtering behavior.
- Do not duplicate existing generic typed-state round-trip tests; extend them
  only for new fields/classes.

### End-to-end acceptance

Run a BTA with `worker_inferencers: {_target_: MultiFlowDual, ...}` and N > 1
subtasks under async worker execution. Assert:

- no winner/reviewer/fixer bleed across subtasks;
- no workspace bleed across subtasks;
- aggregator sees the same worker result semantics as the old `worker_factory`
  config;
- final deliverable behavior remains unchanged.

## Non-Goals

- Do not remove `worker_factory`.
- Do not change the meaning of `workers`.
- Do not implement shared single-instance BTA worker reuse in this plan.
- Do not store raw inferencer objects in serialized run state.
- Do not make bare direct calls concurrency-safe. Shared-definition concurrency
  requires explicit non-legacy run contexts.
- Do not claim arbitrary shared MFDual singleton reuse unless the inherited
  Dual/LWI runtime-field audit is green. The concrete YAML goal uses fresh lazy
  workers and is not blocked by that broader proof.

## If Only One Existing Plan Is Chosen

Choose this integrated Codex plan:

`/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plan/inferencer_architecture/codex/mfi_mfdual_full_definition_state_separation_plan.md`

Reason: it keeps Claude's source-aligned MFI/MFDual state strategy, includes
Revodev's required `worker_inferencers` user-facing goal, and rejects the
shared-instance shortcut until the codebase has enough proof to support it.
