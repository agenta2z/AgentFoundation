# WorkGraph Step-Level Lockstep Coordination for MultiFlow

## Summary

Build MultiFlow's coordinated mode as a WorkGraph-backed step lattice: each flow round becomes a first-class WorkGraph node, so `round N+1` cannot start until every scheduled `round N` node has produced a status. This fixes stale peer visibility, enables step-level checkpoint/retry/resume, and avoids the old hand-rolled `_ainfer_coordinated` loop that would bypass WorkGraph's existing persistence and reporting machinery.

Default behavior stays unchanged: MultiFlow remains independent unless lockstep coordination is explicitly enabled.

## Key Design

- Add `flow_coordination: "independent" | "lockstep" = "independent"` to MultiFlow.
- Keep `coordinated_stop` as a backward-compatible alias:
  - `False` means no behavior change.
  - `True` maps to `flow_coordination="lockstep"` and no longer raises `NotImplementedError`.
- In lockstep mode, MultiFlow does not wrap each whole flow in one `LinearWorkflowInferencer` worker. Instead, it builds deterministic WorkGraph nodes:
  - `flow_i_initial`
  - `flow_i_round01`
  - `flow_i_round02`
  - final aggregator node
- Edges form an all-to-all barrier between rounds:
  - every initial node feeds every `round01` node
  - every `roundN` node feeds every `roundN+1` node
  - aggregator depends on the final scheduled node for each flow
- Use `WorkGraphNode.add_next()` as the only edge construction mechanism. Do not manually append `previous`, because WorkGraph already maintains parent counts through `add_next()`.

## Runtime State and Step Status

- Add a serializable `FlowStepStatus` result shape with:
  - `flow_idx`
  - `round_idx`
  - `state: "ok" | "stopped" | "skipped" | "failed"`
  - `output_text`
  - `output_path`
  - `judgment`
  - `should_continue`
- Store per-attempt coordination state in `MultiFlowAttemptState`, not on the inferencer definition:
  - latest output text per flow
  - latest output path per flow
  - flow active/stopped flags
  - ordered step status ledger
- Each scheduled WorkGraph node must always return one `FlowStepStatus`. A stopped flow returns `skipped` or `stopped` in later rounds instead of disappearing, so downstream barriers never hang.
- Do not use `WorkGraphStopFlags.Terminate` for normal flow stopping. Stop is data-level state; fatal configuration or programming errors may still raise.

## Execution Behavior

- Round 0 nodes call each flow's `initial_inferencer`.
- Round N nodes call each flow's `followup_inferencer`.
- Followup prompt construction reuses the existing peer visibility logic, but now reads from the lockstep state ledger after the previous round barrier has completed. Therefore a fast flow's `round N+1` sees every peer's `round N` output or explicit stopped/skipped status.
- Aggregation reads declaration-ordered latest outputs from `MultiFlowAttemptState`, not the raw WorkGraph output tuple, because resumed/cached nodes may not appear in the final tuple.
- Preserve `propagate_runtime_input` by persisting effective per-flow inputs in `MultiFlowState.flow_inputs`; resume must reuse those persisted values rather than recomputing or mutating definitions.

## Checkpoint, Retry, and Resume

- Stable node names provide stable result paths, enabling WorkGraph node-level checkpoint/resume per flow step.
- Use WorkGraph persistence for step-level checkpointing.
- Avoid retry multiplication:
  - child inferencer retry remains the default retry owner for actual model calls;
  - WorkGraph node retry should not add an additional retry multiplier unless explicitly configured later.
- Resume behavior:
  - completed round nodes load from saved results;
  - missing or failed nodes rerun;
  - downstream nodes wait on loaded plus newly completed parents exactly like a fresh run.
- This design does require rounds to be WorkGraph nodes. There is no honest way to get true WorkGraph step-level checkpoint/retry/resume while keeping each whole flow as a single opaque LWI worker node.

## Integration Points

- Implement the lockstep graph inside MultiFlow only. Do not change generic BTA, LWI, or WorkGraph unless tests reveal a required compatibility fix.
- Preserve the existing independent path:
  - `_reset_cross_flow_state`
  - runtime input propagation
  - current BTA delegation
  - current whole-flow LWI workers
- The concrete target YAML use case is `task/configs/breakdown-multiflow-plan.yaml`, where BTA can use a `MultiFlowDual`/MultiFlow worker under `worker_inferencers`. Lockstep must work in that nested BTA worker position without mutating shared inferencer definitions.
- Support both async and sync inference paths using the same graph topology:
  - async node wrappers call `ainfer`
  - sync node wrappers call `infer`

## Tests

- Existing independent MultiFlow tests must pass unchanged.
- Add lockstep tests with fake slow/fast flows:
  - fast flow round 1 does not start until slow flow initial completes.
  - no peer slot renders as `(no output yet)` after the barrier.
  - peer output text and path are visible in deterministic flow order.
- Add checkpoint/resume tests:
  - complete some round nodes, interrupt, resume, and verify completed nodes load while missing nodes rerun.
  - aggregator uses saved/latest state rather than raw WorkGraph output tuple.
- Add stop/skip tests:
  - one flow stops early while another continues.
  - stopped flow emits skipped statuses for later scheduled rounds.
  - all stopped flows do not deadlock the graph.
- Add retry tests proving retries are not multiplied by both child inferencer retry and WorkGraph retry.
- Add nested integration coverage for the BTA worker case used by `breakdown-multiflow-plan.yaml`.

## Assumptions

- Lockstep coordination is opt-in because it trades latency for better peer consistency.
- `max_dynamic_steps` remains the round bound; lockstep pre-materializes up to the configured maximum and uses cheap skipped nodes for flows that stop early.
- The first implementation favors deterministic pre-materialized topology over dynamic graph expansion, because it is simpler, easier to resume, and less likely to break WorkGraph replay semantics.
- The plan fixes MultiFlow coordination and step-level state. It does not require changing the broader inferencer definition/state separation architecture beyond using the existing run state objects correctly.
