"""LinearWorkflowInferencer — declarative sequential step-chain with loop support.

Codifies the sequential-with-loop pattern used by PlanThenImplementInferencer
and DualInferencer (both Workflow-based) into a single configurable class.
Also capable of expressing ReflectiveInferencer's Sequential mode (which
currently uses manual loops without Workflow).

Each step is declared via a WorkflowStepConfig — specifying the child
inferencer (or raw callable), input transformation, output extraction,
state update, loop configuration, and checkpoint control.

Inherits from both InferencerBase and Workflow so the step chain can
leverage Workflow's checkpoint / loop-resume system.  When
``enable_result_save`` is set and a workspace is provided, the workflow
persists each step's result to disk and can resume from a crash.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.serializable import SerializationMode
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import (
    ResultPassDownMode,
)
from rich_python_utils.common_objects.workflow.common.expansion import ExpansionResult
from rich_python_utils.common_objects.workflow.common.step_wrapper import StepWrapper
from rich_python_utils.common_objects.workflow.workflow import Workflow
from rich_python_utils.io_utils.artifact import artifact_type


@attrs
class WorkflowStepConfig:
    """Declarative configuration for one step in a LinearWorkflowInferencer.

    Each step either delegates to a child ``inferencer`` or calls a raw
    ``step_fn`` callable.  Input/output transformations, state updates,
    and loop configuration are all optional.

    Attributes:
        name: Unique step name (used for loop_back_to resolution and logging).
        inferencer: Child InferencerBase to execute for this step.
        step_fn: Raw async/sync callable alternative to inferencer.
        input_builder: Callable(state) -> inference_input for the child.
        output_extractor: Callable(result) -> extracted value.
        output_state_key: Key in state dict to store the extracted output.
        state_updater: Callable(state, result) -> None for custom state mutation.
        loop_back_to: Step name to loop back to when loop_condition is True.
        loop_condition: Callable(state, result) -> bool.
        max_loop_iterations: Max loop iterations before exhaustion.
        on_loop_exhausted: Callable(state, result) called when loop limit hit.
        enable_result_save: Whether to checkpoint this step's result.
            None means fall back to the Workflow-level setting.
        config_key: Key in inference_config for step-specific config.
        pass_inference_args: Whether to forward **_inference_args to the child.
        enabled: Whether this step is active (False = no-op placeholder).
    """

    name: str = attrib()
    inferencer: Optional[InferencerBase] = attrib(default=None)
    step_fn: Optional[Callable] = attrib(default=None)
    input_builder: Optional[Callable] = attrib(default=None)
    output_extractor: Optional[Callable] = attrib(default=None)
    output_state_key: Optional[str] = attrib(default=None)
    state_updater: Optional[Callable] = attrib(default=None)
    loop_back_to: Optional[str] = attrib(default=None)
    loop_condition: Optional[Callable] = attrib(default=None)
    max_loop_iterations: int = attrib(default=5)
    on_loop_exhausted: Optional[Callable] = attrib(default=None)
    enable_result_save: Optional[bool] = attrib(default=None)
    config_key: Optional[str] = attrib(default=None)
    pass_inference_args: bool = attrib(default=False)
    enabled: bool = attrib(default=True)


class _DynamicStepRegistry(dict):
    """Dict-like registry that matches ``dynamic_step_N`` expansion IDs.

    Workflow's ``_reconstruct_expansions`` does ``expansion_id in registry``
    and ``registry[expansion_id]``.  This class intercepts both so that
    any expansion_id starting with ``"dynamic_step_"`` resolves to a
    factory that rebuilds the correct step wrapper via the captured LWI
    instance.
    """

    def __init__(self, lwi):
        super().__init__()
        self._lwi = lwi

    def __contains__(self, key):
        if isinstance(key, str) and key.startswith("dynamic_step_"):
            return True
        return super().__contains__(key)

    def __getitem__(self, key):
        if isinstance(key, str) and key.startswith("dynamic_step_"):
            lwi = self._lwi

            def _factory(exp_id):
                step_index = int(exp_id.split("_")[-1])
                return [
                    lwi._build_dynamic_step_wrapper(
                        lwi.default_followup_inferencer, step_index
                    )
                ]

            return _factory
        return super().__getitem__(key)


@artifact_type(Workflow, type="json", group="workflows")
@attrs(slots=False)
class LinearWorkflowInferencer(InferencerBase, Workflow):
    """Declarative sequential step-chain inferencer with loop support.

    Generalises the sequential-with-loop pattern shared by PTI and
    DualInferencer into a single configurable class.  Steps are declared
    via :class:`WorkflowStepConfig` objects; the class wires them into
    :class:`StepWrapper` instances and delegates execution to
    :meth:`Workflow._arun`.

    Inherits from both ``InferencerBase`` (for ``infer()``/``ainfer()`` API)
    and ``Workflow`` (for checkpoint/loop-resume infrastructure).

    Usage::

        lwi = LinearWorkflowInferencer(
            step_configs=[
                WorkflowStepConfig(name="plan", inferencer=planner),
                WorkflowStepConfig(name="implement", inferencer=executor,
                                   input_builder=lambda s: s["plan_output"]),
            ],
            response_builder=lambda state: state["implement_output"],
        )
        result = lwi("Design and implement a REST API")

    Attributes:
        step_configs: Ordered list of WorkflowStepConfig declarations.
        response_builder: Callable(state) -> final response.  Defaults to
            returning the full state dict.
        initial_state_factory: Callable(inference_input) -> initial state dict.
        workspace: Optional InferencerWorkspace for checkpoint/output I/O.
            (Inherited from InferencerBase.) The legacy ``workspace_root: str``
            shorthand was removed 2026-05-05; pass
            ``workspace=InferencerWorkspace(root="/path", ...)`` instead.
    """

    step_configs: List[WorkflowStepConfig] = attrib(factory=list)
    response_builder: Optional[Callable] = attrib(default=None)
    initial_state_factory: Optional[Callable] = attrib(default=None)

    # --- New: Iteration Management ---
    iteration_workspace_factory: Optional[Callable[[str, int], str]] = attrib(default=None)
    reset_sessions_per_iteration: bool = attrib(default=False)
    iteration_record_builder: Optional[Callable[[dict], dict]] = attrib(default=None)
    checkpoint_subdir: Optional[str] = attrib(default=None)

    # --- Dynamic Mode (Requirements 1-9) ---
    dynamic_mode: bool = attrib(default=False)
    default_initial_inferencer: Optional[InferencerBase] = attrib(default=None)
    default_followup_inferencer: Optional[InferencerBase] = attrib(default=None)
    end_condition: Optional[Callable[[dict, Any], bool]] = attrib(default=None)
    max_dynamic_steps: int = attrib(default=10)
    inferencer_factory: Optional[Callable] = attrib(default=None)
    dynamic_input_builder: Optional[Callable[[dict, Any], Any]] = attrib(default=None)

    _DERIVED_FROM_WORKSPACE = ()

    _workspace_propagation_skip: frozenset = frozenset((
        "default_initial_inferencer",
        "default_followup_inferencer",
    ))

    # --- Suppress Workflow constructor parameters (init=False) ---
    result_pass_down_mode = attrib(default=ResultPassDownMode.NoPassDown, init=False)
    unpack_single_result = attrib(default=False, init=False)
    ignore_stop_flag_from_saved_results = attrib(default=True, init=False)
    auto_mode = attrib(default=SerializationMode.PREFER_CLEAR_TEXT, init=False)
    checkpoint_mode = attrib(default="jsonfy", init=False)

    # Internal state (not user-facing)
    # M7/§2.9 + Part G (C10): the LinearWorkflow/Workflow RUNNER state machine is
    # virtualized so a SHARED instance run under N concurrent RunContexts keeps
    # isolated runner state.  Every per-run mutable runner attribute below is a
    # compat-property routing to a per-run home when a RunContext is active —
    # run-state lives in the context, not on ``self`` — else an instance backing
    # (byte-identical without a context).
    #
    # Two homes, matching the engine's existing checkpoint classification:
    #   * **Serialized / resume-restored** (``_state`` working dict, ``_loop_counts``,
    #     ``_exec_seq``, ``_splice_*``, ``_step_attempt_counts``) -> the typed
    #     ``LinearWorkflowState`` carrier (per-CALL; survives ``reset_attempt``).
    #     For MFDual the carrier is ``ctx.node().call.runner`` (composed into
    #     ``MFDualState``); for a plain LWI the working dict ``_state``/``_pending_state``
    #     IS ``ctx.node().call`` and the other serialized fields live in a per-path
    #     ``LinearWorkflowState`` lazily held in ``ctx.node().scratch`` (so the dict
    #     ``.call`` and the runner bookkeeping never collide — GT#14).
    #   * **Transient / re-derived each run** (``_steps``, the live expansion flags,
    #     ``_state_picklability_verified``, per-run attempt flags, ``_inference_config``/
    #     ``_inference_args``) -> the non-serialized per-path ``ctx.node().scratch`` dict
    #     (concurrency-isolated; never resumed, matching the re-derived contract).
    #
    # ``_pending_state`` is the canonical template (G1, already shipped) and is kept
    # byte-for-byte; ``_state`` mirrors it exactly so the two stay coherent.

    # ------------------------------------------------------------------
    # Part G runner-state resolution helpers (mirror ``_pending_state``)
    # ------------------------------------------------------------------

    # Reserved key under which a plain-LWI per-path ``LinearWorkflowState`` carrier
    # is held in ``ctx.node().scratch`` (the serialized runner fields' home when
    # ``.call`` is a raw working-state dict rather than a typed ``.call`` state).
    _RUNNER_SCRATCH_KEY = "_lwi_runner_carrier"

    def _runner_carrier(self, create: bool = True):
        """Resolve the per-run ``LinearWorkflowState`` carrier for the serialized
        runner fields (``loop_counts``/``exec_seq``/``splice_*``/``step_attempt_counts``).

        Resolution mirrors ``_pending_state`` exactly:

        * a RunContext is active and ``ctx.node().call`` is a TYPED state with a
          ``runner`` carrier (e.g. ``MFDualState.runner``) -> that carrier (the
          fields ride along with the serialized ``.call`` state);
        * a RunContext is active and ``.call`` is a plain dict / ``None`` (a plain
          LWI, whose working dict lives at ``.call``) -> a per-path
          ``LinearWorkflowState`` lazily created in ``ctx.node().scratch`` (so it is
          isolated per concurrent run yet survives ``reset_attempt``);
        * no active context (legacy / direct call) -> ``None`` (caller uses the
          instance backing — byte-identical to the pre-virtualization behaviour).
        """
        from agent_foundation.common.inferencers.run_context import (
            InferencerStateBase,
            active_run_context,
        )
        from agent_foundation.common.inferencers.run_context.state import (
            LinearWorkflowState,
        )

        ctx = active_run_context()
        if ctx is None:
            return None
        node = ctx.node()
        call = node.call
        if isinstance(call, InferencerStateBase):
            runner = getattr(call, "runner", None)
            if runner is not None:
                return runner
            # Typed ``.call`` without a runner carrier — fall back to the per-path
            # scratch carrier rather than mutating the typed ``.call``.
        carrier = node.scratch.get(self._RUNNER_SCRATCH_KEY)
        if carrier is None and create:
            carrier = LinearWorkflowState()
            node.scratch[self._RUNNER_SCRATCH_KEY] = carrier
        return carrier

    def _runner_scratch(self):
        """Return the per-path transient scratch dict for re-derived runner fields,
        or ``None`` when no RunContext is active (caller uses the instance backing).
        """
        from agent_foundation.common.inferencers.run_context import (
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is None:
            return None
        return ctx.node().scratch

    @property
    def _pending_state(self):
        from agent_foundation.common.inferencers.run_context import (
            InferencerStateBase,
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is not None:
            call = ctx.node().call
            # GT#14/G1: when ``.call`` is a TYPED state (e.g. MFDualState — the dispatch/role
            # state lives there), the workflow working-state dict lives in its ``runner``
            # carrier so the two don't collide. Plain dict/None → the legacy carrier IS .call.
            if isinstance(call, InferencerStateBase):
                runner = getattr(call, "runner", None)
                if runner is not None:
                    return runner.state
            elif isinstance(call, dict) or call is None:
                return call
        return self.__dict__.get("_pending_state_backing")

    @_pending_state.setter
    def _pending_state(self, value):
        from agent_foundation.common.inferencers.run_context import (
            InferencerStateBase,
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is not None:
            call = ctx.node().call
            if isinstance(call, InferencerStateBase):
                runner = getattr(call, "runner", None)
                if runner is not None:
                    runner.state = value
                    return
                # typed state without a runner carrier — fall through to the backing
                # rather than clobbering the typed .call with a raw dict.
            else:
                ctx.node().call = value
                return
        self.__dict__["_pending_state_backing"] = value

    # ------------------------------------------------------------------
    # Part G: serialized runner fields -> the ``LinearWorkflowState`` carrier
    # (``_state`` mirrors ``_pending_state`` exactly; the rest ride the carrier)
    # ------------------------------------------------------------------

    @property
    def _state(self):
        # The workflow working-state dict.  Resolves to the SAME home as
        # ``_pending_state`` (typed ``call.runner.state`` | dict/None ``.call`` |
        # the shared ``_pending_state_backing`` instance backing) so the two stay
        # coherent: the Dual review step reads ``self._state`` then falls back to
        # ``dict(self._pending_state)`` (dual_inferencer.py:1089-1092) and the
        # rest of the engine reads ``self._state`` while ``_ainfer`` seeds via
        # ``self._pending_state`` — they must be one object.
        from agent_foundation.common.inferencers.run_context import (
            InferencerStateBase,
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is not None:
            call = ctx.node().call
            if isinstance(call, InferencerStateBase):
                runner = getattr(call, "runner", None)
                if runner is not None:
                    return runner.state
            elif isinstance(call, dict) or call is None:
                return call
        return self.__dict__.get("_pending_state_backing")

    @_state.setter
    def _state(self, value):
        from agent_foundation.common.inferencers.run_context import (
            InferencerStateBase,
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is not None:
            call = ctx.node().call
            if isinstance(call, InferencerStateBase):
                runner = getattr(call, "runner", None)
                if runner is not None:
                    runner.state = value
                    return
            else:
                ctx.node().call = value
                return
        self.__dict__["_pending_state_backing"] = value

    @property
    def _loop_counts(self):
        carrier = self._runner_carrier()
        if carrier is not None:
            return carrier.loop_counts
        lc = self.__dict__.get("_loop_counts_backing")
        if lc is None:
            lc = {}
            self.__dict__["_loop_counts_backing"] = lc
        return lc

    @_loop_counts.setter
    def _loop_counts(self, value):
        carrier = self._runner_carrier()
        if carrier is not None:
            carrier.loop_counts = value
            return
        self.__dict__["_loop_counts_backing"] = value

    @property
    def _exec_seq(self):
        carrier = self._runner_carrier()
        if carrier is not None:
            return carrier.exec_seq
        return self.__dict__.get("_exec_seq_backing", 0)

    @_exec_seq.setter
    def _exec_seq(self, value):
        carrier = self._runner_carrier()
        if carrier is not None:
            carrier.exec_seq = value
            return
        self.__dict__["_exec_seq_backing"] = value

    @property
    def _step_attempt_counts(self):
        # Accumulator (persists across resume) -> the serialized carrier, never the
        # transient scratch.  Always returns a dict so ``hasattr`` stays True (the
        # base only inits it ``if not hasattr`` — see workflow.py:1174,1513).
        carrier = self._runner_carrier()
        if carrier is not None:
            return carrier.step_attempt_counts
        sac = self.__dict__.get("_step_attempt_counts_backing")
        if sac is None:
            sac = {}
            self.__dict__["_step_attempt_counts_backing"] = sac
        return sac

    @_step_attempt_counts.setter
    def _step_attempt_counts(self, value):
        carrier = self._runner_carrier()
        if carrier is not None:
            carrier.step_attempt_counts = value
            return
        self.__dict__["_step_attempt_counts_backing"] = value

    @property
    def _splice_orig_args(self):
        carrier = self._runner_carrier(create=False)
        if carrier is not None:
            return carrier.splice_orig_args
        return self.__dict__.get("_splice_orig_args_backing")

    @_splice_orig_args.setter
    def _splice_orig_args(self, value):
        carrier = self._runner_carrier()
        if carrier is not None:
            carrier.splice_orig_args = value
            return
        self.__dict__["_splice_orig_args_backing"] = value

    @_splice_orig_args.deleter
    def _splice_orig_args(self):
        # The base engine does ``del self._splice_orig_args`` after the first
        # spliced step executes; clearing to ``None`` keeps the subsequent
        # ``getattr(self, '_splice_orig_args', None)`` reads returning ``None``.
        carrier = self._runner_carrier(create=False)
        if carrier is not None:
            carrier.splice_orig_args = None
            return
        self.__dict__.pop("_splice_orig_args_backing", None)

    @property
    def _splice_orig_kwargs(self):
        carrier = self._runner_carrier(create=False)
        if carrier is not None:
            return carrier.splice_orig_kwargs
        return self.__dict__.get("_splice_orig_kwargs_backing")

    @_splice_orig_kwargs.setter
    def _splice_orig_kwargs(self, value):
        carrier = self._runner_carrier()
        if carrier is not None:
            carrier.splice_orig_kwargs = value
            return
        self.__dict__["_splice_orig_kwargs_backing"] = value

    @_splice_orig_kwargs.deleter
    def _splice_orig_kwargs(self):
        carrier = self._runner_carrier(create=False)
        if carrier is not None:
            carrier.splice_orig_kwargs = None
            return
        self.__dict__.pop("_splice_orig_kwargs_backing", None)

    @property
    def _splice_step_index(self):
        carrier = self._runner_carrier(create=False)
        if carrier is not None:
            return carrier.splice_step_index
        return self.__dict__.get("_splice_step_index_backing")

    @_splice_step_index.setter
    def _splice_step_index(self, value):
        carrier = self._runner_carrier()
        if carrier is not None:
            carrier.splice_step_index = value
            return
        self.__dict__["_splice_step_index_backing"] = value

    @_splice_step_index.deleter
    def _splice_step_index(self):
        carrier = self._runner_carrier(create=False)
        if carrier is not None:
            carrier.splice_step_index = None
            return
        self.__dict__.pop("_splice_step_index_backing", None)

    # ------------------------------------------------------------------
    # Part G: transient runner fields -> the per-path ``NodeRunState.scratch``
    # (re-derived each run; never serialized / resumed)
    # ------------------------------------------------------------------

    @property
    def _steps(self):
        scratch = self._runner_scratch()
        if scratch is not None:
            return scratch.get("_steps")
        return self.__dict__.get("_steps_backing")

    @_steps.setter
    def _steps(self, value):
        scratch = self._runner_scratch()
        if scratch is not None:
            scratch["_steps"] = value
            return
        self.__dict__["_steps_backing"] = value

    @property
    def _inference_config(self):
        scratch = self._runner_scratch()
        if scratch is not None:
            return scratch.get("_inference_config")
        return self.__dict__.get("_inference_config_backing")

    @_inference_config.setter
    def _inference_config(self, value):
        scratch = self._runner_scratch()
        if scratch is not None:
            scratch["_inference_config"] = value
            return
        self.__dict__["_inference_config_backing"] = value

    @property
    def _inference_args(self):
        scratch = self._runner_scratch()
        if scratch is not None:
            return scratch.get("_inference_args")
        return self.__dict__.get("_inference_args_backing")

    @_inference_args.setter
    def _inference_args(self, value):
        scratch = self._runner_scratch()
        if scratch is not None:
            scratch["_inference_args"] = value
            return
        self.__dict__["_inference_args_backing"] = value

    @property
    def _expansion_count(self):
        scratch = self._runner_scratch()
        if scratch is not None:
            return scratch.get("_expansion_count", 0)
        return self.__dict__.get("_expansion_count_backing", 0)

    @_expansion_count.setter
    def _expansion_count(self, value):
        scratch = self._runner_scratch()
        if scratch is not None:
            scratch["_expansion_count"] = value
            return
        self.__dict__["_expansion_count_backing"] = value

    @property
    def _expansion_records(self):
        scratch = self._runner_scratch()
        if scratch is not None:
            recs = scratch.get("_expansion_records")
            if recs is None:
                recs = []
                scratch["_expansion_records"] = recs
            return recs
        recs = self.__dict__.get("_expansion_records_backing")
        if recs is None:
            recs = []
            self.__dict__["_expansion_records_backing"] = recs
        return recs

    @_expansion_records.setter
    def _expansion_records(self, value):
        scratch = self._runner_scratch()
        if scratch is not None:
            scratch["_expansion_records"] = value
            return
        self.__dict__["_expansion_records_backing"] = value

    @property
    def _expansion_active(self):
        scratch = self._runner_scratch()
        if scratch is not None:
            return scratch.get("_expansion_active", False)
        return self.__dict__.get("_expansion_active_backing", False)

    @_expansion_active.setter
    def _expansion_active(self, value):
        scratch = self._runner_scratch()
        if scratch is not None:
            scratch["_expansion_active"] = value
            return
        self.__dict__["_expansion_active_backing"] = value

    @property
    def _migration_needed(self):
        scratch = self._runner_scratch()
        if scratch is not None:
            return scratch.get("_migration_needed", False)
        return self.__dict__.get("_migration_needed_backing", False)

    @_migration_needed.setter
    def _migration_needed(self, value):
        scratch = self._runner_scratch()
        if scratch is not None:
            scratch["_migration_needed"] = value
            return
        self.__dict__["_migration_needed_backing"] = value

    @property
    def _state_picklability_verified(self):
        scratch = self._runner_scratch()
        if scratch is not None:
            return scratch.get("_state_picklability_verified", False)
        return self.__dict__.get("_state_picklability_verified_backing", False)

    @_state_picklability_verified.setter
    def _state_picklability_verified(self, value):
        scratch = self._runner_scratch()
        if scratch is not None:
            scratch["_state_picklability_verified"] = value
            return
        self.__dict__["_state_picklability_verified_backing"] = value

    # NOTE: ``_step_was_previously_attempted`` / ``_previous_attempt_info`` are the
    # resume-marker signals the runner sets in ``_arun`` and that **child step
    # closures read across a child-ctx boundary** (e.g. PTI's
    # ``_build_executor_input`` reads ``self._step_was_previously_attempted`` while
    # the executor's own ``ainfer`` has a child ctx active —
    # ``plan_then_implement_inferencer.py:571``).  Routing them through the *active*
    # node's scratch would write them on the LWI node and read them on the child
    # node -> the signal would be invisible to the consumer.  They are therefore
    # kept on the **instance backing** (byte-identical to pre-virtualization, and
    # readable from any ctx within the run).  Per-run isolation for a shared
    # instance under concurrency would require a captured-run node (cf. Part B's
    # parent-node write) and is out of scope for this commit (the concurrency
    # acceptance covers ``state``/``loop_counts``/``exec_seq``).
    @property
    def _step_was_previously_attempted(self):
        return self.__dict__.get("_step_was_previously_attempted_backing", False)

    @_step_was_previously_attempted.setter
    def _step_was_previously_attempted(self, value):
        self.__dict__["_step_was_previously_attempted_backing"] = value

    @property
    def _previous_attempt_info(self):
        return self.__dict__.get("_previous_attempt_info_backing")

    @_previous_attempt_info.setter
    def _previous_attempt_info(self, value):
        self.__dict__["_previous_attempt_info_backing"] = value

    def __attrs_post_init__(self):
        super(LinearWorkflowInferencer, self).__attrs_post_init__()
        # Workspace is fully managed by InferencerBase.__attrs_post_init__
        # (syncs `self.workspace` → `self._workspace`). No additional setup
        # needed here. The legacy `workspace_root: Optional[str]` shorthand
        # was removed 2026-05-05 — pass `workspace=InferencerWorkspace(...)`
        # for the explicit form.

        # Set parent debuggable for child inferencers
        seen_ids: set = set()
        for sc in self.step_configs:
            inf = sc.inferencer
            if (
                inf is not None
                and isinstance(inf, Debuggable)
                and id(inf) not in seen_ids
            ):
                seen_ids.add(id(inf))
                inf.set_parent_debuggable(self)

        # Dynamic mode: register expansion_step_registry for resume support
        if self.dynamic_mode:
            # Use a prefix-matching dict so that expansion_ids like
            # "dynamic_step_1", "dynamic_step_2", etc. all match the
            # single "dynamic_step" prefix and route to the same factory.
            self.expansion_step_registry = _DynamicStepRegistry(self)

    # ------------------------------------------------------------------
    # Workspace propagation override (hierarchical layout)
    # ------------------------------------------------------------------

    def _propagate_workspace_to_children(self, parent_workspace):
        """LWI override: assign semantic child names in dynamic mode.

        In dynamic mode the initial inferencer gets ``children/initial/``
        and the followup inferencer gets ``children/round01/``.  Both are
        listed in ``_workspace_propagation_skip`` so the base walker skips
        them; this override handles them with semantic names instead of
        the generic attr-slot names the base walker would use.
        """
        if getattr(self, "dynamic_mode", False):
            from agent_foundation.common.inferencers.inferencer_base import (
                InferencerBase,
            )
            for inf, child_name in (
                (getattr(self, "default_initial_inferencer", None), self._dynamic_child_name(0)),
                (getattr(self, "default_followup_inferencer", None), self._dynamic_child_name(1)),
            ):
                if inf is None or not isinstance(inf, InferencerBase):
                    continue
                if getattr(inf, "_workspace", None) is not None:
                    continue
                child_ws = parent_workspace.child(child_name)
                child_ws.ensure_dirs()
                inf._workspace = child_ws

        super()._propagate_workspace_to_children(parent_workspace)

        # Fail-loud: detect when dynamic-mode children didn't get a
        # workspace under THIS LWI's tree (e.g., instance aliased at
        # a higher level stole the workspace slot).
        if getattr(self, "dynamic_mode", False):
            from agent_foundation.common.inferencers.inferencer_base import (
                InferencerBase,
            )
            for inf, child_name in (
                (getattr(self, "default_initial_inferencer", None), self._dynamic_child_name(0)),
                (getattr(self, "default_followup_inferencer", None), self._dynamic_child_name(1)),
            ):
                if inf is None or not isinstance(inf, InferencerBase):
                    continue
                child_ws = getattr(inf, "_workspace", None)
                if child_ws is None:
                    raise RuntimeError(
                        f"LWI dynamic mode: {child_name} inferencer has no "
                        f"workspace after propagation. The instance may be "
                        f"aliased at a higher level (e.g., reviewer_match_second "
                        f"claimed it before flow-level propagation). Ensure "
                        f"_workspace_propagation_skip includes review/fixer slots "
                        f"when dynamic dispatch is enabled."
                    )
                if not child_ws.root.startswith(parent_workspace.root):
                    _logger.warning(
                        "LWI dynamic mode: %s inferencer workspace %r is outside "
                        "this LWI's tree %r — output may land in the wrong "
                        "directory. Likely cause: instance shared between flow "
                        "configs and a higher-level role.",
                        child_name, child_ws.root, parent_workspace.root,
                    )

    # ------------------------------------------------------------------
    # Output finalization (orchestrator override)
    # ------------------------------------------------------------------

    def _finalize_output(self, response):
        """LWI override: symlink last dynamic step's output as own.

        In dynamic mode, the last child (initial or roundNN) wrote the
        full artifact.  Symlinking it to the LWI's own ``outputs/``
        makes it available to the parent aggregator via
        ``resolve_canonical_output_path``.  The base ``_finalize_output``
        would only write the ``<Response>`` summary — the symlink
        preserves the full artifact.
        """
        if getattr(self, "dynamic_mode", False) and self._workspace is not None:
            results = (self._state or {}).get("dynamic_step_results", [])
            if results:
                step_count = len(results)
                consensus_iter = (self._state or {}).get("consensus_iteration_id", 0)
                child_name = self._dynamic_child_name(step_count - 1, consensus_iter)
                child_ws = self._workspace.child(child_name)
                from agent_foundation.common.inferencers.inferencer_workspace import DEFAULT_OUTPUT_FILENAME
                _output_name = self.output_path or DEFAULT_OUTPUT_FILENAME
                _child_out = child_ws.output_path(_output_name) if hasattr(child_ws, "output_path") else None
                _child_out_exists = os.path.isfile(_child_out) if _child_out else False
                self._symlink_child_output(child_ws)
                _own_out = self._workspace.output_path(_output_name) if hasattr(self._workspace, "output_path") else None
                self.log_info(
                    {
                        "child_name": child_name,
                        "child_output": _child_out,
                        "child_exists_before_symlink": _child_out_exists,
                        "own_ws": self._workspace.root,
                        "own_output_path": self.output_path,
                        "own_output": _own_out,
                        "symlink_exists": os.path.exists(_own_out) if _own_out else False,
                        "symlink_islink": os.path.islink(_own_out) if _own_out else False,
                    },
                    log_type="LWISymlink",
                )
                # Emit manifest adjacent to the (symlinked) output
                resolved = self.resolve_output_path()
                if resolved and os.path.isfile(resolved):
                    self._emit_output_manifest(resolved)
                return response
        # Non-dynamic mode: fall through to base leaf behavior
        return super()._finalize_output(response)

    # ------------------------------------------------------------------
    # Iteration workspace helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_iteration_workspace(base, n, factory=None):
        """Return the workspace path for iteration *n*.

        If *factory* is provided, delegate to ``factory(base, n)``.
        Otherwise iteration 1 uses *base* directly and iteration N>1
        uses ``{base}/iteration_{N}/``.
        """
        if factory is not None:
            return factory(base, n)
        if n == 1:
            return base
        return os.path.join(base, f"iteration_{n}")

    def _setup_iteration(self, state):
        """Set up workspace, children, and sessions for a new iteration.

        Called when ``state["iteration"]`` differs from
        ``state["_prev_iteration"]``, indicating a loop-back has
        incremented the iteration counter.

        Steps:
        1. Create a new :class:`InferencerWorkspace` for the iteration.
        2. Re-point child Workflow ``_result_root_override`` via
           :meth:`_setup_child_workflows`.
        3. Optionally reset child inferencer sessions.
        4. Record the completed iteration via :meth:`_record_iteration`.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        iteration = state.get("iteration", 1)

        # 1. Create iteration workspace (skip if no workspace configured)
        if self._workspace is not None:
            iter_path = self._get_iteration_workspace(
                self._workspace.root, iteration, self.iteration_workspace_factory
            )
            ws = InferencerWorkspace(root=iter_path)
            ws.ensure_dirs()
            self._workspace = ws

        # 2. Update child Workflow _result_root_override
        self._setup_child_workflows(state)

        # 3. Optionally reset child inferencer sessions
        # §9.3/N-R2: bind each step's slot so reset_session targets the step's OWN
        # Tier-3 handle (no-op binding without a context -> byte-identical legacy).
        if self.reset_sessions_per_iteration:
            seen_ids: set = set()
            for sc in self.step_configs:
                inf = sc.inferencer
                if inf is not None and id(inf) not in seen_ids:
                    seen_ids.add(id(inf))
                    with self._with_child_ctx(getattr(sc, "name", None) or "step"):
                        if hasattr(inf, "reset_session"):
                            inf.reset_session()
                        elif hasattr(inf, "new_session"):
                            inf.new_session()

        # 4. Record the completed iteration
        self._record_iteration(state)

    def _setup_child_workflows(self, state, *args, **kwargs):
        """Update child Workflow ``_result_root_override`` to the current
        iteration's checkpoint directory.

        Discovers child workflows via :meth:`_find_child_workflows_in`
        (inherited from :class:`Workflow`) and sets each child's
        ``_result_root_override`` to the current workspace's checkpoints
        directory.

        Falls back to the Workflow base class implementation when no
        ``_workspace`` is available (preserves default behavior).
        """
        ws = getattr(self, "_workspace", None)
        if ws is None:
            # Delegate to base Workflow implementation for non-workspace cases
            super()._setup_child_workflows(state, *args, **kwargs)
            return

        children = self._find_child_workflows_in(self)
        for _attr_name, (child, _entry) in children.items():
            child._result_root_override = ws.checkpoints_dir

    def _record_iteration(self, state):
        """Append a snapshot of the current iteration to state["iteration_records"].

        Uses ``iteration_record_builder`` if provided, otherwise snapshots
        all state keys that do NOT start with an underscore.
        """
        if self.iteration_record_builder is not None:
            record = self.iteration_record_builder(state)
        else:
            # Default: snapshot all non-underscore keys
            record = {k: v for k, v in state.items() if not k.startswith("_")}

        records = state.get("iteration_records")
        if records is None:
            state["iteration_records"] = []
            records = state["iteration_records"]
        records.append(record)

    def _write_step_marker(self, step_name):
        """Write a completion marker for the given step.

        Delegates to :meth:`InferencerWorkspace.write_marker` which
        creates ``artifacts/.<step_name>_completed`` with a timestamp.

        Silently returns when no workspace is configured.
        """
        ws = self._workspace
        if ws is None:
            return
        ws.write_marker(step_name)

    # ------------------------------------------------------------------
    # Final result caching
    # ------------------------------------------------------------------

    def _save_final_result(self, state):
        """Save the raw state dict to ``final_result.json`` in the workspace.

        Uses :func:`dict__` for serialization and :func:`write_json` for
        file I/O (same pattern as :meth:`_save_result`).

        Silently returns when no workspace is configured.
        """
        if self._workspace is None:
            return
        from rich_python_utils.common_utils.map_helper import dict__
        from rich_python_utils.io_utils.json_io import write_json

        path = self._workspace.checkpoint_path("final_result.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_json(dict__(state, recursive=True), path, indent=2)

    def _auto_enable_checkpointing(self):
        """Enable checkpoint/resume when workspace is available and no override is set.

        In dynamic mode, ``resume_with_saved_results`` is left disabled because
        Workflow's backward resume scan iterates ``self._steps[i]`` in
        ``range(resume_with_saved_results_int, -1, -1)``, which IndexErrors
        when ``len(self._steps) == 1`` (the initial dynamic-mode step list).
        Dynamic mode has its own resume path via ``expansion_step_registry``.
        """
        if self._result_root_override is None and self._workspace is not None:
            from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
                StepResultSaveOptions,
            )
            self.enable_result_save = StepResultSaveOptions.Always
            self.resume_with_saved_results = not self.dynamic_mode

    def _load_final_result(self):
        """Load cached final result from ``final_result.json``.

        If the file exists, loads the state dict and re-runs
        :attr:`response_builder` (if set) to reconstruct the typed
        response.  Returns ``None`` when no workspace is configured,
        the file does not exist, or deserialization fails.
        """
        if self._workspace is None:
            return None
        from rich_python_utils.io_utils.json_io import read_json

        path = self._workspace.checkpoint_path("final_result.json")
        if not os.path.exists(path):
            return None
        try:
            loaded_state = read_json(path)
            if self.response_builder is not None:
                return self.response_builder(loaded_state)
            return loaded_state
        except Exception:
            self.log_info(
                f"Warning: failed to deserialize final_result.json at {path}, "
                "proceeding with normal execution."
            )
            return None

    # ------------------------------------------------------------------
    # Child inferencer lifecycle management
    # ------------------------------------------------------------------

    async def aconnect(self, **kwargs):
        """Connect unique child inferencers (deduplicated by identity)."""
        seen_ids = set()
        for sc in self.step_configs:
            inf = sc.inferencer
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                if hasattr(inf, "aconnect"):
                    await inf.aconnect(**kwargs)

    async def adisconnect(self):
        """Disconnect unique child inferencers (deduplicated by identity)."""
        seen_ids = set()
        for sc in self.step_configs:
            inf = sc.inferencer
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                if hasattr(inf, "adisconnect"):
                    await inf.adisconnect()

    async def __aenter__(self):
        await self.aconnect()
        return self

    async def __aexit__(self, *exc_info):
        await self.adisconnect()

    # ------------------------------------------------------------------
    # Block WorkNodeBase.run() / arun() — callers must use infer()/ainfer()
    # ------------------------------------------------------------------

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "Use infer() or ainfer() — LinearWorkflowInferencer.run() is "
            "disabled because Workflow._arun() requires state setup that "
            "only _ainfer() provides."
        )

    async def arun(self, *args, **kwargs):
        raise NotImplementedError(
            "Use infer() or ainfer() — LinearWorkflowInferencer.arun() is "
            "disabled because Workflow._arun() requires state setup that "
            "only _ainfer() provides."
        )

    # ------------------------------------------------------------------
    # Dynamic mode helpers
    # ------------------------------------------------------------------

    def _resolve_next_inferencer(self, raw_result):
        """Unpack a dynamic step result into (actual_result, next_inferencer).

        If *raw_result* is a 2-tuple ``(result, next_inferencer)``, returns
        both.  Otherwise returns ``(raw_result, self.default_followup_inferencer)``.

        Raises :class:`ValueError` when no next inferencer can be determined
        (non-tuple result AND ``default_followup_inferencer`` is None).
        """
        if isinstance(raw_result, tuple) and len(raw_result) == 2:
            actual_result, next_inf = raw_result
            # Fall back to default if tuple explicitly contains None
            if next_inf is None:
                next_inf = self.default_followup_inferencer
            if next_inf is None:
                raise ValueError(
                    "Dynamic step returned (result, None) but "
                    "default_followup_inferencer is also None"
                )
            return actual_result, next_inf

        if self.default_followup_inferencer is None:
            raise ValueError(
                "Dynamic step did not specify next inferencer and "
                "default_followup_inferencer is None"
            )
        return raw_result, self.default_followup_inferencer

    def _instantiate_inferencer(self, inferencer):
        """Return a ready-to-use inferencer instance.

        If *inferencer* is already an instance, return it directly.
        If it is a class (type), create a new instance via
        ``inferencer_factory`` (if provided) or by calling the class
        with no arguments.
        """
        if not isinstance(inferencer, type):
            return inferencer
        if self.inferencer_factory is not None:
            return self.inferencer_factory(inferencer)
        return inferencer()

    @staticmethod
    def _dynamic_child_name(step_index, consensus_iter=0):
        """Single source of truth for a dynamic step's on-disk child dir name.

        A ``@staticmethod`` (no instance state) so peer orchestrators (e.g. the MFI's
        per-flow path capture) can reuse the SAME naming convention without an LWI instance —
        ``self._dynamic_child_name(...)`` calls from within the LWI keep working unchanged.

        ``step 0 -> "initial"``, ``step N>=1 -> "round{N:02d}"``, with an
        ``"_iter{K}"`` suffix for consensus iteration ``K>0``. Used by the
        dispatch (the explicit workspace handed to ``_rc_child``), workspace
        propagation, and ``_finalize_output`` so the three can never drift — the
        regression was the dispatch landing output at the ctx slot ``step_{N}``
        while ``_finalize_output`` looked for ``round{NN}``.
        """
        base = "initial" if step_index == 0 else f"round{step_index:02d}"
        return base + (f"_iter{consensus_iter}" if consensus_iter and consensus_iter > 0 else "")

    def _build_dynamic_step_wrapper(self, inferencer, step_index):
        """Build a step wrapper closure for dynamic mode.

        The returned async callable has signature ``(step_input, state)``
        matching the ``step_fn`` convention used by
        :class:`WorkflowStepConfig`.  It captures *self* via closure so
        it can access dynamic-mode configuration and shared state.

        Behaviour:
        1. Build input from state (``original_input`` for step 0,
           previous result for step N) or via ``dynamic_input_builder``.
        2. Call ``inferencer.ainfer(input)``.
        3. Unpack result tuple ``(result, next_inferencer)`` if
           applicable; use ``default_followup_inferencer`` otherwise.
        4. Update state: append to ``dynamic_step_results``, increment
           ``dynamic_step_count``.
        5. Check ``end_condition(state, result)`` and
           ``max_dynamic_steps``.
        6. If continuing: return an :class:`ExpansionResult` with the
           next step wrapper.
        7. If done: return the actual result (no expansion).
        """
        inf_instance = self._instantiate_inferencer(inferencer)

        async def _dynamic_step(*args, **kwargs):
            state = self._state
            step_input = args[0] if args else None
            # 1. Build input
            if step_index == 0:
                inp = state.get("original_input", step_input)
            else:
                prev_results = state.get("dynamic_step_results", [])
                prev = prev_results[-1] if prev_results else step_input
                if self.dynamic_input_builder is not None:
                    # call_maybe_async supports BOTH a sync builder (run inline — byte
                    # identical to the old direct call) and an async builder (awaited).
                    # MultiFlow's cross-flow-sync wrapper is async (it awaits a per-round
                    # step barrier between publishing its output and reading peers').
                    from rich_python_utils.common_utils.async_utils import (
                        call_maybe_async,
                    )

                    inp = await call_maybe_async(self.dynamic_input_builder, state, prev)
                else:
                    inp = prev

            # Part D — Per-step workspace (canonical initial/round{NN} layout)
            # ----------------------------------------------------------------
            # The ctx node stays "step_{N}" (the §2.7 deterministic slot the
            # step___expanded_* checkpoints correlate to), but the on-disk
            # workspace is pinned to children/initial|round{NN} — the dir that
            # propagation, _finalize_output, and the MFI flow-output resolver all
            # address — and passed EXPLICITLY via _rc_child(workspace=...) (M7
            # §2.12 ctx/workspace decoupling, mirroring the BTA worker dispatch).
            # Without this, a step instance with no workspace backing resolves to
            # the path-mirrored children/step_{N}, which _finalize_output never
            # finds -> empty flow deliverable -> aggregator embeds raw <Response>.
            consensus_iter = state.get("consensus_iteration_id", 0) if state else 0
            _step_ws = (
                self._workspace.child(self._dynamic_child_name(step_index, consensus_iter))
                if self._workspace is not None
                else None
            )
            if _step_ws is not None:
                _step_ws.ensure_dirs()
                # step>=2 reuses the followup instance across rounds; reset its
                # session so each round starts clean (step 0/1 use the distinct
                # default initial/followup instances).
                if inf_instance is not None and step_index >= 2 and hasattr(inf_instance, "reset_session"):
                    inf_instance.reset_session()

            # 2. Execute inferencer — forward inference_config and _inference_args
            # (stored by _ainfer at lines 742-743) to match static-mode behavior.
            extra_kwargs = {}
            if self._inference_config:
                extra_kwargs["inference_config"] = self._inference_config
            if self._inference_args:
                extra_kwargs.update(self._inference_args)
            raw_result = await inf_instance.ainfer(
                inp,
                run_context=self._rc_child(f"step_{step_index}", workspace=_step_ws),
                **extra_kwargs,
            )

            # 3. Unpack result tuple
            actual_result, next_inferencer = self._resolve_next_inferencer(raw_result)

            # 4. Update state
            if "dynamic_step_results" not in state:
                state["dynamic_step_results"] = []
            state["dynamic_step_results"].append(actual_result)
            state["dynamic_step_count"] = len(state["dynamic_step_results"])

            # NOTE: No state channel for output paths is maintained because
            # no dynamic_input_builder currently reads
            # state["dynamic_step_output_paths"] — would be speculative
            # infrastructure with no consumer. Builders can re-derive paths
            # from inf_instance._workspace if needed.

            # 5. Check termination
            should_stop = False
            if self.end_condition is not None and self.end_condition(state, actual_result):
                should_stop = True
            if state["dynamic_step_count"] >= self.max_dynamic_steps:
                should_stop = True

            # 6/7. Continue or finish
            if should_stop:
                return actual_result

            # Build next step wrapper and return ExpansionResult
            next_inf_instance = self._instantiate_inferencer(next_inferencer)
            next_wrapper = self._build_dynamic_step_wrapper(
                next_inf_instance, step_index + 1
            )

            return ExpansionResult(
                result=actual_result,
                new_steps=[next_wrapper],
                expansion_id=f"dynamic_step_{step_index + 1}",
                seed=None,
                reconstruct_from_seed=None,
            )

        return _dynamic_step

    def _build_dynamic_initial_steps(self) -> List[StepWrapper]:
        """Build the initial step list for dynamic mode.

        Creates a single :class:`StepWrapper` wrapping the dynamic step
        closure for step 0 (using ``default_initial_inferencer``).  Also
        sets ``max_expansion_events`` on the Workflow so that the
        expansion infrastructure is enabled for subsequent steps.

        Returns:
            A one-element list containing the initial dynamic step
            wrapped in a :class:`StepWrapper`.
        """
        # Enable expansion on Workflow for up to max_dynamic_steps expansions
        self.max_expansion_events = self.max_dynamic_steps

        # Build the step 0 closure using default_initial_inferencer
        step_fn = self._build_dynamic_step_wrapper(
            self.default_initial_inferencer, 0
        )

        def _sync_state(state, result):
            """Keep Workflow's local state variable pointing at self._state."""
            return self._state

        wrapper = StepWrapper(
            step_fn,
            name="dynamic_step_0",
            update_state=_sync_state,
            enable_result_save=False,
        )
        return [wrapper]

    # ------------------------------------------------------------------
    # Step builder
    # ------------------------------------------------------------------

    def _build_steps(self) -> List[StepWrapper]:
        """Convert WorkflowStepConfig list to StepWrapper list.

        Disabled steps become no-op closures (preserving indices so that
        loop_back_to references remain stable).  Each enabled step closure:
        build input → route config → execute inferencer/step_fn → extract
        output → update state in-place.

        Every StepWrapper gets an ``update_state`` callback that returns
        ``self._state`` so Workflow's local ``state`` variable stays in
        sync with the shared dict (in-place mutation only).
        """
        # Validate step configs (deferred from __attrs_post_init__ for subclass support)
        names = [sc.name for sc in self.step_configs]
        if len(names) != len(set(names)):
            raise ValueError(
                "WorkflowStepConfig names must be unique. "
                f"Duplicates found in: {names}"
            )
        for sc in self.step_configs:
            if sc.inferencer is None and sc.step_fn is None and sc.enabled:
                raise ValueError(
                    f"Step '{sc.name}' must have either 'inferencer' or "
                    f"'step_fn' set when enabled=True."
                )
            if sc.loop_back_to is not None and sc.loop_back_to not in names:
                raise ValueError(
                    f"Step '{sc.name}' has loop_back_to='{sc.loop_back_to}' "
                    f"but no step with that name exists."
                )

        # Build-time check: do any steps have loop_back_to configured?
        has_loops = any(sc.loop_back_to is not None for sc in self.step_configs)

        # Build name→index map for loop_back_to resolution
        name_to_index: Dict[str, int] = {
            sc.name: idx for idx, sc in enumerate(self.step_configs)
        }

        def _sync_state(state, result):
            """Keep Workflow's local state variable pointing at self._state."""
            return self._state

        steps: List[StepWrapper] = []
        for idx, sc in enumerate(self.step_configs):
            if not sc.enabled:
                # No-op closure preserving index for loop_back_to stability
                async def _noop(*args, _sc_name=sc.name, **kwargs):
                    return None

                wrapper = StepWrapper(
                    _noop,
                    name=sc.name,
                    update_state=_sync_state,
                    enable_result_save=False,
                )
                steps.append(wrapper)
                continue

            # Build the step closure — capture sc by default arg
            async def _step_impl(*args, _sc=sc, **kwargs):
                state = self._state

                # Iteration workspace setup — detect iteration change
                if has_loops:
                    prev_iteration = state.get("_prev_iteration", state.get("iteration", 1))
                    curr_iteration = state.get("iteration", 1)
                    if curr_iteration != prev_iteration:
                        self._setup_iteration(state)
                        state["_prev_iteration"] = curr_iteration

                # 1. Build input
                if _sc.input_builder is not None:
                    step_input = _sc.input_builder(state)
                else:
                    step_input = state.get("original_input", "")

                # 2. Route config
                extra_kwargs = {}
                if _sc.config_key and self._inference_config:
                    step_config = self._inference_config.get(_sc.config_key)
                    if step_config is not None:
                        extra_kwargs["inference_config"] = step_config

                if _sc.pass_inference_args and self._inference_args:
                    extra_kwargs.update(self._inference_args)

                # 3. Execute
                if _sc.inferencer is not None:
                    result = await _sc.inferencer.ainfer(
                        step_input,
                        run_context=self._rc_child(getattr(_sc, "name", None) or "step"),
                        **extra_kwargs,
                    )
                elif _sc.step_fn is not None:
                    from rich_python_utils.common_utils.async_utils import (
                        call_maybe_async,
                    )
                    result = await call_maybe_async(_sc.step_fn, step_input, state)
                else:
                    result = None

                # 4. Extract output
                if _sc.output_extractor is not None:
                    extracted = _sc.output_extractor(result)
                else:
                    extracted = result

                # 5. Update state in-place
                if _sc.output_state_key is not None:
                    state[_sc.output_state_key] = extracted

                if _sc.state_updater is not None:
                    _sc.state_updater(state, result)

                # Step completion marker
                self._write_step_marker(_sc.name)

                return result

            # Resolve loop_back_to name to index
            loop_back_to_idx = None
            if sc.loop_back_to is not None:
                loop_back_to_idx = name_to_index[sc.loop_back_to]

            wrapper_kwargs: Dict[str, Any] = {
                "name": sc.name,
                "update_state": _sync_state,
            }
            if sc.enable_result_save is not None:
                wrapper_kwargs["enable_result_save"] = sc.enable_result_save
            if loop_back_to_idx is not None:
                wrapper_kwargs["loop_back_to"] = loop_back_to_idx
            if sc.loop_condition is not None:
                wrapper_kwargs["loop_condition"] = sc.loop_condition
            if sc.max_loop_iterations != 5:
                wrapper_kwargs["max_loop_iterations"] = sc.max_loop_iterations
            if sc.on_loop_exhausted is not None:
                wrapper_kwargs["on_loop_exhausted"] = sc.on_loop_exhausted

            steps.append(StepWrapper(_step_impl, **wrapper_kwargs))

        return steps

    # ------------------------------------------------------------------
    # Inference entry points
    # ------------------------------------------------------------------

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        """Async inference — build state, delegate to Workflow._arun().

        Stores inference_config and _inference_args on self for closure
        access by step implementations.
        """
        if inference_config is None:
            inference_config = {}
        elif not isinstance(inference_config, dict):
            raise ValueError("'inference_config' must be a dict")

        # Store context for step closures
        self._inference_config = inference_config
        self._inference_args = _inference_args

        # Workspace was set in __attrs_post_init__ via InferencerBase syncing
        # self.workspace → self._workspace. Nothing more to do here.

        # Final result cache check (Req 8)
        if self.resume_with_saved_results:
            cached = self._load_final_result()
            if cached is not None:
                return cached

        # Build initial state — skip if already set by subclass
        if self._pending_state is None:
            if self.initial_state_factory is not None:
                self._pending_state = self.initial_state_factory(inference_input)
            else:
                self._pending_state = {"original_input": inference_input}

        # Initialize iteration tracking
        if "iteration" not in self._pending_state:
            self._pending_state["iteration"] = 1
        if "iteration_records" not in self._pending_state:
            self._pending_state["iteration_records"] = []
        self._pending_state["_prev_iteration"] = self._pending_state["iteration"]

        # Dynamic mode branch
        if self.dynamic_mode:
            # Validate dynamic mode configuration
            if self.default_initial_inferencer is None:
                raise ValueError(
                    "dynamic_mode=True requires default_initial_inferencer"
                )

            # Initialize dynamic-mode state keys
            if "dynamic_step_results" not in self._pending_state:
                self._pending_state["dynamic_step_results"] = []
            if "dynamic_step_count" not in self._pending_state:
                self._pending_state["dynamic_step_count"] = 0

            # Build initial step (single step using default_initial_inferencer)
            self._steps = self._build_dynamic_initial_steps()

            # Enable expansion on Workflow
            self.max_expansion_events = self.max_dynamic_steps
        else:
            # Existing static mode
            self._steps = self._build_steps()

        # Enable checkpointing when workspace is available
        self._auto_enable_checkpointing()

        # Run the workflow
        await Workflow._arun(self, inference_input, **_inference_args)

        # Build response
        if self.response_builder is not None:
            response = self.response_builder(self._state)
        else:
            response = self._state

        # Save final result (Req 8)
        self._save_final_result(self._state)

        return response

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        """Sync bridge — delegates to _ainfer() via _run_async()."""
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(
            self._ainfer(inference_input, inference_config, **_inference_args)
        )

    # ------------------------------------------------------------------
    # Workflow abstract methods
    # ------------------------------------------------------------------

    def _init_state(self) -> dict:
        """Return the pending state prepared in _ainfer."""
        return self._pending_state or {}

    def _get_result_path(self, result_id, *args, **kwargs):
        """Return path for checkpoint files under workspace checkpoints dir."""
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        ws = getattr(self, "_workspace", None)
        if ws is not None:
            filename = f"step_{result_id}.json"
            if self.checkpoint_subdir:
                filename = os.path.join(self.checkpoint_subdir, f"step_{result_id}.json")
            return ws.checkpoint_path(filename)
        return f"step_{result_id}.json"

    def _save_result(self, result, output_path: str):
        """Save step result as JSON with explicit dict__ pre-conversion."""
        from rich_python_utils.common_utils.map_helper import dict__
        from rich_python_utils.io_utils.json_io import write_json

        if not output_path:
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_json(dict__(result, recursive=True), output_path, indent=2)

    def _load_result(self, result_id, result_path_or_preloaded_result):
        """Load step result from JSON file."""
        from rich_python_utils.io_utils.json_io import read_json

        if isinstance(result_path_or_preloaded_result, str):
            return read_json(result_path_or_preloaded_result)
        return result_path_or_preloaded_result

    def _exists_result(self, result_id, result_path):
        """Check if a JSON result file exists."""
        if not result_path:
            return False
        json_path = result_path
        if not json_path.endswith(".json"):
            json_path = result_path + ".json"
        return os.path.exists(json_path) or os.path.exists(result_path)

    def _handle_abort(self, abort_exc, step_result, state):
        """Handle WorkflowAborted — return state (preserves partial results)."""
        return state

    # ------------------------------------------------------------------
    # Checkpoint overrides
    # ------------------------------------------------------------------

    def _save_loop_checkpoint(
        self, step_index, next_step_index, last_saved_result_id, state, *args, **kwargs
    ):
        """Save loop checkpoint — stringify loop_counts int keys before serialization.

        dict__ converts non-string-keyed dicts to list-of-pairs format,
        which would break _try_load_checkpoint's .items() call on resume.
        Pre-converting to string keys avoids this.
        """
        self._save_checkpoint(
            {
                "version": 1,
                "exec_seq": self._exec_seq,
                "step_index": step_index,
                "result_id": last_saved_result_id,
                "next_step_index": next_step_index,
                "loop_counts": {str(k): v for k, v in self._loop_counts.items()},
                "state": state,
            },
            *args,
            **kwargs,
        )

    def _try_load_checkpoint(self, *args, **kwargs):
        """Load checkpoint and convert string keys in loop_counts back to int."""
        ckpt = super()._try_load_checkpoint(*args, **kwargs)
        if ckpt is not None and "loop_counts" in ckpt:
            lc = ckpt["loop_counts"]
            if isinstance(lc, dict):
                ckpt["loop_counts"] = {int(k): v for k, v in lc.items()}
        return ckpt
