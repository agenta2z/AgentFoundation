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

import os
from typing import Any, Callable, Dict, List, Optional

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.serializable import SerializationMode
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import (
    ResultPassDownMode,
)
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
    enable_result_save: bool = attrib(default=True)
    config_key: Optional[str] = attrib(default=None)
    pass_inference_args: bool = attrib(default=False)
    enabled: bool = attrib(default=True)


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
        workspace_path: Optional workspace root for checkpoint file I/O.
    """

    step_configs: List[WorkflowStepConfig] = attrib(factory=list)
    response_builder: Optional[Callable] = attrib(default=None)
    initial_state_factory: Optional[Callable] = attrib(default=None)
    workspace_path: Optional[str] = attrib(default=None)

    # --- New: Iteration Management ---
    iteration_workspace_factory: Optional[Callable[[str, int], str]] = attrib(default=None)
    reset_sessions_per_iteration: bool = attrib(default=False)
    iteration_record_builder: Optional[Callable[[dict], dict]] = attrib(default=None)
    checkpoint_subdir: Optional[str] = attrib(default=None)

    # --- Suppress Workflow constructor parameters (init=False) ---
    result_pass_down_mode = attrib(default=ResultPassDownMode.NoPassDown, init=False)
    unpack_single_result = attrib(default=False, init=False)
    ignore_stop_flag_from_saved_results = attrib(default=True, init=False)
    auto_mode = attrib(default=SerializationMode.PREFER_CLEAR_TEXT, init=False)
    checkpoint_mode = attrib(default="jsonfy", init=False)

    # Internal state (not user-facing)
    _pending_state: Optional[Dict] = attrib(default=None, init=False)
    _inference_config: Optional[Dict] = attrib(default=None, init=False)
    _inference_args: Optional[Dict] = attrib(default=None, init=False)

    def __attrs_post_init__(self):
        super(LinearWorkflowInferencer, self).__attrs_post_init__()

        # Workspace reconstruction
        if self.workspace_path is not None:
            from agent_foundation.common.inferencers.inferencer_workspace import (
                InferencerWorkspace,
            )
            self._workspace = InferencerWorkspace(root=self.workspace_path)
        else:
            self._workspace = None

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
        if self.workspace_path is not None:
            iter_path = self._get_iteration_workspace(
                self.workspace_path, iteration, self.iteration_workspace_factory
            )
            ws = InferencerWorkspace(root=iter_path)
            ws.ensure_dirs()
            self._workspace = ws

        # 2. Update child Workflow _result_root_override
        self._setup_child_workflows(state)

        # 3. Optionally reset child inferencer sessions
        if self.reset_sessions_per_iteration:
            seen_ids: set = set()
            for sc in self.step_configs:
                inf = sc.inferencer
                if inf is not None and id(inf) not in seen_ids:
                    seen_ids.add(id(inf))
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
        """Enable checkpoint/resume when workspace is available and no override is set."""
        if self._result_root_override is None and self._workspace is not None:
            from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
                StepResultSaveOptions,
            )
            self.enable_result_save = StepResultSaveOptions.Always
            self.resume_with_saved_results = True

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
                        step_input, **extra_kwargs
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
                "enable_result_save": sc.enable_result_save,
            }
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

        # Workspace setup — skip if already set by subclass
        if self._workspace is None and self.workspace_path is not None:
            from agent_foundation.common.inferencers.inferencer_workspace import (
                InferencerWorkspace,
            )
            self._workspace = InferencerWorkspace(root=self.workspace_path)

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

        # Build steps
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
