"""DualInferencer — propose-review-fix consensus loop as a composable inferencer.

Models a single consensus phase: base_inferencer proposes, review_inferencer
reviews, fixer_inferencer addresses issues, and the loop repeats until consensus
or max_iterations. Two DualInferencer instances can be chained with swapped roles
for a full dual-phase workflow (e.g., planning → execution).

Inherits from LinearWorkflowInferencer (which extends both InferencerBase and
Workflow) so the inner consensus loop can leverage Workflow's checkpoint /
loop-resume system.  When ``enable_checkpoint=True`` and ``checkpoint_dir``
is provided, the workflow persists each step's result to disk and can resume
from a crash.
"""

import json
import logging
import os
import re
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Union

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusAttemptRecord,
    ConsensusConfig,
    ConsensusIterationRecord,
    DualInferencerResponse,
    InferencerExecutionError,
    InferencerResponse,
    ReflectionStyles,
    ResponseSelectors,
    severity_at_most,
)
from agent_foundation.common.inferencers.constants import (
    DEFAULT_PLACEHOLDER_DUAL_COUNTER_FEEDBACK,
    DEFAULT_PLACEHOLDER_DUAL_INPUT,
    DEFAULT_PLACEHOLDER_DUAL_ISSUES,
    DEFAULT_PLACEHOLDER_DUAL_PROPOSAL,
    DEFAULT_PLACEHOLDER_DUAL_REASONING,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)
from agent_foundation.common.inferencers.template_defaults import (
    FOLLOWUP_TEMPLATE_DEFAULTS,
    REVIEW_TEMPLATE_DEFAULTS,
)
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.input_and_response import InputAndResponse
from rich_python_utils.common_objects.workflow.common.exceptions import (
    WorkflowAborted,
)
from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
    StepResultSaveOptions,
)
from rich_python_utils.common_objects.workflow.workflow import Workflow
from rich_python_utils.io_utils.artifact import artifact_type
from rich_python_utils.string_utils.formatting.template_manager import (
    TemplateManager,
)
from rich_python_utils.string_utils.xml_helpers import unescape_xml

logger = logging.getLogger(__name__)


class _RoleDisabledError(Exception):
    """Raised by ``DualInferencer._render_role_prompt`` when a role's prompt
    template is not configured AND no implicit default key exists in the
    active TemplateManager.

    The corresponding step (``_step_review_impl`` / ``_step_fix_impl``) catches
    this and silently skips the step. This is the "graceful degradation" path
    for the implicit-default semantics — see the class docstring on
    ``DualInferencer.review_prompt`` for the full design.

    Internal — never propagates to user code.
    """

    def __init__(self, role: str):
        super().__init__(f"Role {role!r} is disabled (no template available)")
        self.role = role


@artifact_type(Workflow, type="json", group="workflows")
@attrs
class DualInferencer(LinearWorkflowInferencer):
    """Propose-review-fix consensus loop as a first-class inferencer.

    Inherits from ``LinearWorkflowInferencer`` (which extends both
    ``InferencerBase`` and ``Workflow``), gaining shared infrastructure
    for checkpoint/loop-resume, step building, and state management.

    MRO: DualInferencer → LinearWorkflowInferencer → InferencerBase →
         Workflow → WorkNodeBase → Serializable → Debuggable →
         Identifiable → Resumable → PostProcessable → ABC

    Workflow-suppressed attrs (result_pass_down_mode, unpack_single_result,
    etc.) are inherited from LWI with ``init=False`` defaults.

    Implements a multi-round consensus workflow:
    1. base_inferencer generates an initial proposal
    2. review_inferencer reviews the proposal
    3. If consensus not reached, fixer_inferencer addresses issues
    4. The improved proposal is re-reviewed (step 2)
    5. Loop continues until consensus or max_iterations

    The fixer can optionally produce counter-feedback rejecting invalid
    review issues, which is passed to the reviewer in the next iteration.

    Usage:
        # Simple 2-agent mode (base_inferencer also fixes):
        dual = DualInferencer(
            base_inferencer=proposer,
            review_inferencer=reviewer,
        )
        result = dual("Design a REST API for user management")

        # 3-agent mode:
        dual = DualInferencer(
            base_inferencer=proposer,
            review_inferencer=reviewer,
            fixer_inferencer=dedicated_fixer,
        )

        # Async with checkpointing (recommended):
        async with DualInferencer(
            ...,
            enable_checkpoint=True,
            checkpoint_dir="/tmp/my_workflow",
        ) as inf:
            result = await inf.ainfer("Design a REST API")

    Attributes:
        base_inferencer: Proposer/planner inferencer.
        review_inferencer: Reviewer/reflector inferencer.
        fixer_inferencer: Fixer inferencer (defaults to base_inferencer).
        consensus_config: Loop configuration (max iterations, threshold, etc.).
        initial_prompt: Raw Jinja2 template string for the initial prompt.
            None means passthrough. In the modern (Phase 2) path, leaves
            self-render via their own template_manager + template_key.
        review_prompt: Raw Jinja2 template for review prompts (fallback only).
        followup_prompt: Raw Jinja2 template for followup/fix prompts (fallback only).
        review_parser: Callable to parse raw review output into structured dict.
        followup_response_parser: Callable to parse counter-feedback from fixer output.
        response_parser: Callable to extract/clean raw output from any sub-inferencer.
        consensus_checker: Callable (parsed_review, threshold) → bool.
        response_selector: How to select the final output from the response object.
        issue_id_format: Format string for issue IDs.
        phase: Label for this consensus phase (for logging/metadata).
        enable_checkpoint: If True, enable Workflow checkpoint/resume.
        checkpoint_dir: Directory for checkpoint files.
    """

    # === Slot-based template role defaults (consumed by config_utils._walk) ===
    SLOT_DEFAULTS: ClassVar[Dict[str, Any]] = {
        "review_inferencer": REVIEW_TEMPLATE_DEFAULTS,
        # Phase 2 (leaf-owned template rendering): symmetric with review.
        # Cascades template_key="followup" to fixer_inferencer leaves so they
        # render plan/main/followup.jinja2 themselves (no YAML declaration
        # needed). Wired atomically with Dual's leaf-rendering path enabling
        # in _step_fix_impl — the new path checks if the leaf has template_key
        # set (which this default ensures) and routes accordingly.
        "fixer_inferencer": FOLLOWUP_TEMPLATE_DEFAULTS,
    }

    # === Template-transparent slots (consumed by config_utils._walk wrapping descent) ===
    # When this Dual fills a role-default slot of an enclosing orchestrator
    # (e.g. BTA's aggregator_inferencer), the parent's defaults pass through
    # to these inner slots instead of being applied to the wrapping Dual.
    # Dual's own SLOT_DEFAULTS (review_inferencer.template_key=review) is
    # applied separately when ``_walk`` enters the Dual node.
    _TEMPLATE_TRANSPARENT_SLOTS: ClassVar[List[str]] = [
        "base_inferencer", "review_inferencer", "fixer_inferencer",
    ]

    _workspace_propagation_skip: frozenset = frozenset((
        "base_inferencer",
        "review_inferencer",
        "fixer_inferencer",
    ))

    base_inferencer: InferencerBase = attrib(default=None)
    review_inferencer: InferencerBase = attrib(default=None)
    fixer_inferencer: Optional[InferencerBase] = attrib(default=None)

    consensus_config: ConsensusConfig = attrib(factory=ConsensusConfig)

    enable_round_audit: bool = attrib(default=True)
    """Emits per-round audit: outputs/round_log.jsonl + children/round_NN/ nav links."""

    initial_prompt: Optional[str] = attrib(default=None)
    # ------------------------------------------------------------------
    # Prompt template resolution semantics (review_prompt / followup_prompt)
    # ------------------------------------------------------------------
    # Both default to None — but None has IMPLICIT semantics:
    #
    # 1. Default (None / unset in YAML):
    #      Try to resolve the implicit template key (``"review"`` / ``"followup"``)
    #      against the active TemplateManager + _template_root_space at
    #      construction time.
    #      - If the template IS found → use it (silent success).
    #      - If the template is NOT found → silently DISABLE the role
    #        (no review step / no fixer step) — i.e. caller didn't ask for
    #        review/fixer behavior, and the domain doesn't provide a default,
    #        so we degrade gracefully to single-shot proposal.
    #
    # 2. Explicit (set in YAML to a template_key or raw template string):
    #      Use the configured value. If the template manager cannot resolve
    #      it, raise ValueError loudly at render time. (Caller asked for it
    #      explicitly; a misconfiguration is a programmer error.)
    #
    # Rationale for this split:
    #   * No more hidden in-Python default strings (which caused the 2026-05
    #     doubly-templated rendering bug).
    #   * Domains that want default review/fixer behavior just need to drop
    #     ``<root>/main/review.jinja2`` and ``<root>/main/followup.jinja2``
    #     into their template tree — no YAML change needed.
    #   * Domains that don't have those files cleanly degrade to single-shot
    #     (no review/fixer) instead of silently rendering broken prompts.
    #   * Explicit misconfiguration always fails loudly.
    #
    # Internal state set during ``__attrs_post_init__``:
    #   ``self._review_role_disabled``    — True iff review step should skip.
    #   ``self._followup_role_disabled``  — True iff fixer step should skip.
    review_prompt: Optional[str] = attrib(default=None)
    followup_prompt: Optional[str] = attrib(default=None)

    review_parser: Callable = attrib(default=None)
    followup_response_parser: Callable = attrib(default=None)
    response_parser: Callable = attrib(default=None)
    consensus_checker: Callable = attrib(default=None)

    response_selector: Union[
        Callable[["InferencerResponse"], Any], ResponseSelectors
    ] = attrib(default=ResponseSelectors.BaseResponse)

    issue_id_format: str = attrib(default="ISS-{iteration:02d}-{index:03d}")
    phase: str = attrib(default="")

    new_session_per_attempt: bool = attrib(default=True)

    # Placeholder keys for template variables
    placeholder_input: str = attrib(default=DEFAULT_PLACEHOLDER_DUAL_INPUT)
    placeholder_proposal: str = attrib(default=DEFAULT_PLACEHOLDER_DUAL_PROPOSAL)
    placeholder_issues: str = attrib(default=DEFAULT_PLACEHOLDER_DUAL_ISSUES)
    placeholder_reasoning: str = attrib(default=DEFAULT_PLACEHOLDER_DUAL_REASONING)
    placeholder_counter_feedback: str = attrib(
        default=DEFAULT_PLACEHOLDER_DUAL_COUNTER_FEEDBACK
    )

    # --- Checkpoint-specific attributes ---
    checkpoint_dir: Optional[str] = attrib(default=None, kw_only=True)
    enable_checkpoint: bool = attrib(default=False, kw_only=True)

    # --- §3 Part B: multi-reviewer panel ---
    # When >1, the review step runs ``review_inferencer`` k times and merges the
    # parsed reviews (deterministic union / dedup / never-downgrade) via
    # ``review_aggregator`` (defaults to ``flow_parsers.merge_reviews``). Default 1
    # => single-reviewer path, byte-identical. ``num_reviewers: k`` is the §2.5
    # ``[reviewer]*k`` panel realized over one declared reviewer definition.
    num_reviewers: int = attrib(default=1, kw_only=True)
    review_aggregator: Optional[Callable] = attrib(default=None, kw_only=True)
    # Optional declarative ``[reviewer] * k`` (e.g. via ``_repeat_``): k INDEPENDENT
    # reviewer instances. When set, panelist ``i`` runs on ``reviewers[i]`` (its own
    # definition state — switch_role/template), not a reused ``review_inferencer``.
    # When None (default), the panel reuses ``review_inferencer`` across distinct
    # ``review/panelist_i`` child contexts (runtime state still isolated). If set,
    # ``num_reviewers`` defaults to ``len(reviewers)``.
    reviewers: Optional[list] = attrib(default=None, kw_only=True)

    # --- Workspace support (opt-in, overrides checkpoint_dir when set) ---

    # --- Workflow-suppressed attrs inherited from LWI (init=False) ---
    # result_pass_down_mode, unpack_single_result,
    # ignore_stop_flag_from_saved_results, auto_mode, max_loop_iterations
    # are all inherited from LinearWorkflowInferencer with init=False.
    # checkpoint_mode is also inherited from LWI (default="jsonfy").

    def __attrs_post_init__(self):
        # --- Domain-specific init BEFORE calling super() ---
        # (LWI's __attrs_post_init__ calls Workflow.__attrs_post_init__
        # which is what DualInferencer needs)

        if (not self.response_types) or self.response_types == (str,):
            self.response_types = (str, InferencerResponse, DualInferencerResponse)

        # Default fixer to base_inferencer (2-agent mode)
        if self.fixer_inferencer is None:
            self.fixer_inferencer = self.base_inferencer

        # Track whether each role was explicitly configured by the caller.
        # Drives error-vs-disable semantics when the template cannot be
        # resolved (see ``_resolve_role_template_for_render``).
        self._review_explicit = self.review_prompt is not None
        self._followup_explicit = self.followup_prompt is not None
        # Set to True at render time if the implicit default key cannot be
        # resolved AND the caller didn't explicitly configure the role —
        # the corresponding step is then skipped.
        self._review_role_disabled = False
        self._followup_role_disabled = False

        # Prompt rendering: leaf-owned (Phase 2). Review/followup leaves
        # self-render via their own template_manager + template_key (set by
        # SLOT_DEFAULTS). The orchestrator assembles the feed dict and passes
        # it via extra_feed=; no orchestrator-level prompt construction needed.
        # _render_role_prompt is kept as a fallback for raw Jinja2 strings
        # (review_prompt / followup_prompt explicitly set).

        # Default parsers
        if self.review_parser is None:
            self.review_parser = DualInferencer._default_parse_review
        if self.followup_response_parser is None:
            self.followup_response_parser = (
                DualInferencer._default_parse_counter_feedback
            )
        if self.response_parser is None:
            self.response_parser = DualInferencer._default_response_parser
        if self.consensus_checker is None:
            self.consensus_checker = self._default_check_consensus

        # Workspace setup. Pass the full `workspace=InferencerWorkspace(...)`
        # object (declarative form) — the legacy `workspace_root: str`
        # shorthand was removed 2026-05-05.
        #
        # If a workspace was pre-set on `_workspace` (rare), make sure dirs
        # exist. Otherwise, the canonical sync happens in
        # `super().__attrs_post_init__()` below: InferencerBase copies
        # `self.workspace → self._workspace`, which fires the property
        # setter (configure + propagate to children).
        if self._workspace is not None:
            self._workspace.ensure_dirs()
        # else: super() will sync from self.workspace.

        # Set parent debuggable for nested inferencers (deduplicate by identity)
        seen_ids = set()
        for inf in (
            self.base_inferencer,
            self.review_inferencer,
            self.fixer_inferencer,
        ):
            if (
                inf is not None
                and isinstance(inf, Debuggable)
                and id(inf) not in seen_ids
            ):
                seen_ids.add(id(inf))
                inf.set_parent_debuggable(self)

        # --- Now call super().__attrs_post_init__() ---
        # This calls LWI's __attrs_post_init__ which calls
        # Workflow.__attrs_post_init__ (via InferencerBase).
        super(DualInferencer, self).__attrs_post_init__()

    # ------------------------------------------------------------------
    # M-AF1/C5: runtime role resolution (no self-mutation under a real ctx)
    # ------------------------------------------------------------------
    def _role_get(self, name: str):
        """Resolve a runtime role (``review_inferencer``/``fixer_inferencer``/``reviewers``):
        under a real (non-legacy) ctx, the per-run value cached in ``ctx.node().scratch``
        (so a shared instance run across N concurrent ctxs never clobbers); else the
        instance attribute — the static definition, or the legacy/legacy-mint value a
        legacy run mutated in place (byte-identical)."""
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is not None and not ctx.legacy_mint:
            scratch = ctx.node().scratch
            if name in scratch:
                return scratch[name]
        return getattr(self, name)

    def _role_set(self, name: str, value, ctx) -> None:
        """Store a runtime-selected role. Under a real (non-legacy) ``ctx`` write
        ``ctx.node().scratch`` (no self-mutation → concurrency-isolated); else mutate the
        instance attribute (legacy / legacy-mint — byte-identical, so post-call
        ``self.review_inferencer`` reflects the selection)."""
        if ctx is not None and not ctx.legacy_mint:
            ctx.node().scratch[name] = value
        else:
            setattr(self, name, value)

    # ------------------------------------------------------------------
    # Commit 3 inherited-field audit: per-run TRANSIENT runtime fields
    # ------------------------------------------------------------------
    # The remaining per-run runtime fields Dual mutated on ``self`` during a
    # call/attempt/round — ``_current_attempt``, ``_current_inference_config``,
    # ``_current_extra_inference_args``, ``_current_round_ws``,
    # ``_last_output_child_ws``, and the rebuilt ``step_configs`` — are all
    # transient (re-derived each call; never serialized/resumed). They are
    # virtualized into the active ctx's ``NodeRunState.scratch`` so a SHARED
    # Dual/MFDual instance run under N concurrent ``RunContext``s never clobbers
    # them across runs; with no active ctx, or a legacy / legacy-mint root, they
    # fall back to an instance ``__dict__`` backing (byte-identical to the
    # pre-virtualization behaviour — the legacy ``infer()`` path).
    #
    # Resolution mirrors ``_role_get``/``_role_set`` EXACTLY (the ``legacy_mint``
    # check + ``ctx.node().scratch`` home), keeping the two seams coherent:
    # ``_state``/``_pending_state`` are already virtualized by LWI's C10/G1
    # compat-properties; these add the rest of the Dual-local runtime surface.
    _RUN_SCRATCH_PREFIX = "_dual_run::"

    def _run_get(self, name: str, default=None):
        """Read a per-run transient field. Under a real (non-legacy) ctx return the
        per-run value cached in ``ctx.node().scratch`` (concurrency-isolated); else the
        instance ``__dict__`` backing (legacy / legacy-mint / no-ctx — byte-identical).
        ``default`` is returned when neither home holds the field, so callers that used
        ``getattr(self, "_current_*", <default>)`` keep their exact semantics."""
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is not None and not ctx.legacy_mint:
            scratch = ctx.node().scratch
            key = self._RUN_SCRATCH_PREFIX + name
            if key in scratch:
                return scratch[key]
            return default
        return self.__dict__.get(name + "_backing", default)

    def _run_set(self, name: str, value) -> None:
        """Write a per-run transient field. Under a real (non-legacy) ctx write
        ``ctx.node().scratch`` (no self-mutation → concurrency-isolated); else mutate the
        instance ``__dict__`` backing (legacy / legacy-mint / no-ctx — byte-identical)."""
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is not None and not ctx.legacy_mint:
            ctx.node().scratch[self._RUN_SCRATCH_PREFIX + name] = value
        else:
            self.__dict__[name + "_backing"] = value

    # ``_last_output_child_ws`` / ``_propose_child_ws`` — the per-run canonical-output
    # trackers set by ``_step_propose_impl`` / ``_step_fix_impl`` and read by
    # ``_finalize_output``. Virtualized as COMPAT-PROPERTIES (same seam as
    # ``step_configs``) so plain ``self._<field>`` access transparently routes through
    # the active ctx's scratch under a real context (concurrency-isolated — a shared
    # Dual/MFDual instance run under N contexts never clobbers them) and the instance
    # ``__dict__`` backing under legacy/no-ctx (byte-identical). They share the exact
    # ``_run_get``/``_run_set`` backing key, so the existing explicit ``_run_set`` calls
    # and these bare assignments resolve to the same home.
    @property
    def _last_output_child_ws(self):
        return self._run_get("_last_output_child_ws", None)

    @_last_output_child_ws.setter
    def _last_output_child_ws(self, value):
        self._run_set("_last_output_child_ws", value)

    @property
    def _propose_child_ws(self):
        return self._run_get("_propose_child_ws", None)

    @_propose_child_ws.setter
    def _propose_child_ws(self, value):
        self._run_set("_propose_child_ws", value)

    # ``step_configs`` is virtualized as a COMPAT-PROPERTY (not via _run_get/_run_set)
    # because it is inherited as an attrs ``attrib`` on LWI and read DIRECTLY as
    # ``self.step_configs`` by the LWI engine itself (``_build_steps`` at build time,
    # ``_setup_iteration`` when reset-per-iteration is on) — a property is the only seam
    # that also routes those inherited direct reads through the active ctx. It is rebuilt
    # per call/attempt and patched in place at the propose-override site
    # (``self.step_configs[0] = ...``); confirmed Dual-local (NOT read by the base
    # rich_python_utils ``Workflow`` engine — only by LWI/Dual). The in-place index
    # assignment mutates the per-run list resolved by the getter, so it stays isolated
    # under a real ctx and byte-identical under legacy. The attrs ``__init__`` assignment
    # (no active ctx at construction) routes to the instance backing.
    @property
    def step_configs(self):
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is not None and not ctx.legacy_mint:
            scratch = ctx.node().scratch
            key = self._RUN_SCRATCH_PREFIX + "step_configs"
            if key in scratch:
                return scratch[key]
        return self.__dict__.get("step_configs_backing", [])

    @step_configs.setter
    def step_configs(self, value):
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is not None and not ctx.legacy_mint:
            ctx.node().scratch[self._RUN_SCRATCH_PREFIX + "step_configs"] = value
            return
        self.__dict__["step_configs_backing"] = value

    # ------------------------------------------------------------------
    # Block WorkNodeBase.run() / arun() — callers must use infer()/ainfer()
    # ------------------------------------------------------------------

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "Use infer() or ainfer() — DualInferencer.run() is disabled "
            "because Workflow._arun() requires state setup that only "
            "_ainfer() provides."
        )

    async def arun(self, *args, **kwargs):
        raise NotImplementedError(
            "Use infer() or ainfer() — DualInferencer.arun() is disabled "
            "because Workflow._arun() requires state setup that only "
            "_ainfer() provides."
        )

    # ------------------------------------------------------------------
    # Workflow abstract method implementation
    # ------------------------------------------------------------------

    def _get_result_path(self, result_id, *args, **kwargs):
        attempt = self._run_get("_current_attempt", 0)
        if self._workspace is not None:
            return self._workspace.checkpoint_path(
                os.path.join(
                    f"attempt_{attempt:02d}", f"step_{result_id}.json"
                )
            )
        if self.checkpoint_dir:
            return os.path.join(
                self.checkpoint_dir,
                f"attempt_{attempt:02d}",
                f"step_{result_id}.json",
            )
        # Child mode: _resolve_result_path will apply _result_root_override
        # to basename.  Include attempt to avoid collisions across attempts.
        return f"step_a{attempt:02d}_{result_id}.json"

    def _try_load_checkpoint(self, *args, **kwargs):
        ckpt = super()._try_load_checkpoint(*args, **kwargs)
        if ckpt is not None and "loop_counts" in ckpt:
            # CRITICAL: JSON mandates string keys. After JSON round-trip,
            # loop_counts keys are strings but Workflow uses int step indices.
            ckpt["loop_counts"] = {int(k): v for k, v in ckpt["loop_counts"].items()}
        return ckpt

    def _save_loop_checkpoint(
        self, step_index, next_step_index, last_saved_result_id, state, *args, **kwargs
    ):
        # Override for two reasons:
        # 1. Skip pickle.dumps(state) validation — base class resets
        #    _state_picklability_verified=False every _arun() call (line 551).
        # 2. CRITICAL: Convert loop_counts int keys to strings BEFORE dict__
        #    sees them. dict__ converts non-string-keyed dicts to list-of-pairs
        #    format [{"key":k,"value":v}] (map_helper.py:548), which would break
        #    _try_load_checkpoint's .items() call on resume.
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

    # ------------------------------------------------------------------
    # Consensus step builders
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Workspace propagation — per-round child workspaces
    # ------------------------------------------------------------------

    def _propagate_workspace_to_children(self, parent_workspace):
        """Dual override: assign ``children/propose/`` to base_inferencer.

        review_inferencer and fixer_inferencer get per-round workspaces
        at runtime (in ``_step_review_impl`` / ``_step_fix_impl``).
        """
        from agent_foundation.common.inferencers.inferencer_base import (
            InferencerBase,
        )
        base = getattr(self, "base_inferencer", None)
        if base is not None and isinstance(base, InferencerBase):
            if getattr(base, "_workspace", None) is None:
                child_ws = parent_workspace.child("propose")
                child_ws.ensure_dirs()
                base._workspace = child_ws
        super()._propagate_workspace_to_children(parent_workspace)

    def _write_step_marker(self, step_name):
        """Write completion markers to both the outer and per-round workspaces.

        Extends LWI's marker (outer ``artifacts/.{step}_completed``) with a
        per-round marker in the child inferencer's workspace — e.g.,
        ``children/round_01/children/review/artifacts/.review_completed``.
        """
        super()._write_step_marker(step_name)
        child_inf = {
            "review": self._role_get("review_inferencer"),
            "fix": self._role_get("fixer_inferencer"),
            "propose": self.base_inferencer,
        }.get(step_name)
        child_ws = getattr(child_inf, "_workspace", None) if child_inf else None
        if child_ws is not None and hasattr(child_ws, "write_marker"):
            try:
                child_ws.write_marker(step_name)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Output finalization (orchestrator override)
    # ------------------------------------------------------------------

    def _finalize_output(self, response):
        """Dual override: symlink to canonical child's output.

        The canonical child is tracked by ``_last_output_child_ws``,
        set by ``_step_propose_impl`` and ``_step_fix_impl``.

        When the fix phase refines the propose output, it typically
        reproduces ``output_path`` but not auxiliary deliverables (e.g.
        per-proposal files from research-propose).  After surfacing the
        fix child's deliverables, we merge any propose-phase deliverables
        that the fix phase did not reproduce — ``_symlink_or_copy`` skips
        entries that already exist, so the fix output always takes
        precedence.

        Safety net (B1): detect when the last review was not-consensus but
        no fix ran — the output is the un-fixed propose, which may contain
        CRITICAL issues the reviewer explicitly flagged.
        """
        # Safety net: check if a review ran but fix never updated the output
        child_ws = self._run_get("_last_output_child_ws", None)
        propose_ws = getattr(self, "_propose_child_ws", None)
        state = getattr(self, "_state", None) or {}
        last_iter = getattr(self, "_last_iteration_record", None)
        if (
            child_ws is not None
            and propose_ws is not None
            and child_ws is propose_ws
            and last_iter is not None
            and not getattr(last_iter, "consensus_reached", True)
        ):
            last_fb = getattr(last_iter, "review_feedback", None) or {}
            sev = last_fb.get("severity", last_fb.get("overall_severity", ""))
            self.log_warning(
                {
                    "event": "DEGRADED_OUTPUT",
                    "message": (
                        "Last review was NOT consensus but fix never ran — "
                        "output is the un-fixed propose. The review's issues "
                        "were NOT addressed."
                    ),
                    "severity": sev,
                    "approved": last_fb.get("approved", last_fb.get("approve")),
                },
                "DegradedOutput",
            )
        if child_ws is not None:
            self._symlink_child_output(child_ws)
            propose_ws = getattr(self, "_propose_child_ws", None)
            if propose_ws is not None and propose_ws is not child_ws:
                propose_fd = getattr(propose_ws, "deliverables_dir", None)
                own_fd = getattr(self._workspace, "deliverables_dir", None)
                if propose_fd and own_fd and os.path.isdir(propose_fd):
                    os.makedirs(own_fd, exist_ok=True)
                    for entry in os.listdir(propose_fd):
                        self._symlink_or_copy(
                            os.path.join(propose_fd, entry),
                            os.path.join(own_fd, entry),
                        )
            resolved = self.resolve_output_path()
            if resolved and os.path.isfile(resolved):
                self._emit_output_manifest(resolved)
            return response
        return super()._finalize_output(response)

    # ------------------------------------------------------------------
    # Finalization — audit bookkeeping
    # ------------------------------------------------------------------

    def _finalize_response(self):
        """Dual audit bookkeeping — surfacing moved to ``_finalize_output``.

        Only the round_log.jsonl audit is relevant here. Output surfacing
        and deliverable promotion are handled by the ``_finalize_output``
        override via ``_last_output_child_ws`` symlinks.
        """
        pass

    def _active_proposer(self):
        """Return the active proposer for this Dual run.

        v1.7 §4.4: Dual surfaces EITHER base OR fixer (never both). Selection
        is based on the latest iteration's counter_feedback:
          - counter_feedback is None → review passed, base is active
          - counter_feedback is non-None → fixer ran, fixer is active

        v1.7.2 BUG FIX: the runtime state structure is
        ``state["attempt_record"]["iterations"]`` (see _pending_state setup
        at line ~489), NOT ``state["consensus_iterations"]``. Earlier code
        read the wrong key, so this method always returned base_inferencer
        in real runs (the fixer never won pass-through surfacing). Tests
        also fabricated the wrong shape and didn't catch this. Now reads
        the canonical attempt_record path with a back-compat fallback for
        any test that fabricates the older synthetic shape.

        Returns base_inferencer when state is not yet initialized.
        """
        state = getattr(self, "_state", None) or {}
        # Preferred (real runtime) path.
        attempt_record = state.get("attempt_record")
        if isinstance(attempt_record, dict):
            iters = attempt_record.get("iterations") or []
        elif attempt_record is not None:
            iters = getattr(attempt_record, "iterations", None) or []
        else:
            # Back-compat fallback for fabricated test states.
            iters = state.get("consensus_iterations") or []
        if not iters:
            return self.base_inferencer
        last = iters[-1]
        if hasattr(last, "counter_feedback"):
            counter = getattr(last, "counter_feedback", None)
        elif isinstance(last, dict):
            counter = last.get("counter_feedback")
        else:
            counter = None
        if counter is None:
            return self.base_inferencer
        return self._role_get("fixer_inferencer") if self._role_get("fixer_inferencer") is not None else self.base_inferencer

    def _resolve_prior_proposer_output_path(self) -> Optional[str]:
        """Resolve the on-disk file path of the active proposer's prior output.

        Used by ``_build_followup_prompt`` and ``_build_review_prompt`` to give
        the LLM a concrete path it can ``cp`` / ``read_file`` against, instead
        of forcing it to "copy mentally" from inline content (which empirically
        leads to regeneration drift; see _docs/_plans/
        dual_inferencer_path_aware_followup_INTEGRATED_plan.md).

        Two-tier deterministic resolution (domain-agnostic — works for plan,
        implementation, evaluation, or any future Dual usage):

          Tier 1 — Deliverable file (preferred for orchestrators: BTA, PTI):
            If the active proposer's workspace has non-empty
            ``final_deliverables/``, return the deliverable file matching the
            proposer's ``_output_path`` basename (typically ``output.md``);
            else the first non-dotfile deliverable in alphabetical order.
            (Dotfiles like ``.self_promoted`` are filtered.)

          Tier 2 — Outputs file (canonical for leaf inferencers):
            If ``outputs/<basename>`` exists on disk, return it.

          Tier 3 — None: no usable file on disk; caller should pass an empty
            string into the prompt feed so Jinja's ``{% if %}`` is falsy.

        Pure: filesystem read only, no mutation; never raises.

        Note on timing: at the call sites in ``_build_followup_prompt`` /
        ``_build_review_prompt``, ``_active_proposer()`` correctly identifies
        who produced the *current* ``state["base_output_str"]`` because
        iteration records are appended AFTER ``_step_fix_impl`` completes
        (see ``_step_fix_impl`` ~line 1032), creating a one-step lag that
        aligns with the "who produced the current proposal?" question.
        """
        proposer = self._active_proposer()
        if proposer is None:
            return None
        # Use _last_output_child_ws — the canonical workspace of whoever
        # last produced output (propose or fix). Under M7 the proposer's
        # workspace may be published to the child context only (not the
        # instance), so direct proposer._workspace can return None.
        ws = self._last_output_child_ws
        if ws is None:
            ws = getattr(proposer, "_workspace", None)
        if ws is None:
            return None

        from agent_foundation.common.inferencers.inferencer_workspace import DEFAULT_OUTPUT_FILENAME
        out_basename = os.path.basename(
            getattr(proposer, "_output_path", None) or DEFAULT_OUTPUT_FILENAME
        )

        # Tier 1: deliverable file
        if getattr(ws, "has_deliverables", False):
            try:
                preferred = ws.deliverable_path(out_basename)
            except Exception:
                preferred = None
            if preferred and os.path.isfile(preferred):
                return preferred
            # Fall back: first non-dotfile deliverable, alphabetically.
            try:
                names = ws.deliverable_paths()
            except Exception:
                names = []
            for name in sorted(
                n for n in (names or []) if not os.path.basename(n).startswith(".")
            ):
                # deliverable_paths() may return either basenames or full paths.
                candidate = (
                    name if os.path.isabs(name) else ws.deliverable_path(name)
                )
                if candidate and os.path.isfile(candidate):
                    return candidate

        # Tier 2: outputs/<basename>
        try:
            out_path = ws.output_path(out_basename) if hasattr(ws, "output_path") else None
        except Exception:
            out_path = None
        if out_path and os.path.isfile(out_path):
            return out_path

        return None

    def _record_round_audit(self, round_idx, phase, inferencer, extra=None,
                            workspace_root_at_phase=None):
        """Record a single round phase to structured log + navigation symlink.

        Fail-safe: exceptions are logged but never propagate — audit must
        not crash the inference run.

        Args:
            round_idx: The round number for this audit entry.
            phase: The phase label (e.g. "propose", "review", "fix").
            inferencer: The child inferencer whose workspace is being recorded.
            extra: Optional dict of additional fields for the log entry.
            workspace_root_at_phase: Optional snapshot of the inferencer's
                workspace root captured BEFORE ainfer(). When provided, this
                is used as the symlink target instead of the live
                ``inferencer._workspace.root`` — which may have been mutated
                by role-reassignment during the call. Defaults to None
                (falls back to live read for backward compatibility).
        """
        if not self.enable_round_audit or self._workspace is None:
            return
        if workspace_root_at_phase is None and getattr(inferencer, "_workspace", None) is None:
            return

        try:
            import json as _json
            from datetime import datetime as _dt

            # Fix #8: use snapshot if provided, else live read
            target = workspace_root_at_phase or str(inferencer._workspace.root)

            log_entry = {
                "round": round_idx,
                "phase": phase,
                "inferencer_class": type(inferencer).__name__,
                "inferencer_workspace": target,
                "timestamp": _dt.utcnow().isoformat(),
                **(extra or {}),
            }
            outputs_dir = getattr(self._workspace, "outputs_dir", None)
            if outputs_dir:
                log_path = os.path.join(outputs_dir, "round_log.jsonl")
                os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(log_entry) + "\n")

            # Cross-worker leakage detection (log-only, no filesystem nav symlinks).
            children_dir = getattr(self._workspace, "children_dir", None)
            if children_dir:
                my_root = str(self._workspace.root).rstrip("/") + "/"
                if not target.startswith(my_root):
                    logger.error(
                        "Audit: cross-worker leakage at round_%02d/%s: "
                        "target %s outside %s",
                        round_idx, phase, target, self._workspace.root,
                    )
        except Exception as exc:
            logger.warning("Round audit for %s/%s failed: %s", round_idx, phase, exc)

    # region Sync/Async Bridge

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        """Sync bridge — delegates to _ainfer() via _run_async()."""
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(
            self._ainfer(inference_input, inference_config, **_inference_args)
        )

    # endregion

    # region Core Consensus Loop

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        """Async inference — core consensus loop using step_configs + Workflow._arun().

        Multi-attempt outer loop: each attempt rebuilds step_configs and runs
        the Workflow via WorkflowStepConfig declarations + _build_steps().

        Args:
            inference_input: The original task/request prompt.
            inference_config: Optional dict with overrides.
            **_inference_args: Additional args passed to sub-inferencers.

        Returns:
            DualInferencerResponse with consensus history and final proposal.
        """
        if inference_config is None:
            inference_config = {}
        elif not isinstance(inference_config, Mapping):
            raise ValueError("'inference_config' must be a mapping")

        config = inference_config.get("consensus_config", self.consensus_config)
        phase = inference_config.get("phase", self.phase)

        all_attempt_records: List[ConsensusAttemptRecord] = []
        final_output = None
        final_review = None
        consensus_achieved = False
        total_iterations = 0

        for attempt in range(1, config.max_consensus_attempts + 1):
            if attempt > 1:
                await self._areset_sub_inferencers()

            logger.info(
                "[%s] Starting consensus attempt %d/%d",
                phase or "DualInferencer",
                attempt,
                config.max_consensus_attempts,
            )

            # Per-run transient working state — virtualized into the active ctx
            # (Commit 3 audit), with an instance backing under legacy/no-ctx.
            # NOT in self._state (not picklable); re-derived each call.
            self._run_set("_current_attempt", attempt)
            self._current_config = config

            self._run_set("_current_inference_config", inference_config)
            self._run_set("_current_extra_inference_args", _inference_args)

            # Build step_configs for this attempt
            self._build_step_configs_for_attempt(config, attempt)

            # Set pending state (LWI convention — uses _pending_state)
            self._pending_state = {
                "inference_input": inference_input,
                "base_output_str": None,
                "counter_feedback_str": None,
                "consensus_reached": False,
                "attempt_record": {
                    "attempt": attempt,
                    "iterations": [],
                    "consensus_reached": False,
                    "final_output": None,
                    "final_feedback": None,
                },
                "total_iterations": total_iterations,
                "consensus_iteration": 0,
                "parsed_review": None,
                "review_output_str": None,
                "review_prompt": None,
                "_consensus_threshold": config.consensus_threshold,
            }

            # Support initial_response_override: skip propose, start at review.
            initial_override = inference_config.get("initial_response_override")
            if initial_override is not None:
                self._pending_state["base_output_str"] = (
                    self._maybe_replace_with_file_reference(
                        initial_override,
                        round_index=0,
                        inference_config=inference_config,
                    )
                )
                self._pending_state["counter_feedback_str"] = None
                self._pending_state["consensus_iteration"] = 0

                # Replace propose step_fn with a passthrough
                pending = dict(self._pending_state)

                async def _initial_plan_passthrough(step_input, state):
                    self._state = pending
                    return pending

                self.step_configs[0] = WorkflowStepConfig(
                    name="propose",
                    step_fn=_initial_plan_passthrough,
                )

            # Configure checkpoint if enabled
            if config.max_iterations <= 0:
                # Propose-only mode: no resume needed (single step)
                self.enable_result_save = False
                self.resume_with_saved_results = False
            elif self.enable_checkpoint and (
                self._workspace is not None or self.checkpoint_dir
            ):
                self.enable_result_save = StepResultSaveOptions.Always
                self.resume_with_saved_results = True
            elif self._result_root_override is not None:
                pass  # Parent already configured via _setup_child_workflows
            else:
                self.enable_result_save = False
                self.resume_with_saved_results = False

            # Build steps from step_configs and run
            self._steps = self._build_steps()
            await Workflow._arun(self, inference_input, **_inference_args)

            # Extract results from self._state
            state = self._state or {}
            attempt_record_dict = state.get("attempt_record", {})

            attempt_record = ConsensusAttemptRecord(
                attempt=attempt_record_dict.get("attempt", attempt),
            )
            # Rebuild iteration records from the dict data
            for iter_rec in attempt_record_dict.get("iterations", []):
                if isinstance(iter_rec, ConsensusIterationRecord):
                    attempt_record.iterations.append(iter_rec)
                elif isinstance(iter_rec, dict):
                    attempt_record.iterations.append(
                        ConsensusIterationRecord(**iter_rec)
                    )
                else:
                    attempt_record.iterations.append(iter_rec)

            attempt_record.consensus_reached = attempt_record_dict.get(
                "consensus_reached", False
            )
            attempt_record.final_output = attempt_record_dict.get("final_output")
            attempt_record.final_feedback = attempt_record_dict.get("final_feedback")

            total_iterations = state.get("total_iterations", total_iterations)

            if not attempt_record.consensus_reached:
                attempt_record.final_output = state.get("base_output_str")
                final_output = state.get("base_output_str")
            else:
                final_output = attempt_record.final_output
                consensus_achieved = True

            all_attempt_records.append(attempt_record)
            if consensus_achieved:
                break

        # Build final review if none captured
        if (
            final_review is None
            and all_attempt_records
            and all_attempt_records[-1].iterations
        ):
            last_iter = all_attempt_records[-1].iterations[-1]
            review_input = getattr(last_iter, "review_input", None)
            review_output = getattr(last_iter, "review_output", None)
            if review_input is not None:
                final_review = InputAndResponse(
                    input=review_input, response=review_output
                )

        # Check if consensus achieved via last attempt's last iteration
        if not consensus_achieved and all_attempt_records:
            last_attempt = all_attempt_records[-1]
            if last_attempt.consensus_reached and last_attempt.final_output:
                final_output = last_attempt.final_output
                consensus_achieved = True
                # Extract review from last iteration
                if last_attempt.iterations:
                    last_iter = last_attempt.iterations[-1]
                    review_input = getattr(last_iter, "review_input", None)
                    review_output = getattr(last_iter, "review_output", None)
                    if review_input is not None:
                        final_review = InputAndResponse(
                            input=review_input, response=review_output
                        )

        logger.info(
            "[%s] Consensus loop complete: achieved=%s, total_iterations=%d, attempts=%d",
            phase or "DualInferencer",
            consensus_achieved,
            total_iterations,
            len(all_attempt_records),
        )

        # Copy last round artifact to outputs/ (workspace mode only)
        self._finalize_response()

        return DualInferencerResponse(
            base_response=final_output or "",
            reflection_response=final_review,
            reflection_style=ReflectionStyles.Sequential,
            response_selector=self.response_selector,
            consensus_history=all_attempt_records,
            total_iterations=total_iterations,
            consensus_achieved=consensus_achieved,
            phase=phase,
        )

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _build_step_configs_for_attempt(self, config, attempt):
        """Build step_configs for a single consensus attempt.

        Creates WorkflowStepConfig entries:
        - max_iterations > 0: 3 steps (propose/review/fix) with fix
          looping back to review up to (max_iterations - 1) times
        - max_iterations <= 0: 1 step (propose only)
        """
        if config.max_iterations <= 0:
            self.step_configs = [
                WorkflowStepConfig(
                    name="propose",
                    step_fn=self._step_propose_impl,
                ),
            ]
        else:
            def _check_loop_condition(state, result):
                if state is None:
                    self.log_warning(
                        {
                            "message": "loop_condition received None state (expected dict), "
                            "falling back to self._state",
                            "self_state_type": type(self._state).__name__,
                            "self_state_keys": list((self._state or {}).keys()),
                        },
                        log_type="StateWarning",
                    )
                effective_state = state if state is not None else (self._state or {})
                return not effective_state.get("consensus_reached", False)

            self.step_configs = [
                WorkflowStepConfig(
                    name="propose",
                    step_fn=self._step_propose_impl,
                ),
                WorkflowStepConfig(
                    name="review",
                    step_fn=self._step_review_impl,
                ),
                WorkflowStepConfig(
                    name="fix",
                    step_fn=self._step_fix_impl,
                    loop_back_to="review",
                    loop_condition=_check_loop_condition,
                    max_loop_iterations=config.max_iterations - 1,
                ),
            ]

    async def _step_propose_impl(self, step_input, state):
        """Propose step — calls base_inferencer to generate initial proposal.

        Uses self._state directly.
        """
        self._check_cancelled()  # §2.1/P-#6: halt at the consensus-step boundary
        state = self._state

        if self.initial_prompt is not None:
            initial_prompt = self._build_initial_prompt(
                state["inference_input"],
                self._run_get("_current_inference_config", {}),
                attempt=state["attempt_record"]["attempt"],
            )
        else:
            initial_prompt = state["inference_input"]

        _sf = f"Round{state['total_iterations'] + 1:02d}"
        self.log_info(
            initial_prompt,
            "InitialPrompt",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=_sf,
        )

        # M7: the canonical propose child is ``self._workspace.child("propose")`` — the
        # base is always dispatched at ``_rc_child("propose")``, so this equals where it
        # wrote. Byte-identical to the old bare ``base._workspace`` read in legacy;
        # correct under a ctx (the bare base read, with the instance backing left None
        # by write-purity, would fall through to the worker ROOT). Snapshot its root
        # BEFORE ainfer() for the round audit.
        _eff_propose_ws = (
            self._workspace.child("propose") if self._workspace is not None else None
        )
        _propose_ws_snapshot = (
            str(_eff_propose_ws.root) if _eff_propose_ws is not None else None
        )
        _raw_base = str(
            await self.base_inferencer.ainfer(
                initial_prompt,
                run_context=self._rc_child("propose"),
                **self._run_get("_current_extra_inference_args", {}),
            )
        )
        # Track canonical output child for _finalize_output symlink. Bare assignment
        # routes through the compat-properties → ctx scratch under a ctx (concurrency-
        # isolated), instance backing under legacy.
        if _eff_propose_ws is not None:
            self._last_output_child_ws = _eff_propose_ws
            self._propose_child_ws = _eff_propose_ws
        _sf = f"Round{state['total_iterations'] + 1:02d}"
        self.log_debug(
            _raw_base,
            "RawBaseResponse",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=_sf,
        )
        base_output_str = self.response_parser(_raw_base)
        if base_output_str is None:
            raise InferencerExecutionError(
                tool=f"<{self.phase or 'DualInferencer'}.propose>",
                error=(
                    "response_parser returned None — model emitted no clean "
                    "<Response> block (likely truncated by max_tokens, deliberately "
                    "skipped after a tool call, or prompt-template echo). "
                    f"raw_size_bytes={len(_raw_base)}"
                ),
            )
        base_output_str = self._maybe_replace_with_file_reference(
            base_output_str,
            round_index=0,
            inference_config=self._run_get("_current_inference_config", {}),
        )
        _sf = f"Round{state['total_iterations'] + 1:02d}"
        self.log_info(
            base_output_str,
            "InitialResponse",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=_sf,
        )

        state["base_output_str"] = base_output_str
        state["counter_feedback_str"] = None
        state["consensus_iteration"] = 0
        self._state = state
        self._record_round_audit(
            state["total_iterations"] + 1, "propose", self.base_inferencer,
            workspace_root_at_phase=_propose_ws_snapshot,
        )
        return base_output_str

    async def _step_review_impl(self, step_input, state):
        """Review step — calls review_inferencer, raises WorkflowAborted on consensus.

        Uses ``consensus_iteration`` instead of ``iteration`` to avoid
        triggering LWI's ``_setup_iteration()`` which creates
        per-iteration workspace subdirectories.

        Checkpoint compat shim: reads with fallback
        ``state.get("consensus_iteration", state.get("iteration", 0))``
        so old checkpoints containing ``"iteration"`` resume correctly.
        """
        self._check_cancelled()  # §2.1/P-#6: halt at the review-step boundary
        state = self._state
        if state is None:
            state = dict(self._pending_state)
            self._state = state

        # Read with fallback for old checkpoint compat
        consensus_iter = state.get("consensus_iteration", state.get("iteration", 0))
        consensus_iter += 1
        state["consensus_iteration"] = consensus_iter
        state["total_iterations"] = state.get("total_iterations", 0) + 1
        iteration = consensus_iter
        total_iters = state["total_iterations"]
        attempt_num = state["attempt_record"]["attempt"]

        # Context path for the primary reviewer — independent of workspace.
        # Single reviewer: review/. Multi-reviewer panel: review/panelist_00/.
        from agent_foundation.common.inferencers.inferencer_workspace import indexed_child_name
        _review_child = self._rc_child("review")
        _has_panel = bool(self._role_get("reviewers"))
        if _has_panel and _review_child is not None:
            _review_child = _review_child.child(indexed_child_name("panelist", 0))

        # Per-round workspace (orthogonal to context path).
        if self._workspace is not None and self._role_get("review_inferencer") is not None:
            round_ws = self._workspace.child(indexed_child_name("round", consensus_iter))
            if _has_panel:
                review_ws = round_ws.child("review").child(indexed_child_name("panelist", 0))
            else:
                review_ws = round_ws.child("review")
            review_ws.ensure_dirs()
            self._publish_workspace_to_ctx(_review_child, review_ws)
            self._run_set("_current_round_ws", round_ws)

        logger.info(
            "[%s] ROUND_TRACE inner_loop_top: iteration=%d, total_iterations=%d",
            self.phase or "DualInferencer",
            iteration,
            total_iters,
        )

        try:
            # Phase 2 (leaf-owned template rendering): route between
            # leaf-side rendering (modern) and orchestrator-side rendering
            # (legacy). When the review leaf can self-render, pass it the
            # raw feed dict via extra_feed= and let it render its own
            # template — eliminates the doubly-templated rendering bug.
            review_feed = self._build_review_feed(
                state["inference_input"],
                state["base_output_str"],
                state["counter_feedback_str"],
                iteration=iteration,
                attempt=attempt_num,
                inference_config=self._run_get("_current_inference_config", {}),
            )
            review_leaf_can_render = self._leaf_can_self_render(
                self._role_get("review_inferencer")
            )
            _review_extra_feed = {
                k: v
                for k, v in review_feed.items()
                if k not in ("input", "__template_space__")
            }
            if review_leaf_can_render:
                # §2.7: render this logging preview under the SAME ./review ctx the
                # dispatch uses (run_context=self._rc_child("review")), so the leaf's
                # _effective_role claims ./review (its own node) and reads the role's
                # RoleState there — NOT the active worker node, which it would tag with
                # this leaf's creator and then collide with the fixer's worker-node
                # pre-render. Bridge is no-op-safe when no ctx is active.
                from agent_foundation.common.inferencers.run_context import (
                    enter_run as _af_enter_run,
                    exit_run as _af_exit_run,
                )

                _rv_rc = _review_child
                _rv_tok = _af_enter_run(_rv_rc) if _rv_rc is not None else None
                try:
                    review_prompt = self._role_get("review_inferencer")._render_prompt(
                        state["inference_input"],
                        extra_feed=_review_extra_feed,
                    )
                finally:
                    if _rv_tok is not None:
                        _af_exit_run(_rv_tok)
            else:
                # Legacy path: orchestrator renders.
                review_prompt = self._render_role_prompt(
                    "review",
                    review_feed,
                    self._run_get("_current_inference_config", {}),
                )
        except _RoleDisabledError:
            # No review template available (and caller didn't explicitly
            # configure one) → degrade gracefully to single-shot proposal:
            # mark consensus reached, treat the base output as final.
            logger.info(
                "[%s] Review role is disabled (no template available) — "
                "skipping review step and treating base output as final.",
                self.phase or "DualInferencer",
            )
            iteration_record = ConsensusIterationRecord(
                iteration=iteration,
                base_output=state["base_output_str"],
                review_input="",
                review_output="",
                review_feedback={"approved": True, "severity": "NONE",
                                 "issues": [], "reasoning": "Review role disabled."},
                consensus_reached=True,
            )
            state["attempt_record"]["iterations"].append(iteration_record)
            state["attempt_record"]["consensus_reached"] = True
            state["attempt_record"]["final_output"] = state["base_output_str"]
            state["attempt_record"]["final_feedback"] = iteration_record.review_feedback
            state["consensus_reached"] = True
            self._last_iteration_record = iteration_record
            self._state = state
            raise WorkflowAborted()
        _sf = f"Round{total_iters:02d}"
        self.log_info(
            review_prompt,
            "ReviewPrompt",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=_sf,
        )

        # Phase 2: dual invocation paths.
        #   Modern (leaf-renders): pass raw input + extra_feed; leaf templates.
        #   Legacy (orchestrator-renders): review_prompt is already rendered;
        #     pass it as the input directly (leaf's _render_prompt is no-op
        #     because it has no template_manager / no template_key — Phase 1d
        #     loud-failure does NOT trigger because template_manager is None).

        # Fix #8: snapshot workspace root BEFORE ainfer() — role reassignment
        # during the call may mutate _workspace.root on the inferencer.
        # In panel mode, the primary reviewer's workspace is at _review_child
        # (e.g. /review/panelist_00), not the bare /review slot.
        _eff_review_ws = None
        if _review_child is not None:
            _eff_review_ws = getattr(_review_child.handles, "get", lambda *a: None)(
                "workspace_override"
            ) if hasattr(_review_child, "handles") else None
        if _eff_review_ws is None:
            _eff_review_ws = self._read_child_workspace(
                self._role_get("review_inferencer"), "review"
            )
        _review_ws_snapshot = (
            str(_eff_review_ws.root) if _eff_review_ws is not None else None
        )
        if review_leaf_can_render:
            # Reuse _review_extra_feed computed above for the logging prompt —
            # same dict, no re-computation, no re-rendering by Dual.
            # The leaf's _render_prompt fires once inside ainfer().
            _raw_review = str(
                await self._role_get("review_inferencer").ainfer(
                    state["inference_input"],
                    run_context=_review_child,
                    extra_feed=_review_extra_feed,
                    **self._run_get("_current_extra_inference_args", {}),
                )
            )
        else:
            _raw_review = str(
                await self._role_get("review_inferencer").ainfer(
                    review_prompt,
                    run_context=_review_child,
                    **self._run_get("_current_extra_inference_args", {}),
                )
            )
        _sf = f"Round{total_iters:02d}"
        self.log_debug(
            _raw_review,
            "RawReviewResponse",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=_sf,
        )

        review_output_str = self.response_parser(_raw_review)
        if review_output_str is None:
            raise InferencerExecutionError(
                tool=f"<{self.phase or 'DualInferencer'}.review>",
                error=(
                    "response_parser returned None — model emitted no clean "
                    "<Response> block (likely truncated by max_tokens, deliberately "
                    "skipped after a tool call, or prompt-template echo). "
                    f"raw_size_bytes={len(_raw_review)}"
                ),
            )
        parsed_review = self.review_parser(review_output_str)
        # §3 Part B: multi-reviewer panel — run k reviewers and merge the parsed
        # reviews. Effective k is ``1 + len(reviewers)`` when a declarative
        # ``reviewers`` panel is supplied (``review_inferencer`` is panelist 0), else
        # ``num_reviewers``. No-op when k<=1 (byte-identical single-reviewer path).
        _panel_extra = (
            list(self._role_get("reviewers")) if self._role_get("reviewers") else None
        )
        _panel_k = (
            1 + len(_panel_extra)
            if _panel_extra and self.num_reviewers <= 1
            else self.num_reviewers
        )
        if _panel_k and _panel_k > 1:
            from agent_foundation.common.inferencers.flow_parsers import (
                merge_reviews,
            )

            # §3: each additional panelist runs under its OWN child node
            # (review/panelist_i) so its state + Tier-3 live handles are isolated
            # from the first reviewer and from each other (no session/subprocess
            # cross-talk when the reviewer is a live-handle leaf). When ``reviewers``
            # is set, panelist i is an INDEPENDENT instance (its own definition
            # state); else the single ``review_inferencer`` is reused per child ctx.
            # No child without an active ctx => legacy-mint, byte-identical.
            _panel = [parsed_review]
            from agent_foundation.common.inferencers.inferencer_workspace import indexed_child_name as _icn
            self._record_round_audit(
                total_iters, "review", self._role_get("review_inferencer"),
                workspace_root_at_phase=_review_ws_snapshot,
                extra={
                    "audit_kind": "panelist",
                    "panelist": _icn("panelist", 0),
                    "approved": parsed_review.get("approved"),
                    "severity": parsed_review.get("severity"),
                    "num_issues": len(parsed_review.get("issues", [])),
                },
            )
            _rev_parent = self._rc_child("review")
            _round_ws = self._run_get("_current_round_ws")
            for _i in range(1, _panel_k):
                _panelist = (
                    _panel_extra[(_i - 1) % len(_panel_extra)]
                    if _panel_extra
                    else self._role_get("review_inferencer")
                )
                from agent_foundation.common.inferencers.inferencer_workspace import indexed_child_name
                _pname = indexed_child_name("panelist", _i)
                _panelist_ctx = (
                    _rev_parent.child(_pname)
                    if _rev_parent is not None
                    else None
                )
                if _round_ws is not None:
                    _panelist_ws = _round_ws.child("review").child(_pname)
                    _panelist_ws.ensure_dirs()
                    self._publish_workspace_to_ctx(_panelist_ctx, _panelist_ws)
                if self._leaf_can_self_render(_panelist):
                    _rk = str(
                        await _panelist.ainfer(
                            state["inference_input"],
                            run_context=_panelist_ctx,
                            extra_feed=_review_extra_feed,
                            **self._run_get("_current_extra_inference_args", {}),
                        )
                    )
                else:
                    _rk = str(
                        await _panelist.ainfer(
                            review_prompt,
                            run_context=_panelist_ctx,
                            **self._run_get("_current_extra_inference_args", {}),
                        )
                    )
                _rk_out = self.response_parser(_rk)
                if _rk_out is not None:
                    _parsed_panelist = self.review_parser(_rk_out)
                    _panel.append(_parsed_panelist)
                else:
                    _parsed_panelist = {"approved": False, "issues": []}
                    _panel.append(_parsed_panelist)
                _panelist_ws_root = None
                if _panelist_ctx is not None:
                    _pw = getattr(_panelist_ctx.handles, "get", lambda *a: None)(
                        "workspace_override"
                    ) if hasattr(_panelist_ctx, "handles") else None
                    if _pw is not None:
                        _panelist_ws_root = str(_pw.root)
                self._record_round_audit(
                    total_iters, "review", _panelist,
                    workspace_root_at_phase=_panelist_ws_root,
                    extra={
                        "audit_kind": "panelist",
                        "panelist": _pname,
                        "approved": _parsed_panelist.get("approved"),
                        "severity": _parsed_panelist.get("severity"),
                        "num_issues": len(_parsed_panelist.get("issues", [])),
                    },
                )
            _aggregator = self.review_aggregator or merge_reviews
            _merged = _aggregator(_panel)
            # Consensus only when EVERY panelist approves; issues are the merged
            # (deduped, never-downgraded) union — the schema _default_check_consensus
            # expects. Other top-level fields carry over from the first review.
            parsed_review = dict(parsed_review)
            parsed_review["issues"] = _merged.get(
                "issues", parsed_review.get("issues", [])
            )
            parsed_review["approved"] = all(
                bool(r.get("approved")) for r in _panel
            )
            # Never-downgrade the top-level severity: it becomes the WORST across the
            # panel, so the consensus check's severity fallback (approved=False ->
            # accept when severity is within threshold) can't be fooled by panelist
            # 0's severity when another panelist flagged a higher one.
            _levels = self.consensus_config.severity_levels or []
            _rank = {s: i for i, s in enumerate(_levels)}
            _panel_sevs = [
                r.get("severity") for r in _panel if r.get("severity") in _rank
            ]
            if _panel_sevs:
                parsed_review["severity"] = max(_panel_sevs, key=_rank.get)
        self.log_info(
            review_output_str,
            "ReviewResponse",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=_sf,
        )
        parsed_review = self._assign_issue_ids(parsed_review, iteration)

        threshold = state.get(
            "_consensus_threshold", self.consensus_config.consensus_threshold
        )
        reached = self.consensus_checker(parsed_review, threshold)

        self.log_info(
            {
                "iteration": iteration,
                "consensus_reached": reached,
                "approved": parsed_review.get("approved"),
                "severity": parsed_review.get("severity", "UNKNOWN"),
                "threshold": threshold,
                "num_issues": len(parsed_review.get("issues", [])),
                "issue_severities": [i.get("severity") for i in parsed_review.get("issues", [])],
            },
            "ReviewIteration",
        )

        iteration_record = ConsensusIterationRecord(
            iteration=iteration,
            base_output=state["base_output_str"],
            review_input=review_prompt,
            review_output=review_output_str,
            review_feedback=parsed_review,
            consensus_reached=reached,
        )

        state["parsed_review"] = parsed_review
        state["review_output_str"] = review_output_str
        state["review_prompt"] = review_prompt
        state["consensus_reached"] = reached
        self._last_iteration_record = iteration_record
        self._state = state

        self._record_round_audit(
            total_iters, "review", self._role_get("review_inferencer"),
            workspace_root_at_phase=_review_ws_snapshot,
            extra={
                "audit_kind": "merged" if _has_panel else "review",
                "consensus_reached": reached,
                "severity": parsed_review.get("severity"),
                "approved": parsed_review.get("approved"),
                "num_issues": len(parsed_review.get("issues", [])),
                "iteration": iteration,
            },
        )

        if reached:
            state["attempt_record"]["iterations"].append(iteration_record)
            state["attempt_record"]["consensus_reached"] = True
            state["attempt_record"]["final_output"] = state["base_output_str"]
            state["attempt_record"]["final_feedback"] = parsed_review
            raise WorkflowAborted()

        return review_output_str

    async def _step_fix_impl(self, step_input, state):
        """Fix step — calls fixer_inferencer, updates proposal in state.

        Uses ``consensus_iteration`` instead of ``iteration`` and
        ``self._pending_state`` for checkpoint state sync.
        """
        self._check_cancelled()  # §2.1/P-#6: halt at the fix-step boundary
        state = self._state
        # Read with fallback for old checkpoint compat
        iteration = state.get("consensus_iteration", state.get("iteration", 0))
        total_iters = state["total_iterations"]
        attempt_num = state["attempt_record"]["attempt"]
        parsed_review = state["parsed_review"]

        # Per-round workspace: assign fixer to round_NN/children/fix/
        round_ws = self._run_get("_current_round_ws", None)
        if round_ws is not None and self._role_get("fixer_inferencer") is not None:
            fix_ws = round_ws.child("fix")
            fix_ws.ensure_dirs()
            # M7 workspace virtualization (see review step): publish to the fix
            # child's context + keep the instance assignment as the fallback.
            _fix_child = self._rc_child("fix")
            self._publish_workspace_to_ctx(_fix_child, fix_ws)
            if _fix_child is None:
                self._role_get("fixer_inferencer")._workspace = fix_ws  # legacy (byte-identical)

        self.log_info(
            {
                "event": "FIX_STEP_ENTER",
                "iteration": iteration,
                "fixer": type(self._role_get("fixer_inferencer")).__name__,
            },
            "FixStep",
        )
        try:
            # Phase 2 (leaf-owned template rendering): build feed dict, then
            # route between leaf-side rendering (modern) and orchestrator-
            # side rendering (legacy). See _step_review_impl for symmetry.
            followup_feed = self._build_followup_feed(
                state["inference_input"],
                state["base_output_str"],
                parsed_review,
                getattr(self, "_current_inference_config", {}),
                iteration=iteration,
                attempt=attempt_num,
                review_output=state.get("review_output_str"),
            )
            fixer_leaf_can_render = self._leaf_can_self_render(
                self._role_get("fixer_inferencer")
            )
            if fixer_leaf_can_render:
                # Modern path: leaf renders. Pre-render for logging via
                # _render_prompt directly (avoids double-render — see
                # review step for rationale).
                _fixer_extra_feed = {
                    k: v
                    for k, v in followup_feed.items()
                    if k not in ("input", "__template_space__")
                }
                # §2.7: render this logging preview under the SAME ./fix ctx the
                # dispatch uses (run_context=self._rc_child("fix")), so the fixer leaf
                # claims ./fix (its own node) and reads the role's RoleState there —
                # not the active worker node (which the reviewer already tagged).
                from agent_foundation.common.inferencers.run_context import (
                    enter_run as _af_enter_run,
                    exit_run as _af_exit_run,
                )

                _fx_rc = self._rc_child("fix")
                _fx_tok = _af_enter_run(_fx_rc) if _fx_rc is not None else None
                try:
                    followup_prompt = self._role_get("fixer_inferencer")._render_prompt(
                        state["inference_input"],
                        extra_feed=_fixer_extra_feed,
                    )
                finally:
                    if _fx_tok is not None:
                        _af_exit_run(_fx_tok)
            else:
                # Legacy path: orchestrator renders.
                followup_prompt = self._render_role_prompt(
                    "followup",
                    followup_feed,
                    getattr(self, "_current_inference_config", {}),
                )
        except _RoleDisabledError as _rde:
            self.log_warning(
                {
                    "event": "FIX_STEP_ROLE_DISABLED",
                    "message": (
                        "Fixer role disabled — fix step skipped. "
                        "consensus_reached forced to True."
                    ),
                    "fixer": type(self._role_get("fixer_inferencer")).__name__,
                    "error": str(_rde),
                },
                "FixStepSkipped",
            )
            iteration_record = getattr(self, "_last_iteration_record", None)
            if iteration_record is not None:
                state["attempt_record"]["iterations"].append(iteration_record)
            state["attempt_record"]["consensus_reached"] = True
            state["attempt_record"]["final_output"] = state["base_output_str"]
            state["consensus_reached"] = True
            self._state = state
            self._pending_state = dict(state)
            raise WorkflowAborted()
        self.log_info(
            followup_prompt,
            "FollowupPrompt",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=f"Round{total_iters:02d}",
        )

        # Phase 2: dual invocation paths (see _step_review_impl).

        # Fix #8: snapshot workspace root BEFORE ainfer() — role reassignment
        # during the call may mutate _workspace.root on the inferencer.
        _eff_fix_ws = self._read_child_workspace(self._role_get("fixer_inferencer"), "fix")
        _fix_ws_snapshot = (
            str(_eff_fix_ws.root) if _eff_fix_ws is not None else None
        )
        if fixer_leaf_can_render:
            # Reuse _fixer_extra_feed computed above — same dict, one render.
            _raw_fix = str(
                await self._role_get("fixer_inferencer").ainfer(
                    state["inference_input"],
                    run_context=self._rc_child("fix"),
                    extra_feed=_fixer_extra_feed,
                    **self._run_get("_current_extra_inference_args", {}),
                )
            )
        else:
            _raw_fix = str(
                await self._role_get("fixer_inferencer").ainfer(
                    followup_prompt,
                    run_context=self._rc_child("fix"),
                    **self._run_get("_current_extra_inference_args", {}),
                )
            )
        # Track canonical output child for _finalize_output symlink
        _eff_fix_ws_out = self._read_child_workspace(self._role_get("fixer_inferencer"), "fix")
        if _eff_fix_ws_out is not None:
            self._run_set("_last_output_child_ws", _eff_fix_ws_out)
        self.log_debug(
            _raw_fix,
            "RawFixResponse",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=f"Round{total_iters:02d}",
        )

        fix_output_str = self.response_parser(_raw_fix)
        if fix_output_str is None:
            raise InferencerExecutionError(
                tool=f"<{self.phase or 'DualInferencer'}.fix>",
                error=(
                    "response_parser returned None — model emitted no clean "
                    "<Response> block (likely truncated by max_tokens, deliberately "
                    "skipped after a tool call, or prompt-template echo). "
                    f"raw_size_bytes={len(_raw_fix)}"
                ),
            )
        self.log_info(
            fix_output_str,
            "FollowupResponse",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=f"Round{total_iters:02d}",
        )
        parsed_counter = self.followup_response_parser(fix_output_str)
        counter_feedback_str = (
            json.dumps(parsed_counter, indent=2)
            if parsed_counter.get("items")
            else None
        )
        improved_proposal = fix_output_str

        iteration_record = getattr(self, "_last_iteration_record", None)
        if iteration_record is not None:
            iteration_record.counter_feedback = parsed_counter
            state["attempt_record"]["iterations"].append(iteration_record)

        state["counter_feedback_str"] = counter_feedback_str
        state["base_output_str"] = self._maybe_replace_with_file_reference(
            improved_proposal,
            round_index=iteration,
            inference_config=getattr(self, "_current_inference_config", {}),
        )
        self._state = state
        self._pending_state = dict(state)
        self._record_round_audit(
            total_iters, "fix", self._role_get("fixer_inferencer"),
            workspace_root_at_phase=_fix_ws_snapshot,
            extra={
                "iteration": iteration_record.iteration if iteration_record else None,
                "has_counter_feedback": counter_feedback_str is not None,
            },
        )
        return fix_output_str

    # endregion

    # region Prompt Builders

    def _is_role_disabled(self, role: str) -> bool:
        """Check whether a role (review/followup) is disabled — i.e. caller
        provided no explicit prompt and the implicit default key was not found
        when ``_render_role_prompt`` last attempted to resolve it.

        Used by ``_step_review_impl`` and ``_step_fix_impl`` to skip steps
        gracefully when no template is available.
        """
        return getattr(self, f"_{role}_role_disabled", False)

    def _render_role_prompt(self, role: str, feed: dict, inference_config: dict) -> str:
        """Render a prompt for a role using an explicit Jinja2 string.

        This is the fallback path — only reached when the leaf inferencer
        cannot self-render (no template_manager or template_key). In the
        modern (Phase 2) architecture, leaves self-render and this method
        is not called.

        Raises:
            _RoleDisabledError: if no prompt string is configured for this role.
        """
        post_process = partial(unescape_xml, unescape_for_html=True)
        prompt_value = getattr(self, f"{role}_prompt", None)

        if self._is_role_disabled(role):
            raise _RoleDisabledError(role)

        if prompt_value is None:
            raise _RoleDisabledError(role)

        from jinja2 import Template
        rendered = Template(prompt_value).render(**feed)
        return post_process(rendered)

    # ─────────────────────────────────────────────────────────────────
    # Phase 2 (leaf-owned template rendering): leaf-side rendering helpers.
    # ─────────────────────────────────────────────────────────────────
    # Architectural direction: orchestrators own workflow & feed-dict
    # assembly; leaves own template selection & rendering. The fallback path
    # (_render_role_prompt) renders raw Jinja2 strings at the orchestrator
    # level; the modern path passes a feed dict to the leaf via extra_feed=
    # and lets the leaf's own _build_template_feed merge it.
    #
    # Detection: a leaf "can self-render" iff:
    #   1. It is a TemplatedInferencerBase subclass (or duck-types it via
    #      having both `template_manager` and `_render_prompt` attributes),
    #   2. Its `template_manager` attribute is non-None (so it knows how
    #      to resolve templates), AND
    #   3. It has `template_key` or `template_root_space` set (so there's
    #      an actual template to render — Phase 2 wiring of
    #      FOLLOWUP_TEMPLATE_DEFAULTS / REVIEW_TEMPLATE_DEFAULTS via
    #      SLOT_DEFAULTS guarantees this for unconfigured slots).
    #
    # When a leaf cannot self-render (legacy custom-formatter path, leaves
    # without template_manager, etc.), Dual falls back to its own
    # _render_role_prompt using the merged feed dict.

    @staticmethod
    def _leaf_can_self_render(leaf) -> bool:
        """True iff the given child inferencer can render its own template.

        Strict check: must be a TemplatedInferencerBase subclass with both
        template_manager AND (template_key OR template_root_space) set.
        Importantly, isinstance is used here rather than duck-typing on
        attribute presence: this avoids false positives on test mocks
        (MagicMock makes getattr return a Mock for any attribute), and
        keeps the semantic crisp ("only inferencers explicitly designed
        for templating self-render").

        See class-level Phase 2 comment for full semantics.
        """
        # Avoid a circular import at module top by importing lazily here.
        from agent_foundation.common.inferencers.templated_inferencer_base import (
            TemplatedInferencerBase,
        )
        if leaf is None:
            return False
        if not isinstance(leaf, TemplatedInferencerBase):
            return False
        if leaf.template_manager is None:
            return False
        if not leaf.template_key and not leaf.template_root_space:
            return False
        return True

    def _build_initial_prompt(
        self,
        inference_input,
        inference_config: dict,
        attempt: int = 1,
    ) -> str:
        """Build the initial prompt from template."""
        feed = {
            self.placeholder_input: inference_input,
            "iteration": 0,
            "attempt": attempt,
            "round_index": 0,
        }
        return self._render_role_prompt("initial", feed, inference_config)

    def _build_review_feed(
        self,
        inference_input,
        proposal: str,
        counter_feedback: Optional[str],
        iteration: int = 1,
        attempt: int = 1,
        inference_config: Optional[dict] = None,
    ) -> dict:
        """Build the review feed dict (Phase 2: pure feed assembly).

        Returns the feed dict consumed by either:
          * Leaf-side rendering — ``review_inferencer.ainfer(input,
            extra_feed=feed)`` — modern path.
          * Orchestrator-side rendering — ``_render_role_prompt("review",
            feed, ...)`` — legacy path.

        Splits feed assembly from rendering so the same dict can be used
        for both paths during the migration period (Phase 2-5).
        """
        config = (inference_config or {}).get(
            "consensus_config", self.consensus_config
        )
        prior_output_path = self._resolve_prior_proposer_output_path() or ""
        # When the reviewer lacks local file access, inline the FULL artifact
        # content (from output.md on disk) as main_response instead of the
        # <Response> summary. CLI reviewers can read the file themselves via
        # prior_output_path; API reviewers cannot.
        main_response = proposal
        reviewer = self._role_get("review_inferencer")
        reviewer_has_local = getattr(reviewer, "has_local_access", False) if reviewer else False
        if not reviewer_has_local and prior_output_path and os.path.isfile(prior_output_path):
            try:
                main_response = open(prior_output_path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                pass
        feed = {
            self.placeholder_input: inference_input,
            self.placeholder_proposal: proposal,
            "iteration": iteration,
            "attempt": attempt,
            "round_index": max(0, iteration - 1),
            # Outer-template slots (always set; safe with empty-string sentinel).
            "main_response": main_response,
            "prior_output_path": prior_output_path,
            # Consensus threshold guidance for review templates.
            "consensus_threshold": str(config.consensus_threshold),
            "approve_hint": config.approve_hint(),
            "approval_guidance": config.approval_guidance(),
        }
        if counter_feedback is not None:
            feed[self.placeholder_counter_feedback] = counter_feedback
        return feed

    def _build_review_prompt(
        self,
        inference_input,
        proposal: str,
        counter_feedback: Optional[str],
        inference_config: dict,
        iteration: int = 1,
        attempt: int = 1,
    ) -> str:
        """Build the review prompt from template (legacy compatibility shim).

        Phase 2: this is now a thin wrapper that builds the feed dict and
        immediately renders via the legacy orchestrator-side path. Modern
        callers should use ``_build_review_feed()`` + leaf-side rendering
        directly. Kept for: (1) external callers that may import this,
        (2) tests that mock at this granularity, (3) legacy YAMLs without
        leaf templating capability.
        """
        feed = self._build_review_feed(
            inference_input, proposal, counter_feedback,
            iteration=iteration, attempt=attempt,
            inference_config=inference_config,
        )
        return self._render_role_prompt("review", feed, inference_config)

    def _build_followup_feed(
        self,
        inference_input,
        proposal: str,
        parsed_review: dict,
        inference_config: dict,
        iteration: int = 1,
        attempt: int = 1,
        review_output: Optional[str] = None,
    ) -> dict:
        """Build the followup feed dict (Phase 2: pure feed assembly).

        Feed dict semantics — the following keys are ALL set, even if some are
        redundant with each other, to support both naming conventions:

          * ``self.placeholder_proposal`` (default ``"proposal"``) →
            inner default template's ``{{ proposal }}`` →
            renders into ``<CurrentProposal>`` tag.
          * ``"main_response"`` →
            outer Jinja template's ``{{ main_response }}`` →
            renders into ``<PriorVersionArtifact>`` tag (plan) or
            ``<PriorImplementation>`` tag (implementation).
            (Set unconditionally so empty-tag bug cannot recur regardless
            of how the YAML configures ``placeholder_proposal``.)
          * ``"prior_output_path"`` →
            outer template's ``{{ prior_output_path }}`` →
            renders the ``cp <prior_path> <output_path>`` instruction.
            Empty string when no prior file is on disk; Jinja
            ``{% if prior_output_path %}`` then renders the graceful fallback.
          * ``"reviewer_response"`` →
            outer template's ``{{ reviewer_response }}`` →
            renders into ``<ReviewerFeedback>`` tag.
            Empty string when ``review_output`` is None (NEVER literal
            ``"None"`` — that would leak into the prompt).
        """
        issues = parsed_review.get("issues", [])
        reasoning = parsed_review.get("reasoning", "")
        config = inference_config.get("consensus_config", self.consensus_config)

        prior_output_path = self._resolve_prior_proposer_output_path() or ""

        feed = {
            self.placeholder_input: inference_input,
            self.placeholder_proposal: proposal,
            self.placeholder_issues: self._serialize_issues(issues),
            self.placeholder_reasoning: reasoning,
            "enable_counter_feedback": config.enable_counter_feedback,
            "iteration": iteration,
            "attempt": attempt,
            "round_index": iteration,
            # Outer-template slots (always set; safe empty-string sentinels).
            "main_response": proposal,
            "prior_output_path": prior_output_path,
            "reviewer_response": review_output if review_output is not None else "",
            # Consensus threshold guidance for followup templates.
            "consensus_threshold": str(config.consensus_threshold),
            "approve_hint": config.approve_hint(),
            "approval_guidance": config.approval_guidance(),
        }
        return feed

    def _build_followup_prompt(
        self,
        inference_input,
        proposal: str,
        parsed_review: dict,
        inference_config: dict,
        iteration: int = 1,
        attempt: int = 1,
        review_output: Optional[str] = None,
    ) -> str:
        """Build the followup prompt from template (legacy compatibility shim).

        Phase 2: thin wrapper around ``_build_followup_feed`` + legacy
        orchestrator-side render. See ``_build_review_prompt`` rationale.
        """
        feed = self._build_followup_feed(
            inference_input, proposal, parsed_review, inference_config,
            iteration=iteration, attempt=attempt, review_output=review_output,
        )
        return self._render_role_prompt("followup", feed, inference_config)

    # endregion

    # region Default Parsers

    @staticmethod
    def _default_response_parser(raw: str) -> Optional[str]:
        """Extract response content from delimiter tags, with prompt-echo defense.

        Three return modes:
        1. No matches at all     -> raw unchanged (legacy passthrough).
        2. >=1 clean matches     -> most-recent clean match's content.
        3. All matches are echoes -> None (hard-failure signal).

        See forensics_round_template_echo_bug for the original corruption
        that motivated this defense.
        """
        from agent_foundation.common.response_parsers.delimiter_parser import (
            _PROMPT_ECHO_MARKERS,
        )

        any_matches = False
        for tag in ("Response", "ImprovedProposal"):
            matches = list(re.finditer(rf"<{tag}>([\s\S]*?)</{tag}>", raw))
            if matches:
                any_matches = True
            for match in reversed(matches):
                content = match.group(1).strip()
                if all(m in content for m in _PROMPT_ECHO_MARKERS):
                    continue
                return content
        if not any_matches:
            return raw
        return None

    @staticmethod
    def _default_parse_review(raw: str) -> dict:
        """Parse structured review JSON from raw reviewer output."""
        match = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
        if match:
            try:
                parsed = json.loads(match.group(1))
                approved = parsed.get("approved", parsed.get("approve", False))
                severity = parsed.get(
                    "severity", parsed.get("overall_severity", "MAJOR")
                )
                return {
                    "approved": approved,
                    "severity": severity,
                    "issues": parsed.get("issues", []),
                    "reasoning": parsed.get("reasoning", ""),
                }
            except json.JSONDecodeError:
                pass

        return {
            "approved": False,
            "severity": "MAJOR",
            "issues": [
                {
                    "severity": "MAJOR",
                    "category": "parsing_error",
                    "description": "Failed to parse structured review from reviewer output.",
                    "location": "N/A",
                    "suggestion": "Ensure reviewer produces valid JSON in ```json blocks.",
                }
            ],
            "reasoning": "Review parsing failed — treating as non-consensus.",
        }

    @staticmethod
    def _default_parse_counter_feedback(raw: str) -> dict:
        """Parse counter-feedback JSON from fixer output."""
        match = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return {
                    "items": parsed.get("items", []),
                    "summary": parsed.get("summary", ""),
                }
            except json.JSONDecodeError:
                pass
        return {"items": [], "summary": ""}

    @staticmethod
    def _default_extract_proposal(raw: str) -> str:
        """Extract improved proposal from fixer output."""
        match = re.search(r"<ImprovedProposal>([\s\S]*?)</ImprovedProposal>", raw)
        if match:
            return match.group(1).strip()
        cleaned = re.sub(r"```json\s*[\s\S]*?\s*```", "", raw).strip()
        return cleaned if cleaned else raw

    def _default_check_consensus(self, parsed_review: dict, threshold) -> bool:
        """Check if consensus is reached based on review feedback.

        Individual issues with severity above *threshold* block consensus
        even when the reviewer set ``approved: true``.
        """
        levels = self.consensus_config.severity_levels

        for issue in parsed_review.get("issues", []):
            issue_sev = issue.get("severity")
            if issue_sev is not None and not severity_at_most(issue_sev, threshold, levels):
                self.log_info(
                    {
                        "result": "REJECTED",
                        "reason": "issue severity exceeds threshold",
                        "issue_severity": issue_sev,
                        "threshold": threshold,
                        "approved": parsed_review.get("approved"),
                        "overall_severity": parsed_review.get("severity"),
                        "num_issues": len(parsed_review.get("issues", [])),
                    },
                    "ConsensusCheck",
                )
                return False

        if parsed_review.get("approved", False):
            self.log_info(
                {
                    "result": "APPROVED",
                    "approved": parsed_review.get("approved"),
                    "severity": parsed_review.get("severity"),
                    "threshold": threshold,
                    "num_issues": len(parsed_review.get("issues", [])),
                },
                "ConsensusCheck",
            )
            return True

        severity_str = parsed_review.get("severity", levels[-1] if levels else "MAJOR")
        reached = severity_at_most(severity_str, threshold, levels)
        self.log_info(
            {
                "result": "REACHED" if reached else "NOT_REACHED",
                "reason": "severity within threshold" if reached else "severity exceeds threshold",
                "approved": parsed_review.get("approved"),
                "severity": severity_str,
                "threshold": threshold,
                "severity_at_most": reached,
                "num_issues": len(parsed_review.get("issues", [])),
            },
            "ConsensusCheck",
        )
        return reached

    # endregion

    # region Utilities

    def _maybe_replace_with_file_reference(
        self,
        response_str: str,
        round_index: int,
        inference_config: dict,
    ) -> str:
        """Persist the base response to disk and return the FULL content.

        Side effect: when ``output_path`` is configured, writes
        ``artifacts/round{NN}_{basename}`` (workspace mode) or the legacy
        ``inference_config["output_path"]`` template path. Existing
        non-empty artifacts are preserved (no overwrite).

        Return value: ``response_str`` unchanged. The reviewer step downstream
        consumes this string verbatim, so substituting a "see file" stub
        would leave the reviewer with nothing to evaluate (the reviewer LLM
        is not instructed to follow the path) and would deadlock the Dual
        loop on guaranteed rejection. Saving the artifact is the value-add;
        truncating the response is not.
        """
        # -- Workspace mode --
        if self._workspace is not None and self.output_path:
            basename = self.output_path
            resolved_path = self._workspace.artifact_path(
                f"round{round_index:02d}_{basename}"
            )
            if os.path.isfile(resolved_path) and os.path.getsize(resolved_path) > 0:
                logger.info(
                    "[DualInferencer] Workspace artifact exists (%d bytes): %s",
                    os.path.getsize(resolved_path),
                    resolved_path,
                )
            else:
                try:
                    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
                    # Explicit utf-8: response_str typically contains LLM
                    # output which routinely uses Unicode characters (arrows,
                    # em-dashes, etc.). Without explicit encoding, Windows'
                    # default cp1252 fails on these as a deterministic
                    # UnicodeEncodeError that the retry layers will mistake
                    # for a transient.
                    with open(resolved_path, "w", encoding="utf-8") as f:
                        f.write(response_str)
                    logger.info(
                        "[DualInferencer] Wrote workspace artifact (%d bytes): %s",
                        len(response_str),
                        resolved_path,
                    )
                except (OSError, UnicodeEncodeError) as e:
                    logger.warning(
                        "[DualInferencer] Failed to write artifact to %s: %s",
                        resolved_path,
                        e,
                    )
            return response_str

        # -- Legacy mode --
        output_path_template = inference_config.get("output_path", "")
        if not output_path_template:
            return response_str
        resolved_path = output_path_template.replace(
            "{{ round_index }}", str(round_index)
        )
        resolved_path = resolved_path.replace("{{round_index}}", str(round_index))
        if os.path.isfile(resolved_path) and os.path.getsize(resolved_path) > 0:
            logger.info(
                "[DualInferencer] Output file exists and is non-empty (%d bytes): %s",
                os.path.getsize(resolved_path),
                resolved_path,
            )
        else:
            try:
                os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(response_str)
                logger.info(
                    "[DualInferencer] Inferencer did not write output file; "
                    "saved raw response (%d bytes) to: %s",
                    len(response_str),
                    resolved_path,
                )
            except (OSError, UnicodeEncodeError) as e:
                logger.warning(
                    "[DualInferencer] Failed to save fallback output to %s: %s",
                    resolved_path,
                    e,
                )
        return response_str

    def _assign_issue_ids(self, parsed_review: dict, iteration: int) -> dict:
        """Assign unique IDs to each issue in the parsed review."""
        for index, issue in enumerate(parsed_review.get("issues", []), start=1):
            issue["id"] = self.issue_id_format.format(iteration=iteration, index=index)
        return parsed_review

    @staticmethod
    def _serialize_issues(issues: list) -> str:
        """Serialize issues list to JSON string for template rendering."""
        return json.dumps(issues, indent=2)

    # endregion

    # region Lifecycle

    def _iter_child_inferencers(self):
        """Active step inferencers: base (propose), review, fixer, and any §3
        ``reviewers`` panel members.

        Used uniformly by ``aconnect`` / ``adisconnect`` /
        ``_areset_sub_inferencers`` (lifecycle) and by
        ``InferencerBase.pre_retry`` (retry-time cleanup). A
        DualInferencer retry implies wholesale restart of the consensus
        loop; resetting all here is consistent with that semantic. The
        ``reviewers`` panel (independent ``[reviewer]*k`` instances) is included so
        the extra panelists are connected/disconnected with the rest, not stranded.
        """
        seen_ids = set()
        for inf in (
            self.base_inferencer,
            self.review_inferencer,
            self.fixer_inferencer,
            *(self.reviewers or []),
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                yield inf

    def _iter_child_slots(self):
        """§9.3/N-Major1: semantic slots matching the ``ctx.child(slot)`` used in
        ``_ainfer`` (propose/review/fix) so lifecycle ops bind each child's OWN
        Tier-3 handle. Mirrors the dedup of ``_iter_child_inferencers``."""
        seen_ids = set()
        for slot, inf in (
            ("propose", self.base_inferencer),
            ("review", self.review_inferencer),
            ("fix", self.fixer_inferencer),
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                yield (slot, inf)

    async def _areset_sub_inferencers(self):
        """Reset all sub-inferencers by disconnecting and reconnecting.

        §9.3/N-R2: each child's reset runs under its own ``ctx.child(slot)`` so a
        ``reset_session`` targets the child's OWN Tier-3 handle (no-op binding
        without an active context -> byte-identical legacy behavior)."""
        for slot, inf in self._iter_child_slots():
            with self._with_child_ctx(slot):
                prev_session_id = getattr(inf, "active_session_id", None)
                await inf.adisconnect()
                if self.new_session_per_attempt:
                    if hasattr(inf, "reset_session"):
                        inf.reset_session()
                    await inf.aconnect()
                else:
                    await inf.aconnect(session_id=prev_session_id)

    async def aconnect(self, **kwargs):
        """Establish connections for all sub-inferencers.

        ``hasattr`` guard preserved (and added compared to the prior
        hardcoded version) so the inherited path stays correct even when
        a subclass override of ``_iter_child_inferencers`` yields a
        broader candidate set whose entries may not all expose
        ``aconnect`` — e.g., :class:`MultiFlowDualInferencer`'s pool
        entries.
        """
        for inf in self._iter_child_inferencers():
            if hasattr(inf, "aconnect"):
                await inf.aconnect(**kwargs)

    async def adisconnect(self):
        """Disconnect all sub-inferencers (symmetric to ``aconnect``)."""
        for inf in self._iter_child_inferencers():
            if hasattr(inf, "adisconnect"):
                await inf.adisconnect()

    # endregion
