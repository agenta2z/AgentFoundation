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
    InferencerResponse,
    ReflectionStyles,
    ResponseSelectors,
    Severity,
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
        prompt_formatter: Shared TemplateManager for all prompts. When set,
            initial_prompt/review_prompt/followup_prompt are used as template_key
            names. When None, those strings are treated as raw Jinja2 templates.
        initial_prompt: Template key (when prompt_formatter is set) or raw Jinja2
            template string for the initial prompt. None means passthrough.
        review_prompt: Template key or raw template for review prompts.
        followup_prompt: Template key or raw template for followup/fix prompts.
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

    prompt_formatter: Callable = attrib(default=None)
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

    # --- Workspace support (opt-in, overrides checkpoint_dir when set) ---

    # --- Workflow-suppressed attrs inherited from LWI (init=False) ---
    # result_pass_down_mode, unpack_single_result,
    # ignore_stop_flag_from_saved_results, auto_mode, max_loop_iterations
    # are all inherited from LinearWorkflowInferencer with init=False.
    # checkpoint_mode is also inherited from LWI (default="jsonfy").

    @property
    def supports_prompt_rendering(self) -> bool:
        return self.prompt_formatter is not None

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

        # Build prompt rendering infrastructure
        if isinstance(self.prompt_formatter, TemplateManager):
            # Shared TemplateManager — initial/review/followup are template_key names.
            # Implicit-default resolution happens lazily at render time.
            self._prompt_tms = None
        elif self.prompt_formatter is not None:
            # Custom formatter provided — wrap each EXPLICITLY configured raw
            # template string. None values fall back to the legacy in-Python
            # defaults (with a one-time logger warning) since there's no
            # template-discovery mechanism here. This preserves backward
            # compatibility for callers that construct DualInferencer with a
            # custom formatter but no explicit prompts; production callers
            # using a shared TemplateManager get the implicit-key-discovery
            # path instead.
            from agent_foundation.common.inferencers.constants import (
                DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE,
                DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE,
            )
            custom_formatter = self.prompt_formatter
            self._prompt_tms = {}
            role_to_default = {
                "initial": None,
                "review": DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE,
                "followup": DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE,
            }
            for role, prompt_str in [
                ("initial", self.initial_prompt),
                ("review", self.review_prompt),
                ("followup", self.followup_prompt),
            ]:
                effective = prompt_str
                if effective is None and role_to_default.get(role) is not None:
                    effective = role_to_default[role]
                    logger.warning(
                        "DualInferencer.%s_prompt is None — falling back to "
                        "in-Python DEFAULT_DUAL_%s_PROMPT_TEMPLATE because no "
                        "TemplateManager is configured. For production use, "
                        "pass a TemplateManager as prompt_formatter so "
                        "implicit template_key resolution can locate "
                        "<root>/main/%s.jinja2 instead.",
                        role,
                        role.upper(),
                        role,
                    )
                if effective is not None:
                    self._prompt_tms[role] = TemplateManager(
                        templates=effective,
                        template_formatter=custom_formatter,
                        enable_templated_feed=True,
                    )
        else:
            # No formatter at all — render raw Jinja2 templates directly using
            # the explicitly configured strings. None values fall back to the
            # legacy in-Python defaults with a one-time warning. Same
            # backward-compatibility rationale as the custom-formatter path
            # above.
            from agent_foundation.common.inferencers.constants import (
                DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE,
                DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE,
            )
            self._prompt_tms = {}
            role_to_default = {
                "initial": None,
                "review": DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE,
                "followup": DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE,
            }
            # Materialize defaults onto the public attribute so the explicit
            # path in _render_role_prompt sees them as "set". This keeps the
            # render contract simple (always renders a non-None template
            # string when in raw-Jinja mode).
            for role in ("review", "followup"):
                current = getattr(self, f"{role}_prompt")
                if current is None:
                    default = role_to_default.get(role)
                    if default is not None:
                        setattr(self, f"{role}_prompt", default)
                        # Mark as explicit so _render_role_prompt doesn't
                        # treat it as "implicit + missing → disable".
                        setattr(self, f"_{role}_explicit", True)
                        logger.warning(
                            "DualInferencer.%s_prompt is None — using "
                            "in-Python DEFAULT_DUAL_%s_PROMPT_TEMPLATE. "
                            "For production use, pass a TemplateManager "
                            "as prompt_formatter to enable implicit "
                            "template-key discovery.",
                            role,
                            role.upper(),
                        )

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
            self.consensus_checker = DualInferencer._default_check_consensus

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
        attempt = getattr(self, "_current_attempt", 0)
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

    # ------------------------------------------------------------------
    # Output finalization (orchestrator override)
    # ------------------------------------------------------------------

    def _finalize_output(self, response):
        """Dual override: symlink to canonical child's output.

        The canonical child is tracked by ``_last_output_child_ws``,
        set by ``_step_propose_impl`` and ``_step_fix_impl``.
        """
        child_ws = getattr(self, "_last_output_child_ws", None)
        if child_ws is not None:
            self._symlink_child_output(child_ws)
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
        return self.fixer_inferencer if self.fixer_inferencer is not None else self.base_inferencer

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

            children_dir = getattr(self._workspace, "children_dir", None)
            if children_dir:
                nav_dir = os.path.join(children_dir, f"round_{round_idx:02d}")
                os.makedirs(nav_dir, exist_ok=True)
                link_path = os.path.join(nav_dir, phase)

                # Fix #7: cross-worker leakage detection
                my_root = str(self._workspace.root).rstrip("/") + "/"
                if not target.startswith(my_root):
                    logger.error(
                        "Audit: cross-worker leakage at round_%02d/%s: "
                        "target %s outside %s",
                        round_idx, phase, target, self._workspace.root,
                    )

                if os.path.islink(link_path):
                    os.unlink(link_path)
                try:
                    os.symlink(target, link_path, target_is_directory=True)
                except (OSError, NotImplementedError):
                    pointer = os.path.join(nav_dir, f"{phase}.pointer.txt")
                    with open(pointer, "w") as f:
                        f.write(f"# Workspace pointer\n# Target: {target}\n")
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

            # Set up instance-level state (NOT in self._state — not picklable)
            self._current_attempt = attempt
            self._current_config = config

            # Non-picklable data stays on self (re-derived on resume)
            self._current_inference_config = inference_config
            self._current_extra_inference_args = _inference_args

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
        state = self._state

        if self.initial_prompt is not None:
            initial_prompt = self._build_initial_prompt(
                state["inference_input"],
                getattr(self, "_current_inference_config", {}),
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

        # Fix #8: snapshot workspace root BEFORE ainfer() — role reassignment
        # during the call may mutate _workspace.root on the inferencer.
        _propose_ws_snapshot = (
            str(self.base_inferencer._workspace.root)
            if getattr(self.base_inferencer, "_workspace", None) is not None
            else None
        )
        _raw_base = str(
            await self.base_inferencer.ainfer(
                initial_prompt, **getattr(self, "_current_extra_inference_args", {})
            )
        )
        # Track canonical output child for _finalize_output symlink
        if getattr(self.base_inferencer, "_workspace", None) is not None:
            self._last_output_child_ws = self.base_inferencer._workspace
        _sf = f"Round{state['total_iterations'] + 1:02d}"
        self.log_debug(
            _raw_base,
            "RawBaseResponse",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=_sf,
        )
        base_output_str = self.response_parser(_raw_base)
        base_output_str = self._maybe_replace_with_file_reference(
            base_output_str,
            round_index=0,
            inference_config=getattr(self, "_current_inference_config", {}),
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

        # Per-round workspace: assign review_inferencer to round_NN/children/review/
        if self._workspace is not None and self.review_inferencer is not None:
            round_ws = self._workspace.child(f"round_{consensus_iter:02d}")
            review_ws = round_ws.child("review")
            review_ws.ensure_dirs()
            self.review_inferencer._workspace = review_ws
            # Store round workspace for fix step to use
            self._current_round_ws = round_ws

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
            )
            review_leaf_can_render = self._leaf_can_self_render(
                self.review_inferencer
            )
            if review_leaf_can_render:
                # Modern path: leaf renders. Get the rendered prompt for
                # logging by calling _render_prompt directly on the leaf —
                # avoids the double-render that would occur if we used
                # ainfer(..., render_only=True) followed by ainfer(...).
                _review_extra_feed = {
                    k: v
                    for k, v in review_feed.items()
                    if k not in ("input", "__template_space__")
                }
                review_prompt = self.review_inferencer._render_prompt(
                    state["inference_input"],
                    extra_feed=_review_extra_feed,
                )
            else:
                # Legacy path: orchestrator renders.
                review_prompt = self._render_role_prompt(
                    "review",
                    review_feed,
                    getattr(self, "_current_inference_config", {}),
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
        _review_ws_snapshot = (
            str(self.review_inferencer._workspace.root)
            if getattr(self.review_inferencer, "_workspace", None) is not None
            else None
        )
        if review_leaf_can_render:
            # Reuse _review_extra_feed computed above for the logging prompt —
            # same dict, no re-computation, no re-rendering by Dual.
            # The leaf's _render_prompt fires once inside ainfer().
            _raw_review = str(
                await self.review_inferencer.ainfer(
                    state["inference_input"],
                    extra_feed=_review_extra_feed,
                    **getattr(self, "_current_extra_inference_args", {}),
                )
            )
        else:
            _raw_review = str(
                await self.review_inferencer.ainfer(
                    review_prompt, **getattr(self, "_current_extra_inference_args", {})
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
        parsed_review = self.review_parser(review_output_str)
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

        logger.info(
            "[%s] Iteration %d: severity=%s, consensus_reached=%s",
            self.phase or "DualInferencer",
            iteration,
            parsed_review.get("severity", "UNKNOWN"),
            reached,
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
            total_iters, "review", self.review_inferencer,
            workspace_root_at_phase=_review_ws_snapshot,
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
        state = self._state
        # Read with fallback for old checkpoint compat
        iteration = state.get("consensus_iteration", state.get("iteration", 0))
        total_iters = state["total_iterations"]
        attempt_num = state["attempt_record"]["attempt"]
        parsed_review = state["parsed_review"]

        # Per-round workspace: assign fixer to round_NN/children/fix/
        round_ws = getattr(self, "_current_round_ws", None)
        if round_ws is not None and self.fixer_inferencer is not None:
            fix_ws = round_ws.child("fix")
            fix_ws.ensure_dirs()
            self.fixer_inferencer._workspace = fix_ws

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
                self.fixer_inferencer
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
                followup_prompt = self.fixer_inferencer._render_prompt(
                    state["inference_input"],
                    extra_feed=_fixer_extra_feed,
                )
            else:
                # Legacy path: orchestrator renders.
                followup_prompt = self._render_role_prompt(
                    "followup",
                    followup_feed,
                    getattr(self, "_current_inference_config", {}),
                )
        except _RoleDisabledError:
            # No followup/fixer template available (and caller didn't
            # explicitly configure one) → skip the fix step and treat the
            # current base output as the final answer.
            logger.info(
                "[%s] Fixer role is disabled (no template available) — "
                "skipping fix step; review verdict stands.",
                self.phase or "DualInferencer",
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
        _fix_ws_snapshot = (
            str(self.fixer_inferencer._workspace.root)
            if getattr(self.fixer_inferencer, "_workspace", None) is not None
            else None
        )
        if fixer_leaf_can_render:
            # Reuse _fixer_extra_feed computed above — same dict, one render.
            _raw_fix = str(
                await self.fixer_inferencer.ainfer(
                    state["inference_input"],
                    extra_feed=_fixer_extra_feed,
                    **getattr(self, "_current_extra_inference_args", {}),
                )
            )
        else:
            _raw_fix = str(
                await self.fixer_inferencer.ainfer(
                    followup_prompt,
                    **getattr(self, "_current_extra_inference_args", {}),
                )
            )
        # Track canonical output child for _finalize_output symlink
        if getattr(self.fixer_inferencer, "_workspace", None) is not None:
            self._last_output_child_ws = self.fixer_inferencer._workspace
        self.log_debug(
            _raw_fix,
            "RawFixResponse",
            is_artifact=True,
            parts_min_size=0,
            parts_subfolder=f"Round{total_iters:02d}",
        )

        fix_output_str = self.response_parser(_raw_fix)
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
            total_iters, "fix", self.fixer_inferencer,
            workspace_root_at_phase=_fix_ws_snapshot,
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
        """Render a prompt template by role name.

        Resolution rules (see class docstring on ``review_prompt`` /
        ``followup_prompt`` for full design rationale):

          1. ``<role>_prompt`` is explicitly set (non-None):
             Use the configured value. If the TemplateManager cannot resolve
             it, raise ValueError (loud failure for explicit misconfig).
          2. ``<role>_prompt`` is None (default) AND we have a shared
             TemplateManager:
             Try the implicit key (``"review"`` / ``"followup"``) against
             the manager. If found, use it. If NOT found, mark the role as
             disabled and raise a sentinel ``_RoleDisabledError`` that the
             step caller treats as "skip this step".
          3. ``<role>_prompt`` is None AND no template-discovery mechanism
             is available (custom formatter / no formatter): the role is
             pre-marked as disabled in ``__attrs_post_init__``; this method
             should not be reached.

        Raises:
            ValueError: if an EXPLICITLY configured prompt cannot be resolved.
            _RoleDisabledError: if the implicit default cannot be found —
                the calling step should catch this and skip the step.
        """
        post_process = partial(unescape_xml, unescape_for_html=True)
        prompt_value = getattr(self, f"{role}_prompt", None)
        explicit = getattr(self, f"_{role}_explicit", False)

        # Pre-disabled (no resolution mechanism available + no explicit value).
        if self._is_role_disabled(role):
            raise _RoleDisabledError(role)

        # Implicit default — try the role-name as the template key.
        resolved_key = prompt_value if prompt_value is not None else role

        if self._prompt_tms is None:
            # Shared TemplateManager mode — prompt_formatter is a TemplateManager.
            # Probe existence first via get_raw_template; this lets us
            # distinguish "found and resolved" from "not found" before
            # actually rendering.
            try:
                raw = self.prompt_formatter.get_raw_template(template_key=resolved_key)
            except Exception:
                raw = None
            if raw is None or raw == "":
                if explicit:
                    raise ValueError(
                        f"DualInferencer.{role}_prompt is set to "
                        f"{resolved_key!r}, but the configured TemplateManager "
                        f"could not resolve it. Either:\n"
                        f"  - Provide a template at the configured root_space "
                        f"matching key {resolved_key!r}, or\n"
                        f"  - Remove the ``{role}_prompt`` setting to fall "
                        f"back to the implicit ``{role}`` key (and skip the "
                        f"role if that's also missing)."
                    )
                # Implicit and not found → silently disable.
                setattr(self, f"_{role}_role_disabled", True)
                raise _RoleDisabledError(role)
            return self.prompt_formatter(
                template_key=resolved_key,
                feed=feed,
                post_process=post_process,
                **inference_config,
            )
        elif self._prompt_tms:
            # Per-role TemplateManager wrappers (custom formatter was provided).
            # These were built only for explicitly-configured roles in
            # __attrs_post_init__; if we got here without a wrapper, the role
            # is disabled.
            wrapper = self._prompt_tms.get(role)
            if wrapper is None:
                raise _RoleDisabledError(role)
            return wrapper(feed=feed, post_process=post_process, **inference_config)
        else:
            # No formatter — render raw Jinja2 template directly. The
            # __attrs_post_init__ materializes legacy defaults onto
            # ``review_prompt``/``followup_prompt`` for this branch, so
            # ``prompt_value`` is non-None here unless someone constructed
            # the inferencer in an exotic way.
            if prompt_value is None:
                raise _RoleDisabledError(role)
            from jinja2 import Template

            rendered = Template(prompt_value).render(**feed)
            return post_process(rendered)

    # ─────────────────────────────────────────────────────────────────
    # Phase 2 (leaf-owned template rendering): leaf-side rendering helpers.
    # ─────────────────────────────────────────────────────────────────
    # Architectural direction: orchestrators own workflow & feed-dict
    # assembly; leaves own template selection & rendering. The legacy path
    # (_render_role_prompt) renders at the orchestrator level using
    # prompt_formatter; the modern path passes a feed dict to the leaf via
    # extra_feed= and lets the leaf's own _build_template_feed merge it.
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

    async def _ainvoke_with_feed(
        self,
        leaf,
        inference_input: str,
        feed: dict,
        extra_inference_args: dict,
    ) -> str:
        """Invoke a child leaf with a feed dict.

        Routes between leaf-side rendering (modern, via extra_feed=) and
        orchestrator-side rendering (legacy, via _render_role_prompt) based
        on whether the leaf can self-render. Returns the raw response string.

        The feed dict's reserved keys ({"input", "__template_space__"}) are
        stripped before passing as extra_feed (the leaf re-derives input
        from inference_input itself; __template_space__ comes from the
        leaf's own template_root_space).
        """
        # Strip reserved keys — the leaf manages these itself.
        leaf_extra_feed = {
            k: v
            for k, v in feed.items()
            if k not in ("input", "__template_space__")
        }
        return str(
            await leaf.ainfer(
                inference_input,
                extra_feed=leaf_extra_feed,
                **extra_inference_args,
            )
        )

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
        prior_output_path = self._resolve_prior_proposer_output_path() or ""
        feed = {
            self.placeholder_input: inference_input,
            self.placeholder_proposal: proposal,
            "iteration": iteration,
            "attempt": attempt,
            "round_index": max(0, iteration - 1),
            # Outer-template slots (always set; safe with empty-string sentinel).
            "main_response": proposal,
            "prior_output_path": prior_output_path,
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
    def _default_response_parser(raw: str) -> str:
        """Extract response content from delimiter tags."""
        for tag in ("Response", "ImprovedProposal"):
            match = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", raw)
            if match:
                return match.group(1).strip()
        return raw

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

    @staticmethod
    def _default_check_consensus(parsed_review: dict, threshold: Severity) -> bool:
        """Check if consensus is reached based on review feedback."""
        if parsed_review.get("approved", False):
            return True
        severity_str = parsed_review.get("severity", "MAJOR")
        try:
            review_severity = Severity(severity_str)
        except ValueError:
            return False
        return severity_at_most(review_severity, threshold)

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
        """Active step inferencers: base (propose), review, fixer.

        Used uniformly by ``aconnect`` / ``adisconnect`` /
        ``_areset_sub_inferencers`` (lifecycle) and by
        ``InferencerBase.pre_retry`` (retry-time cleanup). A
        DualInferencer retry implies wholesale restart of the consensus
        loop; resetting all three here is consistent with that semantic.
        """
        seen_ids = set()
        for inf in (
            self.base_inferencer,
            self.review_inferencer,
            self.fixer_inferencer,
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                yield inf

    async def _areset_sub_inferencers(self):
        """Reset all sub-inferencers by disconnecting and reconnecting."""
        for inf in self._iter_child_inferencers():
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
