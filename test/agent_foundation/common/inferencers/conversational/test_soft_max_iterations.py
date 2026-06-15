"""Tests for soft_max_iterations: the prompt-level self-governance threshold.

Verifies the CI surfaces ``soft_max_iterations`` into the conversation prompt
(so the model is told to stop / use the confirmation tool after N unproductive
rounds) only when it is set, and that the framework default YAML wires
max_iterations=-1 (unbounded) + soft_max_iterations together.
"""

import tempfile
from pathlib import Path

from attr import attrs

import agent_foundation
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
    _CONTINUE_AFTER_TOOLS,
    _USER_ROLE,
    _AGENT_ROLE,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


@attrs(slots=False)
class _MockBase(InferencerBase):
    def _infer(self, inp, cfg=None, **kw):
        return "ok"

    async def _ainfer(self, inp, cfg=None, **kw):
        return "ok"


class TestSoftMaxIterationsPrompt:
    def test_instruction_present_and_value_rendered_when_set(self):
        ci = ConversationalInferencer(base_inferencer=_MockBase(), soft_max_iterations=30)
        rendered = ci._render_prompt("hello")
        # Merged into the Decision Procedure as a step (no standalone section).
        assert "Avoid unproductive loops" in rendered
        assert "30" in rendered
        assert "confirmation" in rendered  # the tool the model is told to use

    def test_instruction_absent_when_unset(self):
        ci = ConversationalInferencer(base_inferencer=_MockBase())
        assert ci.soft_max_iterations is None
        rendered = ci._render_prompt("hello")
        assert "Avoid unproductive loops" not in rendered

    def test_feed_carries_soft_max_iterations(self):
        ci = ConversationalInferencer(base_inferencer=_MockBase(), soft_max_iterations=12)
        ci._render_prompt("hello")
        assert ci._last_template_feed.get("soft_max_iterations") == 12


class TestCurrentTurnRole:
    """The CurrentTurn role distinguishes a real user message ('user') from the
    agent continuing its own work after tools ('agent'), driving the 1a/1b
    split in the Decision Procedure."""

    def test_user_message_renders_user_role(self):
        ci = ConversationalInferencer(base_inferencer=_MockBase())
        rendered = ci._render_prompt("hello there")
        assert f"<{_USER_ROLE}>hello there</{_USER_ROLE}>" in rendered

    def test_self_continuation_renders_agent_role(self):
        ci = ConversationalInferencer(base_inferencer=_MockBase())
        rendered = ci._render_prompt(_CONTINUE_AFTER_TOOLS)
        # The continuation nudge is labeled with agent_role, not mislabeled user.
        assert f"<{_AGENT_ROLE}>{_CONTINUE_AFTER_TOOLS}</{_AGENT_ROLE}>" in rendered
        assert f"<{_USER_ROLE}>{_CONTINUE_AFTER_TOOLS}" not in rendered

    def test_decision_procedure_references_role_variables(self):
        ci = ConversationalInferencer(base_inferencer=_MockBase(), soft_max_iterations=5)
        rendered = ci._render_prompt("hi")
        # 1a/1b reference the injected role labels, not hardcoded strings.
        assert f"role `{_USER_ROLE}`" in rendered
        assert f"role `{_AGENT_ROLE}`" in rendered


class TestDecisionProcedureSopBranch:
    """The Decision Procedure splits 1a handling by whether an SOP is active,
    keyed on the `sop_active` feed flag (sop is not None) — NOT on
    sop_description (empty when the SOP has no description) or inprogress_sops
    (those are suspended, not active)."""

    def test_no_active_sop_renders_no_sop_branch(self):
        ci = ConversationalInferencer(base_inferencer=_MockBase())
        rendered = ci._render_prompt("hi")
        assert ci._last_template_feed["sop_active"] is False
        assert "No SOP is active" in rendered
        assert "An SOP is active" not in rendered

    def test_active_sop_renders_active_branch_even_without_description(self):
        from agent_foundation.common.workflow.sop_state import SOPState

        ci = ConversationalInferencer(base_inferencer=_MockBase())
        # Active SOP with an empty description: sop_active must still be True
        # (this is exactly why we use `sop is not None`, not sop_description).
        ci.sop_state = SOPState(sop=object(), sop_description="")
        rendered = ci._render_prompt("hi")
        assert ci._last_template_feed["sop_active"] is True
        assert "An SOP is active" in rendered
        assert "No SOP is active" not in rendered
        # The SOP context block (status/guidance) shows even with no description,
        # but the <SOPDescription> tag is omitted when there is none.
        assert "## Active SOP Context" in rendered
        assert "<SOPDescription>" not in rendered
        assert "<SOPNextStepGuidance>" in rendered

    def test_no_active_sop_omits_sop_context_block(self):
        ci = ConversationalInferencer(base_inferencer=_MockBase())
        rendered = ci._render_prompt("hi")
        assert "## Active SOP Context" not in rendered


class TestSoftMaxIterationsConfig:
    def test_default_yaml_sets_unbounded_and_soft_threshold(self):
        from agent_foundation.resources.tools import _ci_host

        cfg = (
            Path(agent_foundation.__file__).parent
            / "resources" / "configs" / "conversational" / "default.yaml"
        )
        ci = _ci_host.build_ci_from_config(
            cfg,
            backend="ClaudeCodeCLI",
            backend_dir=cfg.parent / "base_inferencer",
            target_path=tempfile.mkdtemp(),
        )
        assert ci.max_iterations == -1          # hard cap removed (unbounded)
        assert ci.soft_max_iterations == 30     # soft self-governance threshold


def _default_cfg_path():
    return (
        Path(agent_foundation.__file__).parent
        / "resources" / "configs" / "conversational" / "default.yaml"
    )


class TestInjectedBasePath:
    """The path the OpenStartup session factory uses: a pre-built backend `base`
    is injected while the YAML governs the CI-wrapper config."""

    def test_injected_base_is_preserved_and_yaml_governs_wrapper(self):
        from agent_foundation.resources.tools import _ci_host

        base = _MockBase()
        renderer = object()
        reg = {"sometool": object()}
        ci = _ci_host.build_ci_from_config(
            _default_cfg_path(),
            base_inferencer=base,
            prompt_renderer=renderer,
            tool_registry=reg,
            tool_executor=(lambda *a, **k: None),
            extra_sop_dirs=["/tmp/openteam_sops"],
        )
        # YAML governs the wrapper config.
        assert ci.max_iterations == -1
        assert ci.soft_max_iterations == 30
        assert ci.compression_threshold == 8000
        # Runtime objects are the exact injected instances (faithful, no rebuild).
        assert ci.base_inferencer is base
        assert ci.prompt_renderer is renderer
        assert ci.tool_registry is reg
        assert ci.tool_executor is not None
        assert ci._extra_sop_dirs == ["/tmp/openteam_sops"]

    def test_debug_cascade_to_injected_base(self):
        from agent_foundation.resources.tools import _ci_host

        base = _MockBase()
        ci = _ci_host.build_ci_from_config(_default_cfg_path(), base_inferencer=base)
        # _debug_mode: false in the YAML keeps the wrapper off; the base is
        # untouched until an explicit enable cascades.
        assert ci.debug_mode is False
        ci.enable_debug_mode()
        assert ci.debug_mode is True
        assert base.debug_mode is True

    def test_backend_and_base_inferencer_are_mutually_exclusive(self):
        import pytest

        from agent_foundation.resources.tools import _ci_host

        with pytest.raises(ValueError, match="mutually exclusive"):
            _ci_host.build_ci_from_config(
                _default_cfg_path(),
                backend="ClaudeCodeCLI",
                base_inferencer=_MockBase(),
            )
