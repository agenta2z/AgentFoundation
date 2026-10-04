"""Golden renders of ``conversation/main/initial.jinja2`` through
``ConversationalInferencer._render_prompt``.

Locks the classic prompt byte-for-byte across the SOP/identity matrix (SOP
active or idle × paused × in-progress × employee × ``soft_max_iterations`` ×
catalog allow/deny filters) so that refactors of how its sections are produced
(shared section templates, SOP-feed extraction, command discovery) provably
change nothing.

The goldens were rendered by the code as it was before those refactors (rev
``ae236be2ef46``), with the later intended changes applied: the unproduced
``### Skills`` block removed, and completed phases stored as bare ids marked
``✓ (done)`` in the phase outline (``active_completed_phases``). Regenerate
deliberately with ``AF_REGEN_RENDER_GOLDEN=1`` only after an intended wording
change.

Each golden is ``<scenario>.json`` holding the render split on ``"\\n"``;
joining the lines back with ``"\\n"`` reproduces the render exactly, so trailing
spaces, leading blank lines and the missing final newline are compared without
the fixture file itself carrying them.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Callable

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    _CONTINUE_AFTER_TOOLS,
    ConversationalInferencer,
)
from agent_foundation.resources.tools.registry import load_all_tools

_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "render_golden"
_REGEN = os.environ.get("AF_REGEN_RENDER_GOLDEN") == "1"

_USER_MESSAGE = "please help me"
_SUSPENDED_AT = "2026-01-01T00:00:00+00:00"

_EMPLOYEE = {
    "name": "Ada",
    "role": "Research Engineer",
    "mindset": "Be rigorous, cite evidence, and keep the user informed.",
}

_HISTORY = [
    {"role": "user", "content": "hi there"},
    {"role": "assistant", "content": "Hello! How can I help?"},
]


class _FakeBase:
    """Stands in for the backend inferencer: rendering only reads its cwd."""

    effective_cwd = "/workspace/project"
    system_prompt = ""


def _make_ci(**kwargs) -> ConversationalInferencer:
    return ConversationalInferencer(base_inferencer=_FakeBase(), **kwargs)


def _enter(ci: ConversationalInferencer, name: str) -> None:
    """``/sop name``: the active SOP (if any) is paused, ``name`` becomes active."""
    state, error = ci._enter_sop(name)
    assert error is None, error
    if ci.sop_state is not None:
        ci.sop_state.suspension_reason = "paused"
        ci.sop_state.suspended_at = _SUSPENDED_AT
        ci._suspended_sops.insert(0, ci.sop_state)
    ci.sop_state = state


def _suspend(ci: ConversationalInferencer, name: str, reason: str) -> None:
    """Add ``name`` as the most recent suspended SOP without touching the active one."""
    state, error = ci._enter_sop(name)
    assert error is None, error
    state.suspension_reason = reason
    state.suspended_at = _SUSPENDED_AT
    ci._suspended_sops.insert(0, state)


def _idle_bare() -> ConversationalInferencer:
    return _make_ci()


def _idle_employee_history() -> ConversationalInferencer:
    ci = _make_ci(soft_max_iterations=30)
    ci.set_prior_context(
        {
            "employee": _EMPLOYEE,
            "session_root_path": "/workspace/project",
            "workflow_target_path": "/workspace/project/model",
        }
    )
    ci.set_messages(list(_HISTORY))
    return ci


def _idle_filtered_catalog() -> ConversationalInferencer:
    return _make_ci(allowed_sops=["model_optimization", "code_optimization"])


def _idle_disallowed() -> ConversationalInferencer:
    return _make_ci(disallowed_sops=["sop_creation"])


def _idle_allowed_and_disallowed() -> ConversationalInferencer:
    ci = _make_ci(
        allowed_sops=["model_optimization", "code_optimization"],
        disallowed_sops=["code_optimization"],
        soft_max_iterations=30,
    )
    ci.set_prior_context({"employee": _EMPLOYEE})
    return ci


def _idle_catalog_all_filtered() -> ConversationalInferencer:
    return _make_ci(allowed_sops=["no_such_sop"])


def _idle_with_paused() -> ConversationalInferencer:
    ci = _make_ci()
    _enter(ci, "code_optimization")
    ci.sop_state.suspension_reason = "paused"
    ci.sop_state.suspended_at = _SUSPENDED_AT
    ci._suspended_sops.insert(0, ci.sop_state)
    ci.sop_state = None
    return ci


def _idle_inprogress_only() -> ConversationalInferencer:
    ci = _make_ci(disallowed_sops=["model_optimization"], soft_max_iterations=30)
    _suspend(ci, "code_optimization", "exited")
    return ci


def _idle_paused_and_inprogress() -> ConversationalInferencer:
    ci = _make_ci(
        allowed_sops=["model_optimization", "sop_creation"], soft_max_iterations=30
    )
    ci.set_prior_context({"employee": _EMPLOYEE})
    _suspend(ci, "code_optimization", "paused")
    _suspend(ci, "sop_creation", "exited")
    _suspend(ci, "model_optimization", "paused")
    return ci


def _idle_real_tools() -> ConversationalInferencer:
    return _make_ci(tool_registry=load_all_tools())


def _active_sop() -> ConversationalInferencer:
    ci = _make_ci(soft_max_iterations=30)
    ci.set_prior_context({"employee": _EMPLOYEE})
    _enter(ci, "model_optimization")
    return ci


def _active_bare() -> ConversationalInferencer:
    ci = _make_ci()
    _enter(ci, "code_optimization")
    return ci


def _active_with_suspended() -> ConversationalInferencer:
    ci = _make_ci()
    _enter(ci, "sop_creation")
    _enter(ci, "code_optimization")
    _enter(ci, "model_optimization")
    ci._suspended_sops[1].suspension_reason = "exited"
    return ci


def _active_paused_filtered() -> ConversationalInferencer:
    ci = _make_ci(allowed_sops=["sop_creation"], soft_max_iterations=30)
    ci.set_prior_context({"employee": _EMPLOYEE})
    _enter(ci, "code_optimization")
    _enter(ci, "model_optimization")
    return ci


def _active_inprogress_only() -> ConversationalInferencer:
    ci = _make_ci(disallowed_sops=["model_optimization"])
    ci.set_prior_context({"employee": _EMPLOYEE})
    _suspend(ci, "code_optimization", "exited")
    _enter(ci, "sop_creation")
    return ci


def _active_completed_phases() -> ConversationalInferencer:
    """Mid-SOP, completed phases stored as bare ids (as phase completion stores them)."""
    ci = _make_ci(soft_max_iterations=30)
    ci.set_prior_context(
        {"employee": _EMPLOYEE, "workflow_target_path": "/workspace/project/model"}
    )
    _enter(ci, "model_optimization")
    ci.sop_state.completed_phases = ["0a", "0b"]
    ci.sop_state.current_phase = "1"
    return ci


def _active_continuation_with_target() -> ConversationalInferencer:
    ci = _make_ci(soft_max_iterations=12)
    ci.set_prior_context(
        {
            "employee": _EMPLOYEE,
            "session_root_path": "/workspace/project",
            "workflow_target_path": "/workspace/project/model",
        }
    )
    _enter(ci, "model_optimization")
    ci.set_messages(list(_HISTORY) + [{"role": "user", "content": _USER_MESSAGE}])
    return ci


_SCENARIOS: dict[str, tuple[Callable[[], ConversationalInferencer], str]] = {
    "idle_bare": (_idle_bare, _USER_MESSAGE),
    "idle_employee_history": (_idle_employee_history, _USER_MESSAGE),
    "idle_filtered_catalog": (_idle_filtered_catalog, _USER_MESSAGE),
    "idle_disallowed": (_idle_disallowed, _USER_MESSAGE),
    "idle_allowed_and_disallowed": (_idle_allowed_and_disallowed, _USER_MESSAGE),
    "idle_catalog_all_filtered": (_idle_catalog_all_filtered, _USER_MESSAGE),
    "idle_with_paused": (_idle_with_paused, _USER_MESSAGE),
    "idle_inprogress_only": (_idle_inprogress_only, _USER_MESSAGE),
    "idle_paused_and_inprogress": (_idle_paused_and_inprogress, _USER_MESSAGE),
    "idle_real_tools": (_idle_real_tools, _USER_MESSAGE),
    "active_sop": (_active_sop, _USER_MESSAGE),
    "active_bare": (_active_bare, _USER_MESSAGE),
    "active_with_suspended": (_active_with_suspended, _USER_MESSAGE),
    "active_paused_filtered": (_active_paused_filtered, _USER_MESSAGE),
    "active_inprogress_only": (_active_inprogress_only, _USER_MESSAGE),
    "active_completed_phases": (_active_completed_phases, _USER_MESSAGE),
    "active_continuation_with_target": (
        _active_continuation_with_target,
        _CONTINUE_AFTER_TOOLS,
    ),
}


def render_scenarios() -> dict[str, str]:
    """Render every scenario with the importable ``agent_foundation``."""
    return {
        name: build()._render_prompt(message)
        for name, (build, message) in _SCENARIOS.items()
    }


def _golden_path(name: str) -> Path:
    return _GOLDEN_DIR / f"{name}.json"


def _encode_golden(text: str) -> str:
    return json.dumps({"lines": text.split("\n")}, indent=2, ensure_ascii=False) + "\n"


def _decode_golden(content: str) -> str:
    return "\n".join(json.loads(content)["lines"])


class RenderPromptGoldenTest(unittest.TestCase):
    def test_initial_template_renders_match_goldens(self) -> None:
        rendered = render_scenarios()
        if _REGEN:
            _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            for name, text in rendered.items():
                _golden_path(name).write_text(_encode_golden(text), encoding="utf-8")
            return
        for name, text in rendered.items():
            path = _golden_path(name)
            with self.subTest(scenario=name):
                self.assertTrue(path.exists(), f"missing golden {path}")
                golden = _decode_golden(path.read_text(encoding="utf-8"))
                self.assertEqual(golden, text)

    def test_every_golden_has_a_scenario(self) -> None:
        expected = {_golden_path(name).name for name in _SCENARIOS}
        stale = {p.name for p in _GOLDEN_DIR.iterdir()} - expected
        self.assertEqual(stale, set())

    def test_golden_encoding_round_trips_exactly(self) -> None:
        for text in ("", "\n", "a", "a\n", "\n a \n\nb  ", "x\r\ny\t✓\n\n"):
            with self.subTest(text=repr(text)):
                self.assertEqual(_decode_golden(_encode_golden(text)), text)
