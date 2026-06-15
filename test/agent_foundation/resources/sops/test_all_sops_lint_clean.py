"""Commit 6b: every shipped SOP parses cleanly and its tool/phase wiring resolves.

There is no standalone SOP "linter" binary; "lint clean" here means: each SOP
under ``resources/sops`` parses to a non-empty phase graph and computes a
``tool_to_phase_map`` without error. A dedicated check locks in that
``model_optimization`` Phase 3b/4 are wired to ``proposal_selection`` + ``task``.
"""
from pathlib import Path

import pytest

import agent_foundation
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOPManager,
)

_SOPS_DIR = Path(agent_foundation.__file__).parent / "resources" / "sops"


def _all_sop_files():
    return sorted(_SOPS_DIR.glob("*/SOP.md"))


def test_sops_exist():
    assert _all_sop_files(), f"no SOP.md files found under {_SOPS_DIR}"


@pytest.mark.parametrize(
    "sop_path", _all_sop_files(), ids=lambda p: p.parent.name
)
def test_sop_parses_and_maps_tools(sop_path):
    sop = SOPManager.load(sop_path)
    assert sop.phases, f"{sop_path} parsed to no phases"
    # tool_to_phase_map must compute without raising and reference real phases.
    mapping = sop.tool_to_phase_map
    assert isinstance(mapping, dict)
    phase_ids = set(sop.phase_ids)
    for tool_name, phase_id in mapping.items():
        assert phase_id in phase_ids, (
            f"{sop_path.parent.name}: tool {tool_name!r} maps to unknown phase "
            f"{phase_id!r} (known: {sorted(phase_ids)})"
        )


def test_model_optimization_phase3b_and_4_wiring():
    sop = SOPManager.load(_SOPS_DIR / "model_optimization" / "SOP.md")
    mapping = sop.tool_to_phase_map

    # Phase 3b now uses proposal_selection (reverted from the confirmation workaround).
    assert "proposal_selection" in mapping, (
        f"proposal_selection not wired; tool_to_phase_map={mapping}"
    )
    # Phase 4 invokes the task tool with --use-proposal / --proposal-ids.
    assert "task" in mapping, mapping

    # The proposal_selection phase declares the user-input gate.
    ps_phase = sop.get_phase(mapping["proposal_selection"])
    assert ps_phase is not None
    assert ps_phase.requires_user_input, (
        "Phase 3b must keep [__requires user input__]"
    )
