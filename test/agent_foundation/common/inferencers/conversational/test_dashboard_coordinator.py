# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""DashboardCoordinator unit tests — Phase I extraction.

Covers:
- ``normalize_directives`` sets ``metadata.open_dashboard`` correctly for tools
  carrying a Dashboard flag AND whose widget_type is in the dashboard's embeds.
- ``build_seed`` produces the correct seed dict from collected + next
  directives, including the ``auto_implement`` bit.
- ``maybe_open`` no-ops when there's no dashboard-carrier tool.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.dashboard_coordinator import (
    DashboardCoordinator,
)


def _dashboard_tool_def(embeds: list[str]) -> SimpleNamespace:
    """Fake ToolDefinition of tool_type='Dashboard' with the given embeds."""
    return SimpleNamespace(
        tool_type="Dashboard",
        dashboard_config={"embeds": embeds, "label": "Experiment Hub"},
    )


def test_normalize_directives_no_dashboards_registered() -> None:
    """When no Dashboard tools are registered, normalize is a no-op."""
    coord = DashboardCoordinator(
        tool_registry={},
        tool_dispatcher=None,
        prior_context_reader=lambda: {},
    )
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        metadata={"experiment_hub": True},
    )
    coord.normalize_directives([tool])
    assert "open_dashboard" not in tool.metadata


def test_normalize_directives_sugar_flag_sets_open_dashboard() -> None:
    coord = DashboardCoordinator(
        tool_registry={"experiment_hub": _dashboard_tool_def(["proposal_selection"])},
        tool_dispatcher=None,
        prior_context_reader=lambda: {},
    )
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        metadata={"experiment_hub": True},
    )
    coord.normalize_directives([tool])
    assert tool.metadata["open_dashboard"] == "experiment_hub"
    assert "submit_label" in tool.metadata


def test_normalize_directives_explicit_open_dashboard_key() -> None:
    coord = DashboardCoordinator(
        tool_registry={"experiment_hub": _dashboard_tool_def(["proposal_selection"])},
        tool_dispatcher=None,
        prior_context_reader=lambda: {},
    )
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        metadata={"open_dashboard": "experiment_hub"},
    )
    coord.normalize_directives([tool])
    assert tool.metadata["open_dashboard"] == "experiment_hub"


def test_normalize_directives_embeds_gate_rejects_wrong_widget() -> None:
    """If dashboard.embeds doesn't include this tool's widget_type, skip."""
    coord = DashboardCoordinator(
        # embeds only allows "different_widget", not proposal_selection
        tool_registry={"experiment_hub": _dashboard_tool_def(["different_widget"])},
        tool_dispatcher=None,
        prior_context_reader=lambda: {},
    )
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        metadata={"experiment_hub": True},
    )
    coord.normalize_directives([tool])
    # Not set — this tool's widget_type isn't in the dashboard's embeds.
    assert "open_dashboard" not in tool.metadata


def test_build_seed_from_str_collected() -> None:
    coord = DashboardCoordinator(
        tool_registry=None,
        tool_dispatcher=None,
        prior_context_reader=lambda: {},
    )
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        metadata={"open_dashboard": "experiment_hub"},
        output_vars=["selected_proposal_ids"],
    )
    seed = coord.build_seed(tool, "P1,P2,P3")
    assert seed["dashboard_id"] == "experiment_hub"
    assert seed["selected_proposal_ids"] == ["P1", "P2", "P3"]
    assert seed["auto_implement"] is False


def test_build_seed_from_dict_collected() -> None:
    coord = DashboardCoordinator(
        tool_registry=None,
        tool_dispatcher=None,
        prior_context_reader=lambda: {},
    )
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        metadata={"open_dashboard": "experiment_hub"},
        output_vars=["selected_proposal_ids"],
    )
    seed = coord.build_seed(tool, {"selected_proposal_ids": "A,B"})
    assert seed["selected_proposal_ids"] == ["A", "B"]


def test_build_seed_reads_auto_implement_from_directives() -> None:
    coord = DashboardCoordinator(
        tool_registry=None,
        tool_dispatcher=None,
        prior_context_reader=lambda: {},
    )
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        metadata={"open_dashboard": "experiment_hub"},
    )
    seed = coord.build_seed(
        tool, "P1", next_dashboard_directives={"auto_implement": True}
    )
    assert seed["auto_implement"] is True


@pytest.mark.asyncio
async def test_maybe_open_no_carrier_is_noop() -> None:
    coord = DashboardCoordinator(
        tool_registry=None,
        tool_dispatcher=None,
        prior_context_reader=lambda: {},
    )
    tool_without_dashboard = ConversationTool(
        tool_type=ConversationToolType.CLARIFICATION,
    )
    await coord.maybe_open([tool_without_dashboard], "some_collected")


@pytest.mark.asyncio
async def test_maybe_open_no_executor_is_noop() -> None:
    coord = DashboardCoordinator(
        tool_registry=None,
        tool_dispatcher=None,  # no executor available
        prior_context_reader=lambda: {},
    )
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        metadata={"open_dashboard": "experiment_hub"},
    )
    await coord.maybe_open([tool], "P1")
