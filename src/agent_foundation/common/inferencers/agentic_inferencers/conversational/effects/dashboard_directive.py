# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""DashboardDirectiveEffect — merges into _next_dashboard_directives mailbox.

Per-tool signal for dashboard/hub behavior (e.g. ``auto_implement`` from
proposal_selection). DashboardCoordinator.maybe_open() reads the AGGREGATE
directive across all tools in a compound widget from _next_dashboard_directives
in loop-frame — this effect only writes the per-tool contribution.

Merge semantics: same-key double-write with a differing value raises
``HandlerResultMergeConflict`` (matches OverrideNextActionToolArgs /
SetTurnVariables pattern). Same-key same-value is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    HandlerResultMergeConflict,
)

if TYPE_CHECKING:
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )


@dataclass
class DashboardDirectiveEffect:
    directives: dict[str, Any]

    async def apply(self, inferencer: ConversationalInferencer) -> None:
        existing = inferencer._next_dashboard_directives
        if existing is None:
            inferencer._next_dashboard_directives = dict(self.directives)
            return
        for key, value in self.directives.items():
            if key in existing and existing[key] != value:
                raise HandlerResultMergeConflict(
                    field_name="_next_dashboard_directives",
                    key=key,
                    existing=existing[key],
                    new=value,
                )
        existing.update(self.directives)
