# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""DashboardDirectiveEffect — merges into the ``dashboard_directives`` mailbox.

Per-tool signal for dashboard/hub behavior (e.g. ``auto_implement`` from
proposal_selection). DashboardCoordinator.maybe_open() reads the AGGREGATE
directive across all tools in a compound widget from that mailbox in
loop-frame — this effect only writes the per-tool contribution.

Merge semantics: same-key double-write with a differing value raises
``HandlerResultMergeConflict`` (matches OverrideNextActionToolArgs /
SetTurnVariables pattern). Same-key same-value is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    effect_target,
    EffectTarget,
    HandlerResultMergeConflict,
)


@dataclass
class DashboardDirectiveEffect:
    directives: dict[str, Any]

    async def apply(self, inferencer: EffectTarget) -> None:
        mailboxes = effect_target(inferencer).mailboxes
        existing = mailboxes.dashboard_directives
        if existing is None:
            mailboxes.dashboard_directives = dict(self.directives)
            return
        for key, value in self.directives.items():
            if key in existing and existing[key] != value:
                raise HandlerResultMergeConflict(
                    field_name="dashboard_directives",
                    key=key,
                    existing=existing[key],
                    new=value,
                )
        existing.update(self.directives)
