# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""OverrideNextActionToolArgs effect — sets the ``action_overrides`` mailbox.

The CONFIRMATION handler sets this when the user adjusts tool params on the
config-panel widget; the action-tool dispatch loop reads + clears it on the
next iteration.

Merge semantics for bundles: if two effects of this type appear in one bundle,
`apply()` raises `HandlerResultMergeConflict` (whole-dict-replace; no merge).
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
class OverrideNextActionToolArgs:
    overrides: dict[str, Any]

    async def apply(self, inferencer: EffectTarget) -> None:
        mailboxes = effect_target(inferencer).mailboxes
        existing = mailboxes.action_overrides
        if existing is not None:
            raise HandlerResultMergeConflict(
                field_name="action_overrides",
                key=None,
                existing=existing,
                new=self.overrides,
            )
        mailboxes.action_overrides = self.overrides
