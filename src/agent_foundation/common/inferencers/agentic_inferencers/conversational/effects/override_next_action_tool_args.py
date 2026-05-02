# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""OverrideNextActionToolArgs effect — replace dynamic-mailbox via typed attrib.

Replaces the legacy `_pending_param_overrides` dynamic-attribute mailbox.
The CONFIRMATION handler sets this when the user adjusts tool params on the
config-panel widget; the action-tool dispatch loop reads + clears it on the
next iteration.

Merge semantics for bundles: if two effects of this type appear in one bundle,
`apply()` raises `HandlerResultMergeConflict` (whole-dict-replace; no merge).
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
class OverrideNextActionToolArgs:
    overrides: dict[str, Any]

    async def apply(self, inferencer: ConversationalInferencer) -> None:
        existing = inferencer._next_action_tool_overrides
        if existing is not None:
            raise HandlerResultMergeConflict(
                field_name="_next_action_tool_overrides",
                key=None,
                existing=existing,
                new=self.overrides,
            )
        inferencer._next_action_tool_overrides = self.overrides
