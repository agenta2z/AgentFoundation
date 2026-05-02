# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""ApplyContextUpdates effect — write keys into inferencer.prior_context.

Used by handlers that publish context-bag entries (e.g. CONFIRMATION posts
`_confirmation_gate_passed`). Subsequent handlers in the same bundle see
the writes through their fresh `MappingProxyType` views (live, not snapshot).

Callers (current + planned):
- handlers/confirmation.py — gate flag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )


@dataclass
class ApplyContextUpdates:
    updates: dict[str, Any]

    async def apply(self, inferencer: ConversationalInferencer) -> None:
        inferencer.prior_context.update(self.updates)
