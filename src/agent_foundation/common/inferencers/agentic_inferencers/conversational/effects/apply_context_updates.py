# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""ApplyContextUpdates effect — write keys into the target's prior_context
through ``update_prior_context``.

Used by handlers that publish context-bag entries (e.g. CONFIRMATION posts
`_confirmation_gate_passed`). Subsequent handlers in the same bundle see
the writes through their fresh `MappingProxyType` views (live, not snapshot).

Callers (current + planned):
- handlers/confirmation.py — gate flag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    effect_target,
    EffectTarget,
)


@dataclass
class ApplyContextUpdates:
    updates: dict[str, Any]

    async def apply(self, inferencer: EffectTarget) -> None:
        effect_target(inferencer).update_prior_context(**self.updates)
