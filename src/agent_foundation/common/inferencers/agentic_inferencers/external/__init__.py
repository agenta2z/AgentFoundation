"""External SDK-based inferencers package.

This package provides inferencer implementations that wrap external SDKs
for AI agent interactions (Claude Code, Devmate, RovoChat, etc.).

RovoChat inferencer available via:
    ``from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat import RovoChatInferencer``
"""

from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
    SDKInferencerResponse,
)

__all__ = ["SDKInferencerResponse"]
