"""Agentic inferencers package.

This package provides inferencer implementations for agentic AI interactions,
including reflective inferencers and external SDK-based inferencers.

External SDK inferencers (ClaudeCodeInferencer, DevmateSDKInferencer) use lazy
imports to avoid requiring their SDKs at import time. This allows code that
doesn't use these inferencers to import the package without the SDK dependencies.
"""

# Lazy imports for external SDK inferencers
# These are not imported at module level to avoid requiring the SDKs


def __getattr__(name):
    """Lazy import external SDK inferencers."""
    if name == "ClaudeCodeInferencer":
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        return ClaudeCodeInferencer
    elif name == "DevmateSDKInferencer":
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateSDKInferencer,
        )

        return DevmateSDKInferencer
    elif name == "SDKInferencerResponse":
        from agent_foundation.common.inferencers.agentic_inferencers.external import (
            SDKInferencerResponse,
        )

        return SDKInferencerResponse
    elif name == "DualInferencer":
        from agent_foundation.common.inferencers.agentic_inferencers.dual_inferencer import (
            DualInferencer,
        )

        return DualInferencer
    elif name == "ReflectiveInferencer":
        from agent_foundation.common.inferencers.agentic_inferencers.reflective_inferencer import (
            ReflectiveInferencer,
        )

        return ReflectiveInferencer
    elif name == "DevmateCliInferencer":
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
            DevmateCliInferencer,
        )

        return DevmateCliInferencer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClaudeCodeInferencer",
    "DevmateCliInferencer",
    "DevmateSDKInferencer",
    "DualInferencer",
    "ReflectiveInferencer",
    "SDKInferencerResponse",
]
