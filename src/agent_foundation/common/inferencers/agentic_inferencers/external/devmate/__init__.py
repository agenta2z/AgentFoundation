"""Devmate Inferencers - SDK and CLI-based implementations.

This package provides two implementations for interacting with Devmate:

- DevmateSDKInferencer: Async, event-driven SDK using Thrift RPC
- DevmateCliInferencer: Subprocess-based CLI execution

Use SDK for async workflows with fine-grained event handling.
Use CLI for simpler synchronous execution or as a fallback.
"""

from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
    DevmateSDKInferencer,
)

__all__ = ["DevmateSDKInferencer", "DevmateCliInferencer"]
