# pyre-strict
"""Inferencers that wrap arbitrary CLI tools as streaming inferencers.

Public API:
    :class:`ToolAsInferencer` — spawn a subprocess and surface its stdout
        as the inferencer's streamed output, with optional regex-based
        marker callbacks for live event extraction (epoch markers,
        URIs, job IDs, …).
    :class:`ToolInferencerResponse` — the dataclass-shaped response the
        inferencer's ``_ainfer`` returns (also re-emitted as the workflow
        step result so :func:`make_tool_chain` can extract `.parsed` /
        `.stdout` cleanly).
    :func:`make_tool_chain` — convenience factory that wraps N
        ``ToolAsInferencer`` (or any other ``InferencerBase``) instances
        in a fixed-length :class:`LinearWorkflowInferencer`. Replaces a
        bespoke ``ToolChainInferencer`` subclass; LWI already supports
        the structure.
"""

from agent_foundation.common.inferencers.agentic_inferencers.tool_inferencers.tool_as_inferencer import (
    make_tool_chain,
    ToolAsInferencer,
    ToolInferencerResponse,
)


__all__ = [
    "ToolAsInferencer",
    "ToolInferencerResponse",
    "make_tool_chain",
]
