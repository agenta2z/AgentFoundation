# pyre-strict

"""The ``@agentic_function`` framework — a coding surface on ``InferencerBase``.

A decorator that turns a plain typed function into an LLM-backed one: call args
feed a co-located template, a resolved inferencer runs, and a typed parse
pipeline produces the declared return type. See ``decorator.py`` for the
body-role model and the module docstrings for each stage.
"""

from __future__ import annotations

from agent_foundation.common.inferencers.agentic_functions.config import (
    add_trusted_config_root,
    InferencerConfig,
)
from agent_foundation.common.inferencers.agentic_functions.decorator import (
    agentic_function,
    validate_agentic_function,
)
from agent_foundation.common.inferencers.agentic_functions.errors import (
    AgenticFunctionConfigurationError,
    AgenticFunctionError,
    ParseError,
)
from agent_foundation.common.inferencers.agentic_functions.output import (
    Agentic,
    AgenticOutput,
    escalate,
    ESCALATE,
    FromInference,
)
from agent_foundation.common.inferencers.agentic_functions.trace import (
    AgenticFunctionTrace,
)

__all__ = [
    "agentic_function",
    "validate_agentic_function",
    "AgenticOutput",
    "Agentic",
    "ESCALATE",
    "escalate",
    "FromInference",
    "InferencerConfig",
    "add_trusted_config_root",
    "AgenticFunctionError",
    "AgenticFunctionConfigurationError",
    "ParseError",
    "AgenticFunctionTrace",
]
