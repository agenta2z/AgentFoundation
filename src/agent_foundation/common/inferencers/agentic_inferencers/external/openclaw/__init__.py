# pyre-strict

"""OpenClaw inferencer package.

Provides a unified inferencer for the OpenClaw personal AI gateway,
supporting three transport modes via ``OpenClawMode``:

- ``OpenClawMode.PodGateway``   — WebSocket streaming to gateway in Docker/kubectl pod (default)
- ``OpenClawMode.LocalGateway`` — WebSocket streaming to natively running local gateway
- ``OpenClawMode.PodCLI``       — Docker/kubectl subprocess, blocking, always works

Quick start::

    from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
        OpenClawInferencer,
        OpenClawMode,
    )

    # Pod gateway mode (default) — streaming, session restore
    inf = OpenClawInferencer()
    result = inf("what tools do you have access to?")
    print(result)

    # Local gateway (Control UI running natively)
    inf = OpenClawInferencer.from_local_config()
    result = inf("what should I follow up on today?")
    print(result)

    # Pod CLI — simple blocking, no gateway needed
    inf = OpenClawInferencer(mode=OpenClawMode.PodCLI)
    result = inf("say hello")
    print(result)
"""

from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import (
    OpenClawAuthError,
    OpenClawError,
    OpenClawMode,
    OpenClawNotFoundError,
    OpenClawRateLimitError,
    OpenClawTimeoutError,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer import (
    OpenClawInferencer,
)

__all__ = [
    "OpenClawInferencer",
    "OpenClawMode",
    "OpenClawError",
    "OpenClawAuthError",
    "OpenClawNotFoundError",
    "OpenClawRateLimitError",
    "OpenClawTimeoutError",
]
