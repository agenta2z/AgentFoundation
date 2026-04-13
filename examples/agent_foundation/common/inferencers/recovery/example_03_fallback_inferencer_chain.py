#!/usr/bin/env python3
"""Example 03 — Fallback Inferencer Chain: Primary -> Self-Recovery -> External Fallback.

This example demonstrates the two-tier fallback architecture:

    Tier 1: Self-recovery (built-in)
        Every streaming inferencer has a `_ainfer_recovery` method that is
        automatically wired as the FIRST fallback. By default, it delegates
        to `_ainfer` (same function = plain retry). Streaming inferencers
        override it to use cache-based smart recovery (see Example 02).

    Tier 2: External fallback inferencer(s)
        You can set `fallback_inferencer` to one or more alternative inferencers.
        If self-recovery also fails, the system escalates to these external
        fallbacks in order.

The full chain executes like this:

    1. Primary `_ainfer()` call fails
    2. ON_FIRST_FAILURE -> switch to self-recovery (`_ainfer_recovery()`)
    3. Self-recovery gets `max_retry` attempts
    4. If self-recovery also exhausts -> switch to fallback_inferencer[0]
    5. fallback_inferencer[0].ainfer() (full pipeline, with ITS OWN retry logic)
    6. If that fails too -> fallback_inferencer[1].ainfer() ... etc.

Real-world use case:
    "If Claude SDK fails, try Claude CLI. If CLI fails, try a cloud endpoint."

Expected terminal output:

    === Scenario 1: Primary fails, self-recovery succeeds ===
    [FastAPI  ] _ainfer -- CRASH! (APIError: Rate limit exceeded)
    [FastAPI  ] _ainfer_recovery -- SUCCESS: "Recovered via self-healing"
    Result: Recovered via self-healing
    Chain depth reached: self-recovery (tier 1)

    === Scenario 2: Primary + self-recovery fail, external fallback succeeds ===
    [FastAPI  ] _ainfer -- CRASH! (APIError: Service unavailable)
    [FastAPI  ] _ainfer_recovery -- CRASH! (APIError: Still down)
    [CLI      ] ainfer -- SUCCESS: "CLI fallback response"
    Result: CLI fallback response
    Chain depth reached: external fallback (tier 2)

    === Scenario 3: Multi-level external fallback chain ===
    [SDK      ] _ainfer -- CRASH!
    [SDK      ] _ainfer_recovery -- CRASH!
    [CLI      ] ainfer -- CRASH!
    [Cloud    ] ainfer -- SUCCESS: "Cloud endpoint response"
    Result: Cloud endpoint response
    Chain depth reached: external fallback #2 (tier 2, position 2)

Run:
    python examples/agent_foundation/common/inferencers/recovery/example_03_fallback_inferencer_chain.py
"""

import asyncio
import os
import sys

# --- Path setup ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agent_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", "..", ".."))
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_agent_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any, AsyncIterator, Optional

from attr import attrib, attrs

from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from rich_python_utils.common_utils.function_helper import FallbackMode


# ---------------------------------------------------------------------------
# Mock inferencers representing different backends
# ---------------------------------------------------------------------------

@attrs
class MockBackend(StreamingInferencerBase):
    """A configurable mock backend that can simulate success or failure.

    Args:
        name: Human-readable name for logging (e.g., "SDK", "CLI", "Cloud").
        should_crash: If True, _ainfer always crashes.
        recovery_crashes: If True, _ainfer_recovery also crashes.
        response: The response to return on success.
    """
    name: str = attrib(default="Mock")
    should_crash: bool = attrib(default=False)
    recovery_crashes: bool = attrib(default=False)
    response: str = attrib(default="Default response")

    async def _ainfer_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        yield self.response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        if self.should_crash:
            print(f"  [{self.name:8s}] _ainfer -- CRASH! (APIError: Service unavailable)")
            raise RuntimeError(f"{self.name}: Service unavailable")
        print(f"  [{self.name:8s}] _ainfer -- SUCCESS: \"{self.response}\"")
        return self.response

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.response

    async def _ainfer_recovery(self, inference_input, last_exception, last_partial_output,
                                inference_config=None, **kwargs):
        if self.recovery_crashes:
            print(f"  [{self.name:8s}] _ainfer_recovery -- CRASH! (APIError: Still down)")
            raise RuntimeError(f"{self.name}: Recovery also failed")
        print(f"  [{self.name:8s}] _ainfer_recovery -- SUCCESS: \"{self.response} (recovered)\"")
        return f"{self.response} (recovered)"

    async def adisconnect(self):
        pass


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

def separator(title: str):
    print(f"\n{'=' * 3} {title} {'=' * 3}")


async def main():
    print("Fallback Inferencer Chain Demo")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  Primary._ainfer() -> Primary._ainfer_recovery() -> FallbackInferencer[0].ainfer() -> ...")
    print()

    # ── Scenario 1: Self-recovery succeeds ──────────────────────────────
    # Primary crashes, but self-recovery works.
    # The external fallback is never reached.
    separator("Scenario 1: Primary fails, self-recovery succeeds")

    cli_fallback = MockBackend(name="CLI", response="CLI fallback response")

    primary = MockBackend(
        name="FastAPI",
        should_crash=True,          # Primary always crashes
        recovery_crashes=False,     # But self-recovery works
        response="Recovered via self-healing",
        fallback_inferencer=cli_fallback,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        max_retry=2,
        min_retry_wait=0, max_retry_wait=0,
    )

    result = await primary.ainfer("Summarize this document")
    print(f"  Result: {result}")
    print(f"  Chain depth reached: self-recovery (tier 1)")

    # ── Scenario 2: Self-recovery also fails -> external fallback ───────
    # Both primary and self-recovery crash.
    # External fallback (CLI backend) succeeds.
    separator("Scenario 2: Primary + self-recovery fail, external fallback succeeds")

    cli_fallback = MockBackend(name="CLI", response="CLI fallback response")

    primary = MockBackend(
        name="FastAPI",
        should_crash=True,          # Primary crashes
        recovery_crashes=True,      # Self-recovery also crashes
        response="FastAPI response",
        fallback_inferencer=cli_fallback,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        max_retry=1,
        min_retry_wait=0, max_retry_wait=0,
    )

    result = await primary.ainfer("Summarize this document")
    print(f"  Result: {result}")
    print(f"  Chain depth reached: external fallback (tier 2)")

    # ── Scenario 3: Multi-level external fallback chain ─────────────────
    # Primary + recovery fail, first external fallback also fails,
    # second external fallback succeeds.
    separator("Scenario 3: Multi-level external fallback chain")

    cloud_fallback = MockBackend(name="Cloud", response="Cloud endpoint response")
    cli_fallback = MockBackend(
        name="CLI",
        should_crash=True,
        response="CLI response",
        max_retry=1,
        fallback_mode=FallbackMode.NEVER,
        min_retry_wait=0, max_retry_wait=0,
    )

    primary = MockBackend(
        name="SDK",
        should_crash=True,
        recovery_crashes=True,
        response="SDK response",
        fallback_inferencer=[cli_fallback, cloud_fallback],  # list = ordered chain
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        max_retry=1,
        min_retry_wait=0, max_retry_wait=0,
    )

    result = await primary.ainfer("Summarize this document")
    print(f"  Result: {result}")
    print(f"  Chain depth reached: external fallback #2 (tier 2, position 2)")

    # Summary
    separator("Summary")
    print("  The fallback chain is: primary -> self-recovery -> external[0] -> external[1] -> ...")
    print()
    print("  FallbackMode controls WHEN to switch:")
    print("    ON_FIRST_FAILURE: switch immediately on any failure")
    print("    ON_EXHAUSTED:     exhaust max_retry first, then switch")
    print("    NEVER:            no switching, just retry same function")
    print()
    print("  fallback_inferencer can be:")
    print("    - A single InferencerBase instance (one fallback)")
    print("    - A list of InferencerBase instances (ordered chain)")
    print("    - None (no external fallback, only self-recovery)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    asyncio.run(main())
