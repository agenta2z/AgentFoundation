#!/usr/bin/env python3
"""Example 01 — Basic Retry Recovery: Simulating Crashes and Automatic Self-Healing.

This example shows how the retry + fallback system handles transient failures
WITHOUT any caching or streaming. It demonstrates the fundamental mechanism:
when the primary inference call fails, the system automatically retries via
the recovery method.

Key concepts:
    - FallbackMode.ON_FIRST_FAILURE — switch to recovery on the FIRST failure
    - FallbackMode.ON_EXHAUSTED — retry N times, THEN switch to recovery
    - FallbackMode.NEVER — no recovery, just retry the same function
    - max_retry — how many retries the recovery function gets

Expected terminal output:

    === Scenario 1: ON_FIRST_FAILURE (default) ===
    [primary ] Call #1 — CRASH! (RuntimeError: Server overloaded)
    [recovery] Call #1 — SUCCESS: "Recovery result from attempt 1"
    Result: Recovery result from attempt 1
    Total calls: primary=1, recovery=1

    === Scenario 2: ON_EXHAUSTED with max_retry=3 ===
    [primary ] Call #1 — CRASH!
    [primary ] Call #2 — CRASH!
    [primary ] Call #3 — CRASH!
    [recovery] Call #1 — SUCCESS: "Recovery result from attempt 1"
    Result: Recovery result from attempt 1
    Total calls: primary=3, recovery=1

    === Scenario 3: Intermittent failures (fails twice, then succeeds) ===
    [primary ] Call #1 — CRASH! (ConnectionError)
    [recovery] Call #1 — CRASH! (ConnectionError)
    [recovery] Call #2 — SUCCESS: "Finally worked on recovery attempt 2"
    Result: Finally worked on recovery attempt 2
    Total calls: primary=1, recovery=2

    === Scenario 4: FallbackMode.NEVER (no recovery, retry same function) ===
    [primary ] Call #1 — CRASH!
    [primary ] Call #2 — SUCCESS: "Primary succeeded on attempt 2"
    Result: Primary succeeded on attempt 2
    Total calls: primary=2, recovery=0

Run:
    python examples/agent_foundation/common/inferencers/recovery/example_01_basic_retry_recovery.py
"""

import asyncio
import os
import sys

# --- Path setup (make this script runnable standalone) ---
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
# Mock inferencer that simulates crashes
# ---------------------------------------------------------------------------

@attrs
class CrashingInferencer(StreamingInferencerBase):
    """A mock inferencer whose primary _ainfer crashes N times before succeeding.

    Args:
        crash_count: How many times _ainfer should fail before succeeding.
        crash_exception: The exception type to raise on crash.
    """

    crash_count: int = attrib(default=1)
    crash_exception: type = attrib(default=RuntimeError)

    # Internal counters (not constructor params)
    _primary_calls: int = attrib(default=0, init=False, repr=False)
    _recovery_calls: int = attrib(default=0, init=False, repr=False)
    _total_primary_crashes: int = attrib(default=0, init=False, repr=False)

    # -- Abstract method implementations (required by StreamingInferencerBase) --

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        yield "streaming chunk"

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs
    ) -> str:
        self._primary_calls += 1
        if self._total_primary_crashes < self.crash_count:
            self._total_primary_crashes += 1
            msg = "Server overloaded"
            print(f"  [primary ] Call #{self._primary_calls} -- CRASH! ({self.crash_exception.__name__}: {msg})")
            raise self.crash_exception(msg)
        print(f"  [primary ] Call #{self._primary_calls} -- SUCCESS: \"Primary succeeded on attempt {self._primary_calls}\"")
        return f"Primary succeeded on attempt {self._primary_calls}"

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "sync_not_used"

    async def _ainfer_recovery(
        self,
        inference_input: Any,
        last_exception: Optional[Exception],
        last_partial_output: Optional[str],
        inference_config: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """Override recovery to track calls and simulate intermittent recovery failures."""
        self._recovery_calls += 1
        # By default, recovery succeeds
        result = f"Recovery result from attempt {self._recovery_calls}"
        print(f"  [recovery] Call #{self._recovery_calls} -- SUCCESS: \"{result}\"")
        return result

    def reset_counters(self):
        self._primary_calls = 0
        self._recovery_calls = 0
        self._total_primary_crashes = 0


@attrs
class IntermittentCrashInferencer(CrashingInferencer):
    """Like CrashingInferencer, but recovery also fails N times."""

    recovery_crash_count: int = attrib(default=1)
    _total_recovery_crashes: int = attrib(default=0, init=False, repr=False)

    async def _ainfer_recovery(self, inference_input, last_exception, last_partial_output,
                                inference_config=None, **kwargs):
        self._recovery_calls += 1
        if self._total_recovery_crashes < self.recovery_crash_count:
            self._total_recovery_crashes += 1
            print(f"  [recovery] Call #{self._recovery_calls} -- CRASH! (ConnectionError)")
            raise ConnectionError("Intermittent network issue")
        result = f"Finally worked on recovery attempt {self._recovery_calls}"
        print(f"  [recovery] Call #{self._recovery_calls} -- SUCCESS: \"{result}\"")
        return result

    def reset_counters(self):
        super().reset_counters()
        self._total_recovery_crashes = 0


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

def separator(title: str):
    print(f"\n{'=' * 3} {title} {'=' * 3}")


async def main():
    # ── Scenario 1: ON_FIRST_FAILURE (default) ─────────────────────────
    # Primary crashes once -> immediately switches to recovery.
    # Recovery succeeds on first try.
    # Total: 1 primary attempt + 1 recovery attempt = 2 calls.
    separator("Scenario 1: ON_FIRST_FAILURE (default)")

    inf = CrashingInferencer(
        crash_count=1,                              # Primary fails once
        max_retry=3,                                # Recovery gets up to 3 retries
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,  # Switch on first failure (default)
        min_retry_wait=0, max_retry_wait=0,         # No sleep between retries
    )
    result = await inf.ainfer("Tell me a joke")
    print(f"  Result: {result}")
    print(f"  Total calls: primary={inf._primary_calls}, recovery={inf._recovery_calls}")

    # ── Scenario 2: ON_EXHAUSTED with max_retry=3 ──────────────────────
    # Primary crashes 3 times (exhausts retry budget) -> then switches to recovery.
    # With ON_EXHAUSTED, the primary function is retried up to max_retry times
    # BEFORE the recovery method is tried.
    separator("Scenario 2: ON_EXHAUSTED with max_retry=3")

    inf = CrashingInferencer(
        crash_count=99,                             # Primary always fails
        max_retry=3,                                # Retry primary 3 times
        fallback_mode=FallbackMode.ON_EXHAUSTED,    # Exhaust retries first
        min_retry_wait=0, max_retry_wait=0,
    )
    result = await inf.ainfer("Tell me a joke")
    print(f"  Result: {result}")
    print(f"  Total calls: primary={inf._primary_calls}, recovery={inf._recovery_calls}")

    # ── Scenario 3: Intermittent failures ───────────────────────────────
    # Primary crashes once -> switches to recovery -> recovery also crashes once
    # -> recovery retries and succeeds on second attempt.
    # This shows that the recovery method itself gets retried via max_retry.
    separator("Scenario 3: Intermittent failures (fails twice, then succeeds)")

    inf = IntermittentCrashInferencer(
        crash_count=1,                              # Primary fails once
        recovery_crash_count=1,                     # Recovery also fails once
        max_retry=3,                                # Recovery gets 3 retries
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        min_retry_wait=0, max_retry_wait=0,
    )
    result = await inf.ainfer("Tell me a joke")
    print(f"  Result: {result}")
    print(f"  Total calls: primary={inf._primary_calls}, recovery={inf._recovery_calls}")

    # ── Scenario 4: FallbackMode.NEVER ──────────────────────────────────
    # No recovery method — just retry the same primary function.
    # Primary fails once, then succeeds on retry.
    # This is the "classic" retry behavior (pre-fallback feature).
    separator("Scenario 4: FallbackMode.NEVER (no recovery, retry same function)")

    inf = CrashingInferencer(
        crash_count=1,                              # Fails once, then succeeds
        max_retry=3,
        fallback_mode=FallbackMode.NEVER,           # No recovery, just retry
        min_retry_wait=0, max_retry_wait=0,
    )
    result = await inf.ainfer("Tell me a joke")
    print(f"  Result: {result}")
    print(f"  Total calls: primary={inf._primary_calls}, recovery={inf._recovery_calls}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # Suppress retry warnings for cleaner output
    asyncio.run(main())
