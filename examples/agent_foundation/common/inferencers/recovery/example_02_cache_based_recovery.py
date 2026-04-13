#!/usr/bin/env python3
"""Example 02 — Cache-Based Recovery: CONTINUE vs REFERENCE vs RESTART.

This example demonstrates how the streaming inferencer's recovery system uses
cached partial output from a failed streaming attempt to construct a smart
recovery prompt.

When a streaming call fails mid-response (network drop, timeout, server crash),
the inferencer has already cached the partial output chunk-by-chunk. The recovery
system reads that cache and uses one of three strategies:

    FallbackInferMode.CONTINUE
        "Here's what you generated so far. Continue from where you stopped."
        Returns: partial + continuation (concatenated)
        Best for: Long-form text, code generation — avoids re-generating content.

    FallbackInferMode.REFERENCE
        "Here's what a previous attempt produced. Use it as reference, produce
         a complete fresh response."
        Returns: only the new response (no concatenation)
        Best for: Structured output (JSON/XML) where mid-token truncation
                  makes concatenation unsafe.

    FallbackInferMode.RESTART
        Ignores the cache entirely. Just retries with the original prompt.
        Returns: only the new response
        Best for: When partial output is unreliable or not useful.

Expected terminal output:

    === Setup: Simulating a streaming failure ===
    Streaming chunks: "Once upon " -> "a time, " -> "in a land " -> CRASH!
    Cache file written with 3 chunks + failure marker.

    === Mode: CONTINUE ===
    Recovery prompt sent to model:
      "The previous response was interrupted...
       ---BEGIN PARTIAL OUTPUT---
       Once upon a time, in a land
       ---END PARTIAL OUTPUT---
       Continue from where the response was interrupted..."
    Model continuation: "far, far away, there lived a dragon."
    Final result: "Once upon a time, in a land far, far away, there lived a dragon."
    (Note: partial + continuation concatenated)

    === Mode: REFERENCE ===
    Recovery prompt sent to model:
      "A previous attempt was interrupted...
       ---BEGIN PARTIAL (REFERENCE ONLY)---
       Once upon a time, in a land
       ---END PARTIAL---
       The task is: Tell me a story"
    Final result: "Once upon a time, in a magical kingdom, a brave knight..."
    (Note: only the new response — no concatenation)

    === Mode: RESTART ===
    No cache used — original prompt sent as-is.
    Final result: "In a galaxy not so far away, two robots went on an adventure."
    (Note: completely fresh response, cache ignored)

Run:
    python examples/agent_foundation/common/inferencers/recovery/example_02_cache_based_recovery.py
"""

import asyncio
import os
import sys
import tempfile

# --- Path setup ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agent_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", "..", ".."))
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_agent_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any, AsyncIterator, Optional

from attr import attrib, attrs
from contextvars import copy_context

from agent_foundation.common.inferencers.streaming_inferencer_base import (
    FallbackInferMode,
    StreamingInferencerBase,
    _read_partial_from_cache,
)
from agent_foundation.common.inferencers.inferencer_base import (
    _current_fallback_state,
)
from rich_python_utils.common_utils.function_helper import FallbackMode


# ---------------------------------------------------------------------------
# Mock streaming inferencer that crashes mid-stream
# ---------------------------------------------------------------------------

@attrs
class StoryInferencer(StreamingInferencerBase):
    """A mock streaming inferencer that simulates a crash mid-stream.

    On the FIRST call, it yields a few chunks then raises an error.
    On subsequent calls (recovery), it returns a complete response.

    This simulates real-world behavior: the backend starts generating text,
    we cache each chunk as it arrives, then the connection drops. Recovery
    reads the cache and sends an augmented prompt to the model.
    """

    _chunks_before_crash: list = attrib(factory=lambda: ["Once upon ", "a time, ", "in a land "])
    _recovery_responses: dict = attrib(factory=dict)
    _call_count: int = attrib(default=0, init=False, repr=False)
    _recovery_prompts_received: list = attrib(factory=list, init=False, repr=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if not self._recovery_responses:
            self._recovery_responses = {
                "continue": "far, far away, there lived a dragon.",
                "reference": "Once upon a time, in a magical kingdom, a brave knight set out on a quest.",
                "restart": "In a galaxy not so far away, two robots went on an adventure.",
            }

    async def _ainfer_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """First call: yield chunks then crash. Subsequent calls: yield full response."""
        self._call_count += 1

        if self._call_count == 1:
            # First call: stream partial output then crash
            for chunk in self._chunks_before_crash:
                yield chunk
            raise ConnectionError("Network connection dropped mid-stream!")

        # Recovery call: figure out which mode from the prompt content
        self._recovery_prompts_received.append(prompt)
        if "Continue from where" in prompt:
            yield self._recovery_responses["continue"]
        elif "REFERENCE ONLY" in prompt:
            yield self._recovery_responses["reference"]
        else:
            yield self._recovery_responses["restart"]

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "sync_not_used"

    async def adisconnect(self):
        pass


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def separator(title: str):
    print(f"\n{'=' * 3} {title} {'=' * 3}")


async def demo_mode(mode: FallbackInferMode, cache_dir: str):
    """Run a single recovery demo with the given FallbackInferMode."""
    separator(f"Mode: {mode.value.upper()}")

    inf = StoryInferencer(
        cache_folder=cache_dir,
        fallback_infer_mode=mode,
        max_retry=2,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        min_retry_wait=0,
        max_retry_wait=0,
    )

    prompt = "Tell me a story"
    result = await inf.ainfer(prompt)

    # Show what the recovery system did
    if mode == FallbackInferMode.CONTINUE:
        print(f"  Strategy: CONTINUE — concatenates partial + continuation")
        print(f"  Partial cached: \"{inf._chunks_before_crash[0]}...\"")
        if inf._recovery_prompts_received:
            print(f"  Recovery prompt excerpt: \"{inf._recovery_prompts_received[-1][:80]}...\"")
        print(f"  Final result: \"{result}\"")
        print(f"  (partial was prepended to the model's continuation)")

    elif mode == FallbackInferMode.REFERENCE:
        print(f"  Strategy: REFERENCE — show partial as context, get fresh response")
        if inf._recovery_prompts_received:
            print(f"  Recovery prompt excerpt: \"{inf._recovery_prompts_received[-1][:80]}...\"")
        print(f"  Final result: \"{result}\"")
        print(f"  (only the new response — no concatenation)")

    elif mode == FallbackInferMode.RESTART:
        print(f"  Strategy: RESTART — ignore cache, retry with original prompt")
        print(f"  Final result: \"{result}\"")
        print(f"  (completely fresh response, cache was not used)")

    return result


async def main():
    print("Streaming Recovery Demo: How cache-based recovery works")
    print("=" * 60)
    print()
    print("Scenario: A streaming inferencer yields 3 text chunks, then crashes.")
    print("The system caches each chunk as it arrives. On recovery, it reads the")
    print("cache and constructs a recovery prompt based on the configured mode.")
    print()
    print("Chunks that were streamed before crash: ['Once upon ', 'a time, ', 'in a land ']")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Demo each mode
        await demo_mode(FallbackInferMode.CONTINUE, os.path.join(tmpdir, "continue"))
        await demo_mode(FallbackInferMode.REFERENCE, os.path.join(tmpdir, "reference"))
        await demo_mode(FallbackInferMode.RESTART, os.path.join(tmpdir, "restart"))

    # Summary
    separator("Summary")
    print("  CONTINUE  — Best when partial output is valid and can be extended.")
    print("               Returns: cached_partial + model_continuation")
    print("  REFERENCE — Best for structured output (JSON/XML) where concatenation is risky.")
    print("               Returns: fresh_complete_response")
    print("  RESTART   — Best when partial output is unreliable or not useful.")
    print("               Returns: fresh_complete_response (cache ignored)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    asyncio.run(main())
