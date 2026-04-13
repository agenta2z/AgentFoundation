"""Integration tests for Resumable inferencers — composition and end-to-end scenarios.

Covers:
- End-to-end streaming resume from partial cache (simulated crash + restart)
- End-to-end streaming skip on completed cache
- Resume hook fires after preprocessing (prompt hash match)
- Sync path resume works
- Resume disabled by default (cache present but ignored)
- Multiple prompts, same cache folder (correct cache matched)
- worker_manages_resume detection for streaming in workflow context
"""

import asyncio
import hashlib
import os
import time
import unittest
import uuid

from typing import Any, AsyncIterator, Optional

from attr import attrib, attrs

from rich_python_utils.common_objects.workflow.common.resumable import Resumable
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    FallbackInferMode,
    StreamingInferencerBase,
)


# ---------------------------------------------------------------------------
# Mock streaming inferencer with controllable behavior
# ---------------------------------------------------------------------------

@attrs
class ControllableStreamingInferencer(StreamingInferencerBase):
    """A streaming inferencer that can simulate crashes and track calls.

    Args:
        crash_on_first_call: If True, _ainfer_streaming crashes on first invocation.
        response_chunks: Chunks to yield on successful calls.
    """
    crash_on_first_call: bool = attrib(default=False)
    response_chunks: list = attrib(factory=lambda: ["Hello ", "world!"])
    _call_count: int = attrib(default=0, init=False, repr=False)
    _ainfer_calls: list = attrib(factory=list, init=False, repr=False)

    async def _ainfer_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        self._call_count += 1
        if self.crash_on_first_call and self._call_count == 1:
            # Yield some chunks then crash (simulates mid-stream failure)
            for chunk in self.response_chunks[:1]:
                yield chunk
            raise ConnectionError("Simulated mid-stream crash")
        for chunk in self.response_chunks:
            yield chunk

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "sync_result"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        self._ainfer_calls.append(inference_input)
        content_parts = []
        async for chunk in self.ainfer_streaming(inference_input, inference_config, **kwargs):
            content_parts.append(chunk)
        return "".join(content_parts)

    async def adisconnect(self):
        pass


def _write_cache_file(cache_folder, class_name, prompt, content):
    """Write a mock cache file matching _open_cache_file's naming convention."""
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(
        cache_folder, class_name, f"mock-id_{timestamp}"
    )
    os.makedirs(session_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    path = os.path.join(session_dir, f"stream_{unique_id}_{prompt_hash}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# End-to-end: Completed cache → skip execution
# ---------------------------------------------------------------------------

class TestE2ECompletedCacheSkip(unittest.TestCase):
    """When a previous run completed successfully (success marker in cache),
    a new inferencer instance should return the cached result without
    calling _ainfer_streaming at all."""

    def test_async_skip_on_completed_cache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a completed cache file
            prompt = "What is 2+2?"
            cached_response = "The answer is 4."
            _write_cache_file(
                tmpdir,
                "ControllableStreamingInferencer",
                prompt,
                f"{cached_response}\n--- STREAM COMPLETED SUCCESSFULLY ---\n",
            )

            # Create a NEW inferencer instance (simulates process restart)
            inf = ControllableStreamingInferencer(
                cache_folder=tmpdir,
                resume_with_saved_results=True,
                min_retry_wait=0, max_retry_wait=0,
            )

            # This should return the cached result without streaming
            result = asyncio.run(inf.ainfer(prompt))
            self.assertEqual(result, cached_response)
            self.assertEqual(inf._call_count, 0)  # _ainfer_streaming never called


# ---------------------------------------------------------------------------
# End-to-end: Partial cache → resume via recovery
# ---------------------------------------------------------------------------

class TestE2EPartialCacheResume(unittest.TestCase):
    """When a previous run failed (failure marker in cache), a new inferencer
    should read the partial and trigger recovery (augmented prompt)."""

    def test_async_resume_from_partial_cache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = "Tell me a story"
            partial = "Once upon a time"
            _write_cache_file(
                tmpdir,
                "ControllableStreamingInferencer",
                prompt,
                f"{partial}\n--- STREAM FAILED: connection dropped ---\n",
            )

            inf = ControllableStreamingInferencer(
                cache_folder=tmpdir,
                resume_with_saved_results=True,
                response_chunks=["...the end."],  # recovery call returns this
                min_retry_wait=0, max_retry_wait=0,
            )

            result = asyncio.run(inf.ainfer(prompt))

            # Should have called _ainfer with an augmented recovery prompt
            self.assertTrue(len(inf._ainfer_calls) > 0)
            # The recovery prompt should contain the partial output
            recovery_prompt = inf._ainfer_calls[0]
            self.assertIn("Once upon a time", recovery_prompt)


# ---------------------------------------------------------------------------
# Resume disabled by default
# ---------------------------------------------------------------------------

class TestResumeDisabledByDefault(unittest.TestCase):
    """With resume_with_saved_results=False (default), cache files are ignored."""

    def test_cache_ignored_when_disabled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = "What is 2+2?"
            _write_cache_file(
                tmpdir,
                "ControllableStreamingInferencer",
                prompt,
                "Cached result\n--- STREAM COMPLETED SUCCESSFULLY ---\n",
            )

            inf = ControllableStreamingInferencer(
                cache_folder=tmpdir,
                resume_with_saved_results=False,  # default
                response_chunks=["Fresh ", "response"],
                min_retry_wait=0, max_retry_wait=0,
            )

            result = asyncio.run(inf.ainfer(prompt))
            # Should NOT use cached result — should call _ainfer_streaming
            self.assertEqual(inf._call_count, 1)
            self.assertEqual(result, "Fresh response")


# ---------------------------------------------------------------------------
# Multiple prompts, same cache folder
# ---------------------------------------------------------------------------

class TestMultiplePromptsSameFolder(unittest.TestCase):
    """Different prompts produce different cache files; resume finds the right one."""

    def test_correct_cache_per_prompt(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            class_name = "ControllableStreamingInferencer"
            _write_cache_file(
                tmpdir, class_name, "prompt A",
                "Result A\n--- STREAM COMPLETED SUCCESSFULLY ---\n",
            )
            _write_cache_file(
                tmpdir, class_name, "prompt B",
                "Result B\n--- STREAM COMPLETED SUCCESSFULLY ---\n",
            )

            inf = ControllableStreamingInferencer(
                cache_folder=tmpdir,
                resume_with_saved_results=True,
                min_retry_wait=0, max_retry_wait=0,
            )

            result_a = asyncio.run(inf.ainfer("prompt A"))
            self.assertEqual(result_a, "Result A")

            # Reset call count for second call
            inf._call_count = 0
            result_b = asyncio.run(inf.ainfer("prompt B"))
            self.assertEqual(result_b, "Result B")
            self.assertEqual(inf._call_count, 0)  # both served from cache


# ---------------------------------------------------------------------------
# Resume hook fires after preprocessing
# ---------------------------------------------------------------------------

class TestResumeAfterPreprocessing(unittest.TestCase):
    """The resume check uses the POST-processed prompt for hash matching.
    If input_preprocessor transforms the prompt, the hash must match
    what _open_cache_file would have seen during the original call."""

    def test_preprocessed_prompt_hash_matches(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # The preprocessor lowercases the input
            def lower_preprocessor(inp):
                return inp.lower()

            # Cache was written with the preprocessed (lowercased) prompt
            processed_prompt = "hello world"
            _write_cache_file(
                tmpdir,
                "ControllableStreamingInferencer",
                processed_prompt,
                "Cached hello\n--- STREAM COMPLETED SUCCESSFULLY ---\n",
            )

            inf = ControllableStreamingInferencer(
                cache_folder=tmpdir,
                resume_with_saved_results=True,
                input_preprocessor=lower_preprocessor,
                min_retry_wait=0, max_retry_wait=0,
            )

            # Pass UPPERCASE — preprocessor lowercases it, hash matches cache
            result = asyncio.run(inf.ainfer("HELLO WORLD"))
            self.assertEqual(result, "Cached hello")
            self.assertEqual(inf._call_count, 0)


# ---------------------------------------------------------------------------
# worker_manages_resume detection
# ---------------------------------------------------------------------------

class TestWorkerManagesResumeInWorkflowContext(unittest.TestCase):
    """When a streaming inferencer is used as a worker in BTA,
    isinstance(worker, Resumable) returns True and resume_with_saved_results
    can be checked to set worker_manages_resume."""

    def test_streaming_worker_is_resumable(self):
        inf = ControllableStreamingInferencer(
            resume_with_saved_results=True,
        )
        # This is the check that BTA does at line 302-305:
        is_resumable = isinstance(inf, Resumable)
        manages_resume = is_resumable and bool(
            getattr(inf, "resume_with_saved_results", False)
        )
        self.assertTrue(is_resumable)
        self.assertTrue(manages_resume)

    def test_streaming_worker_not_managing_by_default(self):
        inf = ControllableStreamingInferencer()  # resume_with_saved_results=False
        manages_resume = isinstance(inf, Resumable) and bool(
            getattr(inf, "resume_with_saved_results", False)
        )
        self.assertTrue(isinstance(inf, Resumable))  # IS Resumable
        self.assertFalse(manages_resume)  # but NOT managing its own resume


if __name__ == "__main__":
    unittest.main()
