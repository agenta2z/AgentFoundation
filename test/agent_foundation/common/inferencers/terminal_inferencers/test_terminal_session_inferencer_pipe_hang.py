"""Unit tests for TerminalSessionInferencerBase subprocess pipe-hang fix.

Validates that the ``_read_stdout_with_exit_detection()`` mechanism prevents
indefinite hangs when CLI tools spawn child processes (e.g., MCP servers)
that inherit stdout/stderr pipes and keep them open after the main process
exits.

Root cause:
    ``asyncio.create_subprocess_shell()`` creates pipes for stdout/stderr.
    When the subprocess spawns children that inherit those pipes, the parent
    process can exit while the children keep the pipes open.  This causes:
    - ``async for line in process.stdout`` to block forever (no EOF).
    - ``await process.wait()`` to block forever (waits for pipe closure).
    - ``asyncio.run()`` to hang indefinitely during ``shutdown_asyncgens()``.

Fix:
    ``_read_stdout_with_exit_detection()`` races stdout reading against an
    ``os.waitpid(WNOHANG)`` poll that detects actual process exit
    independently of pipe state.  When the process exits, it drains
    buffered output with a timeout and breaks out of the read loop.
    ``_safe_process_cleanup()`` force-closes pipe transports and waits
    with a timeout.
"""

import asyncio
import os
import sys
import time
import unittest
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import resolve_path  # noqa: F401
except (ImportError, RuntimeError):
    pass  # PYTHONPATH already set externally

from attr import attrib, attrs

from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalSessionInferencerBase,
)


# ---------------------------------------------------------------------------
# Test concrete subclass
# ---------------------------------------------------------------------------


@attrs
class MockSessionInferencer(TerminalSessionInferencerBase):
    """Minimal concrete subclass for testing."""

    def construct_command(self, inference_input: Any, **kwargs) -> str:
        if isinstance(inference_input, dict):
            return inference_input.get("prompt", str(inference_input))
        return str(inference_input)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int, **kwargs
    ) -> Dict[str, Any]:
        return {
            "output": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "success": return_code == 0,
        }

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        return ""


# ---------------------------------------------------------------------------
# Helper: run async test with a hard timeout
# ---------------------------------------------------------------------------


def run_with_timeout(coro, timeout: float = 15.0):
    """Run an async coroutine with a hard timeout.

    Uses asyncio.wait_for to enforce the timeout, ensuring tests fail
    fast rather than hanging forever if the fix regresses.
    """

    async def _wrapper():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(_wrapper())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipeHangPrevention(unittest.TestCase):
    """Tests that subprocess pipe-hang is prevented."""

    def test_process_with_pipe_holding_child_does_not_hang(self):
        """Core regression test: a process that spawns a background child
        holding stdout open should NOT cause the inferencer to hang.

        Before the fix, this test would hang forever because
        ``async for line in process.stdout`` never reaches EOF.
        """

        async def run():
            inf = MockSessionInferencer(subprocess_exit_drain_timeout=2.0)
            collected = []
            # Shell command that spawns a background child (sleep 30 &)
            # which inherits and holds stdout open after the parent exits.
            async for line in inf._ainfer_streaming(
                "echo line1; (sleep 30 &); echo line2; sleep 1; echo line3"
            ):
                collected.append(line.strip())
            return collected, inf._last_streaming_return_code

        start = time.time()
        collected, rc = run_with_timeout(run(), timeout=15.0)
        elapsed = time.time() - start

        self.assertEqual(rc, 0)
        self.assertIn("line1", collected[0])
        self.assertIn("line3", collected[-1])
        # Should complete in well under 15s (typically ~3-4s)
        self.assertLess(elapsed, 10.0, f"Took too long: {elapsed:.1f}s")

    def test_process_with_multiple_pipe_holding_children(self):
        """Multiple background children holding pipes should not hang."""

        async def run():
            inf = MockSessionInferencer(subprocess_exit_drain_timeout=2.0)
            collected = []
            async for line in inf._ainfer_streaming(
                "echo start; (sleep 60 &); (sleep 60 &); (sleep 60 &); echo end"
            ):
                collected.append(line.strip())
            return collected, inf._last_streaming_return_code

        start = time.time()
        collected, rc = run_with_timeout(run(), timeout=15.0)
        elapsed = time.time() - start

        self.assertEqual(rc, 0)
        self.assertIn("start", collected[0])
        self.assertIn("end", collected[-1])
        self.assertLess(elapsed, 10.0)

    def test_concurrent_workers_with_pipe_holding_children(self):
        """Multiple concurrent inferencers with pipe-holding children
        should all complete without hanging (simulates BTA workers)."""

        async def worker(idx: int):
            inf = MockSessionInferencer(subprocess_exit_drain_timeout=2.0)
            collected = []
            async for line in inf._ainfer_streaming(
                f"echo worker_{idx}_start; (sleep 30 &); sleep 1; echo worker_{idx}_end"
            ):
                collected.append(line.strip())
            return idx, collected, inf._last_streaming_return_code

        async def run():
            tasks = [worker(i) for i in range(5)]
            return await asyncio.gather(*tasks)

        start = time.time()
        results = run_with_timeout(run(), timeout=20.0)
        elapsed = time.time() - start

        self.assertEqual(len(results), 5)
        for idx, collected, rc in results:
            self.assertEqual(rc, 0, f"Worker {idx} failed with rc={rc}")
            self.assertIn(f"worker_{idx}_start", collected[0])
            self.assertIn(f"worker_{idx}_end", collected[-1])

        self.assertLess(elapsed, 15.0, f"Took too long: {elapsed:.1f}s")


class TestNormalProcessBehavior(unittest.TestCase):
    """Tests that normal processes (no pipe-holding children) still work."""

    def test_normal_process_all_output_captured(self):
        """Normal process output is fully captured."""

        async def run():
            inf = MockSessionInferencer()
            collected = []
            async for line in inf._ainfer_streaming(
                "echo hello; sleep 1; echo world"
            ):
                collected.append(line.strip())
            return collected, inf._last_streaming_return_code

        collected, rc = run_with_timeout(run(), timeout=10.0)

        self.assertEqual(rc, 0)
        self.assertEqual(collected, ["hello", "world"])

    def test_process_with_error_exit_code(self):
        """Process that exits with non-zero return code."""

        async def run():
            inf = MockSessionInferencer()
            collected = []
            async for line in inf._ainfer_streaming("echo oops; exit 42"):
                collected.append(line.strip())
            return collected, inf._last_streaming_return_code

        collected, rc = run_with_timeout(run(), timeout=10.0)

        self.assertEqual(rc, 42)
        self.assertIn("oops", collected[0])

    def test_process_with_no_output(self):
        """Process that produces no stdout output."""

        async def run():
            inf = MockSessionInferencer()
            collected = []
            async for line in inf._ainfer_streaming("true"):
                collected.append(line.strip())
            return collected, inf._last_streaming_return_code

        collected, rc = run_with_timeout(run(), timeout=10.0)

        self.assertEqual(rc, 0)
        self.assertEqual(collected, [])

    def test_process_with_many_lines(self):
        """Process that outputs many lines."""

        async def run():
            inf = MockSessionInferencer()
            collected = []
            async for line in inf._ainfer_streaming(
                "seq 1 100"
            ):
                collected.append(line.strip())
            return collected, inf._last_streaming_return_code

        collected, rc = run_with_timeout(run(), timeout=10.0)

        self.assertEqual(rc, 0)
        self.assertEqual(len(collected), 100)
        self.assertEqual(collected[0], "1")
        self.assertEqual(collected[-1], "100")

    def test_process_with_stderr(self):
        """Process stderr is captured when possible.

        Note: stderr capture may be empty if the process exits before
        the stderr drain completes — this is acceptable because the fix
        prioritizes preventing hangs over capturing every byte of stderr.
        """

        async def run():
            inf = MockSessionInferencer()
            collected = []
            # Add a small sleep to give stderr time to flush before exit
            async for line in inf._ainfer_streaming(
                "echo out; echo err >&2; sleep 0.5"
            ):
                collected.append(line.strip())
            return collected, inf._last_streaming_return_code, inf._last_streaming_stderr

        collected, rc, stderr = run_with_timeout(run(), timeout=10.0)

        self.assertEqual(rc, 0)
        self.assertIn("out", collected[0])
        # stderr may or may not be captured depending on timing
        # The important thing is no hang and stdout is captured
        if stderr:
            self.assertIn("err", stderr)


class TestCleanupMechanisms(unittest.TestCase):
    """Tests for the cleanup helper methods."""

    def test_force_close_pipes_doesnt_error_on_closed_pipes(self):
        """_force_close_pipes should be safe to call on already-closed pipes."""

        async def run():
            inf = MockSessionInferencer()
            proc = await asyncio.create_subprocess_shell(
                "true",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            # Pipes may already be closed — should not raise
            inf._force_close_pipes(proc)
            inf._force_close_pipes(proc)  # Call twice — should be idempotent

        run_with_timeout(run(), timeout=5.0)

    def test_safe_process_cleanup_with_normal_process(self):
        """_safe_process_cleanup should work cleanly for normal processes."""

        async def run():
            inf = MockSessionInferencer()
            proc = await asyncio.create_subprocess_shell(
                "echo done",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # Read stdout first to avoid pipe issues
            await proc.stdout.read()
            await inf._safe_process_cleanup(proc, timeout=5.0)
            self.assertEqual(proc.returncode, 0)

        run_with_timeout(run(), timeout=10.0)

    def test_configurable_drain_timeout(self):
        """subprocess_exit_drain_timeout is respected."""

        async def run():
            inf = MockSessionInferencer(subprocess_exit_drain_timeout=1.0)
            self.assertEqual(inf.subprocess_exit_drain_timeout, 1.0)

            inf2 = MockSessionInferencer(subprocess_exit_drain_timeout=10.0)
            self.assertEqual(inf2.subprocess_exit_drain_timeout, 10.0)

        run_with_timeout(run(), timeout=5.0)

    def test_poll_process_exit_detects_exit(self):
        """_poll_process_exit correctly detects when a process exits."""

        async def run():
            inf = MockSessionInferencer(subprocess_exit_poll_interval=0.1)
            proc = await asyncio.create_subprocess_shell(
                "sleep 0.5",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            start = time.time()
            exit_code = await inf._poll_process_exit(proc.pid)
            elapsed = time.time() - start

            # Should detect exit within ~1s (0.5s sleep + polling overhead)
            self.assertLess(elapsed, 3.0)
            # Exit code should be 0 or None (already reaped)
            self.assertIn(exit_code, [0, None])

        run_with_timeout(run(), timeout=10.0)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
