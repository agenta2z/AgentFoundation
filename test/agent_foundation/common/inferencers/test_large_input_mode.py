"""Tests for LargeInputMode enum, stdin handling, and DevMate CLI bypass fixes."""

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    _convert_large_input_mode,
    LargeInputMode,
    TerminalInferencerResponse,
    TerminalSessionInferencerBase,
)


# === Concrete subclass for testing ===


class _StubTerminalInferencer(TerminalSessionInferencerBase):
    """Minimal concrete subclass for testing base class behavior."""

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        if isinstance(inference_input, dict):
            prompt = inference_input.get("prompt", "")
        else:
            prompt = str(inference_input)
        use_stdin = kwargs.get("use_stdin", False)
        parts = ["echo"]
        if not use_stdin:
            parts.append(f'"{prompt}"')
        return " ".join(parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> TerminalInferencerResponse:
        return TerminalInferencerResponse(
            {
                "output": stdout.strip(),
                "stderr": stderr.strip(),
                "return_code": return_code,
                "success": return_code == 0,
            }
        )

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        return f"--session {session_id}" if session_id else ""


# ==============================================================================
# Phase 1 Tests: LargeInputMode enum + attributes
# ==============================================================================


class TestLargeInputModeEnum(unittest.TestCase):
    """Test LargeInputMode enum values and converter."""

    def test_enum_values(self):
        self.assertEqual(LargeInputMode.INLINE.value, "inline")
        self.assertEqual(LargeInputMode.STDIN.value, "stdin")
        self.assertEqual(LargeInputMode.FILE.value, "file")

    def test_string_converter_lowercase(self):
        self.assertEqual(_convert_large_input_mode("stdin"), LargeInputMode.STDIN)
        self.assertEqual(_convert_large_input_mode("file"), LargeInputMode.FILE)
        self.assertEqual(_convert_large_input_mode("inline"), LargeInputMode.INLINE)

    def test_string_converter_uppercase(self):
        self.assertEqual(_convert_large_input_mode("STDIN"), LargeInputMode.STDIN)
        self.assertEqual(_convert_large_input_mode("FILE"), LargeInputMode.FILE)

    def test_string_converter_mixedcase(self):
        self.assertEqual(_convert_large_input_mode("Stdin"), LargeInputMode.STDIN)

    def test_enum_passthrough(self):
        self.assertEqual(
            _convert_large_input_mode(LargeInputMode.FILE), LargeInputMode.FILE
        )

    def test_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            _convert_large_input_mode("invalid")

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            _convert_large_input_mode(42)


class TestLargeInputModeAttrib(unittest.TestCase):
    """Test large_input_mode attribute on the inferencer."""

    def test_default_inline(self):
        inf = _StubTerminalInferencer()
        self.assertEqual(inf.large_input_mode, LargeInputMode.INLINE)

    def test_explicit_stdin(self):
        inf = _StubTerminalInferencer(large_input_mode=LargeInputMode.STDIN)
        self.assertEqual(inf.large_input_mode, LargeInputMode.STDIN)

    def test_string_stdin(self):
        inf = _StubTerminalInferencer(large_input_mode="stdin")
        self.assertEqual(inf.large_input_mode, LargeInputMode.STDIN)


class TestBackwardCompatAutoPromotion(unittest.TestCase):
    """Test __attrs_post_init__ auto-promotes INLINE → FILE when offload config is set."""

    def test_auto_promote_with_bool_true(self):
        inf = _StubTerminalInferencer(use_file_for_large_arg_exceeding_size=True)
        self.assertEqual(inf.large_input_mode, LargeInputMode.FILE)

    def test_auto_promote_with_list(self):
        inf = _StubTerminalInferencer(use_file_for_large_arg_exceeding_size=["prompt"])
        self.assertEqual(inf.large_input_mode, LargeInputMode.FILE)

    def test_no_promote_when_none(self):
        inf = _StubTerminalInferencer(use_file_for_large_arg_exceeding_size=None)
        self.assertEqual(inf.large_input_mode, LargeInputMode.INLINE)

    def test_no_promote_when_false(self):
        inf = _StubTerminalInferencer(use_file_for_large_arg_exceeding_size=False)
        self.assertEqual(inf.large_input_mode, LargeInputMode.INLINE)

    def test_no_promote_when_explicit_stdin(self):
        """If user explicitly sets STDIN, offload config should NOT override."""
        inf = _StubTerminalInferencer(
            large_input_mode=LargeInputMode.STDIN,
            use_file_for_large_arg_exceeding_size=True,
        )
        self.assertEqual(inf.large_input_mode, LargeInputMode.STDIN)


class TestAttrsPostInitChain(unittest.TestCase):
    """Test that __attrs_post_init__ chain is properly called."""

    def test_parent_post_init_called(self):
        """Verify InferencerBase's __attrs_post_init__ runs (it handles post_response_merger)."""
        inf = _StubTerminalInferencer(post_response_merger="default")
        # InferencerBase.__attrs_post_init__ resolves "default" to a callable
        self.assertTrue(callable(inf.post_response_merger))


# ==============================================================================
# Phase 2 Tests: stdin subprocess, stderr fix, timeout
# ==============================================================================


class TestStdinSubprocessAsync(unittest.TestCase):
    """Test _ainfer_streaming stdin support (STDIN mode)."""

    def test_stdin_mode_uses_pipe(self):
        """Mock subprocess and verify stdin=PIPE + write+drain+close when mode=STDIN."""
        inf = _StubTerminalInferencer(large_input_mode=LargeInputMode.STDIN)

        mock_process = MagicMock()
        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdin.close = MagicMock()
        mock_process.stdin = mock_stdin

        mock_stdout_data = [b"line1\n", b"line2\n"]

        async def mock_stdout_iter():
            for line in mock_stdout_data:
                yield line

        mock_process.stdout = mock_stdout_iter()

        mock_stderr = MagicMock()
        mock_stderr.read = AsyncMock(return_value=b"")
        mock_process.stderr = mock_stderr
        mock_process.wait = AsyncMock()
        mock_process.returncode = 0

        async def run():
            with patch(
                "asyncio.create_subprocess_shell",
                new=AsyncMock(return_value=mock_process),
            ) as mock_create:
                lines = []
                async for line in inf._ainfer_streaming("test prompt"):
                    lines.append(line)

                # Verify stdin was opened (PIPE)
                call_kwargs = mock_create.call_args
                self.assertEqual(
                    call_kwargs.kwargs.get("stdin") or call_kwargs[1].get("stdin"),
                    asyncio.subprocess.PIPE,
                )

                # Verify stdin operations
                mock_stdin.write.assert_called_once_with(b"test prompt")
                mock_stdin.drain.assert_awaited_once()
                mock_stdin.close.assert_called_once()

                self.assertEqual(lines, ["line1\n", "line2\n"])

        asyncio.run(run())


class TestStdinSubprocessSync(unittest.TestCase):
    """Test _infer() stdin support (STDIN mode)."""

    def test_stdin_mode_uses_input(self):
        """Verify input=prompt passed to subprocess.run() when mode=STDIN."""
        inf = _StubTerminalInferencer(large_input_mode=LargeInputMode.STDIN)

        mock_result = MagicMock()
        mock_result.stdout = "output text"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = inf._infer("test prompt")

            call_kwargs = mock_run.call_args
            self.assertEqual(
                call_kwargs.kwargs.get("input") or call_kwargs[1].get("input"),
                "test prompt",
            )
            self.assertTrue(result["success"])


class TestFileModeExclusivity(unittest.TestCase):
    """Test that _maybe_offload_large_args_to_file() only called in FILE mode."""

    def test_offload_not_called_in_stdin_mode(self):
        inf = _StubTerminalInferencer(large_input_mode=LargeInputMode.STDIN)

        with patch.object(inf, "_maybe_offload_large_args_to_file") as mock_offload:
            mock_result = MagicMock()
            mock_result.stdout = "output"
            mock_result.stderr = ""
            mock_result.returncode = 0
            with patch("subprocess.run", return_value=mock_result):
                inf._infer("test")
            mock_offload.assert_not_called()

    def test_offload_not_called_in_inline_mode(self):
        inf = _StubTerminalInferencer(large_input_mode=LargeInputMode.INLINE)

        with patch.object(inf, "_maybe_offload_large_args_to_file") as mock_offload:
            mock_result = MagicMock()
            mock_result.stdout = "output"
            mock_result.stderr = ""
            mock_result.returncode = 0
            with patch("subprocess.run", return_value=mock_result):
                inf._infer("test")
            mock_offload.assert_not_called()


class TestTempFilesInitialization(unittest.TestCase):
    """Test that temp_files is properly initialized to avoid NameError in finally."""

    def test_no_name_error_in_stdin_mode(self):
        """In STDIN mode, temp_files must be initialized before try/finally."""
        inf = _StubTerminalInferencer(large_input_mode=LargeInputMode.STDIN)

        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            # Should NOT raise NameError for temp_files
            result = inf._infer("test")
            self.assertTrue(result["success"])


class TestStderrFixInAinferStreaming(unittest.TestCase):
    """Test that _ainfer_streaming reads stderr BEFORE process.wait()."""

    def test_stderr_read_before_wait(self):
        """Verify stderr is read before process.wait() to prevent deadlock."""
        inf = _StubTerminalInferencer(large_input_mode=LargeInputMode.INLINE)

        call_order = []

        mock_process = MagicMock()
        mock_process.stdout = MagicMock()

        async def mock_stdout_iter():
            yield b"line1\n"

        mock_process.stdout = mock_stdout_iter()

        mock_stderr = MagicMock()

        async def mock_stderr_read():
            call_order.append("stderr_read")
            return b"some stderr"

        mock_stderr.read = mock_stderr_read
        mock_process.stderr = mock_stderr

        async def mock_wait():
            call_order.append("wait")

        mock_process.wait = mock_wait
        mock_process.returncode = 0

        async def run():
            with patch(
                "asyncio.create_subprocess_shell",
                new=AsyncMock(return_value=mock_process),
            ):
                lines = []
                async for line in inf._ainfer_streaming("test"):
                    lines.append(line)

            # stderr_read must come BEFORE wait
            self.assertEqual(call_order, ["stderr_read", "wait"])
            self.assertEqual(inf._last_streaming_stderr, "some stderr")

        asyncio.run(run())


class TestInferSubprocessTimeout(unittest.TestCase):
    """Test that _infer() passes timeout to subprocess.run()."""

    def test_timeout_passed_to_subprocess_run(self):
        """Verify timeout= is set on subprocess.run() call."""
        inf = _StubTerminalInferencer(idle_timeout_seconds=300)

        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            inf._infer("test")

            call_kwargs = mock_run.call_args
            timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
            # Should be max(idle_timeout_seconds, 1800) = 1800
            self.assertEqual(timeout, 1800)

    def test_timeout_zero_disabled(self):
        """idle_timeout_seconds=0 should result in timeout=None."""
        inf = _StubTerminalInferencer(idle_timeout_seconds=0)

        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            inf._infer("test")

            call_kwargs = mock_run.call_args
            timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
            self.assertIsNone(timeout)

    def test_per_call_timeout_override(self):
        """subprocess_timeout_seconds kwarg should override default."""
        inf = _StubTerminalInferencer(idle_timeout_seconds=300)

        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            inf._infer("test", subprocess_timeout_seconds=60)

            call_kwargs = mock_run.call_args
            timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
            self.assertEqual(timeout, 60)

    def test_subprocess_timeout_consumed_from_kwargs(self):
        """subprocess_timeout_seconds should be consumed (not passed to construct_command)."""
        inf = _StubTerminalInferencer(idle_timeout_seconds=300)

        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            with patch.object(
                inf, "construct_command", wraps=inf.construct_command
            ) as mock_cmd:
                inf._infer("test", subprocess_timeout_seconds=60)

                # subprocess_timeout_seconds should NOT appear in kwargs to construct_command
                call_kwargs = mock_cmd.call_args[1]
                self.assertNotIn("subprocess_timeout_seconds", call_kwargs)


class TestInferStreamingStderr(unittest.TestCase):
    """Test that _infer_streaming() reads stderr in finally block."""

    def test_stderr_captured_in_sync_streaming(self):
        """Verify _last_streaming_stderr is set after sync streaming."""
        inf = _StubTerminalInferencer(large_input_mode=LargeInputMode.INLINE)

        mock_process = MagicMock()
        mock_process.stdout = iter(["line1\n", "line2\n"])
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = "some stderr output"
        mock_process.wait.return_value = None
        mock_process.returncode = 0

        with patch("subprocess.Popen", return_value=mock_process):
            lines = list(inf._infer_streaming("test"))

        self.assertEqual(lines, ["line1\n", "line2\n"])
        self.assertEqual(inf._last_streaming_stderr, "some stderr output")


# ==============================================================================
# Phase 3 Tests: ClaudeCode CLI simplification
# ==============================================================================


class TestClaudeCodeCliStdinMode(unittest.TestCase):
    """Test that ClaudeCodeCliInferencer defaults to STDIN mode."""

    def test_default_large_input_mode_is_stdin(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
            ClaudeCodeCliInferencer,
        )

        inf = ClaudeCodeCliInferencer()
        self.assertEqual(inf.large_input_mode, LargeInputMode.STDIN)


_MOCK_SYNC = "agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer.sync_config_to_target"


# ==============================================================================
# Phase 4 Tests: DevMate CLI ainfer() fix
# ==============================================================================


class TestDevmateCliAinferFix(unittest.TestCase):
    """Test that DevMate ainfer() routes through _ainfer_single()."""

    def test_ainfer_calls_ainfer_single(self):
        """Verify ainfer() invokes _ainfer_single() (not _ainfer() directly)."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
            DevmateCliInferencer,
        )

        with patch(_MOCK_SYNC):
            inf = DevmateCliInferencer(target_path="/test/repo")

        mock_result = TerminalInferencerResponse(
            {"output": "test", "success": True, "session_id": "abc123"}
        )

        async def run():
            with patch.object(
                inf, "_ainfer_single", new=AsyncMock(return_value=mock_result)
            ) as mock_ainfer_single:
                result = await inf.ainfer("test prompt")

                mock_ainfer_single.assert_awaited_once()
                self.assertEqual(result["output"], "test")
                self.assertEqual(result["session_id"], "abc123")

        asyncio.run(run())

    def test_ainfer_preserves_session_kwargs(self):
        """Verify session_id and resume are passed through to _ainfer_single()."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
            DevmateCliInferencer,
        )

        with patch(_MOCK_SYNC):
            inf = DevmateCliInferencer(target_path="/test/repo")
        inf.active_session_id = "existing-session"

        mock_result = TerminalInferencerResponse(
            {"output": "test", "success": True, "session_id": "existing-session"}
        )

        async def run():
            with patch.object(
                inf, "_ainfer_single", new=AsyncMock(return_value=mock_result)
            ) as mock_ainfer_single:
                await inf.ainfer("follow up")

                call_kwargs = mock_ainfer_single.call_args[1]
                self.assertEqual(call_kwargs["session_id"], "existing-session")
                self.assertTrue(call_kwargs["resume"])

        asyncio.run(run())


# ==============================================================================
# Phase 5 Tests: DevMate CLI ainfer_streaming() idle timeout
# ==============================================================================


class TestDevmateCliAinferStreamingIdleTimeout(unittest.TestCase):
    """Test idle timeout in ainfer_streaming() via _ainfer_streaming hang."""

    def test_timeout_error_on_hang(self):
        """When _ainfer_streaming hangs, ainfer_streaming should raise TimeoutError."""
        inf = _StubTerminalInferencer(idle_timeout_seconds=1)
        # Keep cleanup fast for the test
        inf._generator_cleanup_timeout = 0.5

        hang_event = None

        async def hanging_generator(*args, **kwargs):
            nonlocal hang_event
            yield "line1\n"
            hang_event = asyncio.Event()
            await hang_event.wait()
            yield "never reached\n"

        async def run():
            with patch.object(inf, "_ainfer_streaming", side_effect=hanging_generator):
                lines = []
                with self.assertRaises(asyncio.TimeoutError):
                    async for line in inf.ainfer_streaming(
                        "test",
                        idle_timeout_seconds=0.1,
                    ):
                        lines.append(line)

                # Should have received the first line before timeout
                self.assertEqual(lines, ["line1\n"])

        asyncio.run(run())


# ==============================================================================
# Phase 7 Tests: DevMate CLI construct_command use_stdin
# ==============================================================================


class TestDevmateCliUseStdin(unittest.TestCase):
    """Test use_stdin in DevMate construct_command()."""

    def test_use_stdin_skips_prompt(self):
        """When use_stdin=True, prompt should not appear in command."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
            DevmateCliInferencer,
        )

        with patch(_MOCK_SYNC):
            inf = DevmateCliInferencer(target_path="/test/repo", no_create_commit=False)

        command_with_stdin = inf.construct_command("Hello world", use_stdin=True)
        command_without_stdin = inf.construct_command("Hello world")

        # With stdin: no prompt= in command
        self.assertNotIn('"prompt=Hello world"', command_with_stdin)
        # Without stdin: prompt= should be in command
        self.assertIn('"prompt=Hello world"', command_without_stdin)

        # Both should have model_name and max_output_tokens
        self.assertIn('"model_name=', command_with_stdin)
        self.assertIn('"max_output_tokens=', command_with_stdin)


# ==============================================================================
# Integration Tests: Full chain
# ==============================================================================


class TestChainInteraction(unittest.TestCase):
    """Test ainfer() → _ainfer_single() → ... produces correct result with session kwargs."""

    def test_chain_with_session(self):
        """Test the full chain: ainfer() → _ainfer_single() → _ainfer() → ainfer_streaming()."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
            DevmateCliInferencer,
        )

        with patch(_MOCK_SYNC):
            inf = DevmateCliInferencer(target_path="/test/repo")

        mock_result = TerminalInferencerResponse(
            {
                "output": "Hello from chain",
                "success": True,
                "session_id": "chain-session-id",
            }
        )

        async def run():
            # Mock _ainfer to return our result (bypass subprocess)
            with patch.object(inf, "_ainfer", new=AsyncMock(return_value=mock_result)):
                result = await inf.ainfer("test prompt", new_session=True)

                self.assertEqual(result["output"], "Hello from chain")
                self.assertEqual(inf.active_session_id, "chain-session-id")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
