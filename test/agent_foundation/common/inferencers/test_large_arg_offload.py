# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""Unit tests for large argument file offload and system_helper utilities.

Tests cover:
- system_helper: get_arg_max, get_available_arg_space, get_max_single_arg_size, get_current_platform
- TerminalSessionInferencerBase: _resolve_large_arg_config,
  _maybe_offload_large_args_to_file, _cleanup_temp_files
"""

import os
import tempfile
import unittest
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from unittest.mock import patch

from attr import attrib, attrs
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalInferencerResponse,
    TerminalSessionInferencerBase,
)
from rich_python_utils.common_utils.system_helper import (
    get_arg_max,
    get_available_arg_space,
    get_current_platform,
    get_max_single_arg_size,
    OperatingSystem,
)


# ---------------------------------------------------------------------------
# Concrete subclass for testing (TerminalSessionInferencerBase is abstract)
# ---------------------------------------------------------------------------
@attrs
class _StubInferencer(TerminalSessionInferencerBase):
    """Minimal concrete subclass for testing base-class helpers."""

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        if isinstance(inference_input, dict):
            return f"echo {inference_input.get('prompt', '')}"
        return f"echo {inference_input}"

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> TerminalInferencerResponse:
        return TerminalInferencerResponse(
            output=stdout, stderr=stderr, return_code=return_code
        )

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        return f"--session-id {session_id}"


# ===================================================================
# Tests for system_helper.py
# ===================================================================


class SystemHelperTest(unittest.TestCase):
    """Tests for system_helper utilities."""

    def test_get_arg_max_returns_positive_int(self) -> None:
        result = get_arg_max()
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_get_arg_max_fallback_when_sysconf_unavailable(self) -> None:
        with patch("os.sysconf", side_effect=AttributeError):
            result = get_arg_max()
            self.assertGreater(result, 0)

    def test_get_available_arg_space_returns_non_negative(self) -> None:
        result = get_available_arg_space()
        self.assertGreaterEqual(result, 0)

    def test_get_available_arg_space_with_custom_safety_margin(self) -> None:
        result_small = get_available_arg_space(safety_margin=0)
        result_large = get_available_arg_space(safety_margin=100_000)
        self.assertGreaterEqual(result_small, result_large)

    def test_get_max_single_arg_size_returns_positive(self) -> None:
        result = get_max_single_arg_size()
        self.assertGreater(result, 0)

    def test_get_max_single_arg_size_with_custom_safety_margin(self) -> None:
        result_small = get_max_single_arg_size(safety_margin=0)
        result_large = get_max_single_arg_size(safety_margin=50_000)
        self.assertGreater(result_small, result_large)

    def test_get_max_single_arg_size_fallback_when_sysconf_unavailable(self) -> None:
        with patch("os.sysconf", side_effect=AttributeError):
            result = get_max_single_arg_size()
            # Fallback: 131072 - 8192 = 122880
            self.assertEqual(result, 122_880)

    def test_get_max_single_arg_size_is_less_than_available_arg_space(self) -> None:
        """MAX_ARG_STRLEN (128 KB) should be much less than ARG_MAX (~2 MB)."""
        max_single = get_max_single_arg_size(safety_margin=0)
        available = get_available_arg_space(safety_margin=0)
        self.assertLess(max_single, available)

    def test_get_current_platform_linux(self) -> None:
        result = get_current_platform(system_str="linux", platform_str="Linux-5.4")
        self.assertEqual(result, OperatingSystem.LINUX)

    def test_get_current_platform_macos(self) -> None:
        result = get_current_platform(system_str="darwin", platform_str="Darwin-20.6")
        self.assertEqual(result, OperatingSystem.MACOS)

    def test_get_current_platform_windows(self) -> None:
        result = get_current_platform(system_str="windows", platform_str="Windows-10")
        self.assertEqual(result, OperatingSystem.WINDOWS)

    def test_get_current_platform_ios(self) -> None:
        result = get_current_platform(
            system_str="darwin", platform_str="Darwin-iPhoneOS-16.0.0"
        )
        self.assertEqual(result, OperatingSystem.IOS)

    def test_get_current_platform_android(self) -> None:
        result = get_current_platform(
            system_str="linux", platform_str="Linux-5.4.0-android"
        )
        self.assertEqual(result, OperatingSystem.ANDROID)

    def test_get_current_platform_mobile_detection_disabled(self) -> None:
        result = get_current_platform(
            identify_mobile_operating_system=False,
            system_str="darwin",
            platform_str="Darwin-iPhoneOS-16.0.0",
        )
        self.assertEqual(result, OperatingSystem.MACOS)


# ===================================================================
# Tests for _resolve_large_arg_config
# ===================================================================


class ResolveLargeArgConfigTest(unittest.TestCase):
    """Tests for TerminalSessionInferencerBase._resolve_large_arg_config."""

    def _make(
        self,
        val: Optional[Union[bool, int, List[str], Dict[str, int]]] = None,
    ) -> _StubInferencer:
        return _StubInferencer(use_file_for_large_arg_exceeding_size=val)

    def test_none_disables(self) -> None:
        self.assertIsNone(self._make(None)._resolve_large_arg_config())

    def test_false_disables(self) -> None:
        self.assertIsNone(self._make(False)._resolve_large_arg_config())

    def test_true_returns_wildcard_with_default_threshold(self) -> None:
        config = self._make(True)._resolve_large_arg_config()
        self.assertIsNotNone(config)
        self.assertIn("*", config)
        self.assertEqual(config["*"], 65_536)

    def test_positive_int_returns_wildcard(self) -> None:
        config = self._make(50_000)._resolve_large_arg_config()
        self.assertIsNotNone(config)
        self.assertEqual(config["*"], 50_000)

    def test_zero_returns_wildcard_zero(self) -> None:
        config = self._make(0)._resolve_large_arg_config()
        self.assertIsNotNone(config)
        self.assertEqual(config["*"], 0)

    def test_negative_int_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self._make(-1)._resolve_large_arg_config()

    def test_list_mode_returns_per_key_thresholds(self) -> None:
        config = self._make(["prompt"])._resolve_large_arg_config()
        self.assertIsNotNone(config)
        self.assertIn("prompt", config)
        self.assertGreater(config["prompt"], 0)

    def test_list_mode_divides_among_args(self) -> None:
        config_one = self._make(["prompt"])._resolve_large_arg_config()
        config_two = self._make(["prompt", "context"])._resolve_large_arg_config()
        self.assertIsNotNone(config_one)
        self.assertIsNotNone(config_two)
        # Two args should have ~half the per-arg threshold of one arg
        self.assertGreater(config_one["prompt"], config_two["prompt"])

    def test_list_mode_respects_max_arg_strlen(self) -> None:
        """Threshold must not exceed MAX_ARG_STRLEN to prevent E2BIG.

        On Linux, each individual argument to execve() is limited to
        MAX_ARG_STRLEN = PAGE_SIZE * 32 (typically 128 KB).  When using
        create_subprocess_shell, the entire command is a single arg to
        /bin/sh -c, so this is the binding constraint.
        """
        config = self._make(["prompt"])._resolve_large_arg_config()
        self.assertIsNotNone(config)
        max_single = get_max_single_arg_size()
        # The per-arg threshold must respect the per-argument limit
        self.assertLessEqual(config["prompt"], max_single)

    def test_dict_mode_returns_as_is(self) -> None:
        val = {"prompt": 100, "context": 200}
        config = self._make(val)._resolve_large_arg_config()
        self.assertEqual(config, val)

    def test_invalid_type_returns_none_with_warning(self) -> None:
        # pyre-ignore[6]: intentionally testing invalid type
        inferencer = self._make(3.14)
        config = inferencer._resolve_large_arg_config()
        self.assertIsNone(config)


# ===================================================================
# Tests for _maybe_offload_large_args_to_file
# ===================================================================


class MaybeOffloadLargeArgsTest(unittest.TestCase):
    """Tests for _maybe_offload_large_args_to_file."""

    def _make(
        self,
        val: Optional[Union[bool, int, List[str], Dict[str, int]]] = None,
        temp_dir: Optional[str] = None,
    ) -> _StubInferencer:
        return _StubInferencer(
            use_file_for_large_arg_exceeding_size=val,
            large_arg_temp_dir=temp_dir,
        )

    def test_disabled_returns_input_unchanged(self) -> None:
        inf = self._make(None)
        result, files = inf._maybe_offload_large_args_to_file({"prompt": "hello"})
        self.assertEqual(result, {"prompt": "hello"})
        self.assertEqual(files, [])

    def test_small_arg_not_offloaded(self) -> None:
        inf = self._make(True)  # 64KB threshold
        result, files = inf._maybe_offload_large_args_to_file({"prompt": "small"})
        self.assertEqual(result, {"prompt": "small"})
        self.assertEqual(files, [])

    def test_large_arg_offloaded_to_file(self) -> None:
        inf = self._make(10)  # 10-byte threshold
        large_text = "A" * 100
        result, files = inf._maybe_offload_large_args_to_file({"prompt": large_text})
        try:
            self.assertEqual(len(files), 1)
            self.assertTrue(os.path.exists(files[0]))
            # Verify file contents
            with open(files[0], "r") as f:
                self.assertEqual(f.read(), large_text)
            # Verify replacement text contains the template's marker
            self.assertIn("saved in file at path:", str(result["prompt"]))
            self.assertNotEqual(result["prompt"], large_text)
        finally:
            _StubInferencer._cleanup_temp_files(files)

    def test_threshold_boundary_at_exactly_threshold(self) -> None:
        """Arg at exactly threshold should NOT be offloaded (uses <=)."""
        inf = self._make(10)
        text = "A" * 10  # exactly 10 bytes
        result, files = inf._maybe_offload_large_args_to_file({"prompt": text})
        self.assertEqual(result, {"prompt": text})
        self.assertEqual(files, [])

    def test_threshold_boundary_one_byte_over(self) -> None:
        """Arg one byte over threshold SHOULD be offloaded."""
        inf = self._make(10)
        text = "A" * 11  # 11 bytes > 10
        result, files = inf._maybe_offload_large_args_to_file({"prompt": text})
        try:
            self.assertEqual(len(files), 1)
            self.assertNotEqual(result["prompt"], text)
        finally:
            _StubInferencer._cleanup_temp_files(files)

    def test_string_input_round_trip(self) -> None:
        """String input should be offloaded and returned as string."""
        inf = self._make(5)
        result, files = inf._maybe_offload_large_args_to_file("A" * 100)
        try:
            self.assertIsInstance(result, str)
            self.assertIn("saved in file at path:", result)
        finally:
            _StubInferencer._cleanup_temp_files(files)

    def test_dict_with_multiple_keys_only_configured_offloaded(self) -> None:
        """Only keys matching the config should be offloaded."""
        inf = self._make({"prompt": 5})  # only prompt, not context
        result, files = inf._maybe_offload_large_args_to_file(
            {"prompt": "A" * 100, "context": "B" * 100}
        )
        try:
            self.assertIn("saved in file at path:", str(result["prompt"]))
            self.assertEqual(result["context"], "B" * 100)  # unchanged
        finally:
            _StubInferencer._cleanup_temp_files(files)

    def test_wildcard_offloads_all_keys(self) -> None:
        """Wildcard '*' config offloads all string args exceeding threshold."""
        inf = self._make(5)  # True → wildcard at 64KB, but int 5 → wildcard at 5
        result, files = inf._maybe_offload_large_args_to_file(
            {"prompt": "A" * 100, "context": "B" * 100}
        )
        try:
            self.assertEqual(len(files), 2)
            self.assertIn("saved in file at path:", str(result["prompt"]))
            self.assertIn("saved in file at path:", str(result["context"]))
        finally:
            _StubInferencer._cleanup_temp_files(files)

    def test_falsy_threshold_zero_offloads_non_empty(self) -> None:
        """Threshold=0 should offload any non-empty string."""
        inf = self._make(0)
        result, files = inf._maybe_offload_large_args_to_file({"prompt": "x"})
        try:
            self.assertEqual(len(files), 1)
        finally:
            _StubInferencer._cleanup_temp_files(files)

    def test_falsy_threshold_zero_skips_empty(self) -> None:
        """Threshold=0 should NOT offload empty strings (0 <= 0)."""
        inf = self._make(0)
        result, files = inf._maybe_offload_large_args_to_file({"prompt": ""})
        self.assertEqual(files, [])

    def test_explicit_key_threshold_overrides_wildcard(self) -> None:
        """Explicit key in config should override wildcard '*'."""
        inf = self._make({"*": 5, "prompt": 1000})
        result, files = inf._maybe_offload_large_args_to_file(
            {"prompt": "A" * 100, "context": "B" * 100}
        )
        try:
            # prompt: 100 bytes <= 1000 threshold → NOT offloaded
            self.assertEqual(result["prompt"], "A" * 100)
            # context: 100 bytes > 5 threshold → offloaded
            self.assertIn("saved in file at path:", str(result["context"]))
        finally:
            _StubInferencer._cleanup_temp_files(files)


# ===================================================================
# Tests for temp file directory and cleanup
# ===================================================================


class TempFileDirAndCleanupTest(unittest.TestCase):
    """Tests for large_arg_temp_dir and _cleanup_temp_files."""

    def test_temp_files_go_to_configured_dir(self) -> None:
        """When large_arg_temp_dir is set, files should be created there."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            inf = _StubInferencer(
                use_file_for_large_arg_exceeding_size=5,
                large_arg_temp_dir=tmp_dir,
            )
            _, files = inf._maybe_offload_large_args_to_file({"prompt": "A" * 100})
            try:
                self.assertEqual(len(files), 1)
                self.assertTrue(files[0].startswith(tmp_dir))
            finally:
                _StubInferencer._cleanup_temp_files(files)

    def test_temp_dir_auto_created(self) -> None:
        """large_arg_temp_dir should be created automatically if it doesn't exist."""
        with tempfile.TemporaryDirectory() as base_dir:
            nested = os.path.join(base_dir, "sub", "nested")
            self.assertFalse(os.path.exists(nested))

            inf = _StubInferencer(
                use_file_for_large_arg_exceeding_size=5,
                large_arg_temp_dir=nested,
            )
            _, files = inf._maybe_offload_large_args_to_file({"prompt": "A" * 100})
            try:
                self.assertTrue(os.path.exists(nested))
                self.assertEqual(len(files), 1)
            finally:
                _StubInferencer._cleanup_temp_files(files)

    def test_cleanup_removes_files(self) -> None:
        inf = _StubInferencer(use_file_for_large_arg_exceeding_size=5)
        _, files = inf._maybe_offload_large_args_to_file({"prompt": "A" * 100})
        self.assertTrue(all(os.path.exists(f) for f in files))
        _StubInferencer._cleanup_temp_files(files)
        self.assertTrue(all(not os.path.exists(f) for f in files))

    def test_cleanup_handles_already_deleted_files(self) -> None:
        """_cleanup_temp_files should not raise if files are already gone."""
        _StubInferencer._cleanup_temp_files(["/tmp/nonexistent_file_xyz.txt"])

    def test_mkstemp_failure_graceful_fallback(self) -> None:
        """When mkstemp fails, the arg should remain inline (no crash)."""
        inf = _StubInferencer(use_file_for_large_arg_exceeding_size=5)
        with patch("tempfile.mkstemp", side_effect=OSError("disk full")):
            result, files = inf._maybe_offload_large_args_to_file({"prompt": "A" * 100})
        self.assertEqual(files, [])
        self.assertEqual(result, {"prompt": "A" * 100})  # unchanged


# ===================================================================
# Tests for concurrency safety (per-call temp files)
# ===================================================================


class ConcurrencySafetyTest(unittest.TestCase):
    """Verify that two calls on the same instance get independent temp file lists."""

    def test_concurrent_calls_independent_temp_files(self) -> None:
        inf = _StubInferencer(use_file_for_large_arg_exceeding_size=5)

        _, files1 = inf._maybe_offload_large_args_to_file({"prompt": "A" * 100})
        _, files2 = inf._maybe_offload_large_args_to_file({"prompt": "B" * 100})

        try:
            # Each call should have its own files
            self.assertEqual(len(files1), 1)
            self.assertEqual(len(files2), 1)
            self.assertNotEqual(files1[0], files2[0])

            # Cleaning up files1 should NOT affect files2
            _StubInferencer._cleanup_temp_files(files1)
            self.assertFalse(os.path.exists(files1[0]))
            self.assertTrue(os.path.exists(files2[0]))
        finally:
            _StubInferencer._cleanup_temp_files(files1)
            _StubInferencer._cleanup_temp_files(files2)


if __name__ == "__main__":
    unittest.main()
