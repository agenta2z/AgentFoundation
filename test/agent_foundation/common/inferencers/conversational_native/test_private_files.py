"""Private files are 0600 whatever was at their path before (invariant 9):
the shared writer, the turn loop's session files, the tool bridge's result
spill and the Claude CLI backend's settings / MCP config."""

from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    private_files,
    turn_loop,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge import (
    tool_bridge,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    AFToolBridge,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.private_files import (
    write_private_file,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.turn_loop import (
    NativeTurnLoopMixin,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _stale(path: Path) -> Path:
    path.write_text("stale " * 100)
    path.chmod(0o644)
    return path


class WritePrivateFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def test_a_new_file_is_0600(self) -> None:
        path = write_private_file(self.dir / "new.txt", "body")
        self.assertEqual((path.read_text(), _mode(path)), ("body", 0o600))

    def test_a_pre_existing_0644_file_is_replaced_by_a_0600_one(self) -> None:
        path = _stale(self.dir / "old.json")
        write_private_file(str(path), "{}")
        self.assertEqual((path.read_text(), _mode(path)), ("{}", 0o600))
        self.assertEqual(os.listdir(self.dir), ["old.json"])  # no temp file left

    def test_a_symlink_at_the_path_is_replaced_not_followed(self) -> None:
        target = _stale(self.dir / "elsewhere.txt")
        link = self.dir / "link.txt"
        link.symlink_to(target)
        write_private_file(link, "secret")
        self.assertFalse(link.is_symlink())
        self.assertEqual((link.read_text(), _mode(link)), ("secret", 0o600))
        self.assertTrue(target.read_text().startswith("stale"))

    def test_a_failed_write_leaves_the_old_file_and_no_temp_file(self) -> None:
        path = _stale(self.dir / "keep.txt")
        with mock.patch.object(
            private_files.os, "replace", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                write_private_file(path, "new")
        self.assertTrue(path.read_text().startswith("stale"))
        self.assertEqual(os.listdir(self.dir), ["keep.txt"])


class TurnLoopSessionFileTest(unittest.TestCase):
    def test_session_files_are_written_by_the_shared_writer(self) -> None:
        directory = Path(tempfile.mkdtemp())
        stale = _stale(directory / "l1_0.md")
        host = SimpleNamespace(session_dir=lambda: directory)
        with mock.patch.object(
            turn_loop, "write_private_file", wraps=write_private_file
        ) as writer:
            path = NativeTurnLoopMixin._write_private(host, "l1_0.md", "L1")
        writer.assert_called_once_with(stale, "L1")
        self.assertEqual((path, path.read_text(), _mode(path)), (stale, "L1", 0o600))


class ToolResultSpillTest(unittest.TestCase):
    def test_a_spill_over_a_pre_existing_0644_file_is_0600(self) -> None:
        spill_dir = Path(tempfile.mkdtemp()) / "tool_results"
        spill_dir.mkdir()
        host = SimpleNamespace(
            native_tool_result_max_chars=200,
            vendor_owns_result_spill=lambda: False,
            spill_dir=lambda: spill_dir,
        )
        bridge = AFToolBridge(host, asyncio.Lock())
        fixed = SimpleNamespace(hex="0123456789abcdef")
        stale = _stale(spill_dir / "tool_result_0123456789ab.txt")
        with mock.patch.object(tool_bridge.uuid, "uuid4", return_value=fixed):
            result = bridge._sized("", "x" * 1000, "")
        self.assertIn(str(stale), result)
        self.assertEqual((stale.read_text(), _mode(stale)), ("x" * 1000, 0o600))
