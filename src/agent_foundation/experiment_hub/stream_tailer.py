# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""WorkspaceStreamTailer — tail inferencer cache files for live streaming.

Reimplemented locally for the Experiment Hub (no ``rankevolve`` import). Pure
``os`` / ``asyncio`` / ``pathlib`` async file-tailer + callback: watches a
workspace's inferencer cache directory for ``stream_*.txt`` files and forwards
new bytes to an async callback, filtering the stream-completion markers. The
submission runner writes the stream file; OpenTeam instantiates the tailer in
the process that owns the WebSocket handle.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_foundation.experiment_hub.hub_markers import (
    STREAM_DONE_MARKER,
    STREAM_FAIL_MARKER,
)


class WorkspaceStreamTailer:
    """Watch an inferencer cache directory for ``stream_*.txt`` files and tail them.

    Args:
        cache_dir: The inferencer cache directory to watch.
        default_phase: Fallback phase label when directory name has no phase marker.
        poll_interval: Seconds between polling iterations.
        replay_existing: If True, do NOT skip pre-existing files (replay all).
        pre_existing_cutoff: Unix timestamp. Files with mtime strictly before
            this cutoff are treated as pre-existing (skipped) unless
            ``replay_existing`` is True. ``0.0`` disables the cutoff.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        default_phase: str = "unknown",
        poll_interval: float = 0.3,
        replay_existing: bool = False,
        pre_existing_cutoff: float = 0.0,
    ) -> None:
        self._cache_dir: str = str(cache_dir)
        self._default_phase: str = default_phase
        self._poll_interval: float = poll_interval
        self._replay_existing: bool = replay_existing
        self._pre_existing_cutoff: float = pre_existing_cutoff

        self._running: bool = True
        self._tailed_bytes: int = 0
        self._accumulated_parts: list[str] = []

        # Internal tracking state
        self._seen_files: set[str] = set()
        self._file_positions: dict[str, int] = {}
        self._file_agents: dict[str, str] = {}
        self._file_phases: dict[str, str] = {}
        self._seen_role_dirs: dict[str, list[str]] = {}
        self._dir_to_round: dict[str, int] = {}

    @property
    def tailed_bytes(self) -> int:
        """Total bytes forwarded to callback."""
        return self._tailed_bytes

    @property
    def accumulated_content(self) -> str:
        """Full concatenated text of all tailed content."""
        return "".join(self._accumulated_parts)

    def stop(self) -> None:
        """Signal the tailer to stop after completing a final drain pass."""
        self._running = False

    async def tail(
        self,
        callback: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Run the tailing loop until ``stop()`` is called.

        Args:
            callback: Async function called with (content, metadata) for each chunk.
        """
        self._pre_populate_seen_files()

        while self._running:
            self._scan_new_files()
            await self._read_new_content(callback)
            await asyncio.sleep(self._poll_interval)

        # Final drain pass: read any remaining bytes after stop signal
        self._scan_new_files()
        await self._read_new_content(callback)

    def _pre_populate_seen_files(self) -> None:
        """Mark pre-existing stream files as seen to avoid replaying old content."""
        if self._replay_existing:
            return
        try:
            for root, _dirs, files in os.walk(self._cache_dir):
                for fname in files:
                    if fname.startswith("stream_") and fname.endswith(".txt"):
                        full_path = os.path.join(root, fname)
                        try:
                            if (
                                self._pre_existing_cutoff > 0
                                and os.path.getmtime(full_path)
                                >= self._pre_existing_cutoff
                            ):
                                continue  # created during this session
                        except OSError:
                            pass
                        self._seen_files.add(full_path)
        except OSError:
            pass

    def _scan_new_files(self) -> None:
        """Discover new ``stream_*.txt`` files and assign agent/phase metadata."""
        try:
            for root, _dirs, files in os.walk(self._cache_dir):
                for fname in files:
                    if not fname.startswith("stream_") or not fname.endswith(".txt"):
                        continue
                    full_path = os.path.join(root, fname)
                    if full_path in self._seen_files:
                        continue
                    self._seen_files.add(full_path)
                    self._file_positions[full_path] = 0

                    parent_name = os.path.basename(root)
                    parent_path = root

                    # Derive agent role
                    if "_base_" in parent_name:
                        role = "base"
                    elif "_review_" in parent_name:
                        role = "review"
                    else:
                        role = "agent"

                    # Assign round number
                    if parent_path not in self._dir_to_round:
                        role_count = len(self._seen_role_dirs.get(role, [])) + 1
                        if role not in self._seen_role_dirs:
                            self._seen_role_dirs[role] = []
                        self._seen_role_dirs[role].append(parent_path)
                        self._dir_to_round[parent_path] = role_count

                    round_num = self._dir_to_round[parent_path]
                    agent_id = f"{role}_round{round_num}" if round_num > 1 else role
                    self._file_agents[full_path] = agent_id

                    # Derive phase from directory name
                    if "_plan_" in parent_name:
                        detected_phase = "plan"
                    elif "_implementation_" in parent_name or "_impl_" in parent_name:
                        detected_phase = "implementation"
                    elif "_analysis_" in parent_name:
                        detected_phase = "analysis"
                    else:
                        detected_phase = self._default_phase
                    self._file_phases[full_path] = detected_phase
        except OSError:
            pass

    async def _read_new_content(
        self,
        callback: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Read new bytes from all tracked files, filter markers, call callback."""
        for fpath in list(self._file_positions.keys()):
            try:
                size = os.path.getsize(fpath)
                pos = self._file_positions[fpath]
                if size <= pos:
                    continue
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    new_content = fh.read()
                    # Use actual file position after read to avoid race
                    # condition where writer appends between getsize() and
                    # read(), which would cause duplicate fragments.
                    actual_pos = fh.tell()
                if not new_content:
                    continue
                self._file_positions[fpath] = actual_pos

                # Filter out stream completion markers
                marker_found = False
                for marker in (STREAM_DONE_MARKER, STREAM_FAIL_MARKER):
                    idx = new_content.find(marker)
                    if idx != -1:
                        new_content = new_content[:idx]
                        marker_found = True
                # Only strip the artifact \n from the leading newline in
                # "\n--- STREAM COMPLETED...". Stripping on every chunk would
                # destroy inter-line newlines that markdown tables need.
                if marker_found:
                    new_content = new_content.rstrip("\n")
                if not new_content:
                    continue

                agent_id = self._file_agents.get(fpath, "agent")
                detected_phase = self._file_phases.get(fpath, self._default_phase)
                self._tailed_bytes += len(new_content)
                self._accumulated_parts.append(new_content)
                await callback(
                    new_content,
                    {"phase": detected_phase, "agent_id": agent_id},
                )
            except OSError:
                pass
