"""Generic atomic JSON-sidecar store for dashboard state files.

Single-process safe (OpenTeam is one FastAPI process): each file is guarded by a
per-instance ``asyncio.Lock`` and writes are atomic (temp file in the same dir,
then ``os.replace``). Concrete Experiment-Hub stores (submissions / setup /
combos / implementations) extend this and supply a mutable-field allow-list.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


class JsonSidecarStore:
    """A single JSON file holding a ``dict`` envelope, with atomic writes + lock."""

    def __init__(
        self, path: str | Path, *, default: dict[str, Any] | None = None
    ) -> None:
        self._path: Path = Path(path)
        self._default: dict[str, Any] = dict(default) if default is not None else {}
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> dict[str, Any]:
        """Read current state (sync). Returns a copy of the default if missing or corrupt."""
        try:
            with open(self._path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else dict(self._default)
        except FileNotFoundError:
            return dict(self._default)
        except (json.JSONDecodeError, OSError):
            return dict(self._default)

    def _write_atomic(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    async def write(self, data: dict[str, Any]) -> None:
        async with self._lock:
            self._write_atomic(data)

    async def update(
        self,
        mutate: Callable[[dict[str, Any]], dict[str, Any] | None],
        *,
        allowed_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        """Atomic read-modify-write under the lock.

        ``mutate(state)`` may mutate in place or return a new dict. When
        ``allowed_fields`` is given, only those top-level keys may change; any
        other key is restored from the prior on-disk state (the single-writer
        mutable-field allow-list pattern ported from RankEvolve's hub writers).
        """
        async with self._lock:
            current = self.read()
            prior = dict(current)
            result = mutate(current)
            new_state = result if isinstance(result, dict) else current
            if allowed_fields is not None:
                for key in list(new_state.keys()):
                    if key not in allowed_fields and key in prior:
                        new_state[key] = prior[key]
                for key, val in prior.items():
                    if key not in allowed_fields:
                        new_state[key] = val
            self._write_atomic(new_state)
            return new_state
