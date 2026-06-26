"""Tier-3: connection-scoped, path-keyed live handles.

Per the plan (§2.0 Note B / §2.1 / V7 / T-MAJOR2):

* Live handles (SDK client + ``_disconnect_fn``/``_connected_loop``/``_connect_lock``,
  subprocess, httpx session + ``_base_url``, live ``session_id``) are **reused
  across many calls** in a branch and carry continuity — they are
  **connection-scoped, NOT per-``ainfer``-call and NOT per-turn**.
* They therefore live in a :class:`LiveHandleStore` whose lifetime is the
  ``aconnect``/``adisconnect`` connection — **distinct from the per-turn Tier-1
  store** (which IS discarded each turn).  Discarding the per-turn state store
  must NOT relaunch a leaf's subprocess or drop its ``session_id``.
* ``LiveHandleStore.get_or_create(path)`` is **fresh on first derivation, reused
  on re-derivation** — resolving "fresh per branch" vs "reused across calls".

These are **never persisted** (a live connection is not a concurrency-safe sink
and cannot be serialized).
"""

from __future__ import annotations

from typing import Any


class LiveHandles:
    """A per-branch bag of live connection handles (NOT serialized).

    Deliberately flexible: leaf inferencers stash backend-specific handles
    (``sdk_client``, ``subprocess``, ``http_client``, ``base_url``,
    ``live_session_id``, ``connect_lock``, ...).  Access via attribute or
    ``get``/``set``; missing attributes read as ``None`` rather than raising, so
    a leaf can probe ``handles.sdk_client`` before connecting.
    """

    __never_persist__ = True

    def __init__(self, path: str = "/", **initial: Any) -> None:
        # Use object.__setattr__ to avoid recursion through our __setattr__.
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "_data", dict(initial))

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails; serve from _data (default None).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("path", "_data"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def set(self, name: str, value: Any) -> None:
        self._data[name] = value

    def clear(self) -> None:
        self._data.clear()

    def __repr__(self) -> str:
        return f"LiveHandles(path={self.path!r}, keys={sorted(self._data)})"


class LiveHandleStore:
    """Connection-scoped, path-keyed store of :class:`LiveHandles`.

    Lifetime = the connection (``aconnect``..``adisconnect``), **not** the per-turn
    run.  ``get_or_create`` is idempotent per path.
    """

    __never_persist__ = True

    def __init__(self) -> None:
        self._by_path: dict[str, LiveHandles] = {}

    def get_or_create(self, path: str) -> LiveHandles:
        handles = self._by_path.get(path)
        if handles is None:
            handles = LiveHandles(path=path)
            self._by_path[path] = handles
        return handles

    def peek(self, path: str) -> LiveHandles | None:
        return self._by_path.get(path)

    def teardown(self, path: str | None = None) -> None:
        """Tear down one branch's handles (``adisconnect``), or all of them."""
        if path is None:
            for handles in self._by_path.values():
                handles.clear()
            self._by_path.clear()
        else:
            handles = self._by_path.pop(path, None)
            if handles is not None:
                handles.clear()

    def __len__(self) -> int:
        return len(self._by_path)
