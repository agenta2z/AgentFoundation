"""Unit tests for ``StreamingInferencerBase._effective_cache_folder``.

v5 Fix #3 — the modern M7 ctx-dispatch path delivers the active workspace
via ``ctx.handles["workspace_override"]`` (set by ``_publish_workspace_to_ctx``),
which deliberately bypasses the ``_workspace`` setter (so a SAME shared
streaming-leaf instance can serve concurrent gathered branches with distinct
workspaces). Without a per-call resolver, ``self.cache_folder`` stays ``None``
for any leaf reached via ctx-dispatch — observed empirically as zero
``stream_*.txt`` files under any flow leaf in the failing run.

These tests pin the resolver's contract:
  * Explicit ``cache_folder`` always wins.
  * No cache, no workspace, no ctx → ``None`` (today's behavior).
  * Instance ``_workspace`` set → derive ``<ws.root>/_runtime/inferencer_cache``.
  * ctx ``workspace_override`` published → derive from THAT (per-branch).
  * Concurrency safety: two ctxs with different overrides on the SAME
    instance → each branch sees its own path. A mutating set-once design
    (``if not self.cache_folder: self.cache_folder = ...``) would FAIL
    this test because branch B would route into branch A's path.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from typing import Any, AsyncIterator

from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrs


@attrs
class _StreamingStub(StreamingInferencerBase):
    """Minimal concrete StreamingInferencerBase — implements just the abstracts."""

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        # Never called by these tests — they only exercise the resolver.
        yield ""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return ""


def _expected(root: str) -> str:
    return os.path.join(str(root), "_runtime", "inferencer_cache")


class TestEffectiveCacheFolderExplicit(unittest.TestCase):
    def test_explicit_cache_folder_returned(self):
        """Explicit ``cache_folder`` takes precedence (the resolver's first branch)."""
        with tempfile.TemporaryDirectory() as explicit:
            inf = _StreamingStub(cache_folder=explicit)
            # Without setting a workspace (which would overwrite cache_folder
            # via the legacy setter), the resolver returns the explicit value.
            self.assertEqual(inf._effective_cache_folder(), explicit)


class TestEffectiveCacheFolderNoSources(unittest.TestCase):
    def test_none_when_nothing_available(self):
        """No explicit cache, no instance workspace, no ctx → None (legacy)."""
        inf = _StreamingStub()
        self.assertIsNone(inf._effective_cache_folder())


class TestEffectiveCacheFolderInstanceWorkspace(unittest.TestCase):
    def test_derives_from_instance_workspace(self):
        with tempfile.TemporaryDirectory() as ws_root:
            inf = _StreamingStub()
            # Bypass the setter side-effects by going through object.__setattr__
            # to mirror what ctx-dispatch would do (no _configure_for_workspace).
            # Actually use the setter here — represents the LEGACY path.
            inf._workspace = InferencerWorkspace(root=ws_root)
            # Setter would have populated cache_folder for the legacy path —
            # the resolver still returns that (explicit > derived).
            self.assertEqual(
                inf._effective_cache_folder(),
                _expected(ws_root),
            )


class TestEffectiveCacheFolderCtxDispatch(unittest.TestCase):
    def test_derives_from_ctx_workspace_override(self):
        """Modern M7 ctx-dispatch path: no instance workspace, override in ctx."""
        with tempfile.TemporaryDirectory() as ws_root:
            inf = _StreamingStub()
            # Confirm legacy path returns None (no instance workspace set,
            # no explicit cache_folder).
            self.assertIsNone(inf._effective_cache_folder())

            ws = InferencerWorkspace(root=ws_root)
            ctx = RunContext.root(workspace=None)
            ctx.handles.set("workspace_override", ws)
            token = enter_run(ctx)
            try:
                self.assertEqual(
                    inf._effective_cache_folder(),
                    _expected(ws_root),
                )
            finally:
                exit_run(token)

            # After exit, the override is gone — resolver back to None.
            self.assertIsNone(inf._effective_cache_folder())

    def test_ctx_workspace_field_used_when_no_override(self):
        """When the ctx itself carries a workspace (root-context case)
        and the instance has no backing, the resolver picks that up too
        — matches the ``_workspace`` getter's secondary fallback."""
        with tempfile.TemporaryDirectory() as ws_root:
            inf = _StreamingStub()
            ws = InferencerWorkspace(root=ws_root)
            ctx = RunContext.root(workspace=ws)
            token = enter_run(ctx)
            try:
                self.assertEqual(
                    inf._effective_cache_folder(),
                    _expected(ws_root),
                )
            finally:
                exit_run(token)


class TestEffectiveCacheFolderConcurrency(unittest.TestCase):
    """The decisive regression test for the WRITE-PURE design: a single
    shared streaming-leaf instance serving two concurrent gathered branches
    with distinct workspaces must yield two distinct cache folders. A
    set-once mutation (``if not self.cache_folder: self.cache_folder = …``)
    would FAIL this test because branch B's resolver would return branch
    A's already-mutated value.
    """

    def test_two_concurrent_ctxs_distinct_workspaces(self):
        with (
            tempfile.TemporaryDirectory() as ws_a,
            tempfile.TemporaryDirectory() as ws_b,
        ):
            shared_inf = _StreamingStub()

            results = {}

            async def _branch(name, ws_root):
                ws = InferencerWorkspace(root=ws_root)
                ctx = RunContext.root(workspace=None)
                ctx.handles.set("workspace_override", ws)
                token = enter_run(ctx)
                try:
                    # Yield once to interleave with the other branch.
                    await asyncio.sleep(0)
                    results[name] = shared_inf._effective_cache_folder()
                    # Yield once more to ensure the other branch ran before
                    # we exit our ctx.
                    await asyncio.sleep(0)
                finally:
                    exit_run(token)

            async def _gather():
                await asyncio.gather(_branch("A", ws_a), _branch("B", ws_b))

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_gather())
            finally:
                loop.close()

            self.assertEqual(results["A"], _expected(ws_a))
            self.assertEqual(results["B"], _expected(ws_b))
            self.assertNotEqual(results["A"], results["B"])


if __name__ == "__main__":
    unittest.main()
