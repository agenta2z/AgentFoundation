"""Commit 1 (D1): BTA proposal_index extraction + truncated-response fallback.

Exercises ``_try_extract_proposal_index`` / ``_read_aggregator_output_text`` in
isolation by binding the real methods onto a lightweight stub ``self`` (the full
inferencer needs a large constructor; these methods only touch ``_workspace`` and
``aggregator_inferencer``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer as _BTA,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


def _make_fence(total_count: int = 1) -> str:
    index = {
        "version": "1",
        "total_count": total_count,
        "groups": [
            {
                "phase": 1,
                "label": "Quick Wins",
                "proposals": [
                    {"id": "P1", "rank": 1, "title": "Add caching"},
                ],
            }
        ],
    }
    return "```json proposal_index\n" + json.dumps(index) + "\n```"


class _StubAggregator:
    def __init__(self, workspace: InferencerWorkspace, output_path: str = "output.md"):
        self._workspace = workspace
        self.output_path = output_path


class _StubBTA:
    """Minimal carrier binding the two real methods under test."""

    _try_extract_proposal_index = _BTA._try_extract_proposal_index
    _read_aggregator_output_text = _BTA._read_aggregator_output_text
    # _read_aggregator_output_text now resolves the aggregator workspace via the
    # override-aware orchestrator read (M7); bind it too. With no active ctx it
    # falls back to the bare ``agg_inf._workspace`` — byte-identical to the prior read.
    _read_child_workspace = _BTA._read_child_workspace

    def __init__(self, workspace, aggregator_inferencer):
        self._workspace = workspace
        self.aggregator_inferencer = aggregator_inferencer


def _sidecar(root: str) -> Path:
    return Path(root) / "outputs" / "proposals.json"


def test_fast_path_fence_in_response(tmp_path):
    """Regression guard: fence present in the response → sidecar written."""
    ws = InferencerWorkspace(root=str(tmp_path))
    stub = _StubBTA(workspace=ws, aggregator_inferencer=None)

    stub._try_extract_proposal_index(f"Here is the plan:\n{_make_fence(2)}\nDone.")

    side = _sidecar(str(tmp_path))
    assert side.is_file()
    data = json.loads(side.read_text())
    assert data["groups"][0]["proposals"][0]["id"] == "P1"


def test_fallback_fence_in_aggregator_file(tmp_path):
    """Truncated response, but the fence lives in the aggregator output file."""
    main_ws = InferencerWorkspace(root=str(tmp_path / "main"))
    agg_ws = InferencerWorkspace(root=str(tmp_path / "agg"))

    # Write the fence to the aggregator's output file on disk.
    agg_out = Path(agg_ws.output_path("output.md"))
    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(f"# Research\n{_make_fence(3)}\n", encoding="utf-8")

    stub = _StubBTA(
        workspace=main_ws,
        aggregator_inferencer=_StubAggregator(agg_ws, output_path="output.md"),
    )

    # The response is truncated — no fence in memory.
    stub._try_extract_proposal_index("...(response truncated at token limit)...")

    side = _sidecar(str(tmp_path / "main"))
    assert side.is_file(), "fallback must recover the fence from disk"
    data = json.loads(side.read_text())
    assert data["groups"][0]["proposals"][0]["id"] == "P1"


def test_truncated_and_aggregator_file_missing(tmp_path):
    """Truncated response AND no aggregator file → clean no-op, no sidecar."""
    main_ws = InferencerWorkspace(root=str(tmp_path / "main"))
    agg_ws = InferencerWorkspace(root=str(tmp_path / "agg"))  # outputs/ never written

    stub = _StubBTA(
        workspace=main_ws,
        aggregator_inferencer=_StubAggregator(agg_ws, output_path="output.md"),
    )

    stub._try_extract_proposal_index("...(response truncated, no fence)...")

    assert not _sidecar(str(tmp_path / "main")).exists()


def test_truncated_no_aggregator_inferencer(tmp_path):
    """No aggregator at all → fallback returns None, clean no-op."""
    main_ws = InferencerWorkspace(root=str(tmp_path / "main"))
    stub = _StubBTA(workspace=main_ws, aggregator_inferencer=None)

    stub._try_extract_proposal_index("no fence here")

    assert not _sidecar(str(tmp_path / "main")).exists()


def test_read_aggregator_output_text_reads_file(tmp_path):
    agg_ws = InferencerWorkspace(root=str(tmp_path / "agg"))
    agg_out = Path(agg_ws.output_path("output.md"))
    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text("hello fence world", encoding="utf-8")

    stub = _StubBTA(
        workspace=InferencerWorkspace(root=str(tmp_path / "main")),
        aggregator_inferencer=_StubAggregator(agg_ws),
    )
    assert stub._read_aggregator_output_text() == "hello fence world"
