"""Mock BTA components for graph visualization testing.

Plug into a real BreakdownThenAggregateInferencer via YAML profiles so
the full graph event pipeline (topology, status, streaming) is exercised
without LLM calls. Used by the ``/mock_task`` developer tool.

Usage::

    import agent_foundation.common.configs.registered_targets  # register aliases
    from rich_python_utils.config_utils import load_config, instantiate
    cfg = load_config("profiles/default.yaml")
    bta = instantiate(cfg)
    result = await bta.ainfer("__mock_input__")
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

_logger = logging.getLogger(__name__)


class MockBreakdownInferencer:
    """Returns a pre-canned JSON subtask list with optional streaming delay.

    Satisfies BTA's ``breakdown_inferencer`` contract: has ``ainfer()``
    and optional ``stream_observer`` attribute.
    """

    def __init__(
        self,
        *,
        subtasks: list[dict] | None = None,
        delay_s: float = 0.5,
        stream_chunks: list[str] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.subtasks = subtasks or [
            {"description": "Research existing implementations", "args": {}},
            {"description": "Identify capability gaps", "args": {}},
            {"description": "Define specifications", "args": {}},
        ]
        self.delay_s = delay_s
        self.stream_chunks = stream_chunks or [
            "Analyzing input document...\n",
            "Identifying required subtasks...\n",
            f"Found {len(self.subtasks)} subtasks.\n",
        ]
        self.stream_observer: Any = None

    async def ainfer(self, *args: Any, **kwargs: Any) -> str:
        per_chunk = self.delay_s / max(1, len(self.stream_chunks))
        for chunk in self.stream_chunks:
            if self.stream_observer is not None:
                await self.stream_observer(chunk)
            await asyncio.sleep(per_chunk)
        return json.dumps({"subtasks": self.subtasks})


class MockWorker:
    """Mocks a single leaf worker — sleeps, streams chunks, returns text.

    Satisfies BTA worker contract: has ``ainfer()``, ``interactive``,
    ``stream_observer``, and ``graph_reporter`` attributes.
    """

    def __init__(
        self,
        *,
        label: str = "mock worker",
        duration_s: float = 1.0,
        stream_chunks: list[str] | None = None,
        should_error: bool = False,
        output: str = "",
        sub_query: str = "",
        index: int = 0,
        **_kwargs: Any,
    ) -> None:
        self.label = label or sub_query or f"worker_{index}"
        self.duration_s = duration_s
        self.stream_chunks = stream_chunks or [
            f"## {self.label}\n\n",
            "Investigating requirements and constraints...\n\n",
            "### Key Findings\n\n",
            "- Finding 1: The system requires careful integration.\n",
            "- Finding 2: Existing patterns can be reused.\n\n",
            "### Recommendations\n\n",
            "Based on analysis, the recommended approach is to...\n",
        ]
        self.should_error = should_error
        self.output = output or f"[Mock output for {self.label}]"
        self.interactive: Any = None
        self.stream_observer: Any = None

    async def ainfer(self, *args: Any, **kwargs: Any) -> str:
        per_chunk = self.duration_s / max(1, len(self.stream_chunks) + 1)
        for chunk in self.stream_chunks:
            await asyncio.sleep(per_chunk)
            if self.stream_observer is not None:
                await self.stream_observer(chunk)
        await asyncio.sleep(per_chunk)
        if self.should_error:
            raise RuntimeError(f"[Mock error in {self.label}]")
        if self.stream_observer is not None and hasattr(self.stream_observer, 'flush'):
            await self.stream_observer.flush()
        return self.output


class MockAggregator:
    """Mocks the aggregator — receives worker outputs, emits synthesis.

    Satisfies BTA ``aggregator_inferencer`` contract.
    """

    def __init__(
        self,
        *,
        delay_s: float = 1.0,
        output_template: str = "# Aggregated Report\n\nAll subtasks completed successfully.",
        stream_chunks: list[str] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.delay_s = delay_s
        self.output_template = output_template
        self.stream_chunks = stream_chunks or [
            "Synthesizing results from all workers...\n",
            "Generating final report...\n",
        ]
        self.stream_observer: Any = None

    async def ainfer(self, *args: Any, **kwargs: Any) -> str:
        per_chunk = self.delay_s / max(1, len(self.stream_chunks) + 1)
        for chunk in self.stream_chunks:
            await asyncio.sleep(per_chunk)
            if self.stream_observer is not None:
                await self.stream_observer(chunk)
        await asyncio.sleep(per_chunk)
        if self.stream_observer is not None and hasattr(self.stream_observer, 'flush'):
            await self.stream_observer.flush()
        return self.output_template
