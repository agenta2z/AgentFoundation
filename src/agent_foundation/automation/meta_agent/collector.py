"""
Trace Collector for the Meta Agent Workflow pipeline.

Runs an agent on a task N times and collects structured execution traces.
Each agent run produces a session directory with JSONL logs and ``.parts/``
artifact files (via SessionLogger + JsonLogger). The collector reads
session log entries via :class:`SessionLogReader` with ``resolve_parts=True``,
which automatically inlines ``__parts_file__`` reference markers,
and converts them into :class:`ExecutionTrace` objects.

Trace file structure per session::

    session_dir/
      manifest.json                    ← Session manifest (read by SessionLogReader)
      turn_001/
        session.jsonl                  ← JSONL with AgentResponse, AgentState entries
        session.jsonl.parts/           ← Extracted artifacts (auto-resolved)
      turn_002/
        session.jsonl
        session.jsonl.parts/
        ...

The collector reads AgentResponse entries to extract ``next_actions``
(Iterable[Iterable[AgentAction]]). HTML snapshots are automatically
inlined from the ``.parts/`` directories by SessionLogReader.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from rich_python_utils.service_utils.session_management.session_logger import (
    SessionLogReader,
)

from science_modeling_tools.automation.meta_agent.errors import TraceCollectionError
from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceActionResult,
    TraceStep,
)

logger = logging.getLogger(__name__)


class TraceCollector:
    """
    Runs an agent on a task N times and collects execution traces.

    Each agent run produces a session directory with JSONL logs and
    ``.parts/`` artifact files. The collector reads these via
    :class:`SessionLogReader` (with ``resolve_parts=True``) and
    converts them into :class:`ExecutionTrace` objects.

    The agent interface is generic — the collector calls
    ``agent.run(task_description, data)`` and captures the result.
    The agent is expected to return an object with a ``session_dir``
    attribute (or the collector falls back to an empty trace).

    Parameters
    ----------
    agent:
        Agent instance with a ``run(task_description, data)`` method.
    synthetic_data_provider:
        Optional provider that generates distinct input data per run.
        Must have a ``generate(count) -> List[Dict[str, Any]]`` method.
    """

    def __init__(
        self,
        agent: Any,
        synthetic_data_provider: Any = None,
    ):
        self._agent = agent
        self._synthetic_data_provider = synthetic_data_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        task_description: str,
        run_count: int,
        input_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ExecutionTrace]:
        """
        Run the agent *run_count* times and return execution traces.

        Parameters
        ----------
        task_description:
            The task for the agent to perform.
        run_count:
            Number of times to run the agent (must be >= 1).
        input_data:
            Optional explicit input data per run. When provided, its
            length must match *run_count*. Overrides the synthetic
            data provider.

        Returns
        -------
        List of :class:`ExecutionTrace`, one per run (including failed
        runs with ``success=False``).

        Raises
        ------
        ValueError
            If *run_count* < 1.
        """
        if run_count < 1:
            raise ValueError(
                f"run_count must be >= 1, got {run_count}"
            )

        # Resolve per-run data
        data_per_run = self._resolve_input_data(run_count, input_data)

        traces: List[ExecutionTrace] = []
        for idx in range(run_count):
            data = data_per_run[idx] if data_per_run else None
            try:
                trace = self._run_single(task_description, data, idx)
            except Exception as exc:
                logger.warning(
                    "Agent run %d failed: %s", idx, exc, exc_info=True
                )
                # Capture partial trace with error status
                trace = ExecutionTrace(
                    trace_id=str(uuid4()),
                    task_description=task_description,
                    steps=[],
                    input_data=data,
                    success=False,
                    error=str(exc),
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )
            traces.append(trace)

        return traces

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def _run_single(
        self,
        task_description: str,
        data: Optional[Dict[str, Any]],
        run_index: int,
    ) -> ExecutionTrace:
        """Execute a single agent run and capture the trace."""
        trace_id = str(uuid4())
        start_time = datetime.now()

        # Call the agent
        result = self._agent.run(task_description, data)

        end_time = datetime.now()

        # Extract session directory from the result
        session_dir = self._extract_session_dir(result)

        # Parse logs into steps (SessionLogReader handles missing dirs gracefully)
        steps: List[TraceStep] = []
        turn_count = 0
        if session_dir:
            steps = self._convert_logs_to_steps(session_dir)
            turn_count = self._count_turns(session_dir)

        return ExecutionTrace(
            trace_id=trace_id,
            task_description=task_description,
            steps=steps,
            input_data=data,
            success=True,
            start_time=start_time,
            end_time=end_time,
            session_dir=session_dir,
            turn_count=turn_count,
        )

    # ------------------------------------------------------------------
    # Log parsing
    # ------------------------------------------------------------------

    def _convert_logs_to_steps(
        self,
        session_dir: str,
    ) -> List[TraceStep]:
        """
        Read session log entries via :class:`SessionLogReader` and convert
        them into :class:`TraceStep` objects.

        ``SessionLogReader`` with ``resolve_parts=True`` automatically
        inlines ``__parts_file__`` reference markers, so HTML snapshots
        from ``.parts/`` directories appear as string values in the
        entry dicts.

        html_before chaining
        --------------------
        Because WebDriver only captures ``body_html_before_last_action``
        when ``incremental_change_mode`` is enabled (not the default),
        ``html_before`` is often ``None`` in the raw artifacts. This
        method chains ``html_after[i-1]`` → ``html_before[i]`` when
        ``html_before[i]`` is missing, ensuring the
        TargetStrategyConverter always has an HTML snapshot to work with.
        """
        try:
            reader = SessionLogReader(session_dir, resolve_parts=True)
        except (OSError, KeyError, ValueError) as exc:
            logger.warning(
                "Could not read session logs from %s: %s", session_dir, exc
            )
            return []

        steps: List[TraceStep] = []
        for entry in reader:
            step = self._entry_to_step(entry)
            if step is not None:
                steps.append(step)

        # Apply html_before chaining
        self._chain_html_before(steps)

        return steps

    def _entry_to_step(self, entry: Dict[str, Any]) -> Optional[TraceStep]:
        """Convert a single JSONL entry to a TraceStep, or None if not actionable."""
        # Look for entries that represent agent actions
        entry_type = entry.get("type", "")

        # AgentResponse entries contain next_actions
        if entry_type == "AgentResponse":
            return self._parse_agent_response_entry(entry)

        # AgentActionResults entries contain action results
        if entry_type == "AgentActionResults":
            return self._parse_action_result_entry(entry)

        # Generic action entries
        action_type = entry.get("action_type")
        if action_type:
            return TraceStep(
                action_type=action_type,
                target=entry.get("target"),
                args=entry.get("args"),
                result=self._parse_result(entry.get("result")),
                timestamp=self._parse_timestamp(entry.get("timestamp")),
                source_url=entry.get("source_url"),
                action_group_index=entry.get("action_group_index", 0),
                parallel_index=entry.get("parallel_index", 0),
                reasoning=entry.get("reasoning"),
                metadata=entry.get("metadata", {}),
                html_before=entry.get("body_html_before_last_action"),
                html_after=entry.get("body_html_after_last_action"),
            )

        return None

    def _parse_agent_response_entry(self, entry: Dict[str, Any]) -> Optional[TraceStep]:
        """Parse an AgentResponse JSONL entry into a TraceStep."""
        data = entry.get("data", entry)
        next_actions = data.get("next_actions")

        if not next_actions:
            return None

        # next_actions is Iterable[Iterable[AgentAction]]
        # Flatten to get the first action for this step
        for group_idx, action_group in enumerate(next_actions):
            if isinstance(action_group, dict):
                # Single action (not nested)
                return self._action_dict_to_step(action_group, group_idx, 0)
            if isinstance(action_group, (list, tuple)):
                for par_idx, action in enumerate(action_group):
                    if isinstance(action, dict):
                        return self._action_dict_to_step(action, group_idx, par_idx)

        return None

    def _action_dict_to_step(
        self,
        action: Dict[str, Any],
        group_index: int,
        parallel_index: int,
    ) -> TraceStep:
        """Convert an action dictionary to a TraceStep."""
        return TraceStep(
            action_type=action.get("action_type", action.get("type", "unknown")),
            target=action.get("target"),
            args=action.get("args", action.get("arguments")),
            timestamp=self._parse_timestamp(action.get("timestamp")),
            source_url=action.get("source_url"),
            action_group_index=group_index,
            parallel_index=parallel_index,
            reasoning=action.get("reasoning"),
            metadata=action.get("metadata", {}),
            html_before=action.get("body_html_before_last_action"),
            html_after=action.get("body_html_after_last_action"),
        )

    def _parse_action_result_entry(self, entry: Dict[str, Any]) -> Optional[TraceStep]:
        """Parse an AgentActionResults JSONL entry into a TraceStep."""
        data = entry.get("data", entry)

        action_type = data.get("action_type", "unknown")
        result_data = data.get("result", {})

        return TraceStep(
            action_type=action_type,
            target=data.get("target"),
            args=data.get("args"),
            result=self._parse_result(result_data),
            timestamp=self._parse_timestamp(data.get("timestamp")),
            source_url=data.get("source_url"),
            action_group_index=data.get("action_group_index", 0),
            parallel_index=data.get("parallel_index", 0),
            reasoning=data.get("reasoning"),
            metadata=data.get("metadata", {}),
            html_before=data.get("body_html_before_last_action"),
            html_after=data.get("body_html_after_last_action"),
        )

    # ------------------------------------------------------------------
    # html_before chaining
    # ------------------------------------------------------------------

    @staticmethod
    def _chain_html_before(steps: List[TraceStep]) -> None:
        """
        Chain ``html_after[i-1]`` → ``html_before[i]`` when
        ``html_before[i]`` is ``None``.

        WebDriver only captures ``body_html_before_last_action`` when
        ``incremental_change_mode`` is enabled. For most action types
        ``html_before`` will be ``None``. This chaining ensures the
        TargetStrategyConverter always has an HTML snapshot.
        """
        for i in range(1, len(steps)):
            if steps[i].html_before is None and steps[i - 1].html_after is not None:
                steps[i].html_before = steps[i - 1].html_after

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_input_data(
        self,
        run_count: int,
        input_data: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Resolve per-run input data from explicit data or synthetic provider."""
        if input_data is not None:
            return input_data

        if self._synthetic_data_provider is not None:
            return self._synthetic_data_provider.generate(run_count)

        return None

    @staticmethod
    def _extract_session_dir(result: Any) -> Optional[str]:
        """Extract session directory path from an agent run result."""
        if result is None:
            return None

        # Try common attribute names
        for attr in ("session_dir", "session_directory", "log_dir"):
            val = getattr(result, attr, None)
            if val is not None:
                return str(val)

        # Try dict access
        if isinstance(result, dict):
            for key in ("session_dir", "session_directory", "log_dir"):
                if key in result:
                    return str(result[key])

        return None

    @staticmethod
    def _count_turns(session_dir: Optional[str]) -> int:
        """Count turns from the session manifest."""
        if not session_dir:
            return 0
        try:
            reader = SessionLogReader(session_dir, resolve_parts=False)
            return len(reader.turns)
        except (OSError, KeyError, ValueError):
            return 0

    @staticmethod
    def _parse_result(result_data: Any) -> Optional[TraceActionResult]:
        """Parse a result dict into a TraceActionResult."""
        if result_data is None:
            return None
        if isinstance(result_data, dict):
            return TraceActionResult(
                success=result_data.get("success", True),
                action_skipped=result_data.get("action_skipped", False),
                skip_reason=result_data.get("skip_reason"),
                value=result_data.get("value"),
                error=result_data.get("error"),
            )
        return None

    @staticmethod
    def _parse_timestamp(ts: Any) -> Optional[datetime]:
        """Parse a timestamp value into a datetime."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            # Try ISO format
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        return None

