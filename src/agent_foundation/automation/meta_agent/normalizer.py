"""
Trace Normalizer for the Meta Agent Workflow pipeline.

Converts raw agent trace steps into canonical format with consistent
action types, target representations, and normalized wait durations.
"""

import copy
import statistics
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set

from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)


# Known canonical action types that don't need mapping.
# These are already in the ActionGraph vocabulary and pass through as-is.
KNOWN_CANONICAL_TYPES: Set[str] = {
    "click",
    "input_text",
    "append_text",
    "scroll",
    "scroll_up_to_element",
    "input_and_submit",
    "visit_url",
    "no_op",
    "wait",
    "extract_text",
    "select_option",
}


class TraceNormalizer:
    """
    Normalizes trace steps to canonical format.

    Per-step normalization:
    - Maps action types to canonical ActionGraph types
      (e.g., "ElementInteraction.Click" → "click")
    - Normalizes target representations
    - Converts UserInputsRequired to wait(True)

    Cross-trace normalization:
    - Normalizes wait durations to median across all traces

    Note: Target consolidation across runs (merging multiple target
    observations into TargetSpecWithFallback) happens in Pattern
    Extraction, not here.
    """

    ACTION_TYPE_MAP: Dict[str, str] = {}

    def __init__(
        self,
        action_metadata: Optional[Any] = None,
        custom_type_map: Optional[Dict[str, str]] = None,
    ):
        self._action_metadata = action_metadata
        self._type_map: Dict[str, str] = {
            **self.ACTION_TYPE_MAP,
            **(custom_type_map or {}),
        }
        # Build the full set of known canonical types, including any
        # types registered in the action metadata registry.
        self._canonical_types: Set[str] = set(KNOWN_CANONICAL_TYPES)
        if self._action_metadata is not None:
            self._canonical_types.update(self._action_metadata.list_actions())
        # Also include all map *values* as canonical
        self._canonical_types.update(self._type_map.values())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, traces: List[ExecutionTrace]) -> List[ExecutionTrace]:
        """
        Normalize all traces to canonical format.

        Returns new ExecutionTrace objects with normalized steps.
        Wait durations are normalized to the median across all traces.
        """
        # First pass: normalize each step individually
        normalized: List[ExecutionTrace] = []
        for trace in traces:
            new_steps = [self.normalize_step(step) for step in trace.steps]
            normalized.append(
                replace(trace, steps=new_steps)
            )

        # Second pass: normalize wait durations across all traces
        self._normalize_wait_durations(normalized)

        return normalized

    def normalize_step(self, step: TraceStep) -> TraceStep:
        """
        Normalize a single trace step.

        - Maps action_type to canonical form
        - Handles UserInputsRequired → wait with args={"wait": True}
        - Normalizes target representation
        """
        raw_type = step.action_type
        canonical_type = self.normalize_action_type(raw_type)
        new_target = self.normalize_target(step.target)

        # Build updated metadata
        new_metadata = dict(step.metadata)

        # Flag unrecognized types
        if canonical_type not in self._canonical_types:
            new_metadata["unrecognized_action_type"] = True

        # Handle UserInputsRequired → wait with human confirmation
        new_args = dict(step.args) if step.args else None
        if raw_type == "UserInputsRequired":
            new_args = new_args or {}
            new_args["wait"] = True

        return replace(
            step,
            action_type=canonical_type,
            target=new_target,
            args=new_args,
            metadata=new_metadata,
        )

    def normalize_action_type(self, raw_type: str) -> str:
        """
        Map raw action type to canonical type.

        1. If raw_type is in the type map, return the mapped value.
        2. Otherwise, pass through as-is (it may already be canonical
           like "visit_url", "wait", etc.).
        3. Unrecognized types are flagged in metadata by normalize_step,
           not here — this method only does the mapping.
        """
        return self._type_map.get(raw_type, raw_type)

    def normalize_target(self, target: Any) -> Any:
        """
        Normalize target to canonical TargetSpec format.

        For now, targets are passed through as-is. Target conversion
        to stable selectors is handled by TargetStrategyConverter
        separately.
        """
        return target

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_wait_durations(
        self, traces: List[ExecutionTrace]
    ) -> None:
        """
        Normalize wait durations to the median across all traces.

        Collects all wait step durations (from args["seconds"]),
        computes the median, and updates every wait step in-place.
        """
        # Collect all wait durations
        durations: List[float] = []
        for trace in traces:
            for step in trace.steps:
                if step.action_type == "wait" and step.args:
                    seconds = step.args.get("seconds")
                    if seconds is not None:
                        try:
                            durations.append(float(seconds))
                        except (TypeError, ValueError):
                            pass

        if not durations:
            return

        median_duration = statistics.median(durations)

        # Update all wait steps with the median duration
        for trace in traces:
            for step in trace.steps:
                if step.action_type == "wait" and step.args:
                    if "seconds" in step.args:
                        step.args["seconds"] = median_duration
