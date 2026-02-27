"""
Target Converter base classes for the Meta Agent Workflow pipeline.

Provides:

- **TargetSpec / TargetSpecWithFallback** — lightweight data containers for
  selector strategies (domain-agnostic).
- **TargetConverterBase** — abstract base class for domain-specific target
  conversion (e.g., web selectors, API endpoints).

The web-specific implementation (``WebTargetConverter``) lives in
``webaxon.automation.meta_agent.web_target_converter``.  The legacy
``TargetStrategyConverter`` name is kept as a lazy re-export for backward
compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from agent_foundation.automation.meta_agent.models import TraceStep

# ---------------------------------------------------------------------------
# Local selector data models (kept lightweight to avoid depending on the
# full ActionGraph schema).
# ---------------------------------------------------------------------------


@dataclass
class TargetSpec:
    """A single selector strategy and its value."""

    strategy: str
    value: str


@dataclass
class TargetSpecWithFallback:
    """Ordered list of selector strategies for a single element."""

    strategies: List[TargetSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base class for target conversion
# ---------------------------------------------------------------------------


class TargetConverterBase(ABC):
    """Abstract base for domain-specific target conversion.

    Subclasses implement :meth:`convert` for their domain (web selectors,
    API endpoints, etc.).  The :meth:`convert_all` convenience method
    applies conversion to steps whose targets match :meth:`should_convert`.
    """

    @abstractmethod
    def convert(self, step: TraceStep) -> TargetSpecWithFallback:
        """Convert a single step's target to stable selectors.

        Only called for steps where :meth:`should_convert` returned
        ``True``.  Implementations should assume the target needs
        conversion and produce a :class:`TargetSpecWithFallback`.
        """
        ...

    def should_convert(self, step: TraceStep) -> bool:
        """Return ``True`` if this step's target should be converted.

        Default implementation matches the pipeline's current behavior:
        converts targets that are strings starting with ``"__id__"``.
        Subclasses may override for domain-specific filtering.
        """
        target = step.target
        return isinstance(target, str) and target.startswith("__id__")

    def convert_all(self, steps: List[TraceStep]) -> List[TraceStep]:
        """Apply :meth:`convert` to steps that :meth:`should_convert`.

        Updates ``step.target`` in-place.  Returns the same list for
        chaining convenience.
        """
        for step in steps:
            if self.should_convert(step):
                step.target = self.convert(step)
        return steps


# ---------------------------------------------------------------------------
# Backward-compatible lazy re-export of TargetStrategyConverter
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    if name == "TargetStrategyConverter":
        try:
            from webaxon.automation.meta_agent.web_target_converter import (
                WebTargetConverter,
            )

            return WebTargetConverter
        except ImportError:
            raise ImportError(
                "TargetStrategyConverter has moved to "
                "webaxon.automation.meta_agent.web_target_converter.WebTargetConverter. "
                "Install WebAgent or use TargetConverterBase for a custom implementation."
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
