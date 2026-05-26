"""SOP linter — static validation of SOP files.

Used by `af-sop inspect` to catch common errors before execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOP,
    SOPManager,
    SOPPhase,
)

logger = logging.getLogger(__name__)


class LintResult:
    """Collection of lint findings for an SOP file."""

    def __init__(self, source: str = ""):
        self.source = source
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        lines = [f"Lint: {self.source}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        if self.ok and not self.warnings:
            lines.append("  OK (no issues)")
        return "\n".join(lines)


def lint_sop(sop: SOP, source: str = "") -> LintResult:
    """Lint a parsed SOP for common issues."""
    result = LintResult(source=source)

    if not sop.phases:
        result.error("SOP has no phases")
        return result

    # Check for [__initial__] phase
    initial_phases = [p for p in sop.phases if "initial" in getattr(p, "directives", [])]
    if not initial_phases:
        result.error(
            "No [__initial__] phase declared. "
            "Mark at least one phase with [__initial__]."
        )

    # Check for duplicate phase IDs
    seen_ids: set[str] = set()
    for phase in sop.phases:
        if phase.id in seen_ids:
            result.error(f"Duplicate phase ID: {phase.id}")
        seen_ids.add(phase.id)

    # Check goto targets exist
    for phase in sop.phases:
        if phase.goto_target and phase.goto_target not in seen_ids:
            result.error(
                f"Phase {phase.id}: __goto__ target 'Phase {phase.goto_target}' "
                f"does not exist. Known phases: {sorted(seen_ids)}"
            )

    # Check depends_on references exist
    for phase in sop.phases:
        for dep in phase.depends_on:
            if dep not in seen_ids:
                result.error(
                    f"Phase {phase.id}: depends_on 'Phase {dep}' "
                    f"does not exist. Known phases: {sorted(seen_ids)}"
                )

    # Check __branch__ has __depends on__
    for phase in sop.phases:
        if getattr(phase, "branch", False) and not phase.depends_on:
            result.error(
                f"Phase {phase.id}: __branch__ requires __depends on__ "
                f"to identify the source phase."
            )

    # Warn about unknown tags
    for phase in sop.phases:
        for tag in getattr(phase, "unknown_tags", []):
            result.warn(f"Phase {phase.id}: unknown tag '{tag}'")

    return result


def lint_file(path: Path) -> LintResult:
    """Load and lint an SOP markdown file."""
    try:
        sop = SOPManager.load(path)
    except Exception as e:
        result = LintResult(source=str(path))
        result.error(f"Failed to parse: {e}")
        return result

    return lint_sop(sop, source=str(path))
