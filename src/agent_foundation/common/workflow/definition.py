"""WorkflowDefinition — a discovered, parsed workflow. Immutable after construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich_python_utils.string_utils.formatting.template_manager.sop_manager import SOP


@dataclass(frozen=True)
class WorkflowDefinition:
    """Static workflow definition parsed from an SOP markdown file."""

    workflow_id: str
    name: str
    description: str
    source_path: Path
    sop: SOP
    raw_markdown: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    available_tools: list[str] = field(default_factory=list)
    requires_tools: list[str] = field(default_factory=list)
    available_modes: list[str] = field(default_factory=lambda: ["default", "yolo"])
