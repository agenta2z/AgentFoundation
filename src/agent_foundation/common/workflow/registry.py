"""WorkflowRegistry — discovers WorkflowDefinitions from resource directories.

Mirrors the existing tools/registry.py and skills/registry.py patterns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOPManager,
)

from agent_foundation.common.workflow.definition import WorkflowDefinition

logger = logging.getLogger(__name__)


class WorkflowNotFound(KeyError):
    """Raised when a workflow definition is not found."""


class WorkflowRegistry:
    """Discovers WorkflowDefinitions from one or more resource directories."""

    def __init__(self, search_paths: list[Path] | None = None):
        self._search_paths = search_paths or self._default_search_paths()
        self._definitions: dict[str, WorkflowDefinition] = {}

    def load_all(self) -> dict[str, WorkflowDefinition]:
        """Scan all search paths for *.md files; parse each as a WorkflowDefinition."""
        for path in self._search_paths:
            if not path.is_dir():
                continue
            for md_file in path.rglob("*.md"):
                try:
                    definition = self._parse_definition(md_file)
                    if definition.workflow_id in self._definitions:
                        logger.warning(
                            "Duplicate workflow id %r in %s",
                            definition.workflow_id,
                            md_file,
                        )
                    self._definitions[definition.workflow_id] = definition
                except Exception as e:
                    logger.warning("Failed to parse workflow %s: %s", md_file, e)
        return dict(self._definitions)

    def get(self, definition_id: str) -> WorkflowDefinition:
        if definition_id not in self._definitions:
            raise WorkflowNotFound(definition_id)
        return self._definitions[definition_id]

    def list_all(self) -> list[WorkflowDefinition]:
        return list(self._definitions.values())

    def _parse_definition(self, md_file: Path) -> WorkflowDefinition:
        raw_markdown = md_file.read_text(encoding="utf-8")
        sop = SOPManager.parse_markdown(raw_markdown)

        workflow_id = md_file.parent.name if md_file.stem == "SOP" else md_file.stem
        name = workflow_id.replace("_", " ").replace("-", " ").title()
        description = ""

        lines = raw_markdown.split("\n")
        desc_lines = []
        for line in lines:
            if line.strip().startswith("## Phase") or line.strip().startswith("### Phase"):
                break
            if line.strip().startswith("# "):
                name = line.strip().lstrip("# ").strip()
                continue
            desc_lines.append(line)
        description = "\n".join(desc_lines).strip()

        available_tools = list(sop.tool_to_phase_map.keys()) if hasattr(sop, "tool_to_phase_map") else []

        return WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description[:500],
            source_path=md_file,
            sop=sop,
            raw_markdown=raw_markdown,
            available_tools=available_tools,
        )

    @staticmethod
    def _default_search_paths() -> list[Path]:
        paths = []

        # Primary: new resources/sops/ layout (PR-1 migration)
        af_sops = (
            Path(__file__).resolve().parent.parent.parent
            / "resources"
            / "sops"
        )
        if af_sops.is_dir():
            for child in sorted(af_sops.iterdir()):
                sop_md = child / "SOP.md"
                if child.is_dir() and sop_md.is_file():
                    paths.append(child)

        user_dir = Path.home() / ".agent_foundation" / "workflows"
        if user_dir.is_dir():
            paths.append(user_dir)

        env_paths = os.environ.get("AGENT_FOUNDATION_WORKFLOW_PATH", "")
        if env_paths:
            for p in env_paths.split(":"):
                pp = Path(p.strip())
                if pp.is_dir():
                    paths.append(pp)

        return paths
