
# pyre-strict

"""Formats ToolDefinition objects into docstring-like API markdown.

Supports two modes:
  1. Template-driven: Jinja2 templates in tools/templates/{tool_type}.jinja2
  2. Fallback: Hardcoded Python formatting (format_tool, format_parameter, etc.)

Templates are tried first. If a template doesn't exist for a given tool_type,
the fallback Python formatting is used. Adding a new tool_type only requires
a new template file — no Python changes needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from attr import attrs, attrib

from agent_foundation.resources.tools.models import (
    ParameterDef,
    SubcommandDef,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

_TEMPLATES_DIR: Path = Path(__file__).parent.parent / "templates"


@attrs
class ToolMarkdownFormatter:
    """Formats ToolDefinition objects into docstring-like API markdown.

    Output resembles Python docstring format but in markdown, suitable
    for injection into LLM conversation prompts.

    Templates in tools/templates/{tool_type}.jinja2 are used when available.
    Falls back to hardcoded Python formatting otherwise.
    """

    include_examples: bool = attrib(default=True)
    include_returns: bool = attrib(default=True)
    compact: bool = attrib(default=False)

    def _load_template(self, tool_type: str):
        """Load a Jinja2 template for the given tool_type, or return None."""
        template_path = _TEMPLATES_DIR / f"{tool_type.lower()}.jinja2"
        if not template_path.is_file():
            return None
        try:
            from jinja2 import Template

            return Template(template_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load tool template: %s", template_path)
            return None

    def format_tool(self, tool: ToolDefinition) -> str:
        """Format a single ToolDefinition into API-doc markdown."""
        lines: list[str] = []

        # Header
        header = f"### {tool.name}"
        if tool.aliases:
            header += f" (aliases: {', '.join(tool.aliases)})"
        lines.append(header)
        if tool.description:
            lines.append(tool.description)

        # Parameters (for tools without subcommands)
        if tool.parameters and not tool.subcommands:
            lines.append("")
            lines.append("**Parameters:**")
            for param in tool.parameters:
                lines.append(self.format_parameter(param))

        # Subcommands
        if tool.subcommands:
            lines.append("")
            lines.append("**Subcommands:**")
            for sub in tool.subcommands:
                lines.append("")
                lines.append(self.format_subcommand(sub, tool.name))

        # Returns
        if self.include_returns and tool.returns:
            lines.append("")
            lines.append(f"**Returns:** {tool.returns}")

        # Examples
        if self.include_examples and tool.examples:
            lines.append("")
            lines.append("**Examples:**")
            for ex in tool.examples:
                lines.append(f"- `{ex}`")

        return "\n".join(lines)

    def format_parameter(self, param: ParameterDef) -> str:
        """Format: `--name` (type, required/optional): Description."""
        name_part = f"`{param.name}`"

        qualifiers: list[str] = [param.type]
        if param.required:
            qualifiers.append("required")
        if param.default is not None:
            qualifiers.append(f"default: {param.default}")
        if param.choices:
            qualifiers.append(f"choices: {', '.join(param.choices)}")
        qualifier_str = ", ".join(qualifiers)

        line = f"- {name_part} ({qualifier_str})"
        if param.description:
            line += f": {param.description}"
        return line

    def format_subcommand(self, sub: SubcommandDef, parent_name: str = "") -> str:
        """Format a subcommand with its own parameter block."""
        lines: list[str] = []
        prefix = parent_name or ""
        lines.append(f"#### {prefix} {sub.name}")
        if sub.description:
            lines.append(sub.description)
        if sub.parameters:
            for param in sub.parameters:
                lines.append(self.format_parameter(param))
        return "\n".join(lines)

    def format_all(self, tools: list[ToolDefinition]) -> str:
        """Format all tools into a combined markdown block for prompt injection."""
        sections: list[str] = []
        for tool in tools:
            sections.append(self.format_tool(tool))
        return "\n\n".join(sections)

    def format_by_type(self, tools: list[ToolDefinition]) -> str:
        """Format tools grouped by tool_type using templates when available.

        Groups tools by their tool_type, then for each group:
        1. Try to load templates/{tool_type.lower()}.jinja2
        2. If found, render the template with the tools as context
        3. If not found, fall back to format_all() (generic action-style)

        Adding a new tool_type only requires a new template file — no Python changes.
        """
        # Group tools by tool_type, preserving insertion order
        groups: dict[str, list[ToolDefinition]] = {}
        for t in tools:
            groups.setdefault(t.tool_type, []).append(t)

        sections: list[str] = []
        for tool_type, group in groups.items():
            template = self._load_template(tool_type)
            if template:
                rendered = template.render(
                    tools=[t.to_dict() for t in group],
                ).strip()
                sections.append(rendered)
            else:
                # Fallback: generic action-style formatting
                sections.append(self.format_all(group))
        return "\n\n".join(sections)

    def _format_conversation_tools(self, tools: list[ToolDefinition]) -> str:
        """Format conversation tools — tries template, falls back to generic."""
        template = self._load_template("Conversation")
        if template:
            return template.render(
                tools=[t.to_dict() for t in tools],
            ).strip()
        # Fallback: generic action-style formatting
        return self.format_all(tools)
