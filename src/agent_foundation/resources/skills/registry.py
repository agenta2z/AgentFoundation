"""Load SKILL.md files into SkillInfo instances.

Parallel to tools/registry.py. Scans resources/skills/*/SKILL.md,
parses YAML frontmatter via parse_skill_md(), returns structured info.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SKILLS_DIR: Path = Path(__file__).parent


@dataclass
class SkillInfo:
    """Parsed skill metadata from a SKILL.md file."""

    name: str
    description: str = ""
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    body: str = ""  # markdown body (without frontmatter)


def _parse_skill_md(content: str) -> tuple[dict[str, Any], str]:
    """Parse SKILL.md into (frontmatter_dict, body_text).

    Inlined to avoid heavy knowledge package imports.
    Uses yaml if available, falls back to basic parsing.
    """
    import re

    pattern = r"^---\s*\n(.*?)^---\s*\n"
    match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
    if not match:
        return {}, content.strip()

    frontmatter_raw = match.group(1)
    body = content[match.end():].strip()

    try:
        import yaml
        frontmatter = yaml.safe_load(frontmatter_raw) or {}
    except ImportError:
        # Fallback: extract name and description from YAML-like text
        frontmatter = {}
        for line in frontmatter_raw.splitlines():
            line = line.strip()
            if line.startswith("name:"):
                frontmatter["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                val = line.split(":", 1)[1].strip()
                if val and val != ">":
                    frontmatter["description"] = val
            elif line.startswith("- ") and "labels" not in frontmatter:
                pass  # skip list items in fallback
        # Try multi-line description
        if "description" not in frontmatter:
            desc_match = re.search(
                r"description:\s*>\s*\n((?:\s+.+\n?)+)", frontmatter_raw
            )
            if desc_match:
                frontmatter["description"] = " ".join(
                    l.strip() for l in desc_match.group(1).splitlines()
                )
    except Exception:
        frontmatter = {}

    return frontmatter, body


def load_skill(name: str, base_dir: Path | None = None) -> SkillInfo:
    """Load a single skill from its SKILL.md file."""
    skill_md = (base_dir or _SKILLS_DIR) / name / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _parse_skill_md(content)

    return SkillInfo(
        name=frontmatter.get("name", name),
        description=frontmatter.get("description", ""),
        labels=frontmatter.get("labels", []),
        metadata=frontmatter.get("metadata", {}),
        file_path=str(skill_md),
        body=body,
    )


def load_all_skills(extra_dirs: list[str | Path] | None = None) -> dict[str, SkillInfo]:
    """Load all skills from framework and optional extra directories.

    Args:
        extra_dirs: Additional directories to scan for SKILL.md files.
            Application-specific skills (e.g., Slack, TWG) can live outside
            the framework package and be loaded via this parameter.

    Returns:
        {name: SkillInfo} — extra_dirs skills override framework skills
        with the same name.
    """
    skills: dict[str, SkillInfo] = {}
    dirs_to_scan = [_SKILLS_DIR] + [Path(d) for d in (extra_dirs or [])]
    for skills_dir in dirs_to_scan:
        if not skills_dir.is_dir():
            continue
        for child in sorted(skills_dir.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                try:
                    skill = load_skill(child.name, base_dir=skills_dir)
                    skills[skill.name] = skill
                except Exception as e:
                    logger.warning("Failed to load skill %s: %s", child.name, e)
    return skills


def format_all_skills(extra_dirs: list[str | Path] | None = None) -> str:
    """Format all skills as markdown text for prompt injection."""
    skills = load_all_skills(extra_dirs=extra_dirs)
    if not skills:
        return ""

    lines = []
    for name, si in sorted(skills.items()):
        desc = si.description.strip().replace("\n", " ")[:150]
        labels = f" [{', '.join(si.labels)}]" if si.labels else ""
        lines.append(f"- **{name}**{labels}: {desc}")
    return "\n".join(lines)
