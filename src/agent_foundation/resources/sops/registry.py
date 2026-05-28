"""SOPRegistry — discovers SOP definitions from resource directories.

Mirrors the skills/registry.py and tools/registry.py patterns.
Each SOP lives in a directory with SOP.md + sop.config.json.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOP,
    SOPManager,
)

logger = logging.getLogger(__name__)

_SOPS_DIR = Path(__file__).resolve().parent


class SOPNotFound(KeyError):
    pass


@dataclass(frozen=True)
class SOPInfo:
    """A discovered SOP definition with its config."""

    name: str
    display_name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    example_requests: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    available_modes: list[str] = field(default_factory=lambda: ["default", "yolo"])
    requires_tools: list[str] = field(default_factory=list)
    yolo_overrides: dict[str, dict] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    body_path: Path = field(default_factory=lambda: Path("."))
    body: str = ""
    folder: Path = field(default_factory=lambda: Path("."))
    sop: SOP = field(default_factory=lambda: SOP([]))


def load_sop(name: str, base_dir: Path | None = None) -> SOPInfo:
    """Load a single SOP from <base_dir>/<name>/SOP.md + sop.config.json."""
    base = base_dir or _SOPS_DIR
    sop_dir = base / name
    sop_md = sop_dir / "SOP.md"
    config_path = sop_dir / "sop.config.json"

    if not sop_md.is_file():
        raise SOPNotFound(f"SOP.md not found at {sop_md}")

    body = sop_md.read_text(encoding="utf-8")
    sop = SOPManager.parse_markdown(body)

    config: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    # Extract description from preamble (lines before first ## Phase)
    desc_lines = []
    for line in body.split("\n"):
        if line.strip().startswith("## Phase") or line.strip().startswith("### Phase"):
            break
        if line.strip().startswith("[__"):
            continue
        if line.strip().startswith("# "):
            continue
        desc_lines.append(line)
    md_description = "\n".join(desc_lines).strip()[:500]

    description = config.get("description") or md_description
    display_name = config.get("display_name") or name.replace("_", " ").replace("-", " ").title()

    # Merge keywords/example_requests: sop.config.json primary, SOP.md fallback
    merge = config.get("_merge_with_markdown", False)
    kw_config = config.get("keywords", [])
    kw_md = sop.keywords
    ex_config = config.get("example_requests", [])
    ex_md = sop.example_requests

    if merge:
        keywords = list(dict.fromkeys(kw_config + kw_md))
        example_requests = list(dict.fromkeys(ex_config + ex_md))
    else:
        keywords = kw_config if kw_config else kw_md
        example_requests = ex_config if ex_config else ex_md

    return SOPInfo(
        name=config.get("name", name),
        display_name=display_name,
        description=description,
        keywords=keywords,
        example_requests=example_requests,
        labels=config.get("labels", []),
        available_modes=config.get("available_modes", ["default", "yolo"]),
        requires_tools=config.get("requires_tools", []),
        yolo_overrides=config.get("yolo_overrides", {}),
        config=config,
        body_path=sop_md,
        body=body,
        folder=sop_dir,
        sop=sop,
    )


def load_all_sops(
    extra_dirs: list[str | Path] | None = None,
) -> dict[str, SOPInfo]:
    """Load all SOPs from framework directory + optional extra directories.

    Later directories override earlier ones on name collision (matches
    load_all_skills/load_all_tools convention).
    """
    search_dirs = [_SOPS_DIR]
    if extra_dirs:
        search_dirs.extend(Path(d) for d in extra_dirs)

    sops: dict[str, SOPInfo] = {}
    for base_dir in search_dirs:
        if not base_dir.is_dir():
            continue
        for child in sorted(base_dir.iterdir()):
            if not child.is_dir():
                continue
            sop_md = child / "SOP.md"
            if not sop_md.is_file():
                continue
            try:
                info = load_sop(child.name, base_dir=base_dir)
                if info.name in sops:
                    logger.warning(
                        "SOP %r in %s overrides existing from %s",
                        info.name,
                        base_dir,
                        sops[info.name].folder,
                    )
                sops[info.name] = info
            except Exception as e:
                logger.warning("Failed to load SOP %s: %s", child.name, e)

    return sops


def format_all_sops(sops: dict[str, SOPInfo] | None = None) -> str:
    """Format all SOPs as a markdown summary for prompt injection."""
    if sops is None:
        sops = load_all_sops()
    if not sops:
        return ""

    lines = []
    for info in sops.values():
        lines.append(f"- **{info.display_name}** (`{info.name}`)")
        if info.description:
            desc = info.description[:200]
            lines.append(f"  {desc}")
        if info.keywords:
            lines.append(f"  Keywords: {', '.join(info.keywords[:8])}")
        if info.example_requests:
            examples = "; ".join(f'"{e}"' for e in info.example_requests[:3])
            lines.append(f"  Examples: {examples}")
    return "\n".join(lines)
