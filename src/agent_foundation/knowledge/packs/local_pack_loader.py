"""
LocalPackLoader — load knowledge packs from local directories or JSON files.

Supports two loading modes:
1. Directory: reads a ``pack.json`` manifest + ``*.md`` / supporting files
2. JSON file: reads ``{manifest: {...}, pieces: [...]}`` format
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from agent_foundation.knowledge.packs.clawhub_adapter import parse_skill_md
from agent_foundation.knowledge.packs.models import (
    KnowledgePack,
    PackInstallResult,
    PackSource,
)
from agent_foundation.knowledge.packs.pack_manager import KnowledgePackManager
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)

logger = logging.getLogger(__name__)

# File extensions treated as skill content (markdown)
SKILL_EXTENSIONS = {".md", ".mdx"}

# Text file extensions for supporting files
TEXT_EXTENSIONS = {
    ".py", ".sh", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
    ".toml", ".rb", ".go", ".rs", ".swift", ".kt", ".java", ".cs", ".cpp",
    ".c", ".h", ".hpp", ".sql", ".csv", ".ini", ".cfg", ".xml", ".html",
    ".css", ".scss", ".sass", ".svg", ".txt", ".md", ".mdx",
}


class LocalPackLoader:
    """Loads knowledge packs from local filesystem sources.

    Args:
        pack_manager: The KnowledgePackManager to install packs into.
    """

    def __init__(self, pack_manager: KnowledgePackManager):
        self.pack_manager = pack_manager

    def load_from_directory(
        self,
        directory: str,
        name: Optional[str] = None,
        spaces: Optional[List[str]] = None,
    ) -> PackInstallResult:
        """Load a pack from a local directory.

        Expected directory structure:
        - ``pack.json`` (optional): manifest with name, version, description, tags
        - ``SKILL.md`` or ``*.md``: primary skill file(s)
        - Any text-based supporting files

        If ``pack.json`` is absent, the directory name is used as the pack name.

        Args:
            directory: Path to the directory containing pack files.
            name: Override pack name (default: from manifest or directory name).

        Returns:
            PackInstallResult with install status.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return PackInstallResult(
                success=False,
                pack_id="",
                error=f"Directory not found: {directory}",
            )

        # Load manifest if present
        manifest_path = dir_path / "pack.json"
        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception as e:
                logger.warning("Failed to read pack.json: %s", e)

        pack_name = name or manifest.get("name") or dir_path.name
        pack_id = f"pack:local:{pack_name}"

        if self.pack_manager.is_installed(pack_id):
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"Pack already installed: {pack_id}",
            )

        # Collect files
        pieces = []
        skill_file = None

        # Look for SKILL.md first
        for candidate in ["SKILL.md", "skill.md"]:
            candidate_path = dir_path / candidate
            if candidate_path.exists():
                skill_file = candidate_path
                break

        # If no SKILL.md, use the first .md file
        if skill_file is None:
            md_files = sorted(dir_path.glob("*.md"))
            if md_files:
                skill_file = md_files[0]

        if skill_file is not None:
            content = skill_file.read_text(encoding="utf-8")
            frontmatter, body = parse_skill_md(content)

            primary_piece = KnowledgePiece(
                content=body,
                piece_id=str(uuid.uuid4()),
                knowledge_type=KnowledgeType.Procedure,
                info_type="skills",
                source=f"pack:local:{pack_name}",
                domain=manifest.get("domain", "agent_skills"),
                custom_tags=[
                    f"pack:local:{pack_name}",
                    f"pack-version:{manifest.get('version', '0.0.0')}",
                ],
                summary=(
                    frontmatter.get("description")
                    or manifest.get("description", "")
                ),
            )
            pieces.append(primary_piece)

            # Merge frontmatter into manifest (frontmatter takes precedence)
            if frontmatter.get("description") and not manifest.get("description"):
                manifest["description"] = frontmatter["description"]
            if frontmatter.get("version") and not manifest.get("version"):
                manifest["version"] = frontmatter["version"]

        # Add supporting files
        for file_path in sorted(dir_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path == skill_file:
                continue
            if file_path.name == "pack.json":
                continue
            if file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            try:
                file_content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            rel_path = file_path.relative_to(dir_path).as_posix()
            supporting_piece = KnowledgePiece(
                content=f"## File: {rel_path}\n\n{file_content}",
                piece_id=str(uuid.uuid4()),
                knowledge_type=KnowledgeType.Fact,
                info_type="context",
                source=f"pack:local:{pack_name}",
                domain=manifest.get("domain", "agent_skills"),
                custom_tags=[
                    f"pack:local:{pack_name}",
                    f"pack-version:{manifest.get('version', '0.0.0')}",
                    f"pack-file:{rel_path}",
                ],
                summary=f"Supporting file '{rel_path}' for pack '{pack_name}'",
            )
            pieces.append(supporting_piece)

        if not pieces:
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"No loadable files found in: {directory}",
            )

        pack = KnowledgePack(
            pack_id=pack_id,
            name=pack_name,
            version=manifest.get("version", "0.0.0"),
            description=manifest.get("description", ""),
            source_type=PackSource.LOCAL,
            source_url=str(dir_path.resolve()),
            source_identifier=pack_name,
            tags=manifest.get("tags", []),
            requirements=manifest.get("requirements", {}),
            properties=manifest.get("properties", {}),
            spaces=spaces or manifest.get("spaces"),
        )

        return self.pack_manager.install(pack, pieces)

    def load_from_json(
        self,
        file_path: str,
    ) -> PackInstallResult:
        """Load a pack from a JSON file.

        Expected format::

            {
                "manifest": {
                    "name": "my-pack",
                    "version": "1.0.0",
                    "description": "...",
                    "tags": [...],
                    "requirements": {...}
                },
                "pieces": [
                    {
                        "content": "...",
                        "knowledge_type": "procedure",
                        "info_type": "skills",
                        ...
                    }
                ]
            }

        Args:
            file_path: Path to the JSON file.

        Returns:
            PackInstallResult with install status.
        """
        path = Path(file_path)
        if not path.exists():
            return PackInstallResult(
                success=False,
                pack_id="",
                error=f"File not found: {file_path}",
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return PackInstallResult(
                success=False,
                pack_id="",
                error=f"Failed to read JSON: {e}",
            )

        manifest = data.get("manifest", {})
        piece_dicts = data.get("pieces", [])

        pack_name = manifest.get("name", path.stem)
        pack_id = f"pack:local:{pack_name}"

        if self.pack_manager.is_installed(pack_id):
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"Pack already installed: {pack_id}",
            )

        if not piece_dicts:
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error="No pieces found in JSON file",
            )

        # Build pieces from dicts, stamping provenance
        pieces = []
        for piece_dict in piece_dicts:
            piece_dict.setdefault("source", f"pack:local:{pack_name}")
            piece_dict.setdefault("domain", manifest.get("domain", "agent_skills"))

            custom_tags = piece_dict.get("custom_tags", [])
            if f"pack:local:{pack_name}" not in custom_tags:
                custom_tags.append(f"pack:local:{pack_name}")
            version_tag = f"pack-version:{manifest.get('version', '0.0.0')}"
            if version_tag not in custom_tags:
                custom_tags.append(version_tag)
            piece_dict["custom_tags"] = custom_tags

            try:
                piece = KnowledgePiece.from_dict(piece_dict)
                pieces.append(piece)
            except Exception as e:
                logger.warning("Skipping invalid piece: %s", e)

        pack = KnowledgePack(
            pack_id=pack_id,
            name=pack_name,
            version=manifest.get("version", "0.0.0"),
            description=manifest.get("description", ""),
            source_type=PackSource.LOCAL,
            source_url=str(path.resolve()),
            source_identifier=pack_name,
            tags=manifest.get("tags", []),
            requirements=manifest.get("requirements", {}),
            properties=manifest.get("properties", {}),
        )

        return self.pack_manager.install(pack, pieces)
