"""
ClawHub integration adapter for Knowledge Packs.

Provides:
- ClawhubClient: HTTP client for the ClawHub public API
- parse_skill_md: YAML frontmatter parser for SKILL.md files
- ClawhubPackAdapter: Imports ClawHub skills as Knowledge Packs
"""

import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import yaml

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

# ClawHub API constants
DEFAULT_BASE_URL = "https://clawhub.ai"
API_PREFIX = "/api/v1"
MAX_FILE_SIZE = 200 * 1024  # 200KB per-file limit


def parse_skill_md(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse a SKILL.md file into frontmatter dict and body.

    Extracts YAML frontmatter delimited by ``---`` at the start of the file.
    If no frontmatter is found, returns an empty dict and the full content.

    Args:
        content: The raw SKILL.md text.

    Returns:
        A tuple of (frontmatter_dict, body_text).
    """
    pattern = r"^---\s*\n(.*?)^---\s*\n"
    match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
    if not match:
        return {}, content.strip()

    frontmatter_raw = match.group(1)
    body = content[match.end():].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_raw) or {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse YAML frontmatter: %s", e)
        frontmatter = {}

    return frontmatter, body


class ClawhubClient:
    """HTTP client for the ClawHub public API.

    Provides typed methods for read-only operations. Handles rate limiting
    by respecting Retry-After headers.

    Args:
        base_url: Registry base URL (default: https://clawhub.ai).
        token: Optional API token for authenticated requests.
        session: Optional requests.Session for connection reuse.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        token: Optional[str] = None,
        session=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token

        if session is not None:
            self._session = session
        else:
            import requests
            self._session = requests.Session()

        if self.token:
            self._session.headers["Authorization"] = f"Bearer {self.token}"
        self._session.headers["Accept"] = "application/json"

    def _url(self, path: str) -> str:
        return f"{self.base_url}{API_PREFIX}{path}"

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        """Make a GET request with rate-limit retry."""
        url = self._url(path)
        response = self._session.get(url, params=params)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "5")
            wait = int(float(retry_after))
            logger.warning("Rate limited, retrying in %ds", wait)
            time.sleep(wait)
            response = self._session.get(url, params=params)

        response.raise_for_status()
        return response.json()

    def _get_text(self, path: str, params: Optional[Dict] = None) -> str:
        """Make a GET request that returns plain text."""
        url = self._url(path)
        response = self._session.get(url, params=params)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "5")
            wait = int(float(retry_after))
            logger.warning("Rate limited, retrying in %ds", wait)
            time.sleep(wait)
            response = self._session.get(url, params=params)

        response.raise_for_status()
        return response.text

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Search for skills.

        Args:
            query: Search query string.
            limit: Max results.

        Returns:
            List of search result dicts with slug, displayName, summary, etc.
        """
        params = {"q": query}
        if limit is not None:
            params["limit"] = str(limit)
        data = self._get("/search", params=params)
        return data.get("results", [])

    def list_skills(
        self,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
    ) -> List[Dict]:
        """List skills from the registry.

        Args:
            limit: Max results (1-200).
            sort: Sort order (updated, downloads, stars, trending).

        Returns:
            List of skill item dicts.
        """
        params = {}
        if limit is not None:
            params["limit"] = str(limit)
        if sort is not None:
            params["sort"] = sort
        data = self._get("/skills", params=params)
        return data.get("items", [])

    def get_skill(self, slug: str) -> Dict:
        """Get skill metadata by slug.

        Args:
            slug: The skill slug.

        Returns:
            Dict with skill, latestVersion, metadata, owner fields.
        """
        return self._get(f"/skills/{slug}")

    def get_version(self, slug: str, version: str) -> Dict:
        """Get specific version details including files manifest.

        Args:
            slug: The skill slug.
            version: Semver version string.

        Returns:
            Dict with skill, version (including files array) fields.
        """
        return self._get(f"/skills/{slug}/versions/{version}")

    def get_file(
        self,
        slug: str,
        path: str,
        version: Optional[str] = None,
    ) -> str:
        """Get raw file content from a skill.

        Args:
            slug: The skill slug.
            path: File path within the bundle.
            version: Optional version (defaults to latest).

        Returns:
            Raw text content of the file.
        """
        params = {"path": path}
        if version is not None:
            params["version"] = version
        return self._get_text(f"/skills/{slug}/file", params=params)


class ClawhubPackAdapter:
    """Imports ClawHub skills as Knowledge Packs.

    Fetches skill metadata and files from the ClawHub API, transforms them
    into KnowledgePiece objects, and installs them via KnowledgePackManager.

    Args:
        pack_manager: The KnowledgePackManager to install packs into.
        client: A ClawhubClient for API access.
    """

    def __init__(
        self,
        pack_manager: KnowledgePackManager,
        client: Optional[ClawhubClient] = None,
    ):
        self.pack_manager = pack_manager
        self.client = client or ClawhubClient()

    @staticmethod
    def _make_pack_id(slug: str) -> str:
        return f"pack:clawhub:{slug}"

    def import_skill(
        self,
        slug: str,
        version: Optional[str] = None,
        spaces: Optional[List[str]] = None,
    ) -> PackInstallResult:
        """Import a ClawHub skill as a Knowledge Pack.

        Fetches the skill and all its files, creates KnowledgePieces, and
        installs them atomically via the pack manager.

        Args:
            slug: The skill slug on ClawHub.
            version: Specific version to import (default: latest).

        Returns:
            PackInstallResult with install status.
        """
        pack_id = self._make_pack_id(slug)

        if self.pack_manager.is_installed(pack_id):
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"Skill already installed: {slug}",
            )

        try:
            # Fetch skill metadata
            skill_data = self.client.get_skill(slug)
            skill_info = skill_data.get("skill", {})
            latest_version = skill_data.get("latestVersion", {})
            resolved_version = version or (latest_version.get("version") if latest_version else None)

            if not resolved_version:
                return PackInstallResult(
                    success=False,
                    pack_id=pack_id,
                    error=f"No version found for skill: {slug}",
                )

            # Fetch version details for files manifest
            version_data = self.client.get_version(slug, resolved_version)
            version_info = version_data.get("version", {})
            files_manifest = version_info.get("files", [])

            # Fetch SKILL.md content
            skill_md_content = self.client.get_file(
                slug, "SKILL.md", version=resolved_version
            )
            frontmatter, body = parse_skill_md(skill_md_content)

            # Extract metadata from frontmatter
            metadata_section = frontmatter.get("metadata", {})
            openclaw = (
                metadata_section.get("openclaw")
                or metadata_section.get("clawdbot")
                or metadata_section.get("clawdis")
                or {}
            )
            requires = openclaw.get("requires", {})

            # Build bundle files manifest (for the primary piece properties)
            bundle_files = [
                {"path": f["path"], "size": f.get("size", 0)}
                for f in files_manifest
            ]

            # Extract content tags from frontmatter (NOT from skill_info.tags
            # which is Record<string, Id<'skillVersions'>> — version pointers)
            content_tags = frontmatter.get("tags", [])
            if not isinstance(content_tags, list):
                content_tags = []

            # Build pieces
            pieces = []

            # Primary piece: SKILL.md body
            primary_piece = KnowledgePiece(
                content=body,
                piece_id=str(uuid.uuid4()),
                knowledge_type=KnowledgeType.Procedure,
                info_type="skills",
                tags=content_tags,
                source=f"pack:clawhub:{slug}",
                domain="agent_skills",
                custom_tags=[
                    f"pack:clawhub:{slug}",
                    f"pack-version:{resolved_version}",
                ],
                summary=frontmatter.get("description") or skill_info.get("summary", ""),
            )
            # Store bundle manifest in the piece for retrieval discovery
            # We use embedding_text to store a concise version for search
            if len(bundle_files) > 1:
                file_list = ", ".join(f["path"] for f in bundle_files if f["path"].lower() != "skill.md")
                primary_piece.embedding_text = (
                    f"{primary_piece.summary or ''} "
                    f"(includes: {file_list})"
                )[:2000]
            pieces.append(primary_piece)

            # Supporting files
            for file_info in files_manifest:
                path = file_info["path"]
                size = file_info.get("size", 0)

                # Skip SKILL.md (already processed) and oversized files
                if path.lower() in ("skill.md", "skill.md"):
                    continue
                if size > MAX_FILE_SIZE:
                    logger.warning(
                        "Skipping oversized file %s (%d bytes) in %s",
                        path, size, slug,
                    )
                    continue

                try:
                    file_content = self.client.get_file(
                        slug, path, version=resolved_version
                    )
                except Exception as e:
                    logger.warning("Failed to fetch %s from %s: %s", path, slug, e)
                    continue

                supporting_piece = KnowledgePiece(
                    content=f"## File: {path}\n\n{file_content}",
                    piece_id=str(uuid.uuid4()),
                    knowledge_type=KnowledgeType.Fact,
                    info_type="context",
                    source=f"pack:clawhub:{slug}",
                    domain="agent_skills",
                    custom_tags=[
                        f"pack:clawhub:{slug}",
                        f"pack-version:{resolved_version}",
                        f"pack-file:{path}",
                    ],
                    summary=f"Supporting file '{path}' for skill '{slug}'",
                )
                pieces.append(supporting_piece)

            # Build pack model
            pack = KnowledgePack(
                pack_id=pack_id,
                name=skill_info.get("displayName") or frontmatter.get("name") or slug,
                version=resolved_version,
                description=frontmatter.get("description") or skill_info.get("summary", ""),
                source_type=PackSource.CLAWHUB,
                source_url=f"{self.client.base_url}/skills/{slug}",
                source_identifier=slug,
                tags=content_tags,
                requirements={
                    "env": requires.get("env", []),
                    "bins": requires.get("bins", []),
                    "any_bins": requires.get("anyBins", []),
                    "config": requires.get("config", []),
                    "os": openclaw.get("os", []),
                },
                properties={
                    "bundle_files": bundle_files,
                    "primary_env": openclaw.get("primaryEnv"),
                    "emoji": openclaw.get("emoji"),
                    "homepage": openclaw.get("homepage") or frontmatter.get("homepage"),
                    "always": openclaw.get("always", False),
                    "owner": skill_data.get("owner", {}).get("handle"),
                },
                spaces=spaces,
            )

            return self.pack_manager.install(pack, pieces)

        except Exception as e:
            logger.error("Failed to import skill %s: %s", slug, e)
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"Import failed: {e}",
            )

    def check_for_updates(self, pack_id: str) -> Optional[str]:
        """Check if a newer version is available for an installed pack.

        Args:
            pack_id: The pack ID (e.g., "pack:clawhub:todoist-cli").

        Returns:
            The newer version string if available, None if up to date.
        """
        pack = self.pack_manager.get(pack_id)
        if pack is None or pack.source_type != PackSource.CLAWHUB:
            return None

        slug = pack.source_identifier
        if not slug:
            return None

        try:
            skill_data = self.client.get_skill(slug)
            latest = skill_data.get("latestVersion", {})
            remote_version = latest.get("version") if latest else None

            if remote_version and remote_version != pack.version:
                return remote_version
            return None
        except Exception as e:
            logger.warning("Failed to check updates for %s: %s", pack_id, e)
            return None

    def update_skill(
        self,
        pack_id: str,
        version: Optional[str] = None,
        spaces: Optional[List[str]] = None,
    ) -> PackInstallResult:
        """Update an installed ClawHub skill to a new version.

        Fetches the new version and uses pack_manager.update() for atomic
        replacement (add new pieces first, deactivate old after).

        Args:
            pack_id: The pack ID to update.
            version: Specific version to update to (default: latest).

        Returns:
            PackInstallResult with update status.
        """
        pack = self.pack_manager.get(pack_id)
        if pack is None:
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"Pack not found: {pack_id}",
            )

        slug = pack.source_identifier
        if not slug:
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error="Pack has no source_identifier (slug)",
            )

        # Uninstall old, then import new
        # We use the update flow: build new pieces, then call pack_manager.update()
        try:
            skill_data = self.client.get_skill(slug)
            latest_version = skill_data.get("latestVersion", {})
            resolved_version = version or (latest_version.get("version") if latest_version else None)

            if not resolved_version:
                return PackInstallResult(
                    success=False,
                    pack_id=pack_id,
                    error=f"No version found for skill: {slug}",
                )

            # Fetch version details
            version_data = self.client.get_version(slug, resolved_version)
            version_info = version_data.get("version", {})
            files_manifest = version_info.get("files", [])
            skill_info = skill_data.get("skill", {})

            # Fetch and parse SKILL.md
            skill_md_content = self.client.get_file(
                slug, "SKILL.md", version=resolved_version
            )
            frontmatter, body = parse_skill_md(skill_md_content)

            metadata_section = frontmatter.get("metadata", {})
            openclaw = (
                metadata_section.get("openclaw")
                or metadata_section.get("clawdbot")
                or metadata_section.get("clawdis")
                or {}
            )
            requires = openclaw.get("requires", {})

            bundle_files = [
                {"path": f["path"], "size": f.get("size", 0)}
                for f in files_manifest
            ]

            # Extract content tags from frontmatter (NOT from skill_info.tags
            # which is Record<string, Id<'skillVersions'>> — version pointers)
            content_tags = frontmatter.get("tags", [])
            if not isinstance(content_tags, list):
                content_tags = []

            # Build new pieces
            new_pieces = []

            primary_piece = KnowledgePiece(
                content=body,
                piece_id=str(uuid.uuid4()),
                knowledge_type=KnowledgeType.Procedure,
                info_type="skills",
                tags=content_tags,
                source=f"pack:clawhub:{slug}",
                domain="agent_skills",
                custom_tags=[
                    f"pack:clawhub:{slug}",
                    f"pack-version:{resolved_version}",
                ],
                summary=frontmatter.get("description") or skill_info.get("summary", ""),
            )
            if len(bundle_files) > 1:
                file_list = ", ".join(f["path"] for f in bundle_files if f["path"].lower() != "skill.md")
                primary_piece.embedding_text = (
                    f"{primary_piece.summary or ''} "
                    f"(includes: {file_list})"
                )[:2000]
            new_pieces.append(primary_piece)

            for file_info in files_manifest:
                path = file_info["path"]
                size = file_info.get("size", 0)
                if path.lower() in ("skill.md",):
                    continue
                if size > MAX_FILE_SIZE:
                    continue
                try:
                    file_content = self.client.get_file(
                        slug, path, version=resolved_version
                    )
                except Exception:
                    continue

                supporting_piece = KnowledgePiece(
                    content=f"## File: {path}\n\n{file_content}",
                    piece_id=str(uuid.uuid4()),
                    knowledge_type=KnowledgeType.Fact,
                    info_type="context",
                    source=f"pack:clawhub:{slug}",
                    domain="agent_skills",
                    custom_tags=[
                        f"pack:clawhub:{slug}",
                        f"pack-version:{resolved_version}",
                        f"pack-file:{path}",
                    ],
                    summary=f"Supporting file '{path}' for skill '{slug}'",
                )
                new_pieces.append(supporting_piece)

            # Build updated pack
            updated_pack = KnowledgePack(
                pack_id=pack_id,
                name=skill_info.get("displayName") or frontmatter.get("name") or slug,
                version=resolved_version,
                description=frontmatter.get("description") or skill_info.get("summary", ""),
                source_type=PackSource.CLAWHUB,
                source_url=f"{self.client.base_url}/skills/{slug}",
                source_identifier=slug,
                tags=content_tags,
                requirements={
                    "env": requires.get("env", []),
                    "bins": requires.get("bins", []),
                    "any_bins": requires.get("anyBins", []),
                    "config": requires.get("config", []),
                    "os": openclaw.get("os", []),
                },
                properties={
                    "bundle_files": bundle_files,
                    "primary_env": openclaw.get("primaryEnv"),
                    "emoji": openclaw.get("emoji"),
                    "homepage": openclaw.get("homepage") or frontmatter.get("homepage"),
                    "always": openclaw.get("always", False),
                    "owner": skill_data.get("owner", {}).get("handle"),
                },
                installed_at=pack.installed_at,  # preserve original install time
                spaces=spaces,
            )

            return self.pack_manager.update(updated_pack, new_pieces)

        except Exception as e:
            logger.error("Failed to update skill %s: %s", slug, e)
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"Update failed: {e}",
            )
