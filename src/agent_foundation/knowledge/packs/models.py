"""
Knowledge Pack data models.

Provides data models for managing bundled knowledge sets (packs). A pack is a
named, versioned collection of KnowledgePieces managed as an atomic unit.
Typical use: importing ClawHub skills or loading local knowledge bundles.

The models follow the project's @attrs pattern for core data classes and
@dataclass for configuration/result types.
"""

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional

from attr import attrs, attrib


class PackStatus(StrEnum):
    """Lifecycle status of a knowledge pack."""

    INSTALLED = "installed"
    UPDATING = "updating"
    UNINSTALLING = "uninstalling"
    FAILED = "failed"


class PackSource(StrEnum):
    """Source type for a knowledge pack."""

    CLAWHUB = "clawhub"
    LOCAL = "local"
    MANUAL = "manual"


@attrs
class KnowledgePack:
    """A named, versioned collection of KnowledgePieces managed as an atomic unit.

    Attributes:
        pack_id: Unique identifier (e.g., "pack:clawhub:todoist-cli").
        name: Human-readable name.
        version: Semver version string.
        description: Short description of the pack.
        source_type: Where this pack came from (clawhub, local, manual).
        source_url: URL of the source registry or file.
        source_identifier: Original identifier in source system (e.g., slug).
        piece_ids: List of KnowledgePiece IDs belonging to this pack.
        status: Current lifecycle status.
        tags: Tags for categorization.
        requirements: Runtime requirements (env vars, bins, OS constraints).
        properties: Arbitrary key-value properties for extensibility.
        installed_at: ISO 8601 timestamp of installation.
        updated_at: ISO 8601 timestamp of last update.
    """

    pack_id: str = attrib()
    name: str = attrib()
    version: str = attrib(default="0.0.0")
    description: str = attrib(default="")
    source_type: PackSource = attrib(default=PackSource.MANUAL)
    source_url: Optional[str] = attrib(default=None)
    source_identifier: Optional[str] = attrib(default=None)
    piece_ids: List[str] = attrib(factory=list)
    status: PackStatus = attrib(default=PackStatus.INSTALLED)
    tags: List[str] = attrib(factory=list)
    requirements: Dict[str, Any] = attrib(factory=dict)
    properties: Dict[str, Any] = attrib(factory=dict)
    installed_at: str = attrib(default=None)
    updated_at: str = attrib(default=None)
    spaces: Optional[List[str]] = attrib(default=None)

    def __attrs_post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if self.installed_at is None:
            self.installed_at = now
        if self.updated_at is None:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pack_id": self.pack_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "source_identifier": self.source_identifier,
            "piece_ids": list(self.piece_ids),
            "status": self.status.value,
            "tags": list(self.tags),
            "requirements": dict(self.requirements),
            "properties": dict(self.properties),
            "installed_at": self.installed_at,
            "updated_at": self.updated_at,
            "spaces": list(self.spaces) if self.spaces else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgePack":
        """Deserialize from dictionary.

        Args:
            data: Dictionary containing KnowledgePack fields.

        Returns:
            A new KnowledgePack instance.

        Raises:
            ValueError: If required fields (pack_id, name) are missing.
        """
        if "pack_id" not in data:
            raise ValueError("Missing required field: 'pack_id'")
        if "name" not in data:
            raise ValueError("Missing required field: 'name'")

        source_type = data.get("source_type", PackSource.MANUAL)
        if isinstance(source_type, str):
            source_type = PackSource(source_type)

        status = data.get("status", PackStatus.INSTALLED)
        if isinstance(status, str):
            status = PackStatus(status)

        return cls(
            pack_id=data["pack_id"],
            name=data["name"],
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            source_type=source_type,
            source_url=data.get("source_url"),
            source_identifier=data.get("source_identifier"),
            piece_ids=data.get("piece_ids", []),
            status=status,
            tags=data.get("tags", []),
            requirements=data.get("requirements", {}),
            properties=data.get("properties", {}),
            installed_at=data.get("installed_at"),
            updated_at=data.get("updated_at"),
            spaces=data.get("spaces"),
        )


@dataclass
class PackInstallResult:
    """Result of a pack install/uninstall/update operation."""

    success: bool
    pack_id: str
    pieces_installed: int = 0
    pieces_removed: int = 0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PackManagerConfig:
    """Configuration for KnowledgePackManager."""

    default_domain: str = "agent_skills"
    default_info_type: str = "skills"
    preserve_history: bool = True
