"""AIEmployeeRole — generic enterprise-aware role definition."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent_foundation.employees.models.enums import AutonomyLevel, RoleStatus
from agent_foundation.employees.models.skill import (
    CommunicationPolicy,
    GuardrailConfig,
    MindsetDirective,
    SOPDefinition,
    SkillConfig,
    ToolConfig,
)

logger = logging.getLogger(__name__)

_YAML_VERSION = "1.0"


@dataclass
class AIEmployeeRole:
    """Generic, enterprise-aware AI employee role definition.

    Produced by Phases 1+2 (create_role + role_setup).
    Independent of any specific team — describes WHAT the role does
    and HOW it does it across the organization.

    Stored as: _data/roles/{role_id}/role.yaml

    The role is analogous to a job description — reusable and enterprise-generic.
    Multiple AIEmployee instances can reference the same AIEmployeeRole.
    """

    # Identity
    id: str                   # e.g. "program_manager"
    name: str                 # e.g. "Program Manager"
    description: str          # brief description for UI display
    version: str = _YAML_VERSION

    # Artifacts (produced by /create-role Phase 1 and /role-setup Phase 2)
    role_document_path: Path | None = None   # path to role_document.md
    skills: list[SkillConfig] = field(default_factory=list)
    tools: ToolConfig = field(default_factory=ToolConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)

    # UI-aligned fields (maps to role_configs.json — consumed by RoleControlPopover.js)
    mindsets: list[MindsetDirective] = field(default_factory=list)
    sops: list[SOPDefinition] = field(default_factory=list)
    communication: CommunicationPolicy = field(default_factory=CommunicationPolicy)

    # Lifecycle
    status: RoleStatus = RoleStatus.draft
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""      # session_id or user that ran /create-role

    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # ---------------------------------------------------------------------------
    # Computed properties
    # ---------------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if role_setup is complete and role can be instantiated as an employee."""
        return (
            self.status == RoleStatus.active
            and self.role_document_path is not None
            and len(self.skills) > 0
        )

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, base_path: Path | None = None) -> "AIEmployeeRole":
        """Construct from a parsed YAML/JSON dict."""
        # Role document path
        rdp = data.get("role_document_path")
        role_doc = None
        if rdp:
            role_doc = Path(rdp)
            if base_path and not role_doc.is_absolute():
                role_doc = base_path / role_doc

        # Skills
        skills = [
            SkillConfig.from_dict(s, base_path=base_path)
            for s in data.get("skills", [])
        ]

        # Status
        try:
            status = RoleStatus(data.get("status", "draft"))
        except ValueError:
            status = RoleStatus.draft

        # Timestamps
        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            return datetime.now(timezone.utc)

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", _YAML_VERSION),
            role_document_path=role_doc,
            skills=skills,
            tools=ToolConfig.from_dict(data.get("tools", {})),
            guardrails=GuardrailConfig.from_dict(data.get("guardrails", {})),
            mindsets=[MindsetDirective.from_dict(m) for m in data.get("mindsets", [])],
            sops=[SOPDefinition.from_dict(s) for s in data.get("sops", [])],
            communication=CommunicationPolicy.from_dict(data.get("communication", {})),
            status=status,
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            created_by=data.get("created_by", ""),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        """Convert to a dict suitable for YAML serialization."""
        d: dict = {
            "version": self.version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
        }
        if self.role_document_path:
            d["role_document_path"] = str(self.role_document_path)
        d["skills"] = [s.to_dict() for s in self.skills]
        d["tools"] = self.tools.to_dict()
        d["mindsets"] = [m.to_dict() for m in self.mindsets]
        d["sops"] = [s.to_dict() for s in self.sops]
        d["communication"] = self.communication.to_dict()
        d["guardrails"] = self.guardrails.to_dict()
        if self.tags:
            d["tags"] = self.tags
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_yaml(cls, path: Path) -> "AIEmployeeRole":
        """Load from role.yaml file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"role.yaml not found: {path}")
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, base_path=path.parent)

    def to_yaml(self, path: Path) -> None:
        """Save to role.yaml file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info("[AIEmployeeRole] Saved role '%s' to %s", self.id, path)

    # ---------------------------------------------------------------------------
    # UI / Prompt bridge methods
    # ---------------------------------------------------------------------------

    def to_role_config(self) -> dict:
        """Convert to role_configs.json format for the dashboard UI.

        Enables AIEmployeeRole to replace fixture data in role_configs.json,
        making the UI serve real role data instead of mock fixtures.
        """
        return {
            "description": self.description,
            "mindsets": [m.to_dict() for m in self.mindsets],
            "sops": [s.to_dict() for s in self.sops],
            "communication": self.communication.to_dict(),
            "guardrails": self.guardrails.to_role_config_guardrails(),
        }

    def to_employee_variable(self) -> dict:
        """Convert to .variables.yaml employee format for prompt injection.

        Enables AIEmployeeRole to populate the employee.mindset template variable,
        replacing the static .variables.yaml mindset dict.

        Returns a dict compatible with the `employee` key in .variables.yaml:
            employee:
              name: ...
              role: ...
              mindset:
                directive_text: directive_text
        """
        return {
            "name": self.name,
            "role": self.description,
            "mindset": {
                m.text: m.text
                for m in self.mindsets
                if m.default_enabled
            },
        }
