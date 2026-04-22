"""AIEmployee — concrete AI employee instance deployed to a specific team."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from agent_foundation.employees.models.enums import EmployeeStatus
from agent_foundation.employees.models.team_context import TeamContext

if TYPE_CHECKING:
    from agent_foundation.employees.models.role import AIEmployeeRole
    from agent_foundation.employees.models.skill import SkillConfig

logger = logging.getLogger(__name__)

_YAML_VERSION = "1.0"


@dataclass
class AIEmployee:
    """Concrete AI employee instance — a role deployed to a specific team.

    Produced by Phase 3 (/team-onboard).
    Holds a reference to AIEmployeeRole (composition, not inheritance).
    This allows employees to change roles and multiple employees to share
    the same role definition.

    Config stored as: _data/employees/{employee_id}/employee.yaml
    Team context: _data/employees/{employee_id}/team_context.yaml
    Runtime state: _data/employees/{employee_id}/state.json (gitignored)
    """

    # Identity
    id: str                   # e.g. "alice_pm"
    persona_name: str         # e.g. "Alice"
    display_name: str         # e.g. "Alice — Platform PM"
    version: str = _YAML_VERSION

    # Role (composition — references AIEmployeeRole by id)
    role_id: str = ""         # references AIEmployeeRole.id
    role: "AIEmployeeRole | None" = field(default=None, repr=False)  # loaded at runtime

    # Team deployment (Phase 3)
    team_context: TeamContext | None = None

    # Persona / communication style
    greeting: str = ""               # e.g. "Hi, I'm Alice..."
    communication_style: str = "professional"

    # Lifecycle
    status: EmployeeStatus = EmployeeStatus.onboarding
    hired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retired_at: datetime | None = None

    # ---------------------------------------------------------------------------
    # Computed properties
    # ---------------------------------------------------------------------------

    @property
    def effective_tools(self) -> list[str]:
        """Merge role tools with team-specific overrides.

        Priority: role enabled tools + team additional_tools - team disabled_tools
        """
        if not self.role:
            return []
        base = set(self.role.tools.effective_tools)
        if self.team_context:
            base.update(self.team_context.additional_tools)
            base.difference_update(self.team_context.disabled_tools)
        return sorted(base)

    @property
    def all_skills(self) -> "list[SkillConfig]":
        """All skills from the role (team-specific additions added in Phase 3 future work)."""
        return self.role.skills if self.role else []

    @property
    def is_active(self) -> bool:
        return self.status == EmployeeStatus.active

    @property
    def team_name(self) -> str:
        return self.team_context.team_name if self.team_context else ""

    @property
    def team_id(self) -> str:
        return self.team_context.team_id if self.team_context else ""

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "AIEmployee":
        """Construct from a parsed YAML dict.

        Note: role is NOT loaded here — call resolve_role(role_registry) after loading.
        """
        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            return datetime.now(timezone.utc)

        def _parse_dt_opt(val: Any) -> datetime | None:
            if val is None:
                return None
            return _parse_dt(val)

        try:
            status = EmployeeStatus(data.get("status", "onboarding"))
        except ValueError:
            status = EmployeeStatus.onboarding

        return cls(
            id=data["id"],
            persona_name=data.get("persona_name", data.get("name", data["id"])),
            display_name=data.get("display_name", data.get("persona_name", data["id"])),
            version=data.get("version", _YAML_VERSION),
            role_id=data.get("role_id", data.get("role", "")),
            role=None,  # resolved separately via resolve_role()
            team_context=None,  # loaded separately via load_team_context()
            greeting=data.get("greeting", ""),
            communication_style=data.get("communication_style", "professional"),
            status=status,
            hired_at=_parse_dt(data.get("hired_at")),
            retired_at=_parse_dt_opt(data.get("retired_at")),
        )

    def to_dict(self) -> dict:
        """Convert to a dict suitable for YAML serialization.

        Does NOT include runtime state (role object, team_context object) —
        those are stored in separate files.
        """
        d: dict = {
            "version": self.version,
            "id": self.id,
            "persona_name": self.persona_name,
            "display_name": self.display_name,
            "role_id": self.role_id,
            "greeting": self.greeting,
            "communication_style": self.communication_style,
            "status": self.status.value,
            "hired_at": self.hired_at.isoformat(),
        }
        if self.retired_at:
            d["retired_at"] = self.retired_at.isoformat()
        return d

    @classmethod
    def from_yaml(cls, path: Path) -> "AIEmployee":
        """Load employee.yaml.

        Note: role and team_context are NOT loaded — call resolve_role() and
        load_team_context() after loading.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"employee.yaml not found: {path}")
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: Path) -> None:
        """Save employee.yaml (without runtime state or team_context)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info("[AIEmployee] Saved employee '%s' to %s", self.id, path)

    def resolve_role(self, role: "AIEmployeeRole") -> None:
        """Attach the resolved AIEmployeeRole object."""
        if role.id != self.role_id:
            raise ValueError(
                f"Role id mismatch: employee expects '{self.role_id}', got '{role.id}'"
            )
        self.role = role

    def load_team_context(self, path: Path) -> None:
        """Load and attach team_context.yaml from the employee directory."""
        if path.exists():
            self.team_context = TeamContext.from_yaml(path)

    # ---------------------------------------------------------------------------
    # Prompt / UI bridge
    # ---------------------------------------------------------------------------

    def to_prompt_prior_context(self) -> dict:
        """Build prior_context dict for ConversationalInferencer.set_prior_context().

        Merges role context + team context into a flat dict for prompt injection.
        """
        ctx: dict = {
            "employee_id": self.id,
            "employee_name": self.persona_name,
            "display_name": self.display_name,
            "greeting": self.greeting,
            "communication_style": self.communication_style,
        }
        if self.role:
            ctx["role_name"] = self.role.name
            ctx["role_description"] = self.role.description
            # Merge role employee variable — exclude "name" to avoid ambiguity
            # with "employee_name" already set above
            ev = self.role.to_employee_variable()
            ctx["role"] = ev.get("role", "")
            ctx["mindset"] = ev.get("mindset", {})
        if self.team_context:
            ctx.update(self.team_context.to_prompt_context())
        return ctx

    def to_dashboard_employee(self) -> dict:
        """Convert to the Employee fixture format for the dashboard API.

        Compatible with OpenStartup's Employee(BaseModel) shape.
        """
        return {
            "id": self.id,
            "name": self.persona_name,
            "type": "ai",
            "role": self.role.name if self.role else self.role_id,
            "status": self.status.value,
            "avatar_url": "",
            "team_ids": [self.team_id] if self.team_id else [],
            "specializations": [s.id for s in self.all_skills],
            "metrics": {},
        }
