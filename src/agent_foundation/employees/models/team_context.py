"""TeamContext — team-specific deployment context for an AIEmployee."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


_YAML_VERSION = "1.0"


@dataclass
class TeamContext:
    """Team-specific context produced by Phase 3 (/team-onboard).

    Contains the concrete integration points, workflows, and focus areas
    for a specific team. Stored as:
        _data/employees/{employee_id}/team_context.yaml

    Answers the questions deferred from Phase 0:
    - What domain does the team operate in?
    - What Jira projects, Slack channels, Confluence spaces do they use?
    - What are their specific focus areas and workflows?
    """

    team_id: str
    team_name: str
    domain: str = ""            # e.g. "Platform Engineering"

    # Concrete integration points (discovered via Phase 3)
    jira_projects: list[str] = field(default_factory=list)      # e.g. ["CTSC", "PLAT"]
    slack_channels: list[str] = field(default_factory=list)     # e.g. ["#platform-team"]
    confluence_spaces: list[str] = field(default_factory=list)  # e.g. ["Platform"]

    # Team-specific specializations of the generic role
    focus_areas: list[str] = field(default_factory=list)

    # Tool overrides (extend or restrict the role's tool config)
    additional_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)

    # Autonomy tuning for this specific team (override role defaults)
    # dict of {tool_name: autonomy_level_override} or {setting: value}
    autonomy_overrides: dict[str, str] = field(default_factory=dict)

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = _YAML_VERSION

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "TeamContext":
        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            return datetime.now(timezone.utc)

        return cls(
            team_id=data["team_id"],
            team_name=data["team_name"],
            domain=data.get("domain", ""),
            jira_projects=data.get("jira_projects", []),
            slack_channels=data.get("slack_channels", []),
            confluence_spaces=data.get("confluence_spaces", []),
            focus_areas=data.get("focus_areas", []),
            additional_tools=data.get("additional_tools", []),
            disabled_tools=data.get("disabled_tools", []),
            autonomy_overrides=data.get("autonomy_overrides", {}),
            created_at=_parse_dt(data.get("created_at")),
            version=data.get("version", _YAML_VERSION),
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "domain": self.domain,
            "jira_projects": self.jira_projects,
            "slack_channels": self.slack_channels,
            "confluence_spaces": self.confluence_spaces,
            "focus_areas": self.focus_areas,
            "additional_tools": self.additional_tools,
            "disabled_tools": self.disabled_tools,
            "autonomy_overrides": self.autonomy_overrides,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "TeamContext":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"team_context.yaml not found: {path}")
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: Path) -> None:
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

    def to_prompt_context(self) -> dict:
        """Return a dict suitable for injection into prior_context / prompt variables."""
        return {
            "team_name": self.team_name,
            "team_domain": self.domain,
            "jira_projects": self.jira_projects,
            "slack_channels": self.slack_channels,
            "confluence_spaces": self.confluence_spaces,
            "focus_areas": self.focus_areas,
        }
