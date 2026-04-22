"""Skill, tool, mindset, SOP, communication and guardrail config models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_foundation.employees.models.enums import AutonomyLevel


# ---------------------------------------------------------------------------
# Skill & Tool
# ---------------------------------------------------------------------------

@dataclass
class SkillConfig:
    """A single skill produced by /role-setup (Phase 2).

    Each skill has a SKILL.md file (injected into LLM prompts) and an
    optional skill.yaml metadata file.
    """
    id: str                           # e.g. "project_tracking"
    display_name: str                 # e.g. "Project Tracking"
    skill_md_path: Path               # path to SKILL.md (relative to role dir)
    metadata_path: Path | None = None # path to skill.yaml (optional)
    description: str = ""             # short description (from skill.yaml or SKILL.md header)

    @classmethod
    def from_dict(cls, data: dict, base_path: Path | None = None) -> "SkillConfig":
        skill_md = Path(data["skill_md_path"])
        if base_path and not skill_md.is_absolute():
            skill_md = base_path / skill_md
        metadata = None
        if data.get("metadata_path"):
            metadata = Path(data["metadata_path"])
            if base_path and not metadata.is_absolute():
                metadata = base_path / metadata
        return cls(
            id=data["id"],
            display_name=data.get("display_name", data["id"]),
            skill_md_path=skill_md,
            metadata_path=metadata,
            description=data.get("description", ""),
        )

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "display_name": self.display_name,
            "skill_md_path": str(self.skill_md_path),
        }
        if self.metadata_path:
            d["metadata_path"] = str(self.metadata_path)
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class ToolConfig:
    """Tool access configuration for a role or employee."""
    enabled: list[str] = field(default_factory=list)   # tool names enabled for this role
    disabled: list[str] = field(default_factory=list)  # explicit overrides (for employee)

    @property
    def effective_tools(self) -> list[str]:
        """Tools available after applying disabled overrides."""
        return [t for t in self.enabled if t not in self.disabled]

    @classmethod
    def from_dict(cls, data: dict) -> "ToolConfig":
        return cls(
            enabled=data.get("enabled", []),
            disabled=data.get("disabled", []),
        )

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "disabled": self.disabled}


# ---------------------------------------------------------------------------
# UI-aligned models (maps to role_configs.json consumed by RoleControlPopover.js)
# ---------------------------------------------------------------------------

@dataclass
class MindsetDirective:
    """Toggleable directive shaping how the agent approaches work.

    Maps directly to role_configs.json mindsets[] format.
    Runtime-toggled via RoleControlPopover.js.
    """
    text: str
    default_enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "MindsetDirective":
        return cls(text=data["text"], default_enabled=data.get("default_enabled", True))

    def to_dict(self) -> dict:
        return {"text": self.text, "default_enabled": self.default_enabled}


@dataclass
class SOPDefinition:
    """Standard Operating Procedure with inline steps.

    Maps directly to role_configs.json sops[] format.
    Optional file ref for detailed Jinja2 SOP template.
    """
    title: str
    trigger: str = ""
    steps: list[str] = field(default_factory=list)
    file: str = ""  # optional: relative path to detailed Jinja2 SOP template

    @classmethod
    def from_dict(cls, data: dict) -> "SOPDefinition":
        return cls(
            title=data["title"],
            trigger=data.get("trigger", ""),
            steps=data.get("steps", []),
            file=data.get("file", ""),
        )

    def to_dict(self) -> dict:
        d: dict = {"title": self.title}
        if self.trigger:
            d["trigger"] = self.trigger
        if self.steps:
            d["steps"] = self.steps
        if self.file:
            d["file"] = self.file
        return d


@dataclass
class CommunicationPolicy:
    """Inter-employee communication permissions.

    Maps directly to role_configs.json communication format.
    """
    allow_all: bool = True
    allowed_roles: list[str] = field(default_factory=list)
    blocked_roles: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "CommunicationPolicy":
        return cls(
            allow_all=data.get("allow_all", True),
            allowed_roles=data.get("allowed_roles", []),
            blocked_roles=data.get("blocked_roles", []),
        )

    def to_dict(self) -> dict:
        return {
            "allow_all": self.allow_all,
            "allowed_roles": self.allowed_roles,
            "blocked_roles": self.blocked_roles,
        }


@dataclass
class GuardrailConfig:
    """Operational guardrails defining autonomy boundaries.

    Merged from role_configs.json guardrails + agent_states.json autonomy sections.
    Also supports SOP-driven tool confirmation gates and free-text rules.
    """
    # From role_configs.json guardrails
    max_autonomy_level: AutonomyLevel = AutonomyLevel.medium
    max_concurrent_tasks: int = 3
    escalation_threshold: str = "2h"        # e.g. "2h", "30m"
    max_token_budget: str = "100K"          # cost control
    output_review: str = "on_errors"        # always | on_errors | never
    approval_required: list[str] = field(default_factory=list)   # always needs human approval

    # From agent_states.json autonomy
    can_auto_approve: list[str] = field(default_factory=list)    # no human approval needed
    max_hours_without_checkin: int = 8

    # SOP-driven tool confirmation gates
    requires_confirmation: list[str] = field(default_factory=list)

    # Free-text guardrail statements (injected into system prompt)
    custom_rules: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "GuardrailConfig":
        autonomy_raw = data.get("max_autonomy_level", data.get("max_autonomy", "medium"))
        try:
            autonomy = AutonomyLevel(autonomy_raw)
        except ValueError:
            autonomy = AutonomyLevel.medium
        return cls(
            max_autonomy_level=autonomy,
            max_concurrent_tasks=int(data.get("max_concurrent_tasks", 3)),
            escalation_threshold=data.get("escalation_threshold", "2h"),
            max_token_budget=data.get("max_token_budget", "100K"),
            output_review=data.get("output_review", "on_errors"),
            approval_required=data.get("approval_required", []),
            can_auto_approve=data.get("can_auto_approve", []),
            max_hours_without_checkin=int(data.get("max_hours_without_checkin", 8)),
            requires_confirmation=data.get("requires_confirmation", []),
            custom_rules=data.get("custom_rules", []),
        )

    def to_dict(self) -> dict:
        return {
            "max_autonomy_level": self.max_autonomy_level.value,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "escalation_threshold": self.escalation_threshold,
            "max_token_budget": self.max_token_budget,
            "output_review": self.output_review,
            "approval_required": self.approval_required,
            "can_auto_approve": self.can_auto_approve,
            "max_hours_without_checkin": self.max_hours_without_checkin,
            "requires_confirmation": self.requires_confirmation,
            "custom_rules": self.custom_rules,
        }

    def to_role_config_guardrails(self) -> dict:
        """Convert to role_configs.json guardrails format for UI compatibility."""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_autonomy": self.max_autonomy_level.value,
            "escalation_threshold": self.escalation_threshold,
            "max_token_budget": self.max_token_budget,
            "output_review": self.output_review,
            "approval_required": self.approval_required,
        }
