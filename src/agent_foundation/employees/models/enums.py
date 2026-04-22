"""Enumerations for the AI Employee framework."""

from __future__ import annotations

from enum import StrEnum


class RoleStatus(StrEnum):
    """Lifecycle status of an AIEmployeeRole."""
    draft = "draft"           # role document created, role_setup not yet run
    active = "active"         # role_setup complete, ready to instantiate employees
    deprecated = "deprecated" # superseded by a newer version


class EmployeeStatus(StrEnum):
    """Operational status of an AIEmployee instance."""
    onboarding = "onboarding" # team-onboard in progress
    active = "active"         # fully operational
    idle = "idle"             # no current task
    blocked = "blocked"       # awaiting human decision
    away = "away"             # temporarily inactive
    retired = "retired"       # permanently deactivated


class AutonomyLevel(StrEnum):
    """Autonomy level for guardrail configuration."""
    high = "high"     # can act and decide independently
    medium = "medium" # needs human approval for key actions
    low = "low"       # assists only; humans make all decisions
