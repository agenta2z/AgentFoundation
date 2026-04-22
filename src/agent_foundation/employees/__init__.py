"""AI Employee framework — AgentFoundation module.

Provides:
- Models: AIEmployeeRole, AIEmployee, TeamContext, and supporting dataclasses
- Registry: RoleRegistry, EmployeeRegistry
- Runtime: StateManager, EmployeeRuntimeState

Quick start::

    from agent_foundation.employees import (
        AIEmployeeRole, AIEmployee, RoleRegistry, EmployeeRegistry, StateManager
    )

    role_registry = RoleRegistry(Path("_data/roles"))
    employee_registry = EmployeeRegistry(Path("_data/employees"), role_registry)

    role = role_registry.get("program_manager")
    employees = employee_registry.list_by_team("team-platform")
"""

from agent_foundation.employees.models import (
    AIEmployee,
    AIEmployeeRole,
    AutonomyLevel,
    CommunicationPolicy,
    EmployeeStatus,
    GuardrailConfig,
    MindsetDirective,
    RoleStatus,
    SkillConfig,
    SOPDefinition,
    TeamContext,
    ToolConfig,
)
from agent_foundation.employees.registry import EmployeeRegistry, RoleRegistry
from agent_foundation.employees.runtime.state_manager import (
    EmployeeRuntimeState,
    StateManager,
)

__all__ = [
    # Models
    "AIEmployee",
    "AIEmployeeRole",
    "AutonomyLevel",
    "CommunicationPolicy",
    "EmployeeStatus",
    "GuardrailConfig",
    "MindsetDirective",
    "RoleStatus",
    "SkillConfig",
    "SOPDefinition",
    "TeamContext",
    "ToolConfig",
    # Registry
    "EmployeeRegistry",
    "RoleRegistry",
    # Runtime
    "EmployeeRuntimeState",
    "StateManager",
]
