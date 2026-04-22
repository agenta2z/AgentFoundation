"""EmployeeRegistry — scan and index AIEmployee instances from disk."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from agent_foundation.employees.models.employee import AIEmployee
from agent_foundation.employees.models.enums import EmployeeStatus

logger = logging.getLogger(__name__)


class EmployeeRegistry:
    """Scans an employees directory and provides lookup by id, team, role, and status.

    Lazy-loads employee.yaml files on first access.
    Automatically resolves role references via RoleRegistry.
    Thread-safe for concurrent reads.

    Directory structure expected:
        employees_dir/
          {employee_id}/
            employee.yaml
            team_context.yaml   (optional)
            state.json          (optional, runtime state)
    """

    def __init__(self, employees_dir: Path, role_registry: "RoleRegistry") -> None:
        from agent_foundation.employees.registry.role_registry import RoleRegistry
        self._employees_dir = Path(employees_dir)
        self._role_registry = role_registry
        self._employees: dict[str, AIEmployee] = {}
        self._lock = threading.RLock()
        self._scanned = False

    def _ensure_scanned(self) -> None:
        with self._lock:
            if not self._scanned:
                self._scan()
                self._scanned = True

    def _scan(self) -> None:
        """Scan employees_dir for employee.yaml files and load them."""
        if not self._employees_dir.exists():
            logger.debug("[EmployeeRegistry] employees_dir does not exist: %s", self._employees_dir)
            return
        for child in sorted(self._employees_dir.iterdir()):
            employee_yaml = child / "employee.yaml"
            if child.is_dir() and employee_yaml.exists():
                try:
                    employee = self._load_employee(child)
                    self._employees[employee.id] = employee
                    logger.debug(
                        "[EmployeeRegistry] Loaded employee '%s' (role=%s)",
                        employee.id, employee.role_id,
                    )
                except Exception as e:
                    logger.warning(
                        "[EmployeeRegistry] Failed to load %s: %s", employee_yaml, e
                    )
        logger.info(
            "[EmployeeRegistry] Loaded %d employees from %s",
            len(self._employees), self._employees_dir,
        )

    def _load_employee(self, employee_dir: Path) -> AIEmployee:
        """Load a single employee from its directory."""
        employee = AIEmployee.from_yaml(employee_dir / "employee.yaml")

        # Resolve role reference
        if employee.role_id:
            role = self._role_registry.get(employee.role_id)
            if role:
                employee.resolve_role(role)
            else:
                logger.warning(
                    "[EmployeeRegistry] Role '%s' not found for employee '%s'",
                    employee.role_id, employee.id,
                )

        # Load team context if present
        team_context_path = employee_dir / "team_context.yaml"
        employee.load_team_context(team_context_path)

        return employee

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def get(self, employee_id: str) -> AIEmployee | None:
        """Get an employee by id. Returns None if not found."""
        self._ensure_scanned()
        return self._employees.get(employee_id)

    def get_or_raise(self, employee_id: str) -> AIEmployee:
        """Get an employee by id. Raises KeyError if not found."""
        emp = self.get(employee_id)
        if emp is None:
            raise KeyError(
                f"Employee not found: '{employee_id}'. "
                f"Available: {list(self._employees.keys())}"
            )
        return emp

    def list_all(self, status: EmployeeStatus | None = None) -> list[AIEmployee]:
        """List all employees, optionally filtered by status."""
        self._ensure_scanned()
        employees = list(self._employees.values())
        if status is not None:
            employees = [e for e in employees if e.status == status]
        return employees

    def list_active(self) -> list[AIEmployee]:
        """List all active employees."""
        return self.list_all(status=EmployeeStatus.active)

    def list_by_team(self, team_id: str) -> list[AIEmployee]:
        """List all employees deployed to a given team."""
        self._ensure_scanned()
        return [
            e for e in self._employees.values()
            if e.team_id == team_id
        ]

    def list_by_role(self, role_id: str) -> list[AIEmployee]:
        """List all employees of a given role."""
        self._ensure_scanned()
        return [e for e in self._employees.values() if e.role_id == role_id]

    def register(self, employee: AIEmployee, save: bool = True) -> None:
        """Register an employee (called after /team-onboard completes).

        If save=True, writes employee.yaml and team_context.yaml to the employees directory.
        """
        with self._lock:
            self._employees[employee.id] = employee
            if save:
                emp_dir = self._employees_dir / employee.id
                emp_dir.mkdir(parents=True, exist_ok=True)
                employee.to_yaml(emp_dir / "employee.yaml")
                if employee.team_context:
                    employee.team_context.to_yaml(emp_dir / "team_context.yaml")
        logger.info(
            "[EmployeeRegistry] Registered employee '%s' (role=%s, team=%s)",
            employee.id, employee.role_id, employee.team_id,
        )

    def invalidate(self) -> None:
        """Force re-scan on next access."""
        with self._lock:
            self._scanned = False
            self._employees.clear()

    def __len__(self) -> int:
        self._ensure_scanned()
        return len(self._employees)

    def __contains__(self, employee_id: str) -> bool:
        self._ensure_scanned()
        return employee_id in self._employees
