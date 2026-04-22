"""RoleRegistry — scan and index AIEmployeeRole definitions from disk."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from agent_foundation.employees.models.enums import RoleStatus
from agent_foundation.employees.models.role import AIEmployeeRole

logger = logging.getLogger(__name__)


class RoleRegistry:
    """Scans a roles directory and provides lookup by id, name, status, and tag.

    Lazy-loads role.yaml files on first access. Thread-safe for concurrent reads.

    Directory structure expected:
        roles_dir/
          {role_id}/
            role.yaml
            role_document.md
            skills/
              ...
    """

    def __init__(
        self,
        roles_dir: Path,
        extra_dirs: list[Path] | None = None,
    ) -> None:
        self._roles_dir = Path(roles_dir)
        self._extra_dirs: list[Path] = [Path(d) for d in (extra_dirs or [])]
        self._roles: dict[str, AIEmployeeRole] = {}
        self._lock = threading.RLock()
        self._scanned = False

    def _ensure_scanned(self) -> None:
        with self._lock:
            if not self._scanned:
                self._scan()
                self._scanned = True

    def _scan(self) -> None:
        """Scan roles_dir (and extra_dirs) for role.yaml files and load them.

        extra_dirs roles override framework roles with the same id (same as load_all_tools pattern).
        """
        dirs_to_scan = [self._roles_dir] + self._extra_dirs
        for scan_dir in dirs_to_scan:
            if not scan_dir.exists():
                logger.debug("[RoleRegistry] scan dir does not exist: %s", scan_dir)
                continue
            for child in sorted(scan_dir.iterdir()):
                role_yaml = child / "role.yaml"
                if child.is_dir() and role_yaml.exists():
                    try:
                        role = AIEmployeeRole.from_yaml(role_yaml)
                        self._roles[role.id] = role
                        logger.debug("[RoleRegistry] Loaded role '%s' from %s", role.id, role_yaml)
                    except Exception as e:
                        logger.warning("[RoleRegistry] Failed to load %s: %s", role_yaml, e)
        logger.info("[RoleRegistry] Loaded %d roles from %s (+%d extra dirs)",
                    len(self._roles), self._roles_dir, len(self._extra_dirs))

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def get(self, role_id: str) -> AIEmployeeRole | None:
        """Get a role by id. Returns None if not found."""
        self._ensure_scanned()
        return self._roles.get(role_id)

    def get_or_raise(self, role_id: str) -> AIEmployeeRole:
        """Get a role by id. Raises KeyError if not found."""
        role = self.get(role_id)
        if role is None:
            raise KeyError(f"Role not found: '{role_id}'. Available: {list(self._roles.keys())}")
        return role

    def list_all(self, status: RoleStatus | None = None) -> list[AIEmployeeRole]:
        """List all roles, optionally filtered by status."""
        self._ensure_scanned()
        roles = list(self._roles.values())
        if status is not None:
            roles = [r for r in roles if r.status == status]
        return roles

    def list_active(self) -> list[AIEmployeeRole]:
        """List roles ready to instantiate as employees."""
        return self.list_all(status=RoleStatus.active)

    def find_by_tag(self, tag: str) -> list[AIEmployeeRole]:
        """Find roles that have a given tag."""
        self._ensure_scanned()
        return [r for r in self._roles.values() if tag in r.tags]

    def find_by_tool(self, tool_name: str) -> list[AIEmployeeRole]:
        """Find roles that have a given tool enabled."""
        self._ensure_scanned()
        return [r for r in self._roles.values() if tool_name in r.tools.effective_tools]

    def register(self, role: AIEmployeeRole, save: bool = True) -> None:
        """Register a role (called after /create-role or /role-setup completes).

        If save=True, writes role.yaml to the roles directory.
        """
        with self._lock:
            self._roles[role.id] = role
            if save:
                role_dir = self._roles_dir / role.id
                role_dir.mkdir(parents=True, exist_ok=True)
                role.to_yaml(role_dir / "role.yaml")
        logger.info("[RoleRegistry] Registered role '%s' (status=%s)", role.id, role.status)

    def update(self, role: AIEmployeeRole, save: bool = True) -> None:
        """Update an existing role (e.g. after role_setup upgrades status to active)."""
        self.register(role, save=save)

    def invalidate(self) -> None:
        """Force re-scan on next access (e.g. after external file changes)."""
        with self._lock:
            self._scanned = False
            self._roles.clear()

    def __len__(self) -> int:
        self._ensure_scanned()
        return len(self._roles)

    def __contains__(self, role_id: str) -> bool:
        self._ensure_scanned()
        return role_id in self._roles
