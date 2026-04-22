"""Tests for the AI Employee framework — models, registry, and state manager."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from agent_foundation.employees import (
    AIEmployee,
    AIEmployeeRole,
    AutonomyLevel,
    CommunicationPolicy,
    EmployeeRegistry,
    EmployeeRuntimeState,
    EmployeeStatus,
    GuardrailConfig,
    MindsetDirective,
    RoleRegistry,
    RoleStatus,
    SkillConfig,
    SOPDefinition,
    StateManager,
    TeamContext,
    ToolConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_role(**overrides) -> AIEmployeeRole:
    defaults = dict(
        id="program_manager",
        name="Program Manager",
        description="Coordinates cross-team projects.",
        status=RoleStatus.active,
        skills=[
            SkillConfig(
                id="project_tracking",
                display_name="Project Tracking",
                skill_md_path=Path("skills/project_tracking/SKILL.md"),
            )
        ],
        tools=ToolConfig(enabled=["jira_search", "slack_send_message"], disabled=[]),
        guardrails=GuardrailConfig(
            max_autonomy_level=AutonomyLevel.medium,
            max_concurrent_tasks=3,
            escalation_threshold="2h",
            approval_required=["architecture_decisions"],
            can_auto_approve=["status_updates"],
            requires_confirmation=["jira_create_issue"],
        ),
        mindsets=[
            MindsetDirective(text="Escalate early", default_enabled=True),
            MindsetDirective(text="Prioritize shipping", default_enabled=False),
        ],
        sops=[
            SOPDefinition(
                title="Sprint Planning",
                trigger="At start of sprint",
                steps=["Review backlog", "Estimate capacity"],
            )
        ],
        communication=CommunicationPolicy(
            allow_all=True,
            allowed_roles=["Software Engineer"],
            blocked_roles=[],
        ),
    )
    defaults.update(overrides)
    return AIEmployeeRole(**defaults)


def _make_team_context(**overrides) -> TeamContext:
    defaults = dict(
        team_id="team-platform",
        team_name="Platform Team",
        domain="Platform Engineering",
        jira_projects=["CTSC", "PLAT"],
        slack_channels=["#platform-team"],
        confluence_spaces=["Platform"],
        focus_areas=["Developer experience"],
        additional_tools=["twg_query"],
        disabled_tools=[],
    )
    defaults.update(overrides)
    return TeamContext(**defaults)


def _make_employee(role: AIEmployeeRole, **overrides) -> AIEmployee:
    emp = AIEmployee(
        id=overrides.pop("id", "alice_pm"),
        persona_name=overrides.pop("persona_name", "Alice"),
        display_name=overrides.pop("display_name", "Alice — Platform PM"),
        role_id=role.id,
        **overrides,
    )
    emp.resolve_role(role)
    emp.team_context = _make_team_context()
    return emp


# ===========================================================================
# Phase A: Model Tests
# ===========================================================================

class TestEnums:
    def test_role_status_values(self):
        assert RoleStatus.draft == "draft"
        assert RoleStatus.active == "active"
        assert RoleStatus.deprecated == "deprecated"

    def test_employee_status_values(self):
        assert EmployeeStatus.onboarding == "onboarding"
        assert EmployeeStatus.active == "active"
        assert EmployeeStatus.retired == "retired"

    def test_autonomy_level_values(self):
        assert AutonomyLevel.high == "high"
        assert AutonomyLevel.medium == "medium"
        assert AutonomyLevel.low == "low"


class TestSkillConfig:
    def test_from_dict_basic(self):
        data = {
            "id": "project_tracking",
            "display_name": "Project Tracking",
            "skill_md_path": "skills/project_tracking/SKILL.md",
        }
        skill = SkillConfig.from_dict(data)
        assert skill.id == "project_tracking"
        assert skill.display_name == "Project Tracking"
        assert skill.skill_md_path == Path("skills/project_tracking/SKILL.md")

    def test_from_dict_with_base_path(self, tmp_path):
        data = {
            "id": "s1",
            "display_name": "S1",
            "skill_md_path": "skills/s1/SKILL.md",
        }
        skill = SkillConfig.from_dict(data, base_path=tmp_path)
        assert skill.skill_md_path == tmp_path / "skills/s1/SKILL.md"

    def test_to_dict_roundtrip(self):
        skill = SkillConfig(
            id="s1", display_name="S1",
            skill_md_path=Path("skills/s1/SKILL.md"),
            description="A skill",
        )
        d = skill.to_dict()
        skill2 = SkillConfig.from_dict(d)
        assert skill2.id == skill.id
        assert skill2.description == skill.description


class TestToolConfig:
    def test_effective_tools(self):
        tc = ToolConfig(enabled=["a", "b", "c"], disabled=["b"])
        assert tc.effective_tools == ["a", "c"]

    def test_roundtrip(self):
        tc = ToolConfig(enabled=["jira", "slack"], disabled=["slack"])
        tc2 = ToolConfig.from_dict(tc.to_dict())
        assert tc2.effective_tools == ["jira"]


class TestMindsetDirective:
    def test_from_dict(self):
        m = MindsetDirective.from_dict({"text": "Ship fast", "default_enabled": False})
        assert m.text == "Ship fast"
        assert m.default_enabled is False

    def test_to_dict(self):
        m = MindsetDirective(text="Be thorough", default_enabled=True)
        d = m.to_dict()
        assert d == {"text": "Be thorough", "default_enabled": True}


class TestSOPDefinition:
    def test_from_dict(self):
        data = {
            "title": "Sprint Planning",
            "trigger": "Start of sprint",
            "steps": ["Step 1", "Step 2"],
            "file": "sops/sprint.jinja2",
        }
        sop = SOPDefinition.from_dict(data)
        assert sop.title == "Sprint Planning"
        assert len(sop.steps) == 2
        assert sop.file == "sops/sprint.jinja2"

    def test_to_dict_omits_empty(self):
        sop = SOPDefinition(title="T")
        d = sop.to_dict()
        assert "trigger" not in d
        assert "steps" not in d


class TestGuardrailConfig:
    def test_from_dict_role_configs_format(self):
        """Parses role_configs.json guardrails format (max_autonomy not max_autonomy_level)."""
        data = {
            "max_autonomy": "high",
            "max_concurrent_tasks": 5,
            "escalation_threshold": "1h",
            "approval_required": ["deployments"],
        }
        g = GuardrailConfig.from_dict(data)
        assert g.max_autonomy_level == AutonomyLevel.high
        assert g.max_concurrent_tasks == 5

    def test_to_role_config_guardrails(self):
        g = GuardrailConfig(
            max_autonomy_level=AutonomyLevel.medium,
            max_concurrent_tasks=3,
            approval_required=["prod_deploy"],
        )
        rc = g.to_role_config_guardrails()
        assert rc["max_autonomy"] == "medium"
        assert rc["max_concurrent_tasks"] == 3
        assert "approval_required" in rc

    def test_invalid_autonomy_defaults_to_medium(self):
        g = GuardrailConfig.from_dict({"max_autonomy": "ultra"})
        assert g.max_autonomy_level == AutonomyLevel.medium


class TestAIEmployeeRole:
    def test_is_ready_active_with_skills(self):
        role = _make_role()
        role.role_document_path = Path("role_document.md")
        assert role.is_ready is True

    def test_is_ready_draft(self):
        role = _make_role(status=RoleStatus.draft)
        assert role.is_ready is False

    def test_is_ready_no_skills(self):
        role = _make_role(skills=[])
        assert role.is_ready is False

    def test_to_role_config(self):
        role = _make_role()
        rc = role.to_role_config()
        assert "description" in rc
        assert len(rc["mindsets"]) == 2
        assert rc["mindsets"][0]["text"] == "Escalate early"
        assert len(rc["sops"]) == 1
        assert rc["guardrails"]["max_autonomy"] == "medium"

    def test_to_employee_variable_only_enabled_mindsets(self):
        role = _make_role()
        ev = role.to_employee_variable()
        assert ev["name"] == "Program Manager"
        # Only enabled mindsets (default_enabled=True) → "Escalate early" only
        assert "Escalate early" in ev["mindset"]
        assert "Prioritize shipping" not in ev["mindset"]

    def test_yaml_roundtrip(self, tmp_path):
        role = _make_role()
        role.role_document_path = Path("role_document.md")
        yaml_path = tmp_path / "role.yaml"
        role.to_yaml(yaml_path)
        role2 = AIEmployeeRole.from_yaml(yaml_path)
        assert role2.id == role.id
        assert role2.name == role.name
        assert role2.status == role.status
        assert len(role2.skills) == 1
        assert len(role2.mindsets) == 2
        assert len(role2.sops) == 1
        assert role2.guardrails.max_autonomy_level == AutonomyLevel.medium
        assert role2.guardrails.max_concurrent_tasks == 3

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AIEmployeeRole.from_yaml(Path("/nonexistent/role.yaml"))


class TestTeamContext:
    def test_from_dict(self):
        data = {
            "team_id": "team-1",
            "team_name": "Team One",
            "domain": "Engineering",
            "jira_projects": ["PROJ"],
            "slack_channels": ["#eng"],
            "confluence_spaces": ["Engineering"],
            "focus_areas": ["Backend"],
            "additional_tools": ["twg_query"],
            "disabled_tools": [],
            "autonomy_overrides": {},
        }
        tc = TeamContext.from_dict(data)
        assert tc.team_id == "team-1"
        assert tc.jira_projects == ["PROJ"]

    def test_yaml_roundtrip(self, tmp_path):
        tc = _make_team_context()
        path = tmp_path / "team_context.yaml"
        tc.to_yaml(path)
        tc2 = TeamContext.from_yaml(path)
        assert tc2.team_id == tc.team_id
        assert tc2.jira_projects == tc.jira_projects

    def test_to_prompt_context(self):
        tc = _make_team_context()
        ctx = tc.to_prompt_context()
        assert ctx["team_name"] == "Platform Team"
        assert "CTSC" in ctx["jira_projects"]


class TestAIEmployee:
    def test_effective_tools_merges_role_and_team(self):
        role = _make_role()
        emp = _make_employee(role)
        # Role: jira_search, slack_send_message; Team adds: twg_query
        tools = emp.effective_tools
        assert "jira_search" in tools
        assert "slack_send_message" in tools
        assert "twg_query" in tools

    def test_effective_tools_respects_disabled(self):
        role = _make_role()
        emp = _make_employee(role)
        emp.team_context.disabled_tools = ["slack_send_message"]
        tools = emp.effective_tools
        assert "slack_send_message" not in tools

    def test_resolve_role_wrong_id_raises(self):
        role = _make_role()
        emp = AIEmployee(id="e1", persona_name="E", display_name="E", role_id="wrong_role")
        with pytest.raises(ValueError, match="Role id mismatch"):
            emp.resolve_role(role)

    def test_all_skills(self):
        role = _make_role()
        emp = _make_employee(role)
        assert len(emp.all_skills) == 1
        assert emp.all_skills[0].id == "project_tracking"

    def test_yaml_roundtrip(self, tmp_path):
        role = _make_role()
        emp = _make_employee(role)
        emp_path = tmp_path / "employee.yaml"
        emp.to_yaml(emp_path)
        emp2 = AIEmployee.from_yaml(emp_path)
        assert emp2.id == emp.id
        assert emp2.persona_name == emp.persona_name
        assert emp2.role_id == emp.role_id

    def test_to_prompt_prior_context(self):
        role = _make_role()
        emp = _make_employee(role)
        ctx = emp.to_prompt_prior_context()
        assert ctx["employee_name"] == "Alice"
        assert ctx["role_name"] == "Program Manager"
        assert ctx["team_name"] == "Platform Team"
        assert "CTSC" in ctx["jira_projects"]

    def test_to_dashboard_employee(self):
        role = _make_role()
        emp = _make_employee(role)
        d = emp.to_dashboard_employee()
        assert d["type"] == "ai"
        assert d["role"] == "Program Manager"
        assert "project_tracking" in d["specializations"]


# ===========================================================================
# Phase B: Registry Tests
# ===========================================================================

class TestRoleRegistry:
    def test_empty_dir(self, tmp_path):
        registry = RoleRegistry(tmp_path / "roles")
        assert len(registry) == 0

    def test_scans_and_loads_roles(self, tmp_path):
        role_dir = tmp_path / "roles"
        role = _make_role()
        role_dir.mkdir()
        role.to_yaml(role_dir / "program_manager" / "role.yaml")

        registry = RoleRegistry(role_dir)
        assert len(registry) == 1
        assert "program_manager" in registry

    def test_get_returns_role(self, tmp_path):
        role_dir = tmp_path / "roles"
        role = _make_role()
        role_dir.mkdir()
        role.to_yaml(role_dir / "program_manager" / "role.yaml")

        registry = RoleRegistry(role_dir)
        found = registry.get("program_manager")
        assert found is not None
        assert found.name == "Program Manager"

    def test_get_nonexistent_returns_none(self, tmp_path):
        registry = RoleRegistry(tmp_path)
        assert registry.get("nonexistent") is None

    def test_get_or_raise_raises(self, tmp_path):
        registry = RoleRegistry(tmp_path)
        with pytest.raises(KeyError):
            registry.get_or_raise("nonexistent")

    def test_register_and_save(self, tmp_path):
        role_dir = tmp_path / "roles"
        role_dir.mkdir()
        registry = RoleRegistry(role_dir)

        role = _make_role()
        registry.register(role, save=True)

        assert registry.get("program_manager") is not None
        assert (role_dir / "program_manager" / "role.yaml").exists()

    def test_list_active_filters_by_status(self, tmp_path):
        role_dir = tmp_path / "roles"
        role_dir.mkdir()
        registry = RoleRegistry(role_dir)

        active_role = _make_role(id="r1", name="R1", status=RoleStatus.active)
        draft_role = _make_role(id="r2", name="R2", status=RoleStatus.draft)
        registry.register(active_role, save=False)
        registry.register(draft_role, save=False)

        active = registry.list_active()
        assert len(active) == 1
        assert active[0].id == "r1"

    def test_find_by_tag(self, tmp_path):
        role_dir = tmp_path / "roles"
        role_dir.mkdir()
        registry = RoleRegistry(role_dir)

        role = _make_role(tags=["delivery", "pm"])
        registry.register(role, save=False)

        found = registry.find_by_tag("delivery")
        assert len(found) == 1

        not_found = registry.find_by_tag("engineering")
        assert len(not_found) == 0

    def test_find_by_tool(self, tmp_path):
        role_dir = tmp_path / "roles"
        role_dir.mkdir()
        registry = RoleRegistry(role_dir)

        role = _make_role()
        registry.register(role, save=False)

        found = registry.find_by_tool("jira_search")
        assert len(found) == 1

        not_found = registry.find_by_tool("nonexistent_tool")
        assert len(not_found) == 0


class TestEmployeeRegistry:
    def _setup(self, tmp_path):
        role_dir = tmp_path / "roles"
        emp_dir = tmp_path / "employees"
        role_dir.mkdir()
        emp_dir.mkdir()

        role = _make_role()
        role.to_yaml(role_dir / "program_manager" / "role.yaml")

        role_registry = RoleRegistry(role_dir)
        emp_registry = EmployeeRegistry(emp_dir, role_registry)
        return role, role_registry, emp_registry, emp_dir

    def test_empty_dir(self, tmp_path):
        role_registry = RoleRegistry(tmp_path / "roles")
        emp_registry = EmployeeRegistry(tmp_path / "employees", role_registry)
        assert len(emp_registry) == 0

    def test_register_and_retrieve(self, tmp_path):
        role, _, emp_registry, _ = self._setup(tmp_path)
        emp = _make_employee(role)
        emp_registry.register(emp, save=True)

        found = emp_registry.get("alice_pm")
        assert found is not None
        assert found.persona_name == "Alice"
        assert found.role is not None
        assert found.role.id == "program_manager"

    def test_list_by_team(self, tmp_path):
        role, _, emp_registry, _ = self._setup(tmp_path)
        emp = _make_employee(role)
        emp_registry.register(emp, save=False)

        found = emp_registry.list_by_team("team-platform")
        assert len(found) == 1

        not_found = emp_registry.list_by_team("team-other")
        assert len(not_found) == 0

    def test_list_by_role(self, tmp_path):
        role, _, emp_registry, _ = self._setup(tmp_path)
        emp = _make_employee(role)
        emp_registry.register(emp, save=False)

        found = emp_registry.list_by_role("program_manager")
        assert len(found) == 1

    def test_loads_from_disk_on_scan(self, tmp_path):
        role, role_registry, _, emp_dir = self._setup(tmp_path)
        emp = _make_employee(role)
        emp_dir2 = emp_dir / "alice_pm"
        emp_dir2.mkdir()
        emp.to_yaml(emp_dir2 / "employee.yaml")
        emp.team_context.to_yaml(emp_dir2 / "team_context.yaml")

        # New registry instance — scans from disk
        emp_registry2 = EmployeeRegistry(emp_dir, role_registry)
        found = emp_registry2.get("alice_pm")
        assert found is not None
        assert found.team_context is not None
        assert found.team_context.team_id == "team-platform"


# ===========================================================================
# Phase D: StateManager Tests
# ===========================================================================

class TestStateManager:
    def test_load_returns_default_if_missing(self, tmp_path):
        sm = StateManager(tmp_path)
        state = sm.load("alice_pm")
        assert state.employee_id == "alice_pm"
        assert state.status == EmployeeStatus.active

    def test_save_and_load_roundtrip(self, tmp_path):
        sm = StateManager(tmp_path)
        state = EmployeeRuntimeState(
            employee_id="alice_pm",
            status=EmployeeStatus.idle,
            metrics={"issues_resolved": 42},
        )
        sm.save(state)

        loaded = sm.load("alice_pm")
        assert loaded.status == EmployeeStatus.idle
        assert loaded.metrics["issues_resolved"] == 42

    def test_update_status(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.update_status("alice_pm", EmployeeStatus.blocked,
                         pending_reason={"reason": "awaiting_decision"})
        state = sm.load("alice_pm")
        assert state.status == EmployeeStatus.blocked
        assert state.pending_reason["reason"] == "awaiting_decision"

    def test_record_metric(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.record_metric("alice_pm", "ai_human_chats", 10)
        state = sm.load("alice_pm")
        assert state.metrics["ai_human_chats"] == 10

    def test_increment_metric(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.increment_metric("alice_pm", "issues_resolved", 1)
        sm.increment_metric("alice_pm", "issues_resolved", 1)
        state = sm.load("alice_pm")
        assert state.metrics["issues_resolved"] == 2

    def test_set_current_task(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.set_current_task("alice_pm", "task-042")
        state = sm.load("alice_pm")
        assert state.current_task_id == "task-042"

    def test_handles_corrupt_json_gracefully(self, tmp_path):
        emp_dir = tmp_path / "alice_pm"
        emp_dir.mkdir()
        (emp_dir / "state.json").write_text("not valid json")
        sm = StateManager(tmp_path)
        state = sm.load("alice_pm")
        assert state.employee_id == "alice_pm"  # returns default


# ===========================================================================
# Integration: role_configs.json compatibility
# ===========================================================================

class TestRoleConfigCompatibility:
    """Verify that AIEmployeeRole.to_role_config() produces output compatible
    with the existing role_configs.json fixture format consumed by the UI."""

    def test_to_role_config_has_required_keys(self):
        role = _make_role()
        rc = role.to_role_config()
        # These are the keys the UI expects (from RoleControlPopover.js)
        assert "description" in rc
        assert "mindsets" in rc
        assert "sops" in rc
        assert "communication" in rc
        assert "guardrails" in rc

    def test_guardrails_has_ui_keys(self):
        role = _make_role()
        rc = role.to_role_config()
        g = rc["guardrails"]
        assert "max_autonomy" in g       # UI expects max_autonomy not max_autonomy_level
        assert "max_concurrent_tasks" in g
        assert "approval_required" in g

    def test_mindsets_format(self):
        role = _make_role()
        rc = role.to_role_config()
        for m in rc["mindsets"]:
            assert "text" in m
            assert "default_enabled" in m

    def test_sops_format(self):
        role = _make_role()
        rc = role.to_role_config()
        for s in rc["sops"]:
            assert "title" in s
