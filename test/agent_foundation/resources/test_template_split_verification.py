"""Verification tests for the template file split (Task 6).

Ensures:
- All 9 wrapper templates exist in AgentFoundation's prompt_templates.
- All _variables/ directories remain in OpenStartup's prompt_templates.
- No _variables/ directories exist in AgentFoundation's active prompt_templates
  (excluding _archived/).
"""

from pathlib import Path

import agent_foundation.resources as _af_res

AF_TEMPLATES_ROOT = Path(_af_res.__file__).parent / "prompt_templates"

# Resolve OpenStartup's prompt_templates relative to the workspace root.
# Walk up from AgentFoundation's resources to the workspace root, then into OpenStartup.
_WORKSPACE_ROOT = AF_TEMPLATES_ROOT.parents[4]  # prompt_templates → resources → agent_foundation → src → AgentFoundation → workspace
OS_TEMPLATES_ROOT = (
    _WORKSPACE_ROOT
    / "OpenStartup"
    / "src"
    / "openteam"
    / "server"
    / "resources"
    / "prompt_templates"
)

# The 9 wrapper templates that should live in AgentFoundation
WRAPPER_TEMPLATES = [
    "plan/main/initial.jinja2",
    "plan/main/review.jinja2",
    "plan/main/followup.jinja2",
    "implementation/main/initial.jinja2",
    "implementation/main/review.jinja2",
    "implementation/main/followup.jinja2",
    "task_breakdown/main/initial.jinja2",
    "deep_research/main/initial.jinja2",
    "conversation/main/initial.jinja2",
]

# _variables/ directories that must remain in OpenStartup
OPENSTARTUP_VARIABLE_DIRS = [
    "_variables/task_preamble",
    "plan/main/_variables",
    "implementation/main/_variables",
    "task_breakdown/main/_variables",
    "deep_research/main/_variables",
    "conversation/main/_variables",
]


class TestWrapperTemplatesInAgentFoundation:
    """All 9 wrapper templates must exist in AgentFoundation."""

    def test_all_wrapper_templates_exist(self) -> None:
        missing = []
        for rel in WRAPPER_TEMPLATES:
            path = AF_TEMPLATES_ROOT / rel
            if not path.is_file():
                missing.append(rel)
        assert not missing, f"Missing wrapper templates in AgentFoundation: {missing}"

    def test_wrapper_template_count(self) -> None:
        """Exactly 9 wrapper templates should be present."""
        count = sum(1 for rel in WRAPPER_TEMPLATES if (AF_TEMPLATES_ROOT / rel).is_file())
        assert count == 9, f"Expected 9 wrapper templates, found {count}"

    def test_wrapper_templates_are_non_empty(self) -> None:
        for rel in WRAPPER_TEMPLATES:
            path = AF_TEMPLATES_ROOT / rel
            assert path.stat().st_size > 0, f"Wrapper template is empty: {rel}"


class TestVariableDirsInOpenStartup:
    """All _variables/ directories must remain in OpenStartup."""

    def test_all_variable_dirs_exist(self) -> None:
        missing = []
        for rel in OPENSTARTUP_VARIABLE_DIRS:
            path = OS_TEMPLATES_ROOT / rel
            if not path.is_dir():
                missing.append(rel)
        assert not missing, f"Missing _variables/ dirs in OpenStartup: {missing}"

    def test_variables_yaml_exists(self) -> None:
        """The .variables.yaml (AI HR persona) must remain in OpenStartup."""
        assert (OS_TEMPLATES_ROOT / ".variables.yaml").is_file()

    def test_global_task_preamble_exists(self) -> None:
        """Global _variables/task_preamble/default.jinja2 must remain."""
        path = OS_TEMPLATES_ROOT / "_variables" / "task_preamble" / "default.jinja2"
        assert path.is_file()


class TestNoVariableDirsInAgentFoundation:
    """AgentFoundation's active prompt_templates must have NO _variables/ dirs.

    This is critical for the variable isolation fallback: when a root has no
    _variables/, TemplateManager creates no VariableLoader for it, and variable
    resolution falls back to self._variable_loader (OpenStartup's loader).
    """

    def test_no_variables_in_active_template_dirs(self) -> None:
        """No _variables/ in active (non-archived) template directories."""
        active_categories = [
            "plan", "implementation", "task_breakdown",
            "deep_research", "conversation",
        ]
        found = []
        for category in active_categories:
            cat_dir = AF_TEMPLATES_ROOT / category
            if cat_dir.is_dir():
                for var_dir in cat_dir.rglob("_variables"):
                    if var_dir.is_dir():
                        found.append(str(var_dir.relative_to(AF_TEMPLATES_ROOT)))
        assert not found, (
            f"Found _variables/ dirs in AgentFoundation active templates: {found}"
        )

    def test_no_top_level_variables_dir(self) -> None:
        """No top-level _variables/ in AgentFoundation's prompt_templates."""
        assert not (AF_TEMPLATES_ROOT / "_variables").is_dir()
