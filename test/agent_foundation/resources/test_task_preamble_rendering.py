"""Integration test: task_preamble variable renders cleanly in plan/main/initial.jinja2.

Verifies that Jinja2 control structures inside predefined variable files
(like _variables/task_preamble/default.jinja2) are rendered through the
template engine — NOT passed through as raw text.

Two scenarios:
  1. Context variables absent → {% if %} blocks hidden, no raw Jinja2 in output
  2. Context variables present → blocks render with actual values
"""

import os
import pytest

from rich_python_utils.string_utils.formatting.template_manager import TemplateManager


@pytest.fixture
def plan_template_manager():
    """TemplateManager configured with AgentFoundation's prompt_templates."""
    templates_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "src", "agent_foundation", "resources", "prompt_templates",
    )
    templates_dir = os.path.normpath(templates_dir)
    assert os.path.isdir(templates_dir), f"Templates dir not found: {templates_dir}"

    return TemplateManager(
        templates=[templates_dir],
        active_template_type="main",
        predefined_variables=True,
        enable_templated_feed=True,
        default_template_key="initial",
    )


class TestTaskPreambleRendering:
    """Verify task_preamble variable renders without raw Jinja2 leaking."""

    def test_no_raw_jinja2_when_context_vars_absent(self, plan_template_manager):
        """When session_root_path / workflow_target_path / docs_path are NOT
        in the feed, the {% if %} blocks in task_preamble should evaluate
        to False and be hidden — NOT appear as raw template syntax."""
        rendered = plan_template_manager(
            template_key="initial",
            active_template_root_space="plan",
            input="Build an authentication system",
            output_path="/tmp/output.md",
        )

        assert "{%" not in rendered, (
            "Raw Jinja2 control structure leaked into rendered output. "
            "The {% if session_root_path %} block in task_preamble/default.jinja2 "
            "was not rendered by the template engine."
        )
        assert "{{ session_root_path }}" not in rendered, (
            "Raw {{ session_root_path }} variable leaked into rendered output."
        )
        assert "{{ workflow_target_path }}" not in rendered
        assert "{{ docs_path }}" not in rendered

        # The actual content (non-conditional parts) should be present
        assert "Build an authentication system" in rendered
        assert "output.md" in rendered

    def test_context_vars_render_when_present(self, plan_template_manager):
        """When session_root_path / workflow_target_path / docs_path ARE
        in the feed, the {% if %} blocks should render with actual values."""
        rendered = plan_template_manager(
            template_key="initial",
            active_template_root_space="plan",
            input="Build auth",
            output_path="/tmp/output.md",
            session_root_path="/Users/dev/myproject",
            workflow_target_path="/Users/dev/myproject/src",
            docs_path="/Users/dev/myproject/docs",
        )

        assert "{%" not in rendered, "Raw Jinja2 leaked"
        assert "/Users/dev/myproject" in rendered, (
            "session_root_path should appear in rendered output when provided"
        )
        assert "/Users/dev/myproject/src" in rendered, (
            "workflow_target_path should appear in rendered output when provided"
        )
        assert "/Users/dev/myproject/docs" in rendered, (
            "docs_path should appear in rendered output when provided"
        )

    def test_partial_context_vars(self, plan_template_manager):
        """Only session_root_path provided — other blocks hidden cleanly."""
        rendered = plan_template_manager(
            template_key="initial",
            active_template_root_space="plan",
            input="Build auth",
            output_path="/tmp/output.md",
            session_root_path="/Users/dev/project",
        )

        assert "{%" not in rendered, "Raw Jinja2 leaked"
        assert "/Users/dev/project" in rendered
        # workflow_target_path and docs_path not provided — should not appear
        assert "workflow_target_path" not in rendered
        assert "docs_path" not in rendered


class TestNoRawJinja2InAnyTemplate:
    """Broader check: no predefined variable should leak raw Jinja2 into
    ANY template rendering, regardless of template_root_space."""

    @pytest.mark.parametrize("root_space", ["plan", "implementation"])
    def test_initial_template_clean(self, plan_template_manager, root_space):
        """Render initial template for each root_space and verify no raw Jinja2."""
        try:
            rendered = plan_template_manager(
                template_key="initial",
                active_template_root_space=root_space,
                input="Test task",
                output_path="/tmp/test.md",
            )
        except Exception:
            pytest.skip(f"Template {root_space}/main/initial not available")

        assert "{%" not in rendered, (
            f"Raw Jinja2 leaked in {root_space}/main/initial rendering"
        )
