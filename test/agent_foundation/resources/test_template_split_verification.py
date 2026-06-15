"""Verification tests for template structure and shared variable loading.

Ensures:
- All 9 wrapper templates exist in AgentFoundation's prompt_templates.
- Shared cross-template variables (_variables/notes/) load correctly.
- {{ notes.local_search_efficiency }} renders in templates that reference it.
"""

from pathlib import Path

import agent_foundation.resources as _af_res

AF_TEMPLATES_ROOT = Path(_af_res.__file__).parent / "prompt_templates"

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


class TestWrapperTemplatesInAgentFoundation:
    """All 9 wrapper templates must exist in AgentFoundation."""

    def test_all_wrapper_templates_exist(self) -> None:
        missing = []
        for rel in WRAPPER_TEMPLATES:
            path = AF_TEMPLATES_ROOT / rel
            if not path.is_file():
                missing.append(rel)
        assert not missing, f"Missing wrapper templates in AgentFoundation: {missing}"

    def test_wrapper_templates_are_non_empty(self) -> None:
        for rel in WRAPPER_TEMPLATES:
            path = AF_TEMPLATES_ROOT / rel
            assert path.stat().st_size > 0, f"Wrapper template is empty: {rel}"


class TestSharedVariableFile:
    """The shared _variables/notes/local_search_efficiency.jinja2 must exist
    and contain the expected NOTES content."""

    def test_file_exists(self) -> None:
        path = AF_TEMPLATES_ROOT / "_variables" / "notes" / "local_search_efficiency.jinja2"
        assert path.is_file(), f"Shared variable file missing: {path}"

    def test_file_has_content(self) -> None:
        path = AF_TEMPLATES_ROOT / "_variables" / "notes" / "local_search_efficiency.jinja2"
        content = path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, "Shared variable file is empty"
        assert "search" in content.lower() or "NEVER" in content, (
            "Shared variable file should contain search-scoping guidance"
        )


class TestSharedVariableLoading:
    """End-to-end: load_variables() with dot-key resolves the shared variable
    via cascade, and the rendered template includes its content."""

    def test_load_variables_finds_shared_file(self) -> None:
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        tm = TemplateManager(
            templates=str(AF_TEMPLATES_ROOT),
            active_template_type="main",
        )
        result = tm.load_variables(
            {"notes.local_search_efficiency": None},
            root_space="task_breakdown",
        )
        assert "notes" in result, "notes key missing from load_variables result"
        assert isinstance(result["notes"], dict), "notes should be a nested dict"
        assert "local_search_efficiency" in result["notes"], (
            "local_search_efficiency key missing from notes dict"
        )
        content = result["notes"]["local_search_efficiency"]
        assert len(content.strip()) > 0, "Loaded content should not be empty"

    def test_task_breakdown_template_renders_with_notes(self) -> None:
        """Render task_breakdown/main/initial.jinja2 with the shared notes variable
        and verify the NOTES content appears in the output."""
        from rich_python_utils.string_utils.formatting.jinja2_format import (
            format_template,
        )
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        tm = TemplateManager(
            templates=str(AF_TEMPLATES_ROOT),
            active_template_type="main",
            template_formatter=format_template,
        )
        feed = tm.load_variables(
            {"notes.local_search_efficiency": None},
            root_space="task_breakdown",
        )
        rendered = tm(
            "initial",
            active_template_root_space="task_breakdown",
            input="Test decomposition request",
            task_preamble="",
            task_instructions="",
            output_path="/tmp/test_output.md",
            max_breakdown=3,
            **feed,
        )
        notes_content = feed["notes"]["local_search_efficiency"]
        assert notes_content in rendered, (
            "Rendered task_breakdown template should contain the shared "
            "notes.local_search_efficiency content"
        )

    def test_implementation_template_renders_with_notes(self) -> None:
        """Render implementation/main/initial.jinja2 with the shared notes variable.
        Catches slash-vs-dot typos ({{ notes/foo }} vs {{ notes.foo }})."""
        from rich_python_utils.string_utils.formatting.jinja2_format import (
            format_template,
        )
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        tm = TemplateManager(
            templates=str(AF_TEMPLATES_ROOT),
            active_template_type="main",
            template_formatter=format_template,
        )
        feed = tm.load_variables(
            {"notes.local_search_efficiency": None},
            root_space="implementation",
        )
        rendered = tm(
            "initial",
            active_template_root_space="implementation",
            input="Test implementation request",
            task_preamble="",
            task_instructions="",
            output_path="/tmp/test_output.md",
            round_index=1,
            **feed,
        )
        notes_content = feed["notes"]["local_search_efficiency"]
        assert notes_content in rendered, (
            "Rendered implementation template should contain the shared "
            "notes.local_search_efficiency content"
        )

    def test_plan_task_instructions_variable_renders_with_notes(self) -> None:
        """Render plan/main/_variables/task_instructions/default.jinja2 directly
        with notes provided. Catches slash-vs-dot typos in _variables files."""
        import jinja2

        path = (
            AF_TEMPLATES_ROOT
            / "plan"
            / "main"
            / "_variables"
            / "task_instructions"
            / "default.jinja2"
        )
        content = path.read_text(encoding="utf-8")
        env = jinja2.Environment(undefined=jinja2.ChainableUndefined)
        tmpl = env.from_string(content)
        rendered = tmpl.render(
            notes={"local_search_efficiency": "SEARCH_EFFICIENCY_SENTINEL"},
        )
        assert "SEARCH_EFFICIENCY_SENTINEL" in rendered, (
            "plan/main/_variables/task_instructions/default.jinja2 should render "
            "notes.local_search_efficiency content (check for slash-vs-dot typo)"
        )

    def test_task_breakdown_template_renders_without_notes(self) -> None:
        """When notes is NOT configured, the template renders cleanly
        (ChainableUndefined produces empty string, no error)."""
        from rich_python_utils.string_utils.formatting.jinja2_format import (
            format_template,
        )
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        tm = TemplateManager(
            templates=str(AF_TEMPLATES_ROOT),
            active_template_type="main",
            template_formatter=format_template,
        )
        rendered = tm(
            "initial",
            active_template_root_space="task_breakdown",
            input="Test decomposition request",
            task_preamble="",
            task_instructions="",
            output_path="/tmp/test_output.md",
            max_breakdown=3,
        )
        assert "Now start your decomposition" in rendered
        assert "UndefinedError" not in rendered

    def test_task_breakdown_implementation_version_resolves_task_instructions(self) -> None:
        """With per-variable version 'implementation', task_instructions resolves
        to task_breakdown/main/_variables/task_instructions/implementation/default.jinja2."""
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        tm = TemplateManager(
            templates=str(AF_TEMPLATES_ROOT),
            active_template_type="main",
        )
        result = tm.load_variables(
            {"task_instructions": "implementation"},
            root_space="task_breakdown",
        )
        assert "task_instructions" in result, (
            "task_instructions should resolve with version='implementation'"
        )
        content = result["task_instructions"]
        assert "execution/implementation" in content, (
            "Implementation task_instructions should contain 'execution/implementation'"
        )
        assert "non-overlapping" in content.lower(), (
            "Implementation task_instructions should mention non-overlapping"
        )
        assert "Reference plan sections" in content, (
            "Implementation task_instructions should instruct to reference plan sections"
        )

    def test_task_breakdown_implementation_instructions_render_in_template(self) -> None:
        """Full render: task_breakdown/main/initial.jinja2 with
        implementation-versioned task_instructions includes the
        implementation-specific content in the rendered output."""
        from rich_python_utils.string_utils.formatting.jinja2_format import (
            format_template,
        )
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        tm = TemplateManager(
            templates=str(AF_TEMPLATES_ROOT),
            active_template_type="main",
            template_formatter=format_template,
        )
        feed = tm.load_variables(
            {"task_instructions": "implementation"},
            root_space="task_breakdown",
        )
        rendered = tm(
            "initial",
            active_template_root_space="task_breakdown",
            input="Implement the approved plan",
            task_preamble="",
            output_path="/tmp/test_output.md",
            max_breakdown=4,
            **feed,
        )
        assert "execution/implementation" in rendered, (
            "Rendered breakdown template with implementation task_instructions "
            "should contain the implementation-specific content"
        )
        assert "Now start your decomposition" in rendered, (
            "Generic breakdown structure should still render"
        )
