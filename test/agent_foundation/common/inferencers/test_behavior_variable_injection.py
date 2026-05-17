"""Test that instructions.behavior.* variables render correctly in prompt templates.

Specifically tests the nested variable pattern where
file_reading_fallback_for_review.jinja2 references {{ file_reading_fallback }}
internally — verifying that the final rendered prompt contains the full
expanded instruction, not literal {{ file_reading_fallback }} text.
"""

import os
from pathlib import Path

import pytest

# Template root: AgentFoundation/src/agent_foundation/resources/prompt_templates/
_AF_ROOT = Path(__file__).resolve().parents[4] / "src" / "agent_foundation"
TEMPLATES_DIR = _AF_ROOT / "resources" / "prompt_templates"


@pytest.fixture
def template_manager():
    """Create a TemplateManager pointed at the real prompt_templates directory."""
    from rich_python_utils.string_utils.formatting.template_manager import (
        TemplateManager,
    )
    from rich_python_utils.string_utils.formatting.jinja2_format import (
        format_template,
    )

    tm = TemplateManager(
        templates=str(TEMPLATES_DIR),
        template_formatter=format_template,
        active_template_root_space="plan",
        active_template_type="main",
        predefined_variables=True,
    )
    return tm


# The base instruction text that should appear in all rendered prompts
BASE_FALLBACK_TEXT = "fall back to command-line tools"
REVIEW_SUFFIX = "verification_gap"
FOLLOWUP_SUFFIX = "unreachable path"


class TestBehaviorVariableResolution:
    """Test that instructions.behavior.* variables resolve from _variables/."""

    def test_base_fallback_variable_loads(self, template_manager):
        """instructions.behavior.file_reading_fallback resolves to file content."""
        result = template_manager.load_variables(
            {"instructions.behavior.file_reading_fallback": None},
            root_space="plan",
        )
        content = result.get("instructions", {}).get("behavior", {}).get("file_reading_fallback", "")
        assert BASE_FALLBACK_TEXT in content, (
            f"Base fallback text not found in resolved variable. Got: {content!r}"
        )

    def test_review_fallback_variable_loads(self, template_manager):
        """instructions.behavior.file_reading_fallback_for_review resolves."""
        result = template_manager.load_variables(
            {"instructions.behavior.file_reading_fallback_for_review": None},
            root_space="plan",
        )
        content = result.get("instructions", {}).get("behavior", {}).get("file_reading_fallback_for_review", "")
        assert REVIEW_SUFFIX in content, (
            f"Review suffix not found. Got: {content!r}"
        )

    def test_followup_fallback_variable_loads(self, template_manager):
        """instructions.behavior.file_reading_fallback_for_followup resolves."""
        result = template_manager.load_variables(
            {"instructions.behavior.file_reading_fallback_for_followup": None},
            root_space="plan",
        )
        content = result.get("instructions", {}).get("behavior", {}).get("file_reading_fallback_for_followup", "")
        assert FOLLOWUP_SUFFIX in content, (
            f"Followup suffix not found. Got: {content!r}"
        )


class TestNestedVariableExpansion:
    """Test that {{ file_reading_fallback }} inside _for_review/_for_followup
    variable files expands to the base instruction content."""

    def test_review_variable_expands_base_reference(self, template_manager):
        """file_reading_fallback_for_review.jinja2 contains {{ file_reading_fallback }}.
        After rendering, the final text should contain the base fallback text,
        NOT the literal '{{ file_reading_fallback }}' string."""
        result = template_manager.load_variables(
            {
                "instructions.behavior.file_reading_fallback": None,
                "instructions.behavior.file_reading_fallback_for_review": None,
            },
            root_space="plan",
        )
        review_content = result["instructions"]["behavior"]["file_reading_fallback_for_review"]

        # If nested expansion works: contains the actual base text
        # If it doesn't work: contains literal "{{ file_reading_fallback }}"
        assert "{{ file_reading_fallback }}" not in review_content, (
            "Nested variable {{ file_reading_fallback }} was NOT expanded — "
            "it appears literally in the rendered content. "
            "The TemplateManager may not render variable file content through Jinja2. "
            f"Got: {review_content!r}"
        )
        assert BASE_FALLBACK_TEXT in review_content, (
            f"Base fallback text missing from expanded review variable. Got: {review_content!r}"
        )

    def test_followup_variable_expands_base_reference(self, template_manager):
        """Same test for the followup variant."""
        result = template_manager.load_variables(
            {
                "instructions.behavior.file_reading_fallback": None,
                "instructions.behavior.file_reading_fallback_for_followup": None,
            },
            root_space="plan",
        )
        followup_content = result["instructions"]["behavior"]["file_reading_fallback_for_followup"]

        assert "{{ file_reading_fallback }}" not in followup_content, (
            "Nested variable {{ file_reading_fallback }} was NOT expanded. "
            f"Got: {followup_content!r}"
        )
        assert BASE_FALLBACK_TEXT in followup_content, (
            f"Base fallback text missing from expanded followup variable. Got: {followup_content!r}"
        )


class TestFullTemplateRendering:
    """End-to-end: render actual review.jinja2 and verify the instruction appears."""

    def test_plan_review_template_contains_fallback_instruction(self, template_manager):
        """Render plan/main/review.jinja2 and verify the file-reading
        fallback instruction is present in the output.

        Uses load_variables to pre-resolve the behavior variable (same
        path as production _build_template_feed), then passes it in
        the feed dict for rendering.
        """
        # Pre-load the behavior variable (production path)
        vars_feed = template_manager.load_variables(
            {"instructions.behavior.file_reading_fallback_for_review": None},
            root_space="plan",
        )
        rendered = template_manager(
            "review",
            active_template_root_space="plan",
            active_template_type="main",
            input="Test user request",
            task_preamble="",
            main_response="Test artifact content",
            reviewer_response="",
            round_index=1,
            prior_output_path="",
            counter_feedback="",
            task_instructions="",
            **vars_feed,
        )
        assert BASE_FALLBACK_TEXT in rendered, (
            f"File-reading fallback instruction not found in rendered review template. "
            f"Length: {len(rendered)}, first 500 chars: {rendered[:500]!r}"
        )
        assert REVIEW_SUFFIX in rendered, (
            f"Review-specific suffix (verification_gap) not found in rendered template."
        )

    def test_plan_followup_template_contains_fallback_instruction(self, template_manager):
        """Render plan/main/followup.jinja2 and verify the file-reading
        fallback instruction (followup variant) is present."""
        vars_feed = template_manager.load_variables(
            {"instructions.behavior.file_reading_fallback_for_followup": None},
            root_space="plan",
        )
        rendered = template_manager(
            "followup",
            active_template_root_space="plan",
            active_template_type="main",
            input="Test user request",
            task_preamble="",
            main_response="Test prior artifact",
            reviewer_response="Test reviewer feedback",
            round_index=1,
            prior_output_path="",
            task_instructions="",
            output_path="/tmp/test_output.md",
            **vars_feed,
        )
        assert BASE_FALLBACK_TEXT in rendered, (
            f"File-reading fallback not found in followup template."
        )
        assert FOLLOWUP_SUFFIX in rendered, (
            f"Followup-specific suffix (unreachable path) not found."
        )

    def test_plan_initial_template_contains_fallback_instruction(self, template_manager):
        """Render plan/main/initial.jinja2 and verify the file-reading
        fallback instruction (base variant) is present."""
        vars_feed = template_manager.load_variables(
            {"instructions.behavior.file_reading_fallback": None},
            root_space="plan",
        )
        rendered = template_manager(
            "initial",
            active_template_root_space="plan",
            active_template_type="main",
            input="Test user request",
            task_preamble="",
            output_path="/tmp/test_output.md",
            **vars_feed,
        )
        assert BASE_FALLBACK_TEXT in rendered, (
            f"File-reading fallback not found in initial template."
        )
