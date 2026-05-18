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

# Followup-specific strengthened language (added 2026-05-17 to address D-1
# from task-7ae9058e: worker_1 fix step regenerated from summary when
# `open_files` rejected the prior artifact path. The followup variable must
# explicitly call out the prior-artifact case and require shell fallback
# before any other alternative — preventing reconstruction-from-summary
# hallucinations.)
#
# These markers pin the meaningful strengthening vs the pre-2026-05-17
# one-liner ("{{ file_reading_fallback }} If BOTH fail, note the unreachable
# path in your Response."). If a future edit softens past this floor (e.g.,
# drops the prior-artifact specificity, drops the MUST ordering, or drops the
# shell-tools fallback callout), these tests will fail. They DO NOT pin
# specific wording beyond the strengthening floor — readers may further
# adjust phrasing as long as the floor markers remain.
FOLLOWUP_PRIOR_ARTIFACT_FALLBACK_MARKERS = (
    "Prior Artifacts/Files",   # explicit prior-artifact callout (the failure mode site)
    "outside workspace",        # the trigger phrase the agent will see
    "MUST first try",           # imperative + ordering: shell BEFORE any other recourse
    "command-line",             # the fallback tool category
)


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


class TestFollowupPriorArtifactReconstructionGuard:
    """Pin the strengthened followup-fallback language that prevents the
    worker_1-style failure mode observed in task-7ae9058e (2026-05-17 deep
    audit, defect D-1): the agent saw `open_files` reject the prior artifact
    path and reconstructed from the in-prompt summary, fabricating
    Spring Boot / Jackson / Kotlin-stdlib package labels in place of the real
    jq / Netty / OpenSSL / gRPC CVE data.

    These markers MUST appear in the followup variable and in any rendered
    followup template; if a future edit softens the guard, these tests will fail.
    """

    def test_followup_variable_has_reconstruction_guard_markers(self, template_manager):
        """The loaded followup variable contains all required guard markers."""
        result = template_manager.load_variables(
            {
                "instructions.behavior.file_reading_fallback": None,
                "instructions.behavior.file_reading_fallback_for_followup": None,
            },
            root_space="plan",
        )
        content = result["instructions"]["behavior"]["file_reading_fallback_for_followup"]
        missing = [m for m in FOLLOWUP_PRIOR_ARTIFACT_FALLBACK_MARKERS if m not in content]
        assert not missing, (
            f"Followup fallback variable is missing required guard markers {missing}. "
            f"This guard prevents reconstruction-from-summary hallucinations. "
            f"Loaded content: {content!r}"
        )

    def test_plan_followup_template_pins_reconstruction_guard(self, template_manager):
        """Rendered plan/main/followup.jinja2 carries the strengthened guard."""
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
            prior_output_path="/tmp/prior.md",
            task_instructions="",
            output_path="/tmp/test_output.md",
            **vars_feed,
        )
        missing = [m for m in FOLLOWUP_PRIOR_ARTIFACT_FALLBACK_MARKERS if m not in rendered]
        assert not missing, (
            f"Rendered plan followup template is missing guard markers {missing}. "
            f"First 1200 chars: {rendered[:1200]!r}"
        )

    def test_implementation_followup_template_pins_reconstruction_guard(self, template_manager):
        """Rendered implementation/main/followup.jinja2 carries the same guard."""
        vars_feed = template_manager.load_variables(
            {"instructions.behavior.file_reading_fallback_for_followup": None},
            root_space="implementation",
        )
        rendered = template_manager(
            "followup",
            active_template_root_space="implementation",
            active_template_type="main",
            input="Test user request",
            task_preamble="",
            main_response="Test prior implementation",
            reviewer_response="Test reviewer feedback",
            round_index=1,
            prior_output_path="/tmp/prior.md",
            task_instructions="",
            output_path="/tmp/test_output.md",
            **vars_feed,
        )
        missing = [m for m in FOLLOWUP_PRIOR_ARTIFACT_FALLBACK_MARKERS if m not in rendered]
        assert not missing, (
            f"Rendered implementation followup template is missing guard markers {missing}. "
            f"First 1200 chars: {rendered[:1200]!r}"
        )
