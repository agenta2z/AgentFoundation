"""Unit tests for `_variables/context/` shared variables.

Verifies that the newly-introduced shared variables under
``_variables/context/`` (and the templates that consume them) work
correctly across rendering scenarios.

Coverage
--------

1. **Source-of-truth files exist and are non-empty**:
   - ``_variables/context/user_request_with_task_preamble.jinja2``
   - ``_variables/context/session/local_paths.jinja2``

2. **Shared-variable rendering**:
   - ``user_request_with_task_preamble`` resolves to
     ``{{ task_preamble }} + ## Original User Request + {{ input }}``.
   - ``context.session.local_paths`` is guarded by ``has_local_access``:
     - Renders path-list block when ``has_local_access=True`` AND at least
       one path is present.
     - Renders empty when ``has_local_access=False`` (any inferencer
       without local file access).
     - Renders empty when ``has_local_access`` is undefined (safety net
       for legacy callers).

3. **Consumer-template integration** (end-to-end through the actual
   wrapper templates that reference these variables):
   - ``plan/main/initial.jinja2`` → contains user-request block
   - ``plan/main/followup.jinja2`` → contains user-request block
   - ``plan/main/review.jinja2`` → contains user-request block
   - ``implementation/main/initial.jinja2`` → contains user-request block
   - ``implementation/main/followup.jinja2`` → contains user-request block
   - ``deep_research/main/initial.jinja2`` → contains user-request block
   - ``task_breakdown/main/initial.jinja2`` → contains user-request block
   - ``individual_proposal/main/initial.jinja2`` → DOES NOT use the
     wrapper (different semantic: ``{{ input }}`` is research context,
     not user query).

4. **No regressions**:
   - No raw Jinja2 syntax (``{% ... %}`` / ``{{ ... }}``) leaks into
     rendered output.
   - Inputs are correctly interpolated, not literal.
"""

import os
from pathlib import Path

import pytest

import agent_foundation.resources as _af_res
from rich_python_utils.string_utils.formatting.template_manager import (
    TemplateManager,
)


AF_TEMPLATES_ROOT = Path(_af_res.__file__).parent / "prompt_templates"

# Files that MUST use the new ``user_request_with_task_preamble`` global.
TEMPLATES_USING_USER_REQUEST_VAR = [
    ("plan", "initial"),
    ("plan", "followup"),
    ("plan", "review"),
    ("implementation", "initial"),
    ("implementation", "followup"),
    ("deep_research", "initial"),
    ("task_breakdown", "initial"),
]

# Files that should NOT use the wrapper (different semantic).
TEMPLATES_NOT_USING_USER_REQUEST_VAR = [
    ("individual_proposal", "initial"),  # {{ input }} = research context
]

# Files where ``context.session.local_paths`` should appear in the
# task_preamble (after recent refactor).
TEMPLATES_WITH_LOCAL_PATHS_VAR = [
    "plan/main/_variables/task_preamble/default.jinja2",
    "implementation/main/_variables/task_preamble/default.jinja2",
    "deep_research/main/_variables/task_preamble/default.jinja2",
    "individual_proposal/main/_variables/task_preamble/subtask/default.jinja2",
]


# ---------------------------------------------------------------------------
# Section 1: Source-of-truth files exist
# ---------------------------------------------------------------------------


class TestSharedVariableFilesExist:
    """The shared variable files under _variables/context/ must exist."""

    def test_user_request_with_task_preamble_file_exists(self) -> None:
        path = AF_TEMPLATES_ROOT / "_variables" / "context" / "user_request_with_task_preamble.jinja2"
        assert path.is_file(), f"Missing: {path}"

    def test_user_request_with_task_preamble_has_required_content(self) -> None:
        path = AF_TEMPLATES_ROOT / "_variables" / "context" / "user_request_with_task_preamble.jinja2"
        content = path.read_text(encoding="utf-8")
        assert "{{ task_preamble }}" in content, (
            "Source-of-truth variable should render task_preamble"
        )
        assert "{{ input }}" in content, (
            "Source-of-truth variable should render input"
        )
        assert "Original User Request" in content, (
            "Source-of-truth variable should label the input section"
        )

    def test_session_local_paths_file_exists(self) -> None:
        path = AF_TEMPLATES_ROOT / "_variables" / "context" / "session" / "local_paths.jinja2"
        assert path.is_file(), f"Missing: {path}"

    def test_session_local_paths_has_required_content(self) -> None:
        path = AF_TEMPLATES_ROOT / "_variables" / "context" / "session" / "local_paths.jinja2"
        content = path.read_text(encoding="utf-8")
        assert "has_local_access" in content, (
            "local_paths must be guarded by has_local_access"
        )
        assert "session_root_path" in content, (
            "local_paths should list session_root_path"
        )


# ---------------------------------------------------------------------------
# Section 2: Consumer templates reference the shared variable
# ---------------------------------------------------------------------------


class TestConsumerTemplatesUseSharedVariables:
    """The 7 templates listed in TEMPLATES_USING_USER_REQUEST_VAR must
    reference ``{{ user_request_with_task_preamble }}`` (NOT inline
    ``{{ task_preamble }} ... {{ input }}``)."""

    @pytest.mark.parametrize("space,template_key", TEMPLATES_USING_USER_REQUEST_VAR)
    def test_template_uses_user_request_variable(
        self, space: str, template_key: str
    ) -> None:
        path = AF_TEMPLATES_ROOT / space / "main" / f"{template_key}.jinja2"
        assert path.is_file(), f"Template missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert "{{ user_request_with_task_preamble }}" in content, (
            f"{space}/main/{template_key}.jinja2 should reference the "
            "shared user_request_with_task_preamble variable"
        )

    @pytest.mark.parametrize("space,template_key", TEMPLATES_USING_USER_REQUEST_VAR)
    def test_template_does_not_duplicate_inline_pattern(
        self, space: str, template_key: str
    ) -> None:
        """No template should have BOTH {{ task_preamble }} AND {{ input }}
        directly adjacent — they should come via the shared variable."""
        path = AF_TEMPLATES_ROOT / space / "main" / f"{template_key}.jinja2"
        content = path.read_text(encoding="utf-8")
        # Inline pattern is when {{ task_preamble }} and {{ input }} appear
        # with at most 5 lines between them (excluding via the shared var).
        lines = content.split("\n")
        preamble_lines = [i for i, l in enumerate(lines) if "{{ task_preamble }}" in l]
        input_lines = [i for i, l in enumerate(lines) if "{{ input }}" in l]
        for p in preamble_lines:
            for i in input_lines:
                if 0 < i - p <= 5:
                    pytest.fail(
                        f"{space}/main/{template_key}.jinja2 has inline "
                        f"{{{{ task_preamble }}}} ... {{{{ input }}}} at lines "
                        f"{p+1}, {i+1} — should use the shared variable instead"
                    )


class TestSpecialCaseTemplates:
    """Templates that intentionally do NOT use the wrapper (different
    semantic for {{ input }})."""

    @pytest.mark.parametrize("space,template_key", TEMPLATES_NOT_USING_USER_REQUEST_VAR)
    def test_individual_proposal_keeps_research_context_semantic(
        self, space: str, template_key: str
    ) -> None:
        """individual_proposal/main/initial.jinja2 uses {{ input }} as
        research context, NOT user request. It should keep its inline
        rendering to preserve semantic clarity."""
        path = AF_TEMPLATES_ROOT / space / "main" / f"{template_key}.jinja2"
        assert path.is_file(), f"Template missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert "{{ user_request_with_task_preamble }}" not in content, (
            f"{space}/main/{template_key}.jinja2 should NOT use the "
            "user-request wrapper — its {{ input }} is research context, "
            "not a user query"
        )


class TestConsumerTemplatesUseLocalPaths:
    """Task-preamble templates should reference the shared local_paths
    variable instead of inlining path-printing blocks."""

    @pytest.mark.parametrize("rel_path", TEMPLATES_WITH_LOCAL_PATHS_VAR)
    def test_template_uses_local_paths_variable(self, rel_path: str) -> None:
        path = AF_TEMPLATES_ROOT / rel_path
        assert path.is_file(), f"Template missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert "{{ context.session.local_paths }}" in content, (
            f"{rel_path} should reference the shared "
            "context.session.local_paths variable"
        )

    @pytest.mark.parametrize("rel_path", TEMPLATES_WITH_LOCAL_PATHS_VAR)
    def test_template_no_inline_path_block(self, rel_path: str) -> None:
        """The refactored task_preamble files should NOT have an inline
        '- session_root_path: {{ session_root_path }}' block (that's now
        encapsulated inside the shared variable)."""
        path = AF_TEMPLATES_ROOT / rel_path
        content = path.read_text(encoding="utf-8")
        # The shared var is the only legitimate place that contains this
        # phrase; the refactored task_preambles should be free of it.
        assert "- session_root_path:" not in content, (
            f"{rel_path} still contains inline session_root_path block — "
            "should be replaced by {{ context.session.local_paths }}"
        )


# ---------------------------------------------------------------------------
# Section 3: End-to-end rendering through TemplateManager
# ---------------------------------------------------------------------------


@pytest.fixture
def tm() -> TemplateManager:
    """TemplateManager wired to AgentFoundation prompt_templates."""
    return TemplateManager(
        templates=[str(AF_TEMPLATES_ROOT)],
        active_template_type="main",
        predefined_variables=True,
        enable_templated_feed=True,
        default_template_key="initial",
    )


SENTINEL_PREAMBLE = "<<<TEST_PREAMBLE_SENTINEL>>>"
SENTINEL_INPUT = "<<<TEST_INPUT_SENTINEL>>>"
SENTINEL_PATH = "/tmp/test_root_sentinel_xyz"


class TestEndToEndRendering:
    """Render real consumer templates and verify the shared variables
    interpolate correctly."""

    @pytest.mark.parametrize("space,template_key", TEMPLATES_USING_USER_REQUEST_VAR)
    def test_user_request_block_renders_both_preamble_and_input(
        self, tm: TemplateManager, space: str, template_key: str
    ) -> None:
        """End-to-end: the wrapper should expand to include both
        ``task_preamble`` AND ``input``, with the ``## Original User
        Request`` subheader between them."""
        rendered = tm(
            template_key=template_key,
            active_template_root_space=space,
            input=SENTINEL_INPUT,
            output_path="/tmp/output.md",
            task_preamble=SENTINEL_PREAMBLE,
        )
        assert SENTINEL_PREAMBLE in rendered, (
            f"{space}/{template_key}: task_preamble sentinel missing "
            "from rendered output"
        )
        assert SENTINEL_INPUT in rendered, (
            f"{space}/{template_key}: input sentinel missing from "
            "rendered output"
        )
        assert "Original User Request" in rendered, (
            f"{space}/{template_key}: '## Original User Request' "
            "subheader missing — wrapper variable not expanded correctly"
        )

    @pytest.mark.parametrize("space,template_key", TEMPLATES_USING_USER_REQUEST_VAR)
    def test_no_raw_jinja_leak(
        self, tm: TemplateManager, space: str, template_key: str
    ) -> None:
        """Rendered output must contain ZERO raw Jinja2 markers."""
        rendered = tm(
            template_key=template_key,
            active_template_root_space=space,
            input=SENTINEL_INPUT,
            output_path="/tmp/output.md",
        )
        assert "{{" not in rendered, (
            f"{space}/{template_key}: raw {{{{ leaked into output"
        )
        assert "{%" not in rendered, (
            f"{space}/{template_key}: raw {{% leaked into output"
        )
        assert "{{ user_request_with_task_preamble }}" not in rendered
        assert "{{ context.session.local_paths }}" not in rendered


class TestLocalPathsHasLocalAccessGuard:
    """Verify ``context.session.local_paths`` gates on ``has_local_access``."""

    def test_renders_when_has_local_access_true_and_paths_present(
        self, tm: TemplateManager
    ) -> None:
        rendered = tm(
            template_key="initial",
            active_template_root_space="plan",
            input=SENTINEL_INPUT,
            output_path="/tmp/output.md",
            session_root_path=SENTINEL_PATH,
            has_local_access=True,
        )
        assert SENTINEL_PATH in rendered, (
            "local_paths block should include session_root_path when "
            "has_local_access=True"
        )

    def test_hidden_when_has_local_access_false(
        self, tm: TemplateManager
    ) -> None:
        rendered = tm(
            template_key="initial",
            active_template_root_space="plan",
            input=SENTINEL_INPUT,
            output_path="/tmp/output.md",
            session_root_path=SENTINEL_PATH,
            has_local_access=False,
        )
        assert SENTINEL_PATH not in rendered, (
            "local_paths block should be hidden when has_local_access=False "
            "(API inferencer can't use shell paths)"
        )

    def test_hidden_when_has_local_access_undefined(
        self, tm: TemplateManager
    ) -> None:
        """Safety net: legacy callers that never set has_local_access
        should get the empty/safe default (no path leakage)."""
        rendered = tm(
            template_key="initial",
            active_template_root_space="plan",
            input=SENTINEL_INPUT,
            output_path="/tmp/output.md",
            session_root_path=SENTINEL_PATH,
            # has_local_access deliberately omitted
        )
        assert SENTINEL_PATH not in rendered, (
            "local_paths block should be hidden when has_local_access "
            "is undefined (safe default)"
        )


class TestSpecialCaseRendering:
    """individual_proposal/initial.jinja2 must NOT label {{ input }} as
    'Original User Request' — it's research context."""

    def test_individual_proposal_does_not_wrap_input_as_user_request(
        self, tm: TemplateManager
    ) -> None:
        rendered = tm(
            template_key="initial",
            active_template_root_space="individual_proposal",
            input=SENTINEL_INPUT,
            output_path="/tmp/output.md",
            task_preamble=SENTINEL_PREAMBLE,
        )
        assert SENTINEL_INPUT in rendered, "research-context input missing"
        # The shared wrapper's specific subheader must NOT appear
        # (individual_proposal uses its own "Research Context" label).
        assert "## Original User Request" not in rendered, (
            "individual_proposal incorrectly using the user_request wrapper "
            "— input is research context, not user query"
        )
