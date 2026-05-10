"""Tests for the path-aware followup mechanism in DualInferencer.

Covers:
  - F1: _resolve_prior_proposer_output_path() helper — pure unit tests over
        all 3 tiers and edge cases (12 methods).
  - F2: _build_followup_prompt / _build_review_prompt feed-dict shape — 8 methods.
  - F3: Real-Jinja end-to-end render of plan/main/followup.jinja2 — 5 methods.
  - F4: Real-Jinja end-to-end render of plan/main/review.jinja2 — 3 methods.
  - F5: IsolatedAsyncioTestCase E2E through full consensus loop — 3 methods.

See _docs/_plans/dual_inferencer_path_aware_followup_INTEGRATED_plan.md.

All tests run in <10 seconds total. Zero LLM calls.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import jinja2

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    DualInferencerResponse,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)


# =====================================================================
# Helpers
# =====================================================================


class _FakeWorkspace:
    """Minimal workspace stub matching the API the helper uses.

    Mirrors the surface of inferencer_workspace.InferencerWorkspace just
    enough for _resolve_prior_proposer_output_path() to traverse it.
    """

    def __init__(
        self,
        root: str,
        deliverables_subdir: str | None = "final_deliverables",
    ):
        self.root = root
        self.outputs_dir = os.path.join(root, "outputs")
        self.deliverables_dir = (
            os.path.join(self.outputs_dir, deliverables_subdir)
            if deliverables_subdir
            else None
        )
        os.makedirs(self.outputs_dir, exist_ok=True)
        if self.deliverables_dir:
            os.makedirs(self.deliverables_dir, exist_ok=True)

    @property
    def has_deliverables(self) -> bool:
        d = self.deliverables_dir
        return bool(d and os.path.isdir(d) and os.listdir(d))

    def deliverable_path(self, relative: str) -> str | None:
        if not self.deliverables_dir:
            return None
        return os.path.join(self.deliverables_dir, relative)

    def deliverable_paths(self) -> list[str]:
        if not self.deliverables_dir or not os.path.isdir(self.deliverables_dir):
            return []
        return sorted(os.listdir(self.deliverables_dir))

    def output_path(self, relative: str) -> str:
        return os.path.join(self.outputs_dir, relative)


def _make_dual_with_proposers(base, fixer=None, state=None):
    """Construct a DualInferencer instance via __new__ with just enough
    state to exercise _resolve_prior_proposer_output_path() and
    _active_proposer().

    Bypasses __init__ deliberately because the helper is fully isolated
    from any other Dual state (it touches only base_inferencer,
    fixer_inferencer, and _state).
    """
    dual = DualInferencer.__new__(DualInferencer)
    dual.base_inferencer = base
    dual.fixer_inferencer = fixer
    dual._state = state if state is not None else {}
    return dual


def _mock_proposer(workspace=None, output_path="output.md"):
    proposer = MagicMock()
    proposer._workspace = workspace
    proposer._output_path = output_path
    return proposer


def _make_mock_inferencer(response=None, side_effect=None, on_call=None):
    """Mirror of the helper used in test_dual_inferencer_resume.py, with
    optional on_call callback that lets tests capture the rendered input."""
    inf = MagicMock()
    if on_call is not None:

        async def _capture(inp, *args, **kwargs):
            return on_call(inp)

        inf.ainfer = AsyncMock(side_effect=_capture)
    elif side_effect is not None:
        inf.ainfer = AsyncMock(side_effect=side_effect)
    else:
        inf.ainfer = AsyncMock(return_value=response or "mock response")
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()
    inf.reset_session = MagicMock()
    return inf


def _review_json(approved: bool, severity: str = "COSMETIC") -> str:
    review = {
        "approved": approved,
        "severity": severity,
        "issues": []
        if approved
        else [
            {
                "severity": severity,
                "category": "test",
                "description": "Test issue",
                "location": "N/A",
                "suggestion": "Fix it",
            }
        ],
        "reasoning": "Test reasoning.",
    }
    return f"```json\n{json.dumps(review, indent=2)}\n```"


# Path to the actual plan/main templates (used by F3 and F4 real-Jinja tests).
# Use __file__-relative path resolution (NOT CWD-relative) so tests are
# portable.
_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[5]
    / "src"
    / "agent_foundation"
    / "resources"
    / "prompt_templates"
    / "plan"
    / "main"
)


# =====================================================================
# F1 — _resolve_prior_proposer_output_path() — Pure Unit Tests
# =====================================================================


class TestResolvePriorProposerOutputPath(unittest.TestCase):
    """Cover all 3 tiers + 4 edge cases + active proposer switching."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # -- Tier 1: deliverable file --

    def test_T1_returns_deliverable_when_output_md_present(self):
        ws = _FakeWorkspace(self.tmp)
        deliv = os.path.join(ws.deliverables_dir, "output.md")
        with open(deliv, "w") as f:
            f.write("# plan content")
        dual = _make_dual_with_proposers(_mock_proposer(workspace=ws))
        self.assertEqual(dual._resolve_prior_proposer_output_path(), deliv)

    def test_T1_prefers_output_path_basename_over_alphabetical(self):
        ws = _FakeWorkspace(self.tmp)
        for fn in ("a_first.md", "output.md", "z_last.md"):
            with open(os.path.join(ws.deliverables_dir, fn), "w") as f:
                f.write("x")
        dual = _make_dual_with_proposers(_mock_proposer(workspace=ws))
        self.assertEqual(
            dual._resolve_prior_proposer_output_path(),
            os.path.join(ws.deliverables_dir, "output.md"),
        )

    def test_T1_skips_dotfile_falls_through_to_T2(self):
        ws = _FakeWorkspace(self.tmp)
        # Only the dotfile marker; no real deliverable.
        with open(os.path.join(ws.deliverables_dir, ".self_promoted"), "w") as f:
            f.write("")
        # But there IS a Tier-2 fallback.
        with open(os.path.join(ws.outputs_dir, "output.md"), "w") as f:
            f.write("x")
        dual = _make_dual_with_proposers(_mock_proposer(workspace=ws))
        self.assertEqual(
            dual._resolve_prior_proposer_output_path(),
            os.path.join(ws.outputs_dir, "output.md"),
        )

    def test_T1_alphabetical_fallback_when_preferred_absent(self):
        ws = _FakeWorkspace(self.tmp)
        with open(os.path.join(ws.deliverables_dir, "report.md"), "w") as f:
            f.write("x")
        with open(os.path.join(ws.deliverables_dir, "summary.json"), "w") as f:
            f.write("{}")
        dual = _make_dual_with_proposers(
            _mock_proposer(workspace=ws, output_path="not_present.md")
        )
        # report.md is alphabetically first non-dotfile.
        self.assertEqual(
            dual._resolve_prior_proposer_output_path(),
            os.path.join(ws.deliverables_dir, "report.md"),
        )

    # -- Tier 2: outputs file --

    def test_T2_returns_outputs_md_when_no_deliverables(self):
        ws = _FakeWorkspace(self.tmp, deliverables_subdir=None)
        out = os.path.join(ws.outputs_dir, "output.md")
        with open(out, "w") as f:
            f.write("# plan")
        dual = _make_dual_with_proposers(_mock_proposer(workspace=ws))
        self.assertEqual(dual._resolve_prior_proposer_output_path(), out)

    def test_T2_uses_configured_output_path_basename(self):
        ws = _FakeWorkspace(self.tmp, deliverables_subdir=None)
        out = os.path.join(ws.outputs_dir, "my_report.md")
        with open(out, "w") as f:
            f.write("x")
        dual = _make_dual_with_proposers(
            _mock_proposer(workspace=ws, output_path="my_report.md")
        )
        self.assertEqual(dual._resolve_prior_proposer_output_path(), out)

    # -- Tier 3 / edge cases --

    def test_T3_returns_None_when_neither_exists(self):
        ws = _FakeWorkspace(self.tmp, deliverables_subdir=None)
        dual = _make_dual_with_proposers(_mock_proposer(workspace=ws))
        self.assertIsNone(dual._resolve_prior_proposer_output_path())

    def test_proposer_None_returns_None(self):
        dual = _make_dual_with_proposers(None)
        self.assertIsNone(dual._resolve_prior_proposer_output_path())

    def test_proposer_workspace_None_returns_None(self):
        dual = _make_dual_with_proposers(_mock_proposer(workspace=None))
        self.assertIsNone(dual._resolve_prior_proposer_output_path())

    def test_proposer_without_output_path_uses_default_basename(self):
        ws = _FakeWorkspace(self.tmp, deliverables_subdir=None)
        out = os.path.join(ws.outputs_dir, "output.md")
        with open(out, "w") as f:
            f.write("x")
        proposer = MagicMock()
        proposer._workspace = ws
        # Configure as if the attribute is None — helper must default to "output.md".
        proposer._output_path = None
        dual = _make_dual_with_proposers(proposer)
        self.assertEqual(dual._resolve_prior_proposer_output_path(), out)

    # -- Active proposer switching --

    def test_after_fix_iteration_resolves_fixer_path(self):
        base_ws = _FakeWorkspace(os.path.join(self.tmp, "base"))
        fixer_ws = _FakeWorkspace(os.path.join(self.tmp, "fixer"))
        base_deliv = os.path.join(base_ws.deliverables_dir, "output.md")
        fixer_deliv = os.path.join(fixer_ws.deliverables_dir, "output.md")
        with open(base_deliv, "w") as f:
            f.write("base plan")
        with open(fixer_deliv, "w") as f:
            f.write("fixer plan")
        dual = _make_dual_with_proposers(
            base=_mock_proposer(workspace=base_ws),
            fixer=_mock_proposer(workspace=fixer_ws),
            # Simulate a state where last iteration had counter_feedback set
            # (fix ran), so _active_proposer() returns the fixer.
            state={"attempt_record": {"iterations": [{"counter_feedback": "needs"}]}},
        )
        self.assertEqual(dual._resolve_prior_proposer_output_path(), fixer_deliv)

    def test_two_agent_mode_fixer_None_resolves_base(self):
        """If fixer_inferencer is None, _active_proposer() falls back to base."""
        base_ws = _FakeWorkspace(self.tmp)
        base_deliv = os.path.join(base_ws.deliverables_dir, "output.md")
        with open(base_deliv, "w") as f:
            f.write("base")
        dual = _make_dual_with_proposers(
            base=_mock_proposer(workspace=base_ws),
            fixer=None,
            state={"attempt_record": {"iterations": [{"counter_feedback": "needs"}]}},
        )
        self.assertEqual(dual._resolve_prior_proposer_output_path(), base_deliv)


# =====================================================================
# F2 — Builder Feed-Dict Shape Tests
# =====================================================================


def _stub_dual_for_builder_tests(prior_path=None):
    """Build a Dual stub that intercepts _render_role_prompt to return the
    feed dict directly (so tests can assert on it)."""
    dual = DualInferencer.__new__(DualInferencer)
    dual.base_inferencer = MagicMock(_workspace=None)
    dual.fixer_inferencer = None
    dual._state = {}
    # Stub placeholders to default values.
    dual.placeholder_input = "input"
    dual.placeholder_proposal = "proposal"
    dual.placeholder_issues = "issues"
    dual.placeholder_reasoning = "reasoning"
    dual.placeholder_counter_feedback = "counter_feedback"
    dual.consensus_config = MagicMock(enable_counter_feedback=False)
    dual._serialize_issues = lambda issues: f"ISSUES({len(issues)})"
    # Force the helper to return the desired path (None or string).
    dual._resolve_prior_proposer_output_path = MagicMock(return_value=prior_path)
    # Intercept render to return the feed dict so tests can inspect it.
    dual._render_role_prompt = MagicMock(side_effect=lambda role, feed, cfg: feed)
    return dual


class TestBuildFollowupPromptFeedDict(unittest.TestCase):
    def test_followup_feed_includes_main_response(self):
        dual = _stub_dual_for_builder_tests(prior_path="/tmp/x/output.md")
        feed = dual._build_followup_prompt(
            inference_input="INPUT",
            proposal="PROPOSAL_TEXT",
            parsed_review={"issues": [], "reasoning": "ok"},
            inference_config={},
            review_output="REVIEW_RAW",
        )
        self.assertEqual(feed["main_response"], "PROPOSAL_TEXT")

    def test_followup_feed_includes_prior_output_path(self):
        dual = _stub_dual_for_builder_tests(prior_path="/tmp/x/output.md")
        feed = dual._build_followup_prompt(
            inference_input="i",
            proposal="p",
            parsed_review={"issues": [], "reasoning": ""},
            inference_config={},
        )
        self.assertEqual(feed["prior_output_path"], "/tmp/x/output.md")

    def test_followup_feed_emits_empty_string_when_path_None(self):
        dual = _stub_dual_for_builder_tests(prior_path=None)
        feed = dual._build_followup_prompt(
            inference_input="i",
            proposal="p",
            parsed_review={"issues": [], "reasoning": ""},
            inference_config={},
        )
        self.assertEqual(feed["prior_output_path"], "")
        # Must NOT be the literal string "None".
        self.assertNotEqual(feed["prior_output_path"], "None")
        self.assertIsNotNone(feed["prior_output_path"])

    def test_followup_feed_includes_proposal_for_backward_compat(self):
        """Even though main_response is set, placeholder_proposal must still be set."""
        dual = _stub_dual_for_builder_tests()
        feed = dual._build_followup_prompt(
            inference_input="i",
            proposal="PROPOSAL",
            parsed_review={"issues": [], "reasoning": ""},
            inference_config={},
        )
        self.assertEqual(feed["proposal"], "PROPOSAL")

    def test_followup_feed_includes_reviewer_response_when_provided(self):
        dual = _stub_dual_for_builder_tests()
        feed = dual._build_followup_prompt(
            inference_input="i",
            proposal="p",
            parsed_review={"issues": [], "reasoning": ""},
            inference_config={},
            review_output="REVIEW_RAW",
        )
        self.assertEqual(feed["reviewer_response"], "REVIEW_RAW")

    def test_followup_feed_emits_empty_string_when_review_output_None(self):
        dual = _stub_dual_for_builder_tests()
        feed = dual._build_followup_prompt(
            inference_input="i",
            proposal="p",
            parsed_review={"issues": [], "reasoning": ""},
            inference_config={},
            review_output=None,
        )
        self.assertEqual(feed["reviewer_response"], "")
        self.assertNotEqual(feed["reviewer_response"], "None")


class TestBuildReviewPromptFeedDict(unittest.TestCase):
    def test_review_feed_includes_main_response_and_prior_output_path(self):
        dual = _stub_dual_for_builder_tests(prior_path="/tmp/x/output.md")
        feed = dual._build_review_prompt(
            inference_input="i",
            proposal="PROPOSAL",
            counter_feedback=None,
            inference_config={},
        )
        self.assertEqual(feed["main_response"], "PROPOSAL")
        self.assertEqual(feed["prior_output_path"], "/tmp/x/output.md")

    def test_review_feed_emits_empty_string_when_path_None(self):
        dual = _stub_dual_for_builder_tests(prior_path=None)
        feed = dual._build_review_prompt(
            inference_input="i",
            proposal="p",
            counter_feedback=None,
            inference_config={},
        )
        self.assertEqual(feed["prior_output_path"], "")


# =====================================================================
# F3 — Real-Jinja End-to-End Render of plan/main/followup.jinja2
# =====================================================================


class TestFollowupTemplateRendersPathAware(unittest.TestCase):
    """Render the actual plan/main/followup.jinja2 with the feed dict
    produced by _build_followup_prompt and assert the resulting prompt
    text contains the path, the cp instruction, and a non-empty
    <ProposedDocument> tag."""

    def setUp(self):
        if not (_TEMPLATE_DIR / "followup.jinja2").is_file():
            self.skipTest(f"Template not found at {_TEMPLATE_DIR}")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # Set up a real on-disk deliverable to be discovered by the helper.
        self.deliv_dir = os.path.join(self.tmp, "outputs", "final_deliverables")
        os.makedirs(self.deliv_dir, exist_ok=True)
        self.prior_file = os.path.join(self.deliv_dir, "output.md")
        with open(self.prior_file, "w") as f:
            f.write("# Title\n\n## 1. Section\nBody.")

    def _build_feed(self, prior_path):
        dual = _stub_dual_for_builder_tests(prior_path=prior_path)
        return dual._build_followup_prompt(
            inference_input="USER REQUEST",
            proposal="PRIOR PROPOSAL TEXT",
            parsed_review={"issues": [], "reasoning": "ok"},
            inference_config={},
            review_output="REVIEW_RAW",
        )

    def _render_template(self, feed, output_path="/tmp/dest/output.md"):
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
            undefined=jinja2.StrictUndefined,
        )
        # Provide template variables that aren't provided by the builder.
        feed = dict(feed)
        feed.setdefault("task_preamble", "")
        feed.setdefault("output_path", output_path)
        feed.setdefault("enable_elegant_mode", False)
        feed.setdefault("instructions", MagicMock())
        feed.setdefault("task_instructions", "")
        return env.get_template("followup.jinja2").render(**feed)

    def test_rendered_prompt_contains_prior_output_path_literal(self):
        feed = self._build_feed(self.prior_file)
        rendered = self._render_template(feed)
        self.assertIn(self.prior_file, rendered)

    def test_rendered_prompt_contains_cp_instruction(self):
        feed = self._build_feed(self.prior_file)
        out = "/tmp/dest/output.md"
        rendered = self._render_template(feed, output_path=out)
        # Match `cp <prior_file> <output_path>` allowing surrounding whitespace.
        pattern = rf"cp\s+{re.escape(self.prior_file)}\s+{re.escape(out)}"
        self.assertRegex(rendered, pattern)

    def test_rendered_prompt_proposed_document_tag_populated(self):
        feed = self._build_feed(self.prior_file)
        rendered = self._render_template(feed)
        # Tag must contain the proposal text — empty-tag bug must NOT recur.
        m = re.search(
            r"<ProposedDocument>\s*(.*?)\s*</ProposedDocument>",
            rendered,
            re.DOTALL,
        )
        self.assertIsNotNone(m, "Could not find <ProposedDocument> tag")
        self.assertIn("PRIOR PROPOSAL TEXT", m.group(1))

    def test_rendered_prompt_falls_back_gracefully_when_path_empty(self):
        feed = self._build_feed(prior_path=None)  # helper returns None → ""
        rendered = self._render_template(feed)
        # No `cp ` instruction.
        self.assertNotIn("cp ", rendered)
        # Fallback wording present.
        self.assertIn("on-disk path is unavailable", rendered)
        # Inline content still in <ProposedDocument>.
        self.assertIn("PRIOR PROPOSAL TEXT", rendered)

    def test_rendered_prompt_does_not_leak_literal_None(self):
        feed = self._build_feed(prior_path=None)
        rendered = self._render_template(feed)
        # The empty-string sentinel must prevent "None" from leaking into
        # any of the path-related blocks (we don't check the entire prompt
        # since some literal "None" values might legitimately appear in
        # template body text).
        # Specifically check the path-block area:
        self.assertNotIn("`None`", rendered)


# =====================================================================
# F4 — Real-Jinja End-to-End Render of plan/main/review.jinja2
# =====================================================================


class TestReviewTemplateRendersPathAware(unittest.TestCase):
    def setUp(self):
        if not (_TEMPLATE_DIR / "review.jinja2").is_file():
            self.skipTest(f"Template not found at {_TEMPLATE_DIR}")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.prior_file = os.path.join(self.tmp, "output.md")
        with open(self.prior_file, "w") as f:
            f.write("doc")

    def _render(self, prior_path):
        dual = _stub_dual_for_builder_tests(prior_path=prior_path)
        feed = dual._build_review_prompt(
            inference_input="USER REQUEST",
            proposal="DOC TEXT",
            counter_feedback=None,
            inference_config={},
        )
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
            undefined=jinja2.StrictUndefined,
        )
        feed = dict(feed)
        feed.setdefault("task_preamble", "")
        feed.setdefault("output_path", "/tmp/dest/output.md")
        feed.setdefault("enable_elegant_mode", False)
        feed.setdefault("enable_deep_mode", False)
        feed.setdefault("instructions", MagicMock())
        feed.setdefault("task_instructions", "")
        # review.jinja2 references {{ counter_feedback }} inside an `{% if %}`
        # but StrictUndefined chokes on the lookup itself; provide an empty
        # default so the conditional evaluates falsy.
        feed.setdefault("counter_feedback", "")
        return env.get_template("review.jinja2").render(**feed)

    def test_review_rendered_prompt_contains_prior_output_path(self):
        rendered = self._render(self.prior_file)
        self.assertIn(self.prior_file, rendered)
        self.assertIn("read_file", rendered)

    def test_review_rendered_prompt_falls_back_gracefully(self):
        rendered = self._render(prior_path=None)
        self.assertNotIn("read_file", rendered)
        self.assertIn("DOC TEXT", rendered)

    def test_review_rendered_prompt_proposed_document_populated(self):
        rendered = self._render(self.prior_file)
        m = re.search(
            r"<ProposedDocument>\s*(.*?)\s*</ProposedDocument>",
            rendered,
            re.DOTALL,
        )
        self.assertIsNotNone(m)
        self.assertIn("DOC TEXT", m.group(1))


# =====================================================================
# F5 — IsolatedAsyncioTestCase E2E (full consensus loop)
# =====================================================================


class TestPathAwareE2E(unittest.IsolatedAsyncioTestCase):
    """Run the full Dual consensus loop with mocked LLMs and assert that
    the fixer's captured prompt input contains the prior path.

    Uses inline followup_prompt / review_prompt templates that reference
    ``{{ prior_output_path }}`` so we can exercise the integration without
    needing a TemplateManager configured. This mirrors how production
    topologies (e.g. plan/main/followup.jinja2) reference the variable.
    """

    # Inline templates with the same path-aware contract as production
    # plan/main/{followup,review}.jinja2.
    _FOLLOWUP_TMPL = (
        "FOLLOWUP_PROMPT\n"
        "<OriginalRequest>{{ input }}</OriginalRequest>\n"
        "{% if prior_output_path %}"
        "PRIOR_PATH: {{ prior_output_path }}\n"
        "FIRST_ACTION: cp {{ prior_output_path }} <output>\n"
        "{% else %}"
        "PRIOR_PATH_UNAVAILABLE\n"
        "{% endif %}"
        "<ProposedDocument>{{ main_response }}</ProposedDocument>\n"
        "<ReviewerFeedback>{{ reviewer_response }}</ReviewerFeedback>\n"
    )
    _REVIEW_TMPL = (
        "REVIEW_PROMPT\n"
        "<OriginalRequest>{{ input }}</OriginalRequest>\n"
        "{% if prior_output_path %}"
        "PRIOR_PATH: {{ prior_output_path }}\n"
        "{% endif %}"
        "<ProposedDocument>{{ main_response }}</ProposedDocument>\n"
    )

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _make_workspace(self, name):
        ws = _FakeWorkspace(os.path.join(self.tmp, name))
        return ws

    async def test_fixer_input_contains_prior_path_when_workspace_present(self):
        # Base proposer: writes a real deliverable to its workspace.
        base_ws = self._make_workspace("base")
        prior_file = os.path.join(base_ws.deliverables_dir, "output.md")
        with open(prior_file, "w") as f:
            f.write("# Plan\n\n## 1. Section\nContent.")

        captured_fixer_inputs: list[str] = []

        # Build mock proposers.
        base = _make_mock_inferencer(response="<Response>summary</Response>")
        base._workspace = base_ws
        base._output_path = "output.md"

        reviewer = _make_mock_inferencer(
            side_effect=[
                _review_json(approved=False, severity="MAJOR"),
                _review_json(approved=True),
            ]
        )

        fixer = _make_mock_inferencer(
            on_call=lambda inp: captured_fixer_inputs.append(inp)
            or "<ImprovedProposal>fixed</ImprovedProposal>"
        )

        dual = DualInferencer(
            base_inferencer=base,
            review_inferencer=reviewer,
            fixer_inferencer=fixer,
            followup_prompt=self._FOLLOWUP_TMPL,
            review_prompt=self._REVIEW_TMPL,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("test request")
        self.assertIsInstance(result, DualInferencerResponse)

        # The fixer must have been invoked exactly once.
        self.assertEqual(len(captured_fixer_inputs), 1)
        fixer_input = captured_fixer_inputs[0]
        # Strongest assertion: the actual prior file path appears in fixer prompt.
        self.assertIn(prior_file, fixer_input)
        # And the cp instruction with that path as source.
        self.assertRegex(fixer_input, rf"cp\s+{re.escape(prior_file)}")
        # <ProposedDocument> tag must be populated (Bug A regression guard).
        self.assertIn("<ProposedDocument>summary</ProposedDocument>", fixer_input)

    async def test_fixer_input_no_path_when_no_workspace(self):
        """Backward-compat: when proposer has no workspace, no path is added."""
        captured: list[str] = []
        base = _make_mock_inferencer(response="<Response>summary</Response>")
        # Explicitly NO workspace.
        base._workspace = None
        reviewer = _make_mock_inferencer(
            side_effect=[
                _review_json(approved=False, severity="MAJOR"),
                _review_json(approved=True),
            ]
        )
        fixer = _make_mock_inferencer(
            on_call=lambda inp: captured.append(inp)
            or "<ImprovedProposal>fixed</ImprovedProposal>"
        )

        dual = DualInferencer(
            base_inferencer=base,
            review_inferencer=reviewer,
            fixer_inferencer=fixer,
            followup_prompt=self._FOLLOWUP_TMPL,
            review_prompt=self._REVIEW_TMPL,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        await dual._ainfer("test request")
        self.assertEqual(len(captured), 1)
        # Fallback wording should appear; no `cp ` line.
        self.assertNotIn("cp ", captured[0])
        self.assertIn("PRIOR_PATH_UNAVAILABLE", captured[0])

    async def test_review_inferencer_input_contains_prior_path(self):
        """Symmetric assertion for the reviewer."""
        base_ws = self._make_workspace("base")
        prior_file = os.path.join(base_ws.deliverables_dir, "output.md")
        with open(prior_file, "w") as f:
            f.write("plan content")

        captured_reviewer_inputs: list[str] = []

        base = _make_mock_inferencer(response="<Response>summary</Response>")
        base._workspace = base_ws
        base._output_path = "output.md"

        reviewer = _make_mock_inferencer(
            on_call=lambda inp: captured_reviewer_inputs.append(inp)
            or _review_json(approved=True)
        )

        dual = DualInferencer(
            base_inferencer=base,
            review_inferencer=reviewer,
            review_prompt=self._REVIEW_TMPL,
            consensus_config=ConsensusConfig(
                max_iterations=2,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        await dual._ainfer("test request")
        self.assertGreaterEqual(len(captured_reviewer_inputs), 1)
        # Reviewer should also see the path (via review template path-aware block).
        self.assertIn(prior_file, captured_reviewer_inputs[0])


if __name__ == "__main__":
    unittest.main()
