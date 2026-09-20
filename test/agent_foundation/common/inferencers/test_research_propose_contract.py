"""Locks the research_propose OUTPUT CONTRACT at the template-render layer (F5) so it
cannot silently regress.

Renders the REAL ``plan/main/initial.jinja2`` through the REAL ``TemplateManager``,
mirroring ``TemplatedInferencerBase._render_prompt`` and the task-tool
``_template_manager`` config (default.yaml:93-101). Asserts:

1. A FLOW-LEAF render (``master_version=research_propose`` — the value injected into
   EVERY node via ``overrides["_template_master_version"]``, executor.py:1333) puts the
   ``proposal_index`` instruction IN the deliverable-file task section (it "rides IN the
   deliverable file") — NOT in the ``<Response>`` stdout summary. This is the channel-
   split contract fix; if it regresses, flow leaves stop writing the fence into the file
   the aggregator merges and ``executor.py`` parses.
2. No variable-file control syntax (``{% %}`` / ``{# #}``) leaks into the rendered
   prompt (the raw-injection class of bug — variable files are regex-composed; the
   ``enable_templated_feed`` re-render must process their control blocks).
3. A PLAIN run (no research_propose master) carries NO proposal_index.
"""

import unittest


def _render(master_version=None, separate=False, has_local=True):
    from agent_foundation.resources import PROMPT_TEMPLATES_ROOT
    from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
        TemplateManager,
    )

    # Mirror the task-tool _template_manager config (default.yaml:93-101).
    tm = TemplateManager(
        templates=[str(PROMPT_TEMPLATES_ROOT)],
        active_template_type="main",
        predefined_variables=False,
        default_template_key="initial",
        enable_templated_feed=True,
    )
    feed = {
        "output_path": "/ws/outputs/output.md",
        "has_local_access": has_local,
        "context": {"user_request_with_task_preamble": "REQUEST_BODY"},
    }
    if separate:
        feed["separate_proposal_files"] = True
    kwargs = {"active_template_root_space": "plan", **feed}
    if master_version:
        kwargs["master_version"] = master_version
    return tm("initial", **kwargs)


class TestResearchProposeContract(unittest.TestCase):
    def test_flow_leaf_proposal_index_rides_in_file(self):
        out = _render(master_version="research_propose")
        self.assertIn("Structured Proposal Index", out)
        self.assertIn("```json proposal_index", out)
        self.assertIn("rides IN the deliverable file", out)
        # Placement: the index instruction lives in the "## Your Task" file-deliverable
        # section, BEFORE the "## Response Format" stdout-summary section.
        t = out.find("## Your Task")
        p = out.find("Structured Proposal Index")
        r = out.find("## Response Format")
        self.assertGreaterEqual(t, 0, "no '## Your Task' section rendered")
        self.assertLess(t, p, "proposal_index must be inside the task section")
        self.assertTrue(r == -1 or p < r, "proposal_index must precede Response Format")

    def test_flow_leaf_no_control_syntax_leak(self):
        for sep in (False, True):
            out = _render(master_version="research_propose", separate=sep)
            self.assertNotIn("{%", out, f"Jinja control leaked (separate={sep})")
            self.assertNotIn("{#", out, f"Jinja comment leaked (separate={sep})")

    def test_separate_proposal_files_branch_processed(self):
        # With separate_proposal_files=True the {% if %} branch must be PROCESSED
        # (its content included), not leaked — proves the re-render handles the flat
        # variable file's control syntax.
        out = _render(master_version="research_propose", separate=True)
        self.assertIn("Per-Proposal File Output", out)
        self.assertNotIn("{%", out)

    def test_plain_run_has_no_proposal_index(self):
        out = _render(master_version=None)
        self.assertNotIn("Structured Proposal Index", out)
        self.assertNotIn("```json proposal_index", out)
        self.assertNotIn("{%", out)
        self.assertNotIn("{#", out)


if __name__ == "__main__":
    unittest.main()
