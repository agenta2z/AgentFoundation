"""Locks the AUTHOR-SNAPSHOT contract for ``<OriginalTaskInstructions>``.

``task_instructions`` is a predefined variable RE-RESOLVED per leaf, and the
research_propose variant embeds ``{{ output_path }}``. A reviewer/fixer that renders
it itself therefore binds the author's placeholders to ITSELF — which is how a review
panelist's ``<OriginalTaskInstructions>`` came to read "Conclude your deliverable
(.../panelist_00/outputs/output.md)", an instruction no author ever received.

The fix: each leaf records what IT rendered (``_capture_rendered_task_instructions``),
orchestrators report their author's snapshot down a single protocol method
(``_proposer_task_instructions``), the Dual relays it as ``prior_task_instructions``,
and the templates print that verbatim instead of re-rendering. These tests lock each
link, plus the end-state guard: even a deliberately leaky ``task_instructions`` in the
consumer's feed can no longer reach the rendered block.
"""

import unittest


def _template_manager(predefined_variables: bool):
    from agent_foundation.resources import PROMPT_TEMPLATES_ROOT
    from rich_python_utils.string_utils.formatting.jinja2_format import format_template
    from rich_python_utils.string_utils.formatting.template_manager import (
        TemplateManager,
    )

    # Shape proven to resolve real ``_variables/`` files (see
    # test_behavior_variable_injection), plus the task-tool's
    # ``enable_templated_feed`` (default.yaml:93-101) that re-renders their
    # embedded ``{{ }}`` against the feed — the behaviour under test.
    return TemplateManager(
        templates=str(PROMPT_TEMPLATES_ROOT),
        template_formatter=format_template,
        active_template_root_space="plan",
        active_template_type="main",
        predefined_variables=predefined_variables,
        default_template_key="initial",
        enable_templated_feed=True,
    )


def _author_leaf(output_path, *, declare_variable=True, separate=False):
    """A minimal templated leaf that renders the research_propose proposer contract."""
    from agent_foundation.common.inferencers.templated_inferencer_base import (
        TemplatedInferencerBase,
    )
    from attr import attrs

    @attrs(slots=False)
    class _Leaf(TemplatedInferencerBase):
        def _infer(self, inp, inference_config=None, **kw):  # pragma: no cover
            return "unused"

    leaf = _Leaf()
    leaf.template_manager = _template_manager(predefined_variables=True)
    leaf.template_root_space = "plan"
    leaf.template_key = "initial"
    leaf.template_master_version = "research_propose"
    # Two real shapes: the tool config pins the variant explicitly on the
    # review/fixer/aggregator slots (``task_instructions: research_propose``), while
    # flow leaves declare nothing and let ``__call__`` auto-discover it — in which
    # case ``_build_template_feed`` never pre-places the variable in the feed.
    if declare_variable:
        leaf.template_variables = {"task_instructions": "research_propose"}
    leaf.output_path = output_path  # absolute -> resolve_output_path returns as-is
    leaf.has_local_access = True
    if separate:
        leaf.template_extra_feed = {"separate_proposal_files": True}
    return leaf


class TestLeafCapturesItsOwnRender(unittest.TestCase):
    """1a — a leaf snapshots the contract it actually rendered, fully resolved."""

    AUTHOR_PATH = "/ws/author/outputs/output.md"

    def _render_and_snapshot(self, **kwargs):
        leaf = _author_leaf(self.AUTHOR_PATH, **kwargs)
        leaf._render_prompt("the user request")
        return leaf._last_rendered_task_instructions

    def test_snapshot_is_the_research_propose_contract(self):
        snap = self._render_and_snapshot()
        self.assertTrue(snap, "leaf must record what it rendered")
        self.assertIn("Structured Proposal Index", snap)
        # The proposer contract, NOT the aggregator's merge brief.
        self.assertNotIn("aggregating/integrating", snap)

    def test_snapshot_carries_the_AUTHORS_path(self):
        snap = self._render_and_snapshot()
        self.assertIn(self.AUTHOR_PATH, snap)

    def test_snapshot_is_brace_free(self):
        # The load-bearing invariant: any surviving placeholder would be re-rendered
        # against the CONSUMER's feed downstream — exactly the leak being fixed.
        snap = self._render_and_snapshot()
        self.assertNotIn("{{", snap)
        self.assertNotIn("{%", snap)

    def test_snapshot_works_without_declared_template_variables(self):
        # Fallback path (leaf relies on __call__ auto-discovery).
        snap = self._render_and_snapshot(declare_variable=False)
        self.assertIn("Structured Proposal Index", snap)
        self.assertIn(self.AUTHOR_PATH, snap)
        self.assertNotIn("{{", snap)

    def test_snapshot_reflects_the_authors_separate_proposal_files(self):
        # The flag is the author's, not the consumer's: the per-proposal-file section
        # is present only when the AUTHOR rendered with it on.
        on = self._render_and_snapshot(separate=True)
        off = self._render_and_snapshot(separate=False)
        self.assertIn("Per-Proposal File Output", on)
        self.assertNotIn("Per-Proposal File Output", off)

    def test_leaf_reports_its_snapshot_through_the_protocol(self):
        leaf = _author_leaf(self.AUTHOR_PATH)
        self.assertEqual(leaf._proposer_task_instructions(), "")  # nothing rendered yet
        leaf._render_prompt("the user request")
        self.assertEqual(
            leaf._proposer_task_instructions(), leaf._last_rendered_task_instructions
        )


class TestProposerProtocolDelegation(unittest.TestCase):
    """1b — every node type reports the contract from its INPUT side."""

    def test_base_default_is_empty(self):
        from agent_foundation.common.inferencers.inferencer_base import InferencerBase

        self.assertEqual(InferencerBase._proposer_task_instructions(object()), "")

    def test_multiflow_prefers_winner_then_any_flow(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
            MultiFlowInferencer,
        )

        class _Leaf:
            def __init__(self, snap):
                self._snap = snap

            def _proposer_task_instructions(self):
                return self._snap

        winner, other = _Leaf("WINNER_CONTRACT"), _Leaf("OTHER_CONTRACT")

        class _FakeMFI:
            flow_configs = [{"initial_inferencer": other}]
            _winner = None

            def get_winner_inferencer(self):
                return self._winner

            _proposer_task_instructions = (
                MultiFlowInferencer._proposer_task_instructions
            )

        mfi = _FakeMFI()
        self.assertEqual(mfi._proposer_task_instructions(), "OTHER_CONTRACT")
        mfi._winner = winner
        self.assertEqual(mfi._proposer_task_instructions(), "WINNER_CONTRACT")

    def test_multiflow_never_consults_an_aggregator(self):
        # The aggregator is the OUTPUT side; it renders the merge brief, so it must
        # never be a source for the reviewer's task-contract reference.
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
            MultiFlowInferencer,
        )

        class _Aggregator:
            def _proposer_task_instructions(self):  # pragma: no cover
                raise AssertionError("aggregator must not be consulted")

        class _FakeMFI:
            flow_configs = []
            aggregator_inferencer = _Aggregator()
            multi_flow_aggregator_inferencer = _Aggregator()

            def get_winner_inferencer(self):
                return None

            _proposer_task_instructions = (
                MultiFlowInferencer._proposer_task_instructions
            )

        self.assertEqual(_FakeMFI()._proposer_task_instructions(), "")

    def test_dual_prefers_stored_state_over_live_walk(self):
        # By review/fix time the winning flow may have re-rendered as the fixer and
        # overwritten its own snapshot — the value captured at propose-completion wins.
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )

        class _Base:
            def _proposer_task_instructions(self):
                return "LIVE_FIXER_CONTEXT"

        class _FakeDual:
            base_inferencer = _Base()
            _state = {"prior_task_instructions": "CAPTURED_AT_PROPOSE"}
            _prior_task_instructions = DualInferencer._prior_task_instructions
            _proposer_task_instructions = DualInferencer._proposer_task_instructions

        dual = _FakeDual()
        self.assertEqual(dual._proposer_task_instructions(), "CAPTURED_AT_PROPOSE")
        dual._state = {}
        self.assertEqual(dual._proposer_task_instructions(), "LIVE_FIXER_CONTEXT")

    def test_dual_prior_is_empty_without_state(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )

        class _FakeDual:
            _state = None
            _prior_task_instructions = DualInferencer._prior_task_instructions

        self.assertEqual(_FakeDual()._prior_task_instructions(), "")

    def test_bta_reports_harvested_worker_contract(self):
        # Workers are factory-ephemeral, so the contract is published UP as they
        # finish rather than reached by a static drill.
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )

        class _FakeBTA:
            _worker_task_instructions = ""
            _proposer_task_instructions = (
                BreakdownThenAggregateInferencer._proposer_task_instructions
            )

        bta = _FakeBTA()
        self.assertEqual(bta._proposer_task_instructions(), "")
        bta._worker_task_instructions = "WORKER_CONTRACT"
        self.assertEqual(bta._proposer_task_instructions(), "WORKER_CONTRACT")


class TestReviewTemplateConsumesSnapshotOnly(unittest.TestCase):
    """1e/1f — the rendered block comes from the snapshot, and ONLY from it."""

    AUTHOR = "AUTHOR_CONTRACT_MARKER at /ws/author/outputs/output.md"
    REVIEWER_PATH = "/ws/panelist_00/outputs/output.md"

    def _render(self, template_key, **extra):
        tm = _template_manager(predefined_variables=False)
        feed = {
            "main_response": "the artifact under review",
            "context": {"user_request_with_task_preamble": "REQUEST_BODY"},
            # A deliberately LEAKY author-scoped variable in the consumer's feed:
            # the templates must no longer read it.
            "task_instructions": f"LEAKY_MARKER at {self.REVIEWER_PATH}",
            "output_path": self.REVIEWER_PATH,
            **extra,
        }
        return tm(template_key, active_template_root_space="plan", **feed)

    def test_review_prints_the_snapshot_verbatim(self):
        out = self._render("review", prior_task_instructions=self.AUTHOR)
        self.assertIn("<OriginalTaskInstructions>", out)
        self.assertIn(self.AUTHOR, out)

    def test_review_ignores_a_leaky_task_instructions(self):
        out = self._render("review", prior_task_instructions=self.AUTHOR)
        self.assertNotIn("LEAKY_MARKER", out)
        self.assertNotIn(self.REVIEWER_PATH, out)

    def test_review_omits_the_block_when_no_snapshot(self):
        # Graceful degradation: no author resolved -> no reference block at all,
        # rather than a re-render bound to the reviewer.
        out = self._render("review", prior_task_instructions="")
        self.assertNotIn("<OriginalTaskInstructions>", out)
        self.assertNotIn("LEAKY_MARKER", out)

    def test_followup_prints_snapshot_but_keeps_its_own_write_target(self):
        out = self._render("followup", prior_task_instructions=self.AUTHOR)
        self.assertIn(self.AUTHOR, out)
        self.assertNotIn("LEAKY_MARKER", out)
        # The fixer IS an author: its own Output Requirements still name its path.
        self.assertIn(self.REVIEWER_PATH, out)


if __name__ == "__main__":
    unittest.main()
