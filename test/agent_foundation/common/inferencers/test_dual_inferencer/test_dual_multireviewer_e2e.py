"""§3 Part B — END-TO-END: num_reviewers>1 / reviewers=[...] drive the REAL
DualInferencer._step_review_impl (propose -> review), issuing k reviewer calls and
merging the panel. Uses the leaf-render path so no Jinja review template is needed.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)


def _mk(resp):
    inf = MagicMock()
    inf.ainfer = AsyncMock(return_value=resp)
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()
    inf.reset_session = MagicMock()
    return inf


def _rj(approved, loc="f:1"):
    review = {
        "approved": approved,
        "severity": "COSMETIC",
        "issues": []
        if approved
        else [{"severity": "COSMETIC", "category": "x", "description": "d", "location": loc}],
        "reasoning": "r",
    }
    return f"```json\n{json.dumps(review, indent=2)}\n```"


def _run_review_step(dual, review_responses):
    """Drive the REAL propose+review steps once; return after the review step
    (swallowing the consensus WorkflowAborted)."""
    dual._leaf_can_self_render = lambda inf: True  # leaf-render -> no template needed

    async def _go():
        st = {
            "inference_input": "request",
            "total_iterations": 0,
            "attempt_record": {"attempt": 1, "iterations": []},
        }
        dual._state = st
        await dual._step_propose_impl("request", st)
        try:
            await dual._step_review_impl("request", st)
        except Exception:  # WorkflowAborted on consensus is expected
            pass

    asyncio.run(_go())


class MultiReviewerE2E(unittest.TestCase):
    def test_num_reviewers_2_issues_two_real_reviewer_calls(self):
        """The REAL _step_review_impl runs the reviewer TWICE for num_reviewers=2."""
        review = _mk(_rj(True))
        dual = DualInferencer(
            base_inferencer=_mk("proposal"),
            review_inferencer=review,
            num_reviewers=2,
            consensus_config=ConsensusConfig(
                max_iterations=3, max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )
        _run_review_step(dual, None)
        self.assertEqual(review.ainfer.call_count, 2)

    def test_reviewers_list_invokes_each_independent_panelist_once(self):
        """reviewers=[r1] + review_inferencer=r0 -> the real review step calls r0 once
        (panelist 0) and the INDEPENDENT r1 once (panelist 1)."""
        r0, r1 = _mk(_rj(True)), _mk(_rj(True))
        dual = DualInferencer(
            base_inferencer=_mk("proposal"),
            review_inferencer=r0,
            reviewers=[r1],
            consensus_config=ConsensusConfig(
                max_iterations=3, max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )
        _run_review_step(dual, None)
        self.assertEqual(r0.ainfer.call_count, 1)
        self.assertEqual(r1.ainfer.call_count, 1)

    def test_single_reviewer_default_is_byte_identical_one_call(self):
        """num_reviewers=1 (default) -> exactly one reviewer call (no panel)."""
        review = _mk(_rj(True))
        dual = DualInferencer(
            base_inferencer=_mk("proposal"),
            review_inferencer=review,
            consensus_config=ConsensusConfig(
                max_iterations=3, max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )
        _run_review_step(dual, None)
        self.assertEqual(review.ainfer.call_count, 1)


if __name__ == "__main__":
    unittest.main()
