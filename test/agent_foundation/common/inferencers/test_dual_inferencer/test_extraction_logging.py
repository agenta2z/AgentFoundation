

"""Unit tests verifying that DualInferencer logs post-extraction content
for ReviewResponse and FollowupResponse (not the raw pre-extraction text).

These tests use mock inferencers that return canned responses with
<Response>...</Response> tags, then verify that the `log_info` calls
for ReviewResponse/FollowupResponse receive the *extracted* content
(without tags), while RawReviewResponse/RawFixResponse receive the
original raw text (with tags).
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_inferencer(response: str):
    """Create a mock InferencerBase whose ainfer() returns `response`."""
    inf = MagicMock()
    inf.ainfer = AsyncMock(return_value=response)
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()
    return inf


def _build_review_json(approved: bool, severity: str = "COSMETIC") -> str:
    """Build a valid review JSON block inside <Response> tags."""
    review = {
        "approved": approved,
        "severity": severity,
        "issues": [],
        "reasoning": "Looks good.",
    }
    return (
        "<thinking>\nAnalyzing the proposal carefully...\n</thinking>\n\n"
        "<Response>\n"
        f"```json\n{json.dumps(review, indent=2)}\n```\n"
        "</Response>"
    )


def _build_fix_response(proposal: str) -> str:
    """Build a fixer response with <Response> tags wrapping the proposal."""
    return (
        "<thinking>\nAddressing the review feedback...\n</thinking>\n\n"
        f"<Response>\n{proposal}\n</Response>"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class DualInferencerExtractionLoggingTest(unittest.TestCase):
    """Verify that ReviewResponse and FollowupResponse are logged post-extraction."""

    def _run_async(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_review_response_logged_post_extraction(self):
        """ReviewResponse should contain extracted content (no <Response> tags)."""
        # Setup: base proposes, reviewer approves (consensus on first round)
        raw_review = _build_review_json(approved=True)
        base_inf = _make_mock_inferencer("The proposal content")
        review_inf = _make_mock_inferencer(raw_review)

        dual = DualInferencer(
            base_inferencer=base_inf,
            review_inferencer=review_inf,
            consensus_config=ConsensusConfig(
                max_iterations=1,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        # Execute: run consensus loop and capture log_info calls
        with patch.object(dual, "log_info") as mock_log_info, \
             patch.object(dual, "log_debug"):
            self._run_async(dual._ainfer("test request"))

        # Assert: find the ReviewResponse log_info call
        review_calls = [
            c for c in mock_log_info.call_args_list
            if len(c.args) >= 2 and c.args[1] == "ReviewResponse"
        ]
        self.assertEqual(len(review_calls), 1, "Expected exactly one ReviewResponse log")

        logged_content = review_calls[0].args[0]

        # The logged content should NOT contain <Response> tags (extraction happened)
        self.assertNotIn("<Response>", logged_content)
        self.assertNotIn("</Response>", logged_content)

        # The logged content should NOT contain <thinking> tags (extraction strips outer content)
        self.assertNotIn("<thinking>", logged_content)

        # The logged content SHOULD contain the JSON review block
        self.assertIn("```json", logged_content)
        self.assertIn('"approved": true', logged_content)

    def test_raw_review_response_logged_pre_extraction(self):
        """RawReviewResponse should contain full raw text including <Response> tags."""
        raw_review = _build_review_json(approved=True)
        base_inf = _make_mock_inferencer("The proposal content")
        review_inf = _make_mock_inferencer(raw_review)

        dual = DualInferencer(
            base_inferencer=base_inf,
            review_inferencer=review_inf,
            consensus_config=ConsensusConfig(
                max_iterations=1,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        with patch.object(dual, "log_debug") as mock_log_debug, \
             patch.object(dual, "log_info"):
            self._run_async(dual._ainfer("test request"))

        raw_calls = [
            c for c in mock_log_debug.call_args_list
            if len(c.args) >= 2 and c.args[1] == "RawReviewResponse"
        ]
        self.assertEqual(len(raw_calls), 1, "Expected exactly one RawReviewResponse log")

        logged_content = raw_calls[0].args[0]

        # Raw log SHOULD contain <Response> tags and <thinking> block
        self.assertIn("<Response>", logged_content)
        self.assertIn("</Response>", logged_content)
        self.assertIn("<thinking>", logged_content)

    def test_followup_response_logged_post_extraction(self):
        """FollowupResponse should contain extracted content (no <Response> tags)."""
        # Setup: reviewer rejects first, then we check FollowupResponse logging
        raw_review_reject = _build_review_json(approved=False, severity="MAJOR")
        raw_review_approve = _build_review_json(approved=True)
        improved_proposal = "Improved proposal after fixing issues"
        raw_fix = _build_fix_response(improved_proposal)

        base_inf = _make_mock_inferencer("Initial proposal")
        # Reviewer: reject first time, approve second time
        review_inf = MagicMock()
        review_inf.ainfer = AsyncMock(
            side_effect=[raw_review_reject, raw_review_approve]
        )
        review_inf.aconnect = AsyncMock()
        review_inf.adisconnect = AsyncMock()

        fixer_inf = _make_mock_inferencer(raw_fix)

        dual = DualInferencer(
            base_inferencer=base_inf,
            review_inferencer=review_inf,
            fixer_inferencer=fixer_inf,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        with patch.object(dual, "log_info") as mock_log_info, \
             patch.object(dual, "log_debug"):
            self._run_async(dual._ainfer("test request"))

        # Assert: find the FollowupResponse log_info call
        followup_calls = [
            c for c in mock_log_info.call_args_list
            if len(c.args) >= 2 and c.args[1] == "FollowupResponse"
        ]
        self.assertEqual(
            len(followup_calls), 1, "Expected exactly one FollowupResponse log"
        )

        logged_content = followup_calls[0].args[0]

        # The logged content should NOT contain <Response> tags
        self.assertNotIn("<Response>", logged_content)
        self.assertNotIn("</Response>", logged_content)
        self.assertNotIn("<thinking>", logged_content)

        # The logged content SHOULD contain the extracted proposal
        self.assertIn(improved_proposal, logged_content)

    def test_raw_fix_response_logged_pre_extraction(self):
        """RawFixResponse should contain full raw text including <Response> tags."""
        raw_review_reject = _build_review_json(approved=False, severity="MAJOR")
        raw_review_approve = _build_review_json(approved=True)
        raw_fix = _build_fix_response("Fixed proposal")

        base_inf = _make_mock_inferencer("Initial proposal")
        review_inf = MagicMock()
        review_inf.ainfer = AsyncMock(
            side_effect=[raw_review_reject, raw_review_approve]
        )
        review_inf.aconnect = AsyncMock()
        review_inf.adisconnect = AsyncMock()
        fixer_inf = _make_mock_inferencer(raw_fix)

        dual = DualInferencer(
            base_inferencer=base_inf,
            review_inferencer=review_inf,
            fixer_inferencer=fixer_inf,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        with patch.object(dual, "log_debug") as mock_log_debug, \
             patch.object(dual, "log_info"):
            self._run_async(dual._ainfer("test request"))

        raw_fix_calls = [
            c for c in mock_log_debug.call_args_list
            if len(c.args) >= 2 and c.args[1] == "RawFixResponse"
        ]
        self.assertEqual(
            len(raw_fix_calls), 1, "Expected exactly one RawFixResponse log"
        )

        logged_content = raw_fix_calls[0].args[0]

        # Raw log SHOULD still have <Response> tags
        self.assertIn("<Response>", logged_content)
        self.assertIn("</Response>", logged_content)
        self.assertIn("<thinking>", logged_content)

    def test_review_vs_raw_review_differ(self):
        """ReviewResponse and RawReviewResponse should contain different content."""
        raw_review = _build_review_json(approved=True)
        base_inf = _make_mock_inferencer("The proposal")
        review_inf = _make_mock_inferencer(raw_review)

        dual = DualInferencer(
            base_inferencer=base_inf,
            review_inferencer=review_inf,
            consensus_config=ConsensusConfig(
                max_iterations=1,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        with patch.object(dual, "log_info") as mock_log_info, \
             patch.object(dual, "log_debug") as mock_log_debug:
            self._run_async(dual._ainfer("test request"))

        # Extract logged content for both
        review_calls = [
            c for c in mock_log_info.call_args_list
            if len(c.args) >= 2 and c.args[1] == "ReviewResponse"
        ]
        raw_calls = [
            c for c in mock_log_debug.call_args_list
            if len(c.args) >= 2 and c.args[1] == "RawReviewResponse"
        ]

        self.assertEqual(len(review_calls), 1)
        self.assertEqual(len(raw_calls), 1)

        review_content = review_calls[0].args[0]
        raw_content = raw_calls[0].args[0]

        # They should be DIFFERENT (the whole point of this fix)
        self.assertNotEqual(
            review_content,
            raw_content,
            "ReviewResponse and RawReviewResponse should differ after extraction fix",
        )


if __name__ == "__main__":
    unittest.main()
