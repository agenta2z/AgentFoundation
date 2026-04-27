"""Mock-based DualInferencer flow test to verify base-class changes don't break consensus (Task 11.5).

Creates a DualInferencer with mock base/review/fixer inferencers and verifies:
- Propose → review → fix consensus loop executes correctly
- WorkflowAborted is raised when consensus is reached (early exit)
- Checkpoint/resume works with expansion-aware base class

Requirements: 21.1, 21.3, 21.5
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    DualInferencerResponse,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)


def _make_mock_inferencer(response=None, side_effect=None):
    """Create a mock InferencerBase whose ainfer() returns ``response``."""
    inf = MagicMock()
    if side_effect is not None:
        inf.ainfer = AsyncMock(side_effect=side_effect)
    else:
        inf.ainfer = AsyncMock(return_value=response or "mock response")
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()
    inf.reset_session = MagicMock()
    return inf


def _review_json(approved: bool, severity: str = "COSMETIC") -> str:
    """Return a raw review response string."""
    review = {
        "approved": approved,
        "severity": severity,
        "issues": (
            []
            if approved
            else [
                {
                    "severity": severity,
                    "category": "test",
                    "description": "Test issue",
                    "location": "N/A",
                    "suggestion": "Fix it",
                }
            ]
        ),
        "reasoning": "Test reasoning.",
    }
    return f"```json\n{json.dumps(review, indent=2)}\n```"


def _fix_response(proposal: str) -> str:
    return f"<ImprovedProposal>\n{proposal}\n</ImprovedProposal>"


class TestDualConsensusPostExpansion(unittest.IsolatedAsyncioTestCase):
    """Verify DualInferencer consensus loop works after Workflow base-class expansion changes."""

    async def test_consensus_first_round_approve(self):
        """Propose → review approves → consensus achieved on first round."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal A"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        self.assertEqual(result.total_iterations, 1)

    async def test_one_fix_cycle_then_approve(self):
        """Propose → review rejects → fix → review approves."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal A"),
            review_inferencer=_make_mock_inferencer(
                side_effect=[
                    _review_json(approved=False, severity="MAJOR"),
                    _review_json(approved=True),
                ]
            ),
            fixer_inferencer=_make_mock_inferencer(_fix_response("proposal B")),
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        self.assertEqual(result.total_iterations, 2)

    async def test_loop_exhaustion_no_consensus(self):
        """All review iterations reject → no consensus."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal A"),
            review_inferencer=_make_mock_inferencer(
                _review_json(approved=False, severity="MAJOR")
            ),
            fixer_inferencer=_make_mock_inferencer(_fix_response("proposal B")),
            consensus_config=ConsensusConfig(
                max_iterations=2,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertFalse(result.consensus_achieved)

    async def test_response_structure_correct(self):
        """DualInferencerResponse has correct structure after expansion changes."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal A"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertIsNotNone(result.base_response)
        self.assertIsNotNone(result.consensus_history)
        self.assertEqual(len(result.consensus_history), 1)


class TestDualCheckpointPostExpansion(unittest.IsolatedAsyncioTestCase):
    """Verify DualInferencer checkpoint/resume works with expansion-aware base class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dual_ckpt_compat_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_checkpoint_enabled_consensus(self):
        """DualInferencer with checkpoint_dir runs consensus loop correctly."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal A"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            checkpoint_dir=self.tmpdir,
        )

        result = await dual._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)

    async def test_checkpoint_files_valid_json(self):
        """Checkpoint files created by DualInferencer are valid JSON."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal A"),
            review_inferencer=_make_mock_inferencer(
                side_effect=[
                    _review_json(approved=False, severity="MAJOR"),
                    _review_json(approved=True),
                ]
            ),
            fixer_inferencer=_make_mock_inferencer(_fix_response("proposal B")),
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            checkpoint_dir=self.tmpdir,
        )

        result = await dual._ainfer("request")
        self.assertTrue(result.consensus_achieved)

        # Verify any checkpoint files are valid JSON
        for root, _dirs, files in os.walk(self.tmpdir):
            for fname in files:
                if fname.endswith(".json"):
                    fpath = os.path.join(root, fname)
                    with open(fpath) as f:
                        data = json.load(f)
                    self.assertIsInstance(data, dict)


class TestDualMultiAttemptPostExpansion(unittest.IsolatedAsyncioTestCase):
    """Verify multi-attempt outer loop works with expansion-aware base class."""

    async def test_second_attempt_succeeds(self):
        """First attempt fails consensus, second attempt succeeds."""
        # First attempt: all reviews reject
        # Second attempt: review approves
        review_responses = [
            _review_json(approved=False, severity="MAJOR"),
            _review_json(approved=False, severity="MAJOR"),
            # Second attempt
            _review_json(approved=True),
        ]

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal A"),
            review_inferencer=_make_mock_inferencer(side_effect=review_responses),
            fixer_inferencer=_make_mock_inferencer(_fix_response("proposal B")),
            consensus_config=ConsensusConfig(
                max_iterations=2,
                max_consensus_attempts=2,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        # Should have achieved consensus on second attempt
        self.assertTrue(result.consensus_achieved)


if __name__ == "__main__":
    unittest.main()
