"""DualInferencer consensus tests — Layer 1, R2.

Coverage focus: GAPS not covered by existing test files.

Existing coverage (test_dual_inferencer/test_dual_post_expansion_compat.py):
- consensus first round approve, one fix cycle then approve, loop exhaustion,
  response structure, checkpoint enabled/valid JSON, multi-attempt 2nd succeeds

Genuine gaps filled here:
- 2-agent mode (no fixer_inferencer → base used as fixer) (R2.5)
- Counter-feedback content reaches next review iteration's prompt (R2.6)
- Severity threshold determines exit condition (R2.2, R2.3)
"""

import json
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


def _review_json(approved: bool, severity: str = "COSMETIC", issues=None) -> str:
    """Standard review-response JSON envelope."""
    if issues is None:
        issues = ([] if approved else [
            {
                "severity": severity,
                "category": "test",
                "description": "Test issue",
                "location": "N/A",
                "suggestion": "Fix it",
            }
        ])
    review = {
        "approved": approved,
        "severity": severity,
        "issues": issues,
        "reasoning": "Test reasoning.",
    }
    return f"```json\n{json.dumps(review, indent=2)}\n```"


# ---------------------------------------------------------------------------
# R2.5: 2-agent mode — no fixer_inferencer means base is used for both
# ---------------------------------------------------------------------------


class TestTwoAgentMode(unittest.IsolatedAsyncioTestCase):
    """Validates: R2.5 — 2-agent mode (no fixer)."""

    async def test_two_agent_mode_uses_base_as_fixer(self):
        """Without fixer_inferencer, base_inferencer is invoked for both
        propose and fix steps."""
        base = _make_mock_inferencer(side_effect=[
            "initial proposal",          # propose
            "fixed proposal",            # fix (base used as fixer)
        ])
        dual = DualInferencer(
            base_inferencer=base,
            review_inferencer=_make_mock_inferencer(side_effect=[
                _review_json(approved=False, severity="MAJOR"),  # 1st review: reject
                _review_json(approved=True),                     # 2nd review: approve
            ]),
            # NB: no fixer_inferencer
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        # base.ainfer was called twice (once propose, once fix)
        self.assertEqual(base.ainfer.call_count, 2)


# ---------------------------------------------------------------------------
# R2.2 / R2.3: Severity threshold determines exit
# ---------------------------------------------------------------------------


class TestSeverityThreshold(unittest.IsolatedAsyncioTestCase):
    """Validates: R2.2 (≤ threshold → exit) / R2.3 (> threshold → fix)."""

    async def test_severity_at_threshold_exits_without_fixer(self):
        """When review severity equals consensus_threshold, the loop exits
        and fixer is never invoked."""
        fixer = _make_mock_inferencer("should not be called")
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(
                # NOT approved but severity = COSMETIC = threshold → exits
                _review_json(approved=False, severity="COSMETIC")
            ),
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        # Fixer should NOT have been invoked
        fixer.ainfer.assert_not_called()

    async def test_severity_above_threshold_invokes_fixer(self):
        """When review severity exceeds consensus_threshold, fixer runs."""
        fixer = _make_mock_inferencer("fixed")
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(side_effect=[
                _review_json(approved=False, severity="MAJOR"),  # MAJOR > COSMETIC → fix
                _review_json(approved=True),
            ]),
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        # Fixer SHOULD have been invoked once
        self.assertEqual(fixer.ainfer.call_count, 1)


# ---------------------------------------------------------------------------
# R2.6: Counter-feedback propagation
# ---------------------------------------------------------------------------


class TestCounterFeedbackPropagation(unittest.IsolatedAsyncioTestCase):
    """Validates: R2.6 — fixer's counter_feedback reaches next review iteration."""

    async def test_counter_feedback_in_next_review_input(self):
        """Counter-feedback emitted by fixer in iteration N is included in the
        review prompt for iteration N+1.

        Approach: install a custom review_inferencer that captures every
        review prompt; fixer emits a unique marker string in counter_feedback;
        verify the marker appears in the second review's prompt.
        """
        captured_review_inputs = []

        async def review_capture(prompt, *args, **kwargs):
            captured_review_inputs.append(str(prompt))
            # First review: reject; second: approve
            if len(captured_review_inputs) == 1:
                return _review_json(approved=False, severity="MAJOR")
            return _review_json(approved=True)

        review = MagicMock()
        review.ainfer = AsyncMock(side_effect=review_capture)
        review.aconnect = AsyncMock()
        review.adisconnect = AsyncMock()
        review.reset_session = MagicMock()

        # Fixer emits a unique marker in its counter-feedback envelope
        UNIQUE_MARKER = "COUNTER-FEEDBACK-MARKER-XYZ-12345"
        fixer_response = json.dumps({
            "items": [{"reasoning": UNIQUE_MARKER}],
            "summary": "see items",
        })

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("initial proposal"),
            review_inferencer=review,
            fixer_inferencer=_make_mock_inferencer(fixer_response),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                consensus_threshold=Severity.COSMETIC,
                enable_counter_feedback=True,
            ),
        )

        result = await dual._ainfer("request")

        # Two review calls happened
        self.assertEqual(len(captured_review_inputs), 2)
        # Second review prompt should contain the counter-feedback marker
        self.assertIn(
            UNIQUE_MARKER, captured_review_inputs[1],
            "Counter-feedback content from fixer should appear in next review's prompt",
        )


# ---------------------------------------------------------------------------
# Issue-level severity blocking (E-03 bug fix)
# ---------------------------------------------------------------------------


class TestIssueLevelSeverityBlocking(unittest.IsolatedAsyncioTestCase):
    """approved=True must be overridden when any issue exceeds threshold."""

    async def test_approved_with_major_issue_blocks_consensus(self):
        """approved=True but a MAJOR issue exists → consensus blocked, fixer runs."""
        fixer = _make_mock_inferencer("fixed proposal")
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(side_effect=[
                _review_json(
                    approved=True, severity="COSMETIC",
                    issues=[{
                        "severity": "MAJOR",
                        "category": "logic",
                        "description": "Wrong classification",
                        "location": "line 10",
                        "suggestion": "Reclassify",
                    }],
                ),
                _review_json(approved=True),
            ]),
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        self.assertEqual(fixer.ainfer.call_count, 1)

    async def test_approved_with_cosmetic_issues_reaches_consensus(self):
        """approved=True + only COSMETIC issues → consensus reached."""
        fixer = _make_mock_inferencer("should not run")
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(
                _review_json(
                    approved=True, severity="COSMETIC",
                    issues=[{
                        "severity": "COSMETIC",
                        "category": "style",
                        "description": "Typo",
                        "location": "line 5",
                        "suggestion": "Fix typo",
                    }],
                ),
            ),
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        fixer.ainfer.assert_not_called()

    async def test_approved_no_issues_reaches_consensus(self):
        """approved=True + empty issues list → consensus reached."""
        fixer = _make_mock_inferencer("should not run")
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(
                _review_json(approved=True, issues=[]),
            ),
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        fixer.ainfer.assert_not_called()


# ---------------------------------------------------------------------------
# Generic severity_levels
# ---------------------------------------------------------------------------


class TestGenericSeverityLevels(unittest.IsolatedAsyncioTestCase):
    """severity_levels allows custom severity scales."""

    async def test_numeric_severity_levels(self):
        """Numeric-string severity scale: severity ≤ threshold → consensus."""
        fixer = _make_mock_inferencer("fixed")
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(
                _review_json(approved=False, severity="1", issues=[]),
            ),
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(
                severity_levels=("0", "1", "2", "3", "4", "5"),
                consensus_threshold="2",
                max_iterations=3,
            ),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        fixer.ainfer.assert_not_called()

    async def test_numeric_severity_above_threshold_blocks(self):
        """Numeric-string severity scale: severity > threshold → fixer runs."""
        fixer = _make_mock_inferencer("fixed")
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(side_effect=[
                _review_json(approved=False, severity="4", issues=[]),
                _review_json(approved=True, issues=[]),
            ]),
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(
                severity_levels=("0", "1", "2", "3", "4", "5"),
                consensus_threshold="2",
                max_iterations=3,
            ),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        self.assertEqual(fixer.ainfer.call_count, 1)

    async def test_custom_string_severity_levels(self):
        """Custom string labels: LOW/MEDIUM/HIGH scale."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(
                _review_json(approved=False, severity="LOW", issues=[]),
            ),
            consensus_config=ConsensusConfig(
                severity_levels=('LOW', 'MEDIUM', 'HIGH'),
                consensus_threshold='LOW',
                max_iterations=3,
            ),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)

    async def test_unknown_severity_blocks_consensus(self):
        """Issue with severity not in severity_levels blocks consensus."""
        fixer = _make_mock_inferencer("fixed")
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(side_effect=[
                _review_json(
                    approved=True, severity="COSMETIC",
                    issues=[{
                        "severity": "UNKNOWN_LEVEL",
                        "category": "test",
                        "description": "Test",
                        "location": "N/A",
                        "suggestion": "N/A",
                    }],
                ),
                _review_json(approved=True, issues=[]),
            ]),
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(max_iterations=3),
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        self.assertEqual(fixer.ainfer.call_count, 1)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestSeverityBackwardCompat(unittest.TestCase):
    """Existing Severity enum usage must continue working."""

    def test_severity_enum_still_works_as_threshold(self):
        """ConsensusConfig(consensus_threshold=Severity.COSMETIC) still works."""
        config = ConsensusConfig(consensus_threshold=Severity.COSMETIC)
        self.assertEqual(config.consensus_threshold, "COSMETIC")
        self.assertIn(config.consensus_threshold, config.severity_levels)

    def test_default_severity_levels(self):
        """Default severity_levels matches the standard 5-level scale."""
        config = ConsensusConfig()
        self.assertEqual(
            config.severity_levels,
            ('NONE', 'COSMETIC', 'MINOR', 'MAJOR', 'CRITICAL'),
        )

    def test_invalid_threshold_raises(self):
        """consensus_threshold not in severity_levels raises ValueError."""
        with self.assertRaises(ValueError):
            ConsensusConfig(consensus_threshold='INVALID')


if __name__ == "__main__":
    unittest.main()
