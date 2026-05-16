"""Tests for DualInferencer output manifest generation (Fix #3).

Verifies that _finalize_response() writes an ``output_manifest.json``
file to the workspace outputs/ directory with the expected structure.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import (
    InferencerWorkspace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_inferencer(response=None):
    """Create a mock InferencerBase whose ainfer() returns ``response``."""
    inf = MagicMock()
    inf.ainfer = MagicMock(return_value=response or "mock response")
    inf.aconnect = MagicMock()
    inf.adisconnect = MagicMock()
    inf.reset_session = MagicMock()
    inf._workspace = None
    return inf


def _make_workspace(tmpdir):
    """Create an InferencerWorkspace with the standard layout."""
    ws = InferencerWorkspace(
        root=tmpdir,
        use_final_deliverables_folder=True,
    )
    ws.ensure_dirs()
    return ws


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFinalizeResponseEmitsManifest(unittest.TestCase):
    """Verify _finalize_response writes output_manifest.json with correct keys."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_finalize_response_emits_top_level_manifest(self):
        """_finalize_response() creates output_manifest.json with expected keys."""
        ws = _make_workspace(self.tmpdir)

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal A"),
            review_inferencer=_make_mock_inferencer("review"),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        # Wire the workspace and output_path onto the Dual
        dual._workspace = ws
        dual.output_path = "output.md"

        # Set up the _state dict that _finalize_response reads
        dual._state = {
            "attempt_record": {
                "iterations": [
                    {
                        "proposal": "proposal A",
                        "review": {"approved": True},
                        "counter_feedback": None,
                    }
                ],
            },
            "consensus_reached": True,
        }

        # Write a dummy artifact so the round-copy path does not fail
        artifact_path = os.path.join(ws.artifacts_dir, "round01_output.md")
        with open(artifact_path, "w") as f:
            f.write("final output content")

        # Call _finalize_response
        dual._finalize_response()

        # Assert manifest file exists
        manifest_path = ws.output_path("output_manifest.json")
        self.assertTrue(
            os.path.isfile(manifest_path),
            f"output_manifest.json should exist at {manifest_path}",
        )

        # Assert manifest contains the expected keys
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        expected_keys = {
            "source",
            "active_proposer",
            "total_iterations",
            "consensus_achieved",
            "deliverable_file",
        }
        self.assertTrue(
            expected_keys.issubset(manifest.keys()),
            f"Manifest keys {set(manifest.keys())} should contain {expected_keys}",
        )

        # Verify specific values
        self.assertEqual(manifest["source"], "DualInferencer")
        self.assertEqual(manifest["active_proposer"], "base")
        self.assertEqual(manifest["total_iterations"], 1)
        self.assertTrue(manifest["consensus_achieved"])
        self.assertEqual(manifest["deliverable_file"], "output.md")

    def test_manifest_reports_fixer_when_counter_feedback_present(self):
        """When the last iteration has counter_feedback, active_proposer is 'fixer'."""
        ws = _make_workspace(self.tmpdir)

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer("review"),
            fixer_inferencer=_make_mock_inferencer("fixed proposal"),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        dual._workspace = ws
        dual.output_path = "output.md"

        # State with counter_feedback in the last iteration (fixer ran)
        dual._state = {
            "attempt_record": {
                "iterations": [
                    {
                        "proposal": "proposal",
                        "review": {"approved": False},
                        "counter_feedback": "Fixed the issues.",
                    },
                    {
                        "proposal": "fixed proposal",
                        "review": {"approved": True},
                        "counter_feedback": "Applied all fixes.",
                    },
                ],
            },
            "consensus_reached": True,
        }

        dual._finalize_response()

        manifest_path = ws.output_path("output_manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["active_proposer"], "fixer")
        self.assertEqual(manifest["total_iterations"], 2)
        self.assertTrue(manifest["consensus_achieved"])

    def test_manifest_reports_no_consensus(self):
        """Manifest correctly reports consensus_achieved=False."""
        ws = _make_workspace(self.tmpdir)

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer("review"),
            consensus_config=ConsensusConfig(
                max_iterations=2,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        dual._workspace = ws
        dual.output_path = "output.md"

        dual._state = {
            "attempt_record": {
                "iterations": [
                    {"proposal": "v1", "review": {"approved": False}, "counter_feedback": None},
                    {"proposal": "v2", "review": {"approved": False}, "counter_feedback": None},
                ],
            },
            "consensus_reached": False,
        }

        dual._finalize_response()

        manifest_path = ws.output_path("output_manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertFalse(manifest["consensus_achieved"])
        self.assertEqual(manifest["total_iterations"], 2)

    def test_manifest_skipped_when_no_workspace(self):
        """No crash when _workspace is None (non-workspace mode)."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer("review"),
            consensus_config=ConsensusConfig(
                max_iterations=1,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        dual._workspace = None
        dual.output_path = None

        # Should not crash
        dual._finalize_response()


if __name__ == "__main__":
    unittest.main()
