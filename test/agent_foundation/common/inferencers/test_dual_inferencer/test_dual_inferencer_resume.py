

"""Comprehensive tests for DualInferencer checkpoint/resume from different
workspace conditions.

Organized into tiers:

Tier 1 — Backward Compatibility (checkpoint disabled):
  No checkpoint files, no resume.  Verifies the Workflow inheritance
  does not change any observable behavior.

Tier 2 — Checkpoint-Enabled Normal Completion:
  Full runs with checkpointing.  Verifies files are created and
  results match non-checkpoint runs.

Tier 3 — Resume from Crash at Different Steps:
  Simulates crashes at propose/review/fix, then resumes.  Verifies
  the correct sub-inferencer calls are skipped or replayed.

Tier 4 — State Restoration Correctness:
  Verifies counters (total_iterations, iteration), proposal text,
  and consensus flag survive resume.

Tier 5 — Multi-Attempt:
  max_consensus_attempts > 1 with/without checkpointing.

Tier 6 — Edge Cases:
  Corrupted checkpoints, missing dirs, empty checkpoint_dir.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Return a raw review response string (no <Response> tags)."""
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


def _count_json_files(directory: str) -> int:
    """Recursively count .json files under *directory*."""
    count = 0
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".json"):
                count += 1
    return count


def _count_pkl_files(directory: str) -> int:
    """Recursively count .pkl files under *directory*."""
    count = 0
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".pkl"):
                count += 1
    return count


# =====================================================================
# Tier 1 — Backward Compatibility (checkpoint disabled)
# =====================================================================


class Tier1_BackwardCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    """enable_checkpoint=False (default).  No files written, identical API."""

    async def test_consensus_first_round(self):
        """Propose → review approves → WorkflowAborted path → consensus."""
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
        self.assertEqual(len(result.consensus_history), 1)
        self.assertTrue(result.consensus_history[0].consensus_reached)
        self.assertEqual(len(result.consensus_history[0].iterations), 1)

    async def test_one_fix_cycle(self):
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

        self.assertTrue(result.consensus_achieved)
        self.assertEqual(result.total_iterations, 2)
        attempt = result.consensus_history[0]
        self.assertEqual(len(attempt.iterations), 2)
        self.assertFalse(attempt.iterations[0].consensus_reached)
        self.assertTrue(attempt.iterations[1].consensus_reached)

    async def test_loop_exhaustion(self):
        """All review iterations reject → no consensus."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(
                _review_json(approved=False, severity="CRITICAL")
            ),
            fixer_inferencer=_make_mock_inferencer(_fix_response("still bad")),
            consensus_config=ConsensusConfig(
                max_iterations=2,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertFalse(result.consensus_achieved)
        # 2 review rounds: initial review + one after fix
        self.assertEqual(result.total_iterations, 2)

    async def test_run_and_arun_blocked(self):
        """WorkNodeBase.run() and arun() must raise NotImplementedError."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer(),
            review_inferencer=_make_mock_inferencer(),
        )
        with self.assertRaises(NotImplementedError):
            dual.run("x")
        with self.assertRaises(NotImplementedError):
            await dual.arun("x")

    async def test_two_agent_mode(self):
        """fixer_inferencer=None → base_inferencer also fixes."""
        base = _make_mock_inferencer(
            side_effect=[
                "proposal A",  # propose
                _fix_response("proposal B"),  # fix (base doubles as fixer)
            ]
        )
        dual = DualInferencer(
            base_inferencer=base,
            review_inferencer=_make_mock_inferencer(
                side_effect=[
                    _review_json(approved=False, severity="MAJOR"),
                    _review_json(approved=True),
                ]
            ),
            # fixer_inferencer=None → defaults to base_inferencer
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")
        self.assertTrue(result.consensus_achieved)
        # base_inferencer.ainfer was called twice (propose + fix)
        self.assertEqual(base.ainfer.call_count, 2)


# =====================================================================
# Tier 2 — Checkpoint-Enabled Normal Completion
# =====================================================================


class Tier2_CheckpointNormalCompletionTest(unittest.IsolatedAsyncioTestCase):
    """enable_checkpoint=True, no crashes.  Files written, results correct."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_checkpoint_creates_files(self):
        """Checkpoint-enabled first-round consensus creates attempt_01/ with JSON files."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        result = await dual._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        attempt_dir = os.path.join(self.tmpdir, "attempt_01")
        self.assertTrue(os.path.isdir(attempt_dir))
        json_count = _count_json_files(attempt_dir)
        self.assertGreater(json_count, 0, "Expected at least one .json checkpoint file")
        self.assertEqual(
            _count_pkl_files(attempt_dir),
            0,
            "No .pkl files should be created — DualInferencer uses JSON checkpoints",
        )

    async def test_checkpoint_files_are_valid_json(self):
        """All checkpoint files contain valid, human-readable JSON."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        await dual._ainfer("request")

        attempt_dir = os.path.join(self.tmpdir, "attempt_01")
        for root, _dirs, files in os.walk(attempt_dir):
            for fname in files:
                if fname.endswith(".json"):
                    fpath = os.path.join(root, fname)
                    with open(fpath, encoding="utf-8") as f:
                        content = f.read()
                    parsed = json.loads(content)
                    self.assertIsNotNone(
                        parsed,
                        f"JSON checkpoint {fname} should parse to a non-None value",
                    )

    async def test_checkpoint_results_match_no_checkpoint(self):
        """Checkpoint-enabled run produces same outcome as non-checkpoint run."""
        review_seq = [
            _review_json(approved=False, severity="MAJOR"),
            _review_json(approved=True),
        ]

        # Without checkpoint
        dual_off = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(side_effect=list(review_seq)),
            fixer_inferencer=_make_mock_inferencer(_fix_response("fixed")),
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
        )
        r_off = await dual_off._ainfer("request")

        # With checkpoint
        dual_on = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(side_effect=list(review_seq)),
            fixer_inferencer=_make_mock_inferencer(_fix_response("fixed")),
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )
        r_on = await dual_on._ainfer("request")

        self.assertEqual(r_off.consensus_achieved, r_on.consensus_achieved)
        self.assertEqual(r_off.total_iterations, r_on.total_iterations)
        self.assertEqual(
            len(r_off.consensus_history), len(r_on.consensus_history)
        )
        self.assertEqual(
            r_off.consensus_history[0].consensus_reached,
            r_on.consensus_history[0].consensus_reached,
        )

    async def test_checkpoint_dir_auto_created(self):
        """checkpoint_dir is created automatically if it doesn't exist."""
        nested = os.path.join(self.tmpdir, "deep", "nested", "dir")
        self.assertFalse(os.path.exists(nested))

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=1,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=nested,
        )

        result = await dual._ainfer("request")
        self.assertTrue(result.consensus_achieved)
        self.assertTrue(os.path.isdir(nested))


# =====================================================================
# Tier 3 — Resume from Crash at Different Steps
# =====================================================================


class Tier3_ResumeFromCrashTest(unittest.IsolatedAsyncioTestCase):
    """Simulate crashes at various steps, then resume.

    Checkpoint save order in Workflow._arun for steps [propose, review, fix]:

    Step 0 (propose) completes →
      save propose___seq1.json
      save __wf_checkpoint__ {next_step_index=1, state=...}
    Step 1 (review) completes →
      save review___seq2.json
      save __wf_checkpoint__ {next_step_index=2, state=...}
    Step 2 (fix) completes →
      save fix___seq3.json
      loop_back_to="review" → save __wf_checkpoint__ {next_step_index=1, state=...}
      (or if loop exhausted → advance to step 3 → loop exits)

    If a step CRASHES, its result is NOT saved and the checkpoint from
    the PREVIOUS step is the last persisted state.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_crash_at_review_resume_skips_propose(self):
        """Crash during review step → resume skips propose, runs review."""
        # --- Run 1: propose succeeds, review crashes ---
        base_run1 = _make_mock_inferencer("proposal v1")
        review_run1 = _make_mock_inferencer(
            side_effect=RuntimeError("review crash")
        )

        dual_run1 = DualInferencer(
            base_inferencer=base_run1,
            review_inferencer=review_run1,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        with self.assertRaises(RuntimeError) as ctx:
            await dual_run1._ainfer("request")
        self.assertIn("review crash", str(ctx.exception))
        base_run1.ainfer.assert_called_once()  # propose ran once

        # Verify checkpoint exists (saved after propose completed)
        attempt_dir = os.path.join(self.tmpdir, "attempt_01")
        self.assertTrue(os.path.isdir(attempt_dir))

        # --- Run 2: resume — propose should NOT be called ---
        base_run2 = _make_mock_inferencer("should not be called")
        review_run2 = _make_mock_inferencer(_review_json(approved=True))

        dual_run2 = DualInferencer(
            base_inferencer=base_run2,
            review_inferencer=review_run2,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        result = await dual_run2._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        # Base was NOT called in run 2 — resumed past propose
        base_run2.ainfer.assert_not_called()
        # Review was called in run 2
        review_run2.ainfer.assert_called_once()

    async def test_crash_at_fix_resume_skips_propose_and_review(self):
        """Crash during fix step → resume skips propose+review, runs fix."""
        # --- Run 1: propose OK, review rejects, fixer crashes ---
        base_run1 = _make_mock_inferencer("proposal v1")
        review_run1 = _make_mock_inferencer(
            _review_json(approved=False, severity="MAJOR")
        )
        fixer_run1 = _make_mock_inferencer(
            side_effect=RuntimeError("fixer crash")
        )

        dual_run1 = DualInferencer(
            base_inferencer=base_run1,
            review_inferencer=review_run1,
            fixer_inferencer=fixer_run1,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        with self.assertRaises(RuntimeError) as ctx:
            await dual_run1._ainfer("request")
        self.assertIn("fixer crash", str(ctx.exception))

        # --- Run 2: resume with working fixer ---
        base_run2 = _make_mock_inferencer("should not be called")
        review_run2 = _make_mock_inferencer(_review_json(approved=True))
        fixer_run2 = _make_mock_inferencer(_fix_response("fixed proposal"))

        dual_run2 = DualInferencer(
            base_inferencer=base_run2,
            review_inferencer=review_run2,
            fixer_inferencer=fixer_run2,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        result = await dual_run2._ainfer("request")

        self.assertIsInstance(result, DualInferencerResponse)
        # Base NOT called — resumed past propose
        base_run2.ainfer.assert_not_called()
        # Fixer called once — resumed at fix step
        fixer_run2.ainfer.assert_called_once()
        # Review called once — loop-back after fix
        review_run2.ainfer.assert_called_once()

    async def test_crash_at_second_review_iteration(self):
        """Crash at review during second loop iteration.

        Flow (run 1): propose → review1(reject) → fix1 → review2(crash)
        Resume (run 2): starts at review2 (fix1 checkpoint), → review2(approve)
        """
        # --- Run 1 ---
        base_run1 = _make_mock_inferencer("proposal")
        # review: reject once, then crash on second call
        review_run1 = _make_mock_inferencer(
            side_effect=[
                _review_json(approved=False, severity="MAJOR"),
                RuntimeError("review crash iter 2"),
            ]
        )
        fixer_run1 = _make_mock_inferencer(_fix_response("fixed v1"))

        dual_run1 = DualInferencer(
            base_inferencer=base_run1,
            review_inferencer=review_run1,
            fixer_inferencer=fixer_run1,
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        with self.assertRaises(RuntimeError) as ctx:
            await dual_run1._ainfer("request")
        self.assertIn("review crash iter 2", str(ctx.exception))

        # Verify: propose=1, review=2 (reject + crash), fix=1
        self.assertEqual(base_run1.ainfer.call_count, 1)
        self.assertEqual(review_run1.ainfer.call_count, 2)
        self.assertEqual(fixer_run1.ainfer.call_count, 1)

        # --- Run 2: resume at review (second iteration) ---
        base_run2 = _make_mock_inferencer("not called")
        review_run2 = _make_mock_inferencer(_review_json(approved=True))
        fixer_run2 = _make_mock_inferencer("not called")

        dual_run2 = DualInferencer(
            base_inferencer=base_run2,
            review_inferencer=review_run2,
            fixer_inferencer=fixer_run2,
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        result = await dual_run2._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        base_run2.ainfer.assert_not_called()
        fixer_run2.ainfer.assert_not_called()
        review_run2.ainfer.assert_called_once()


# =====================================================================
# Tier 4 — State Restoration Correctness
# =====================================================================


class Tier4_StateRestorationTest(unittest.IsolatedAsyncioTestCase):
    """Verify counters, proposal text, and flags survive resume."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_proposal_text_preserved_across_resume(self):
        """The proposal from run 1's propose step is used by run 2's review."""
        original_proposal = "UNIQUE_PROPOSAL_TEXT_12345"

        # Run 1: propose with unique text, then crash at review
        base_run1 = _make_mock_inferencer(original_proposal)
        review_run1 = _make_mock_inferencer(
            side_effect=RuntimeError("crash")
        )

        dual_run1 = DualInferencer(
            base_inferencer=base_run1,
            review_inferencer=review_run1,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        with self.assertRaises(RuntimeError):
            await dual_run1._ainfer("request")

        # Run 2: resume — review gets the original proposal
        captured_review_prompt = []
        review_run2 = _make_mock_inferencer(_review_json(approved=True))
        original_ainfer = review_run2.ainfer

        async def _capture_and_forward(prompt, **kw):
            captured_review_prompt.append(prompt)
            return await original_ainfer(prompt, **kw)

        review_run2.ainfer = AsyncMock(side_effect=_capture_and_forward)

        dual_run2 = DualInferencer(
            base_inferencer=_make_mock_inferencer("not called"),
            review_inferencer=review_run2,
            consensus_config=ConsensusConfig(
                max_iterations=3,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        result = await dual_run2._ainfer("request")
        self.assertTrue(result.consensus_achieved)

        # The review prompt should contain the original proposal text
        self.assertEqual(len(captured_review_prompt), 1)
        self.assertIn(
            original_proposal,
            captured_review_prompt[0],
            "Review prompt should contain the proposal from run 1",
        )

    async def test_iteration_counter_correct_after_resume(self):
        """total_iterations is correct after crash + resume."""
        # Run 1: propose → review(reject) → fix → review(crash) = 2 iterations started
        base_run1 = _make_mock_inferencer("proposal")
        review_run1 = _make_mock_inferencer(
            side_effect=[
                _review_json(approved=False, severity="MAJOR"),
                RuntimeError("crash"),
            ]
        )
        fixer_run1 = _make_mock_inferencer(_fix_response("fixed"))

        dual_run1 = DualInferencer(
            base_inferencer=base_run1,
            review_inferencer=review_run1,
            fixer_inferencer=fixer_run1,
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        with self.assertRaises(RuntimeError):
            await dual_run1._ainfer("request")

        # Run 2: resume at review (iter 2) → approve
        dual_run2 = DualInferencer(
            base_inferencer=_make_mock_inferencer("not called"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            fixer_inferencer=_make_mock_inferencer("not called"),
            consensus_config=ConsensusConfig(
                max_iterations=5,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        result = await dual_run2._ainfer("request")

        self.assertTrue(result.consensus_achieved)
        # Total: iter 1 (run 1) + iter 2 (partially run1, completed run2)
        # The state restored from checkpoint had total_iterations=1 (after fix)
        # The resumed review increments it to 2
        self.assertGreaterEqual(result.total_iterations, 2)


# =====================================================================
# Tier 5 — Multi-Attempt
# =====================================================================


class Tier5_MultiAttemptTest(unittest.IsolatedAsyncioTestCase):
    """max_consensus_attempts > 1."""

    async def test_second_attempt_succeeds(self):
        """First attempt exhausts iterations, second attempt achieves consensus."""
        review_responses = [
            # Attempt 1: iter 1 reject, iter 2 reject (exhausted)
            _review_json(approved=False, severity="CRITICAL"),
            _review_json(approved=False, severity="CRITICAL"),
            # Attempt 2: approve immediately
            _review_json(approved=True),
        ]

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(side_effect=review_responses),
            fixer_inferencer=_make_mock_inferencer(_fix_response("fixed")),
            consensus_config=ConsensusConfig(
                max_iterations=2,
                max_consensus_attempts=2,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("request")

        self.assertEqual(len(result.consensus_history), 2)
        self.assertFalse(result.consensus_history[0].consensus_reached)
        self.assertTrue(result.consensus_history[1].consensus_reached)
        self.assertTrue(result.consensus_achieved)

    async def test_multi_attempt_with_checkpoint(self):
        """Multi-attempt with checkpointing creates per-attempt directories."""
        tmpdir = tempfile.mkdtemp()
        try:
            review_responses = [
                _review_json(approved=False, severity="CRITICAL"),
                _review_json(approved=False, severity="CRITICAL"),
                _review_json(approved=True),
            ]

            dual = DualInferencer(
                base_inferencer=_make_mock_inferencer("proposal"),
                review_inferencer=_make_mock_inferencer(
                    side_effect=review_responses
                ),
                fixer_inferencer=_make_mock_inferencer(_fix_response("fixed")),
                consensus_config=ConsensusConfig(
                    max_iterations=2,
                    max_consensus_attempts=2,
                    consensus_threshold=Severity.COSMETIC,
                ),
                enable_checkpoint=True,
                checkpoint_dir=tmpdir,
            )

            result = await dual._ainfer("request")

            self.assertTrue(result.consensus_achieved)
            # Both attempt directories should exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "attempt_01")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "attempt_02")))
            # No pkl files anywhere
            self.assertEqual(
                _count_pkl_files(tmpdir),
                0,
                "No .pkl files should exist — DualInferencer uses JSON checkpoints",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# =====================================================================
# Tier 6 — Edge Cases
# =====================================================================


class Tier6_EdgeCasesTest(unittest.IsolatedAsyncioTestCase):
    """Corrupted checkpoints, empty checkpoint_dir, etc."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_empty_checkpoint_dir_disables_checkpointing(self):
        """checkpoint_dir='' with enable_checkpoint=True → no crash, no files."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=1,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir="",
        )

        # _get_result_path returns "" when checkpoint_dir is falsy,
        # which means _exists_result(path="") returns False → no resume attempted
        result = await dual._ainfer("request")
        self.assertTrue(result.consensus_achieved)

    async def test_corrupted_checkpoint_falls_back_gracefully(self):
        """Write garbage to checkpoint file → workflow starts from scratch."""
        # Create a valid-looking checkpoint directory with corrupted file
        attempt_dir = os.path.join(self.tmpdir, "attempt_01")
        os.makedirs(attempt_dir, exist_ok=True)
        ckpt_path = os.path.join(attempt_dir, "step___wf_checkpoint__.json")
        with open(ckpt_path, "w") as f:
            f.write("not valid json")

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=1,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=True,
            checkpoint_dir=self.tmpdir,
        )

        # Should not crash — _try_load_checkpoint catches exceptions and
        # falls back to backward scan, which also finds nothing → fresh start
        result = await dual._ainfer("request")
        self.assertTrue(result.consensus_achieved)

    async def test_checkpoint_disabled_no_files_written(self):
        """enable_checkpoint=False → zero checkpoint files created."""
        ckpt_dir = os.path.join(self.tmpdir, "should_stay_empty")
        os.makedirs(ckpt_dir, exist_ok=True)

        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(_review_json(approved=True)),
            consensus_config=ConsensusConfig(
                max_iterations=1,
                max_consensus_attempts=1,
                consensus_threshold=Severity.COSMETIC,
            ),
            enable_checkpoint=False,  # explicit
            checkpoint_dir=ckpt_dir,
        )

        result = await dual._ainfer("request")
        self.assertTrue(result.consensus_achieved)
        self.assertEqual(
            _count_json_files(ckpt_dir),
            0,
            "No .json files should be written when checkpointing is disabled",
        )
        self.assertEqual(
            _count_pkl_files(ckpt_dir),
            0,
            "No .pkl files should be written when checkpointing is disabled",
        )


if __name__ == "__main__":
    unittest.main()
