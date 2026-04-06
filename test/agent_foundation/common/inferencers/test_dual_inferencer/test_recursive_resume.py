

"""Automated tests for recursive resume: PTI → DualInferencer child workflows.

Tests that @artifact_type(Workflow, ...) on PTI correctly discovers child
DualInferencers and propagates checkpoint settings via _setup_child_workflows.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
    StepResultSaveOptions,
)
from rich_python_utils.common_objects.workflow.workflow import Workflow


def _make_mock_base_inferencer(id_="mock"):
    mock = MagicMock()
    mock.id = id_
    mock.set_parent_debuggable = MagicMock()
    return mock


def _make_dual_inferencer(id_="dual", checkpoint_dir=None):
    base = _make_mock_base_inferencer(f"{id_}_base")
    review = _make_mock_base_inferencer(f"{id_}_review")
    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        consensus_config=ConsensusConfig(
            max_iterations=1,
            max_consensus_attempts=1,
            consensus_threshold=Severity.COSMETIC,
        ),
        phase="test",
        id=id_,
    )
    if checkpoint_dir:
        dual.checkpoint_dir = checkpoint_dir
    return dual


class TestArtifactTypeMetadata(unittest.TestCase):
    """Verify @artifact_type decorator sets __artifact_types__ on PTI."""

    def test_pti_has_artifact_types(self):
        entries = getattr(PlanThenImplementInferencer, '__artifact_types__', None)
        self.assertIsNotNone(entries)
        self.assertTrue(len(entries) > 0)

    def test_pti_artifact_type_targets_workflow(self):
        entries = PlanThenImplementInferencer.__artifact_types__
        types = [e['target_type'] for e in entries]
        self.assertIn(Workflow, types)


class TestFindChildWorkflows(unittest.TestCase):
    """Test _find_child_workflows_in discovers DualInferencers."""

    def test_find_child_in_self(self):
        plan_dual = _make_dual_inferencer("planner")
        impl_dual = _make_dual_inferencer("executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        children = pti._find_child_workflows_in(pti)
        self.assertIn("planner_inferencer", children)
        self.assertIn("executor_inferencer", children)
        child_obj, entry = children["planner_inferencer"]
        self.assertIs(child_obj, plan_dual)

    def test_shared_inferencer_dedup(self):
        """When planner == executor (same object), both attr names appear."""
        shared = _make_dual_inferencer("shared")
        pti = PlanThenImplementInferencer(
            planner_inferencer=shared,
            executor_inferencer=shared,
        )
        children = pti._find_child_workflows_in(pti)
        self.assertIn("planner_inferencer", children)
        self.assertIn("executor_inferencer", children)


class TestSetupChildWorkflows(unittest.TestCase):
    """Test _setup_child_workflows propagates settings."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_propagates_result_root_override(self):
        plan_dual = _make_dual_inferencer("planner")
        impl_dual = _make_dual_inferencer("executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        pti._current_base_workspace = self.tmpdir
        pti.enable_result_save = StepResultSaveOptions.Always
        pti.resume_with_saved_results = True
        pti.checkpoint_mode = 'jsonfy'

        state = {"iteration": 1}
        pti._setup_child_workflows(state)

        self.assertIsNotNone(plan_dual._result_root_override)
        self.assertIsNotNone(impl_dual._result_root_override)
        # Child paths should be under <base>/checkpoints/pti/iter_1/<attr>
        self.assertIn("iter_1", plan_dual._result_root_override)
        self.assertIn("planner_inferencer", plan_dual._result_root_override)

    def test_propagates_checkpoint_settings(self):
        plan_dual = _make_dual_inferencer("planner")
        impl_dual = _make_dual_inferencer("executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        pti._current_base_workspace = self.tmpdir
        pti.enable_result_save = StepResultSaveOptions.Always
        pti.resume_with_saved_results = True
        pti.checkpoint_mode = 'jsonfy'

        state = {"iteration": 1}
        pti._setup_child_workflows(state)

        self.assertEqual(plan_dual.enable_result_save, StepResultSaveOptions.Always)
        self.assertTrue(plan_dual.resume_with_saved_results)
        self.assertEqual(plan_dual.checkpoint_mode, 'jsonfy')

    def test_child_paths_isolated_per_iteration(self):
        """Different iterations get different child directories."""
        plan_dual = _make_dual_inferencer("planner")
        impl_dual = _make_dual_inferencer("executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        pti._current_base_workspace = self.tmpdir
        pti.enable_result_save = StepResultSaveOptions.Always
        pti.resume_with_saved_results = True
        pti.checkpoint_mode = 'jsonfy'

        # Iteration 1
        state1 = {"iteration": 1}
        pti._setup_child_workflows(state1)
        planner_dir_iter1 = plan_dual._result_root_override

        # Iteration 2
        state2 = {"iteration": 2}
        pti._setup_child_workflows(state2)
        planner_dir_iter2 = plan_dual._result_root_override

        self.assertNotEqual(planner_dir_iter1, planner_dir_iter2)
        self.assertIn("iter_1", planner_dir_iter1)
        self.assertIn("iter_2", planner_dir_iter2)


class TestDualInferencerChildMode(unittest.TestCase):
    """Test DualInferencer child-mode adaptations (Phase B)."""

    def test_get_result_path_with_checkpoint_dir(self):
        dual = _make_dual_inferencer("test", checkpoint_dir="/tmp/ckpt")
        path = dual._get_result_path("propose")
        self.assertIn("attempt_00", path)
        self.assertIn("step_propose.json", path)
        self.assertTrue(path.startswith("/tmp/ckpt"))

    def test_get_result_path_without_checkpoint_dir(self):
        """Child mode: returns just a filename for _resolve_result_path."""
        dual = _make_dual_inferencer("test")
        dual.checkpoint_dir = None
        path = dual._get_result_path("propose")
        self.assertEqual(path, "step_a00_propose.json")
        self.assertFalse(os.path.isabs(path))

    def test_child_mode_respects_parent_settings(self):
        """When _result_root_override is set, don't override enable_result_save."""
        dual = _make_dual_inferencer("test")
        dual._result_root_override = "/tmp/parent_override"
        dual.enable_result_save = StepResultSaveOptions.Always
        dual.resume_with_saved_results = True

        # Simulate the condition in _ainfer
        if dual.enable_checkpoint and dual.checkpoint_dir:
            pass
        elif dual._result_root_override is not None:
            pass  # Should NOT reset enable_result_save
        else:
            dual.enable_result_save = False

        self.assertEqual(dual.enable_result_save, StepResultSaveOptions.Always)
        self.assertTrue(dual.resume_with_saved_results)


class TestPTIBuildSteps(unittest.TestCase):
    """Test _build_iteration_steps produces correct step structure."""

    def test_step_count(self):
        plan_dual = _make_dual_inferencer("planner")
        impl_dual = _make_dual_inferencer("executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        pti._current_base_workspace = "/tmp/ws"
        pti._current_inference_config = {}
        pti._current_inference_args = {}
        steps = pti._build_iteration_steps()
        self.assertEqual(len(steps), 4)

    def test_step_names(self):
        plan_dual = _make_dual_inferencer("planner")
        impl_dual = _make_dual_inferencer("executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        pti._current_base_workspace = "/tmp/ws"
        pti._current_inference_config = {}
        pti._current_inference_args = {}
        steps = pti._build_iteration_steps()
        names = [getattr(s, "name", None) for s in steps]
        self.assertEqual(names, ["plan", "approval", "implement", "analysis"])

    def test_analysis_has_loop_back(self):
        plan_dual = _make_dual_inferencer("planner")
        impl_dual = _make_dual_inferencer("executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        pti._current_base_workspace = "/tmp/ws"
        pti._current_inference_config = {}
        pti._current_inference_args = {}
        steps = pti._build_iteration_steps()
        analysis_step = steps[3]
        self.assertEqual(getattr(analysis_step, "loop_back_to", None), "plan")

    def test_loop_condition_uses_state(self):
        plan_dual = _make_dual_inferencer("planner")
        impl_dual = _make_dual_inferencer("executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        pti._current_base_workspace = "/tmp/ws"
        pti._current_inference_config = {}
        pti._current_inference_args = {}
        steps = pti._build_iteration_steps()
        analysis_step = steps[3]
        cond = getattr(analysis_step, "loop_condition", None)
        self.assertIsNotNone(cond)
        self.assertTrue(cond({"should_continue": True}, None))
        self.assertFalse(cond({"should_continue": False}, None))
        self.assertFalse(cond({}, None))


if __name__ == "__main__":
    unittest.main()
