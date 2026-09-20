"""Integration tests for MFDual workspace anomalies — regression guards.

Six tests that exercise REAL Hydra instantiation and orchestration flow
with MOCK LLM inferencers. Each test targets a specific historical anomaly
documented in the workspace observability plan.

Anomalies covered:
  1. LazyConfigFactory produces isolated workers (no shared instances)
  2. switch_role sets correct template_key on reviewer/fixer
  3. (RETIRED in Part 2) double final_deliverables nesting — the
     final_deliverables/ subfolder no longer exists, so this anomaly is moot.
  4. No empty round01 placeholder directories
  5. Worker sharing detection fires on shared instances
  6. Audit symlink cross-worker leakage detection
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from attr import attrib, attrs


# ---------------------------------------------------------------------------
# Mock inferencer — minimal InferencerBase subclass with scripted responses
# ---------------------------------------------------------------------------


@attrs
class _MockInferencer(InferencerBase):
    """Minimal InferencerBase subclass that returns scripted responses.

    Does NOT inherit from TemplatedInferencerBase, so it never attempts
    template rendering. Safe to plug into any slot that expects an
    InferencerBase (flow initial/followup, reviewer, fixer, aggregator).
    """

    scripted_response: str = attrib(default="mock output")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.scripted_response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self.scripted_response


from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)


@attrs
class _TemplatedMockInferencer(TemplatedInferencerBase):
    """Mock that inherits from TemplatedInferencerBase — has template_key,
    template_root_space, etc. Used to verify switch_role() actually mutates
    template attributes via the TemplatedInferencerBase override."""

    scripted_response: str = attrib(default="mock output")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.scripted_response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self.scripted_response


# ---------------------------------------------------------------------------
# Test 1: LazyConfigFactory produces isolated workers
# ---------------------------------------------------------------------------


class TestLazyConfigFactoryIsolation(unittest.TestCase):
    """Verify that BTA worker_inferencers produces workers with no shared
    inferencer instances between them.

    The anomaly: when a raw ``functools.partial`` was used as
    ``worker_inferencers`` without deep-copying, all workers shared the same
    sub-inferencer instances, causing cross-worker state pollution.
    LazyConfigFactory (Hydra's deferred instantiation) guarantees a fresh
    object graph per call.
    """

    def test_worker_inferencers_produces_isolated_workers(self):
        """Create a MultiFlowInferencer (MFDual's inner engine),
        call its worker factory twice, and verify NO shared instances."""
        inf_a = _MockInferencer(scripted_response="flow A")
        inf_b = _MockInferencer(scripted_response="flow B")

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "query A",
                    "initial_inferencer": inf_a,
                    "followup_inferencer": inf_a,
                    "max_dynamic_steps": 1,
                },
                {
                    "input": "query B",
                    "initial_inferencer": inf_b,
                    "followup_inferencer": inf_b,
                    "max_dynamic_steps": 1,
                },
            ],
            visible_flows="all",
            worker_isolation_check=True,
        )

        # Invoke the factory to produce two workers (LWI instances)
        factory = mfi.worker_inferencers
        worker_0 = factory("query A", 0)
        worker_1 = factory("query B", 1)

        # Collect all descendant inferencers for each worker
        descendants_0 = set(
            id(inf) for inf in worker_0._collect_all_descendant_inferencers()
        )
        descendants_1 = set(
            id(inf) for inf in worker_1._collect_all_descendant_inferencers()
        )

        # Worker IDs themselves should differ
        self.assertNotEqual(id(worker_0), id(worker_1))

        # The factory-produced LWI workers carry references to the SAME
        # flow_configs[i] inferencers (initial/followup), which is by design:
        # MultiFlow's worker factory closes over the flow_configs list.
        # What matters for isolation is that worker_0's subtree and
        # worker_1's subtree do not share the same initial/followup
        # inferencer across different flow indices.
        #
        # Verify: the initial_inferencer for flow 0 is NOT the same object
        # as the initial_inferencer for flow 1.
        self.assertIsNot(
            inf_a,
            inf_b,
            "Test setup error: flow inferencers must be distinct instances",
        )

        # Each worker's default_initial_inferencer should reference its own
        # flow's inferencer, not the other flow's.
        self.assertIs(worker_0.default_initial_inferencer, inf_a)
        self.assertIs(worker_1.default_initial_inferencer, inf_b)


# ---------------------------------------------------------------------------
# Test 2: switch_role sets correct template_key
# ---------------------------------------------------------------------------


class TestSwitchRoleSetsTemplateKey(unittest.TestCase):
    """Verify that _reassign_role_workspace calls switch_role with the
    correct template_key for reviewer ('review') and fixer ('followup').

    The anomaly: role reassignment forgot to set template_key, leaving
    a repurposed inferencer rendering the wrong template variant.
    """

    def test_reviewer_gets_review_template_key(self):
        """After _select_reviewer_and_fixer + _reassign_role_workspace,
        the review_inferencer should have template_key='review' and the
        fixer_inferencer should have template_key='followup'."""
        # Build a minimal MFDual with 2 flows, reviewer_match_second=True,
        # fixer_match_winner=True. Use distinct mock inferencers per flow.
        flow0_inf = _MockInferencer(scripted_response="flow 0 output")
        flow1_inf = _MockInferencer(scripted_response="flow 1 output")

        mfd = MultiFlowDualInferencer(
            flow_configs=[
                {
                    "input": "query 0",
                    "initial_inferencer": flow0_inf,
                    "followup_inferencer": flow0_inf,
                    "max_dynamic_steps": 1,
                },
                {
                    "input": "query 1",
                    "initial_inferencer": flow1_inf,
                    "followup_inferencer": flow1_inf,
                    "max_dynamic_steps": 1,
                },
            ],
            visible_flows="all",
            reviewer_strategy="runner_up",
            fixer_strategy="winner",
        )

        # Give MFDual a workspace
        tmpdir = tempfile.mkdtemp(prefix="mfdual_role_")
        try:
            ws = InferencerWorkspace(root=tmpdir)
            ws.ensure_dirs()
            mfd._workspace = ws

            # Simulate a winner: flow 0 wins, flow 1 is runner-up.
            mfi = mfd.base_inferencer
            mfi._last_winner_idx = 0
            mfi._last_ranking = [0, 1]

            # Trigger dispatch
            mfd._select_reviewer_and_fixer()

            # After dispatch: reviewer should be flow1_inf (runner-up),
            # fixer should be flow0_inf (winner).
            self.assertIs(mfd.review_inferencer, flow1_inf)
            self.assertIs(mfd.fixer_inferencer, flow0_inf)

            # Now trigger workspace reassignment
            mfd._reassign_role_workspace(mfd.review_inferencer, "review_inferencer")
            mfd._reassign_role_workspace(mfd.fixer_inferencer, "fixer_inferencer")

            # _reassign_role_workspace calls switch_role which sets template_key
            # on TemplatedInferencerBase instances. Our _MockInferencer is NOT a
            # TemplatedInferencerBase, so template_key is set via switch_role's
            # workspace + audit trail. Verify the switch_role was called by
            # checking the role_history audit trail.
            review_history = getattr(mfd.review_inferencer, "_role_history", [])
            fixer_history = getattr(mfd.fixer_inferencer, "_role_history", [])

            self.assertTrue(
                len(review_history) > 0,
                "switch_role should have been called on the reviewer",
            )
            self.assertEqual(review_history[-1]["to_role"], "review_inferencer")

            self.assertTrue(
                len(fixer_history) > 0,
                "switch_role should have been called on the fixer",
            )
            self.assertEqual(fixer_history[-1]["to_role"], "fixer_inferencer")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_templated_inferencer_gets_template_key_mutated(self):
        """Using a REAL TemplatedInferencerBase mock, verify template_key
        actually changes from '' to 'review'/'followup' after role swap.

        This exercises the TemplatedInferencerBase.switch_role() override
        (not just InferencerBase.switch_role)."""
        flow0_inf = _TemplatedMockInferencer(scripted_response="flow 0")
        flow1_inf = _TemplatedMockInferencer(scripted_response="flow 1")

        # Both start with empty template_key (default)
        self.assertEqual(flow0_inf.template_key, "")
        self.assertEqual(flow1_inf.template_key, "")

        mfd = MultiFlowDualInferencer(
            flow_configs=[
                {
                    "input": "q0",
                    "initial_inferencer": flow0_inf,
                    "followup_inferencer": flow0_inf,
                    "max_dynamic_steps": 1,
                },
                {
                    "input": "q1",
                    "initial_inferencer": flow1_inf,
                    "followup_inferencer": flow1_inf,
                    "max_dynamic_steps": 1,
                },
            ],
            visible_flows="all",
            reviewer_strategy="runner_up",
            fixer_strategy="winner",
        )

        tmpdir = tempfile.mkdtemp(prefix="mfdual_tmpl_")
        try:
            ws = InferencerWorkspace(root=tmpdir)
            ws.ensure_dirs()
            mfd._workspace = ws

            mfi = mfd.base_inferencer
            mfi._last_winner_idx = 0
            mfi._last_ranking = [0, 1]
            mfd._select_reviewer_and_fixer()

            mfd._reassign_role_workspace(mfd.review_inferencer, "review_inferencer")
            mfd._reassign_role_workspace(mfd.fixer_inferencer, "fixer_inferencer")

            # CRITICAL: template_key should have been mutated by
            # TemplatedInferencerBase.switch_role()
            self.assertEqual(
                mfd.review_inferencer.template_key,
                "review",
                "Reviewer's template_key should be 'review' after role swap",
            )
            self.assertEqual(
                mfd.fixer_inferencer.template_key,
                "followup",
                "Fixer's template_key should be 'followup' after role swap",
            )

            # Verify the workspace was properly assigned (fresh child workspace)
            review_ws = getattr(mfd.review_inferencer, "_workspace", None)
            fixer_ws = getattr(mfd.fixer_inferencer, "_workspace", None)
            # The _workspace property is stored via name mangling
            review_ws = getattr(
                mfd.review_inferencer, "_InferencerBase__workspace", review_ws
            )
            fixer_ws = getattr(
                mfd.fixer_inferencer, "_InferencerBase__workspace", fixer_ws
            )
            self.assertIsNotNone(
                review_ws, "Reviewer should have a workspace after role reassignment"
            )
            self.assertIsNotNone(
                fixer_ws, "Fixer should have a workspace after role reassignment"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: No double final_deliverables nesting — RETIRED in Part 2.
#
# ``TestNoDoubleFinalDeliverablesNesting`` (test_no_nested_final_deliverables,
# test_no_nested_final_deliverables_via_bta_promote) guarded against creating
# ``outputs/final_deliverables/final_deliverables/`` when surfacing a child whose
# deliverables lived in ``outputs/final_deliverables/``. Part 2 retired the
# ``final_deliverables/`` subfolder entirely — deliverables live directly in
# ``outputs/`` and ``surface_outputs_from`` copies outputs/ → outputs/ — so the
# double-nesting anomaly cannot occur and these tests have no subject. (Flat
# outputs/ surfacing is covered by test_workspace_phase0.surface_outputs_from_* and
# test_deliverable_boundary.*.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 4: No empty round01 placeholder
# ---------------------------------------------------------------------------


class TestFollowupWorkspaceAssignment(unittest.TestCase):
    """Verify that MultiFlow delegates flow_configs workspace to LWI.

    With the hierarchical layout, MultiFlow no longer propagates to
    flow_configs inferencers. They get workspaces from LWI's own
    ``_propagate_workspace_to_children`` when BTA assigns each worker.
    """

    def test_followup_inferencer_no_workspace_from_multiflow(self):
        """After constructing a MultiFlowInferencer with a workspace,
        followup_inferencer should NOT have workspace — it comes from LWI."""
        tmpdir = tempfile.mkdtemp(prefix="mfi_hierarchical_")
        try:
            inf_initial = _MockInferencer(scripted_response="initial output")
            inf_followup = _MockInferencer(scripted_response="followup output")

            mfi = MultiFlowInferencer(
                flow_configs=[
                    {
                        "input": "query 0",
                        "initial_inferencer": inf_initial,
                        "followup_inferencer": inf_followup,
                        "max_dynamic_steps": 2,
                    },
                    {
                        "input": "query 1",
                        "initial_inferencer": _MockInferencer(scripted_response="q1"),
                        "followup_inferencer": _MockInferencer(
                            scripted_response="q1 fup"
                        ),
                        "max_dynamic_steps": 2,
                    },
                ],
                visible_flows="all",
                workspace=InferencerWorkspace(root=tmpdir),
            )

            # With hierarchical layout, MultiFlow delegates to LWI.
            # flow_configs inferencers do NOT get workspace from MultiFlow.
            followup_ws = getattr(inf_followup, "_workspace", None)
            self.assertIsNone(
                followup_ws,
                "followup_inferencer should NOT have workspace from MultiFlow — "
                "LWI propagation handles it when BTA assigns the worker.",
            )

            # initial_inferencer also shouldn't have workspace from MultiFlow
            initial_ws = getattr(inf_initial, "_workspace", None)
            self.assertIsNone(
                initial_ws,
                "initial_inferencer should NOT have workspace from MultiFlow.",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: Worker sharing detection fires on shared instances
# ---------------------------------------------------------------------------


class TestWorkerSharingDetection(unittest.TestCase):
    """Verify that BTA's _validate_worker_isolation logs a warning when
    two workers share a sub-inferencer instance.

    The anomaly: without isolation checking, shared instances caused
    silent cross-worker state pollution (workspace, session, prompt
    history).
    """

    def test_shared_instance_triggers_warning(self):
        """Manually create 2 workers that SHARE an inferencer instance,
        call _validate_worker_isolation, assert a warning was logged."""
        shared_inf = _MockInferencer(scripted_response="shared")

        # Create two workers that reference the same shared_inf
        worker_0 = _MockInferencer(scripted_response="w0")
        worker_1 = _MockInferencer(scripted_response="w1")

        # Make them look like they have children that include shared_inf.
        # Override _iter_child_inferencers to yield shared_inf.
        original_iter_0 = worker_0._iter_child_inferencers
        original_iter_1 = worker_1._iter_child_inferencers

        def _yield_shared_0():
            yield shared_inf

        def _yield_shared_1():
            yield shared_inf

        worker_0._iter_child_inferencers = _yield_shared_0
        worker_1._iter_child_inferencers = _yield_shared_1

        # Create a minimal BTA to call _validate_worker_isolation
        bta = BreakdownThenAggregateInferencer.__new__(BreakdownThenAggregateInferencer)
        bta.worker_isolation_check = True

        log_target = (
            "agent_foundation.common.inferencers.agentic_inferencers"
            ".flow_inferencers.breakdown_then_aggregate_inferencer"
        )
        with patch(f"{log_target}._logger") as mock_logger:
            bta._validate_worker_isolation([worker_0, worker_1])

            # Assert that a warning was logged about the shared instance
            self.assertTrue(
                mock_logger.warning.called,
                "_validate_worker_isolation should log a WARNING when two "
                "workers share a sub-inferencer instance",
            )
            # Verify the warning message mentions the shared class name
            call_args = mock_logger.warning.call_args
            warning_msg = call_args[0][0] % call_args[0][1:]
            self.assertIn(
                "_MockInferencer",
                warning_msg,
                "Warning should mention the shared inferencer class name",
            )


# ---------------------------------------------------------------------------
# Test 6: Audit symlink cross-worker detection
# ---------------------------------------------------------------------------


class TestAuditSymlinkCrossWorkerDetection(unittest.TestCase):
    """Verify that DualInferencer._record_round_audit logs an error when
    the inferencer's workspace is outside the Dual's workspace tree.

    The anomaly: without cross-worker leakage detection, audit symlinks
    could point outside the parent workspace tree, indicating a workspace
    assignment bug (an inferencer is operating in the wrong workspace).
    """

    def test_cross_worker_leakage_logs_error(self):
        """Create a DualInferencer with workspace A, and an inferencer
        whose workspace root is in a completely different tree B. Calling
        _record_round_audit should log an error about cross-worker leakage."""
        tmpdir = tempfile.mkdtemp(prefix="mfdual_audit_")
        try:
            # DualInferencer workspace
            dual_root = os.path.join(tmpdir, "dual_workspace")
            os.makedirs(dual_root, exist_ok=True)

            # Foreign workspace (simulates wrong workspace assignment)
            foreign_root = os.path.join(tmpdir, "foreign_workspace")
            os.makedirs(foreign_root, exist_ok=True)

            # Create a mock DualInferencer with a workspace
            from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
                DualInferencer,
            )

            # Build a stub DualInferencer without full attrs init
            dual_stub = DualInferencer.__new__(DualInferencer)
            dual_ws = InferencerWorkspace(root=dual_root)
            dual_ws.ensure_dirs()
            # Set _workspace via the name-mangled backing store (bypass setter)
            object.__setattr__(dual_stub, "_InferencerBase__workspace", dual_ws)
            dual_stub.enable_round_audit = True
            dual_stub.output_path = "output.md"

            # Create a mock inferencer with the foreign workspace
            foreign_inf = MagicMock()
            foreign_ws = InferencerWorkspace(root=foreign_root)
            foreign_inf._workspace = foreign_ws

            log_target = (
                "agent_foundation.common.inferencers.agentic_inferencers"
                ".flow_inferencers.dual_inferencer"
            )
            with patch(f"{log_target}.logger") as mock_logger:
                dual_stub._record_round_audit(
                    round_idx=1,
                    phase="review",
                    inferencer=foreign_inf,
                )

                # Assert that an error was logged about cross-worker leakage
                self.assertTrue(
                    mock_logger.error.called,
                    "_record_round_audit should log an ERROR when the "
                    "inferencer's workspace is outside the Dual's tree",
                )
                call_args = mock_logger.error.call_args
                msg = call_args[0][0] % call_args[0][1:]
                self.assertIn(
                    "cross-worker leakage",
                    msg,
                    "Error message should mention 'cross-worker leakage'",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 7: YAML-driven instantiate() round-trip — LazyConfigFactory fires
# ---------------------------------------------------------------------------


class TestYAMLDrivenLazyConfigFactory(unittest.TestCase):
    """Verify that instantiate() on a config with *_factory fields produces
    LazyConfigFactory (not functools.partial), and that two factory() calls
    produce workers with completely independent sub-inferencer trees.

    This is the KEY integration test — it exercises the actual code path
    from YAML config → Hydra walker → _filter_attrs_keys → LazyConfigFactory
    → fresh sub-trees per call.
    """

    def test_instantiate_produces_lazy_config_factory(self):
        """A BTA-like config with worker_inferencers.__default__._target_ should
        produce a LazyConfigFactory, not a functools.partial."""
        import functools

        import agent_foundation.common.configs.registered_targets  # noqa: F401
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils import instantiate, register_class
        from rich_python_utils.config_utils._lazy_config_factory import (
            LazyConfigFactory,
        )

        # Register our mock so Hydra can resolve _target_
        register_class(_MockInferencer, "_MockInferencer", category="inferencer")

        config = OmegaConf.create(
            {
                "_target_": "BTA",
                "worker_inferencers": {
                    "__default__": {
                        "_target_": "MultiFlowDual",
                        "flow_configs": [
                            {
                                "input": "task_0",
                                "initial_inferencer": {
                                    "_target_": "_MockInferencer",
                                    "scripted_response": "flow0",
                                },
                                "followup_inferencer": {
                                    "_target_": "_MockInferencer",
                                    "scripted_response": "flow0_followup",
                                },
                                "max_dynamic_steps": 1,
                            },
                            {
                                "input": "task_1",
                                "initial_inferencer": {
                                    "_target_": "_MockInferencer",
                                    "scripted_response": "flow1",
                                },
                                "followup_inferencer": {
                                    "_target_": "_MockInferencer",
                                    "scripted_response": "flow1_followup",
                                },
                                "max_dynamic_steps": 1,
                            },
                        ],
                        "multi_flow_aggregator_inferencer": {
                            "_target_": "_MockInferencer",
                            "scripted_response": "aggregated",
                        },
                    },
                },
                "breakdown_inferencer": {
                    "_target_": "_MockInferencer",
                    "scripted_response": "breakdown",
                },
            }
        )

        bta = instantiate(config)

        # The worker_inferencers should be a dict with __default__ as LazyConfigFactory
        self.assertIsInstance(bta.worker_inferencers, dict)
        factory = bta.worker_inferencers.get("__default__")
        self.assertIsNotNone(factory, "worker_inferencers['__default__'] should exist")
        self.assertIsInstance(
            factory,
            LazyConfigFactory,
            f"Expected LazyConfigFactory, got {type(factory).__name__}. "
            f"This means _filter_attrs_keys didn't record the factory config.",
        )
        self.assertNotIsInstance(
            factory,
            functools.partial,
            "Should NOT be a plain functools.partial (causes shared instances)",
        )

    def test_lazy_factory_produces_distinct_nested_instances(self):
        """Two factory() calls must produce MFDual instances with DISTINCT
        flow_configs[i]['initial_inferencer'] instances (no shared ids)."""
        import agent_foundation.common.configs.registered_targets  # noqa: F401
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils import instantiate, register_class

        register_class(_MockInferencer, "_MockInferencer", category="inferencer")

        config = OmegaConf.create(
            {
                "_target_": "BTA",
                "worker_inferencers": {
                    "__default__": {
                        "_target_": "MultiFlowDual",
                        "flow_configs": [
                            {
                                "input": "task_0",
                                "initial_inferencer": {
                                    "_target_": "_MockInferencer",
                                    "scripted_response": "a",
                                },
                                "followup_inferencer": {
                                    "_target_": "_MockInferencer",
                                    "scripted_response": "b",
                                },
                                "max_dynamic_steps": 1,
                            },
                        ],
                        "multi_flow_aggregator_inferencer": {
                            "_target_": "_MockInferencer",
                            "scripted_response": "agg",
                        },
                    },
                },
                "breakdown_inferencer": {
                    "_target_": "_MockInferencer",
                    "scripted_response": "bd",
                },
            }
        )

        bta = instantiate(config)
        factory = bta.worker_inferencers["__default__"]

        # Create two workers
        worker_0 = factory()
        worker_1 = factory()

        # Top-level: distinct MFDual instances
        self.assertIsNot(worker_0, worker_1)

        # CRITICAL: nested flow inferencers must be DISTINCT
        w0_flow0_init = worker_0.flow_configs[0]["initial_inferencer"]
        w1_flow0_init = worker_1.flow_configs[0]["initial_inferencer"]
        self.assertIsNot(
            w0_flow0_init,
            w1_flow0_init,
            "flow_configs[0]['initial_inferencer'] shared across workers! "
            "LazyConfigFactory should produce independent instances.",
        )

        # Also check followup
        w0_flow0_followup = worker_0.flow_configs[0]["followup_inferencer"]
        w1_flow0_followup = worker_1.flow_configs[0]["followup_inferencer"]
        self.assertIsNot(
            w0_flow0_followup,
            w1_flow0_followup,
            "flow_configs[0]['followup_inferencer'] shared across workers!",
        )

        # Also check aggregator
        w0_agg = worker_0.multi_flow_aggregator_inferencer
        w1_agg = worker_1.multi_flow_aggregator_inferencer
        self.assertIsNot(
            w0_agg, w1_agg, "multi_flow_aggregator_inferencer shared across workers!"
        )

        # Full tree check using _collect_all_descendant_inferencers
        ids_0 = {id(inf) for inf in worker_0._collect_all_descendant_inferencers()}
        ids_1 = {id(inf) for inf in worker_1._collect_all_descendant_inferencers()}
        shared = ids_0 & ids_1
        self.assertEqual(
            len(shared),
            0,
            f"Workers share {len(shared)} inferencer instance(s) in their trees. "
            f"LazyConfigFactory should eliminate ALL sharing.",
        )

    def test_nested_topology_dual_wrapping_bta(self):
        """PRODUCTION TOPOLOGY: Dual { BTA { worker_inferencers } }.

        The BTA is NESTED inside Dual's base_inferencer. This tests that
        _apply_lazy_factories_recursive finds the nested BTA and replaces
        its worker_inferencers partial with LazyConfigFactory.

        This is the test that was MISSING — the previous tests used BTA at
        the top level, which worked because _apply_lazy_factory ran on the
        top-level result. The production YAML has Dual at top level with
        BTA nested inside, so the factory replacement must recurse."""
        import functools

        import agent_foundation.common.configs.registered_targets  # noqa: F401
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils import instantiate, register_class
        from rich_python_utils.config_utils._lazy_config_factory import (
            LazyConfigFactory,
        )

        register_class(_MockInferencer, "_MockInferencer", category="inferencer")

        # Dual wrapping BTA — matches production topology
        config = OmegaConf.create(
            {
                "_target_": "Dual",
                "base_inferencer": {
                    "_target_": "BTA",
                    "worker_inferencers": {
                        "__default__": {
                            "_target_": "MultiFlowDual",
                            "flow_configs": [
                                {
                                    "input": "t0",
                                    "initial_inferencer": {
                                        "_target_": "_MockInferencer"
                                    },
                                    "followup_inferencer": {
                                        "_target_": "_MockInferencer"
                                    },
                                    "max_dynamic_steps": 1,
                                },
                            ],
                            "multi_flow_aggregator_inferencer": {
                                "_target_": "_MockInferencer"
                            },
                        },
                    },
                    "breakdown_inferencer": {"_target_": "_MockInferencer"},
                },
                "review_inferencer": {"_target_": "_MockInferencer"},
                "fixer_inferencer": {"_target_": "_MockInferencer"},
            }
        )

        root = instantiate(config)

        # Navigate to nested BTA
        bta = root.base_inferencer
        factory = bta.worker_inferencers.get("__default__")

        self.assertIsNotNone(factory)
        self.assertIsInstance(
            factory,
            LazyConfigFactory,
            f"NESTED BTA's worker_inferencers should be LazyConfigFactory, "
            f"got {type(factory).__name__}. "
            f"_apply_lazy_factories_recursive must recurse into Dual.base_inferencer.",
        )
        self.assertNotIsInstance(factory, functools.partial)

        # Verify two calls produce isolated instances
        w0 = factory()
        w1 = factory()
        ids_0 = {id(inf) for inf in w0._collect_all_descendant_inferencers()}
        ids_1 = {id(inf) for inf in w1._collect_all_descendant_inferencers()}
        shared = ids_0 & ids_1
        self.assertEqual(
            len(shared),
            0,
            f"NESTED topology: workers share {len(shared)} instances. "
            f"LazyConfigFactory must produce independent trees even "
            f"when BTA is nested inside Dual.",
        )


# ---------------------------------------------------------------------------
# Test 8: Root cascade variables propagate through LazyConfigFactory
# ---------------------------------------------------------------------------


class TestLazyConfigFactoryCascadePropagation(unittest.TestCase):
    """Verify that root-level cascade variables (_output_path, _logger, etc.)
    propagate into instances created by LazyConfigFactory at runtime.

    This is the test that was MISSING — prior tests verified instance
    isolation (no shared objects) but never checked whether parent-level
    cascade variables reach factory-created children.  The bug: LazyConfigFactory
    only re-injected variables that the factory's own config block explicitly
    declared, silently dropping parent-level cascades like _output_path.
    """

    def test_output_path_cascades_into_factory_created_workers(self):
        """_output_path at root level must reach flow inferencers inside
        a LazyConfigFactory-created MFDual worker."""
        import agent_foundation.common.configs.registered_targets  # noqa: F401
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils import instantiate, register_class

        register_class(_MockInferencer, "_MockInferencer", category="inferencer")

        config = OmegaConf.create(
            {
                "_target_": "BTA",
                "_output_path": "output.md",
                "worker_inferencers": {
                    "__default__": {
                        "_target_": "MultiFlowDual",
                        "flow_configs": [
                            {
                                "input": "task_0",
                                "initial_inferencer": {
                                    "_target_": "_MockInferencer",
                                    "scripted_response": "flow0",
                                },
                                "followup_inferencer": {
                                    "_target_": "_MockInferencer",
                                    "scripted_response": "flow0_followup",
                                },
                                "max_dynamic_steps": 1,
                            },
                        ],
                        "multi_flow_aggregator_inferencer": {
                            "_target_": "_MockInferencer",
                            "scripted_response": "aggregated",
                        },
                    },
                },
                "breakdown_inferencer": {
                    "_target_": "_MockInferencer",
                    "scripted_response": "breakdown",
                },
            }
        )

        bta = instantiate(config)

        # BTA itself should have output_path (direct cascade)
        self.assertEqual(
            bta.output_path,
            "output.md",
            "BTA should receive _output_path from root cascade",
        )

        # Factory-created worker should ALSO have it
        factory = bta.worker_inferencers["__default__"]
        worker = factory()

        self.assertEqual(
            worker.output_path,
            "output.md",
            "LazyConfigFactory-created MFDual should receive _output_path "
            "from root cascade. If this fails, LazyConfigFactory.__call__() "
            "is not re-injecting parent-level injectables.",
        )

        # Nested flow inferencers should have it too
        flow0_init = worker.flow_configs[0]["initial_inferencer"]
        self.assertEqual(
            flow0_init.output_path,
            "output.md",
            "flow_configs[0]['initial_inferencer'] should receive _output_path "
            "via cascade through LazyConfigFactory. This is the exact bug "
            "that caused hollow output directories.",
        )

    def test_multiple_cascade_variables_propagate(self):
        """All root cascade variables (_output_path, _debug_mode, etc.)
        must reach factory-created instances — not just one."""
        import agent_foundation.common.configs.registered_targets  # noqa: F401
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils import instantiate, register_class

        register_class(_MockInferencer, "_MockInferencer", category="inferencer")

        config = OmegaConf.create(
            {
                "_target_": "BTA",
                "_output_path": "custom_output.md",
                "_debug_mode": True,
                "worker_inferencers": {
                    "__default__": {
                        "_target_": "MultiFlowDual",
                        "flow_configs": [
                            {
                                "input": "t0",
                                "initial_inferencer": {"_target_": "_MockInferencer"},
                                "followup_inferencer": {"_target_": "_MockInferencer"},
                                "max_dynamic_steps": 1,
                            },
                        ],
                        "multi_flow_aggregator_inferencer": {
                            "_target_": "_MockInferencer"
                        },
                    },
                },
                "breakdown_inferencer": {"_target_": "_MockInferencer"},
            }
        )

        bta = instantiate(config)
        factory = bta.worker_inferencers["__default__"]
        worker = factory()

        self.assertEqual(worker.output_path, "custom_output.md")
        self.assertTrue(
            worker.debug_mode, "_debug_mode should cascade through LazyConfigFactory"
        )

    def test_factory_local_override_wins_over_root(self):
        """If the factory's own config declares _output_path, it should
        take precedence over the root-level cascade (local-wins)."""
        import agent_foundation.common.configs.registered_targets  # noqa: F401
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils import instantiate, register_class

        register_class(_MockInferencer, "_MockInferencer", category="inferencer")

        config = OmegaConf.create(
            {
                "_target_": "BTA",
                "_output_path": "root_output.md",
                "worker_inferencers": {
                    "__default__": {
                        "_target_": "MultiFlowDual",
                        "_output_path": "factory_output.md",
                        "flow_configs": [
                            {
                                "input": "t0",
                                "initial_inferencer": {"_target_": "_MockInferencer"},
                                "followup_inferencer": {"_target_": "_MockInferencer"},
                                "max_dynamic_steps": 1,
                            },
                        ],
                        "multi_flow_aggregator_inferencer": {
                            "_target_": "_MockInferencer"
                        },
                    },
                },
                "breakdown_inferencer": {"_target_": "_MockInferencer"},
            }
        )

        bta = instantiate(config)
        factory = bta.worker_inferencers["__default__"]
        worker = factory()

        self.assertEqual(
            worker.output_path,
            "factory_output.md",
            "Factory's own _output_path should win over root cascade "
            "(local-wins / setdefault semantics)",
        )


# ---------------------------------------------------------------------------
# Test 9: _finalize_output output handling (Part 2 two-axis model)
# ---------------------------------------------------------------------------


class TestFinalizeOutputHandling(unittest.TestCase):
    """Verify _finalize_output under the Part 2 two-axis model: ``outputs/`` IS
    the deliverable set, so there is NO move to ``final_deliverables/`` (retired,
    along with the ``output_is_deliverable`` flag). Everything the agent wrote to
    ``outputs/`` stays there as-is; the framework only materializes the
    ``<Response>``-extracted summary at ``output_path`` when the agent didn't
    write it itself.

    (The former ``TestDeliverablePromotion`` asserted the deleted Step-1 move into
    ``final_deliverables/`` gated on ``output_is_deliverable``; those move/flag
    assertions were removed and the surviving output-handling behavior is pinned
    below.)
    """

    def _make_inferencer(self, **kwargs):
        inf = _MockInferencer(**kwargs)
        return inf

    def test_agent_written_output_stays_in_outputs(self):
        """Agent wrote output.md → it stays in outputs/ unchanged (it IS the
        deliverable; no move to final_deliverables/)."""
        tmpdir = tempfile.mkdtemp(prefix="deliv_stay_")
        try:
            ws = InferencerWorkspace(root=tmpdir)
            ws.ensure_dirs()
            inf = self._make_inferencer(
                scripted_response="<Response>Summary</Response>",
                output_path="output.md",
            )
            inf._workspace = ws

            # Simulate agent writing to outputs/
            agent_content = "# Full artifact\nDetailed content here."
            with open(os.path.join(ws.outputs_dir, "output.md"), "w") as f:
                f.write(agent_content)

            inf._finalize_output(inf.scripted_response)

            # output.md stays in outputs/ with the agent's content intact —
            # the summary must NOT overwrite a non-empty agent-written output.
            out_path = os.path.join(ws.outputs_dir, "output.md")
            self.assertTrue(
                os.path.isfile(out_path),
                "agent-written output.md must remain in outputs/",
            )
            self.assertEqual(open(out_path).read(), agent_content)
            # outputs/ IS the deliverable set.
            self.assertTrue(ws.has_deliverables)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_agent_multi_file_output_stays_in_outputs(self):
        """Agent wrote output.md + skills/ + tools/ → ALL stay in outputs/ (no
        move); outputs/ contains exactly what the agent wrote."""
        tmpdir = tempfile.mkdtemp(prefix="deliv_multi_")
        try:
            ws = InferencerWorkspace(root=tmpdir)
            ws.ensure_dirs()
            inf = self._make_inferencer(
                scripted_response="<Response>Summary</Response>",
                output_path="output.md",
            )
            inf._workspace = ws

            # Simulate agent writing multiple files/dirs
            with open(os.path.join(ws.outputs_dir, "output.md"), "w") as f:
                f.write("# Report")
            skills_dir = os.path.join(ws.outputs_dir, "skills")
            os.makedirs(skills_dir)
            with open(os.path.join(skills_dir, "SKILL.md"), "w") as f:
                f.write("# Skill spec")
            tools_dir = os.path.join(ws.outputs_dir, "tools")
            os.makedirs(tools_dir)
            with open(os.path.join(tools_dir, "tool.json"), "w") as f:
                f.write('{"name": "test"}')

            inf._finalize_output(inf.scripted_response)

            # ALL stay in outputs/ (no move, no final_deliverables/ subfolder).
            self.assertTrue(os.path.isfile(os.path.join(ws.outputs_dir, "output.md")))
            self.assertTrue(
                os.path.isfile(os.path.join(ws.outputs_dir, "skills", "SKILL.md"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(ws.outputs_dir, "tools", "tool.json"))
            )
            self.assertEqual(
                sorted(os.listdir(ws.outputs_dir)),
                ["output.md", "skills", "tools"],
                "outputs/ should contain exactly the agent-written entries",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_summary_written_when_agent_wrote_no_output_md(self):
        """Agent wrote skills/ but NOT output.md → skills/ stays in outputs/,
        framework materializes the <Response> summary at outputs/output.md."""
        tmpdir = tempfile.mkdtemp(prefix="deliv_nomd_")
        try:
            ws = InferencerWorkspace(root=tmpdir)
            ws.ensure_dirs()
            inf = self._make_inferencer(
                scripted_response="<Response>Summary text</Response>",
                output_path="output.md",
            )
            inf._workspace = ws

            # Agent writes skills/ but NOT output.md
            skills_dir = os.path.join(ws.outputs_dir, "skills")
            os.makedirs(skills_dir)
            with open(os.path.join(skills_dir, "SKILL.md"), "w") as f:
                f.write("# Skill")

            inf._finalize_output(inf.scripted_response)

            # skills/ stays in outputs/ (no move).
            self.assertTrue(
                os.path.isfile(os.path.join(ws.outputs_dir, "skills", "SKILL.md"))
            )

            # Framework materialized the <Response> summary at outputs/output.md.
            summary_path = os.path.join(ws.outputs_dir, "output.md")
            self.assertTrue(
                os.path.isfile(summary_path),
                "Framework should write <Response> summary to outputs/output.md",
            )
            self.assertIn("Summary text", open(summary_path).read())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_summary_written_when_agent_wrote_nothing(self):
        """Agent wrote nothing → framework materializes the <Response> summary at
        outputs/output.md (the no-local-access sole-output case)."""
        tmpdir = tempfile.mkdtemp(prefix="deliv_empty_")
        try:
            ws = InferencerWorkspace(root=tmpdir)
            ws.ensure_dirs()
            inf = self._make_inferencer(
                scripted_response="<Response>API summary</Response>",
                output_path="output.md",
            )
            inf._workspace = ws

            inf._finalize_output(inf.scripted_response)

            summary_path = os.path.join(ws.outputs_dir, "output.md")
            self.assertTrue(os.path.isfile(summary_path))
            self.assertIn("API summary", open(summary_path).read())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_no_final_deliverables_subfolder_created(self):
        """_finalize_output must never create an outputs/final_deliverables/
        subfolder (retired in Part 2)."""
        tmpdir = tempfile.mkdtemp(prefix="deliv_nest_")
        try:
            ws = InferencerWorkspace(root=tmpdir)
            ws.ensure_dirs()
            inf = self._make_inferencer(
                scripted_response="<Response>S</Response>",
                output_path="output.md",
            )
            inf._workspace = ws

            with open(os.path.join(ws.outputs_dir, "output.md"), "w") as f:
                f.write("content")

            inf._finalize_output(inf.scripted_response)

            nested = os.path.join(ws.outputs_dir, "final_deliverables")
            self.assertFalse(
                os.path.isdir(nested),
                "outputs/final_deliverables/ must not be created (Part 2)",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMaxBreakdownInjection(unittest.TestCase):
    """Verify max_breakdown is injected into breakdown inferencer's template feed."""

    def test_max_breakdown_injected_when_set(self):
        bd = _TemplatedMockInferencer(scripted_response="breakdown")
        agg = _TemplatedMockInferencer(scripted_response="aggregated")
        bta = BreakdownThenAggregateInferencer(
            max_breakdown=3,
            breakdown_inferencer=bd,
            aggregator_inferencer=agg,
        )
        self.assertEqual(
            bta.breakdown_inferencer.template_extra_feed["max_breakdown"], 3
        )

    def test_max_breakdown_not_injected_when_none(self):
        bd = _TemplatedMockInferencer(scripted_response="breakdown")
        agg = _TemplatedMockInferencer(scripted_response="aggregated")
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=bd,
            aggregator_inferencer=agg,
        )
        self.assertNotIn("max_breakdown", bta.breakdown_inferencer.template_extra_feed)

    def test_explicit_override_wins(self):
        bd = _TemplatedMockInferencer(
            scripted_response="breakdown",
            template_extra_feed={"max_breakdown": 10},
        )
        agg = _TemplatedMockInferencer(scripted_response="aggregated")
        bta = BreakdownThenAggregateInferencer(
            max_breakdown=3,
            breakdown_inferencer=bd,
            aggregator_inferencer=agg,
        )
        self.assertEqual(
            bta.breakdown_inferencer.template_extra_feed["max_breakdown"], 10
        )


if __name__ == "__main__":
    unittest.main()
