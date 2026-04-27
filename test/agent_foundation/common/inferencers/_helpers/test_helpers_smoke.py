"""Smoke test that the _helpers package imports correctly under pytest.

Helpers are imported via the test-package prefix `test.agent_foundation...`
because pytest uses the test directory as rootdir; the production
`agent_foundation` namespace from src/ does not contain `_helpers`.
"""

import unittest


class TestHelpersImport(unittest.TestCase):
    def test_mock_inferencer_imports(self):
        from test.agent_foundation.common.inferencers._helpers.mock_inferencer import (
            MockInferencer,
            SequentialMockInferencer,
        )
        m = MockInferencer(response="hello")
        self.assertEqual(m._infer("any"), "hello")

        seq = SequentialMockInferencer(responses=["a", "b", "c"])
        self.assertEqual(seq._infer("x"), "a")
        self.assertEqual(seq._infer("x"), "b")
        self.assertEqual(seq._infer("x"), "c")
        # exhaustion: returns last
        self.assertEqual(seq._infer("x"), "c")

    def test_factories_import(self):
        from test.agent_foundation.common.inferencers._helpers.factories import (
            _make_mock_inferencer,
            _make_sequential_mock,
            _make_lwi,
            _make_dual,
            _make_pti,
            _make_bta,
            _make_mfi,
        )
        # smoke construction
        mock = _make_mock_inferencer("x")
        self.assertEqual(mock._infer("a"), "x")

    def test_realistic_responses_import(self):
        from test.agent_foundation.common.inferencers._helpers.realistic_responses import (
            DUAL_REVIEW_RESPONSE_MAJOR,
            DUAL_REVIEW_RESPONSE_COSMETIC,
            BTA_BREAKDOWN_NUMBERED,
            PTI_PLAN_OUTPUT,
        )
        self.assertIn("MAJOR", DUAL_REVIEW_RESPONSE_MAJOR)
        self.assertIn("COSMETIC", DUAL_REVIEW_RESPONSE_COSMETIC)
        self.assertIn("Research", BTA_BREAKDOWN_NUMBERED)
        self.assertIn("Plan", PTI_PLAN_OUTPUT)
