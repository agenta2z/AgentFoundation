"""Tests for hierarchical graph visualization components.

Covers: NamespacedGraphReporter, child_reporter composition, GraphTopologyEvent
fields, send_graph_event resilience, and rate limiting.
"""

import asyncio
import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.graph_events import (
    GraphTopologyEvent,
    NodeStatusEvent,
    NodeStreamEvent,
)
from agent_foundation.ui.graph_interactive_adapter import (
    NamespacedGraphReporter,
    WebSocketGraphReporter,
)


class FakeWsInteractive:
    """Minimal WebSocket interactive mock for testing."""

    def __init__(self):
        self.events_sent: list[tuple] = []
        self.send_graph_event = AsyncMock(side_effect=self._record)

    async def _record(self, event, task_id=""):
        self.events_sent.append((event, task_id))


class TestGraphTopologyEvent(unittest.TestCase):
    """Step 1: GraphTopologyEvent has parent_node_id, version, is_container."""

    def test_default_fields(self):
        evt = GraphTopologyEvent(nodes=[], edges=[])
        self.assertEqual(evt.parent_node_id, "")
        self.assertEqual(evt.version, 0)

    def test_parent_node_id(self):
        evt = GraphTopologyEvent(
            nodes=[{"id": "w0", "is_container": True}],
            edges=[],
            parent_node_id="worker_0",
            version=2,
        )
        self.assertEqual(evt.parent_node_id, "worker_0")
        self.assertEqual(evt.version, 2)
        self.assertTrue(evt.nodes[0]["is_container"])


class TestNamespacedGraphReporter(unittest.TestCase):
    """Step 2: NamespacedGraphReporter prefixes node IDs correctly."""

    def setUp(self):
        self.ws = FakeWsInteractive()
        self.root = WebSocketGraphReporter(self.ws, task_id="t1")

    def test_qualify_node_id(self):
        child = NamespacedGraphReporter(self.root, "worker_0")
        self.assertEqual(child._qualify("breakdown"), "worker_0/breakdown")
        self.assertEqual(child._qualify("worker_1"), "worker_0/worker_1")

    def test_topology_prefixes_and_sets_parent(self):
        child = NamespacedGraphReporter(self.root, "worker_0")
        evt = GraphTopologyEvent(
            nodes=[{"id": "w0"}, {"id": "w1"}],
            edges=[{"source": "w0", "target": "w1"}],
        )
        asyncio.get_event_loop().run_until_complete(child.on_graph_topology(evt))
        sent_evt = self.ws.events_sent[0][0]
        self.assertEqual(sent_evt.parent_node_id, "worker_0")
        self.assertEqual(sent_evt.nodes[0]["id"], "worker_0/w0")
        self.assertEqual(sent_evt.edges[0]["source"], "worker_0/w0")
        self.assertEqual(sent_evt.edges[0]["target"], "worker_0/w1")

    def test_status_prefixes_node_id(self):
        child = NamespacedGraphReporter(self.root, "worker_0")
        asyncio.get_event_loop().run_until_complete(
            child.on_node_status("w0", "running")
        )
        sent_evt = self.ws.events_sent[0][0]
        self.assertIsInstance(sent_evt, NodeStatusEvent)
        self.assertEqual(sent_evt.node_id, "worker_0/w0")

    def test_stream_prefixes_node_id(self):
        child = NamespacedGraphReporter(self.root, "worker_0")
        asyncio.get_event_loop().run_until_complete(
            child.on_node_stream("w0", "hello")
        )
        sent_evt = self.ws.events_sent[0][0]
        self.assertIsInstance(sent_evt, NodeStreamEvent)
        self.assertEqual(sent_evt.node_id, "worker_0/w0")


class TestChildReporterRecursive(unittest.TestCase):
    """Step 2: child_reporter composes prefixes recursively: a/b/c."""

    def setUp(self):
        self.ws = FakeWsInteractive()
        self.root = WebSocketGraphReporter(self.ws, task_id="t1")

    def test_two_level_nesting(self):
        child = self.root.child_reporter("worker_0")
        grandchild = child.child_reporter("worker_2")
        self.assertEqual(grandchild._qualify("breakdown"), "worker_0/worker_2/breakdown")

    def test_three_level_nesting(self):
        a = self.root.child_reporter("a")
        b = a.child_reporter("b")
        c = b.child_reporter("c")
        self.assertEqual(c._qualify("node"), "a/b/c/node")

    def test_topology_at_depth_3(self):
        a = self.root.child_reporter("a")
        b = a.child_reporter("b")
        c = b.child_reporter("c")
        evt = GraphTopologyEvent(
            nodes=[{"id": "leaf"}], edges=[],
        )
        asyncio.get_event_loop().run_until_complete(c.on_graph_topology(evt))
        sent_evt = self.ws.events_sent[0][0]
        self.assertEqual(sent_evt.parent_node_id, "a/b/c")
        self.assertEqual(sent_evt.nodes[0]["id"], "a/b/c/leaf")


class TestSendGraphEventResilience(unittest.TestCase):
    """Step 4: send_graph_event swallows errors and has circuit breaker."""

    def test_on_graph_topology_swallows_error(self):
        ws = FakeWsInteractive()
        ws.send_graph_event = AsyncMock(side_effect=ConnectionError("closed"))
        reporter = WebSocketGraphReporter(ws, task_id="t1")
        evt = GraphTopologyEvent(nodes=[], edges=[])
        # Should NOT raise
        asyncio.get_event_loop().run_until_complete(reporter.on_graph_topology(evt))

    def test_on_node_status_swallows_error(self):
        ws = FakeWsInteractive()
        ws.send_graph_event = AsyncMock(side_effect=RuntimeError("boom"))
        reporter = WebSocketGraphReporter(ws, task_id="t1")
        asyncio.get_event_loop().run_until_complete(
            reporter.on_node_status("w0", "running")
        )

    def test_on_node_stream_swallows_error(self):
        ws = FakeWsInteractive()
        ws.send_graph_event = AsyncMock(side_effect=OSError("pipe broken"))
        reporter = WebSocketGraphReporter(ws, task_id="t1")
        asyncio.get_event_loop().run_until_complete(
            reporter.on_node_stream("w0", "content", is_final=True)
        )


class TestRateLimiting(unittest.TestCase):
    """Step 12: Rate limiter on WebSocketGraphReporter."""

    def test_rate_limit_drops_excess_stream_events(self):
        ws = FakeWsInteractive()
        reporter = WebSocketGraphReporter(ws, task_id="t1", max_msg_per_sec=5)
        loop = asyncio.get_event_loop()
        for i in range(10):
            loop.run_until_complete(
                reporter.on_node_stream(f"w{i}", f"chunk{i}", is_final=False)
            )
        # First 5 should pass, rest should be rate-limited
        self.assertEqual(len(ws.events_sent), 5)

    def test_is_final_always_passes(self):
        ws = FakeWsInteractive()
        reporter = WebSocketGraphReporter(ws, task_id="t1", max_msg_per_sec=2)
        loop = asyncio.get_event_loop()
        # Exhaust rate limit
        for i in range(3):
            loop.run_until_complete(
                reporter.on_node_stream(f"w{i}", "chunk", is_final=False)
            )
        # is_final should still pass
        loop.run_until_complete(
            reporter.on_node_stream("w0", "final", is_final=True)
        )
        # 2 regular + 1 final = 3
        self.assertEqual(len(ws.events_sent), 3)

    def test_status_not_rate_limited(self):
        ws = FakeWsInteractive()
        reporter = WebSocketGraphReporter(ws, task_id="t1", max_msg_per_sec=2)
        loop = asyncio.get_event_loop()
        # Status events should always pass regardless of rate limit
        for i in range(5):
            loop.run_until_complete(
                reporter.on_node_status(f"w{i}", "running")
            )
        self.assertEqual(len(ws.events_sent), 5)


class TestMockBtaComponents(unittest.TestCase):
    """Step 5: Mock BTA components work correctly."""

    def test_mock_worker_streams_and_returns(self):
        from agent_foundation.common.inferencers.mock_inferencers.mock_bta_components import MockWorker
        worker = MockWorker(label="test", duration_s=0.1)
        chunks = []
        async def observer(chunk):
            chunks.append(chunk)
        worker.stream_observer = observer
        result = asyncio.get_event_loop().run_until_complete(worker.ainfer())
        self.assertTrue(len(chunks) > 0)
        self.assertIn("test", result)

    def test_mock_worker_error(self):
        from agent_foundation.common.inferencers.mock_inferencers.mock_bta_components import MockWorker
        worker = MockWorker(should_error=True, duration_s=0.05)
        with self.assertRaises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(worker.ainfer())

    def test_mock_breakdown_returns_json(self):
        import json
        from agent_foundation.common.inferencers.mock_inferencers.mock_bta_components import MockBreakdownInferencer
        bd = MockBreakdownInferencer(delay_s=0.05)
        result = asyncio.get_event_loop().run_until_complete(bd.ainfer())
        parsed = json.loads(result)
        self.assertIn("subtasks", parsed)
        self.assertEqual(len(parsed["subtasks"]), 3)

    def test_mock_aggregator_returns_text(self):
        from agent_foundation.common.inferencers.mock_inferencers.mock_bta_components import MockAggregator
        agg = MockAggregator(delay_s=0.05)
        result = asyncio.get_event_loop().run_until_complete(agg.ainfer())
        self.assertIn("Aggregated", result)


if __name__ == "__main__":
    unittest.main()
