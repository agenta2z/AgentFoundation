"""
Unit tests for concrete post-processors.

Tests FlatStringPostProcessor, GroupedDictPostProcessor,
AggregatingPostProcessor, and BudgetAwarePostProcessor with
concrete examples and edge cases.

Requirements: 8.1, 8.2, 9.1, 9.2, 10.1, 10.2, 11.1, 11.2, 11.3
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
_rpu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))

import pytest
from agent_foundation.knowledge.retrieval.formatter import (
    KnowledgeFormatter, RetrievalResult,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece
from agent_foundation.knowledge.retrieval.retrieval_pipeline import (
    AgenticRetrievalResult, SubQuery,
)
from agent_foundation.knowledge.retrieval.post_processors import (
    AggregatingPostProcessor, BudgetAwarePostProcessor,
    CONTEXT_BUDGET, FlatStringPostProcessor, GroupedDictPostProcessor,
)
from agent_foundation.knowledge.retrieval.utils import count_tokens


# ---- Helpers ----

def _piece(pid, info_type="context", content=None):
    return KnowledgePiece(
        content=content or f"Content for {pid}",
        piece_id=pid, info_type=info_type,
    )


def _meta(eid, props=None, spaces=None):
    return EntityMetadata(
        entity_id=eid, entity_type="user",
        properties=props or {"name": eid},
        spaces=spaces or ["main"],
    )


def _result(pieces=None, metadata=None, global_metadata=None, graph_context=None):
    r = RetrievalResult()
    r.metadata = metadata
    r.global_metadata = global_metadata
    r.pieces = pieces or []
    r.graph_context = graph_context or []
    return r


def _edge(node_id, rel="RELATED", depth=1, score=0.5, piece=None, label=""):
    return {
        "target_node_id": node_id, "relation_type": rel,
        "depth": depth, "score": score, "piece": piece, "target_label": label,
    }


# ---- FlatStringPostProcessor ----


class TestFlatStringPostProcessor:
    """Requirements: 8.1, 8.2"""

    def test_produces_same_output_as_formatter(self):
        r = _result(
            pieces=[(_piece("p1"), 0.9), (_piece("p2"), 0.7)],
            metadata=_meta("user:1"),
        )
        formatter = KnowledgeFormatter()
        expected = formatter.format(r)
        pp = FlatStringPostProcessor(formatter=formatter)
        assert pp.process(r) == expected

    def test_handles_list_input(self):
        r = _result(pieces=[(_piece("p1"), 0.9)])
        formatter = KnowledgeFormatter()
        expected = formatter.format(r)
        pp = FlatStringPostProcessor(formatter=formatter)
        assert pp.process([r]) == expected

    def test_empty_result_returns_empty_string(self):
        pp = FlatStringPostProcessor()
        assert pp.process(_result()) == ""

    def test_with_graph_context(self):
        r = _result(graph_context=[_edge("n1", "WORKS_AT")])
        formatter = KnowledgeFormatter()
        expected = formatter.format(r)
        pp = FlatStringPostProcessor(formatter=formatter)
        assert pp.process(r) == expected


# ---- GroupedDictPostProcessor ----


class TestGroupedDictPostProcessor:
    """Requirements: 9.1, 9.2"""

    def test_routes_metadata_to_metadata_info_type(self):
        r = _result(metadata=_meta("user:1", {"name": "Alice"}))
        pp = GroupedDictPostProcessor(
            default_formatter=lambda m, p, g: f"meta:{len(m.properties)}" if m else "",
            metadata_info_type="user_profile",
        )
        out = pp.process(r)
        assert "user_profile" in out
        assert out["user_profile"] == "meta:1"

    def test_routes_pieces_by_info_type(self):
        r = _result(pieces=[
            (_piece("p1", info_type="context"), 0.9),
            (_piece("p2", info_type="instructions"), 0.8),
            (_piece("p3", info_type="context"), 0.7),
        ])
        pp = GroupedDictPostProcessor(
            default_formatter=lambda m, p, g: f"pieces:{len(p)}",
        )
        out = pp.process(r)
        assert out["context"] == "pieces:2"
        assert out["instructions"] == "pieces:1"

    def test_routes_graph_edge_with_linked_piece(self):
        linked = _piece("lp1", info_type="skills")
        r = _result(graph_context=[_edge("n1", piece=linked)])
        pp = GroupedDictPostProcessor(
            default_formatter=lambda m, p, g: f"graph:{len(g)}",
        )
        out = pp.process(r)
        assert "skills" in out
        assert out["skills"] == "graph:1"

    def test_routes_depth1_edge_to_user_profile(self):
        r = _result(graph_context=[_edge("n1", depth=1)])
        pp = GroupedDictPostProcessor(
            default_formatter=lambda m, p, g: f"graph:{len(g)}",
            active_entity_id="user:1",
        )
        out = pp.process(r)
        assert "user_profile" in out

    def test_routes_other_edges_to_context(self):
        r = _result(graph_context=[_edge("n1", depth=0)])
        pp = GroupedDictPostProcessor(
            default_formatter=lambda m, p, g: f"graph:{len(g)}",
        )
        out = pp.process(r)
        assert "context" in out

    def test_global_metadata_fallback(self):
        r = _result(global_metadata=_meta("global", {"setting": "val"}))
        pp = GroupedDictPostProcessor(
            default_formatter=lambda m, p, g: f"meta:{m.entity_id}" if m else "",
            metadata_info_type="user_profile",
        )
        out = pp.process(r)
        assert out["user_profile"] == "meta:global"

    def test_global_metadata_not_used_when_entity_metadata_present(self):
        r = _result(
            metadata=_meta("user:1", {"name": "Alice"}),
            global_metadata=_meta("global", {"setting": "val"}),
        )
        pp = GroupedDictPostProcessor(
            default_formatter=lambda m, p, g: f"meta:{m.entity_id}" if m else "",
            metadata_info_type="user_profile",
        )
        out = pp.process(r)
        assert out["user_profile"] == "meta:user:1"

    def test_consolidation_called(self):
        r = _result(pieces=[(_piece("p1"), 0.9)])
        calls = []

        class MockConsolidator:
            def consolidate(self, query, output):
                calls.append((query, output))
                return {k: v.upper() for k, v in output.items()}

        pp = GroupedDictPostProcessor(
            default_formatter=lambda m, p, g: "formatted",
            consolidator=MockConsolidator(),
        )
        out = pp.process(r, query="test query")
        assert len(calls) == 1
        assert calls[0][0] == "test query"
        assert out["context"] == "FORMATTED"

    def test_empty_result_returns_empty_dict(self):
        pp = GroupedDictPostProcessor(default_formatter=lambda m, p, g: "x")
        assert pp.process(_result()) == {}

    def test_type_formatters_override_default(self):
        r = _result(pieces=[(_piece("p1", info_type="skills"), 0.9)])
        pp = GroupedDictPostProcessor(
            type_formatters={"skills": lambda m, p, g: "custom_skills"},
            default_formatter=lambda m, p, g: "default",
        )
        out = pp.process(r)
        assert out["skills"] == "custom_skills"


# ---- AggregatingPostProcessor ----


class TestAggregatingPostProcessor:
    """Requirements: 10.1, 10.2"""

    def test_max_strategy(self):
        r1 = _result(pieces=[(_piece("p1"), 0.8), (_piece("p2"), 0.6)])
        r2 = _result(pieces=[(_piece("p1"), 0.5), (_piece("p3"), 0.9)])
        sqs = [SubQuery(query="q1", weight=1.0), SubQuery(query="q2", weight=1.0)]
        pp = AggregatingPostProcessor(aggregation_strategy="max", top_k=10, min_results=0)
        out = pp.process([r1, r2], sub_queries=sqs)
        scores = {sp.piece.piece_id: sp.score for sp in out.pieces}
        assert abs(scores["p1"] - 0.8) < 1e-9
        assert abs(scores["p2"] - 0.6) < 1e-9
        assert abs(scores["p3"] - 0.9) < 1e-9

    def test_sum_strategy(self):
        r1 = _result(pieces=[(_piece("p1"), 0.8)])
        r2 = _result(pieces=[(_piece("p1"), 0.5)])
        sqs = [SubQuery(query="q1", weight=1.0), SubQuery(query="q2", weight=1.0)]
        pp = AggregatingPostProcessor(aggregation_strategy="sum", top_k=10, min_results=0)
        out = pp.process([r1, r2], sub_queries=sqs)
        scores = {sp.piece.piece_id: sp.score for sp in out.pieces}
        assert abs(scores["p1"] - 1.3) < 1e-9

    def test_weighted_sum_same_as_sum(self):
        r1 = _result(pieces=[(_piece("p1"), 0.8)])
        sqs = [SubQuery(query="q1", weight=2.0)]
        pp_sum = AggregatingPostProcessor(aggregation_strategy="sum", top_k=10, min_results=0)
        pp_ws = AggregatingPostProcessor(aggregation_strategy="weighted_sum", top_k=10, min_results=0)
        out_sum = pp_sum.process([r1], sub_queries=sqs)
        out_ws = pp_ws.process([r1], sub_queries=sqs)
        assert abs(out_sum.pieces[0].score - out_ws.pieces[0].score) < 1e-9

    def test_weights_applied(self):
        r1 = _result(pieces=[(_piece("p1"), 0.5)])
        sqs = [SubQuery(query="q1", weight=2.0)]
        pp = AggregatingPostProcessor(aggregation_strategy="max", top_k=10, min_results=0)
        out = pp.process([r1], sub_queries=sqs)
        assert abs(out.pieces[0].score - 1.0) < 1e-9

    def test_top_k_truncation(self):
        pieces = [(_piece(f"p{i}"), 0.9 - 0.01 * i) for i in range(10)]
        r = _result(pieces=pieces)
        sqs = [SubQuery(query="q1")]
        pp = AggregatingPostProcessor(top_k=3, min_results=0)
        out = pp.process([r], sub_queries=sqs)
        assert len(out.pieces) == 3

    def test_sorted_by_score_desc(self):
        r = _result(pieces=[(_piece("p1"), 0.3), (_piece("p2"), 0.9), (_piece("p3"), 0.6)])
        sqs = [SubQuery(query="q1")]
        pp = AggregatingPostProcessor(top_k=10, min_results=0)
        out = pp.process([r], sub_queries=sqs)
        scores = [sp.score for sp in out.pieces]
        assert scores == sorted(scores, reverse=True)

    def test_needs_fallback_when_below_min_results(self):
        r = _result(pieces=[(_piece("p1"), 0.9)])
        sqs = [SubQuery(query="q1")]
        pp = AggregatingPostProcessor(top_k=10, min_results=5)
        out = pp.process([r], sub_queries=sqs)
        assert out.needs_fallback is True

    def test_no_needs_fallback_when_enough_results(self):
        pieces = [(_piece(f"p{i}"), 0.9) for i in range(5)]
        r = _result(pieces=pieces)
        sqs = [SubQuery(query="q1")]
        pp = AggregatingPostProcessor(top_k=10, min_results=3)
        out = pp.process([r], sub_queries=sqs)
        assert out.needs_fallback is False

    def test_sub_queries_propagated(self):
        r = _result(pieces=[(_piece("p1"), 0.9)])
        sqs = [SubQuery(query="q1", domain="d1")]
        pp = AggregatingPostProcessor(top_k=10, min_results=0)
        out = pp.process([r], sub_queries=sqs)
        assert out.sub_queries == sqs

    def test_fallback_merge_dedup(self):
        primary = _result(pieces=[(_piece("p1"), 0.8)])
        fallback = _result(pieces=[(_piece("p1"), 0.3), (_piece("p2"), 0.7)])
        sqs = [SubQuery(query="q1")]
        pp = AggregatingPostProcessor(top_k=10, min_results=0)
        out = pp.process([fallback, primary], sub_queries=sqs, is_fallback=True)
        assert out.used_fallback is True
        scores = {sp.piece.piece_id: sp.score for sp in out.pieces}
        assert abs(scores["p1"] - 0.8) < 1e-9  # primary wins
        assert "p2" in scores

    def test_empty_results(self):
        pp = AggregatingPostProcessor(top_k=10, min_results=0)
        out = pp.process([_result()], sub_queries=[SubQuery(query="q1")])
        assert out.pieces == []


# ---- BudgetAwarePostProcessor ----


class TestBudgetAwarePostProcessor:
    """Requirements: 11.1, 11.2, 11.3"""

    def test_output_within_budget(self):
        pieces = [
            (_piece(f"p{i}", info_type="context", content="word " * 50), 0.9)
            for i in range(5)
        ]
        r = _result(pieces=pieces)
        pp = BudgetAwarePostProcessor(available_tokens=200)
        out = pp.process(r)
        tokens = count_tokens(out)
        assert tokens <= 200 + len(CONTEXT_BUDGET)

    def test_priority_order(self):
        pieces = []
        for it in CONTEXT_BUDGET:
            pieces.append((_piece(f"{it}_0", info_type=it, content=f"Content for {it}"), 0.9))
        r = _result(pieces=pieces)
        pp = BudgetAwarePostProcessor(available_tokens=50000)
        out = pp.process(r)
        headers = {
            "skills": "## Available Skills",
            "instructions": "## Instructions",
            "context": "## Relevant Context",
            "episodic": "## Recent History",
            "user_profile": "## User Preferences",
        }
        last_pos = -1
        for it in CONTEXT_BUDGET:
            pos = out.find(headers[it])
            if pos >= 0:
                assert pos > last_pos, f"{it} at {pos} before previous at {last_pos}"
                last_pos = pos

    def test_empty_result(self):
        pp = BudgetAwarePostProcessor()
        assert pp.process(_result()) == ""

    def test_handles_list_input(self):
        r = _result(pieces=[(_piece("p1", info_type="context", content="hello"), 0.9)])
        pp = BudgetAwarePostProcessor(available_tokens=5000)
        out = pp.process([r])
        assert "hello" in out

    def test_respects_per_type_budget(self):
        # Create many context pieces that exceed the context budget
        pieces = [
            (_piece(f"p{i}", info_type="context", content="word " * 200), 0.9)
            for i in range(20)
        ]
        r = _result(pieces=pieces)
        pp = BudgetAwarePostProcessor(available_tokens=50000)
        out = pp.process(r)
        # Output should be limited by the context budget (3000 tokens)
        tokens = count_tokens(out)
        assert tokens <= CONTEXT_BUDGET["context"] + len(CONTEXT_BUDGET)
