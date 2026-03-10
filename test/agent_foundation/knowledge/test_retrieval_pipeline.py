"""
Unit tests for the RetrievalPipeline.

Tests single-query path (no expander), multi-query path (with expander),
fallback logic, error handling (expander failure, post-processor errors),
and that multi-query path calls retrieve_metadata/retrieve_identity_graph
once and retrieve_pieces/retrieve_search_graph N times.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 14.1, 14.2, 14.3
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

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
from agent_foundation.knowledge.retrieval.formatter import RetrievalResult
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.retrieval_pipeline import (
    AgenticRetrievalResult, PostProcessor, QueryExpander,
    RetrievalPipeline, SubQuery,
)


def _make_piece(piece_id, info_type="context"):
    return KnowledgePiece(content=f"Content {piece_id}", piece_id=piece_id, info_type=info_type)


def _make_result(num_pieces=2, metadata=None, graph_context=None):
    pieces = [(_make_piece(f"p{i}"), 0.9 - 0.1 * i) for i in range(num_pieces)]
    r = RetrievalResult()
    r.metadata = metadata
    r.pieces = pieces
    r.graph_context = graph_context or []
    return r


class RecordingPP(PostProcessor):
    def __init__(self, return_value="processed"):
        self.calls = []
        self.return_value = return_value

    def process(self, results, query="", sub_queries=None, **kwargs):
        self.calls.append(dict(results=results, query=query, sub_queries=sub_queries, kwargs=kwargs))
        return self.return_value


class FixedExpander(QueryExpander):
    def __init__(self, sqs):
        self._sqs = sqs

    def expand(self, query):
        return self._sqs


class FailingExpander(QueryExpander):
    def expand(self, query):
        raise RuntimeError("Expander failed")


class FailingPP(PostProcessor):
    def process(self, results, **kwargs):
        raise ValueError("Post-processor failed")


def _mock_kb(retrieve_result=None, metadata=(None, None), identity_ctx=None,
             pieces_side=None, search_side=None):
    kb = MagicMock()
    kb.retrieve.return_value = retrieve_result or _make_result()
    kb.retrieve_metadata.return_value = metadata
    kb.retrieve_identity_graph.return_value = identity_ctx or []
    if pieces_side is not None:
        kb.retrieve_pieces.side_effect = pieces_side
    else:
        kb.retrieve_pieces.return_value = []
    if search_side is not None:
        kb.retrieve_search_graph.side_effect = search_side
    else:
        kb.retrieve_search_graph.return_value = []
    return kb


# ---- Single-Query Path ----


class TestSingleQueryPath:
    """Requirements: 7.2, 7.4, 14.4"""

    def test_delegates_to_kb_retrieve(self):
        result = _make_result(num_pieces=3)
        kb = _mock_kb(retrieve_result=result)
        pp = RecordingPP("formatted")
        pipe = RetrievalPipeline(kb=kb, post_processor=pp)
        out = pipe.execute("test query", entity_id="user:1")
        assert out == "formatted"
        kb.retrieve.assert_called_once_with(query="test query", entity_id="user:1")
        kb.retrieve_metadata.assert_not_called()
        kb.retrieve_pieces.assert_not_called()
        kb.retrieve_search_graph.assert_not_called()
        kb.retrieve_identity_graph.assert_not_called()

    def test_passes_kwargs_to_retrieve(self):
        kb = _mock_kb()
        pp = RecordingPP()
        pipe = RetrievalPipeline(kb=kb, post_processor=pp)
        pipe.execute("q", entity_id="e", spaces=["main"], include_global=False)
        kb.retrieve.assert_called_once_with(
            query="q", entity_id="e", spaces=["main"], include_global=False
        )

    def test_passes_result_and_query_to_pp(self):
        result = _make_result()
        kb = _mock_kb(retrieve_result=result)
        pp = RecordingPP()
        pipe = RetrievalPipeline(kb=kb, post_processor=pp)
        pipe.execute("my query")
        assert len(pp.calls) == 1
        assert pp.calls[0]["results"] is result
        assert pp.calls[0]["query"] == "my query"

    def test_handles_empty_result(self):
        empty = RetrievalResult()
        kb = _mock_kb(retrieve_result=empty)
        pp = RecordingPP()
        pipe = RetrievalPipeline(kb=kb, post_processor=pp)
        pipe.execute("q")
        assert pp.calls[0]["results"] is empty


# ---- Multi-Query Path ----


class TestMultiQueryPath:
    """Requirements: 7.1, 7.3, 14.1, 14.2, 14.3"""

    def test_calls_l1_l3b_once(self):
        sqs = [SubQuery(query="sq1"), SubQuery(query="sq2"), SubQuery(query="sq3")]
        kb = _mock_kb(pieces_side=[[], [], []], search_side=[[], [], []])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q", entity_id="user:1")
        assert kb.retrieve_metadata.call_count == 1
        assert kb.retrieve_identity_graph.call_count == 1

    def test_calls_l2_l3a_per_sub_query(self):
        sqs = [SubQuery(query="sq1"), SubQuery(query="sq2")]
        kb = _mock_kb(pieces_side=[[], []], search_side=[[], []])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp, top_k=10)
        pipe.execute("q", entity_id="user:1")
        assert kb.retrieve_pieces.call_count == 2
        assert kb.retrieve_search_graph.call_count == 2

    def test_passes_sub_query_fields_to_retrieve_pieces(self):
        sqs = [SubQuery(query="sq1", domain="testing", tags=["t1"])]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp, top_k=7)
        pipe.execute("q", entity_id="user:1", spaces=["main"])
        kw = kb.retrieve_pieces.call_args.kwargs
        assert kw["query"] == "sq1"
        assert kw["domain"] == "testing"
        assert kw["tags"] == ["t1"]
        assert kw["top_k"] == 7
        assert kw["spaces"] == ["main"]

    def test_assembles_per_sub_query_results(self):
        sqs = [SubQuery(query="sq1"), SubQuery(query="sq2")]
        p1, p2 = _make_piece("p1"), _make_piece("p2")
        meta = EntityMetadata(entity_id="u", entity_type="user", properties={"k": "v"})
        id_ctx = [{"target_node_id": "n1", "relation_type": "IDENTITY", "depth": 0, "score": 1.0, "target_label": ""}]
        search_ctx = [{"target_node_id": "s1", "relation_type": "SEARCH_HIT", "depth": 0, "score": 0.7, "target_label": ""}]
        kb = _mock_kb(
            metadata=(meta, None), identity_ctx=id_ctx,
            pieces_side=[[(p1, 0.9)], [(p2, 0.8)]],
            search_side=[search_ctx, []],
        )
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q", entity_id="user:1")
        results = pp.calls[0]["results"]
        assert isinstance(results, list) and len(results) == 2
        assert results[0].metadata is not None and results[0].metadata.entity_id == "u"
        assert results[1].metadata is not None
        assert len(results[0].pieces) == 1 and results[0].pieces[0][0].piece_id == "p1"
        assert len(results[1].pieces) == 1 and results[1].pieces[0][0].piece_id == "p2"

    def test_passes_sub_queries_to_pp(self):
        sqs = [SubQuery(query="sq1")]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q")
        assert pp.calls[0]["sub_queries"] == sqs

    def test_builds_dedup_set_from_l2_pieces(self):
        sqs = [SubQuery(query="sq1")]
        p1 = _make_piece("p1", info_type="context")
        kb = _mock_kb(pieces_side=[[(p1, 0.9)]], search_side=[[]])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q")
        kw = kb.retrieve_search_graph.call_args.kwargs
        assert kw["already_retrieved_piece_ids"] == {"p1": "context"}

    def test_no_dedup_when_no_pieces(self):
        sqs = [SubQuery(query="sq1")]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q")
        kw = kb.retrieve_search_graph.call_args.kwargs
        assert kw["already_retrieved_piece_ids"] is None

    def test_graph_context_is_copied_per_result(self):
        sqs = [SubQuery(query="sq1"), SubQuery(query="sq2")]
        id_ctx = [{"target_node_id": "n1", "relation_type": "IDENTITY", "depth": 0, "score": 1.0, "target_label": ""}]
        kb = _mock_kb(identity_ctx=id_ctx, pieces_side=[[], []], search_side=[[], []])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q")
        results = pp.calls[0]["results"]
        results[0].graph_context.append({"extra": True})
        assert len(results[0].graph_context) != len(results[1].graph_context)


# ---- Fallback Logic ----


class TestFallbackLogic:
    """Requirements: 7.5"""

    def test_fallback_triggered_when_needs_fallback_true(self):
        sqs = [SubQuery(query="sq1")]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        first_out = AgenticRetrievalResult(needs_fallback=True)
        final_out = AgenticRetrievalResult(needs_fallback=False, used_fallback=True)
        call_count = [0]

        class TrackingPP(PostProcessor):
            def process(self, results, **kwargs):
                call_count[0] += 1
                return first_out if call_count[0] == 1 else final_out

        pipe = RetrievalPipeline(
            kb=kb, expander=FixedExpander(sqs),
            post_processor=TrackingPP(), min_results=5, top_k=10,
        )
        out = pipe.execute("q", entity_id="user:1")
        kb.retrieve.assert_called_once()
        assert out is final_out

    def test_no_fallback_when_needs_fallback_false(self):
        sqs = [SubQuery(query="sq1")]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        pp = RecordingPP(AgenticRetrievalResult(needs_fallback=False))
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q")
        kb.retrieve.assert_not_called()

    def test_no_fallback_for_string_output(self):
        kb = _mock_kb()
        pp = RecordingPP("formatted string")
        pipe = RetrievalPipeline(kb=kb, post_processor=pp)
        pipe.execute("q")
        assert kb.retrieve.call_count == 1

    def test_no_fallback_for_dict_output(self):
        kb = _mock_kb()
        pp = RecordingPP({"context": "data"})
        pipe = RetrievalPipeline(kb=kb, post_processor=pp)
        pipe.execute("q")
        assert kb.retrieve.call_count == 1

    def test_needs_fallback_checks_attribute(self):
        pipe = RetrievalPipeline(kb=MagicMock(), post_processor=RecordingPP())
        assert pipe._needs_fallback(AgenticRetrievalResult(needs_fallback=True)) is True
        assert pipe._needs_fallback(AgenticRetrievalResult(needs_fallback=False)) is False
        assert pipe._needs_fallback("string") is False
        assert pipe._needs_fallback({"dict": "out"}) is False
        assert pipe._needs_fallback(None) is False

    def test_fallback_passes_is_fallback_to_pp(self):
        sqs = [SubQuery(query="sq1")]
        fallback_result = _make_result(num_pieces=5)
        kb = _mock_kb(retrieve_result=fallback_result, pieces_side=[[]], search_side=[[]])
        call_args = []

        class TrackingPP(PostProcessor):
            def process(self, results, **kwargs):
                call_args.append(dict(results=results, kwargs=kwargs))
                if len(call_args) == 1:
                    return AgenticRetrievalResult(needs_fallback=True)
                return AgenticRetrievalResult(needs_fallback=False, used_fallback=True)

        pipe = RetrievalPipeline(
            kb=kb, expander=FixedExpander(sqs),
            post_processor=TrackingPP(), min_results=5, top_k=10,
        )
        pipe.execute("q", entity_id="user:1")
        assert len(call_args) == 2
        assert call_args[1]["kwargs"].get("is_fallback") is True
        assert isinstance(call_args[1]["results"], list)
        assert call_args[1]["results"][0] is fallback_result


# ---- Error Handling ----


class TestErrorHandling:
    """Requirements: 7.1, 7.5"""

    def test_expander_failure_propagates(self):
        kb = _mock_kb()
        pipe = RetrievalPipeline(kb=kb, expander=FailingExpander(), post_processor=RecordingPP())
        with pytest.raises(RuntimeError, match="Expander failed"):
            pipe.execute("q")

    def test_pp_error_is_re_raised_single_query(self):
        kb = _mock_kb()
        pipe = RetrievalPipeline(kb=kb, post_processor=FailingPP())
        with pytest.raises(ValueError, match="Post-processor failed"):
            pipe.execute("q")

    def test_pp_error_is_re_raised_multi_query(self):
        sqs = [SubQuery(query="sq1")]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=FailingPP())
        with pytest.raises(ValueError, match="Post-processor failed"):
            pipe.execute("q")


# ---- Call Count Verification ----


class TestCallCountVerification:
    """Requirements: 14.1, 14.2, 14.3"""

    def test_one_sub_query(self):
        sqs = [SubQuery(query="sq1")]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q", entity_id="user:1")
        assert kb.retrieve_metadata.call_count == 1
        assert kb.retrieve_identity_graph.call_count == 1
        assert kb.retrieve_pieces.call_count == 1
        assert kb.retrieve_search_graph.call_count == 1
        kb.retrieve.assert_not_called()

    def test_five_sub_queries(self):
        sqs = [SubQuery(query=f"sq{i}") for i in range(5)]
        kb = _mock_kb(pieces_side=[[] for _ in range(5)], search_side=[[] for _ in range(5)])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q")
        assert kb.retrieve_metadata.call_count == 1
        assert kb.retrieve_identity_graph.call_count == 1
        assert kb.retrieve_pieces.call_count == 5
        assert kb.retrieve_search_graph.call_count == 5

    def test_l2_l3a_use_correct_sub_query_params(self):
        sqs = [
            SubQuery(query="alpha", domain="d1", tags=["t1"]),
            SubQuery(query="beta", domain="d2", tags=["t2", "t3"]),
        ]
        kb = _mock_kb(pieces_side=[[], []], search_side=[[], []])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp, top_k=15)
        pipe.execute("q", entity_id="user:1", spaces=["main"])
        kw0 = kb.retrieve_pieces.call_args_list[0].kwargs
        assert kw0["query"] == "alpha" and kw0["domain"] == "d1" and kw0["tags"] == ["t1"]
        kw1 = kb.retrieve_pieces.call_args_list[1].kwargs
        assert kw1["query"] == "beta" and kw1["domain"] == "d2" and kw1["tags"] == ["t2", "t3"]
        assert kb.retrieve_search_graph.call_args_list[0].kwargs["query"] == "alpha"
        assert kb.retrieve_search_graph.call_args_list[1].kwargs["query"] == "beta"

    def test_l1_receives_entity_id_and_spaces(self):
        sqs = [SubQuery(query="sq1")]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q", entity_id="user:1", spaces=["main"], include_global=False)
        kb.retrieve_metadata.assert_called_once_with(
            entity_id="user:1", include_global=False, spaces=["main"],
        )

    def test_l3b_receives_entity_id_and_spaces(self):
        sqs = [SubQuery(query="sq1")]
        kb = _mock_kb(pieces_side=[[]], search_side=[[]])
        pp = RecordingPP(AgenticRetrievalResult())
        pipe = RetrievalPipeline(kb=kb, expander=FixedExpander(sqs), post_processor=pp)
        pipe.execute("q", entity_id="user:1", spaces=["main"])
        kb.retrieve_identity_graph.assert_called_once_with(
            entity_id="user:1", spaces=["main"],
        )
