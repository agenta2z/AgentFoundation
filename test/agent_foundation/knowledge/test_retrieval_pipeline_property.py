"""
Property-based tests for the retrieval pipeline and post-processors.

Feature: retrieval-pipeline-refactor
- Property 14: Pipeline Path A Equivalence
- Property 15: Pipeline Path C Score Aggregation
- Property 16: Pipeline Fallback Trigger
- Property 17: Budget-Aware Output Respects Token Limits
- Property 19: Pipeline Multi-Query L1/L3b Singleton Execution
- Property 20: Pipeline Path B Equivalence (GroupedDictPostProcessor)

**Validates: Requirements 8.1, 8.2, 9.1, 9.2, 10.1, 10.2, 10.3, 11.1, 11.2, 11.3, 14.1, 14.2**
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, call

# Path resolution for imports
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

_test_dir = str(Path(__file__).resolve().parent)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from agent_foundation.knowledge.retrieval.formatter import (
    KnowledgeFormatter,
    RetrievalResult,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece
from agent_foundation.knowledge.retrieval.retrieval_pipeline import (
    AgenticRetrievalResult,
    PostProcessor,
    QueryExpander,
    RetrievalPipeline,
    SubQuery,
)
from agent_foundation.knowledge.retrieval.post_processors import (
    AggregatingPostProcessor,
    BudgetAwarePostProcessor,
    CONTEXT_BUDGET,
    FlatStringPostProcessor,
    GroupedDictPostProcessor,
)
from agent_foundation.knowledge.retrieval.utils import count_tokens

from conftest import knowledge_piece_strategy, entity_metadata_strategy


# ── Hypothesis strategies ────────────────────────────────────────────────────

_info_type_strategy = st.sampled_from(["user_profile", "instructions", "context", "skills", "episodic"])


@st.composite
def scored_piece_strategy(draw):
    """Generate a ScoredPiece with a random KnowledgePiece and score."""
    piece = draw(knowledge_piece_strategy())
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return ScoredPiece(piece=piece, score=score)


@st.composite
def sub_query_strategy(draw):
    """Generate a SubQuery with random query, domain, tags, weight."""
    query = draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip()))
    domain = draw(st.one_of(st.none(), st.sampled_from(["general", "testing", "data"])))
    tags = draw(st.one_of(st.none(), st.lists(st.text(min_size=1, max_size=10), max_size=3)))
    weight = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    return SubQuery(query=query, domain=domain, tags=tags, weight=weight)


@st.composite
def graph_context_entry_strategy(draw):
    """Generate a graph context entry dict."""
    _relation_type_strategy = st.sampled_from([
        "SEARCH_HIT", "RELATED", "WORKS_AT", "SELLS", "LOCATED_IN", "KNOWS", "IDENTITY",
    ])
    _identifier_text = st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
        min_size=1,
        max_size=30,
    )
    has_piece = draw(st.booleans())
    piece = None
    if has_piece:
        piece = draw(knowledge_piece_strategy())
    return {
        "relation_type": draw(_relation_type_strategy),
        "target_node_id": draw(_identifier_text),
        "target_label": draw(st.text(max_size=20)),
        "piece": piece,
        "depth": draw(st.integers(min_value=0, max_value=5)),
        "score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    }


@st.composite
def retrieval_result_strategy(draw):
    """Generate a RetrievalResult with random metadata, pieces, and graph context."""
    has_metadata = draw(st.booleans())
    has_global_metadata = draw(st.booleans())
    metadata = draw(entity_metadata_strategy()) if has_metadata else None
    global_metadata = draw(entity_metadata_strategy()) if has_global_metadata else None

    num_pieces = draw(st.integers(min_value=0, max_value=5))
    pieces = []
    for _ in range(num_pieces):
        piece = draw(knowledge_piece_strategy())
        score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        pieces.append((piece, score))

    num_ctx = draw(st.integers(min_value=0, max_value=5))
    graph_context = [draw(graph_context_entry_strategy()) for _ in range(num_ctx)]

    result = RetrievalResult()
    result.metadata = metadata
    result.global_metadata = global_metadata
    result.pieces = pieces
    result.graph_context = graph_context
    return result


# ── Property 14: Pipeline Path A Equivalence ─────────────────────────────────


class TestPipelinePathAEquivalence:
    """Feature: retrieval-pipeline-refactor, Property 14: Pipeline Path A Equivalence

    *For any* RetrievalResult, a RetrievalPipeline configured with no QueryExpander
    and a FlatStringPostProcessor shall produce output identical to
    KnowledgeFormatter().format(result).

    **Validates: Requirements 8.1, 8.2**
    """

    @given(result=retrieval_result_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_pipeline_path_a_equivalence(self, result: RetrievalResult):
        """Pipeline with FlatStringPostProcessor == KnowledgeFormatter.format()."""
        formatter = KnowledgeFormatter()
        expected = formatter.format(result)

        post_processor = FlatStringPostProcessor(formatter=formatter)
        actual = post_processor.process(result, query="test")

        assert actual == expected

    @given(result=retrieval_result_strategy())
    @settings(max_examples=100)
    def test_pipeline_path_a_equivalence_with_list_input(self, result: RetrievalResult):
        """When given a list, FlatStringPostProcessor uses the first element."""
        formatter = KnowledgeFormatter()
        expected = formatter.format(result)

        post_processor = FlatStringPostProcessor(formatter=formatter)
        actual = post_processor.process([result], query="test")

        assert actual == expected


# ── Property 15: Pipeline Path C Score Aggregation ───────────────────────────


class TestPipelinePathCScoreAggregation:
    """Feature: retrieval-pipeline-refactor, Property 15: Pipeline Path C Score Aggregation

    *For any* list of sub-query results with associated weights, the
    AggregatingPostProcessor shall produce aggregated scores matching the
    current AgenticRetriever._aggregate_scores() logic.

    **Validates: Requirements 10.1, 10.2**
    """

    @given(
        sub_queries=st.lists(sub_query_strategy(), min_size=1, max_size=4),
        strategy=st.sampled_from(["max", "sum", "weighted_sum"]),
    )
    @settings(max_examples=100)
    def test_pipeline_path_c_aggregation(
        self,
        sub_queries: List[SubQuery],
        strategy: str,
    ):
        """Aggregated scores match manual computation for all strategies."""
        # Build RetrievalResults with pieces that may overlap across sub-queries
        # Use a shared pool of piece_ids so some pieces appear in multiple results
        piece_pool = []
        for i in range(5):
            piece_pool.append(
                KnowledgePiece(
                    content=f"piece content {i}",
                    piece_id=f"piece_{i}",
                    info_type="context",
                )
            )

        results = []
        for sq in sub_queries:
            # Each sub-query gets 1-3 random pieces from the pool
            import random
            num = min(len(piece_pool), max(1, len(piece_pool) // 2))
            selected = piece_pool[:num]
            pieces = [
                (p, 0.5 + 0.1 * idx) for idx, p in enumerate(selected)
            ]
            r = RetrievalResult()
            r.pieces = pieces
            results.append(r)

        post_processor = AggregatingPostProcessor(
            aggregation_strategy=strategy,
            top_k=10,
            min_results=0,  # Don't trigger fallback
        )
        output = post_processor.process(results, sub_queries=sub_queries)

        assert isinstance(output, AgenticRetrievalResult)
        assert output.sub_queries == sub_queries
        assert not output.used_fallback

        # Manually compute expected scores
        piece_scores: Dict[str, List[float]] = {}
        for i, result in enumerate(results):
            sq = sub_queries[i] if i < len(sub_queries) else SubQuery(query="")
            weight = sq.weight
            for piece, score in result.pieces:
                pid = piece.piece_id
                if pid not in piece_scores:
                    piece_scores[pid] = []
                piece_scores[pid].append(score * weight)

        for sp in output.pieces:
            pid = sp.piece.piece_id
            assert pid in piece_scores
            scores = piece_scores[pid]
            if strategy == "max":
                expected = max(scores)
            else:  # sum or weighted_sum
                expected = sum(scores)
            assert abs(sp.score - expected) < 1e-9, (
                f"Score mismatch for {pid}: expected {expected}, got {sp.score}"
            )

        # Verify sorted by (-score, piece_id)
        for i in range(len(output.pieces) - 1):
            a, b = output.pieces[i], output.pieces[i + 1]
            assert (
                (-a.score, a.piece.piece_id) <= (-b.score, b.piece.piece_id)
            ), "Output not sorted by (-score, piece_id)"


# ── Property 16: Pipeline Fallback Trigger ───────────────────────────────────


class TestPipelineFallbackTrigger:
    """Feature: retrieval-pipeline-refactor, Property 16: Pipeline Fallback Trigger

    *For any* pipeline execution where the aggregated result contains fewer
    pieces than min_results, the pipeline shall execute a fallback retrieval
    with the original query and merge the fallback results into the output.

    **Validates: Requirements 10.3**
    """

    @given(
        min_results=st.integers(min_value=1, max_value=10),
        num_pieces=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100)
    def test_pipeline_fallback_trigger(
        self,
        min_results: int,
        num_pieces: int,
    ):
        """needs_fallback is True when aggregated pieces < min_results (before top_k)."""
        # Build a single RetrievalResult with num_pieces pieces
        pieces = []
        for i in range(num_pieces):
            piece = KnowledgePiece(
                content=f"content {i}",
                piece_id=f"piece_{i}",
                info_type="context",
            )
            pieces.append((piece, 0.5 + 0.1 * i))

        result = RetrievalResult()
        result.pieces = pieces

        sub_queries = [SubQuery(query="test query")]

        post_processor = AggregatingPostProcessor(
            aggregation_strategy="max",
            top_k=100,  # Large top_k so truncation doesn't affect the test
            min_results=min_results,
        )
        output = post_processor.process([result], sub_queries=sub_queries)

        # needs_fallback should be True when num_pieces < min_results
        if num_pieces < min_results:
            assert output.needs_fallback, (
                f"Expected needs_fallback=True when {num_pieces} < {min_results}"
            )
        else:
            assert not output.needs_fallback, (
                f"Expected needs_fallback=False when {num_pieces} >= {min_results}"
            )

    @given(
        num_primary=st.integers(min_value=0, max_value=3),
        num_fallback=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_fallback_merge_dedup(
        self,
        num_primary: int,
        num_fallback: int,
    ):
        """Fallback merge deduplicates by piece_id, keeping higher score."""
        # Create primary pieces
        primary_pieces = []
        for i in range(num_primary):
            piece = KnowledgePiece(
                content=f"primary {i}",
                piece_id=f"piece_{i}",
                info_type="context",
            )
            primary_pieces.append((piece, 0.8))

        # Create fallback pieces (some overlap with primary)
        fallback_pieces = []
        for i in range(num_fallback):
            piece = KnowledgePiece(
                content=f"fallback {i}",
                piece_id=f"piece_{i}",  # Overlapping IDs
                info_type="context",
            )
            fallback_pieces.append((piece, 0.3))

        primary_result = RetrievalResult()
        primary_result.pieces = primary_pieces

        fallback_result = RetrievalResult()
        fallback_result.pieces = fallback_pieces

        sub_queries = [SubQuery(query="test")]

        post_processor = AggregatingPostProcessor(
            aggregation_strategy="max",
            top_k=100,
            min_results=0,
        )

        # Process as fallback: first result is fallback, rest are per-sub-query
        output = post_processor.process(
            [fallback_result, primary_result],
            sub_queries=sub_queries,
            is_fallback=True,
        )

        assert output.used_fallback

        # Verify dedup: no duplicate piece_ids
        seen_ids = set()
        for sp in output.pieces:
            assert sp.piece.piece_id not in seen_ids, (
                f"Duplicate piece_id: {sp.piece.piece_id}"
            )
            seen_ids.add(sp.piece.piece_id)

        # For overlapping pieces, the higher score should win
        for sp in output.pieces:
            pid = sp.piece.piece_id
            idx = int(pid.split("_")[1])
            if idx < num_primary:
                # Overlapping piece: primary score (0.8) > fallback score (0.3)
                assert sp.score == 0.8, (
                    f"Expected score 0.8 for overlapping piece {pid}, got {sp.score}"
                )


# ── Property 17: Budget-Aware Output Respects Token Limits ───────────────────


class TestBudgetAwareTokenLimits:
    """Feature: retrieval-pipeline-refactor, Property 17: Budget-Aware Output Respects Token Limits

    *For any* set of ScoredPiece objects and any available_tokens budget, the
    BudgetAwarePostProcessor shall produce output whose total token count does
    not exceed available_tokens, and sections shall appear in the CONTEXT_BUDGET
    priority order.

    **Validates: Requirements 11.1, 11.2, 11.3**
    """

    @given(
        available_tokens=st.integers(min_value=10, max_value=5000),
        num_pieces=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_budget_aware_token_limits(
        self,
        available_tokens: int,
        num_pieces: int,
    ):
        """Output token count does not exceed available_tokens (plus separator overhead).

        The BudgetAwarePostProcessor joins sections with "\\n\\n" separators.
        These separators add a small number of characters not counted against
        the per-section budget. This matches the existing
        BudgetAwareKnowledgeProvider behavior.
        """
        # Generate pieces across different info_types
        info_types = list(CONTEXT_BUDGET.keys())
        pieces = []
        for i in range(num_pieces):
            info_type = info_types[i % len(info_types)]
            piece = KnowledgePiece(
                content=f"This is content for piece number {i} with some text to fill space " * 3,
                piece_id=f"piece_{i}",
                info_type=info_type,
                updated_at="2024-01-15T10:00:00+00:00",
            )
            pieces.append((piece, 0.5 + 0.05 * i))

        result = RetrievalResult()
        result.pieces = pieces

        post_processor = BudgetAwarePostProcessor(
            available_tokens=available_tokens,
        )
        output = post_processor.process(result)

        assert isinstance(output, str)

        token_count = count_tokens(output)
        # Allow up to 5 tokens of separator overhead ("\n\n" between up to 5 sections)
        # This matches the existing BudgetAwareKnowledgeProvider behavior where
        # section separators are not counted against the budget.
        max_separator_overhead = len(CONTEXT_BUDGET)  # max 5 sections
        assert token_count <= available_tokens + max_separator_overhead, (
            f"Output tokens ({token_count}) exceed budget ({available_tokens}) "
            f"+ max separator overhead ({max_separator_overhead})"
        )

    @given(
        num_pieces=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=100)
    def test_budget_aware_priority_order(
        self,
        num_pieces: int,
    ):
        """Sections appear in CONTEXT_BUDGET priority order."""
        # Create pieces for each info_type
        priority_order = list(CONTEXT_BUDGET.keys())
        pieces = []
        for i, info_type in enumerate(priority_order):
            for j in range(max(1, num_pieces // len(priority_order))):
                piece = KnowledgePiece(
                    content=f"Content for {info_type} piece {j}",
                    piece_id=f"{info_type}_{j}",
                    info_type=info_type,
                    updated_at="2024-01-15T10:00:00+00:00",
                )
                pieces.append((piece, 0.9 - 0.1 * i))

        result = RetrievalResult()
        result.pieces = pieces

        post_processor = BudgetAwarePostProcessor(
            available_tokens=50000,  # Large budget to include all sections
        )
        output = post_processor.process(result)

        # Verify sections appear in priority order
        # Each section has a header like "## Available Skills", "## Instructions", etc.
        section_headers = {
            "skills": "## Available Skills",
            "instructions": "## Instructions",
            "context": "## Relevant Context",
            "episodic": "## Recent History",
            "user_profile": "## User Preferences",
        }

        last_pos = -1
        for info_type in priority_order:
            header = section_headers[info_type]
            pos = output.find(header)
            if pos >= 0:
                assert pos > last_pos, (
                    f"Section '{info_type}' at pos {pos} appears before "
                    f"previous section at pos {last_pos}"
                )
                last_pos = pos


# ── Property 19: Pipeline Multi-Query L1/L3b Singleton Execution ─────────────


class TestPipelineL1L3bSingletonExecution:
    """Feature: retrieval-pipeline-refactor, Property 19: Pipeline Multi-Query L1/L3b Singleton Execution

    *For any* pipeline execution with a QueryExpander that produces N sub-queries
    (N >= 1), retrieve_metadata() and retrieve_identity_graph() shall each be
    called exactly once, while retrieve_pieces() and retrieve_search_graph()
    shall each be called exactly N times.

    **Validates: Requirements 14.1, 14.2**
    """

    @given(
        num_sub_queries=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_pipeline_l1_l3b_singleton_execution(
        self,
        num_sub_queries: int,
    ):
        """L1/L3b called once, L2/L3a called N times."""
        sub_queries = [
            SubQuery(query=f"sub query {i}", weight=1.0)
            for i in range(num_sub_queries)
        ]

        # Create a mock expander
        class MockExpander(QueryExpander):
            def expand(self, query: str) -> List[SubQuery]:
                return sub_queries

        # Create a mock KB with tracked calls
        mock_kb = MagicMock()
        mock_kb.retrieve_metadata.return_value = (None, None)
        mock_kb.retrieve_identity_graph.return_value = []
        mock_kb.retrieve_pieces.return_value = []
        mock_kb.retrieve_search_graph.return_value = []

        # Create a simple post-processor that returns the results as-is
        class PassthroughPostProcessor(PostProcessor):
            def process(self, results, **kwargs):
                if isinstance(results, list):
                    return AgenticRetrievalResult(pieces=[], sub_queries=sub_queries)
                return results

        pipeline = RetrievalPipeline(
            kb=mock_kb,
            expander=MockExpander(),
            post_processor=PassthroughPostProcessor(),
            top_k=10,
        )

        pipeline.execute("test query", entity_id="user_1")

        # Verify L1 (retrieve_metadata) called exactly once
        assert mock_kb.retrieve_metadata.call_count == 1, (
            f"retrieve_metadata called {mock_kb.retrieve_metadata.call_count} times, expected 1"
        )

        # Verify L3b (retrieve_identity_graph) called exactly once
        assert mock_kb.retrieve_identity_graph.call_count == 1, (
            f"retrieve_identity_graph called {mock_kb.retrieve_identity_graph.call_count} times, expected 1"
        )

        # Verify L2 (retrieve_pieces) called N times
        assert mock_kb.retrieve_pieces.call_count == num_sub_queries, (
            f"retrieve_pieces called {mock_kb.retrieve_pieces.call_count} times, expected {num_sub_queries}"
        )

        # Verify L3a (retrieve_search_graph) called N times
        assert mock_kb.retrieve_search_graph.call_count == num_sub_queries, (
            f"retrieve_search_graph called {mock_kb.retrieve_search_graph.call_count} times, expected {num_sub_queries}"
        )


# ── Property 20: Pipeline Path B Equivalence ─────────────────────────────────


class TestPipelinePathBEquivalence:
    """Feature: retrieval-pipeline-refactor, Property 20: Pipeline Path B Equivalence (GroupedDictPostProcessor)

    *For any* RetrievalResult containing metadata, pieces, and graph context entries,
    a GroupedDictPostProcessor shall produce output equivalent to
    KnowledgeProvider.__call__() for the same input. Specifically:
    (1) metadata routes to metadata_info_type,
    (2) pieces route by piece.info_type,
    (3) graph edges with linked pieces route by linked_piece.info_type,
    (4) depth-1 edges from the user node (no linked piece) route to "user_profile",
    (5) other edges route to "context",
    (6) each group is formatted by the matching per-type formatter.

    **Validates: Requirements 9.1, 9.2**
    """

    @given(result=retrieval_result_strategy())
    @settings(max_examples=100)
    def test_pipeline_path_b_equivalence(self, result: RetrievalResult):
        """GroupedDictPostProcessor grouping matches KnowledgeProvider._group_by_info_type()."""
        active_entity_id = "user_123"

        # Use a simple formatter that just counts items per group
        def counting_formatter(metadata, pieces, graph_context):
            parts = []
            if metadata and hasattr(metadata, "properties") and metadata.properties:
                parts.append(f"meta:{len(metadata.properties)}")
            if pieces:
                parts.append(f"pieces:{len(pieces)}")
            if graph_context:
                parts.append(f"graph:{len(graph_context)}")
            return ",".join(parts) if parts else ""

        post_processor = GroupedDictPostProcessor(
            type_formatters={},
            default_formatter=counting_formatter,
            metadata_info_type="user_profile",
            active_entity_id=active_entity_id,
        )

        output = post_processor.process(result, query="test")
        assert isinstance(output, dict)

        # Manually compute expected grouping
        expected_groups: Dict[str, dict] = {}

        def _ensure(it: str):
            if it not in expected_groups:
                expected_groups[it] = {"metadata": None, "pieces": [], "graph_context": []}
            return expected_groups[it]

        # Metadata routing
        if result.metadata and result.metadata.properties:
            _ensure("user_profile")["metadata"] = result.metadata

        if result.global_metadata and result.global_metadata.properties:
            g = _ensure("user_profile")
            if g["metadata"] is None:
                g["metadata"] = result.global_metadata

        # Piece routing
        for piece, score in result.pieces:
            info_type = piece.info_type or "context"
            _ensure(info_type)["pieces"].append((piece, score))

        # Graph edge routing
        for edge in result.graph_context:
            linked_piece = edge.get("piece")
            if linked_piece is not None and hasattr(linked_piece, "info_type"):
                info_type = linked_piece.info_type or "context"
                _ensure(info_type)["graph_context"].append(edge)
            elif edge.get("depth", 0) == 1 and active_entity_id is not None:
                _ensure("user_profile")["graph_context"].append(edge)
            else:
                _ensure("context")["graph_context"].append(edge)

        # Verify same keys
        assert set(output.keys()) == set(expected_groups.keys()), (
            f"Key mismatch: output={set(output.keys())}, expected={set(expected_groups.keys())}"
        )

        # Verify each group has the right counts
        for info_type, group_data in expected_groups.items():
            expected_formatted = counting_formatter(
                group_data["metadata"],
                group_data["pieces"],
                group_data["graph_context"],
            )
            assert output[info_type] == expected_formatted, (
                f"Mismatch for {info_type}: output={output[info_type]!r}, "
                f"expected={expected_formatted!r}"
            )

    @given(result=retrieval_result_strategy())
    @settings(max_examples=100)
    def test_grouped_dict_consolidation(self, result: RetrievalResult):
        """When consolidator is set, it is called with (query, output)."""
        consolidation_called = []

        class MockConsolidator:
            def consolidate(self, query, output):
                consolidation_called.append((query, output))
                return {k: v.upper() if isinstance(v, str) else v for k, v in output.items()}

        def simple_formatter(metadata, pieces, graph_context):
            return "formatted"

        post_processor = GroupedDictPostProcessor(
            default_formatter=simple_formatter,
            consolidator=MockConsolidator(),
        )

        # Only test if result has any data to produce groups
        has_data = (
            (result.metadata and result.metadata.properties)
            or (result.global_metadata and result.global_metadata.properties)
            or result.pieces
            or result.graph_context
        )

        output = post_processor.process(result, query="test query")

        if has_data:
            assert len(consolidation_called) == 1
            assert consolidation_called[0][0] == "test query"
            # All values should be uppercased by the mock consolidator
            for v in output.values():
                assert v == v.upper()
        else:
            # No groups → consolidator not called (empty dict)
            assert output == {}
