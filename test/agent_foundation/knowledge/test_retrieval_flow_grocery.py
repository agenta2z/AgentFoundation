"""
Integration tests for the full retrieval pipeline with grocery store mock data.

Exercises all 4 retrieval layers (L1 metadata, L2 pieces, L3a search graph,
L3b identity graph), the GroupedDictPostProcessor, and the KnowledgeConsolidator
with a mock LLM function.

Uses the file-based mock knowledge store at ``_mock_knowledge_store/`` to validate
real term-overlap search with stemming, and the multi-edge graph walk fix.
"""
import sys
from pathlib import Path

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

_test_knowledge_dir = str(Path(__file__).resolve().parent)
if _test_knowledge_dir not in sys.path:
    sys.path.insert(0, _test_knowledge_dir)

import pytest

from agent_foundation.knowledge.retrieval.post_processors import GroupedDictPostProcessor
from agent_foundation.knowledge.retrieval.knowledge_consolidator import KnowledgeConsolidator
from agent_foundation.knowledge.retrieval.models.enums import ConsolidationMode

from _grocery_kb_helpers import load_grocery_kb, QUERY, USER_ENTITY_ID


# ── Test class ──────────────────────────────────────────────────────────────


class TestGroceryRetrievalFlow:
    """Integration tests for the full retrieval pipeline with grocery store data."""

    @pytest.fixture
    def grocery_kb(self, tmp_path):
        kb = load_grocery_kb(tmp_path)
        yield kb
        kb.close()

    # ── L1: Metadata ────────────────────────────────────────────────────

    def test_l1_metadata_returns_user_profile(self, grocery_kb):
        meta, global_meta = grocery_kb.retrieve_metadata(USER_ENTITY_ID)
        assert meta is not None
        assert meta.properties["name"] == "Alex Johnson"

    # ── L2: Knowledge Pieces ────────────────────────────────────────────

    def test_l2_returns_pieces(self, grocery_kb):
        pieces = grocery_kb.retrieve_pieces(QUERY)
        assert len(pieces) >= 1

    def test_l2_safeway_query_finds_safeway_membership(self, grocery_kb):
        pieces = grocery_kb.retrieve_pieces(QUERY)
        piece_ids = [p.piece_id for p, s in pieces]
        assert "alex-johnson-safeway-membership" in piece_ids

    def test_l2_stemming_improves_procedure_match(self, grocery_kb):
        """With stemming, 'prices' matches 'price'/'pricing' in the procedure piece."""
        pieces = grocery_kb.retrieve_pieces(QUERY)
        procedure = next(
            (p for p, s in pieces if p.piece_id == "grocery-store-shopping-procedure"),
            None,
        )
        assert procedure is not None, "Procedure piece should be found"
        procedure_score = next(
            s for p, s in pieces if p.piece_id == "grocery-store-shopping-procedure"
        )
        safeway_score = next(
            s for p, s in pieces if p.piece_id == "alex-johnson-safeway-membership"
        )
        # With stemming both should score similarly (prices→price matches content)
        assert procedure_score >= safeway_score * 0.9

    # ── L3a: Search-Based Graph ─────────────────────────────────────────

    def test_l3a_finds_safeway_node(self, grocery_kb):
        ctx = grocery_kb.retrieve_search_graph(QUERY)
        assert len(ctx) >= 1
        search_hits = [e for e in ctx if e["relation_type"] == "SEARCH_HIT"]
        safeway_hits = [
            e for e in search_hits
            if e["target_node_id"] == "service:safeway"
        ]
        assert len(safeway_hits) >= 1

    def test_l3a_walks_to_procedure_from_safeway(self, grocery_kb):
        ctx = grocery_kb.retrieve_search_graph(QUERY)
        uses_proc = [e for e in ctx if e["relation_type"] == "USES_PROCEDURE"]
        assert len(uses_proc) >= 1
        proc_ids = [e["target_node_id"] for e in uses_proc]
        assert "procedure:grocery-shopping" in proc_ids

    # ── L3b: Identity Graph ─────────────────────────────────────────────

    def test_l3b_finds_all_neighbors(self, grocery_kb):
        ctx = grocery_kb.retrieve_identity_graph(USER_ENTITY_ID)
        node_ids = {e["target_node_id"] for e in ctx}
        assert "service:safeway" in node_ids
        assert "service:qfc" in node_ids
        assert "service:whole-foods" in node_ids
        assert "procedure:grocery-shopping" in node_ids

    def test_l3b_multi_edge_captures_both_member_and_shops(self, grocery_kb):
        """After multi-edge fix, both MEMBER_OF and SHOPS_AT should appear."""
        ctx = grocery_kb.retrieve_identity_graph(USER_ENTITY_ID)
        safeway_rels = [
            e["relation_type"]
            for e in ctx
            if e["target_node_id"] == "service:safeway"
        ]
        assert "MEMBER_OF" in safeway_rels
        assert "SHOPS_AT" in safeway_rels

    # ── Full retrieve() ─────────────────────────────────────────────────

    def test_full_retrieve_produces_all_layers(self, grocery_kb):
        result = grocery_kb.retrieve(QUERY)
        assert result.metadata is not None  # L1
        assert len(result.pieces) >= 1  # L2
        assert len(result.graph_context) >= 1  # L3

    # ── GroupedDictPostProcessor ─────────────────────────────────────────

    def test_grouped_dict_routes_by_info_type(self, grocery_kb):
        result = grocery_kb.retrieve(QUERY)
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
        )
        output = processor.process(result, query=QUERY)
        # user_profile group should exist (metadata + membership pieces)
        assert "user_profile" in output
        # instructions group should exist (grocery procedure)
        assert "instructions" in output

    # ── KnowledgeConsolidator ────────────────────────────────────────────

    def test_consolidator_with_mock_llm(self, grocery_kb):
        """Full pipeline with a mock LLM consolidator."""

        def mock_llm(prompt: str) -> str:
            return "[CONSOLIDATED] Safeway membership and grocery shopping procedure."

        consolidator = KnowledgeConsolidator(
            llm_fn=mock_llm,
            mode=ConsolidationMode.ENABLED,
        )
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
            consolidator=consolidator,
        )
        result = grocery_kb.retrieve(QUERY)
        output = processor.process(result, query=QUERY)
        assert "consolidated_knowledge" in output
        assert "CONSOLIDATED" in output["consolidated_knowledge"]

    def test_consolidator_disabled_mode_skips(self, grocery_kb):
        call_count = {"n": 0}

        def mock_llm(prompt: str) -> str:
            call_count["n"] += 1
            return "should not be called"

        consolidator = KnowledgeConsolidator(
            llm_fn=mock_llm,
            mode=ConsolidationMode.DISABLED,
        )
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
            consolidator=consolidator,
        )
        result = grocery_kb.retrieve(QUERY)
        output = processor.process(result, query=QUERY)
        assert "consolidated_knowledge" not in output
        assert call_count["n"] == 0

    def test_consolidator_receives_correct_knowledge_content(self, grocery_kb):
        """Verify the LLM receives the right content for consolidation."""
        received_prompts = []

        def capture_llm(prompt: str) -> str:
            received_prompts.append(prompt)
            return "consolidated output"

        consolidator = KnowledgeConsolidator(
            llm_fn=capture_llm,
            mode=ConsolidationMode.ENABLED,
        )
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
            consolidator=consolidator,
        )
        result = grocery_kb.retrieve(QUERY)
        processor.process(result, query=QUERY)
        assert len(received_prompts) == 1
        prompt = received_prompts[0]
        # Prompt should contain the query (rendered by KnowledgeConsolidation.hbs template)
        assert "egg prices" in prompt.lower()
