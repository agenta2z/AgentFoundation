"""Unit tests for the skill synthesizer module."""

import json
from typing import List, Optional, Tuple

import pytest

from agent_foundation.knowledge.ingestion.skill_synthesizer import (
    SkillSynthesisConfig,
    SkillSynthesisResult,
    SkillSynthesizer,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for testing."""

    def __init__(self, pieces: Optional[List[KnowledgePiece]] = None, score: float = 0.8):
        self._pieces: dict[str, KnowledgePiece] = {}
        self._score = score
        for p in (pieces or []):
            self._pieces[p.piece_id] = p

    def add(self, piece: KnowledgePiece) -> str:
        self._pieces[piece.piece_id] = piece
        return piece.piece_id

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        return self._pieces.get(piece_id)

    def update(self, piece: KnowledgePiece) -> bool:
        if piece.piece_id in self._pieces:
            self._pieces[piece.piece_id] = piece
            return True
        return False

    def remove(self, piece_id: str) -> bool:
        return self._pieces.pop(piece_id, None) is not None

    def search(
        self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5
    ) -> List[Tuple[KnowledgePiece, float]]:
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            results.append((p, self._score))
        return results[:top_k]

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            if knowledge_type is not None and p.knowledge_type != knowledge_type:
                continue
            results.append(p)
        return results


def _make_piece(content: str, tags: Optional[List[str]] = None, domain: str = "general") -> KnowledgePiece:
    return KnowledgePiece(
        content=content,
        tags=tags or [],
        domain=domain,
    )


def _make_llm_fn(is_skill_worthy: bool = True, skill_name: str = "test-skill"):
    """Create a mock LLM function that returns a valid synthesis response."""
    def llm_fn(prompt: str) -> str:
        if is_skill_worthy:
            return json.dumps({
                "is_skill_worthy": True,
                "confidence": 0.9,
                "reasoning": "Pieces form a coherent skill",
                "synthesized_skill": {
                    "name": skill_name,
                    "description": "A test synthesized skill",
                    "steps": [
                        {"step": 1, "description": "First step"},
                        {"step": 2, "description": "Second step"},
                    ],
                },
            })
        else:
            return json.dumps({
                "is_skill_worthy": False,
                "confidence": 0.2,
                "reasoning": "Pieces do not form a coherent skill",
                "synthesized_skill": None,
            })
    return llm_fn


class TestSkillSynthesisConfig:
    """Tests for SkillSynthesisConfig defaults and custom values."""

    def test_defaults(self):
        config = SkillSynthesisConfig()
        assert config.min_pieces_for_skill == 3
        assert config.min_avg_similarity == 0.75
        assert config.max_neighbors == 10

    def test_custom_values(self):
        config = SkillSynthesisConfig(
            min_pieces_for_skill=5,
            min_avg_similarity=0.80,
            max_neighbors=20,
        )
        assert config.min_pieces_for_skill == 5
        assert config.min_avg_similarity == 0.80
        assert config.max_neighbors == 20


class TestSkillSynthesisResult:
    """Tests for SkillSynthesisResult."""

    def test_basic_result(self):
        result = SkillSynthesisResult(
            is_skill_worthy=True,
            confidence=0.9,
            reasoning="Good skill",
        )
        assert result.is_skill_worthy is True
        assert result.confidence == 0.9
        assert result.synthesized_skill is None

    def test_result_with_skill(self):
        skill = {"name": "test", "steps": []}
        result = SkillSynthesisResult(
            is_skill_worthy=True,
            confidence=0.9,
            reasoning="Good skill",
            synthesized_skill=skill,
        )
        assert result.synthesized_skill == skill


class TestClusterDetection:
    """Tests for cluster detection and threshold gating (Req 15.1, 15.4)."""

    def test_too_few_neighbors_returns_none(self):
        """When fewer than min_pieces neighbors exist, return None."""
        # Only 1 piece in store + new piece = 2, below threshold of 3
        store = InMemoryPieceStore(
            pieces=[_make_piece("neighbor 1")],
            score=0.85,
        )
        synthesizer = SkillSynthesizer(
            piece_store=store,
            llm_fn=_make_llm_fn(),
        )
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None

    def test_no_neighbors_returns_none(self):
        """When store is empty, return None."""
        store = InMemoryPieceStore(pieces=[], score=0.9)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None

    def test_low_similarity_neighbors_filtered_out(self):
        """Neighbors below min_avg_similarity are filtered out."""
        # 3 pieces in store but score is below threshold
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.5)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None

    def test_avg_similarity_below_threshold_returns_none(self):
        """When avg similarity of candidates is below threshold, return None."""
        # We need a store that returns mixed scores. Use score=0.74 (below 0.75 threshold)
        # but above 0.75 for filtering... tricky. Let's use score exactly at boundary.
        # With score=0.75, candidates pass the filter, but avg_sim = 0.75 which is NOT < 0.75
        # So let's use 0.74 which fails the filter entirely.
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.74)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None

    def test_cluster_meets_thresholds_calls_llm(self):
        """When cluster meets thresholds, LLM is called for synthesis."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        calls = []

        def tracking_llm(prompt):
            calls.append(prompt)
            return _make_llm_fn()(prompt)

        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=tracking_llm)
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert len(calls) == 1
        assert result is not None


class TestLLMSynthesis:
    """Tests for LLM-based synthesis (Req 15.2, 15.4, 15.5)."""

    def test_llm_says_not_skill_worthy_returns_none(self):
        """When LLM determines pieces don't form a skill, return None."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        synthesizer = SkillSynthesizer(
            piece_store=store,
            llm_fn=_make_llm_fn(is_skill_worthy=False),
        )
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None

    def test_llm_failure_returns_none_and_logs_warning(self, caplog):
        """When LLM call fails, return None and log warning (Req 15.5)."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)

        def failing_llm(prompt):
            raise RuntimeError("LLM service unavailable")

        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=failing_llm)
        import logging
        with caplog.at_level(logging.WARNING):
            result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None
        assert any("Skill synthesis failed" in r.message for r in caplog.records)

    def test_llm_returns_invalid_json_returns_none(self, caplog):
        """When LLM returns invalid JSON, return None."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)

        def bad_json_llm(prompt):
            return "not valid json"

        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=bad_json_llm)
        import logging
        with caplog.at_level(logging.WARNING):
            result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None

    def test_no_llm_fn_returns_none(self, caplog):
        """When no LLM function is provided, return None (Req 15.5)."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=None)
        import logging
        with caplog.at_level(logging.WARNING):
            result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None


class TestSkillPieceCreation:
    """Tests for synthesized skill piece properties (Req 15.3)."""

    def test_skill_piece_has_procedure_type(self):
        """Synthesized skill must have knowledge_type=Procedure."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is not None
        assert result.knowledge_type == KnowledgeType.Procedure

    def test_skill_piece_has_skills_info_type(self):
        """Synthesized skill must have info_type='skills'."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is not None
        assert result.info_type == "skills"

    def test_skill_piece_aggregates_tags(self):
        """Synthesized skill must aggregate tags from source pieces."""
        pieces = [
            _make_piece("piece 1", tags=["python", "ml"]),
            _make_piece("piece 2", tags=["optimization"]),
            _make_piece("piece 3", tags=["python", "training"]),
        ]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        new_piece = _make_piece("new piece", tags=["deep-learning"])
        result = synthesizer.check_and_synthesize(new_piece)
        assert result is not None
        # Should contain tags from all source pieces (new_piece + neighbors)
        result_tags = set(result.tags)
        assert "python" in result_tags
        assert "ml" in result_tags
        assert "optimization" in result_tags
        assert "training" in result_tags
        assert "deep-learning" in result_tags

    def test_skill_piece_has_source_skill_synthesis(self):
        """Synthesized skill must have source='skill_synthesis'."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is not None
        assert result.source == "skill_synthesis"

    def test_skill_piece_content_includes_steps(self):
        """Synthesized skill content should include the steps from LLM."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        synthesizer = SkillSynthesizer(
            piece_store=store,
            llm_fn=_make_llm_fn(skill_name="my-cool-skill"),
        )
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is not None
        assert "my-cool-skill" in result.content
        assert "First step" in result.content
        assert "Second step" in result.content

    def test_skill_piece_inherits_domain_from_first_source(self):
        """Synthesized skill domain comes from the first source piece (new_piece)."""
        pieces = [_make_piece(f"piece {i}", domain="model_optimization") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        new_piece = _make_piece("new piece", domain="training_efficiency")
        result = synthesizer.check_and_synthesize(new_piece)
        assert result is not None
        # First source piece is new_piece
        assert result.domain == "training_efficiency"


class TestCustomConfig:
    """Tests for custom configuration values."""

    def test_higher_min_pieces_threshold(self):
        """With min_pieces=5, 3 neighbors + 1 new = 4 is not enough."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        config = SkillSynthesisConfig(min_pieces_for_skill=5)
        synthesizer = SkillSynthesizer(
            piece_store=store, llm_fn=_make_llm_fn(), config=config
        )
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        assert result is None

    def test_exact_threshold_boundary(self):
        """Score exactly at min_avg_similarity should pass the filter."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.75)
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        result = synthesizer.check_and_synthesize(_make_piece("new piece"))
        # score=0.75 >= 0.75 threshold, so candidates pass filter
        # avg_sim=0.75 which is NOT < 0.75, so it proceeds to LLM
        assert result is not None


class TestEntityIdScoping:
    """Tests for entity_id scoping in search."""

    def test_search_uses_new_piece_entity_id(self):
        """Search should use the new piece's entity_id for scoping."""
        pieces = [_make_piece(f"piece {i}") for i in range(3)]
        store = InMemoryPieceStore(pieces=pieces, score=0.85)
        search_calls = []
        original_search = store.search

        def tracking_search(query, entity_id=None, **kwargs):
            search_calls.append(entity_id)
            return original_search(query, entity_id=entity_id, **kwargs)

        store.search = tracking_search
        new_piece = KnowledgePiece(content="new piece", entity_id="user-123")
        synthesizer = SkillSynthesizer(piece_store=store, llm_fn=_make_llm_fn())
        synthesizer.check_and_synthesize(new_piece)
        assert search_calls[0] == "user-123"
