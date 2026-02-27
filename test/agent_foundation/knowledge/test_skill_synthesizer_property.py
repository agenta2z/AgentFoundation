"""
Property-based tests for the skill synthesizer module.

Feature: knowledge-module-migration
- Property 23: Skill synthesis threshold gate
- Property 24: Synthesized skill piece type invariant

**Validates: Requirements 15.1, 15.3**
"""
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pytest
from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.ingestion.skill_synthesizer import (
    SkillSynthesisConfig,
    SkillSynthesizer,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


# ── Test Helpers ──────────────────────────────────────────────────────────────


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for property testing with controllable search scores."""

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


def _make_piece(content: str = "test content", tags: Optional[List[str]] = None, domain: str = "general") -> KnowledgePiece:
    return KnowledgePiece(content=content, tags=tags or [], domain=domain)


def _make_success_llm_fn(skill_name: str = "synthesized-skill"):
    """Create a mock LLM function that always returns a valid skill synthesis."""
    def llm_fn(prompt: str) -> str:
        return json.dumps({
            "is_skill_worthy": True,
            "confidence": 0.9,
            "reasoning": "Pieces form a coherent skill",
            "synthesized_skill": {
                "name": skill_name,
                "description": "A synthesized skill",
                "steps": [
                    {"step": 1, "description": "Step one"},
                    {"step": 2, "description": "Step two"},
                ],
            },
        })
    return llm_fn


# ── Strategies ────────────────────────────────────────────────────────────────

# Content strings for generating pieces
_content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=5,
    max_size=200,
)

# min_pieces_for_skill: at least 2 (need at least 1 neighbor + new piece)
_min_pieces_strategy = st.integers(min_value=2, max_value=10)

# min_avg_similarity: a threshold in (0, 1)
_similarity_threshold_strategy = st.floats(
    min_value=0.1, max_value=0.99, allow_nan=False, allow_infinity=False
)

# Number of neighbors in the store (for threshold gate tests)
_neighbor_count_strategy = st.integers(min_value=0, max_value=15)


# Feature: knowledge-module-migration, Property 23: Skill synthesis threshold gate


class TestSkillSynthesisThresholdGate:
    """Property 23: Skill synthesis threshold gate.

    For any new piece where the number of similar neighbors above
    min_avg_similarity is less than min_pieces_for_skill - 1,
    check_and_synthesize() should return None.

    **Validates: Requirements 15.1**
    """

    @given(
        content=_content_strategy,
        min_pieces=_min_pieces_strategy,
        min_sim=_similarity_threshold_strategy,
        num_neighbors=_neighbor_count_strategy,
    )
    @settings(max_examples=100)
    def test_too_few_high_scoring_neighbors_returns_none(
        self, content: str, min_pieces: int, min_sim: float, num_neighbors: int
    ):
        """When the number of neighbors with score >= min_avg_similarity is less
        than min_pieces_for_skill - 1, synthesis should return None.

        We set the store score above the threshold but ensure the count of
        neighbors is insufficient (num_neighbors < min_pieces - 1).

        **Validates: Requirements 15.1**
        """
        assume(len(content.strip()) > 0)
        # Ensure we have fewer neighbors than needed
        assume(num_neighbors < min_pieces - 1)

        # All neighbors score above threshold, but there aren't enough of them
        score = min(min_sim + 0.01, 1.0)
        pieces = [_make_piece(f"neighbor {i}") for i in range(num_neighbors)]
        store = InMemoryPieceStore(pieces=pieces, score=score)

        config = SkillSynthesisConfig(
            min_pieces_for_skill=min_pieces,
            min_avg_similarity=min_sim,
        )

        synthesizer = SkillSynthesizer(
            piece_store=store,
            llm_fn=_make_success_llm_fn(),
            config=config,
        )

        result = synthesizer.check_and_synthesize(_make_piece(content))
        assert result is None

    @given(
        content=_content_strategy,
        min_pieces=_min_pieces_strategy,
        min_sim=_similarity_threshold_strategy,
    )
    @settings(max_examples=100)
    def test_all_neighbors_below_threshold_returns_none(
        self, content: str, min_pieces: int, min_sim: float
    ):
        """When all neighbors score below min_avg_similarity, they are filtered
        out and the cluster is too small, so synthesis returns None.

        **Validates: Requirements 15.1**
        """
        assume(len(content.strip()) > 0)
        assume(min_sim > 0.05)  # Ensure room for a score below threshold

        # Put plenty of neighbors in the store, but all score below threshold
        score = max(min_sim - 0.01, 0.0)
        pieces = [_make_piece(f"neighbor {i}") for i in range(min_pieces + 5)]
        store = InMemoryPieceStore(pieces=pieces, score=score)

        config = SkillSynthesisConfig(
            min_pieces_for_skill=min_pieces,
            min_avg_similarity=min_sim,
        )

        synthesizer = SkillSynthesizer(
            piece_store=store,
            llm_fn=_make_success_llm_fn(),
            config=config,
        )

        result = synthesizer.check_and_synthesize(_make_piece(content))
        assert result is None

    @given(
        content=_content_strategy,
        min_pieces=_min_pieces_strategy,
    )
    @settings(max_examples=100)
    def test_empty_store_returns_none(self, content: str, min_pieces: int):
        """When the store is empty (zero neighbors), synthesis always returns None.

        **Validates: Requirements 15.1**
        """
        assume(len(content.strip()) > 0)

        store = InMemoryPieceStore(pieces=[], score=0.9)

        config = SkillSynthesisConfig(min_pieces_for_skill=min_pieces)

        synthesizer = SkillSynthesizer(
            piece_store=store,
            llm_fn=_make_success_llm_fn(),
            config=config,
        )

        result = synthesizer.check_and_synthesize(_make_piece(content))
        assert result is None


# Feature: knowledge-module-migration, Property 24: Synthesized skill piece type invariant


class TestSynthesizedSkillPieceTypeInvariant:
    """Property 24: Synthesized skill piece type invariant.

    For any successfully synthesized skill piece, its knowledge_type should be
    KnowledgeType.Procedure and its info_type should be "skills".

    **Validates: Requirements 15.3**
    """

    @given(
        content=_content_strategy,
        num_neighbors=st.integers(min_value=2, max_value=8),
        score=st.floats(min_value=0.76, max_value=1.0, allow_nan=False, allow_infinity=False),
        domain=st.sampled_from(["general", "model_optimization", "training_efficiency", "debugging"]),
    )
    @settings(max_examples=100)
    def test_synthesized_piece_has_procedure_type_and_skills_info_type(
        self, content: str, num_neighbors: int, score: float, domain: str
    ):
        """When synthesis succeeds, the result piece must have
        knowledge_type=Procedure and info_type="skills".

        We ensure the cluster meets thresholds (enough high-scoring neighbors)
        and provide a working LLM function that returns a valid skill.

        **Validates: Requirements 15.3**
        """
        assume(len(content.strip()) > 0)

        # Use default config: min_pieces=3, min_avg_similarity=0.75
        # num_neighbors >= 2 ensures candidates + new_piece >= 3
        pieces = [_make_piece(f"neighbor {i}", domain=domain) for i in range(num_neighbors)]
        store = InMemoryPieceStore(pieces=pieces, score=score)

        synthesizer = SkillSynthesizer(
            piece_store=store,
            llm_fn=_make_success_llm_fn(),
            config=SkillSynthesisConfig(),  # defaults: min_pieces=3, min_sim=0.75
        )

        result = synthesizer.check_and_synthesize(_make_piece(content, domain=domain))

        # Synthesis should succeed given enough high-scoring neighbors
        assert result is not None
        assert result.knowledge_type == KnowledgeType.Procedure
        assert result.info_type == "skills"

    @given(
        content=_content_strategy,
        num_neighbors=st.integers(min_value=2, max_value=8),
        score=st.floats(min_value=0.76, max_value=1.0, allow_nan=False, allow_infinity=False),
        tags=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=2, max_size=15),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_synthesized_piece_source_is_skill_synthesis(
        self, content: str, num_neighbors: int, score: float, tags: List[str]
    ):
        """When synthesis succeeds, the result piece must have source='skill_synthesis'.

        **Validates: Requirements 15.3**
        """
        assume(len(content.strip()) > 0)

        pieces = [_make_piece(f"neighbor {i}", tags=tags) for i in range(num_neighbors)]
        store = InMemoryPieceStore(pieces=pieces, score=score)

        synthesizer = SkillSynthesizer(
            piece_store=store,
            llm_fn=_make_success_llm_fn(),
            config=SkillSynthesisConfig(),
        )

        result = synthesizer.check_and_synthesize(_make_piece(content, tags=tags))

        assert result is not None
        assert result.source == "skill_synthesis"
