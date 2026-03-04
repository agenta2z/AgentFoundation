"""Unit tests for KnowledgeUpdater."""

import json
from typing import List, Optional, Tuple

import pytest

from agent_foundation.knowledge.ingestion.knowledge_updater import (
    KnowledgeUpdater,
    UpdateConfig,
)
from agent_foundation.knowledge.prompt_templates import render_prompt
from agent_foundation.knowledge.retrieval.models.enums import UpdateAction
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for testing."""

    def __init__(
        self,
        pieces: Optional[List[KnowledgePiece]] = None,
        search_score: float = 0.90,
    ):
        self._pieces = {p.piece_id: p for p in (pieces or [])}
        self._search_score = search_score
        self._fail_on_update = False

    def add(self, piece: KnowledgePiece) -> str:
        self._pieces[piece.piece_id] = piece
        return piece.piece_id

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        return self._pieces.get(piece_id)

    def update(self, piece: KnowledgePiece) -> bool:
        if self._fail_on_update:
            raise RuntimeError("Simulated update failure")
        if piece.piece_id in self._pieces:
            self._pieces[piece.piece_id] = piece
            return True
        return False

    def remove(self, piece_id: str) -> bool:
        if piece_id in self._pieces:
            del self._pieces[piece_id]
            return True
        return False

    def search(
        self,
        query: str,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
        tags: List[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[KnowledgePiece, float]]:
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            results.append((p, self._search_score))
        return results[:top_k]

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        return list(self._pieces.values())


def _make_piece(
    content: str = "Original content",
    piece_id: str = None,
    domain: str = "general",
    tags: Optional[List[str]] = None,
    version: int = 1,
) -> KnowledgePiece:
    return KnowledgePiece(
        content=content,
        piece_id=piece_id,
        domain=domain,
        tags=tags or ["test"],
        version=version,
    )


def _make_replace_llm_fn():
    """LLM that always returns replace action."""
    def llm_fn(prompt: str) -> str:
        return json.dumps({
            "action": "replace",
            "confidence": 0.9,
            "reasoning": "New content is an update",
            "changes_summary": "Replaced content",
        })
    return llm_fn


def _make_merge_llm_fn(strategy: str = "append"):
    """LLM that always returns merge action."""
    def llm_fn(prompt: str) -> str:
        return json.dumps({
            "action": "merge",
            "confidence": 0.8,
            "reasoning": "Content should be combined",
            "merge_strategy": strategy,
            "changes_summary": "Merged content",
        })
    return llm_fn


def _make_no_change_llm_fn():
    """LLM that always returns no_change action."""
    def llm_fn(prompt: str) -> str:
        return json.dumps({
            "action": "no_change",
            "confidence": 0.7,
            "reasoning": "No update needed",
            "changes_summary": "",
        })
    return llm_fn


# ── UpdateConfig Tests ───────────────────────────────────────────────


class TestUpdateConfig:
    def test_defaults(self):
        config = UpdateConfig()
        assert config.similarity_threshold == 0.80
        assert config.max_candidates == 20
        assert config.max_updates == 3
        assert config.require_confirmation is False
        assert config.preserve_history is True

    def test_custom_values(self):
        config = UpdateConfig(
            similarity_threshold=0.95,
            max_candidates=10,
            max_updates=1,
            require_confirmation=True,
            preserve_history=False,
        )
        assert config.similarity_threshold == 0.95
        assert config.max_updates == 1
        assert config.preserve_history is False


# ── update_by_id Tests ───────────────────────────────────────────────


class TestUpdateById:
    def test_piece_not_found_returns_failure(self):
        store = InMemoryPieceStore()
        updater = KnowledgeUpdater(piece_store=store)
        result = updater.update_by_id("nonexistent", "new content")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_replace_without_llm_preserves_history(self):
        piece = _make_piece("Old content")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(piece_store=store)

        result = updater.update_by_id(piece.piece_id, "New content")

        assert result.success is True
        assert result.old_version == 1
        assert result.new_version == 2
        # Old piece should be deactivated
        old = store.get_by_id(piece.piece_id)
        assert old.is_active is False
        # New piece should exist with supersedes link
        new = store.get_by_id(result.piece_id)
        assert new is not None
        assert new.content == "New content"
        assert new.supersedes == piece.piece_id
        assert new.version == 2

    def test_replace_without_llm_no_history(self):
        piece = _make_piece("Old content")
        store = InMemoryPieceStore(pieces=[piece])
        config = UpdateConfig(preserve_history=False)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        result = updater.update_by_id(piece.piece_id, "New content")

        assert result.success is True
        assert result.piece_id == piece.piece_id
        updated = store.get_by_id(piece.piece_id)
        assert updated.content == "New content"
        assert updated.version == 2

    def test_replace_with_llm(self):
        piece = _make_piece("Old content")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store, llm_fn=_make_replace_llm_fn()
        )

        result = updater.update_by_id(piece.piece_id, "New content")

        assert result.success is True
        assert result.details["action"] == "replace"

    def test_merge_with_llm_append(self):
        piece = _make_piece("Part A")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store, llm_fn=_make_merge_llm_fn("append")
        )

        result = updater.update_by_id(piece.piece_id, "Part B")

        assert result.success is True
        new = store.get_by_id(result.piece_id)
        assert "Part A" in new.content
        assert "Part B" in new.content

    def test_merge_with_llm_prepend(self):
        piece = _make_piece("Part A")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store, llm_fn=_make_merge_llm_fn("prepend")
        )

        result = updater.update_by_id(piece.piece_id, "Part B")

        assert result.success is True
        new = store.get_by_id(result.piece_id)
        assert new.content.startswith("Part B")

    def test_merge_with_llm_interleave(self):
        piece = _make_piece("Para1\n\nPara2")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store, llm_fn=_make_merge_llm_fn("interleave")
        )

        result = updater.update_by_id(piece.piece_id, "NewA\n\nNewB")

        assert result.success is True
        new = store.get_by_id(result.piece_id)
        assert "Para1" in new.content
        assert "NewA" in new.content

    def test_no_change_from_llm(self):
        piece = _make_piece("Content")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store, llm_fn=_make_no_change_llm_fn()
        )

        result = updater.update_by_id(piece.piece_id, "New content")

        assert result.success is False
        assert "reason" in result.details

    def test_llm_failure_defaults_to_no_change(self):
        piece = _make_piece("Content")
        store = InMemoryPieceStore(pieces=[piece])

        def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        updater = KnowledgeUpdater(piece_store=store, llm_fn=failing_llm)
        result = updater.update_by_id(piece.piece_id, "New content")

        assert result.success is False

    def test_content_hash_recomputed_in_place(self):
        piece = _make_piece("Old content")
        old_hash = piece.content_hash
        store = InMemoryPieceStore(pieces=[piece])
        config = UpdateConfig(preserve_history=False)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        updater.update_by_id(piece.piece_id, "Completely new content")

        updated = store.get_by_id(piece.piece_id)
        assert updated.content_hash != old_hash
        assert updated.content_hash == updated._compute_content_hash()

    def test_embedding_recomputed_on_update(self):
        piece = _make_piece("Old content")
        store = InMemoryPieceStore(pieces=[piece])
        config = UpdateConfig(preserve_history=False)

        def mock_embedding_fn(text: str) -> List[float]:
            return [0.1, 0.2, 0.3]

        updater = KnowledgeUpdater(
            piece_store=store,
            embedding_fn=mock_embedding_fn,
            config=config,
        )

        updater.update_by_id(piece.piece_id, "New content")

        updated = store.get_by_id(piece.piece_id)
        assert updated.embedding == [0.1, 0.2, 0.3]

    def test_embedding_computed_for_new_piece_with_history(self):
        piece = _make_piece("Old content")
        store = InMemoryPieceStore(pieces=[piece])

        def mock_embedding_fn(text: str) -> List[float]:
            return [0.4, 0.5, 0.6]

        updater = KnowledgeUpdater(
            piece_store=store, embedding_fn=mock_embedding_fn
        )

        result = updater.update_by_id(piece.piece_id, "New content")

        new = store.get_by_id(result.piece_id)
        assert new.embedding == [0.4, 0.5, 0.6]

    def test_domain_and_tags_passed_through(self):
        piece = _make_piece("Content", domain="testing", tags=["old"])
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(piece_store=store)

        result = updater.update_by_id(
            piece.piece_id, "New", domain="debugging", tags=["new"]
        )

        new = store.get_by_id(result.piece_id)
        assert new.domain == "debugging"
        assert new.tags == ["new"]


# ── update_by_content Tests ──────────────────────────────────────────


class TestUpdateByContent:
    def test_no_matches_returns_empty(self):
        store = InMemoryPieceStore(search_score=0.5)
        piece = _make_piece("Some content")
        store.add(piece)
        updater = KnowledgeUpdater(piece_store=store)

        results = updater.update_by_content("Unrelated content")

        assert results == []

    def test_matches_above_threshold_updated(self):
        piece = _make_piece("Similar content")
        store = InMemoryPieceStore(pieces=[piece], search_score=0.90)
        updater = KnowledgeUpdater(piece_store=store)

        results = updater.update_by_content("Updated similar content")

        assert len(results) == 1
        assert results[0].success is True

    def test_max_updates_respected(self):
        pieces = [_make_piece(f"Content {i}") for i in range(5)]
        store = InMemoryPieceStore(pieces=pieces, search_score=0.90)
        config = UpdateConfig(max_updates=2)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        results = updater.update_by_content("New content")

        assert len(results) <= 2

    def test_domain_filter_applied(self):
        p1 = _make_piece("Content A", domain="testing")
        p2 = _make_piece("Content B", domain="debugging")
        store = InMemoryPieceStore(pieces=[p1, p2], search_score=0.90)
        updater = KnowledgeUpdater(piece_store=store)

        results = updater.update_by_content(
            "New content", domain="testing"
        )

        # Only p1 should match the domain filter
        assert len(results) == 1

    def test_inactive_pieces_filtered_out(self):
        piece = _make_piece("Content")
        piece.is_active = False
        store = InMemoryPieceStore(pieces=[piece], search_score=0.90)
        updater = KnowledgeUpdater(piece_store=store)

        results = updater.update_by_content("New content")

        assert results == []

    def test_with_llm_no_change_skipped(self):
        piece = _make_piece("Content")
        store = InMemoryPieceStore(pieces=[piece], search_score=0.90)
        updater = KnowledgeUpdater(
            piece_store=store, llm_fn=_make_no_change_llm_fn()
        )

        results = updater.update_by_content("New content")

        assert results == []

    def test_without_llm_merges_content(self):
        piece = _make_piece("Original")
        store = InMemoryPieceStore(pieces=[piece], search_score=0.90)
        updater = KnowledgeUpdater(piece_store=store)

        results = updater.update_by_content("Addition")

        assert len(results) == 1
        new = store.get_by_id(results[0].piece_id)
        assert "Original" in new.content
        assert "Addition" in new.content


# ── Rollback Tests ───────────────────────────────────────────────────


class TestAtomicUpdateRollback:
    def test_rollback_on_deactivation_failure(self):
        piece = _make_piece("Content")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(piece_store=store)

        # Count pieces before
        pieces_before = len(store._pieces)

        # Make update fail during deactivation (after add succeeds)
        original_update = store.update

        call_count = [0]

        def failing_update(p):
            call_count[0] += 1
            # First call is the deactivation of old piece — fail it
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure")
            return original_update(p)

        store.update = failing_update

        result = updater.update_by_id(piece.piece_id, "New content")

        assert result.success is False
        assert "failed" in result.error.lower()
        # The new piece should have been rolled back (removed)
        # Only the original piece should remain
        assert len(store._pieces) == pieces_before

    def test_new_piece_inherits_metadata(self):
        piece = _make_piece(
            "Content",
            domain="testing",
            tags=["tag1", "tag2"],
        )
        piece.secondary_domains = ["debugging"]
        piece.custom_tags = ["custom1"]
        piece.entity_id = "entity-1"
        piece.source = "test-source"
        piece.space = "personal"
        piece.spaces = ["personal"]
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(piece_store=store)

        result = updater.update_by_id(piece.piece_id, "New content")

        new = store.get_by_id(result.piece_id)
        assert new.domain == "testing"
        assert new.tags == ["tag1", "tag2"]
        assert new.secondary_domains == ["debugging"]
        assert new.custom_tags == ["custom1"]
        assert new.entity_id == "entity-1"
        assert new.source == "test-source"
        assert new.space == "personal"


# ── _compute_final_content Tests ─────────────────────────────────────


# ── Spaces Preservation Tests ────────────────────────────────────────


class TestSpacesPreservation:
    """Tests for KnowledgeUpdater preserving the spaces field (Req 16.1, 16.2)."""

    def test_preserve_history_copies_spaces_to_new_piece(self):
        """In preserve_history mode, the new piece should inherit spaces from the existing piece."""
        piece = _make_piece("Original content")
        piece.spaces = ["personal", "main"]
        piece.space = "personal"
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(piece_store=store)

        result = updater.update_by_id(piece.piece_id, "Updated content")

        assert result.success
        new_piece = store.get_by_id(result.piece_id)
        assert new_piece.spaces == ["personal", "main"]
        assert new_piece.space == "personal"

    def test_preserve_history_copies_single_space(self):
        """Single-space pieces should also have spaces preserved."""
        piece = _make_piece("Original content")
        piece.spaces = ["developmental"]
        piece.space = "developmental"
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(piece_store=store)

        result = updater.update_by_id(piece.piece_id, "Updated content")

        assert result.success
        new_piece = store.get_by_id(result.piece_id)
        assert new_piece.spaces == ["developmental"]
        assert new_piece.space == "developmental"

    def test_in_place_update_preserves_spaces(self):
        """In-place update mode should not overwrite the existing spaces field."""
        piece = _make_piece("Original content")
        piece.spaces = ["personal", "main"]
        piece.space = "personal"
        store = InMemoryPieceStore(pieces=[piece])
        config = UpdateConfig(preserve_history=False)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        result = updater.update_by_id(piece.piece_id, "Updated content")

        assert result.success
        updated = store.get_by_id(result.piece_id)
        assert updated.spaces == ["personal", "main"]
        assert updated.space == "personal"

    def test_preserve_history_spaces_are_independent_copy(self):
        """The new piece's spaces should be an independent copy, not a reference."""
        piece = _make_piece("Original content")
        piece.spaces = ["personal", "main"]
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(piece_store=store)

        result = updater.update_by_id(piece.piece_id, "Updated content")

        new_piece = store.get_by_id(result.piece_id)
        # Mutating the new piece's spaces should not affect the original
        new_piece.spaces.append("developmental")
        assert "developmental" not in piece.spaces


class TestComputeFinalContent:
    def setup_method(self):
        store = InMemoryPieceStore()
        self.updater = KnowledgeUpdater(piece_store=store)

    def test_replace_returns_new_content(self):
        result = self.updater._compute_final_content(
            "old", "new", UpdateAction.REPLACE, {}
        )
        assert result == "new"

    def test_merge_append(self):
        result = self.updater._compute_final_content(
            "A", "B", UpdateAction.MERGE, {"merge_strategy": "append"}
        )
        assert result == "A\n\nB"

    def test_merge_prepend(self):
        result = self.updater._compute_final_content(
            "A", "B", UpdateAction.MERGE, {"merge_strategy": "prepend"}
        )
        assert result == "B\n\nA"

    def test_merge_interleave(self):
        result = self.updater._compute_final_content(
            "P1\n\nP2", "N1\n\nN2", UpdateAction.MERGE,
            {"merge_strategy": "interleave"},
        )
        assert "P1" in result
        assert "N1" in result
        assert "P2" in result
        assert "N2" in result

    def test_merge_default_is_append(self):
        result = self.updater._compute_final_content(
            "A", "B", UpdateAction.MERGE, {}
        )
        assert result == "A\n\nB"

    def test_no_change_returns_existing(self):
        result = self.updater._compute_final_content(
            "existing", "new", UpdateAction.NO_CHANGE, {}
        )
        assert result == "existing"


# ── Instruction-mode LLM mock factories ─────────────────────────────


def _make_instruction_mode_llm_fn(
    generated: str = "LLM-generated updated content",
    action: str = "replace",
):
    """LLM that returns input_mode=instruction on intent call,
    then returns generated_content on content-generation call.

    Distinguishes calls by checking for the "Determine Input Mode" marker
    in the intent prompt vs "Apply the user's instruction" in the generation prompt.
    """
    def llm_fn(prompt: str) -> str:
        if "Determine Input Mode" in prompt:
            # Intent analysis call
            return json.dumps({
                "input_mode": "instruction",
                "action": action,
                "confidence": 0.95,
                "reasoning": "User gave an instruction, not content",
                "changes_summary": "Applied instruction",
            })
        elif "Apply the user's instruction" in prompt:
            # Content generation call
            return json.dumps({
                "generated_content": generated,
            })
        else:
            # Fallback — shouldn't happen
            return json.dumps({"action": "no_change"})
    return llm_fn


def _make_content_mode_llm_fn(action: str = "replace"):
    """LLM that always returns input_mode=content (existing behavior)."""
    def llm_fn(prompt: str) -> str:
        return json.dumps({
            "input_mode": "content",
            "action": action,
            "confidence": 0.9,
            "reasoning": "User provided actual content",
            "changes_summary": "Replaced content",
        })
    return llm_fn


def _make_failing_generation_llm_fn():
    """LLM that returns instruction mode intent, but raises on generation call."""
    def llm_fn(prompt: str) -> str:
        if "Determine Input Mode" in prompt:
            return json.dumps({
                "input_mode": "instruction",
                "action": "replace",
                "confidence": 0.9,
                "reasoning": "Instruction detected",
                "changes_summary": "Would apply instruction",
            })
        elif "Apply the user's instruction" in prompt:
            raise RuntimeError("LLM generation failed")
        else:
            return json.dumps({"action": "no_change"})
    return llm_fn


# ── Instruction-mode Tests ──────────────────────────────────────────


class TestInstructionModeUpdate:
    """Tests for instruction-mode updates where LLM generates content."""

    def test_instruction_replace_generates_content(self):
        """When LLM classifies as instruction mode, generated content replaces piece."""
        piece = _make_piece("Grocery store shopping procedure: 1. Get cart 2. Shop 3. Checkout")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store,
            llm_fn=_make_instruction_mode_llm_fn(
                generated="Grocery store price-checking procedure: 1. Get scanner 2. Scan items 3. Compare prices",
                action="replace",
            ),
        )

        result = updater.update_by_id(
            piece.piece_id,
            'change "grocery store shopping procedure" to "grocery store price-checking procedure"',
            update_instruction='change "grocery store shopping procedure" to "grocery store price-checking procedure"',
        )

        assert result.success is True
        new = store.get_by_id(result.piece_id)
        assert "price-checking" in new.content
        # The instruction text should NOT be stored as content
        assert "change" not in new.content.lower().split()[0:2]

    def test_instruction_merge_generates_content(self):
        """Instruction mode with merge action: LLM-generated content used directly,
        not double-merged."""
        piece = _make_piece("Section A: Original info")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store,
            llm_fn=_make_instruction_mode_llm_fn(
                generated="Section A: Original info\n\nSection B: Added by instruction",
                action="merge",
            ),
        )

        result = updater.update_by_id(
            piece.piece_id,
            "add a section B with some extra info",
            update_instruction="add a section B with some extra info",
        )

        assert result.success is True
        new = store.get_by_id(result.piece_id)
        assert "Section A: Original info" in new.content
        assert "Section B: Added by instruction" in new.content
        # Should NOT have double content (existing appended to generated)
        assert new.content.count("Section A: Original info") == 1

    def test_content_mode_uses_programmatic_path(self):
        """When LLM returns input_mode=content, existing programmatic behavior unchanged."""
        piece = _make_piece("Old content")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store,
            llm_fn=_make_content_mode_llm_fn(action="replace"),
        )

        result = updater.update_by_id(piece.piece_id, "Actual new content")

        assert result.success is True
        new = store.get_by_id(result.piece_id)
        # Content mode replace: new_content is used directly
        assert new.content == "Actual new content"

    def test_generation_failure_returns_error(self):
        """When content generation fails, OperationResult(success=False) returned."""
        piece = _make_piece("Original content")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store,
            llm_fn=_make_failing_generation_llm_fn(),
        )

        result = updater.update_by_id(
            piece.piece_id,
            "rename X to Y",
            update_instruction="rename X to Y",
        )

        assert result.success is False
        assert "generation failed" in result.error.lower()
        # Original content should be preserved
        original = store.get_by_id(piece.piece_id)
        assert original.content == "Original content"
        assert original.is_active is True

    def test_no_update_instruction_defaults_to_content_mode(self):
        """update_by_content without update_instruction never enters instruction mode."""
        piece = _make_piece("Some content")
        store = InMemoryPieceStore(pieces=[piece], search_score=0.90)

        # Even if LLM returns instruction mode, without update_instruction
        # the guard prevents entering instruction path
        def llm_fn(prompt: str) -> str:
            return json.dumps({
                "input_mode": "instruction",
                "action": "replace",
                "confidence": 0.9,
                "reasoning": "Test",
                "changes_summary": "Test",
            })

        updater = KnowledgeUpdater(piece_store=store, llm_fn=llm_fn)

        results = updater.update_by_content("New replacement content")

        # Should use programmatic path (REPLACE with new_content)
        assert len(results) == 1
        assert results[0].success is True
        new = store.get_by_id(results[0].piece_id)
        assert new.content == "New replacement content"

    def test_update_by_id_instruction_mode(self):
        """update_by_id also supports instruction mode."""
        piece = _make_piece("Step 1: Do A\nStep 2: Do B\nStep 3: Do C")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store,
            llm_fn=_make_instruction_mode_llm_fn(
                generated="Step 1: Do X\nStep 2: Do B\nStep 3: Do C",
            ),
        )

        result = updater.update_by_id(
            piece.piece_id,
            "change step 1 from A to X",
            update_instruction="change step 1 from A to X",
        )

        assert result.success is True
        new = store.get_by_id(result.piece_id)
        assert "Do X" in new.content
        assert "Do B" in new.content

    def test_update_by_id_generation_failure(self):
        """update_by_id returns failure result on generation failure."""
        piece = _make_piece("Content here")
        store = InMemoryPieceStore(pieces=[piece])
        updater = KnowledgeUpdater(
            piece_store=store,
            llm_fn=_make_failing_generation_llm_fn(),
        )

        result = updater.update_by_id(
            piece.piece_id,
            "rewrite everything",
            update_instruction="rewrite everything",
        )

        assert result.success is False
        assert "generation failed" in result.error.lower()

    def test_update_by_content_instruction_mode(self):
        """update_by_content supports instruction mode with update_instruction."""
        piece = _make_piece("The cat sat on the mat")
        store = InMemoryPieceStore(pieces=[piece], search_score=0.90)
        updater = KnowledgeUpdater(
            piece_store=store,
            llm_fn=_make_instruction_mode_llm_fn(
                generated="The dog sat on the mat",
            ),
        )

        results = updater.update_by_content(
            'change "cat" to "dog"',
            update_instruction='change "cat" to "dog"',
        )

        assert len(results) == 1
        assert results[0].success is True
        new = store.get_by_id(results[0].piece_id)
        assert "dog" in new.content
        assert new.content == "The dog sat on the mat"

    def test_update_by_content_generation_failure_appends_error(self):
        """update_by_content appends failure result (doesn't silently skip)."""
        piece = _make_piece("Original content")
        store = InMemoryPieceStore(pieces=[piece], search_score=0.90)
        updater = KnowledgeUpdater(
            piece_store=store,
            llm_fn=_make_failing_generation_llm_fn(),
        )

        results = updater.update_by_content(
            "rewrite this",
            update_instruction="rewrite this",
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "generation failed" in results[0].error.lower()


# ── Prompt Template Tests ───────────────────────────────────────────


class TestStripJsonFences:
    """Tests for _strip_json_fences helper."""

    def setup_method(self):
        store = InMemoryPieceStore()
        self.updater = KnowledgeUpdater(piece_store=store)

    def test_plain_json_unchanged(self):
        raw = '{"action": "replace"}'
        assert self.updater._strip_json_fences(raw) == raw

    def test_strips_json_fence(self):
        raw = '```json\n{"action": "replace"}\n```'
        assert self.updater._strip_json_fences(raw) == '{"action": "replace"}'

    def test_strips_plain_fence(self):
        raw = '```\n{"key": "val"}\n```'
        assert self.updater._strip_json_fences(raw) == '{"key": "val"}'

    def test_handles_whitespace(self):
        raw = '  ```json\n{"a": 1}\n```  '
        assert self.updater._strip_json_fences(raw) == '{"a": 1}'

    def test_fenced_llm_response_parses(self):
        """End-to-end: LLM returning fenced JSON still works."""
        piece = _make_piece("Old content")
        store = InMemoryPieceStore(pieces=[piece])

        def fenced_llm(prompt: str) -> str:
            if "Determine Input Mode" in prompt:
                return '```json\n' + json.dumps({
                    "input_mode": "instruction",
                    "action": "replace",
                    "confidence": 0.9,
                    "reasoning": "Test",
                    "changes_summary": "Test",
                }) + '\n```'
            elif "Apply the user's instruction" in prompt:
                return '```json\n' + json.dumps({
                    "generated_content": "Fenced result content",
                }) + '\n```'
            return json.dumps({"action": "no_change"})

        updater = KnowledgeUpdater(piece_store=store, llm_fn=fenced_llm)
        result = updater.update_by_id(
            piece.piece_id, "rename X", update_instruction="rename X"
        )
        assert result.success is True
        new = store.get_by_id(result.piece_id)
        assert new.content == "Fenced result content"


class TestPromptTemplates:
    def test_content_generation_prompt_format(self):
        """render_prompt("quality/UpdateContentGeneration") works with expected placeholders."""
        result = render_prompt(
            "quality/UpdateContentGeneration",
            existing_content="Some existing content",
            update_instruction="Change X to Y",
        )
        assert "Some existing content" in result
        assert "Change X to Y" in result
