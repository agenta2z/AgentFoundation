"""
Unit tests for KnowledgeType enum and KnowledgePiece data model.

Tests cover:
- KnowledgeType enum values and string conversion
- KnowledgePiece creation with defaults and explicit values
- Auto-generated piece_id via UUID
- Tag normalization (lowercase, stripped, empty removal)
- Content validation in from_dict
- Serialization round-trip (to_dict / from_dict)
- embedding_text optional field

Requirements: 1.1, 1.2, 1.4, 1.5, 1.6
"""
import sys
import uuid
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pytest

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgeType,
    KnowledgePiece,
)


class TestKnowledgeType:
    """Tests for the KnowledgeType enum."""

    def test_enum_values(self):
        """All expected enum members exist with correct string values."""
        assert KnowledgeType.Fact == "fact"
        assert KnowledgeType.Instruction == "instruction"
        assert KnowledgeType.Preference == "preference"
        assert KnowledgeType.Procedure == "procedure"
        assert KnowledgeType.Note == "note"
        assert KnowledgeType.Episodic == "episodic"

    def test_enum_from_value(self):
        """KnowledgeType can be constructed from string values."""
        assert KnowledgeType("fact") == KnowledgeType.Fact
        assert KnowledgeType("instruction") == KnowledgeType.Instruction

    def test_enum_is_str(self):
        """KnowledgeType members are strings (StrEnum)."""
        assert isinstance(KnowledgeType.Fact, str)
        assert str(KnowledgeType.Fact) == "fact"


class TestKnowledgePieceCreation:
    """Tests for KnowledgePiece construction and defaults."""

    def test_minimal_creation(self):
        """Creating a KnowledgePiece with only content sets all defaults."""
        piece = KnowledgePiece(content="Test knowledge")
        assert piece.content == "Test knowledge"
        assert piece.piece_id is not None
        assert piece.knowledge_type == KnowledgeType.Fact
        assert piece.tags == []
        assert piece.entity_id is None
        assert piece.source is None
        assert piece.embedding_text is None
        assert piece.created_at is not None
        assert piece.updated_at is not None

    def test_auto_generated_piece_id_is_valid_uuid(self):
        """Auto-generated piece_id is a valid UUID string. (Req 1.2)"""
        piece = KnowledgePiece(content="Test")
        # Should not raise
        uuid.UUID(piece.piece_id)

    def test_explicit_piece_id_preserved(self):
        """Providing an explicit piece_id preserves it."""
        piece = KnowledgePiece(content="Test", piece_id="custom-id-123")
        assert piece.piece_id == "custom-id-123"

    def test_explicit_timestamps_preserved(self):
        """Providing explicit timestamps preserves them."""
        piece = KnowledgePiece(
            content="Test",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-06-15T12:00:00+00:00",
        )
        assert piece.created_at == "2024-01-01T00:00:00+00:00"
        assert piece.updated_at == "2024-06-15T12:00:00+00:00"

    def test_all_fields_explicit(self):
        """Creating with all fields explicitly set."""
        piece = KnowledgePiece(
            content="Prefers organic eggs",
            piece_id="abc-123",
            knowledge_type=KnowledgeType.Preference,
            tags=["food", "grocery"],
            entity_id="user:xinli",
            source="conversation",
            embedding_text="User xinli prefers organic eggs",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        assert piece.content == "Prefers organic eggs"
        assert piece.piece_id == "abc-123"
        assert piece.knowledge_type == KnowledgeType.Preference
        assert piece.tags == ["food", "grocery"]
        assert piece.entity_id == "user:xinli"
        assert piece.source == "conversation"
        assert piece.embedding_text == "User xinli prefers organic eggs"


class TestTagNormalization:
    """Tests for tag normalization behavior. (Req 1.5)"""

    def test_tags_lowercased(self):
        """Tags are converted to lowercase."""
        piece = KnowledgePiece(content="Test", tags=["Food", "GROCERY", "Shopping"])
        assert piece.tags == ["food", "grocery", "shopping"]

    def test_tags_stripped(self):
        """Tags are stripped of leading/trailing whitespace."""
        piece = KnowledgePiece(content="Test", tags=["  food  ", " grocery", "shopping "])
        assert piece.tags == ["food", "grocery", "shopping"]

    def test_empty_tags_removed(self):
        """Empty and whitespace-only tags are removed."""
        piece = KnowledgePiece(content="Test", tags=["food", "", "  ", "grocery"])
        assert piece.tags == ["food", "grocery"]

    def test_combined_normalization(self):
        """Tags are stripped, lowercased, and empty ones removed together."""
        piece = KnowledgePiece(content="Test", tags=["  FOOD  ", "", " Grocery ", "  "])
        assert piece.tags == ["food", "grocery"]


class TestToDict:
    """Tests for to_dict serialization."""

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all fields."""
        piece = KnowledgePiece(
            content="Test content",
            piece_id="test-id",
            knowledge_type=KnowledgeType.Instruction,
            tags=["tag1"],
            entity_id="user:test",
            source="manual",
            embedding_text="override text",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        d = piece.to_dict()
        assert d["content"] == "Test content"
        assert d["piece_id"] == "test-id"
        assert d["knowledge_type"] == "instruction"
        assert d["tags"] == ["tag1"]
        assert d["entity_id"] == "user:test"
        assert d["source"] == "manual"
        assert d["embedding_text"] == "override text"
        assert d["created_at"] == "2024-01-01T00:00:00+00:00"
        assert d["updated_at"] == "2024-01-01T00:00:00+00:00"

    def test_to_dict_knowledge_type_is_string(self):
        """knowledge_type is serialized as its string value, not the enum."""
        piece = KnowledgePiece(content="Test", knowledge_type=KnowledgeType.Procedure)
        d = piece.to_dict()
        assert d["knowledge_type"] == "procedure"
        assert isinstance(d["knowledge_type"], str)

    def test_to_dict_none_fields(self):
        """Optional fields that are None are included as None."""
        piece = KnowledgePiece(content="Test")
        d = piece.to_dict()
        assert d["entity_id"] is None
        assert d["source"] is None
        assert d["embedding_text"] is None


class TestFromDict:
    """Tests for from_dict deserialization and validation."""

    def test_from_dict_basic(self):
        """from_dict creates a valid KnowledgePiece from a dictionary."""
        data = {
            "content": "Test content",
            "piece_id": "test-id",
            "knowledge_type": "instruction",
            "tags": ["tag1", "tag2"],
            "entity_id": "user:test",
            "source": "manual",
            "embedding_text": "override",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
        piece = KnowledgePiece.from_dict(data)
        assert piece.content == "Test content"
        assert piece.piece_id == "test-id"
        assert piece.knowledge_type == KnowledgeType.Instruction
        assert piece.tags == ["tag1", "tag2"]
        assert piece.entity_id == "user:test"
        assert piece.source == "manual"
        assert piece.embedding_text == "override"

    def test_from_dict_minimal(self):
        """from_dict with only content auto-generates other fields."""
        piece = KnowledgePiece.from_dict({"content": "Just content"})
        assert piece.content == "Just content"
        assert piece.piece_id is not None
        assert piece.knowledge_type == KnowledgeType.Fact
        assert piece.tags == []

    def test_from_dict_missing_content_raises(self):
        """from_dict raises ValueError when content key is missing. (Req 1.4)"""
        with pytest.raises(ValueError, match="Missing required field.*content"):
            KnowledgePiece.from_dict({"tags": ["test"]})

    def test_from_dict_empty_content_raises(self):
        """from_dict raises ValueError when content is empty string. (Req 1.4)"""
        with pytest.raises(ValueError, match="non-empty string"):
            KnowledgePiece.from_dict({"content": ""})

    def test_from_dict_whitespace_content_raises(self):
        """from_dict raises ValueError when content is whitespace only. (Req 1.4)"""
        with pytest.raises(ValueError, match="non-empty string"):
            KnowledgePiece.from_dict({"content": "   "})

    def test_from_dict_non_string_content_raises(self):
        """from_dict raises ValueError when content is not a string."""
        with pytest.raises(ValueError, match="non-empty string"):
            KnowledgePiece.from_dict({"content": 123})

    def test_from_dict_normalizes_tags(self):
        """from_dict normalizes tags through __attrs_post_init__."""
        piece = KnowledgePiece.from_dict({"content": "Test", "tags": ["  FOOD  ", ""]})
        assert piece.tags == ["food"]


class TestRoundTrip:
    """Tests for serialization round-trip. (Req 1.6)"""

    def test_round_trip_preserves_data(self):
        """to_dict followed by from_dict preserves all data."""
        original = KnowledgePiece(
            content="Costco membership benefits",
            piece_id="piece-001",
            knowledge_type=KnowledgeType.Fact,
            tags=["costco", "membership"],
            entity_id="user:xinli",
            source="conversation",
            embedding_text="User xinli Costco membership benefits",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-06-15T12:00:00+00:00",
        )
        d = original.to_dict()
        restored = KnowledgePiece.from_dict(d)

        assert restored.content == original.content
        assert restored.piece_id == original.piece_id
        assert restored.knowledge_type == original.knowledge_type
        assert restored.tags == original.tags
        assert restored.entity_id == original.entity_id
        assert restored.source == original.source
        assert restored.embedding_text == original.embedding_text
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at

    def test_round_trip_with_none_optionals(self):
        """Round-trip works when optional fields are None."""
        original = KnowledgePiece(content="Simple fact")
        d = original.to_dict()
        restored = KnowledgePiece.from_dict(d)

        assert restored.content == original.content
        assert restored.piece_id == original.piece_id
        assert restored.knowledge_type == original.knowledge_type
        assert restored.tags == original.tags
        assert restored.entity_id == original.entity_id
        assert restored.source == original.source
        assert restored.embedding_text == original.embedding_text


class TestEmbeddingText:
    """Tests for the embedding_text optional field."""

    def test_embedding_text_default_none(self):
        """embedding_text defaults to None."""
        piece = KnowledgePiece(content="Test")
        assert piece.embedding_text is None

    def test_embedding_text_set_explicitly(self):
        """embedding_text can be set to a custom string."""
        piece = KnowledgePiece(
            content="Prefers organic eggs",
            embedding_text="User xinli prefers organic eggs for grocery shopping",
        )
        assert piece.embedding_text == "User xinli prefers organic eggs for grocery shopping"

    def test_embedding_text_in_to_dict(self):
        """embedding_text is included in to_dict output."""
        piece = KnowledgePiece(content="Test", embedding_text="enriched text")
        d = piece.to_dict()
        assert d["embedding_text"] == "enriched text"

    def test_embedding_text_round_trip(self):
        """embedding_text survives round-trip serialization."""
        piece = KnowledgePiece(content="Test", embedding_text="enriched text")
        restored = KnowledgePiece.from_dict(piece.to_dict())
        assert restored.embedding_text == "enriched text"
