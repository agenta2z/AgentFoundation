"""
Property-based tests for enhanced KnowledgePiece model.

Feature: knowledge-module-migration
- Property 1: KnowledgePiece serialization round trip
- Property 2: KnowledgePiece backward-compatible deserialization
- Property 3: Content hash determinism

Feature: knowledge-space-restructuring
- Property 1: Space/Spaces Synchronization Invariant
- Property 2: KnowledgePiece Dict Serialization Round-Trip

**Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.10, 1.11**
"""
import hashlib
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

from hypothesis import given, settings, strategies as st

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)

# Import strategies from conftest
_test_dir = Path(__file__).resolve().parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
from conftest import knowledge_piece_strategy


# ── Shared helper strategies ─────────────────────────────────────────────────

_non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

_identifier_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
)

_timestamp_strategy = st.from_regex(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\+00:00",
    fullmatch=True,
)

_tag_strategy = st.text(min_size=1, max_size=30)


# Feature: knowledge-module-migration, Property 1: KnowledgePiece serialization round trip


class TestKnowledgePieceSerializationRoundTrip:
    """Property 1: KnowledgePiece serialization round trip.

    For any valid KnowledgePiece (with all fields populated, including new
    fields like domain, content_hash, space, merge_strategy, versioning, etc.),
    calling to_dict() then from_dict() on the result should produce a
    KnowledgePiece with equivalent field values.

    **Validates: Requirements 1.10**
    """

    @given(piece=knowledge_piece_strategy(include_new_fields=True))
    @settings(max_examples=100)
    def test_to_dict_from_dict_round_trip_all_fields(self, piece: KnowledgePiece):
        """to_dict() followed by from_dict() preserves all fields including new ones.

        **Validates: Requirements 1.10**
        """
        serialized = piece.to_dict()
        restored = KnowledgePiece.from_dict(serialized)

        # Original fields
        assert restored.content == piece.content
        assert restored.piece_id == piece.piece_id
        assert restored.knowledge_type == piece.knowledge_type
        assert restored.info_type == piece.info_type
        assert restored.tags == piece.tags
        assert restored.entity_id == piece.entity_id
        assert restored.source == piece.source
        assert restored.embedding_text == piece.embedding_text
        assert restored.created_at == piece.created_at
        assert restored.updated_at == piece.updated_at

        # Retrieval classification fields
        assert restored.domain == piece.domain
        assert restored.secondary_domains == piece.secondary_domains
        assert restored.custom_tags == piece.custom_tags

        # Deduplication fields
        assert restored.content_hash == piece.content_hash
        assert restored.embedding == piece.embedding

        # Space
        assert restored.space == piece.space

        # Merge strategy fields
        assert restored.merge_strategy == piece.merge_strategy
        assert restored.merge_processed == piece.merge_processed
        assert restored.pending_merge_suggestion == piece.pending_merge_suggestion
        assert restored.merge_suggestion_reason == piece.merge_suggestion_reason
        assert restored.suggestion_status == piece.suggestion_status

        # Validation fields
        assert restored.validation_status == piece.validation_status
        assert restored.validation_issues == piece.validation_issues

        # Versioning fields
        assert restored.supersedes == piece.supersedes
        assert restored.is_active == piece.is_active
        assert restored.version == piece.version

        # Progressive disclosure
        assert restored.summary == piece.summary


# Feature: knowledge-module-migration, Property 2: KnowledgePiece backward-compatible deserialization


class TestKnowledgePieceBackwardCompatibleDeserialization:
    """Property 2: KnowledgePiece backward-compatible deserialization.

    For any dictionary containing only the original KnowledgePiece fields
    (content, piece_id, knowledge_type, info_type, tags, entity_id, source,
    embedding_text, created_at, updated_at), calling from_dict() should produce
    a KnowledgePiece where all new fields have their documented default values.

    **Validates: Requirements 1.11**
    """

    @given(
        content=_non_empty_text,
        piece_id=_identifier_text,
        knowledge_type=st.sampled_from(list(KnowledgeType)),
        info_type=st.sampled_from(["user_profile", "instructions", "context"]),
        tags=st.lists(_tag_strategy, max_size=5),
        entity_id=st.one_of(st.none(), _identifier_text),
        source=st.one_of(st.none(), st.text(max_size=50)),
        embedding_text=st.one_of(st.none(), st.text(max_size=100)),
        created_at=_timestamp_strategy,
        updated_at=_timestamp_strategy,
    )
    @settings(max_examples=100)
    def test_from_dict_with_only_original_fields_uses_defaults(
        self,
        content,
        piece_id,
        knowledge_type,
        info_type,
        tags,
        entity_id,
        source,
        embedding_text,
        created_at,
        updated_at,
    ):
        """from_dict() with only original fields sets documented defaults for new fields.

        **Validates: Requirements 1.11**
        """
        original_dict = {
            "content": content,
            "piece_id": piece_id,
            "knowledge_type": knowledge_type.value,
            "info_type": info_type,
            "tags": tags,
            "entity_id": entity_id,
            "source": source,
            "embedding_text": embedding_text,
            "created_at": created_at,
            "updated_at": updated_at,
        }

        piece = KnowledgePiece.from_dict(original_dict)

        # Original fields should be preserved
        assert piece.content == content
        assert piece.piece_id == piece_id
        assert piece.knowledge_type == knowledge_type
        assert piece.info_type == info_type
        assert piece.entity_id == entity_id
        assert piece.source == source
        assert piece.embedding_text == embedding_text
        assert piece.created_at == created_at
        assert piece.updated_at == updated_at

        # New fields should have documented defaults
        assert piece.domain == "general"
        assert piece.secondary_domains == []
        assert piece.custom_tags == []
        assert piece.embedding is None
        assert piece.space == "main"
        assert piece.merge_strategy is None
        assert piece.merge_processed is False
        assert piece.pending_merge_suggestion is None
        assert piece.merge_suggestion_reason is None
        assert piece.suggestion_status is None
        assert piece.validation_status == "not_validated"
        assert piece.validation_issues == []
        assert piece.supersedes is None
        assert piece.is_active is True
        assert piece.version == 1
        assert piece.summary is None

        # content_hash should be auto-computed (not None)
        assert piece.content_hash is not None


# Feature: knowledge-module-migration, Property 3: Content hash determinism


class TestContentHashDeterminism:
    """Property 3: Content hash determinism.

    For any content string, the auto-computed content_hash should equal the
    first 16 characters of the SHA256 hex digest of the whitespace-normalized
    content. Additionally, two KnowledgePieces with content that differs only
    in whitespace should produce the same content_hash.

    **Validates: Requirements 1.2**
    """

    @given(content=_non_empty_text)
    @settings(max_examples=100)
    def test_content_hash_matches_sha256_of_normalized_content(self, content):
        """content_hash equals first 16 chars of SHA256 of whitespace-normalized content.

        **Validates: Requirements 1.2**
        """
        piece = KnowledgePiece(content=content)

        normalized = " ".join(content.split())
        expected_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

        assert piece.content_hash == expected_hash

    @given(
        words=st.lists(st.text(min_size=1, max_size=20).filter(lambda s: s.strip()), min_size=1, max_size=10),
        extra_spaces=st.lists(
            st.text(
                alphabet=st.sampled_from([" ", "\t", "\n", "\r"]),
                min_size=1,
                max_size=5,
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_whitespace_only_difference_produces_same_hash(self, words, extra_spaces):
        """Two pieces with content differing only in whitespace produce the same hash.

        **Validates: Requirements 1.2**
        """
        # Build content with single spaces
        content_normal = " ".join(words)
        if not content_normal.strip():
            return  # skip if all words are whitespace-only

        # Build content with extra whitespace between words
        padded_words = []
        for i, word in enumerate(words):
            padded_words.append(word)
            if i < len(words) - 1:
                ws = extra_spaces[i % len(extra_spaces)]
                padded_words.append(ws)
        content_padded = "".join(padded_words)
        if not content_padded.strip():
            return  # skip if result is whitespace-only

        piece_normal = KnowledgePiece(content=content_normal)
        piece_padded = KnowledgePiece(content=content_padded)

        assert piece_normal.content_hash == piece_padded.content_hash


# ── Shared strategies for space testing ──────────────────────────────────────

# Valid space names (the three known spaces)
_valid_space = st.sampled_from(["main", "personal", "developmental"])

# Space strings that may need normalization (whitespace, mixed case, etc.)
_messy_space = st.one_of(
    _valid_space,
    _valid_space.map(lambda s: s.upper()),
    _valid_space.map(lambda s: s.capitalize()),
    _valid_space.map(lambda s: f"  {s}  "),
    _valid_space.map(lambda s: f"\t{s}\n"),
)

# Non-empty list of space strings (may contain duplicates, whitespace, mixed case)
_spaces_list = st.lists(_messy_space, min_size=1, max_size=6)

# Possibly empty list of space strings (to test empty-list fallback)
_spaces_list_possibly_empty = st.lists(_messy_space, min_size=0, max_size=6)


# Feature: knowledge-space-restructuring, Property 1: Space/Spaces Synchronization Invariant


class TestSpacesSynchronizationInvariant:
    """Property 1: Space/Spaces Synchronization Invariant.

    For any KnowledgePiece, regardless of how it was constructed (with space
    only, spaces only, or both), the invariant piece.space == piece.spaces[0]
    SHALL always hold, and piece.spaces SHALL be a non-empty list of valid
    space strings.

    **Validates: Requirements 1.2, 1.3, 1.4**
    """

    @given(space=_messy_space)
    @settings(max_examples=100)
    def test_space_only_construction_syncs_spaces(self, space):
        """When constructed with space only (no spaces), spaces == [normalized(space)].

        **Validates: Requirements 1.2, 1.3**
        """
        piece = KnowledgePiece(content="test content", space=space)

        # spaces should be derived from space
        assert len(piece.spaces) >= 1
        assert piece.space == piece.spaces[0]
        # The space value should be normalized (stripped, lowercased)
        expected = space.strip().lower()
        assert piece.spaces == [expected]
        assert piece.space == expected

    @given(spaces=_spaces_list)
    @settings(max_examples=100)
    def test_spaces_only_construction_syncs_space(self, spaces):
        """When constructed with spaces only, space == spaces[0] after normalization.

        **Validates: Requirements 1.2, 1.4**
        """
        piece = KnowledgePiece(content="test content", spaces=spaces)

        # Invariant: space == spaces[0]
        assert piece.space == piece.spaces[0]
        # spaces must be non-empty
        assert len(piece.spaces) >= 1
        # All spaces should be normalized
        for s in piece.spaces:
            assert s == s.strip().lower()
            assert len(s) > 0

    @given(space=_messy_space, spaces=_spaces_list)
    @settings(max_examples=100)
    def test_both_space_and_spaces_construction(self, space, spaces):
        """When constructed with both space and spaces, spaces wins and space is synced.

        **Validates: Requirements 1.2, 1.4**
        """
        piece = KnowledgePiece(content="test content", space=space, spaces=spaces)

        # Invariant: space == spaces[0]
        assert piece.space == piece.spaces[0]
        # spaces must be non-empty
        assert len(piece.spaces) >= 1
        # All spaces should be normalized
        for s in piece.spaces:
            assert s == s.strip().lower()
            assert len(s) > 0

    @settings(max_examples=100)
    @given(data=st.data())
    def test_default_construction_has_main(self, data):
        """When constructed with neither space nor spaces, defaults to ["main"].

        **Validates: Requirements 1.2, 1.3**
        """
        piece = KnowledgePiece(content="test content")

        assert piece.space == "main"
        assert piece.spaces == ["main"]
        assert piece.space == piece.spaces[0]

    @given(spaces=_spaces_list_possibly_empty)
    @settings(max_examples=100)
    def test_empty_spaces_defaults_to_main(self, spaces):
        """When spaces is empty or normalizes to empty, defaults to ["main"].

        **Validates: Requirements 1.2, 1.4**
        """
        piece = KnowledgePiece(content="test content", spaces=spaces)

        # spaces must always be non-empty after normalization
        assert len(piece.spaces) >= 1
        assert piece.space == piece.spaces[0]
        # If input was empty, should default to main
        if not spaces or not any(s.strip() for s in spaces):
            assert piece.spaces == ["main"]
            assert piece.space == "main"

    @given(spaces=_spaces_list)
    @settings(max_examples=100)
    def test_spaces_are_deduplicated_preserving_order(self, spaces):
        """Duplicate spaces are removed while preserving first-occurrence order.

        **Validates: Requirements 1.2, 1.4**
        """
        piece = KnowledgePiece(content="test content", spaces=spaces)

        # No duplicates in the result
        assert len(piece.spaces) == len(set(piece.spaces))
        # Invariant still holds
        assert piece.space == piece.spaces[0]

    @given(spaces=_spaces_list)
    @settings(max_examples=100)
    def test_spaces_are_normalized_lowercase_stripped(self, spaces):
        """All space strings are stripped and lowercased.

        **Validates: Requirements 1.2, 1.4**
        """
        piece = KnowledgePiece(content="test content", spaces=spaces)

        for s in piece.spaces:
            assert s == s.strip()
            assert s == s.lower()
            assert len(s) > 0  # no empty strings

        # Invariant still holds
        assert piece.space == piece.spaces[0]

    @given(piece=knowledge_piece_strategy(include_new_fields=True))
    @settings(max_examples=100)
    def test_invariant_holds_for_random_full_pieces(self, piece):
        """The sync invariant holds for fully random KnowledgePiece instances.

        **Validates: Requirements 1.2, 1.3, 1.4**
        """
        assert piece.space == piece.spaces[0]
        assert len(piece.spaces) >= 1
        for s in piece.spaces:
            assert s == s.strip().lower()
            assert len(s) > 0


# Feature: knowledge-space-restructuring, Property 2: KnowledgePiece Dict Serialization Round-Trip


class TestKnowledgePieceDictSerializationRoundTrip:
    """Property 2: KnowledgePiece Dict Serialization Round-Trip.

    For any valid KnowledgePiece with arbitrary spaces values,
    KnowledgePiece.from_dict(piece.to_dict()).spaces SHALL equal piece.spaces,
    and KnowledgePiece.from_dict(piece.to_dict()).space SHALL equal piece.space.
    Additionally, for any dict with a space field but no spaces field,
    from_dict SHALL produce spaces == [space].

    **Validates: Requirements 1.5, 1.6, 1.7**
    """

    @given(piece=knowledge_piece_strategy(include_new_fields=True))
    @settings(max_examples=100)
    def test_round_trip_preserves_spaces_and_space(self, piece):
        """to_dict() → from_dict() preserves both spaces and space fields.

        **Validates: Requirements 1.5, 1.7**
        """
        serialized = piece.to_dict()
        restored = KnowledgePiece.from_dict(serialized)

        assert restored.spaces == piece.spaces
        assert restored.space == piece.space
        # Invariant still holds on the restored piece
        assert restored.space == restored.spaces[0]

    @given(space=_messy_space)
    @settings(max_examples=100)
    def test_missing_spaces_key_derives_from_space(self, space):
        """When dict has space but no spaces key, from_dict produces spaces == [normalized(space)].

        **Validates: Requirements 1.6**
        """
        data = {
            "content": "test content",
            "space": space,
        }
        # Ensure 'spaces' key is absent
        assert "spaces" not in data

        piece = KnowledgePiece.from_dict(data)

        expected_space = space.strip().lower()
        assert piece.spaces == [expected_space]
        assert piece.space == expected_space
        assert piece.space == piece.spaces[0]
