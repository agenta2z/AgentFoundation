"""
Property-based tests for EntityMetadata serialization round-trip.

# Feature: agent-knowledge-base, Property 1: Data model serialization round-trip
# **Validates: Requirements 1.6**
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

from hypothesis import given, settings, strategies as st

from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata

# Import strategies from conftest (same directory, added to sys.path)
_test_dir = Path(__file__).resolve().parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
from conftest import entity_metadata_strategy


# Feature: agent-knowledge-base, Property 1: Data model serialization round-trip


class TestEntityMetadataSerializationRoundTrip:
    """Property 1: Data model serialization round-trip for EntityMetadata.

    For any valid EntityMetadata object, calling to_dict() then from_dict()
    on the result SHALL produce an object equivalent to the original
    (all fields match).

    **Validates: Requirements 1.6**
    """

    @given(metadata=entity_metadata_strategy())
    @settings(max_examples=200)
    def test_to_dict_from_dict_round_trip(self, metadata: EntityMetadata):
        """to_dict() followed by from_dict() produces an equivalent EntityMetadata.

        **Validates: Requirements 1.6**
        """
        serialized = metadata.to_dict()
        restored = EntityMetadata.from_dict(serialized)

        assert restored.entity_id == metadata.entity_id, (
            f"entity_id mismatch: {metadata.entity_id!r} -> {restored.entity_id!r}"
        )
        assert restored.entity_type == metadata.entity_type, (
            f"entity_type mismatch: {metadata.entity_type!r} -> {restored.entity_type!r}"
        )
        assert restored.properties == metadata.properties, (
            f"properties mismatch: {metadata.properties!r} -> {restored.properties!r}"
        )
        assert restored.created_at == metadata.created_at, (
            f"created_at mismatch: {metadata.created_at!r} -> {restored.created_at!r}"
        )
        assert restored.updated_at == metadata.updated_at, (
            f"updated_at mismatch: {metadata.updated_at!r} -> {restored.updated_at!r}"
        )


# Feature: agent-knowledge-base, Property 20: Metadata key-value round-trip


class TestEntityMetadataKeyValueRoundTrip:
    """Property 20: Metadata key-value round-trip.

    For any EntityMetadata instance and any key-value pair where the value is
    JSON-serializable, calling set(key, value) then get(key) SHALL return the
    original value.

    **Validates: Requirements 1.6 (metadata layer)**
    """

    @given(
        metadata=entity_metadata_strategy(),
        key=st.text(min_size=1),
        value=st.recursive(
            st.one_of(
                st.none(),
                st.booleans(),
                st.integers(min_value=-1000, max_value=1000),
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                st.text(max_size=50),
            ),
            lambda children: st.one_of(
                st.lists(children, max_size=5),
                st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
            ),
            max_leaves=10,
        ),
    )
    @settings(max_examples=200)
    def test_set_then_get_returns_original_value(self, metadata: EntityMetadata, key: str, value):
        """set(key, value) followed by get(key) returns the original value.

        **Validates: Requirements 1.6**
        """
        metadata.set(key, value)
        retrieved = metadata.get(key)

        assert retrieved == value, (
            f"Round-trip failed for key={key!r}: set {value!r}, got {retrieved!r}"
        )
