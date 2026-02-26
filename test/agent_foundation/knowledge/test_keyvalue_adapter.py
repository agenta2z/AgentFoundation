"""
Unit tests for KeyValueMetadataStore adapter.

Tests that the adapter correctly implements the MetadataStore ABC by
delegating to a MemoryKeyValueService backend. Covers CRUD operations,
entity type filtering, round-trip serialization, and close delegation.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
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

# Also add SciencePythonUtils src to path
_spu_src = Path(__file__).resolve().parents[4] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest

from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from science_modeling_tools.knowledge.stores.metadata.base import MetadataStore
from science_modeling_tools.knowledge.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)


@pytest.fixture
def kv_service():
    """Create a fresh MemoryKeyValueService for each test."""
    return MemoryKeyValueService()


@pytest.fixture
def store(kv_service):
    """Create a KeyValueMetadataStore backed by MemoryKeyValueService."""
    return KeyValueMetadataStore(kv_service=kv_service)


class TestImplementsABC:
    """Requirement 11.1: KeyValueMetadataStore implements MetadataStore ABC."""

    def test_is_instance_of_metadata_store(self, store):
        """KeyValueMetadataStore should be an instance of MetadataStore."""
        assert isinstance(store, MetadataStore)

    def test_has_all_abstract_methods(self, store):
        """KeyValueMetadataStore should implement all MetadataStore methods."""
        assert hasattr(store, "get_metadata")
        assert hasattr(store, "save_metadata")
        assert hasattr(store, "delete_metadata")
        assert hasattr(store, "list_entities")
        assert hasattr(store, "close")


class TestSaveAndGet:
    """Requirements 11.2, 11.3: save then get returns equivalent metadata."""

    def test_save_then_get_returns_equivalent_metadata(self, store):
        """Saving metadata and getting it back should return equivalent data."""
        meta = EntityMetadata(
            entity_id="user:alice",
            entity_type="user",
            properties={"name": "Alice", "location": "Seattle"},
        )
        store.save_metadata(meta)
        result = store.get_metadata("user:alice")

        assert result is not None
        assert result.entity_id == meta.entity_id
        assert result.entity_type == meta.entity_type
        assert result.properties == meta.properties
        assert result.created_at == meta.created_at
        assert result.updated_at == meta.updated_at

    def test_save_overwrites_existing_metadata(self, store):
        """Saving metadata for an existing entity should overwrite it."""
        meta_v1 = EntityMetadata(
            entity_id="user:bob",
            entity_type="user",
            properties={"name": "Bob"},
        )
        store.save_metadata(meta_v1)

        meta_v2 = EntityMetadata(
            entity_id="user:bob",
            entity_type="user",
            properties={"name": "Robert", "age": 30},
        )
        store.save_metadata(meta_v2)

        result = store.get_metadata("user:bob")
        assert result is not None
        assert result.properties == {"name": "Robert", "age": 30}

    def test_get_nonexistent_returns_none(self, store):
        """Getting metadata for a non-existent entity should return None."""
        result = store.get_metadata("user:nobody")
        assert result is None

    def test_save_uses_entity_type_as_namespace(self, store, kv_service):
        """save_metadata should use entity_type as the KV namespace."""
        meta = EntityMetadata(
            entity_id="user:alice",
            entity_type="user",
            properties={"name": "Alice"},
        )
        store.save_metadata(meta)

        # Verify the data is stored under the "user" namespace
        data = kv_service.get("user:alice", namespace="user")
        assert data is not None
        assert data["entity_id"] == "user:alice"

    def test_save_uses_entity_id_as_key(self, store, kv_service):
        """save_metadata should use entity_id as the KV key."""
        meta = EntityMetadata(
            entity_id="store:costco",
            entity_type="store",
            properties={"location": "Seattle"},
        )
        store.save_metadata(meta)

        # Verify the key is the entity_id
        keys = kv_service.keys(namespace="store")
        assert "store:costco" in keys

    def test_round_trip_preserves_all_fields(self, store):
        """Round-trip should preserve all EntityMetadata fields."""
        meta = EntityMetadata(
            entity_id="tool:calculator",
            entity_type="tool",
            properties={"version": "1.0", "features": ["add", "subtract"]},
        )
        store.save_metadata(meta)
        result = store.get_metadata("tool:calculator")

        assert result.to_dict() == meta.to_dict()

    def test_entity_without_colon_uses_default_type(self, store):
        """An entity ID without ':' should use 'default' as entity_type for namespace."""
        meta = EntityMetadata(
            entity_id="plain_id",
            entity_type="default",
            properties={"key": "value"},
        )
        store.save_metadata(meta)
        result = store.get_metadata("plain_id")

        assert result is not None
        assert result.entity_id == "plain_id"
        assert result.properties == {"key": "value"}


class TestDelete:
    """Requirement 11.1: delete operations via adapter."""

    def test_delete_existing_returns_true(self, store):
        """Deleting an existing entity should return True."""
        meta = EntityMetadata(entity_id="user:charlie", entity_type="user")
        store.save_metadata(meta)
        assert store.delete_metadata("user:charlie") is True

    def test_delete_nonexistent_returns_false(self, store):
        """Deleting a non-existent entity should return False."""
        assert store.delete_metadata("user:nobody") is False

    def test_get_after_delete_returns_none(self, store):
        """Getting metadata after deletion should return None."""
        meta = EntityMetadata(entity_id="user:dave", entity_type="user")
        store.save_metadata(meta)
        store.delete_metadata("user:dave")
        assert store.get_metadata("user:dave") is None


class TestListEntities:
    """Requirements 11.4, 11.5: list entities with optional type filtering."""

    def test_list_entities_returns_all_ids(self, store):
        """list_entities() without filter should return all entity IDs."""
        store.save_metadata(
            EntityMetadata(entity_id="user:alice", entity_type="user")
        )
        store.save_metadata(
            EntityMetadata(entity_id="user:bob", entity_type="user")
        )
        store.save_metadata(
            EntityMetadata(entity_id="store:costco", entity_type="store")
        )

        result = sorted(store.list_entities())
        assert result == ["store:costco", "user:alice", "user:bob"]

    def test_list_entities_with_type_filter(self, store):
        """list_entities(entity_type=...) should return only matching entities."""
        store.save_metadata(
            EntityMetadata(entity_id="user:alice", entity_type="user")
        )
        store.save_metadata(
            EntityMetadata(entity_id="user:bob", entity_type="user")
        )
        store.save_metadata(
            EntityMetadata(entity_id="store:costco", entity_type="store")
        )

        users = sorted(store.list_entities(entity_type="user"))
        assert users == ["user:alice", "user:bob"]

        stores = store.list_entities(entity_type="store")
        assert stores == ["store:costco"]

    def test_list_entities_empty_store(self, store):
        """list_entities() on an empty store should return an empty list."""
        assert store.list_entities() == []

    def test_list_entities_nonexistent_type(self, store):
        """list_entities() with a type that has no entities should return empty."""
        store.save_metadata(
            EntityMetadata(entity_id="user:alice", entity_type="user")
        )
        assert store.list_entities(entity_type="tool") == []

    def test_list_entities_after_delete(self, store):
        """list_entities should not include deleted entities."""
        store.save_metadata(
            EntityMetadata(entity_id="user:alice", entity_type="user")
        )
        store.save_metadata(
            EntityMetadata(entity_id="user:bob", entity_type="user")
        )
        store.delete_metadata("user:alice")

        result = store.list_entities(entity_type="user")
        assert result == ["user:bob"]


class TestClose:
    """Adapter close should delegate to the underlying KV service."""

    def test_close_delegates_to_kv_service(self, store, kv_service):
        """close() should delegate to the underlying KV service's close()."""
        store.close()
        # MemoryKeyValueService sets _closed=True on close
        assert kv_service._closed is True

    def test_close_is_idempotent(self, store):
        """Calling close() multiple times should not raise."""
        store.close()
        store.close()  # Should not raise
