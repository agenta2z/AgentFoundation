"""
Unit tests for EntityMetadata data model.

Tests cover:
- EntityMetadata creation with defaults and explicit values
- Auto-generated timestamps (created_at, updated_at)
- get/set property access
- keys/items iteration
- set() updates updated_at timestamp
- Serialization round-trip (to_dict / from_dict)
- from_dict handles missing optional fields gracefully
- from_dict raises ValueError for missing required fields

Requirements: 1.1, 1.6
"""
import sys
import time
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

from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata


class TestEntityMetadataCreation:
    """Tests for EntityMetadata construction and defaults."""

    def test_minimal_creation(self):
        """Creating with only entity_id and entity_type sets all defaults."""
        meta = EntityMetadata(entity_id="user:xinli", entity_type="user")
        assert meta.entity_id == "user:xinli"
        assert meta.entity_type == "user"
        assert meta.properties == {}
        assert meta.created_at is not None
        assert meta.updated_at is not None

    def test_creation_with_properties(self):
        """Creating with properties preserves them."""
        props = {"name": "Xinli", "location": "Seattle"}
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties=props,
        )
        assert meta.properties == {"name": "Xinli", "location": "Seattle"}

    def test_explicit_timestamps_preserved(self):
        """Providing explicit timestamps preserves them."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-06-15T12:00:00+00:00",
        )
        assert meta.created_at == "2024-01-01T00:00:00+00:00"
        assert meta.updated_at == "2024-06-15T12:00:00+00:00"

    def test_auto_generated_timestamps_are_iso8601(self):
        """Auto-generated timestamps are valid ISO 8601 strings."""
        meta = EntityMetadata(entity_id="app:test", entity_type="app")
        # ISO 8601 timestamps contain 'T' separator and timezone info
        assert "T" in meta.created_at
        assert "T" in meta.updated_at

    def test_all_fields_explicit(self):
        """Creating with all fields explicitly set."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={"name": "Xinli", "zip": "98121"},
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-06-15T12:00:00+00:00",
        )
        assert meta.entity_id == "user:xinli"
        assert meta.entity_type == "user"
        assert meta.properties == {"name": "Xinli", "zip": "98121"}
        assert meta.created_at == "2024-01-01T00:00:00+00:00"
        assert meta.updated_at == "2024-06-15T12:00:00+00:00"


class TestGetSet:
    """Tests for get/set property access."""

    def test_get_existing_key(self):
        """get() returns the value for an existing key."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={"name": "Xinli"},
        )
        assert meta.get("name") == "Xinli"

    def test_get_missing_key_returns_default(self):
        """get() returns default when key is not found."""
        meta = EntityMetadata(entity_id="user:xinli", entity_type="user")
        assert meta.get("name") is None
        assert meta.get("name", "Unknown") == "Unknown"

    def test_set_new_key(self):
        """set() adds a new property."""
        meta = EntityMetadata(entity_id="user:xinli", entity_type="user")
        meta.set("name", "Xinli")
        assert meta.get("name") == "Xinli"

    def test_set_overwrites_existing_key(self):
        """set() overwrites an existing property value."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={"name": "Old Name"},
        )
        meta.set("name", "New Name")
        assert meta.get("name") == "New Name"

    def test_set_updates_updated_at(self):
        """set() updates the updated_at timestamp."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        old_updated = meta.updated_at
        time.sleep(0.01)  # Ensure timestamp changes
        meta.set("name", "Xinli")
        assert meta.updated_at != old_updated
        assert meta.updated_at > old_updated

    def test_set_various_value_types(self):
        """set() works with various value types."""
        meta = EntityMetadata(entity_id="user:xinli", entity_type="user")
        meta.set("name", "Xinli")
        meta.set("age", 30)
        meta.set("active", True)
        meta.set("preferences", ["organic", "local"])
        meta.set("config", {"theme": "dark"})

        assert meta.get("name") == "Xinli"
        assert meta.get("age") == 30
        assert meta.get("active") is True
        assert meta.get("preferences") == ["organic", "local"]
        assert meta.get("config") == {"theme": "dark"}


class TestKeysItems:
    """Tests for keys() and items() iteration."""

    def test_keys_empty(self):
        """keys() returns empty iterable for no properties."""
        meta = EntityMetadata(entity_id="user:xinli", entity_type="user")
        assert list(meta.keys()) == []

    def test_keys_returns_property_keys(self):
        """keys() returns all property keys."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={"name": "Xinli", "location": "Seattle"},
        )
        assert set(meta.keys()) == {"name", "location"}

    def test_items_empty(self):
        """items() returns empty iterable for no properties."""
        meta = EntityMetadata(entity_id="user:xinli", entity_type="user")
        assert list(meta.items()) == []

    def test_items_returns_key_value_pairs(self):
        """items() returns all (key, value) pairs."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={"name": "Xinli", "location": "Seattle"},
        )
        items = dict(meta.items())
        assert items == {"name": "Xinli", "location": "Seattle"}


class TestToDict:
    """Tests for to_dict serialization."""

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all fields."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={"name": "Xinli"},
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-06-15T12:00:00+00:00",
        )
        d = meta.to_dict()
        assert d["entity_id"] == "user:xinli"
        assert d["entity_type"] == "user"
        assert d["properties"] == {"name": "Xinli"}
        assert d["created_at"] == "2024-01-01T00:00:00+00:00"
        assert d["updated_at"] == "2024-06-15T12:00:00+00:00"

    def test_to_dict_empty_properties(self):
        """to_dict works with empty properties."""
        meta = EntityMetadata(entity_id="app:test", entity_type="app")
        d = meta.to_dict()
        assert d["properties"] == {}

    def test_to_dict_properties_is_copy(self):
        """to_dict returns a copy of properties, not a reference."""
        meta = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={"name": "Xinli"},
        )
        d = meta.to_dict()
        d["properties"]["name"] = "Modified"
        assert meta.get("name") == "Xinli"


class TestFromDict:
    """Tests for from_dict deserialization and validation."""

    def test_from_dict_full(self):
        """from_dict creates a valid EntityMetadata from a complete dictionary."""
        data = {
            "entity_id": "user:xinli",
            "entity_type": "user",
            "properties": {"name": "Xinli", "location": "Seattle"},
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-06-15T12:00:00+00:00",
        }
        meta = EntityMetadata.from_dict(data)
        assert meta.entity_id == "user:xinli"
        assert meta.entity_type == "user"
        assert meta.properties == {"name": "Xinli", "location": "Seattle"}
        assert meta.created_at == "2024-01-01T00:00:00+00:00"
        assert meta.updated_at == "2024-06-15T12:00:00+00:00"

    def test_from_dict_minimal(self):
        """from_dict with only required fields auto-generates defaults."""
        meta = EntityMetadata.from_dict({
            "entity_id": "user:xinli",
            "entity_type": "user",
        })
        assert meta.entity_id == "user:xinli"
        assert meta.entity_type == "user"
        assert meta.properties == {}
        assert meta.created_at is not None
        assert meta.updated_at is not None

    def test_from_dict_missing_entity_id_raises(self):
        """from_dict raises ValueError when entity_id is missing."""
        with pytest.raises(ValueError, match="Missing required field.*entity_id"):
            EntityMetadata.from_dict({"entity_type": "user"})

    def test_from_dict_missing_entity_type_raises(self):
        """from_dict raises ValueError when entity_type is missing."""
        with pytest.raises(ValueError, match="Missing required field.*entity_type"):
            EntityMetadata.from_dict({"entity_id": "user:xinli"})

    def test_from_dict_missing_optional_fields(self):
        """from_dict handles missing optional fields gracefully."""
        data = {
            "entity_id": "user:xinli",
            "entity_type": "user",
        }
        meta = EntityMetadata.from_dict(data)
        # properties defaults to empty dict
        assert meta.properties == {}
        # timestamps auto-generated
        assert meta.created_at is not None
        assert meta.updated_at is not None

    def test_from_dict_preserves_explicit_timestamps(self):
        """from_dict preserves explicitly provided timestamps."""
        data = {
            "entity_id": "user:xinli",
            "entity_type": "user",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-06-15T12:00:00+00:00",
        }
        meta = EntityMetadata.from_dict(data)
        assert meta.created_at == "2024-01-01T00:00:00+00:00"
        assert meta.updated_at == "2024-06-15T12:00:00+00:00"


class TestRoundTrip:
    """Tests for serialization round-trip. (Req 1.6)"""

    def test_round_trip_preserves_data(self):
        """to_dict followed by from_dict preserves all data."""
        original = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={"name": "Xinli", "location": "Seattle", "zip": "98121"},
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-06-15T12:00:00+00:00",
        )
        d = original.to_dict()
        restored = EntityMetadata.from_dict(d)

        assert restored.entity_id == original.entity_id
        assert restored.entity_type == original.entity_type
        assert restored.properties == original.properties
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at

    def test_round_trip_empty_properties(self):
        """Round-trip works with empty properties."""
        original = EntityMetadata(entity_id="app:test", entity_type="app")
        d = original.to_dict()
        restored = EntityMetadata.from_dict(d)

        assert restored.entity_id == original.entity_id
        assert restored.entity_type == original.entity_type
        assert restored.properties == original.properties
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at

    def test_round_trip_complex_properties(self):
        """Round-trip works with complex nested property values."""
        original = EntityMetadata(
            entity_id="user:xinli",
            entity_type="user",
            properties={
                "name": "Xinli",
                "preferences": ["organic", "local"],
                "config": {"theme": "dark", "notifications": True},
                "score": 4.5,
            },
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        d = original.to_dict()
        restored = EntityMetadata.from_dict(d)

        assert restored.properties == original.properties
