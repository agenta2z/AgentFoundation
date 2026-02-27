"""
EntityMetadata data model.

Provides structured key-value storage for any entity (user, app, tool, global).
Think of it as a typed profile/config with get/set access, serialization, and
automatic timestamp management.

Requirements: 1.1, 1.6
"""
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

from attr import attrs, attrib


@attrs
class EntityMetadata:
    """Structured key-value storage for any entity.

    Attributes:
        entity_id: Entity identifier, e.g., "user:xinli", "app:grocery_checker".
        entity_type: Entity type, e.g., "user", "app", "tool", "global".
        properties: Key-value pairs storing entity metadata.
        created_at: ISO 8601 creation timestamp. Auto-generated if None.
        updated_at: ISO 8601 last-update timestamp. Auto-generated if None.
    """
    entity_id: str = attrib()
    entity_type: str = attrib()
    properties: Dict[str, Any] = attrib(factory=dict)
    created_at: str = attrib(default=None)
    updated_at: str = attrib(default=None)

    def __attrs_post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    def get(self, key: str, default: Any = None) -> Any:
        """Get a property by key.

        Args:
            key: The property key to look up.
            default: Value to return if key is not found.

        Returns:
            The property value, or default if not found.
        """
        return self.properties.get(key, default)

    def set(self, key: str, value: Any):
        """Set a property. Updates updated_at timestamp.

        Args:
            key: The property key to set.
            value: The value to associate with the key.
        """
        self.properties[key] = value
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def keys(self) -> Iterable[str]:
        """Iterate over property keys.

        Returns:
            An iterable of property key strings.
        """
        return self.properties.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Iterate over (key, value) pairs.

        Returns:
            An iterable of (key, value) tuples.
        """
        return self.properties.items()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of this EntityMetadata with all fields.
        """
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "properties": dict(self.properties),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityMetadata":
        """Deserialize from dictionary. Handles missing optional fields gracefully.

        Args:
            data: Dictionary containing EntityMetadata fields.
                  Required: entity_id, entity_type.
                  Optional: properties, created_at, updated_at.

        Returns:
            A new EntityMetadata instance.

        Raises:
            ValueError: If required fields (entity_id, entity_type) are missing.
        """
        if "entity_id" not in data:
            raise ValueError("Missing required field: 'entity_id'")
        if "entity_type" not in data:
            raise ValueError("Missing required field: 'entity_type'")

        return cls(
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            properties=data.get("properties", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
