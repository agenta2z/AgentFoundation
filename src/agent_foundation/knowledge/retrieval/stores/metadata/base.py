"""
MetadataStore abstract base class.

Defines the abstract interface for metadata storage backends. All metadata
store implementations (file-based, SQLite, etc.) must implement this interface.

The MetadataStore manages structured key-value metadata for entities. Each entity
has an EntityMetadata object containing properties, timestamps, and type information.

Requirements: 2.1
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata


class MetadataStore(ABC):
    """Abstract base class for metadata storage backends.

    Provides the interface for CRUD operations on entity metadata. Implementations
    may use file-based JSON storage, SQLite, or other backends.

    All implementations must support:
    - Getting metadata by entity ID
    - Saving (creating or overwriting) metadata
    - Deleting metadata by entity ID
    - Listing entity IDs with optional type filtering

    The ``close()`` method is a concrete no-op by default. Subclasses that hold
    external connections (e.g., SQLite) should override it to release resources.
    """

    @abstractmethod
    def get_metadata(self, entity_id: str) -> Optional[EntityMetadata]:
        """Get metadata for an entity.

        Args:
            entity_id: The unique identifier of the entity, e.g., "user:xinli".

        Returns:
            The EntityMetadata for the given entity, or None if not found.
        """
        ...

    @abstractmethod
    def save_metadata(self, metadata: EntityMetadata) -> None:
        """Save or overwrite metadata for an entity.

        If metadata for the entity already exists, it is replaced entirely.
        If it does not exist, a new entry is created.

        Args:
            metadata: The EntityMetadata to persist.
        """
        ...

    @abstractmethod
    def delete_metadata(self, entity_id: str) -> bool:
        """Delete metadata for an entity.

        Args:
            entity_id: The unique identifier of the entity to delete.

        Returns:
            True if the entity existed and was deleted, False if not found.
        """
        ...

    @abstractmethod
    def list_entities(self, entity_type: str = None) -> List[str]:
        """List entity IDs, optionally filtered by type.

        Args:
            entity_type: If provided, only return entities of this type.
                         If None, return all entity IDs.

        Returns:
            A list of entity ID strings matching the filter criteria.
        """
        ...

    def close(self):
        """Close any underlying connections.

        Default no-op for file-based stores. Override for stores with
        external connections (e.g., SQLite) to release resources.
        """
        pass
