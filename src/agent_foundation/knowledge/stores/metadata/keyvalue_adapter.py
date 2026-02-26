"""
KeyValueMetadataStore — MetadataStore adapter backed by KeyValueServiceBase.

Implements the MetadataStore ABC by delegating all operations to a
general-purpose KeyValueServiceBase instance. This adapter bridges the
domain-specific knowledge module with the generic key-value service from
SciencePythonUtils.

Mapping:
    - entity_type → KV namespace
    - entity_id   → KV key
    - EntityMetadata.to_dict() → KV value (JSON-serializable dict)

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""
from typing import List, Optional

from attr import attrs, attrib

from rich_python_utils.service_utils.keyvalue_service.keyvalue_service_base import (
    KeyValueServiceBase,
)
from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from science_modeling_tools.knowledge.stores.metadata.base import MetadataStore
from science_modeling_tools.knowledge.utils import parse_entity_type


@attrs
class KeyValueMetadataStore(MetadataStore):
    """MetadataStore backed by any KeyValueServiceBase.

    Delegates all CRUD operations to a generic key-value service, using
    entity_type as the namespace and entity_id as the key. EntityMetadata
    objects are serialized to/from dictionaries via to_dict/from_dict.

    Attributes:
        kv_service: The underlying key-value service instance.
    """

    kv_service: KeyValueServiceBase = attrib()

    def get_metadata(self, entity_id: str) -> Optional[EntityMetadata]:
        """Get metadata for an entity.

        Retrieves the dictionary from the KV service using entity_type as
        namespace and entity_id as key, then deserializes to EntityMetadata.

        Args:
            entity_id: The unique identifier of the entity, e.g., "user:xinli".

        Returns:
            The EntityMetadata for the given entity, or None if not found.
        """
        entity_type = parse_entity_type(entity_id)
        data = self.kv_service.get(entity_id, namespace=entity_type)
        return EntityMetadata.from_dict(data) if data else None

    def save_metadata(self, metadata: EntityMetadata) -> None:
        """Save or overwrite metadata for an entity.

        Serializes the EntityMetadata to a dictionary and stores it in the
        KV service using entity_type as namespace and entity_id as key.

        Args:
            metadata: The EntityMetadata to persist.
        """
        self.kv_service.put(
            metadata.entity_id,
            metadata.to_dict(),
            namespace=metadata.entity_type,
        )

    def delete_metadata(self, entity_id: str) -> bool:
        """Delete metadata for an entity.

        Args:
            entity_id: The unique identifier of the entity to delete.

        Returns:
            True if the entity existed and was deleted, False if not found.
        """
        entity_type = parse_entity_type(entity_id)
        return self.kv_service.delete(entity_id, namespace=entity_type)

    def list_entities(self, entity_type: str = None) -> List[str]:
        """List entity IDs, optionally filtered by type.

        When entity_type is provided, returns keys from that namespace only.
        When entity_type is None, aggregates keys across all namespaces.

        Args:
            entity_type: If provided, only return entities of this type.
                         If None, return all entity IDs.

        Returns:
            A list of entity ID strings matching the filter criteria.
        """
        if entity_type:
            return self.kv_service.keys(namespace=entity_type)
        # Aggregate across all namespaces
        all_keys = []
        for ns in self.kv_service.namespaces():
            all_keys.extend(self.kv_service.keys(namespace=ns))
        return all_keys

    def close(self):
        """Close the underlying key-value service."""
        self.kv_service.close()
