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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from attr import attrs, attrib

from rich_python_utils.service_utils.data_operation_record import (
    DataOperationRecord,
    generate_operation_id,
)
from rich_python_utils.service_utils.keyvalue_service.keyvalue_service_base import (
    KeyValueServiceBase,
)
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.stores.metadata.base import MetadataStore
from agent_foundation.knowledge.retrieval.utils import parse_entity_type


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

    def get_metadata(
        self,
        entity_id: str,
        include_inactive: bool = False,
    ) -> Optional[EntityMetadata]:
        """Get metadata for an entity.

        Retrieves the dictionary from the KV service using entity_type as
        namespace and entity_id as key, then deserializes to EntityMetadata.
        Returns None for soft-deleted entities unless include_inactive=True.

        Args:
            entity_id: The unique identifier of the entity, e.g., "user:xinli".
            include_inactive: If True, return soft-deleted entities too.

        Returns:
            The EntityMetadata for the given entity, or None if not found
            (or soft-deleted when include_inactive=False).
        """
        entity_type = parse_entity_type(entity_id)
        data = self.kv_service.get(entity_id, namespace=entity_type)
        if not data:
            return None
        metadata = EntityMetadata.from_dict(data)
        if not include_inactive and not metadata.is_active:
            return None
        return metadata

    def save_metadata(
        self,
        metadata: EntityMetadata,
        operation_id: Optional[str] = None,
    ) -> None:
        """Save or overwrite metadata for an entity with history tracking.

        Fetches existing metadata first. If existing, appends an UPDATE
        record with properties_before/after. If new, appends an ADD record.

        Args:
            metadata: The EntityMetadata to persist.
            operation_id: Optional shared operation ID for batch grouping.
        """
        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("MetadataStore", "save")
        entity_type = parse_entity_type(metadata.entity_id)

        existing_data = self.kv_service.get(metadata.entity_id, namespace=entity_type)
        if existing_data:
            existing = EntityMetadata.from_dict(existing_data)
            metadata.history = existing.history + metadata.history  # Preserve existing history
            metadata.history.append(DataOperationRecord(
                operation="update",
                timestamp=now,
                operation_id=op_id,
                source="KeyValueMetadataStore.save_metadata",
                properties_before=dict(existing.properties),
                properties_after=dict(metadata.properties),
            ))
        else:
            metadata.history.append(DataOperationRecord(
                operation="add",
                timestamp=now,
                operation_id=op_id,
                source="KeyValueMetadataStore.save_metadata",
            ))

        metadata.updated_at = now
        self.kv_service.put(
            metadata.entity_id,
            metadata.to_dict(),
            namespace=metadata.entity_type,
        )

    def delete_metadata(
        self,
        entity_id: str,
        operation_id: Optional[str] = None,
    ) -> bool:
        """Soft-delete metadata for an entity.

        Fetches the existing metadata, sets is_active=False, appends a
        DELETE history record, and overwrites in the KV store.

        Args:
            entity_id: The unique identifier of the entity to delete.
            operation_id: Optional shared operation ID for batch grouping.

        Returns:
            True if the entity existed and was soft-deleted,
            False if not found or already inactive.
        """
        entity_type = parse_entity_type(entity_id)
        data = self.kv_service.get(entity_id, namespace=entity_type)
        if not data:
            return False

        metadata = EntityMetadata.from_dict(data)
        if not metadata.is_active:
            return False  # Already soft-deleted

        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("MetadataStore", "delete")
        metadata.is_active = False
        metadata.history.append(DataOperationRecord(
            operation="delete",
            timestamp=now,
            operation_id=op_id,
            source="KeyValueMetadataStore.delete_metadata",
            properties_before=dict(metadata.properties),
            details={"delete_mode": "soft"},
        ))
        metadata.updated_at = now
        self.kv_service.put(entity_id, metadata.to_dict(), namespace=entity_type)
        return True

    def list_entities(
        self,
        entity_type: str = None,
        include_inactive: bool = False,
    ) -> List[str]:
        """List entity IDs, optionally filtered by type.

        When entity_type is provided, returns keys from that namespace only.
        When entity_type is None, aggregates keys across all namespaces.

        Args:
            entity_type: If provided, only return entities of this type.
                         If None, return all entity IDs.
            include_inactive: If True, include soft-deleted entities.

        Returns:
            A list of entity ID strings matching the filter criteria.
        """
        if entity_type:
            keys = self.kv_service.keys(namespace=entity_type)
        else:
            keys = []
            for ns in self.kv_service.namespaces():
                keys.extend(self.kv_service.keys(namespace=ns))

        if include_inactive:
            return keys

        # Filter out inactive entities
        active_keys = []
        for key in keys:
            et = parse_entity_type(key)
            data = self.kv_service.get(key, namespace=et)
            if data and data.get("is_active", True):
                active_keys.append(key)
        return active_keys

    def close(self):
        """Close the underlying key-value service."""
        self.kv_service.close()
