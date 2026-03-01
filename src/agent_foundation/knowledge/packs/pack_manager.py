"""
KnowledgePackManager — atomic install/uninstall/update for knowledge packs.

Manages packs across three existing stores (MetadataStore, KnowledgePieceStore,
EntityGraphStore) without requiring any ABC changes. Follows the atomicity
pattern from KnowledgeUpdater: add new first, deactivate old after, best-effort
rollback on failure.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphEdge,
    GraphNode,
)

from agent_foundation.knowledge.packs.models import (
    KnowledgePack,
    PackInstallResult,
    PackManagerConfig,
    PackSource,
    PackStatus,
)
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece

logger = logging.getLogger(__name__)

PACK_ENTITY_TYPE = "pack"
PACK_NODE_TYPE = "pack"
CONTAINS_RELATION = "CONTAINS"


class KnowledgePackManager:
    """Manages knowledge packs as atomic bundles of KnowledgePieces.

    Uses the three existing stores from KnowledgeBase:
    - MetadataStore: pack manifest as EntityMetadata(entity_type="pack")
    - EntityGraphStore: pack node with CONTAINS edges to piece nodes
    - KnowledgePieceStore: actual KnowledgePiece objects with provenance tags
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        config: Optional[PackManagerConfig] = None,
    ):
        self.kb = kb
        self.config = config or PackManagerConfig()
        self._metadata_store = kb.metadata_store
        self._piece_store = kb.piece_store
        self._graph_store = kb.graph_store

    # ── Install ──────────────────────────────────────────────────────────

    def install(
        self,
        pack: KnowledgePack,
        pieces: List[KnowledgePiece],
    ) -> PackInstallResult:
        """Atomically install a pack and its pieces.

        Steps:
        1. Add all pieces to the piece store
        2. Save pack metadata to the metadata store
        3. Create pack graph node and CONTAINS edges to piece nodes

        On failure, rolls back pieces that were already added.

        Args:
            pack: The KnowledgePack to install.
            pieces: The KnowledgePiece objects belonging to this pack.

        Returns:
            PackInstallResult with success status and details.
        """
        if self.is_installed(pack.pack_id):
            return PackInstallResult(
                success=False,
                pack_id=pack.pack_id,
                error=f"Pack already installed: {pack.pack_id}",
            )

        added_piece_ids = []
        try:
            # Step 1: Add all pieces
            for piece in pieces:
                if pack.spaces is not None:
                    piece.spaces = list(pack.spaces)
                    piece.space = piece.spaces[0]
                self._piece_store.add(piece)
                added_piece_ids.append(piece.piece_id)

            # Record piece IDs on the pack
            pack.piece_ids = added_piece_ids
            pack.status = PackStatus.INSTALLED

            # Step 2: Save pack metadata
            metadata = self._pack_to_metadata(pack)
            self._metadata_store.save_metadata(metadata)

            # Step 3: Create graph structure
            self._create_pack_graph(pack.pack_id, added_piece_ids)

            logger.info(
                "Installed pack %s with %d pieces", pack.pack_id, len(added_piece_ids)
            )
            return PackInstallResult(
                success=True,
                pack_id=pack.pack_id,
                pieces_installed=len(added_piece_ids),
            )

        except Exception as e:
            logger.error("Install failed for %s: %s. Rolling back.", pack.pack_id, e)
            # Rollback: remove pieces that were added
            for pid in added_piece_ids:
                try:
                    self._piece_store.remove(pid)
                except Exception:
                    pass
            # Rollback: remove metadata if it was saved
            try:
                self._metadata_store.delete_metadata(pack.pack_id)
            except Exception:
                pass
            # Rollback: remove graph node (cascades edges)
            try:
                self._graph_store.remove_node(pack.pack_id)
            except Exception:
                pass

            return PackInstallResult(
                success=False,
                pack_id=pack.pack_id,
                error=f"Install failed: {e}",
            )

    # ── Uninstall ────────────────────────────────────────────────────────

    def uninstall(
        self,
        pack_id: str,
        hard_delete: bool = False,
    ) -> PackInstallResult:
        """Atomically uninstall a pack and its pieces.

        Args:
            pack_id: The ID of the pack to uninstall.
            hard_delete: If True, permanently remove pieces.
                         If False, soft-delete (set is_active=False).

        Returns:
            PackInstallResult with success status.
        """
        pack = self.get(pack_id)
        if pack is None:
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"Pack not found: {pack_id}",
            )

        piece_ids = self._get_piece_ids_from_graph(pack_id)
        if not piece_ids:
            piece_ids = pack.piece_ids

        removed_count = 0
        try:
            for pid in piece_ids:
                if hard_delete:
                    if self._piece_store.remove(pid):
                        removed_count += 1
                else:
                    piece = self._piece_store.get_by_id(pid)
                    if piece:
                        piece.is_active = False
                        piece.updated_at = datetime.now(timezone.utc).isoformat()
                        self._piece_store.update(piece)
                        removed_count += 1

            # Remove graph node (cascades CONTAINS edges)
            self._graph_store.remove_node(pack_id)

            # Remove metadata
            self._metadata_store.delete_metadata(pack_id)

            logger.info(
                "Uninstalled pack %s (%d pieces %s)",
                pack_id,
                removed_count,
                "hard-deleted" if hard_delete else "soft-deleted",
            )
            return PackInstallResult(
                success=True,
                pack_id=pack_id,
                pieces_removed=removed_count,
            )

        except Exception as e:
            logger.error("Uninstall failed for %s: %s", pack_id, e)
            return PackInstallResult(
                success=False,
                pack_id=pack_id,
                error=f"Uninstall failed: {e}",
            )

    # ── Update ───────────────────────────────────────────────────────────

    def update(
        self,
        pack: KnowledgePack,
        new_pieces: List[KnowledgePiece],
    ) -> PackInstallResult:
        """Atomically update a pack with new pieces.

        Follows the add-new-deactivate-old pattern:
        1. Add all new pieces FIRST
        2. Deactivate old pieces AFTER
        3. Update metadata and graph

        Args:
            pack: The updated KnowledgePack (same pack_id, new version).
            new_pieces: The new KnowledgePiece objects.

        Returns:
            PackInstallResult with success status.
        """
        existing = self.get(pack.pack_id)
        if existing is None:
            return PackInstallResult(
                success=False,
                pack_id=pack.pack_id,
                error=f"Pack not found: {pack.pack_id}",
            )

        old_piece_ids = self._get_piece_ids_from_graph(pack.pack_id)
        if not old_piece_ids:
            old_piece_ids = existing.piece_ids

        added_piece_ids = []
        try:
            # Step 1: Add all new pieces FIRST
            for piece in new_pieces:
                if pack.spaces is not None:
                    piece.spaces = list(pack.spaces)
                    piece.space = piece.spaces[0]
                self._piece_store.add(piece)
                added_piece_ids.append(piece.piece_id)

            # Step 2: Deactivate old pieces AFTER
            for pid in old_piece_ids:
                old_piece = self._piece_store.get_by_id(pid)
                if old_piece and old_piece.is_active:
                    old_piece.is_active = False
                    old_piece.updated_at = datetime.now(timezone.utc).isoformat()
                    self._piece_store.update(old_piece)

            # Step 3: Update metadata
            pack.piece_ids = added_piece_ids
            pack.status = PackStatus.INSTALLED
            pack.updated_at = datetime.now(timezone.utc).isoformat()
            metadata = self._pack_to_metadata(pack)
            self._metadata_store.save_metadata(metadata)

            # Step 4: Rebuild graph (remove old, create new)
            self._graph_store.remove_node(pack.pack_id)
            self._create_pack_graph(pack.pack_id, added_piece_ids)

            logger.info(
                "Updated pack %s: %d old pieces deactivated, %d new pieces added",
                pack.pack_id,
                len(old_piece_ids),
                len(added_piece_ids),
            )
            return PackInstallResult(
                success=True,
                pack_id=pack.pack_id,
                pieces_installed=len(added_piece_ids),
                pieces_removed=len(old_piece_ids),
            )

        except Exception as e:
            logger.error("Update failed for %s: %s. Rolling back.", pack.pack_id, e)
            # Rollback: remove new pieces
            for pid in added_piece_ids:
                try:
                    self._piece_store.remove(pid)
                except Exception:
                    pass
            return PackInstallResult(
                success=False,
                pack_id=pack.pack_id,
                error=f"Update failed: {e}",
            )

    # ── Query ────────────────────────────────────────────────────────────

    def get(self, pack_id: str) -> Optional[KnowledgePack]:
        """Reconstruct a KnowledgePack from MetadataStore.

        Args:
            pack_id: The pack ID to look up.

        Returns:
            The KnowledgePack if found, None otherwise.
        """
        metadata = self._metadata_store.get_metadata(pack_id)
        if metadata is None or metadata.entity_type != PACK_ENTITY_TYPE:
            return None
        return self._metadata_to_pack(metadata)

    def list_packs(
        self,
        source_type: Optional[PackSource] = None,
    ) -> List[KnowledgePack]:
        """List all installed packs, optionally filtered by source type.

        Args:
            source_type: If specified, only return packs from this source.

        Returns:
            List of KnowledgePack objects.
        """
        entity_ids = self._metadata_store.list_entities(
            entity_type=PACK_ENTITY_TYPE,
        )
        packs = []
        for eid in entity_ids:
            pack = self.get(eid)
            if pack is None:
                continue
            if source_type and pack.source_type != source_type:
                continue
            packs.append(pack)
        return packs

    def get_pack_pieces(self, pack_id: str) -> List[KnowledgePiece]:
        """Get all pieces belonging to a pack via graph CONTAINS edges.

        Args:
            pack_id: The pack ID.

        Returns:
            List of KnowledgePiece objects in the pack.
        """
        piece_ids = self._get_piece_ids_from_graph(pack_id)
        pieces = []
        for pid in piece_ids:
            piece = self._piece_store.get_by_id(pid)
            if piece:
                pieces.append(piece)
        return pieces

    def is_installed(self, pack_id: str) -> bool:
        """Check if a pack is installed.

        Args:
            pack_id: The pack ID to check.

        Returns:
            True if the pack exists in the metadata store.
        """
        metadata = self._metadata_store.get_metadata(pack_id)
        return metadata is not None and metadata.entity_type == PACK_ENTITY_TYPE

    def get_pack_for_piece(self, piece_id: str) -> Optional[KnowledgePack]:
        """Reverse lookup: find which pack owns a given piece.

        Uses incoming CONTAINS edges on the piece node.

        Args:
            piece_id: The piece ID to look up.

        Returns:
            The owning KnowledgePack if found, None otherwise.
        """
        relations = self._graph_store.get_relations(
            piece_id, relation_type=CONTAINS_RELATION, direction="incoming"
        )
        for rel in relations:
            pack = self.get(rel.source_id)
            if pack is not None:
                return pack
        return None

    # ── Internal helpers ─────────────────────────────────────────────────

    def _pack_to_metadata(self, pack: KnowledgePack) -> EntityMetadata:
        """Convert a KnowledgePack to EntityMetadata for storage."""
        return EntityMetadata(
            entity_id=pack.pack_id,
            entity_type=PACK_ENTITY_TYPE,
            properties=pack.to_dict(),
            created_at=pack.installed_at,
            updated_at=pack.updated_at,
        )

    def _metadata_to_pack(self, metadata: EntityMetadata) -> KnowledgePack:
        """Reconstruct a KnowledgePack from stored EntityMetadata."""
        return KnowledgePack.from_dict(metadata.properties)

    def _create_pack_graph(self, pack_id: str, piece_ids: List[str]) -> None:
        """Create a pack graph node with CONTAINS edges to piece nodes."""
        pack_node = GraphNode(
            node_id=pack_id,
            node_type=PACK_NODE_TYPE,
            label=pack_id,
        )
        self._graph_store.add_node(pack_node)

        for pid in piece_ids:
            # Ensure piece node exists
            piece_node = self._graph_store.get_node(pid)
            if piece_node is None:
                piece_node = GraphNode(
                    node_id=pid,
                    node_type="knowledge_piece",
                    label=pid,
                )
                self._graph_store.add_node(piece_node)

            edge = GraphEdge(
                source_id=pack_id,
                target_id=pid,
                edge_type=CONTAINS_RELATION,
            )
            self._graph_store.add_relation(edge)

    def _get_piece_ids_from_graph(self, pack_id: str) -> List[str]:
        """Get piece IDs from CONTAINS edges in the graph."""
        relations = self._graph_store.get_relations(
            pack_id, relation_type=CONTAINS_RELATION, direction="outgoing"
        )
        return [rel.target_id for rel in relations]
