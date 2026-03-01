"""
LanceDBKnowledgePieceStore — LanceDB-based knowledge piece storage backend.

Uses ``lancedb`` for vector storage with native hybrid search combining
vector similarity (ANN) and full-text search (BM25 via Tantivy). This is
the most capable knowledge piece store — it indexes both ``embedding_text``
(vector) and ``content`` (FTS) in a single store.

Architecture:
    - A single LanceDB table stores knowledge pieces with columns for
      piece_id, content, embedding_text, knowledge_type, tags (JSON string),
      entity_id, source, created_at, updated_at, and vector.
    - The ``embedding_function`` parameter is required and must be a callable
      that accepts a string and returns a list of floats (the embedding vector).
      Typically ``SentenceTransformer('all-MiniLM-L6-v2').encode``.
    - Search performs separate vector ANN and BM25 FTS queries, then combines
      scores: ``score = hybrid_alpha * vector_score + (1 - hybrid_alpha) * bm25_score``
    - ``hybrid_alpha`` defaults to 0.7 (70% vector, 30% BM25).
    - Tag filtering is done post-query since LanceDB doesn't support JSON
      array containment.
    - Both vector and BM25 scores are normalized to [0.0, 1.0] before combining.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 4.2, 4.4, 4.6
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from attr import attrs, attrib

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore

logger = logging.getLogger(__name__)

# Sentinel value for global pieces (entity_id=None) in LanceDB.
_GLOBAL_ENTITY_SENTINEL = "__global__"



def _piece_to_record(piece, vector):
    """Convert a KnowledgePiece and its embedding vector to a LanceDB record."""
    return {
        "piece_id": piece.piece_id,
        "content": piece.content,
        "embedding_text": piece.embedding_text or "",
        "knowledge_type": piece.knowledge_type.value if piece.knowledge_type else KnowledgeType.Fact.value,
        "tags": json.dumps(piece.tags, ensure_ascii=False),
        "entity_id": piece.entity_id if piece.entity_id is not None else _GLOBAL_ENTITY_SENTINEL,
        "source": piece.source or "",
        "created_at": piece.created_at or "",
        "updated_at": piece.updated_at or "",
        "vector": vector,
        # New fields
        "domain": piece.domain or "general",
        "content_hash": piece.content_hash or "",
        "space": piece.space or "main",
        "is_active": piece.is_active,
        "version": piece.version,
        "summary": piece.summary or "",
        "validation_status": piece.validation_status or "not_validated",
        "merge_strategy": piece.merge_strategy or "",
        "supersedes": piece.supersedes or "",
        # List fields stored as JSON strings
        "secondary_domains": json.dumps(piece.secondary_domains, ensure_ascii=False),
        "custom_tags": json.dumps(piece.custom_tags, ensure_ascii=False),
        "validation_issues": json.dumps(piece.validation_issues, ensure_ascii=False),
        # Multi-space membership
        "spaces": json.dumps(piece.spaces, ensure_ascii=False),
        "primary_space": piece.spaces[0] if piece.spaces else "main",
        # Space suggestion fields
        "pending_space_suggestions": json.dumps(piece.pending_space_suggestions, ensure_ascii=False) if piece.pending_space_suggestions else "",
        "space_suggestion_reasons": json.dumps(piece.space_suggestion_reasons, ensure_ascii=False) if piece.space_suggestion_reasons else "",
        "space_suggestion_status": piece.space_suggestion_status or "",
    }





def _record_to_piece(record):
    """Convert a LanceDB record dict back to a KnowledgePiece."""
    entity_id = record.get("entity_id")
    if entity_id == _GLOBAL_ENTITY_SENTINEL:
        entity_id = None

    source = record.get("source", "")
    if source == "":
        source = None

    embedding_text = record.get("embedding_text", "")
    if embedding_text == "":
        embedding_text = None

    try:
        tags = json.loads(record.get("tags", "[]"))
    except (json.JSONDecodeError, TypeError):
        tags = []

    knowledge_type_str = record.get("knowledge_type", KnowledgeType.Fact.value)
    if isinstance(knowledge_type_str, str):
        knowledge_type = KnowledgeType(knowledge_type_str)
    else:
        knowledge_type = knowledge_type_str

    # Deserialize list fields from JSON strings
    def _parse_json_list(value):
        if not value:
            return []
        if isinstance(value, list):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []

    secondary_domains = _parse_json_list(record.get("secondary_domains", "[]"))
    custom_tags = _parse_json_list(record.get("custom_tags", "[]"))
    validation_issues = _parse_json_list(record.get("validation_issues", "[]"))

    # Deserialize optional string fields (empty string → None)
    summary = record.get("summary", "") or None
    merge_strategy = record.get("merge_strategy", "") or None
    supersedes = record.get("supersedes", "") or None
    content_hash = record.get("content_hash", "") or None

    # Deserialize multi-space membership
    spaces_raw = record.get("spaces")
    if spaces_raw:
        spaces = _parse_json_list(spaces_raw)
        if not spaces:
            spaces = [record.get("space", "main")]
    else:
        # Fallback for records without the spaces column (pre-migration data)
        spaces = [record.get("space", "main")]
    # Ignore primary_space — it's derived from spaces[0]

    # Deserialize space suggestion fields
    pending_space_suggestions_raw = record.get("pending_space_suggestions", "")
    pending_space_suggestions = _parse_json_list(pending_space_suggestions_raw) if pending_space_suggestions_raw else None
    if pending_space_suggestions is not None and len(pending_space_suggestions) == 0:
        pending_space_suggestions = None

    space_suggestion_reasons_raw = record.get("space_suggestion_reasons", "")
    space_suggestion_reasons = _parse_json_list(space_suggestion_reasons_raw) if space_suggestion_reasons_raw else None
    if space_suggestion_reasons is not None and len(space_suggestion_reasons) == 0:
        space_suggestion_reasons = None

    space_suggestion_status = record.get("space_suggestion_status", "") or None

    return KnowledgePiece(
        content=record.get("content", ""),
        piece_id=record.get("piece_id", ""),
        knowledge_type=knowledge_type,
        tags=tags,
        entity_id=entity_id,
        source=source,
        embedding_text=embedding_text,
        created_at=record.get("created_at", ""),
        updated_at=record.get("updated_at", ""),
        # New fields
        domain=record.get("domain", "general"),
        secondary_domains=secondary_domains,
        custom_tags=custom_tags,
        content_hash=content_hash,
        space=record.get("space", "main"),
        is_active=record.get("is_active", True),
        version=record.get("version", 1),
        summary=summary,
        validation_status=record.get("validation_status", "not_validated"),
        validation_issues=validation_issues,
        merge_strategy=merge_strategy,
        supersedes=supersedes,
        # Multi-space membership
        spaces=spaces,
        # Space suggestion fields
        pending_space_suggestions=pending_space_suggestions,
        space_suggestion_reasons=space_suggestion_reasons,
        space_suggestion_status=space_suggestion_status,
    )



def _get_embedding_text(piece):
    """Get the text to embed for a piece. Uses embedding_text if available, else content."""
    return piece.embedding_text if piece.embedding_text else piece.content


@attrs
class LanceDBKnowledgePieceStore(KnowledgePieceStore):
    """LanceDB-based knowledge piece store with hybrid search.

    Stores knowledge pieces in a LanceDB table with vector embeddings for
    semantic similarity search and a full-text search index on the content
    column for BM25 keyword search. Hybrid search combines both:
    ``score = hybrid_alpha * vector_score + (1 - hybrid_alpha) * bm25_score``

    Attributes:
        db_path: Directory for LanceDB data files.
        embedding_function: A callable that accepts a string and returns a
            list of floats (the embedding vector).
        table_name: Name of the LanceDB table.
        hybrid_alpha: Balance between vector and FTS search.
            0.0 = pure FTS, 1.0 = pure vector. Defaults to 0.7.
    """

    db_path: str = attrib()
    embedding_function: Callable = attrib()
    table_name: str = attrib(default="knowledge_pieces")
    hybrid_alpha: float = attrib(default=0.7)
    _db: Any = attrib(init=False, default=None)
    _table: Any = attrib(init=False, default=None)
    _fts_index_created: bool = attrib(init=False, default=False)

    @property
    def supports_space_filter(self) -> bool:
        """LanceDB natively supports space filtering via SQL WHERE clauses."""
        return True

    def __attrs_post_init__(self):
        """Initialize LanceDB connection and open or create the table."""
        import lancedb as _lancedb

        os.makedirs(self.db_path, exist_ok=True)
        self._db = _lancedb.connect(self.db_path)

        existing_tables = self._db.table_names()
        if self.table_name in existing_tables:
            self._table = self._db.open_table(self.table_name)
            self._fts_index_created = True
            self._migrate_schema_if_needed()
        else:
            self._table = None
            self._fts_index_created = False

    def _migrate_schema_if_needed(self):
        """Check for missing spaces/primary_space columns and migrate if needed.

        Migration is idempotent — if columns already exist, no action is taken.
        Derives ``spaces`` from the existing ``space`` column as ``[space]``
        and sets ``primary_space`` to ``space`` for each existing record.
        """
        if self._table is None:
            return
        try:
            sample = self._table.search().limit(1).to_list()
        except Exception as exc:
            logger.warning("LanceDB schema migration check failed: %s", exc)
            return
        if not sample:
            return
        if "spaces" in sample[0] and "primary_space" in sample[0]:
            return  # Already migrated

        logger.info("Migrating LanceDB table '%s' to add spaces columns...", self.table_name)
        try:
            all_records = self._table.search().limit(100000).to_list()
            for record in all_records:
                space = record.get("space", "main")
                record["spaces"] = json.dumps([space], ensure_ascii=False)
                record["primary_space"] = space
                record["pending_space_suggestions"] = ""
                record["space_suggestion_reasons"] = ""
                record["space_suggestion_status"] = ""
            self._db.drop_table(self.table_name)
            self._table = self._db.create_table(self.table_name, all_records)
            self._fts_index_created = False
            self._create_fts_index()
            logger.info("Schema migration complete for table '%s' (%d records).", self.table_name, len(all_records))
        except Exception as exc:
            logger.error("LanceDB schema migration failed for table '%s': %s", self.table_name, exc)

    def _ensure_table(self, first_record):
        """Create the table with the first record if it doesn't exist yet."""
        if self._table is not None:
            return
        self._table = self._db.create_table(self.table_name, [first_record])
        self._create_fts_index()

    def _create_fts_index(self):
        """Create the FTS index on the content column for BM25 search."""
        if self._fts_index_created or self._table is None:
            return
        try:
            self._table.create_fts_index("content", replace=True)
            self._fts_index_created = True
        except Exception as exc:
            logger.warning("Failed to create FTS index: %s", exc)

    def _embed(self, text):
        """Embed a text string using the configured embedding function."""
        result = self.embedding_function(text)
        if hasattr(result, "tolist"):
            return result.tolist()
        return list(result)

    def add(self, piece):
        """Add a knowledge piece to the LanceDB table.

        Raises ValueError if a piece with the same piece_id already exists.
        """
        if self._table is not None:
            existing = (
                self._table.search()
                .where(f"piece_id = '{_escape_sql(piece.piece_id)}'")
                .limit(1)
                .to_list()
            )
            if existing:
                raise ValueError(
                    f"Duplicate piece_id: '{piece.piece_id}' already exists"
                )

        embed_text = _get_embedding_text(piece)
        vector = self._embed(embed_text)
        record = _piece_to_record(piece, vector)

        if self._table is None:
            self._ensure_table(record)
        else:
            self._table.add([record])
            self._rebuild_fts_index()

        return piece.piece_id

    def get_by_id(self, piece_id):
        """Get a knowledge piece by its ID."""
        if self._table is None:
            return None
        try:
            results = (
                self._table.search()
                .where(f"piece_id = '{_escape_sql(piece_id)}'")
                .limit(1)
                .to_list()
            )
        except Exception as exc:
            logger.warning("LanceDB get_by_id error for '%s': %s", piece_id, exc)
            return None
        if not results:
            return None
        return _record_to_piece(results[0])

    def update(self, piece):
        """Update an existing knowledge piece. Returns True if found and updated."""
        if self._table is None:
            return False

        existing = (
            self._table.search()
            .where(f"piece_id = '{_escape_sql(piece.piece_id)}'")
            .limit(1)
            .to_list()
        )
        if not existing:
            return False

        now = datetime.now(timezone.utc).isoformat()
        piece.updated_at = now

        self._table.delete(f"piece_id = '{_escape_sql(piece.piece_id)}'")

        embed_text = _get_embedding_text(piece)
        vector = self._embed(embed_text)
        record = _piece_to_record(piece, vector)
        self._table.add([record])
        self._rebuild_fts_index()

        return True

    def remove(self, piece_id):
        """Remove a knowledge piece. Returns True if existed and was removed."""
        if self._table is None:
            return False

        existing = (
            self._table.search()
            .where(f"piece_id = '{_escape_sql(piece_id)}'")
            .limit(1)
            .to_list()
        )
        if not existing:
            return False

        self._table.delete(f"piece_id = '{_escape_sql(piece_id)}'")
        self._rebuild_fts_index()
        return True

    def search(self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5, spaces=None):
        """Hybrid search combining vector similarity and BM25 full-text search.

        score = hybrid_alpha * vector_score + (1 - hybrid_alpha) * bm25_score
        Both vector and BM25 scores are normalized to [0.0, 1.0] before combining.
        """
        if not query or not query.strip():
            return []
        if self._table is None:
            return []

        where_clause = _build_where_clause(entity_id, knowledge_type, spaces=spaces)
        fetch_limit = top_k * 5 if tags else top_k

        # Vector search
        vector_scores = {}
        try:
            query_vector = self._embed(query)
            vector_builder = self._table.search(query_vector).metric("cosine")
            if where_clause:
                vector_builder = vector_builder.where(where_clause)
            vector_results = vector_builder.limit(fetch_limit).to_list()

            if vector_results:
                for row in vector_results:
                    pid = row.get("piece_id", "")
                    distance = row.get("_distance", 1.0)
                    score = max(0.0, 1.0 - distance)
                    vector_scores[pid] = score
        except Exception as exc:
            logger.warning("LanceDB vector search error: %s", exc)

        # FTS (BM25) search
        bm25_scores = {}
        if self._fts_index_created:
            try:
                fts_builder = self._table.search(query, query_type="fts")
                if where_clause:
                    fts_builder = fts_builder.where(where_clause)
                fts_results = fts_builder.limit(fetch_limit).to_list()

                if fts_results:
                    raw_scores = []
                    for row in fts_results:
                        pid = row.get("piece_id", "")
                        score = row.get("_score", 0.0)
                        if score is None:
                            score = 0.0
                        raw_scores.append((pid, float(score)))

                    if raw_scores:
                        max_score = max(s for _, s in raw_scores)
                        if max_score > 0:
                            for pid, raw in raw_scores:
                                bm25_scores[pid] = raw / max_score
                        else:
                            for pid, _ in raw_scores:
                                bm25_scores[pid] = 0.0
            except Exception as exc:
                logger.warning("LanceDB FTS search error: %s", exc)

        # Combine scores
        all_piece_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        if not all_piece_ids:
            return []

        combined = {}
        alpha = self.hybrid_alpha
        for pid in all_piece_ids:
            v_score = vector_scores.get(pid, 0.0)
            b_score = bm25_scores.get(pid, 0.0)
            combined[pid] = alpha * v_score + (1.0 - alpha) * b_score

        # Retrieve full pieces and apply tag filtering
        scored_pieces = []
        for pid, score in combined.items():
            piece = self.get_by_id(pid)
            if piece is None:
                continue
            score = max(0.0, min(1.0, score))
            scored_pieces.append((piece, score))

        if tags:
            filter_tags = {t.strip().lower() for t in tags if t.strip()}
            scored_pieces = [
                (piece, score)
                for piece, score in scored_pieces
                if filter_tags.issubset(set(piece.tags))
            ]

        scored_pieces.sort(key=lambda x: (-x[1], x[0].piece_id))
        return scored_pieces[:top_k]

    def list_all(self, entity_id=None, knowledge_type=None, spaces=None, limit=10000):
        """List all pieces matching the given filters.

        Args:
            entity_id: If specified, list only this entity's pieces.
            knowledge_type: If specified, filter to this knowledge type only.
            spaces: If specified, filter to pieces belonging to at least one of these spaces.
            limit: Maximum number of records to return. Defaults to 10000 for
                backward compatibility. Use a larger value for migration use cases
                that need to iterate all pieces without truncation.
        """
        if self._table is None:
            return []

        where_clause = _build_where_clause(entity_id, knowledge_type, spaces=spaces)

        try:
            if where_clause:
                results = (
                    self._table.search()
                    .where(where_clause)
                    .limit(limit)
                    .to_list()
                )
            else:
                results = self._table.search().limit(limit).to_list()
        except Exception as exc:
            logger.warning("LanceDB list_all error: %s", exc)
            return []

        return [_record_to_piece(row) for row in results]

    def find_by_content_hash(self, content_hash, entity_id=None):
        """Find a piece by content hash using indexed SQL WHERE clause.

        Overrides the default linear-scan implementation with an O(1) lookup
        using a SQL WHERE clause on the content_hash column.

        Args:
            content_hash: SHA256 hash prefix (16 chars).
            entity_id: If provided, also check entity-scoped pieces.

        Returns:
            The matching piece if found, None otherwise.
        """
        if self._table is None or not content_hash:
            return None

        escaped_hash = _escape_sql(content_hash)

        # Search global pieces first
        try:
            where = f"content_hash = '{escaped_hash}' AND entity_id = '{_escape_sql(_GLOBAL_ENTITY_SENTINEL)}'"
            results = self._table.search().where(where).limit(1).to_list()
            if results:
                return _record_to_piece(results[0])
        except Exception as exc:
            logger.warning("LanceDB find_by_content_hash error (global): %s", exc)

        # Then check entity-scoped pieces if entity_id provided
        if entity_id:
            try:
                where = f"content_hash = '{escaped_hash}' AND entity_id = '{_escape_sql(entity_id)}'"
                results = self._table.search().where(where).limit(1).to_list()
                if results:
                    return _record_to_piece(results[0])
            except Exception as exc:
                logger.warning("LanceDB find_by_content_hash error (entity): %s", exc)

        return None

    def close(self):
        """Close LanceDB connection and release resources."""
        self._table = None
        self._db = None
        self._fts_index_created = False

    def _rebuild_fts_index(self):
        """Rebuild the FTS index after data modifications."""
        if self._table is None:
            return
        try:
            self._table.create_fts_index("content", replace=True)
            self._fts_index_created = True
        except Exception as exc:
            logger.warning("Failed to rebuild FTS index: %s", exc)


def _escape_sql(value):
    """Escape single quotes in a string for use in SQL WHERE clauses."""
    return value.replace("'", "''")


def _escape_sql_like(value):
    """Escape LIKE wildcards in addition to single quotes.

    Prevents unintended matching when space names contain ``%`` or ``_``.
    """
    return value.replace("'", "''").replace("%", "\\%").replace("_", "\\_")


def _build_where_clause(entity_id, knowledge_type, spaces=None):
    """Build a SQL WHERE clause string for LanceDB filtering.

    Args:
        entity_id: Entity scope filter.
        knowledge_type: Knowledge type filter.
        spaces: Optional list of space strings. When provided, generates a
            dual-strategy WHERE clause using ``primary_space IN (...)`` as the
            fast indexed path combined with ``spaces LIKE`` fallback for
            multi-space pieces.
    """
    conditions = []

    entity_value = entity_id if entity_id is not None else _GLOBAL_ENTITY_SENTINEL
    conditions.append(f"entity_id = '{_escape_sql(entity_value)}'")

    if knowledge_type is not None:
        conditions.append(
            f"knowledge_type = '{_escape_sql(knowledge_type.value)}'"
        )

    if spaces:
        in_values = ", ".join(f"'{_escape_sql(s)}'" for s in spaces)
        like_conditions = [f"spaces LIKE '%\"{_escape_sql_like(s)}\"%'" for s in spaces]
        conditions.append(
            f"(primary_space IN ({in_values}) OR {' OR '.join(like_conditions)})"
        )

    if not conditions:
        return None
    return " AND ".join(conditions)
