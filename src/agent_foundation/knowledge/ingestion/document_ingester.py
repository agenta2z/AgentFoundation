"""
DocumentIngester — end-to-end pipeline for ingesting long-form documents.

This module provides the main ingestion pipeline that:
1. Chunks long documents into manageable pieces
2. Sends each chunk to an LLM for structuring with domain classification
3. Validates and merges the results
4. Loads into a KnowledgeBase

The pipeline handles documents of any length by leveraging the MarkdownChunker
and the new domain taxonomy-aware structuring prompts.

Progress Callbacks:
    The ingester supports optional progress callbacks for real-time UI updates.
    Pass a ProgressCallback function to receive status messages during ingestion.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_foundation.knowledge.ingestion.chunker import (
    ChunkerConfig,
    DocumentChunk,
    MarkdownChunker,
)
from agent_foundation.knowledge.ingestion.debug_session import IngestionDebugSession
from agent_foundation.knowledge.ingestion.prompts.structuring_prompt import (
    get_structuring_prompt,
)
from agent_foundation.knowledge.retrieval.data_loader import KnowledgeDataLoader
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.ingestion.deduplicator import (
    DedupConfig,
    ThreeTierDeduplicator,
)
from agent_foundation.knowledge.ingestion.merge_strategy import (
    MergeStrategyConfig,
    MergeStrategyManager,
)
from agent_foundation.knowledge.ingestion.validator import (
    KnowledgeValidator,
    ValidationConfig,
)
from agent_foundation.knowledge.retrieval.models.enums import (
    DedupAction,
    MergeAction,
    MergeType,
)
from agent_foundation.knowledge.retrieval.models.results import MergeCandidate

logger = logging.getLogger(__name__)

# Type alias for progress callback function
ProgressCallback = Callable[[str], None]


def _noop_progress(message: str) -> None:
    """Default no-op progress callback."""
    pass


@dataclass
class IngestionResult:
    """Result of a document ingestion operation.

    Attributes:
        success: Whether the ingestion succeeded.
        chunks_processed: Number of chunks processed.
        pieces_created: Number of knowledge pieces created.
        metadata_created: Number of metadata entries created.
        graph_nodes_created: Number of graph nodes created.
        graph_edges_created: Number of graph edges created.
        errors: List of error messages from failed chunks.
        source_file: Source file path if applicable.
    """

    success: bool
    chunks_processed: int = 0
    pieces_created: int = 0
    metadata_created: int = 0
    graph_nodes_created: int = 0
    graph_edges_created: int = 0
    errors: List[str] = None
    source_file: Optional[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class IngesterConfig:
    """Configuration for the DocumentIngester.

    Attributes:
        max_retries: Maximum LLM call retries per chunk (default 3).
        chunker_config: Configuration for the MarkdownChunker.
        full_schema: Whether to use full schema output (default True).
        merge_graphs: Whether to merge graphs across chunks (default True).
        dedupe_pieces: Whether to deduplicate pieces by content hash (default True).
        debug_session: Optional debug session for saving artifacts.
    """

    max_retries: int = 3
    chunker_config: Optional[ChunkerConfig] = None
    full_schema: bool = True
    merge_graphs: bool = True
    dedupe_pieces: bool = True
    debug_session: Optional[IngestionDebugSession] = None


class DocumentIngester:
    """End-to-end pipeline for ingesting long-form documents into a KnowledgeBase.

    This class orchestrates the full ingestion workflow:
    1. Chunking: Split long documents using MarkdownChunker
    2. Structuring: Send each chunk to LLM with domain taxonomy prompt
    3. Validation: Validate LLM output against schema
    4. Merging: Combine results from all chunks
    5. Loading: Load into KnowledgeBase via KnowledgeDataLoader

    Supports optional progress callbacks for real-time UI updates during
    long-running ingestion operations.
    """

    def __init__(
        self,
        inferencer: Callable[[str], Any],
        config: Optional[IngesterConfig] = None,
        on_progress: Optional[ProgressCallback] = None,
        debug_session: Optional[IngestionDebugSession] = None,
        deduplicator: Optional[ThreeTierDeduplicator] = None,
        merge_manager: Optional[MergeStrategyManager] = None,
        validator: Optional[KnowledgeValidator] = None,
    ):
        """Initialize the DocumentIngester.

        Args:
            inferencer: Any callable implementing the inferencer protocol
                (prompt -> response string or InferencerResponse).
            config: Optional IngesterConfig. Uses defaults if None.
            on_progress: Optional callback for progress updates.
            debug_session: Optional debug session for saving artifacts.
                Takes precedence over config.debug_session if both provided.
            deduplicator: Optional ThreeTierDeduplicator for dedup.
            merge_manager: Optional MergeStrategyManager for merge handling.
            validator: Optional KnowledgeValidator for validation.
        """
        self.inferencer = inferencer
        self.config = config or IngesterConfig()
        self.chunker = MarkdownChunker(self.config.chunker_config)
        self.on_progress = on_progress or _noop_progress
        self.debug_session = debug_session or self.config.debug_session

        self._deduplicator = deduplicator
        self._merge_manager = merge_manager
        self._validator = validator

    def _report(self, message: str) -> None:
        """Report progress to the callback and logger."""
        logger.info(message)
        self.on_progress(message)

    def ingest_file(
        self,
        file_path: str,
        kb: KnowledgeBase,
    ) -> IngestionResult:
        """Ingest a file into the KnowledgeBase.

        Args:
            file_path: Path to the file to ingest.
            kb: KnowledgeBase to populate.

        Returns:
            IngestionResult with counts and status.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        return self.ingest_text(content, kb, source_file=file_path)

    def ingest_text(
        self,
        text: str,
        kb: KnowledgeBase,
        source_file: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest text content into the KnowledgeBase.

        Args:
            text: The text content to ingest.
            kb: KnowledgeBase to populate.
            source_file: Optional source file path for metadata.

        Returns:
            IngestionResult with counts and status.
        """
        result = IngestionResult(success=True, source_file=source_file)
        doc_id = source_file or f"doc_{id(text) % 100000}"

        # Step 1: Chunk the document
        self._report("Analyzing document structure...")
        chunks = self.chunker.chunk_document(text, source_file=source_file)
        if not chunks:
            self._report("No content to process")
            return result

        self._report(f"Split into {len(chunks)} chunk(s)")

        if self.debug_session:
            for chunk in chunks:
                self.debug_session.save_chunk(
                    doc_id=doc_id,
                    chunk_idx=chunk.chunk_index,
                    content=chunk.content,
                    metadata={
                        "header_context": chunk.header_context,
                        "total_chunks": chunk.total_chunks,
                        "source_file": chunk.source_file,
                    },
                )

        # Step 2: Process each chunk through LLM
        all_structured_data: List[Dict[str, Any]] = []

        for chunk in chunks:
            try:
                self._report(
                    f"Processing chunk {chunk.chunk_index + 1}/{chunk.total_chunks}..."
                )
                structured = self._process_chunk(chunk, doc_id=doc_id)
                if structured:
                    pieces = structured.get("pieces", [])
                    if pieces:
                        domains = set(p.get("domain", "general") for p in pieces)
                        self._report(
                            f"  → {len(pieces)} piece(s): {', '.join(sorted(domains))}"
                        )
                    all_structured_data.append(structured)
                    result.chunks_processed += 1
            except Exception as e:
                error_msg = f"Chunk {chunk.chunk_index}: {e}"
                self._report(f"  ✗ Failed: {e}")
                result.errors.append(error_msg)

        if not all_structured_data:
            result.success = False
            self._report("No data extracted from any chunk")
            return result

        # Step 3: Merge results from all chunks
        self._report("Merging results...")
        merged = self._merge_results(all_structured_data)

        if self.debug_session:
            self.debug_session.save_merged(doc_id=doc_id, data=merged)

        # Step 4: Add source metadata to pieces
        if source_file:
            for piece in merged.get("pieces", []):
                piece["source"] = source_file

        # Step 4.5: Apply enhancements (dedup, validation, merge)
        merged, enhancement_counts, pieces_to_deactivate = (
            self._apply_enhancements(merged)
        )

        if enhancement_counts.get("updated"):
            logger.info("Updated %d pieces", enhancement_counts["updated"])
        if enhancement_counts.get("merged"):
            logger.info("Merged %d pieces", enhancement_counts["merged"])

        # Step 5: Load into KnowledgeBase (atomic)
        try:
            self._report("Saving to knowledge base...")
            counts = self._load_into_kb(merged, kb, pieces_to_deactivate)
            result.pieces_created = counts.get("pieces", 0)
            result.metadata_created = counts.get("metadata", 0)
            result.graph_nodes_created = counts.get("graph_nodes", 0)
            result.graph_edges_created = counts.get("graph_edges", 0)
            self._report(
                f"Done: {result.pieces_created} pieces, "
                f"{result.graph_nodes_created} nodes"
            )
        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to load into KB: {e}")
            self._report(f"✗ Save failed: {e}")

        return result

    def _process_chunk(
        self, chunk: DocumentChunk, doc_id: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Process a single chunk through the LLM."""
        context = ""
        if chunk.header_context:
            context = f"Section context: {chunk.header_context}"
        if chunk.total_chunks > 1:
            context += f"\n(Chunk {chunk.chunk_index + 1} of {chunk.total_chunks})"

        prompt = get_structuring_prompt(
            user_input=chunk.content,
            context=context,
            full_schema=self.config.full_schema,
        )

        if self.debug_session:
            self.debug_session.save_prompt(
                doc_id=doc_id,
                chunk_idx=chunk.chunk_index,
                prompt=prompt,
            )

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._call_llm(prompt)

                if self.debug_session:
                    self.debug_session.save_response(
                        doc_id=doc_id,
                        chunk_idx=chunk.chunk_index,
                        response=response,
                    )

                structured = self._parse_and_validate(response)

                if self.debug_session:
                    self.debug_session.save_structured(
                        doc_id=doc_id,
                        chunk_idx=chunk.chunk_index,
                        data=structured,
                    )

                return structured
            except Exception as e:
                last_error = e
                logger.debug(
                    "Chunk %d attempt %d failed: %s",
                    chunk.chunk_index,
                    attempt + 1,
                    e,
                )

        raise ValueError(
            f"Failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM inferencer with duck-typed response handling."""
        response = self.inferencer(prompt)
        if hasattr(response, "select_response"):
            return response.select_response().response
        return str(response)

    def _parse_and_validate(self, llm_response: str) -> Dict[str, Any]:
        """Parse and validate the LLM response JSON."""
        text = llm_response.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

        if not isinstance(data, dict):
            raise ValueError("Response must be a JSON object")

        if self.config.full_schema:
            if "metadata" not in data:
                data["metadata"] = {}
            if "pieces" not in data:
                data["pieces"] = []
            if "graph" not in data:
                data["graph"] = {"nodes": [], "edges": []}

        required_piece_fields = ["piece_id", "content", "knowledge_type", "info_type"]
        for i, piece in enumerate(data.get("pieces", [])):
            missing = [f for f in required_piece_fields if f not in piece]
            if missing:
                raise ValueError(
                    f"Piece {i} missing required fields: {', '.join(missing)}"
                )
            kt = piece.get("knowledge_type", "fact")
            try:
                KnowledgeType(kt)
            except ValueError:
                raise ValueError(f"Piece {i} has invalid knowledge_type: {kt}")

        return data

    def _merge_results(self, all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge structured data from multiple chunks."""
        merged: Dict[str, Any] = {
            "metadata": {},
            "pieces": [],
            "graph": {"nodes": [], "edges": []},
        }

        seen_piece_ids: set = set()
        seen_node_ids: set = set()
        seen_edges: set = set()

        for data in all_data:
            merged["metadata"].update(data.get("metadata", {}))

            for piece in data.get("pieces", []):
                piece_id = piece.get("piece_id", "")
                if self.config.dedupe_pieces and piece_id in seen_piece_ids:
                    continue
                seen_piece_ids.add(piece_id)
                merged["pieces"].append(piece)

            if self.config.merge_graphs:
                graph = data.get("graph", {})
                for node in graph.get("nodes", []):
                    node_id = node.get("node_id", "")
                    if node_id not in seen_node_ids:
                        seen_node_ids.add(node_id)
                        merged["graph"]["nodes"].append(node)

                for edge in graph.get("edges", []):
                    edge_key = (
                        edge.get("source_id", ""),
                        edge.get("target_id", ""),
                        edge.get("edge_type", ""),
                    )
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        merged["graph"]["edges"].append(edge)

        return merged

    def _apply_enhancements(
        self,
        data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, int], List[str]]:
        """Apply deduplication and validation to pieces before loading.

        Returns:
            Tuple of (modified data dict, enhancement counts dict,
            list of piece_ids to deactivate).
        """
        counts = {"deduped": 0, "failed_validation": 0, "updated": 0, "merged": 0}
        pieces_to_deactivate: List[str] = []

        if not (self._deduplicator or self._validator):
            return data, counts, []

        enhanced_pieces = []

        for piece_dict in data.get("pieces", []):
            piece = KnowledgePiece.from_dict(piece_dict)

            # Validation: failed pieces go to developmental space (Req 21.4)
            if self._validator:
                val_result = self._validator.validate(piece)
                if not val_result.is_valid:
                    piece.validation_status = "failed"
                    piece.validation_issues = val_result.issues
                    piece.space = "developmental"
                    counts["failed_validation"] += 1

            # Deduplication
            if self._deduplicator:
                dedup_result = self._deduplicator.deduplicate(piece)

                if dedup_result.action == DedupAction.NO_OP:
                    counts["deduped"] += 1
                    continue

                elif dedup_result.action == DedupAction.UPDATE:
                    if dedup_result.existing_piece_id:
                        piece.supersedes = dedup_result.existing_piece_id
                        pieces_to_deactivate.append(
                            dedup_result.existing_piece_id
                        )
                        piece.version = 2
                        counts["updated"] += 1

                elif dedup_result.action == DedupAction.MERGE:
                    if self._merge_manager and dedup_result.existing_piece_id:
                        existing_id: str = dedup_result.existing_piece_id

                        candidate = MergeCandidate(
                            piece_id=existing_id,
                            similarity=dedup_result.similarity_score or 0.9,
                            merge_type=MergeType.OVERLAPPING,
                            reason=dedup_result.reason or "Dedup suggested merge",
                        )
                        merge_result = self._merge_manager.apply_strategy(
                            piece, [candidate]
                        )

                        if merge_result.action == MergeAction.MERGED:
                            counts["merged"] += 1
                            continue

                        elif merge_result.action in (
                            MergeAction.PENDING_REVIEW,
                            MergeAction.DEFERRED,
                        ):
                            piece = KnowledgePiece.from_dict(piece.to_dict())

            enhanced_pieces.append(piece.to_dict())

        data["pieces"] = enhanced_pieces
        return data, counts, pieces_to_deactivate

    def _load_into_kb(
        self,
        data: Dict[str, Any],
        kb: KnowledgeBase,
        pieces_to_deactivate: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Load structured data into the KnowledgeBase.

        Atomic loading: loads new pieces first, then deactivates old ones.
        """
        pieces_to_deactivate = pieces_to_deactivate or []

        # Cache existing pieces and build version map
        version_map: Dict[str, int] = {}
        existing_pieces: Dict[str, KnowledgePiece] = {}
        for piece_id in pieces_to_deactivate:
            existing = kb.piece_store.get_by_id(piece_id)
            if existing:
                version_map[piece_id] = existing.version
                existing_pieces[piece_id] = existing

        # Correct version numbers for pieces that supersede others
        for piece_dict in data.get("pieces", []):
            supersedes_id = piece_dict.get("supersedes")
            if supersedes_id and supersedes_id in version_map:
                piece_dict["version"] = version_map[supersedes_id] + 1

        # Load new pieces FIRST (atomicity - Req 21.3)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            counts = KnowledgeDataLoader.load(kb, temp_path)
        finally:
            os.unlink(temp_path)

        # Deactivate old pieces AFTER loading new ones (atomicity - Req 21.3)
        for piece_id in pieces_to_deactivate:
            try:
                existing = existing_pieces.get(piece_id)
                if existing:
                    existing.is_active = False
                    existing.updated_at = (
                        datetime.now(timezone.utc).isoformat()
                    )
                    kb.piece_store.update(existing)
            except Exception as e:
                logger.warning(
                    "Failed to deactivate piece %s: %s", piece_id, e
                )

        return counts


def ingest_markdown_files(
    file_paths: List[str],
    kb: KnowledgeBase,
    inferencer: Callable[[str], Any],
    config: Optional[IngesterConfig] = None,
) -> Dict[str, IngestionResult]:
    """Convenience function to ingest multiple markdown files."""
    ingester = DocumentIngester(inferencer, config)
    results: Dict[str, IngestionResult] = {}

    for path in file_paths:
        try:
            result = ingester.ingest_file(path, kb)
            results[path] = result
            logger.info(
                "Ingested %s: %d pieces, %d errors",
                path,
                result.pieces_created,
                len(result.errors),
            )
        except Exception as e:
            results[path] = IngestionResult(
                success=False,
                errors=[str(e)],
                source_file=path,
            )
            logger.error("Failed to ingest %s: %s", path, e)

    return results


def ingest_directory(
    directory: str,
    kb: KnowledgeBase,
    inferencer: Callable[[str], Any],
    pattern: str = "*.md",
    config: Optional[IngesterConfig] = None,
) -> Dict[str, IngestionResult]:
    """Convenience function to ingest all matching files in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    file_paths = [str(p) for p in dir_path.glob(pattern)]
    if not file_paths:
        logger.warning("No files matching '%s' found in %s", pattern, directory)
        return {}

    return ingest_markdown_files(file_paths, kb, inferencer, config)
