"""
KnowledgeUpdater — handles knowledge updates by ID or semantic matching.

Supports two modes:
1. Exact Update: Target a specific piece_id
2. Semantic Update: Find similar pieces and update them

Design Decisions:
- LLM determines action AND input_mode; content handling depends on mode
- input_mode="content": programmatic merge/replace (no truncation)
- input_mode="instruction": second LLM call generates content from instruction
- Add new piece first, deactivate old after; rollback on failure
- Use ``X if X is not None else Y`` for empty list handling
- Compute embedding for new pieces
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional

from agent_foundation.knowledge.prompt_templates import render_prompt
from rich_python_utils.service_utils.data_operation_record import (
    DataOperationRecord,
    generate_operation_id,
)
from agent_foundation.knowledge.retrieval.models.enums import UpdateAction
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import OperationResult
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore

logger = logging.getLogger(__name__)


@dataclass
class UpdateConfig:
    """Configuration for knowledge updates."""

    similarity_threshold: float = 0.80
    max_candidates: int = 20
    max_updates: int = 3
    require_confirmation: bool = False
    preserve_history: bool = True


class KnowledgeUpdater:
    """Handles knowledge updates by ID or semantic matching.

    NOTE: This implementation assumes that piece_store.get_by_id()
    returns pieces regardless of is_active status. If your store implementation
    filters out inactive pieces, soft-delete and restore may not work correctly.
    """

    def __init__(
        self,
        piece_store: KnowledgePieceStore,
        llm_fn: Optional[Callable[[str], str]] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        config: Optional[UpdateConfig] = None,
    ):
        self.piece_store = piece_store
        self.llm_fn = llm_fn
        self.embedding_fn = embedding_fn
        self.config = config or UpdateConfig()

    # ── Exact Update (by ID) ─────────────────────────────────────────────

    def update_by_id(
        self,
        piece_id: str,
        new_content: str,
        update_instruction: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> OperationResult:
        """Update a specific piece by its ID.

        Args:
            piece_id: The ID of the piece to update.
            new_content: The new content (or content to merge).
            update_instruction: Optional instruction for LLM.
            domain: Optional new domain (None = keep existing).
            tags: Optional new tags (None = keep existing).

        Returns:
            OperationResult with success status and details.
        """
        existing = self.piece_store.get_by_id(piece_id)
        if existing is None:
            return OperationResult(
                success=False,
                operation="update",
                piece_id=piece_id,
                error=f"Piece not found: {piece_id}",
            )

        if self.llm_fn:
            analysis = self._analyze_update_intent(
                existing, new_content, update_instruction
            )

            if analysis["action"] == UpdateAction.NO_CHANGE:
                return OperationResult(
                    success=False,
                    operation="update",
                    piece_id=piece_id,
                    old_version=existing.version,
                    new_version=existing.version,
                    details={"reason": analysis["reasoning"]},
                )

            action = analysis["action"]

            # Instruction mode: LLM generates content from instruction
            generated_content = None
            if analysis.get("input_mode") == "instruction" and update_instruction:
                generated_content = self._generate_updated_content(
                    existing.content, update_instruction
                )
                if generated_content is None:
                    return OperationResult(
                        success=False,
                        operation="update",
                        piece_id=piece_id,
                        error="Instruction-mode content generation failed",
                    )

            if generated_content is not None:
                # Instruction mode: LLM already produced the complete final
                # content. Bypass _compute_final_content to avoid double-merging.
                final_content = generated_content
            else:
                # Content mode: programmatic merge/replace as before
                final_content = self._compute_final_content(
                    existing.content, new_content, action, analysis
                )

            final_domain = (
                analysis.get("updated_domain")
                if analysis.get("updated_domain") is not None
                else domain
            )
            if analysis.get("clear_tags"):
                final_tags = []
            elif analysis.get("updated_tags") is not None:
                final_tags = analysis["updated_tags"]
            else:
                final_tags = tags
            summary = analysis.get("changes_summary", "")
        else:
            action = UpdateAction.REPLACE
            final_content = new_content
            final_domain = domain
            final_tags = tags
            summary = "Direct replacement (no LLM analysis)"

        return self._apply_update(
            existing=existing,
            action=action,
            new_content=final_content,
            new_domain=final_domain if final_domain is not None else None,
            new_tags=final_tags if final_tags is not None else None,
            summary=summary,
        )

    # ── Semantic Update (free-text) ──────────────────────────────────────

    def update_by_content(
        self,
        new_content: str,
        update_instruction: Optional[str] = None,
        entity_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[OperationResult]:
        """Find and update pieces similar to the provided content.

        WARNING: This may update MULTIPLE pieces. Use max_updates config to limit.

        Args:
            new_content: The new/updated content.
            update_instruction: Optional instruction for LLM.
            entity_id: Optional entity scope for search.
            domain: Optional domain filter.

        Returns:
            List of OperationResult for each piece that was updated.
        """
        candidates = self.piece_store.search(
            query=new_content,
            entity_id=entity_id,
            top_k=self.config.max_candidates,
        )

        # Filter by threshold and active status
        matches = [
            (piece, score)
            for piece, score in candidates
            if score >= self.config.similarity_threshold
            and getattr(piece, "is_active", True)
        ]

        if not matches:
            logger.info("No similar pieces found for semantic update")
            return []

        if domain:
            matches = [
                (p, s)
                for p, s in matches
                if p.domain == domain
                or domain in getattr(p, "secondary_domains", [])
            ]

        results = []
        for piece, score in matches[: self.config.max_updates]:
            if self.llm_fn:
                analysis = self._analyze_update_intent(
                    piece, new_content, update_instruction
                )

                if analysis["action"] == UpdateAction.NO_CHANGE:
                    continue

                action = analysis["action"]

                # Instruction mode: LLM generates content from instruction
                generated_content = None
                if analysis.get("input_mode") == "instruction" and update_instruction:
                    generated_content = self._generate_updated_content(
                        piece.content, update_instruction
                    )
                    if generated_content is None:
                        results.append(OperationResult(
                            success=False,
                            operation="update",
                            piece_id=piece.piece_id,
                            error="Instruction-mode content generation failed",
                        ))
                        continue

                if generated_content is not None:
                    # Instruction mode: LLM already produced the complete final
                    # content. Bypass _compute_final_content to avoid double-merging.
                    final_content = generated_content
                else:
                    # Content mode: programmatic merge/replace as before
                    final_content = self._compute_final_content(
                        piece.content, new_content, action, analysis
                    )

                final_domain = analysis.get("updated_domain")
                if analysis.get("clear_tags"):
                    final_tags = []
                else:
                    final_tags = analysis.get("updated_tags")
                summary = analysis.get("changes_summary", "")

                result = self._apply_update(
                    existing=piece,
                    action=action,
                    new_content=final_content,
                    new_domain=final_domain,
                    new_tags=final_tags,
                    summary=summary,
                )
            else:
                result = self._apply_update(
                    existing=piece,
                    action=UpdateAction.MERGE,
                    new_content=f"{piece.content}\n\n---\n\n{new_content}",
                    new_domain=None,
                    new_tags=None,
                    summary=f"Merged with similar piece (score: {score:.3f})",
                )

            results.append(result)

        return results

    # ── Internal Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """Strip markdown code fences from LLM JSON responses."""
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = stripped.index("\n") if "\n" in stripped else len(stripped)
            stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()

    def _compute_final_content(
        self,
        existing_content: str,
        new_content: str,
        action: UpdateAction,
        analysis: dict,
    ) -> str:
        """Compute final content based on action. Done programmatically to
        avoid truncation."""
        if action == UpdateAction.REPLACE:
            return new_content

        if action == UpdateAction.MERGE:
            merge_strategy = analysis.get("merge_strategy", "append")
            if merge_strategy == "prepend":
                return f"{new_content}\n\n{existing_content}"
            elif merge_strategy == "interleave":
                existing_paras = existing_content.split("\n\n")
                new_paras = new_content.split("\n\n")
                interleaved = []
                for i in range(max(len(existing_paras), len(new_paras))):
                    if i < len(existing_paras):
                        interleaved.append(existing_paras[i])
                    if i < len(new_paras):
                        interleaved.append(new_paras[i])
                return "\n\n".join(interleaved)
            else:  # append (default)
                return f"{existing_content}\n\n{new_content}"

        return existing_content  # NO_CHANGE

    def _analyze_update_intent(
        self,
        existing: KnowledgePiece,
        new_content: str,
        update_instruction: Optional[str] = None,
    ) -> dict:
        """Use LLM to analyze update intent. Returns dict with action and
        metadata."""
        instruction_text = (
            f"User instruction: {update_instruction}" if update_instruction else ""
        )

        prompt = render_prompt(
            "quality/UpdateIntent",
            existing_id=existing.piece_id,
            existing_content=existing.content[:2000],
            existing_length=str(len(existing.content)),
            existing_domain=existing.domain,
            existing_tags=", ".join(existing.tags),
            existing_updated_at=existing.updated_at or "unknown",
            new_content=new_content[:2000],
            new_length=str(len(new_content)),
            update_instruction=instruction_text,
        )

        try:
            response = self.llm_fn(prompt)
            parsed = json.loads(self._strip_json_fences(response))

            try:
                action = UpdateAction(parsed.get("action", "no_change").lower())
            except ValueError:
                logger.warning(
                    "Invalid action from LLM: %s. Defaulting to NO_CHANGE.",
                    parsed.get("action"),
                )
                action = UpdateAction.NO_CHANGE

            return {
                "action": action,
                "input_mode": parsed.get("input_mode", "content"),
                "merge_strategy": parsed.get("merge_strategy", "append"),
                "updated_domain": parsed.get("updated_domain"),
                "updated_tags": parsed.get("updated_tags"),
                "clear_tags": parsed.get("clear_tags", False),
                "changes_summary": parsed.get("changes_summary", ""),
                "reasoning": parsed.get("reasoning", ""),
                "confidence": parsed.get("confidence", 0.0),
            }

        except Exception as e:
            logger.warning(
                "LLM update analysis failed: %s. Defaulting to NO_CHANGE.", e
            )
            return {
                "action": UpdateAction.NO_CHANGE,
                "reasoning": f"LLM error: {e}",
                "changes_summary": "",
            }

    def _generate_updated_content(
        self,
        existing_content: str,
        update_instruction: str,
    ) -> Optional[str]:
        """Use LLM to generate updated content from an instruction.

        Called only in instruction mode. The LLM applies the user's instruction
        to the existing content and returns the complete final content.

        Args:
            existing_content: The full existing piece content (not truncated).
            update_instruction: The user's natural language instruction.

        Returns:
            The generated content string, or None if generation failed.
        """
        prompt = render_prompt(
            "quality/UpdateContentGeneration",
            existing_content=existing_content,
            update_instruction=update_instruction,
        )

        try:
            response = self.llm_fn(prompt)
            parsed = json.loads(self._strip_json_fences(response))
            generated = parsed.get("generated_content")
            if not generated:
                logger.warning("LLM returned empty generated_content")
                return None
            return generated
        except Exception as e:
            logger.warning("LLM content generation failed: %s", e)
            return None

    def _apply_update(
        self,
        existing: KnowledgePiece,
        action: UpdateAction,
        new_content: str,
        new_domain: Optional[str],
        new_tags: Optional[List[str]],
        summary: str,
    ) -> OperationResult:
        """Apply the update to the piece store.

        Atomicity: Add new piece first, deactivate old after.
        Rollback on failure. Appends DataOperationRecord history.
        """
        old_version = existing.version
        now = datetime.now(timezone.utc).isoformat()
        op_id = generate_operation_id("KnowledgeUpdater", summary[:20])

        if self.config.preserve_history:
            # Compute embedding for new piece
            new_embedding = None
            if self.embedding_fn:
                try:
                    embedding_text = new_content[:2000]
                    new_embedding = self.embedding_fn(embedding_text)
                except Exception as e:
                    logger.warning("Failed to compute embedding: %s", e)

            # Create new piece with supersedes link
            new_piece = KnowledgePiece(
                content=new_content,
                domain=new_domain if new_domain is not None else existing.domain,
                tags=new_tags if new_tags is not None else existing.tags,
                secondary_domains=existing.secondary_domains,
                custom_tags=existing.custom_tags,
                knowledge_type=existing.knowledge_type,
                info_type=existing.info_type,
                entity_id=existing.entity_id,
                source=existing.source,
                space=existing.space,
                spaces=list(existing.spaces),
                supersedes=existing.piece_id,
                version=old_version + 1,
                embedding=new_embedding,
                embedding_text=new_content[:2000],
            )

            # ADD record on new piece
            new_piece.history.append(DataOperationRecord(
                operation="add",
                timestamp=now,
                operation_id=op_id,
                reason=summary,
                source="KnowledgeUpdater",
                details={
                    "action": action.value,
                    "supersedes": existing.piece_id,
                },
            ))

            try:
                # Add new piece FIRST
                self.piece_store.add(new_piece)
                new_piece_id = new_piece.piece_id

                # Deactivate old piece AFTER new is added
                existing.is_active = False
                existing.updated_at = now
                existing.history.append(DataOperationRecord(
                    operation="delete",
                    timestamp=now,
                    operation_id=op_id,
                    reason=summary,
                    source="KnowledgeUpdater",
                    details={
                        "delete_mode": "soft",
                        "superseded_by": new_piece_id,
                    },
                ))
                self.piece_store.update(existing)

            except Exception as e:
                # Rollback — remove the new piece if it was added
                logger.error("Update failed: %s. Rolling back.", e)
                try:
                    self.piece_store.remove(new_piece.piece_id)
                except Exception:
                    pass  # Best effort rollback
                return OperationResult(
                    success=False,
                    operation="update",
                    piece_id=existing.piece_id,
                    error=f"Update failed: {e}",
                )
        else:
            # In-place update — capture content_before
            content_before = existing.content
            existing.content = new_content

            # Recompute content_hash for dedup consistency
            existing.content_hash = existing._compute_content_hash()

            if new_domain is not None:
                existing.domain = new_domain
            if new_tags is not None:
                existing.tags = new_tags
            existing.version = old_version + 1
            existing.updated_at = now

            # Update embedding
            if self.embedding_fn:
                try:
                    existing.embedding = self.embedding_fn(new_content[:2000])
                    existing.embedding_text = new_content[:2000]
                except Exception as e:
                    logger.warning("Failed to update embedding: %s", e)

            # UPDATE record with content_before/content_after
            existing.history.append(DataOperationRecord(
                operation="update",
                timestamp=now,
                operation_id=op_id,
                reason=summary,
                source="KnowledgeUpdater",
                content_before=content_before,
                content_after=new_content,
                details={"action": action.value},
            ))

            self.piece_store.update(existing)
            new_piece_id = existing.piece_id

        return OperationResult(
            success=True,
            operation="update",
            piece_id=new_piece_id,
            old_version=old_version,
            new_version=old_version + 1,
            details={
                "action": action.value,
                "summary": summary,
                "operation_id": op_id,
            },
        )
