"""
SpaceClassifier: Rule-based classification of knowledge into spaces.

Determines which spaces (main, personal, developmental) a knowledge unit
belongs to based on configurable rules. Supports dual-mode classification:
- "auto" rules are applied immediately to the piece's spaces field.
- "suggestion" rules store pending suggestions for human review.

Default rules:
1. Personal rule (auto, priority=10): entity_id starts with "user:" → "personal"
2. Personal content rule (auto, priority=10): info_type == "user_profile" → "personal"
3. Developmental rule (auto, priority=100, exclusive): validation_status == "failed" → "developmental"
4. Main rule (auto, priority=0): default fallback → "main"

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from rich_python_utils.service_utils.graph_service.graph_node import GraphNode

logger = logging.getLogger(__name__)


@dataclass
class SpaceRule:
    """A single classification rule.

    Attributes:
        name: Human-readable rule name.
        space: Space to assign (e.g., "personal").
        condition: Predicate on a dict representation of the knowledge unit.
        priority: Higher priority rules are evaluated first.
        exclusive: If True, this rule's space is the only one assigned.
        mode: "auto" (apply immediately) or "suggestion" (pending human review).
    """

    name: str
    space: str
    condition: Callable[[Dict[str, Any]], bool]
    priority: int = 0
    exclusive: bool = False
    mode: str = "auto"


@dataclass
class ClassificationResult:
    """Result of space classification with dual-mode support.

    Attributes:
        auto_spaces: Spaces to apply immediately.
        suggested_spaces: Spaces pending human review.
        suggestion_reasons: Reasons for each suggestion.
    """

    auto_spaces: List[str] = field(default_factory=lambda: ["main"])
    suggested_spaces: List[str] = field(default_factory=list)
    suggestion_reasons: List[str] = field(default_factory=list)


def _default_rules() -> List[SpaceRule]:
    """Create the default set of classification rules.

    Rules (evaluated highest priority first):
    1. Developmental (priority=100, exclusive): validation_status == "failed"
    2. Personal entity (priority=10): entity_id starts with "user:"
    3. Personal content (priority=10): info_type == "user_profile"
    4. Main fallback (priority=0): always matches
    """
    return [
        SpaceRule(
            name="developmental",
            space="developmental",
            condition=lambda d: d.get("validation_status") == "failed",
            priority=100,
            exclusive=True,
            mode="auto",
        ),
        SpaceRule(
            name="personal_entity",
            space="personal",
            condition=lambda d: isinstance(d.get("entity_id"), str)
            and d["entity_id"].startswith("user:"),
            priority=10,
            exclusive=False,
            mode="auto",
        ),
        SpaceRule(
            name="personal_content",
            space="personal",
            condition=lambda d: d.get("info_type") == "user_profile",
            priority=10,
            exclusive=False,
            mode="auto",
        ),
        SpaceRule(
            name="main_default",
            space="main",
            condition=lambda _: True,
            priority=0,
            exclusive=False,
            mode="auto",
        ),
    ]


@dataclass
class SpaceClassifier:
    """Rule-based classifier that determines spaces for knowledge units.

    Evaluates rules in priority order (highest first). Exclusive rules short-circuit
    and return only their space. Non-exclusive rules are additive. Auto-mode rules
    populate auto_spaces; suggestion-mode rules populate suggested_spaces with reasons.

    Attributes:
        rules: List of SpaceRule instances. Defaults to the standard rule set.
    """

    rules: List[SpaceRule] = field(default_factory=_default_rules)

    def _classify(self, data: Dict[str, Any]) -> ClassificationResult:
        """Core classification logic operating on a dict representation.

        Args:
            data: Dictionary of attributes from the knowledge unit.

        Returns:
            ClassificationResult with auto_spaces and suggested_spaces.
        """
        auto_spaces: List[str] = []
        suggested_spaces: List[str] = []
        suggestion_reasons: List[str] = []

        # Sort rules by priority descending
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            try:
                if rule.condition(data):
                    if rule.exclusive:
                        # Exclusive rule: return only this space, ignore all others
                        if rule.mode == "auto":
                            return ClassificationResult(
                                auto_spaces=[rule.space],
                                suggested_spaces=[],
                                suggestion_reasons=[],
                            )
                        else:
                            # Exclusive suggestion: suggest only this space
                            return ClassificationResult(
                                auto_spaces=["main"],
                                suggested_spaces=[rule.space],
                                suggestion_reasons=[
                                    f"Rule '{rule.name}' suggests exclusive space '{rule.space}'"
                                ],
                            )

                    if rule.mode == "auto":
                        if rule.space not in auto_spaces:
                            auto_spaces.append(rule.space)
                    else:
                        if rule.space not in suggested_spaces:
                            suggested_spaces.append(rule.space)
                            suggestion_reasons.append(
                                f"Rule '{rule.name}' suggests space '{rule.space}'"
                            )
            except Exception:
                logger.warning(
                    "SpaceClassifier rule '%s' raised an exception, skipping",
                    rule.name,
                    exc_info=True,
                )

        # Ensure auto_spaces is non-empty (default to ["main"])
        if not auto_spaces:
            auto_spaces = ["main"]

        return ClassificationResult(
            auto_spaces=auto_spaces,
            suggested_spaces=suggested_spaces,
            suggestion_reasons=suggestion_reasons,
        )

    def classify_piece(self, piece: KnowledgePiece) -> ClassificationResult:
        """Determine spaces for a knowledge piece.

        Converts the piece to a dict-like representation for rule evaluation.

        Args:
            piece: The KnowledgePiece to classify.

        Returns:
            ClassificationResult with auto and suggested spaces.
        """
        data = {
            "entity_id": piece.entity_id,
            "info_type": piece.info_type,
            "validation_status": piece.validation_status,
            "knowledge_type": piece.knowledge_type.value if piece.knowledge_type else None,
            "domain": piece.domain,
            "tags": piece.tags,
            "space": piece.space,
            "spaces": piece.spaces,
            "content": piece.content,
            "source": piece.source,
        }
        return self._classify(data)

    def classify_metadata(self, metadata: EntityMetadata) -> ClassificationResult:
        """Determine spaces for entity metadata.

        Args:
            metadata: The EntityMetadata to classify.

        Returns:
            ClassificationResult with auto and suggested spaces.
        """
        data = {
            "entity_id": metadata.entity_id,
            "entity_type": metadata.entity_type,
            "properties": metadata.properties,
            "info_type": None,
            "validation_status": None,
        }
        return self._classify(data)

    def classify_graph_node(self, node: GraphNode) -> ClassificationResult:
        """Determine spaces for a graph node.

        Args:
            node: The GraphNode to classify.

        Returns:
            ClassificationResult with auto and suggested spaces.
        """
        data = {
            "entity_id": node.node_id,
            "node_type": node.node_type,
            "properties": node.properties,
            "info_type": None,
            "validation_status": None,
        }
        return self._classify(data)
