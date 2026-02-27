"""
Temporal decay scoring for knowledge retrieval.

Applies exponential decay to piece scores based on age, so fresher
knowledge is prioritised over stale content.  Evergreen info types
(e.g. skills, instructions) are exempt from decay.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Set

from agent_foundation.knowledge.retrieval.models.results import ScoredPiece

logger = logging.getLogger(__name__)


@dataclass
class TemporalDecayConfig:
    """Configuration for temporal decay scoring."""

    enabled: bool = True
    half_life_days: float = 30.0
    min_score_multiplier: float = 0.01
    evergreen_info_types: Set[str] = field(
        default_factory=lambda: {"skills", "instructions"}
    )


def apply_temporal_decay(
    pieces: List[ScoredPiece],
    config: TemporalDecayConfig,
) -> List[ScoredPiece]:
    """Apply exponential decay to piece scores based on age.

    Formula: ``score *= max(e^(-ln(2)/half_life * age_days), min_score_multiplier)``

    Pieces whose ``info_type`` is in ``config.evergreen_info_types`` are
    skipped (their scores remain unchanged).

    When ``config.enabled`` is False the input list is returned unchanged.

    Returns a new list sorted by descending score.
    """
    if not config.enabled:
        return pieces

    now = datetime.now(timezone.utc)
    decay_lambda = math.log(2) / config.half_life_days

    result: List[ScoredPiece] = []
    for sp in pieces:
        if sp.info_type in config.evergreen_info_types:
            result.append(sp)
            continue

        age_days = _compute_age_days(sp.updated_at, now)
        if age_days is None:
            # Cannot determine age – leave score untouched
            result.append(sp)
            continue

        multiplier = max(
            math.exp(-decay_lambda * age_days),
            config.min_score_multiplier,
        )
        sp.score = sp.score * multiplier
        result.append(sp)

    result.sort(key=lambda p: p.score, reverse=True)
    return result


def _compute_age_days(updated_at: str, now: datetime) -> float | None:
    """Parse an ISO 8601 timestamp and return age in fractional days."""
    if not updated_at:
        return None
    try:
        dt = datetime.fromisoformat(updated_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = now - dt
        return delta.total_seconds() / 86400
    except (ValueError, TypeError):
        logger.warning("Could not parse updated_at timestamp: %s", updated_at)
        return None
