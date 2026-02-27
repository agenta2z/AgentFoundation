"""
Property-based tests for temporal decay scoring.

Feature: knowledge-module-migration
- Property 8: Temporal decay with floor and evergreen exemption
- Property 9: Temporal decay passthrough when disabled

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**
"""
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from hypothesis import given, settings, strategies as st

from agent_foundation.knowledge.retrieval.temporal_decay import (
    apply_temporal_decay,
    TemporalDecayConfig,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece


# ── Strategies ────────────────────────────────────────────────────────────────

# Info types that are NOT evergreen (will be decayed)
_non_evergreen_info_types = st.sampled_from(
    ["context", "user_profile", "episodic", "supplementary"]
)

# Info types that ARE evergreen (exempt from decay)
_evergreen_info_types = st.sampled_from(["skills", "instructions"])


def _make_scored_piece(info_type: str, score: float, updated_at: str) -> ScoredPiece:
    """Create a ScoredPiece with the given parameters."""
    piece = KnowledgePiece(
        content=f"content_{info_type}_{score}",
        info_type=info_type,
        updated_at=updated_at,
    )
    return ScoredPiece(piece=piece, score=score)


@st.composite
def temporal_decay_config(draw):
    """Generate a valid TemporalDecayConfig with enabled=True."""
    half_life = draw(
        st.floats(min_value=1.0, max_value=365.0, allow_nan=False, allow_infinity=False)
    )
    min_mult = draw(
        st.floats(min_value=0.001, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    return TemporalDecayConfig(
        enabled=True,
        half_life_days=half_life,
        min_score_multiplier=min_mult,
        evergreen_info_types={"skills", "instructions"},
    )


@st.composite
def piece_with_known_age(draw, info_type_strategy):
    """Generate a (info_type, score, age_days) tuple for building ScoredPieces.

    Uses age_days >= 0.01 (about 15 minutes) to avoid timing race conditions
    between the test and the implementation's datetime.now() calls.
    """
    info_type = draw(info_type_strategy)
    score = draw(
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    age_days = draw(
        st.floats(min_value=0.01, max_value=365.0, allow_nan=False, allow_infinity=False)
    )
    return info_type, score, age_days


@st.composite
def mixed_decay_inputs(draw):
    """Generate a list of piece specs (mixing evergreen and non-evergreen)
    and a TemporalDecayConfig."""
    config = draw(temporal_decay_config())
    n = draw(st.integers(min_value=1, max_value=15))

    piece_specs = []
    for _ in range(n):
        is_evergreen = draw(st.booleans())
        info_st = _evergreen_info_types if is_evergreen else _non_evergreen_info_types
        spec = draw(piece_with_known_age(info_st))
        piece_specs.append(spec)

    return piece_specs, config


def _build_pieces_from_specs(piece_specs, now):
    """Build fresh ScoredPiece objects from specs using a fixed 'now' reference.

    Returns (pieces, original_scores, ages) where ages maps piece_id -> age_days.
    """
    pieces = []
    original_scores = {}
    ages = {}
    for info_type, score, age_days in piece_specs:
        updated_at = (now - timedelta(days=age_days)).isoformat()
        sp = _make_scored_piece(info_type, score, updated_at)
        original_scores[sp.piece.piece_id] = score
        ages[sp.piece.piece_id] = age_days
        pieces.append(sp)
    return pieces, original_scores, ages


# ── Property 8: Temporal decay with floor and evergreen exemption ─────────────


class TestTemporalDecayWithFloorAndEvergreenExemption:
    """Property 8: Temporal decay with floor and evergreen exemption.

    For any list of ScoredPieces with known timestamps, after applying temporal
    decay: (a) each non-evergreen piece's score should equal
    original_score * max(e^(-ln(2)/half_life * age_days), min_score_multiplier),
    (b) each evergreen piece's score should remain unchanged, and (c) the output
    should be sorted by descending score.

    **Validates: Requirements 7.1, 7.2, 7.3**
    """

    @given(data=mixed_decay_inputs())
    @settings(max_examples=100)
    def test_non_evergreen_scores_match_decay_formula(self, data):
        """Non-evergreen piece scores match the exponential decay formula with floor.

        **Validates: Requirements 7.1, 7.2**
        """
        piece_specs, config = data

        # Build fresh pieces right before calling apply_temporal_decay
        # to minimize timing drift between our 'now' and the implementation's.
        now = datetime.now(timezone.utc)
        pieces, original_scores, ages = _build_pieces_from_specs(piece_specs, now)

        result = apply_temporal_decay(pieces, config)

        decay_lambda = math.log(2) / config.half_life_days

        for sp in result:
            pid = sp.piece.piece_id
            orig = original_scores[pid]
            if sp.info_type not in config.evergreen_info_types:
                age_days = ages[pid]
                expected_mult = max(
                    math.exp(-decay_lambda * age_days),
                    config.min_score_multiplier,
                )
                expected_score = orig * expected_mult
                # Use abs_tol to handle tiny timing drift (< 1 second)
                # and rel_tol for normal floating point comparison
                assert math.isclose(sp.score, expected_score, rel_tol=1e-4, abs_tol=1e-9), (
                    f"Score mismatch for {pid}: got {sp.score}, "
                    f"expected {expected_score} (orig={orig}, age={age_days}d)"
                )

    @given(data=mixed_decay_inputs())
    @settings(max_examples=100)
    def test_evergreen_scores_unchanged(self, data):
        """Evergreen piece scores remain unchanged after temporal decay.

        **Validates: Requirements 7.3**
        """
        piece_specs, config = data

        now = datetime.now(timezone.utc)
        pieces, original_scores, _ = _build_pieces_from_specs(piece_specs, now)

        result = apply_temporal_decay(pieces, config)

        for sp in result:
            if sp.info_type in config.evergreen_info_types:
                pid = sp.piece.piece_id
                assert sp.score == original_scores[pid], (
                    f"Evergreen piece {pid} score changed: "
                    f"got {sp.score}, expected {original_scores[pid]}"
                )

    @given(data=mixed_decay_inputs())
    @settings(max_examples=100)
    def test_output_sorted_descending_by_score(self, data):
        """Output is sorted by descending score after decay.

        **Validates: Requirements 7.1**
        """
        piece_specs, config = data

        now = datetime.now(timezone.utc)
        pieces, _, _ = _build_pieces_from_specs(piece_specs, now)

        result = apply_temporal_decay(pieces, config)

        scores = [sp.score for sp in result]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending: {scores}"
        )


# ── Property 9: Temporal decay passthrough when disabled ──────────────────────


class TestTemporalDecayPassthroughWhenDisabled:
    """Property 9: Temporal decay passthrough when disabled.

    For any list of ScoredPieces, when TemporalDecayConfig.enabled is False,
    the output list should be identical to the input list.

    **Validates: Requirements 7.4**
    """

    @given(
        piece_specs=st.lists(
            piece_with_known_age(
                st.sampled_from(
                    ["context", "user_profile", "skills", "instructions", "episodic"]
                )
            ),
            min_size=0,
            max_size=15,
        ),
    )
    @settings(max_examples=100)
    def test_disabled_returns_identical_list(self, piece_specs):
        """When enabled=False, output is identical to input (same order, same scores).

        **Validates: Requirements 7.4**
        """
        now = datetime.now(timezone.utc)
        pieces = []
        for info_type, score, age_days in piece_specs:
            updated_at = (now - timedelta(days=age_days)).isoformat()
            sp = _make_scored_piece(info_type, score, updated_at)
            pieces.append(sp)

        original_scores = [sp.score for sp in pieces]
        original_ids = [sp.piece.piece_id for sp in pieces]

        config = TemporalDecayConfig(enabled=False)
        result = apply_temporal_decay(pieces, config)

        # Same length
        assert len(result) == len(pieces), (
            f"Length mismatch: got {len(result)}, expected {len(pieces)}"
        )

        # Same order and scores
        for i, (r, orig_score, orig_id) in enumerate(
            zip(result, original_scores, original_ids)
        ):
            assert r.piece.piece_id == orig_id, (
                f"Order changed at index {i}: got {r.piece.piece_id}, expected {orig_id}"
            )
            assert r.score == orig_score, (
                f"Score changed at index {i}: got {r.score}, expected {orig_score}"
            )
