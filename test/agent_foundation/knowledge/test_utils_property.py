"""
Property-based tests for knowledge module utility functions.

# Feature: agent-knowledge-base, Property 25: Entity ID sanitization round-trip
# **Validates: Requirements 3.2**

Feature: knowledge-module-migration
- Property 5: Cosine similarity mathematical properties
- Property 6: Token count approximation

**Validates: Requirements 3.1, 3.2, 3.4**
"""
import math
import sys
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pytest
from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.retrieval.utils import (
    cosine_similarity,
    count_tokens,
    sanitize_id,
    unsanitize_id,
)


# Feature: agent-knowledge-base, Property 25: Entity ID sanitization round-trip
# **Validates: Requirements 3.2**


class TestSanitizeIdRoundTripProperty:
    """Property 25: Entity ID sanitization round-trip.

    For any entity_id string, unsanitize_id(sanitize_id(entity_id)) SHALL produce
    the original entity_id. Additionally, sanitize_id SHALL produce distinct outputs
    for distinct inputs (no collisions).
    """

    @given(entity_id=st.text())
    @settings(max_examples=200)
    def test_round_trip_identity(self, entity_id: str):
        """unsanitize_id(sanitize_id(entity_id)) == entity_id for all strings.

        **Validates: Requirements 3.2**
        """
        sanitized = sanitize_id(entity_id)
        restored = unsanitize_id(sanitized)
        assert restored == entity_id, (
            f"Round-trip failed: {entity_id!r} -> {sanitized!r} -> {restored!r}"
        )

    @given(data=st.data())
    @settings(max_examples=200)
    def test_injectivity_no_collisions(self, data):
        """sanitize_id SHALL produce distinct outputs for distinct inputs.

        For any two distinct entity_id strings a and b,
        sanitize_id(a) != sanitize_id(b).

        **Validates: Requirements 3.2**
        """
        a = data.draw(st.text(), label="a")
        b = data.draw(st.text(), label="b")

        if a != b:
            assert sanitize_id(a) != sanitize_id(b), (
                f"Collision: sanitize_id({a!r}) == sanitize_id({b!r}) == {sanitize_id(a)!r}"
            )


# ── Shared strategies for vector generation ──────────────────────────────────

# Use a bounded float range to avoid overflow in dot product / magnitude calculations.
# Squaring values > ~1e154 overflows to inf, producing NaN in cosine similarity.
# Squaring values < ~1e-154 underflows to 0.0, making the magnitude zero.
_bounded_float = st.floats(
    min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False
)

# Generate non-zero floats with magnitude large enough that squaring won't
# underflow to 0.0 (which would make the vector appear zero-magnitude).
# sqrt(sys.float_info.min) ≈ 1.5e-154, so 1e-150 is a safe lower bound.
_nonzero_bounded_float = (
    st.floats(min_value=1e-150, max_value=1e10, allow_nan=False, allow_infinity=False)
    | st.floats(min_value=-1e10, max_value=-1e-150, allow_nan=False, allow_infinity=False)
)

_vector_strategy = st.lists(_bounded_float, min_size=1, max_size=50)

# Build a non-zero vector by ensuring at least one element is non-zero:
# draw one guaranteed non-zero float, then fill the rest with bounded floats.
def _build_non_zero_vector(data):
    """Strategy helper: returns a list with at least one non-zero element."""
    size = data.draw(st.integers(min_value=1, max_value=50), label="vec_size")
    # Pick a random position for the guaranteed non-zero element
    nz_index = data.draw(st.integers(min_value=0, max_value=size - 1), label="nz_idx")
    elements = []
    for i in range(size):
        if i == nz_index:
            elements.append(data.draw(_nonzero_bounded_float, label=f"nz_elem"))
        else:
            elements.append(data.draw(_bounded_float, label=f"elem_{i}"))
    return elements


# Feature: knowledge-module-migration, Property 5: Cosine similarity mathematical properties


class TestCosineSimilarityMathematicalProperties:
    """Property 5: Cosine similarity mathematical properties.

    For any non-zero vector v, cosine_similarity(v, v) should equal 1.0.
    For any two vectors a and b of equal dimension, cosine_similarity(a, b)
    should equal cosine_similarity(b, a) (symmetry).
    For any two vectors of different dimensions, cosine_similarity should
    raise ValueError.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_self_similarity_equals_one(self, data):
        """cosine_similarity(v, v) == 1.0 for any non-zero vector v.

        **Validates: Requirements 3.1**
        """
        v = _build_non_zero_vector(data)
        result = cosine_similarity(v, v)
        assert math.isclose(result, 1.0, rel_tol=1e-9), (
            f"Self-similarity should be 1.0, got {result} for vector of length {len(v)}"
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_symmetry(self, data):
        """cosine_similarity(a, b) == cosine_similarity(b, a) for equal-dimension vectors.

        **Validates: Requirements 3.1**
        """
        dim = data.draw(st.integers(min_value=1, max_value=50), label="dim")
        a = data.draw(
            st.lists(_bounded_float, min_size=dim, max_size=dim), label="a"
        )
        b = data.draw(
            st.lists(_bounded_float, min_size=dim, max_size=dim), label="b"
        )

        result_ab = cosine_similarity(a, b)
        result_ba = cosine_similarity(b, a)

        assert math.isclose(result_ab, result_ba, rel_tol=1e-9, abs_tol=1e-15), (
            f"Symmetry violated: cosine_similarity(a, b)={result_ab} != "
            f"cosine_similarity(b, a)={result_ba}"
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_dimension_mismatch_raises_value_error(self, data):
        """cosine_similarity raises ValueError for vectors of different dimensions.

        **Validates: Requirements 3.2**
        """
        dim_a = data.draw(st.integers(min_value=1, max_value=50), label="dim_a")
        dim_b = data.draw(st.integers(min_value=1, max_value=50), label="dim_b")
        assume(dim_a != dim_b)

        a = data.draw(
            st.lists(_bounded_float, min_size=dim_a, max_size=dim_a), label="a"
        )
        b = data.draw(
            st.lists(_bounded_float, min_size=dim_b, max_size=dim_b), label="b"
        )

        with pytest.raises(ValueError):
            cosine_similarity(a, b)


# Feature: knowledge-module-migration, Property 6: Token count approximation


class TestTokenCountApproximation:
    """Property 6: Token count approximation.

    For any string, count_tokens(text) should equal len(text) // 4.

    **Validates: Requirements 3.4**
    """

    @given(text=st.text(max_size=10000))
    @settings(max_examples=100)
    def test_count_tokens_equals_len_div_4(self, text):
        """count_tokens(text) == len(text) // 4 for any string.

        **Validates: Requirements 3.4**
        """
        result = count_tokens(text)
        expected = len(text) // 4
        assert result == expected, (
            f"count_tokens returned {result}, expected {expected} "
            f"for text of length {len(text)}"
        )
