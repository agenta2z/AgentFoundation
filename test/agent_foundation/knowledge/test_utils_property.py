"""
Property-based tests for knowledge module utility functions.

# Feature: agent-knowledge-base, Property 25: Entity ID sanitization round-trip
# **Validates: Requirements 3.2**
"""
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

from hypothesis import given, settings, strategies as st

from science_modeling_tools.knowledge.utils import sanitize_id, unsanitize_id


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
