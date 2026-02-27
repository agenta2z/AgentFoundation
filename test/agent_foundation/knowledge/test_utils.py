"""
Unit tests for knowledge module utility functions: sanitize_id, unsanitize_id, parse_entity_type.
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

import pytest
from agent_foundation.knowledge.retrieval.utils import sanitize_id, unsanitize_id, parse_entity_type


# ── sanitize_id ──────────────────────────────────────────────────────────────

class TestSanitizeId:
    """Tests for sanitize_id percent-encoding."""

    def test_encodes_colon(self):
        assert sanitize_id("user:xinli") == "user%3Axinli"

    def test_encodes_forward_slash(self):
        assert sanitize_id("path/to/thing") == "path%2Fto%2Fthing"

    def test_encodes_backslash(self):
        assert sanitize_id("back\\slash") == "back%5Cslash"

    def test_encodes_percent_first(self):
        """Percent must be encoded before other characters to avoid double-encoding."""
        assert sanitize_id("100%") == "100%25"

    def test_encodes_percent_before_colon(self):
        """Ensure '%' is encoded before ':' so '%3A' in input doesn't get mangled."""
        assert sanitize_id("%3A") == "%253A"

    def test_multiple_special_chars(self):
        assert sanitize_id("a:b/c\\d%e") == "a%3Ab%2Fc%5Cd%25e"

    def test_no_special_chars(self):
        assert sanitize_id("simple_id") == "simple_id"

    def test_empty_string(self):
        assert sanitize_id("") == ""

    def test_only_colons(self):
        assert sanitize_id(":::") == "%3A%3A%3A"

    def test_entity_id_with_type(self):
        assert sanitize_id("store:costco") == "store%3Acostco"


# ── unsanitize_id ────────────────────────────────────────────────────────────

class TestUnsanitizeId:
    """Tests for unsanitize_id percent-decoding."""

    def test_decodes_colon(self):
        assert unsanitize_id("user%3Axinli") == "user:xinli"

    def test_decodes_forward_slash(self):
        assert unsanitize_id("path%2Fto%2Fthing") == "path/to/thing"

    def test_decodes_backslash(self):
        assert unsanitize_id("back%5Cslash") == "back\\slash"

    def test_decodes_percent_last(self):
        """Percent must be decoded last to avoid premature decoding."""
        assert unsanitize_id("100%25") == "100%"

    def test_decodes_encoded_percent_colon(self):
        """'%253A' should decode to '%3A', not to ':'."""
        assert unsanitize_id("%253A") == "%3A"

    def test_multiple_special_chars(self):
        assert unsanitize_id("a%3Ab%2Fc%5Cd%25e") == "a:b/c\\d%e"

    def test_no_encoded_chars(self):
        assert unsanitize_id("simple_id") == "simple_id"

    def test_empty_string(self):
        assert unsanitize_id("") == ""


# ── sanitize/unsanitize round-trip ───────────────────────────────────────────

class TestSanitizeRoundTrip:
    """Tests that sanitize_id and unsanitize_id are inverses."""

    def test_round_trip_with_colon(self):
        original = "user:xinli"
        assert unsanitize_id(sanitize_id(original)) == original

    def test_round_trip_with_slash(self):
        original = "path/to/thing"
        assert unsanitize_id(sanitize_id(original)) == original

    def test_round_trip_with_backslash(self):
        original = "back\\slash"
        assert unsanitize_id(sanitize_id(original)) == original

    def test_round_trip_with_percent(self):
        original = "100%"
        assert unsanitize_id(sanitize_id(original)) == original

    def test_round_trip_with_all_special(self):
        original = "a:b/c\\d%e"
        assert unsanitize_id(sanitize_id(original)) == original

    def test_round_trip_plain_string(self):
        original = "simple_id_123"
        assert unsanitize_id(sanitize_id(original)) == original

    def test_round_trip_empty(self):
        assert unsanitize_id(sanitize_id("")) == ""

    def test_round_trip_already_looks_encoded(self):
        """Input that looks like it's already encoded should still round-trip."""
        original = "%3A"
        assert unsanitize_id(sanitize_id(original)) == original


# ── parse_entity_type ────────────────────────────────────────────────────────

class TestParseEntityType:
    """Tests for parse_entity_type extraction."""

    def test_user_type(self):
        assert parse_entity_type("user:xinli") == "user"

    def test_store_type(self):
        assert parse_entity_type("store:costco") == "store"

    def test_app_type(self):
        assert parse_entity_type("app:grocery_checker") == "app"

    def test_no_colon_returns_default(self):
        assert parse_entity_type("plain_id") == "default"

    def test_empty_string_returns_default(self):
        assert parse_entity_type("") == "default"

    def test_multiple_colons_returns_first_part(self):
        """Only the first colon is used as the separator."""
        assert parse_entity_type("a:b:c") == "a"

    def test_colon_at_start(self):
        """Leading colon means empty type string."""
        assert parse_entity_type(":name") == ""

    def test_colon_at_end(self):
        assert parse_entity_type("type:") == "type"
