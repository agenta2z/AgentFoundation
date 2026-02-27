"""
Property-based tests for domain taxonomy module.

Feature: knowledge-module-migration
- Property 15: Taxonomy domain validation
- Property 16: Taxonomy prompt formatting completeness

**Validates: Requirements 11.3, 11.4**
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
from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.ingestion.taxonomy import (
    DOMAIN_TAXONOMY,
    format_taxonomy_for_prompt,
    get_all_domains,
    get_domain_tags,
    validate_domain,
)


# ── Strategies ────────────────────────────────────────────────────────────────

# Strategy that generates strings guaranteed NOT to be valid domain keys
_valid_domains = list(DOMAIN_TAXONOMY.keys())

_invalid_domain_strategy = st.text(min_size=0, max_size=100).filter(
    lambda s: s not in _valid_domains
)

_valid_domain_strategy = st.sampled_from(_valid_domains)


# Feature: knowledge-module-migration, Property 15: Taxonomy domain validation


class TestTaxonomyDomainValidation:
    """Property 15: Taxonomy domain validation.

    For any string not in the DOMAIN_TAXONOMY keys, validate_domain() should
    return False, and get_domain_tags() should raise ValueError. For any string
    in the DOMAIN_TAXONOMY keys, validate_domain() should return True.

    **Validates: Requirements 11.4**
    """

    @given(domain=_invalid_domain_strategy)
    @settings(max_examples=100)
    def test_invalid_domain_validate_returns_false(self, domain: str):
        """validate_domain() returns False for any string not in DOMAIN_TAXONOMY keys.

        **Validates: Requirements 11.4**
        """
        assert validate_domain(domain) is False

    @given(domain=_invalid_domain_strategy)
    @settings(max_examples=100)
    def test_invalid_domain_get_tags_raises_value_error(self, domain: str):
        """get_domain_tags() raises ValueError for any string not in DOMAIN_TAXONOMY keys.

        **Validates: Requirements 11.4**
        """
        with pytest.raises(ValueError, match="Unknown domain"):
            get_domain_tags(domain)

    @given(domain=_valid_domain_strategy)
    @settings(max_examples=100)
    def test_valid_domain_validate_returns_true(self, domain: str):
        """validate_domain() returns True for any string in DOMAIN_TAXONOMY keys.

        **Validates: Requirements 11.4**
        """
        assert validate_domain(domain) is True


# Feature: knowledge-module-migration, Property 16: Taxonomy prompt formatting completeness


class TestTaxonomyPromptFormattingCompleteness:
    """Property 16: Taxonomy prompt formatting completeness.

    For any call to format_taxonomy_for_prompt(), the returned string should
    contain every domain name from get_all_domains().

    **Validates: Requirements 11.3**
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_format_contains_all_domains(self, data):
        """format_taxonomy_for_prompt() output contains every domain name.

        **Validates: Requirements 11.3**
        """
        result = format_taxonomy_for_prompt()
        all_domains = get_all_domains()

        # Pick a random domain to verify (Hypothesis explores all of them across runs)
        domain = data.draw(_valid_domain_strategy, label="domain")
        assert domain in result, (
            f"Domain '{domain}' not found in formatted taxonomy prompt"
        )

    def test_format_contains_every_domain_exhaustively(self):
        """format_taxonomy_for_prompt() output contains every single domain name.

        **Validates: Requirements 11.3**
        """
        result = format_taxonomy_for_prompt()
        all_domains = get_all_domains()

        for domain in all_domains:
            assert domain in result, (
                f"Domain '{domain}' not found in formatted taxonomy prompt"
            )
