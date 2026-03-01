"""Unit tests for the domain taxonomy module."""

import pytest

from agent_foundation.knowledge.ingestion.taxonomy import (
    DOMAIN_TAXONOMY,
    format_taxonomy_for_prompt,
    get_all_domains,
    get_domain_tags,
    validate_domain,
    validate_tags,
)


EXPECTED_DOMAINS = [
    "model_optimization",
    "model_architecture",
    "feature_engineering",
    "training_efficiency",
    "inference_efficiency",
    "data_engineering",
    "model_evaluation",
    "infrastructure",
    "debugging",
    "testing",
    "workflow",
    "agent_skills",
    "external_knowledge",
    "general",
]


class TestDomainTaxonomy:
    """Tests for the DOMAIN_TAXONOMY dictionary."""

    def test_has_14_domains(self):
        assert len(DOMAIN_TAXONOMY) == 14

    def test_all_expected_domains_present(self):
        for domain in EXPECTED_DOMAINS:
            assert domain in DOMAIN_TAXONOMY, f"Missing domain: {domain}"

    def test_each_domain_has_description_and_tags(self):
        for domain, info in DOMAIN_TAXONOMY.items():
            assert "description" in info, f"{domain} missing description"
            assert "tags" in info, f"{domain} missing tags"
            assert isinstance(info["description"], str)
            assert isinstance(info["tags"], list)
            assert len(info["tags"]) > 0, f"{domain} has no tags"


class TestGetAllDomains:
    def test_returns_all_14_domains(self):
        domains = get_all_domains()
        assert len(domains) == 14

    def test_returns_list_of_strings(self):
        domains = get_all_domains()
        assert all(isinstance(d, str) for d in domains)

    def test_contains_expected_domains(self):
        domains = get_all_domains()
        for expected in EXPECTED_DOMAINS:
            assert expected in domains


class TestGetDomainTags:
    def test_returns_tags_for_valid_domain(self):
        tags = get_domain_tags("general")
        assert isinstance(tags, list)
        assert "background" in tags

    def test_raises_for_invalid_domain(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            get_domain_tags("nonexistent_domain")

    def test_error_lists_valid_domains(self):
        with pytest.raises(ValueError, match="Valid domains"):
            get_domain_tags("fake")


class TestValidateDomain:
    def test_valid_domain_returns_true(self):
        assert validate_domain("general") is True

    def test_invalid_domain_returns_false(self):
        assert validate_domain("nonexistent") is False

    def test_empty_string_returns_false(self):
        assert validate_domain("") is False


class TestValidateTags:
    def test_valid_tags_returns_true(self):
        assert validate_tags("general", ["background", "reference"]) is True

    def test_invalid_tag_returns_false(self):
        assert validate_tags("general", ["background", "not-a-tag"]) is False

    def test_empty_tags_returns_true(self):
        assert validate_tags("general", []) is True

    def test_raises_for_invalid_domain(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            validate_tags("nonexistent", ["some-tag"])


class TestFormatTaxonomyForPrompt:
    def test_contains_all_domain_names(self):
        result = format_taxonomy_for_prompt()
        for domain in get_all_domains():
            assert domain in result

    def test_contains_descriptions(self):
        result = format_taxonomy_for_prompt()
        for info in DOMAIN_TAXONOMY.values():
            assert info["description"] in result

    def test_returns_non_empty_string(self):
        result = format_taxonomy_for_prompt()
        assert isinstance(result, str)
        assert len(result) > 0
