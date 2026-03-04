"""
Unit tests for knowledge prompt templates.

Verifies that TemplateManager loads all expected templates, each template
renders with sample feed variables, version variants resolve correctly,
and the high-level helpers produce well-formed prompts.

Requirements: prompt template migration (all knowledge LLM prompts → .hbs)
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

_spu_src = Path(__file__).resolve().parents[3] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest

from agent_foundation.knowledge.prompt_templates import (
    KNOWLEDGE_TEMPLATE_MANAGER,
    render_prompt,
    get_structuring_prompt,
    get_classification_prompt,
)
from agent_foundation.knowledge.prompt_templates._config import PROMPT_CONFIGS


# ── Expected template keys ──────────────────────────────────────────────────

EXPECTED_KEYS = [
    "ingestion/Structuring",
    "ingestion/Structuring.PiecesOnly",
    "ingestion/Classification",
    "ingestion/Structuring.Cli",
    "quality/DedupJudge",
    "quality/MergeCandidate",
    "quality/MergeExecution",
    "quality/UpdateIntent",
    "quality/UpdateContentGeneration",
    "quality/Validation",
    "quality/SkillSynthesis",
    "retrieval/QueryDecomposition",
    "retrieval/KnowledgeConsolidation",
]


# ── Sample feed variables for each template ─────────────────────────────────

SAMPLE_FEEDS = {
    "ingestion/Structuring": {
        "domain_taxonomy": "- general\n- cooking",
        "context": "Header: Cooking > Recipes",
        "user_input": "I like to make pasta with garlic.",
    },
    "ingestion/Classification": {
        "domain_taxonomy": "- general\n- cooking",
        "content": "Use olive oil for sauteing garlic.",
        "tags": "cooking, garlic",
        "info_type": "instructions",
        "knowledge_type": "instruction",
    },
    "quality/DedupJudge": {
        "similarity": "0.912",
        "existing_content": "Pasta cooks in 8-10 minutes.",
        "existing_domain": "cooking",
        "existing_tags": "pasta, cooking_time",
        "existing_created_at": "2024-01-15T10:00:00",
        "new_content": "Cook pasta for 8 to 10 minutes in boiling water.",
        "new_domain": "cooking",
        "new_tags": "pasta, boiling",
    },
    "quality/MergeCandidate": {
        "new_content": "Garlic bread is made by toasting bread with garlic butter.",
        "new_domain": "cooking",
        "new_tags": "garlic_bread, bread",
        "candidates_formatted": "Candidate 1 (ID: abc-123):\nContent: Garlic bread recipe...",
    },
    "quality/MergeExecution": {
        "piece_a_content": "Use butter and garlic for garlic bread.",
        "piece_a_domain": "cooking",
        "piece_a_tags": "garlic_bread, butter",
        "piece_b_content": "Toast the bread at 375F for 10 minutes.",
        "piece_b_domain": "cooking",
        "piece_b_tags": "garlic_bread, toasting",
    },
    "quality/UpdateIntent": {
        "existing_id": "garlic-bread-recipe",
        "existing_content": "Use butter and minced garlic on French bread.",
        "existing_length": "45",
        "existing_domain": "cooking",
        "existing_tags": "garlic_bread, french_bread",
        "existing_updated_at": "2024-01-15T10:00:00",
        "new_content": "Use garlic butter with parsley on Italian bread, toast at 400F.",
        "new_length": "62",
        "update_instruction": "User says: Update with Italian bread variation.",
    },
    "quality/UpdateContentGeneration": {
        "existing_content": "Garlic bread uses butter and minced garlic on French bread.",
        "update_instruction": "Add information about the Italian bread variation with parsley.",
    },
    "quality/Validation": {
        "content": "Boil pasta in salted water for 8-10 minutes.",
        "domain": "cooking",
        "source": "culinary_guide",
        "created_at": "2024-03-01T12:00:00",
        "checks_to_perform": "correctness, authenticity",
    },
    "quality/SkillSynthesis": {
        "pieces_formatted": (
            "Piece 1 (procedure):\nStep 1: Boil water with salt\n\n"
            "Piece 2 (procedure):\nStep 2: Add pasta and cook 8-10 min\n\n"
            "Piece 3 (procedure):\nStep 3: Drain and toss with sauce"
        ),
        "num_pieces": "3",
        "avg_similarity": "0.890",
        "common_tags": "pasta, cooking, boiling",
        "domains": "cooking",
    },
    "retrieval/QueryDecomposition": {
        "domains_section": "Available domains: cooking, nutrition\n\n",
        "query": "How long should I cook pasta and what sauce goes with it?",
    },
    "retrieval/KnowledgeConsolidation": {
        "query": "Best way to cook pasta",
        "retrieved_knowledge": (
            "1. Boil water with salt first.\n"
            "2. Cook pasta for 8-10 minutes.\n"
            "3. Save pasta water for sauce."
        ),
    },
}


# ── Tests ────────────────────────────────────────────────────────────────────


def _flatten_template_keys(templates_dict: dict, prefix: str = "") -> set:
    """Flatten nested templates dict into slash-separated key set."""
    keys = set()
    for k, v in templates_dict.items():
        full_key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            keys.update(_flatten_template_keys(v, full_key))
        else:
            keys.add(full_key)
    return keys


class TestTemplateManagerLoadsAllKeys:
    """Verify all 13 expected template keys are loaded."""

    def test_all_expected_keys_present(self):
        loaded_keys = _flatten_template_keys(KNOWLEDGE_TEMPLATE_MANAGER.templates)
        for key in EXPECTED_KEYS:
            assert key in loaded_keys, f"Missing template key: {key}"

    def test_no_unexpected_keys(self):
        """All loaded keys should be in our expected set."""
        loaded_keys = _flatten_template_keys(KNOWLEDGE_TEMPLATE_MANAGER.templates)
        for key in loaded_keys:
            assert key in EXPECTED_KEYS, f"Unexpected template key: {key}"


class TestRenderPromptProducesOutput:
    """Each template renders non-empty with sample feed variables."""

    @pytest.mark.parametrize(
        "template_key",
        [k for k in SAMPLE_FEEDS.keys()],
        ids=[k.split("/")[-1] for k in SAMPLE_FEEDS.keys()],
    )
    def test_render_produces_nonempty(self, template_key):
        result = render_prompt(template_key, **SAMPLE_FEEDS[template_key])
        assert isinstance(result, str)
        assert len(result) > 50, f"Template {template_key} rendered too short"

    @pytest.mark.parametrize(
        "template_key",
        [k for k in SAMPLE_FEEDS.keys()],
        ids=[k.split("/")[-1] for k in SAMPLE_FEEDS.keys()],
    )
    def test_render_contains_feed_values(self, template_key):
        """At least one feed value appears in the rendered output."""
        feed = SAMPLE_FEEDS[template_key]
        result = render_prompt(template_key, **feed)
        # Check that at least one feed value appears verbatim
        found = any(str(v) in result for v in feed.values() if len(str(v)) > 5)
        assert found, f"No feed values found in rendered {template_key}"


class TestStructuringVersionMechanism:
    """Version variants resolve correctly via template_version parameter."""

    def test_pieces_only_variant(self):
        result = render_prompt(
            "ingestion/Structuring",
            template_version="PiecesOnly",
            domain_taxonomy="- general",
            user_input="Test input for pieces only.",
        )
        assert isinstance(result, str)
        assert len(result) > 50

    def test_cli_variant(self):
        result = render_prompt(
            "ingestion/Structuring",
            template_version="Cli",
            user_input="Test input for CLI variant.",
        )
        assert isinstance(result, str)
        assert "metadata" in result
        assert "pieces" in result
        assert "graph" in result

    def test_cli_variant_contains_user_input(self):
        result = render_prompt(
            "ingestion/Structuring",
            template_version="Cli",
            user_input="My unique test phrase here.",
        )
        assert "My unique test phrase here." in result


class TestGetStructuringPromptHelper:
    """High-level helper loads taxonomy and renders prompt."""

    def test_full_schema(self):
        result = get_structuring_prompt("Test user input", context="some context")
        assert isinstance(result, str)
        assert "Test user input" in result
        assert len(result) > 200

    def test_pieces_only_schema(self):
        result = get_structuring_prompt("Test user input", full_schema=False)
        assert isinstance(result, str)
        assert "Test user input" in result

    def test_empty_context_uses_placeholder(self):
        result = get_structuring_prompt("Test user input", context="")
        assert "No additional context" in result


class TestGetClassificationPromptHelper:
    """Helper loads taxonomy and renders classification prompt."""

    def test_basic_classification(self):
        result = get_classification_prompt(
            content="Use olive oil for cooking.",
            tags=["cooking", "oil"],
            info_type="instructions",
            knowledge_type="instruction",
        )
        assert isinstance(result, str)
        assert "olive oil" in result
        assert len(result) > 100

    def test_empty_tags(self):
        result = get_classification_prompt(
            content="Test content",
            tags=[],
            info_type="context",
            knowledge_type="fact",
        )
        assert "(none)" in result


class TestPromptConfigs:
    """Advisory config dict covers all template keys."""

    def test_all_base_templates_have_config(self):
        """Every non-variant template key has an entry in PROMPT_CONFIGS."""
        base_keys = [
            k for k in EXPECTED_KEYS
            if "." not in k.split("/")[-1]  # skip version variants
        ]
        for key in base_keys:
            assert key in PROMPT_CONFIGS, f"Missing config for {key}"

    def test_configs_have_required_fields(self):
        for key, config in PROMPT_CONFIGS.items():
            assert "temperature" in config, f"{key} missing temperature"
            assert "max_tokens" in config, f"{key} missing max_tokens"
