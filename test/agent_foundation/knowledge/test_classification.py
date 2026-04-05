# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""
Tests for knowledge classification quality metrics and evaluation.

This module provides tests for:
1. Classification accuracy evaluation against human labels
2. Retrieval quality evaluation (precision, recall)
3. Domain taxonomy validation
4. Structuring prompt output validation
"""

import unittest
from typing import Dict, List, Tuple

from agent_foundation.knowledge.ingestion.taxonomy import (
    DOMAIN_TAXONOMY,
    format_taxonomy_for_prompt,
    get_all_domains,
    get_domain_tags,
    validate_domain,
    validate_tags,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)


def evaluate_classification_accuracy(
    sample_pieces: List[KnowledgePiece],
    human_labels: Dict[str, Dict],
) -> Dict[str, float]:
    """Evaluate LLM classification against human labels.

    Args:
        sample_pieces: List of classified KnowledgePiece objects.
        human_labels: Dict mapping piece_id to {domain, tags, info_type}.

    Returns:
        Dict with domain_accuracy, info_type_accuracy, tag_jaccard_mean.
    """
    domain_correct = 0
    info_type_correct = 0
    tag_overlap_scores = []

    for piece in sample_pieces:
        labels = human_labels.get(piece.piece_id, {})

        if piece.domain == labels.get("domain"):
            domain_correct += 1

        if piece.info_type == labels.get("info_type"):
            info_type_correct += 1

        pred_tags = set(piece.tags + piece.custom_tags)
        true_tags = set(labels.get("tags", []))
        if pred_tags or true_tags:
            union = pred_tags | true_tags
            if union:
                overlap = len(pred_tags & true_tags) / len(union)
                tag_overlap_scores.append(overlap)

    n = len(sample_pieces) if sample_pieces else 1
    return {
        "domain_accuracy": domain_correct / n,
        "info_type_accuracy": info_type_correct / n,
        "tag_jaccard_mean": (
            sum(tag_overlap_scores) / len(tag_overlap_scores)
            if tag_overlap_scores
            else 0.0
        ),
    }


def evaluate_retrieval_quality(
    test_queries: List[Tuple[str, List[str]]],
    retrieve_func,
    top_k: int = 5,
) -> Dict[str, float]:
    """Evaluate retrieval precision and recall.

    Args:
        test_queries: List of (query, relevant_piece_ids) tuples.
        retrieve_func: Function that takes query and returns List[(piece, score)].
        top_k: Number of results to consider.

    Returns:
        Dict with precision@k and recall@k.
    """
    precisions = []
    recalls = []

    for query, relevant_ids in test_queries:
        results = retrieve_func(query, top_k=top_k)
        retrieved_ids = {p.piece_id for p, _ in results}
        relevant_set = set(relevant_ids)

        if retrieved_ids:
            precision = len(retrieved_ids & relevant_set) / len(retrieved_ids)
            precisions.append(precision)

        if relevant_set:
            recall = len(retrieved_ids & relevant_set) / len(relevant_set)
            recalls.append(recall)

    return {
        "precision@k": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall@k": sum(recalls) / len(recalls) if recalls else 0.0,
    }


class TaxonomyTest(unittest.TestCase):
    """Tests for domain taxonomy validation."""

    def test_all_domains_have_description(self):
        """Every domain in taxonomy should have a description."""
        for domain, info in DOMAIN_TAXONOMY.items():
            self.assertIn("description", info, f"Domain '{domain}' missing description")
            self.assertTrue(
                info["description"], f"Domain '{domain}' has empty description"
            )

    def test_all_domains_have_tags(self):
        """Every domain should have at least one tag."""
        for domain, info in DOMAIN_TAXONOMY.items():
            self.assertIn("tags", info, f"Domain '{domain}' missing tags")
            self.assertIsInstance(
                info["tags"], list, f"Domain '{domain}' tags is not a list"
            )
            self.assertGreater(len(info["tags"]), 0, f"Domain '{domain}' has no tags")

    def test_general_domain_exists(self):
        """The 'general' fallback domain should exist."""
        self.assertIn("general", DOMAIN_TAXONOMY)

    def test_get_all_domains(self):
        """get_all_domains() should return all domain names."""
        domains = get_all_domains()
        self.assertEqual(set(domains), set(DOMAIN_TAXONOMY.keys()))

    def test_get_domain_tags_valid(self):
        """get_domain_tags() should return tags for valid domain."""
        tags = get_domain_tags("model_optimization")
        self.assertIsInstance(tags, list)
        self.assertIn("flash-attention", tags)

    def test_get_domain_tags_invalid(self):
        """get_domain_tags() should raise ValueError for invalid domain."""
        with self.assertRaises(ValueError):
            get_domain_tags("nonexistent_domain")

    def test_validate_domain(self):
        """validate_domain() should return True for valid domains."""
        self.assertTrue(validate_domain("model_optimization"))
        self.assertTrue(validate_domain("general"))
        self.assertFalse(validate_domain("nonexistent"))

    def test_validate_tags_all_valid(self):
        """validate_tags() should return empty list for valid tags."""
        invalid = validate_tags("model_optimization", ["flash-attention", "profiling"])
        self.assertEqual(invalid, [])

    def test_validate_tags_some_invalid(self):
        """validate_tags() should return invalid tags."""
        invalid = validate_tags(
            "model_optimization", ["flash-attention", "invalid-tag", "another-invalid"]
        )
        self.assertEqual(set(invalid), {"invalid-tag", "another-invalid"})

    def test_format_taxonomy_for_prompt(self):
        """format_taxonomy_for_prompt() should produce non-empty string."""
        formatted = format_taxonomy_for_prompt()
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 100)
        self.assertIn("model_optimization", formatted)
        self.assertIn("flash-attention", formatted)


class KnowledgePieceClassificationTest(unittest.TestCase):
    """Tests for KnowledgePiece domain classification."""

    def test_piece_default_domain(self):
        """KnowledgePiece should default to 'general' domain."""
        piece = KnowledgePiece(content="Test content")
        self.assertEqual(piece.domain, "general")

    def test_piece_default_secondary_domains(self):
        """KnowledgePiece should default to empty secondary_domains."""
        piece = KnowledgePiece(content="Test content")
        self.assertEqual(piece.secondary_domains, [])

    def test_piece_default_custom_tags(self):
        """KnowledgePiece should default to empty custom_tags."""
        piece = KnowledgePiece(content="Test content")
        self.assertEqual(piece.custom_tags, [])

    def test_piece_with_domain(self):
        """KnowledgePiece should accept domain parameter."""
        piece = KnowledgePiece(
            content="Optimize flash attention", domain="model_optimization"
        )
        self.assertEqual(piece.domain, "model_optimization")

    def test_piece_with_secondary_domains(self):
        """KnowledgePiece should accept secondary_domains parameter."""
        piece = KnowledgePiece(
            content="Flash attention on H100",
            domain="model_optimization",
            secondary_domains=["training_efficiency", "infrastructure"],
        )
        self.assertEqual(
            piece.secondary_domains, ["training_efficiency", "infrastructure"]
        )

    def test_piece_with_custom_tags(self):
        """KnowledgePiece should accept custom_tags parameter."""
        piece = KnowledgePiece(
            content="Custom feature implementation",
            custom_tags=["custom-feature", "experimental"],
        )
        self.assertEqual(piece.custom_tags, ["custom-feature", "experimental"])

    def test_piece_to_dict_includes_new_fields(self):
        """to_dict() should include domain, secondary_domains, custom_tags."""
        piece = KnowledgePiece(
            content="Test",
            domain="debugging",
            secondary_domains=["testing"],
            tags=["error-analysis"],
            custom_tags=["my-tag"],
        )
        d = piece.to_dict()
        self.assertEqual(d["domain"], "debugging")
        self.assertEqual(d["secondary_domains"], ["testing"])
        self.assertEqual(d["custom_tags"], ["my-tag"])

    def test_piece_from_dict_with_new_fields(self):
        """from_dict() should parse domain, secondary_domains, custom_tags."""
        data = {
            "content": "Test content",
            "domain": "workflow",
            "secondary_domains": ["debugging"],
            "tags": ["setup"],
            "custom_tags": ["my-custom"],
        }
        piece = KnowledgePiece.from_dict(data)
        self.assertEqual(piece.domain, "workflow")
        self.assertEqual(piece.secondary_domains, ["debugging"])
        self.assertEqual(piece.custom_tags, ["my-custom"])

    def test_piece_from_dict_backward_compatible(self):
        """from_dict() should work with legacy data missing new fields."""
        data = {
            "content": "Legacy content",
            "tags": ["old-tag"],
        }
        piece = KnowledgePiece.from_dict(data)
        self.assertEqual(piece.domain, "general")
        self.assertEqual(piece.secondary_domains, [])
        self.assertEqual(piece.custom_tags, [])


class KnowledgeTypeTest(unittest.TestCase):
    """Tests for KnowledgeType enum."""

    def test_example_type_exists(self):
        """KnowledgeType should include 'example' type."""
        self.assertEqual(KnowledgeType.Example.value, "example")

    def test_all_expected_types_exist(self):
        """KnowledgeType should have all expected values."""
        expected = {
            "fact",
            "instruction",
            "preference",
            "procedure",
            "note",
            "episodic",
            "example",
        }
        actual = {kt.value for kt in KnowledgeType}
        self.assertEqual(expected, actual)


class ClassificationAccuracyTest(unittest.TestCase):
    """Tests for classification accuracy evaluation."""

    def test_perfect_accuracy(self):
        """Perfect classification should yield 1.0 accuracy."""
        pieces = [
            KnowledgePiece(
                content="Test 1",
                piece_id="p1",
                domain="model_optimization",
                tags=["flash-attention"],
                info_type="context",
            ),
            KnowledgePiece(
                content="Test 2",
                piece_id="p2",
                domain="debugging",
                tags=["error-analysis"],
                info_type="instructions",
            ),
        ]
        labels = {
            "p1": {
                "domain": "model_optimization",
                "tags": ["flash-attention"],
                "info_type": "context",
            },
            "p2": {
                "domain": "debugging",
                "tags": ["error-analysis"],
                "info_type": "instructions",
            },
        }

        metrics = evaluate_classification_accuracy(pieces, labels)
        self.assertEqual(metrics["domain_accuracy"], 1.0)
        self.assertEqual(metrics["info_type_accuracy"], 1.0)
        self.assertEqual(metrics["tag_jaccard_mean"], 1.0)

    def test_zero_accuracy(self):
        """Completely wrong classification should yield 0.0 accuracy."""
        pieces = [
            KnowledgePiece(
                content="Test",
                piece_id="p1",
                domain="debugging",
                tags=["error-analysis"],
                info_type="skills",
            ),
        ]
        labels = {
            "p1": {
                "domain": "model_optimization",
                "tags": ["flash-attention"],
                "info_type": "context",
            },
        }

        metrics = evaluate_classification_accuracy(pieces, labels)
        self.assertEqual(metrics["domain_accuracy"], 0.0)
        self.assertEqual(metrics["info_type_accuracy"], 0.0)
        self.assertEqual(metrics["tag_jaccard_mean"], 0.0)

    def test_partial_tag_overlap(self):
        """Partial tag overlap should yield intermediate Jaccard score."""
        pieces = [
            KnowledgePiece(
                content="Test",
                piece_id="p1",
                domain="model_optimization",
                tags=["flash-attention", "profiling"],
                info_type="context",
            ),
        ]
        labels = {
            "p1": {
                "domain": "model_optimization",
                "tags": ["flash-attention", "memory-optimization"],
                "info_type": "context",
            },
        }

        metrics = evaluate_classification_accuracy(pieces, labels)
        self.assertAlmostEqual(metrics["tag_jaccard_mean"], 1 / 3)

    def test_empty_pieces_list(self):
        """Empty pieces list should not raise errors."""
        metrics = evaluate_classification_accuracy([], {})
        self.assertEqual(metrics["domain_accuracy"], 0.0)
        self.assertEqual(metrics["tag_jaccard_mean"], 0.0)


class RetrievalQualityTest(unittest.TestCase):
    """Tests for retrieval quality evaluation."""

    def test_perfect_retrieval(self):
        """Perfect retrieval should yield 1.0 precision and recall."""
        pieces = [
            (KnowledgePiece(content="A", piece_id="p1"), 0.9),
            (KnowledgePiece(content="B", piece_id="p2"), 0.8),
        ]

        def mock_retrieve(query, top_k=5):
            return pieces[:top_k]

        test_queries = [("test query", ["p1", "p2"])]
        metrics = evaluate_retrieval_quality(test_queries, mock_retrieve, top_k=2)

        self.assertEqual(metrics["precision@k"], 1.0)
        self.assertEqual(metrics["recall@k"], 1.0)

    def test_partial_retrieval(self):
        """Partial retrieval should yield intermediate scores."""
        pieces = [
            (KnowledgePiece(content="A", piece_id="p1"), 0.9),
            (KnowledgePiece(content="B", piece_id="p3"), 0.8),
        ]

        def mock_retrieve(query, top_k=5):
            return pieces[:top_k]

        test_queries = [("test query", ["p1", "p2"])]
        metrics = evaluate_retrieval_quality(test_queries, mock_retrieve, top_k=2)

        self.assertAlmostEqual(metrics["precision@k"], 0.5)
        self.assertAlmostEqual(metrics["recall@k"], 0.5)

    def test_empty_queries(self):
        """Empty queries list should not raise errors."""

        def mock_retrieve(query, top_k=5):
            return []

        metrics = evaluate_retrieval_quality([], mock_retrieve, top_k=5)
        self.assertEqual(metrics["precision@k"], 0.0)
        self.assertEqual(metrics["recall@k"], 0.0)


if __name__ == "__main__":
    unittest.main()
