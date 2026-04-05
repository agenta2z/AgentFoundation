# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""
Tests for knowledge ingestion pipeline.

This module provides tests for:
1. MarkdownChunker - document chunking
2. DocumentIngester - end-to-end ingestion pipeline
3. Structuring prompts - prompt generation
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

from agent_foundation.knowledge.ingestion.chunker import (
    chunk_markdown_file,
    ChunkerConfig,
    estimate_tokens,
    MarkdownChunker,
)
from agent_foundation.knowledge.ingestion.document_ingester import (
    DocumentIngester,
    IngesterConfig,
    IngestionResult,
)
from agent_foundation.knowledge.ingestion.prompts.structuring_prompt import (
    get_classification_prompt,
    get_structuring_prompt,
)


class MarkdownChunkerTest(unittest.TestCase):
    """Tests for MarkdownChunker."""

    def test_empty_document(self):
        """Empty document should produce no chunks."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk_document("")
        self.assertEqual(chunks, [])

    def test_whitespace_only_document(self):
        """Whitespace-only document should produce no chunks."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk_document("   \n\n  \t  ")
        self.assertEqual(chunks, [])

    def test_simple_document_single_chunk(self):
        """Small document should produce single chunk."""
        content = "# Title\n\nSome content here."
        chunker = MarkdownChunker()
        chunks = chunker.chunk_document(content)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].total_chunks, 1)
        self.assertIn("Title", chunks[0].content)

    def test_document_with_multiple_sections(self):
        """Document with headers should preserve structure."""
        content = """# Phase 1

Setup instructions.

## Prerequisites

- Item 1
- Item 2

# Phase 2

More content.
"""
        chunker = MarkdownChunker(ChunkerConfig(max_chars=50))
        chunks = chunker.chunk_document(content)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.total_chunks, len(chunks))

    def test_header_context_tracking(self):
        """Chunker should track header hierarchy."""
        content = """# Top Level

Intro text.

## Second Level

Some content.

### Third Level

More content.
"""
        chunker = MarkdownChunker(ChunkerConfig(max_chars=100))
        chunks = chunker.chunk_document(content)

        self.assertGreater(len(chunks), 0)

    def test_large_section_splitting(self):
        """Large sections should be split."""
        long_content = "A" * 20000
        content = f"# Big Section\n\n{long_content}"

        chunker = MarkdownChunker(ChunkerConfig(max_chars=5000))
        chunks = chunker.chunk_document(content)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 6000)

    def test_source_file_metadata(self):
        """Source file should be set in chunk metadata."""
        content = "# Test\n\nContent."
        chunker = MarkdownChunker()
        chunks = chunker.chunk_document(content, source_file="/path/to/file.md")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_file, "/path/to/file.md")

    def test_chunk_indices(self):
        """Chunk indices should be sequential."""
        content = "\n\n".join([f"# Section {i}\n\nContent {i}" for i in range(5)])
        chunker = MarkdownChunker(ChunkerConfig(max_chars=50))
        chunks = chunker.chunk_document(content)

        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)


class ChunkMarkdownFileTest(unittest.TestCase):
    """Tests for chunk_markdown_file function."""

    def test_chunk_file(self):
        """chunk_markdown_file should read and chunk a file."""
        content = "# Test\n\nSome content."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            chunks = chunk_markdown_file(temp_path)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].source_file, temp_path)
        finally:
            Path(temp_path).unlink()

    def test_chunk_file_not_found(self):
        """chunk_markdown_file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            chunk_markdown_file("/nonexistent/path.md")


class EstimateTokensTest(unittest.TestCase):
    """Tests for estimate_tokens function."""

    def test_estimate_tokens(self):
        """Should estimate ~1 token per 4 chars."""
        text = "A" * 100
        tokens = estimate_tokens(text)
        self.assertEqual(tokens, 25)

    def test_estimate_tokens_empty(self):
        """Empty string should have 0 tokens."""
        self.assertEqual(estimate_tokens(""), 0)


class StructuringPromptTest(unittest.TestCase):
    """Tests for structuring prompt generation."""

    def test_get_structuring_prompt_full_schema(self):
        """Full schema prompt should include all sections."""
        prompt = get_structuring_prompt("test input", full_schema=True)

        self.assertIn("test input", prompt)
        self.assertIn("metadata", prompt)
        self.assertIn("pieces", prompt)
        self.assertIn("graph", prompt)
        self.assertIn("Domain Taxonomy", prompt)

    def test_get_structuring_prompt_pieces_only(self):
        """Pieces-only prompt should not mention graph/metadata."""
        prompt = get_structuring_prompt("test input", full_schema=False)

        self.assertIn("test input", prompt)
        self.assertIn("pieces", prompt)

    def test_get_structuring_prompt_with_context(self):
        """Context should be included in prompt."""
        prompt = get_structuring_prompt(
            "test input",
            context="Section: # Phase 1 > ## Setup",
            full_schema=True,
        )

        self.assertIn("Section: # Phase 1 > ## Setup", prompt)

    def test_get_classification_prompt(self):
        """Classification prompt should include piece info."""
        prompt = get_classification_prompt(
            content="Flash attention optimization",
            tags=["flash-attention"],
            info_type="context",
            knowledge_type="procedure",
        )

        self.assertIn("Flash attention optimization", prompt)
        self.assertIn("flash-attention", prompt)
        self.assertIn("context", prompt)
        self.assertIn("procedure", prompt)


class DocumentIngesterTest(unittest.TestCase):
    """Tests for DocumentIngester."""

    def _create_mock_inferencer(self, response: Dict[str, Any]) -> MagicMock:
        """Create a mock inferencer that returns the given response."""
        mock = MagicMock()
        mock.return_value = json.dumps(response)
        return mock

    def _create_mock_kb(self) -> MagicMock:
        """Create a mock KnowledgeBase."""
        kb = MagicMock()
        kb.metadata_store = MagicMock()
        kb.piece_store = MagicMock()
        kb.graph_store = MagicMock()
        return kb

    def test_ingestion_result_defaults(self):
        """IngestionResult should have sensible defaults."""
        result = IngestionResult(success=True)
        self.assertTrue(result.success)
        self.assertEqual(result.chunks_processed, 0)
        self.assertEqual(result.pieces_created, 0)
        self.assertEqual(result.errors, [])

    def test_ingester_config_defaults(self):
        """IngesterConfig should have sensible defaults."""
        config = IngesterConfig()
        self.assertEqual(config.max_retries, 3)
        self.assertTrue(config.full_schema)
        self.assertTrue(config.merge_graphs)
        self.assertTrue(config.dedupe_pieces)

    def test_parse_and_validate_valid_response(self):
        """Valid JSON response should be parsed."""
        response = {
            "metadata": {},
            "pieces": [
                {
                    "piece_id": "test-piece",
                    "content": "Test content",
                    "knowledge_type": "fact",
                    "info_type": "context",
                }
            ],
            "graph": {"nodes": [], "edges": []},
        }

        mock_inferencer = self._create_mock_inferencer(response)
        ingester = DocumentIngester(mock_inferencer)

        parsed = ingester._parse_and_validate(json.dumps(response))

        self.assertEqual(parsed["pieces"][0]["piece_id"], "test-piece")

    def test_parse_and_validate_adds_missing_sections(self):
        """Missing sections should be added."""
        response = {
            "pieces": [
                {
                    "piece_id": "test",
                    "content": "Test",
                    "knowledge_type": "fact",
                    "info_type": "context",
                }
            ]
        }

        mock_inferencer = self._create_mock_inferencer(response)
        ingester = DocumentIngester(mock_inferencer)

        parsed = ingester._parse_and_validate(json.dumps(response))

        self.assertIn("metadata", parsed)
        self.assertIn("graph", parsed)

    def test_parse_and_validate_strips_markdown_fences(self):
        """Markdown code fences should be stripped."""
        response = {
            "metadata": {},
            "pieces": [
                {
                    "piece_id": "test",
                    "content": "Test",
                    "knowledge_type": "fact",
                    "info_type": "context",
                }
            ],
            "graph": {"nodes": [], "edges": []},
        }

        mock_inferencer = self._create_mock_inferencer(response)
        ingester = DocumentIngester(mock_inferencer)

        fenced = f"```json\n{json.dumps(response)}\n```"
        parsed = ingester._parse_and_validate(fenced)

        self.assertEqual(parsed["pieces"][0]["piece_id"], "test")

    def test_parse_and_validate_invalid_json(self):
        """Invalid JSON should raise ValueError."""
        mock_inferencer = MagicMock()
        ingester = DocumentIngester(mock_inferencer)

        with self.assertRaises(ValueError) as ctx:
            ingester._parse_and_validate("not valid json")

        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_parse_and_validate_missing_piece_fields(self):
        """Missing piece fields should raise ValueError."""
        response = {
            "metadata": {},
            "pieces": [{"piece_id": "test"}],
            "graph": {"nodes": [], "edges": []},
        }

        mock_inferencer = self._create_mock_inferencer(response)
        ingester = DocumentIngester(mock_inferencer)

        with self.assertRaises(ValueError) as ctx:
            ingester._parse_and_validate(json.dumps(response))

        self.assertIn("missing required fields", str(ctx.exception))

    def test_parse_and_validate_invalid_knowledge_type(self):
        """Invalid knowledge_type should raise ValueError."""
        response = {
            "metadata": {},
            "pieces": [
                {
                    "piece_id": "test",
                    "content": "Test",
                    "knowledge_type": "invalid_type",
                    "info_type": "context",
                }
            ],
            "graph": {"nodes": [], "edges": []},
        }

        mock_inferencer = self._create_mock_inferencer(response)
        ingester = DocumentIngester(mock_inferencer)

        with self.assertRaises(ValueError) as ctx:
            ingester._parse_and_validate(json.dumps(response))

        self.assertIn("invalid knowledge_type", str(ctx.exception))

    def test_merge_results_deduplicates_pieces(self):
        """Merge should deduplicate pieces by piece_id."""
        mock_inferencer = MagicMock()
        ingester = DocumentIngester(mock_inferencer)

        data1 = {
            "metadata": {},
            "pieces": [
                {"piece_id": "p1", "content": "First"},
                {"piece_id": "p2", "content": "Second"},
            ],
            "graph": {"nodes": [], "edges": []},
        }
        data2 = {
            "metadata": {},
            "pieces": [
                {"piece_id": "p1", "content": "Duplicate"},
                {"piece_id": "p3", "content": "Third"},
            ],
            "graph": {"nodes": [], "edges": []},
        }

        merged = ingester._merge_results([data1, data2])

        piece_ids = [p["piece_id"] for p in merged["pieces"]]
        self.assertEqual(len(piece_ids), 3)
        self.assertIn("p1", piece_ids)
        self.assertIn("p2", piece_ids)
        self.assertIn("p3", piece_ids)

    def test_merge_results_combines_graph(self):
        """Merge should combine graph nodes and edges."""
        mock_inferencer = MagicMock()
        ingester = DocumentIngester(mock_inferencer)

        data1 = {
            "metadata": {},
            "pieces": [],
            "graph": {
                "nodes": [{"node_id": "n1", "node_type": "type1"}],
                "edges": [{"source_id": "n1", "target_id": "n2", "edge_type": "REL"}],
            },
        }
        data2 = {
            "metadata": {},
            "pieces": [],
            "graph": {
                "nodes": [{"node_id": "n2", "node_type": "type2"}],
                "edges": [],
            },
        }

        merged = ingester._merge_results([data1, data2])

        self.assertEqual(len(merged["graph"]["nodes"]), 2)
        self.assertEqual(len(merged["graph"]["edges"]), 1)


if __name__ == "__main__":
    unittest.main()
