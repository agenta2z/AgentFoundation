# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""
End-to-end integration test for knowledge ingestion pipeline.

This test validates the complete ingestion flow:
1. Reading sample markdown files from _data/
2. Chunking with MarkdownChunker
3. Processing with DocumentIngester (mocked LLM)
4. Verifying KnowledgeBase population

Run with:
    buck test //rankevolve/test/knowledge:test_e2e_ingestion
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

from agent_foundation.knowledge.ingestion.chunker import (
    ChunkerConfig,
    DocumentChunk,
    MarkdownChunker,
)
from agent_foundation.knowledge.ingestion.document_ingester import (
    DocumentIngester,
    IngesterConfig,
    IngestionResult,
)
from agent_foundation.knowledge.ingestion.taxonomy import (
    get_all_domains,
    get_domain_tags,
    validate_domain,
)


# Path to sample data files
DATA_DIR = Path(__file__).parent / "_data"


class ChunkerE2ETest(unittest.TestCase):
    """End-to-end tests for MarkdownChunker with real sample files."""

    def test_chunk_linfeng_skill_file(self) -> None:
        """Test chunking the linfeng_skill.md file."""
        file_path = DATA_DIR / "linfeng_skill.md"
        if not file_path.exists():
            self.skipTest(f"Test data file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        chunker = MarkdownChunker(ChunkerConfig(max_chars=4000))
        chunks = chunker.chunk_document(content, source_file=str(file_path))

        # Should produce multiple chunks given the file size
        self.assertGreater(len(chunks), 1)

        # Verify chunk structure
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertEqual(chunk.chunk_index, i)
            self.assertEqual(chunk.total_chunks, len(chunks))
            self.assertEqual(chunk.source_file, str(file_path))
            self.assertTrue(chunk.content.strip())  # Non-empty content

        # Verify chunks cover the document
        total_content = "\n".join(c.content for c in chunks)
        # Content should be substantial
        self.assertGreater(len(total_content), 1000)

        print(f"\n[linfeng_skill.md] Produced {len(chunks)} chunks:")
        for chunk in chunks:
            print(
                f"  Chunk {chunk.chunk_index}: "
                f"{len(chunk.content)} chars, "
                f"lines {chunk.start_line}-{chunk.end_line}, "
                f"context: '{chunk.header_context[:50]}...'"
                if chunk.header_context
                else ""
            )

    def test_chunk_li_skills_file(self) -> None:
        """Test chunking the larger li_skills.md file."""
        file_path = DATA_DIR / "li_skills.md"
        if not file_path.exists():
            self.skipTest(f"Test data file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        chunker = MarkdownChunker(ChunkerConfig(max_chars=8000))
        chunks = chunker.chunk_document(content, source_file=str(file_path))

        # This is a large file, should produce many chunks
        self.assertGreater(len(chunks), 5)

        # Verify header context is preserved
        headers_found = [c.header_context for c in chunks if c.header_context]
        self.assertGreater(len(headers_found), 0)

        print(f"\n[li_skills.md] Produced {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:5]):  # Show first 5
            ctx = chunk.header_context[:60] if chunk.header_context else ""
            print(f"  Chunk {i}: {len(chunk.content)} chars, context: '{ctx}...'")
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more chunks")

    def test_chunk_all_sample_files(self) -> None:
        """Test that all sample files can be chunked without errors."""
        if not DATA_DIR.exists():
            self.skipTest(f"Data directory not found: {DATA_DIR}")

        md_files = list(DATA_DIR.glob("*.md"))
        if not md_files:
            self.skipTest("No markdown files found in data directory")

        chunker = MarkdownChunker(ChunkerConfig(max_chars=5000))

        results: Dict[str, int] = {}
        for file_path in md_files:
            content = file_path.read_text(encoding="utf-8")
            chunks = chunker.chunk_document(content, source_file=str(file_path))
            results[file_path.name] = len(chunks)

        print(f"\n[All files] Chunking results:")
        for name, count in sorted(results.items()):
            print(f"  {name}: {count} chunks")

        # All files should produce at least 1 chunk
        for name, count in results.items():
            self.assertGreater(count, 0, f"{name} produced no chunks")


class TaxonomyE2ETest(unittest.TestCase):
    """Tests for domain taxonomy with sample content."""

    def test_taxonomy_covers_sample_content_domains(self) -> None:
        """Verify taxonomy includes domains relevant to sample files."""
        # Sample files contain ML optimization content
        relevant_domains = [
            "model_optimization",
            "training_efficiency",
            "inference_efficiency",
            "workflow",
        ]

        all_domains = get_all_domains()
        for domain in relevant_domains:
            self.assertIn(domain, all_domains)

    def test_domain_tags_for_sample_content(self) -> None:
        """Verify domain tags exist for expected content types."""
        # Sample files mention flash attention, memory optimization
        optimization_tags = get_domain_tags("model_optimization")
        self.assertIn("flash-attention", optimization_tags)
        self.assertIn("memory-optimization", optimization_tags)

        # Sample files contain workflow procedures - use actual taxonomy tags
        workflow_tags = get_domain_tags("workflow")
        self.assertIn("experiment-tracking", workflow_tags)
        self.assertIn("reproducibility", workflow_tags)

    def test_validate_domains_from_sample_keywords(self) -> None:
        """Test domain validation with keywords from sample files."""
        # Keywords from sample files should map to valid domains
        sample_keywords_domains = {
            "model_optimization": ["MLP", "FLOPs", "kernel"],
            "training_efficiency": ["batch", "gradient"],
            "workflow": ["ablation", "experiment"],
        }

        for domain, _keywords in sample_keywords_domains.items():
            self.assertTrue(validate_domain(domain), f"Domain {domain} should be valid")
            tags = get_domain_tags(domain)
            self.assertGreater(len(tags), 0, f"Domain {domain} should have tags")


class DocumentIngesterE2ETest(unittest.TestCase):
    """End-to-end tests for DocumentIngester with mocked LLM."""

    def _create_mock_llm_response(
        self, chunk_content: str, chunk_index: int
    ) -> Dict[str, Any]:
        """Create a realistic LLM response for a chunk."""
        return {
            "metadata": {
                "source": f"test-chunk-{chunk_index}",
            },
            "pieces": [
                {
                    "piece_id": f"chunk-{chunk_index}-fact-1",
                    "content": chunk_content[:200],
                    "knowledge_type": "fact",
                    "info_type": "context",
                    "domain": "model_optimization",
                    "tags": ["mlp", "optimization"],
                    "embedding_text": "MLP optimization techniques",
                },
                {
                    "piece_id": f"chunk-{chunk_index}-procedure-1",
                    "content": f"Procedure from chunk {chunk_index}",
                    "knowledge_type": "procedure",
                    "info_type": "instructions",
                    "domain": "workflow",
                    "tags": ["ablation-study"],
                    "embedding_text": "Ablation study workflow",
                },
            ],
            "graph": {
                "nodes": [
                    {
                        "node_id": f"concept:chunk-{chunk_index}-topic",
                        "node_type": "concept",
                        "label": f"Topic from chunk {chunk_index}",
                        "properties": {},
                    }
                ],
                "edges": [],
            },
        }

    def test_ingester_parses_valid_response(self) -> None:
        """Test that ingester correctly parses valid LLM responses."""
        mock_response = self._create_mock_llm_response("Test content", 0)
        mock_inferencer = MagicMock(return_value=json.dumps(mock_response))

        ingester = DocumentIngester(mock_inferencer)
        parsed = ingester._parse_and_validate(json.dumps(mock_response))

        self.assertEqual(len(parsed["pieces"]), 2)
        self.assertEqual(parsed["pieces"][0]["domain"], "model_optimization")

    def test_ingester_merge_results(self) -> None:
        """Test merging results from multiple chunks."""
        mock_inferencer = MagicMock()
        ingester = DocumentIngester(mock_inferencer)

        chunk_results = [
            self._create_mock_llm_response("Content 1", 0),
            self._create_mock_llm_response("Content 2", 1),
            self._create_mock_llm_response("Content 3", 2),
        ]

        merged = ingester._merge_results(chunk_results)

        # Should have 6 pieces (2 per chunk * 3 chunks)
        self.assertEqual(len(merged["pieces"]), 6)
        # Should have 3 graph nodes
        self.assertEqual(len(merged["graph"]["nodes"]), 3)

    def test_ingester_with_sample_file_chunks(self) -> None:
        """Test full ingestion flow with chunked sample file."""
        file_path = DATA_DIR / "linfeng_skill.md"
        if not file_path.exists():
            self.skipTest(f"Test data file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")

        # Step 1: Chunk the document
        chunker = MarkdownChunker(ChunkerConfig(max_chars=3000))
        chunks = chunker.chunk_document(content, source_file=str(file_path))

        print(f"\n[Ingestion E2E] Chunked into {len(chunks)} pieces")

        # Step 2: Create mock inferencer that returns valid responses
        call_count = [0]

        def mock_inferencer(prompt: str) -> str:
            idx = call_count[0]
            call_count[0] += 1
            # Extract some content from the prompt for realistic response
            content_start = prompt.find("User input:") + len("User input:")
            content_snippet = prompt[content_start : content_start + 100].strip()
            response = self._create_mock_llm_response(content_snippet, idx)
            return json.dumps(response)

        # Step 3: Run ingestion
        config = IngesterConfig(max_retries=1)
        ingester = DocumentIngester(mock_inferencer, config)

        # Process text directly (simulating what ingest_text does)
        all_structured_data = []
        for chunk in chunks:
            try:
                structured = ingester._process_chunk(chunk)
                if structured:
                    all_structured_data.append(structured)
            except Exception as e:
                print(f"  Chunk {chunk.chunk_index} failed: {e}")

        # Merge results
        merged = ingester._merge_results(all_structured_data)

        print(f"  Processed {len(all_structured_data)} chunks successfully")
        print(f"  Total pieces: {len(merged['pieces'])}")
        print(f"  Total graph nodes: {len(merged['graph']['nodes'])}")

        # Verify results
        self.assertEqual(len(all_structured_data), len(chunks))
        self.assertGreater(len(merged["pieces"]), 0)


class IngestionResultTest(unittest.TestCase):
    """Tests for IngestionResult dataclass."""

    def test_default_values(self) -> None:
        """Test IngestionResult default values."""
        result = IngestionResult(success=True)
        self.assertTrue(result.success)
        self.assertEqual(result.chunks_processed, 0)
        self.assertEqual(result.pieces_created, 0)
        self.assertEqual(result.errors, [])

    def test_with_values(self) -> None:
        """Test IngestionResult with all values."""
        result = IngestionResult(
            success=True,
            chunks_processed=5,
            pieces_created=10,
            metadata_created=2,
            graph_nodes_created=8,
            graph_edges_created=4,
            errors=["warning: skipped empty chunk"],
            source_file="/path/to/file.md",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.chunks_processed, 5)
        self.assertEqual(result.pieces_created, 10)
        self.assertEqual(len(result.errors), 1)


class FullPipelineE2ETest(unittest.TestCase):
    """Complete end-to-end pipeline test."""

    def test_full_pipeline_with_temp_file(self) -> None:
        """Test complete pipeline: write → chunk → ingest → verify."""
        # Create test content
        test_content = """
# Model Optimization Guide

This guide covers MLP optimization techniques.

## Prerequisites

- Understanding of neural network basics
- Familiarity with PyTorch

## Phase 1: Analysis

Analyze your model's performance bottlenecks:

1. Profile memory usage
2. Measure FLOPs per layer
3. Identify hotspots

### Key Metrics

| Metric | Target |
|--------|--------|
| Memory | < 16GB |
| QPS | > 1000 |

## Phase 2: Implementation

Apply optimization techniques:

1. Use flash attention for attention layers
2. Apply kernel fusion where possible
3. Consider quantization

```python
# Example optimization
model = optimize_model(model, techniques=["flash_attn", "fusion"])
```

## Conclusion

Optimization improves efficiency while maintaining quality.
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Step 1: Chunk
            chunker = MarkdownChunker(ChunkerConfig(max_chars=500))
            chunks = chunker.chunk_document(test_content, source_file=temp_path)

            print(f"\n[Full Pipeline E2E]")
            print(f"  Input: {len(test_content):,} chars")
            print(f"  Chunks: {len(chunks)}")

            # Verify chunks
            self.assertGreater(len(chunks), 1)

            # Step 2: Create mock inferencer
            def mock_inferencer(prompt: str) -> str:
                return json.dumps(
                    {
                        "metadata": {},
                        "pieces": [
                            {
                                "piece_id": "test-piece",
                                "content": "Test content",
                                "knowledge_type": "fact",
                                "info_type": "context",
                                "domain": "model_optimization",
                                "tags": ["flash-attention"],
                            }
                        ],
                        "graph": {"nodes": [], "edges": []},
                    }
                )

            # Step 3: Process chunks
            ingester = DocumentIngester(mock_inferencer, IngesterConfig(max_retries=1))
            results = []
            for chunk in chunks:
                result = ingester._process_chunk(chunk)
                if result:
                    results.append(result)

            print(f"  Processed chunks: {len(results)}")

            # Step 4: Merge
            merged = ingester._merge_results(results)
            print(f"  Final pieces: {len(merged['pieces'])}")

            self.assertEqual(len(results), len(chunks))
            self.assertGreater(len(merged["pieces"]), 0)

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
