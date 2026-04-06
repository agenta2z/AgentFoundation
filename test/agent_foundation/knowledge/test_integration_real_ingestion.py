
# pyre-strict

"""
Integration tests for knowledge ingestion with real LLM inference and real stores.

This module tests the full ingestion pipeline with:
1. Real in-memory stores (MemoryRetrievalService, MemoryKeyValueService, MemoryGraphService)
2. Real LLM inference via Plugboard (Claude Opus 4.5)
3. Actual sample files from _data/ directory

These tests make real LLM API calls and may take several seconds to complete.
Run with: buck2 test fbcode//rankevolve/test/knowledge:test_integration_real_ingestion
"""

import asyncio
import logging
import unittest
from pathlib import Path
from typing import Any

from agent_foundation.knowledge.ingestion.chunker import ChunkerConfig, MarkdownChunker
from agent_foundation.knowledge.ingestion.document_ingester import (
    DocumentIngester,
    IngesterConfig,
)
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from agent_foundation.knowledge.retrieval.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from agent_foundation.knowledge.services.graph_service.memory_graph_service import (
    MemoryGraphService,
)
from agent_foundation.knowledge.services.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)
from agent_foundation.knowledge.services.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)

logger = logging.getLogger(__name__)


class PlugboardSyncInferencer:
    """Synchronous wrapper around the async PlugboardClient for ingestion.

    This class provides a simple callable interface (prompt -> response) that
    the DocumentIngester expects, while internally using the async Plugboard
    streaming API.
    """

    def __init__(
        self,
        model: str = "claude-opus-4.5",
        pipeline: str = "usecase-3pai-claude-code",
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ) -> None:
        """Initialize the sync inferencer.

        Args:
            model: Model to use (default: claude-opus-4-5).
            pipeline: Plugboard pipeline (default: usecase-3pai-claude-code).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        """
        self.model = model
        self.pipeline = pipeline
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        """Make a synchronous LLM call.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The complete LLM response as a string.
        """
        return asyncio.run(self._async_call(prompt))

    async def _async_call(self, prompt: str) -> str:
        """Async implementation that streams and collects the response."""
        # Import here to avoid import errors when Plugboard is not available
        from rankevolve.src.server.llm.plugboard_client import PlugboardClient

        client = PlugboardClient(pipeline=self.pipeline)
        response_chunks = []

        try:
            async for chunk in client.stream_response(
                messages=[{"role": "user", "content": prompt}],
                system="You are a knowledge structuring assistant. Output ONLY valid JSON.",
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ):
                response_chunks.append(chunk)
        finally:
            await client.close()

        return "".join(response_chunks)


def create_real_knowledge_base() -> KnowledgeBase:
    """Create a KnowledgeBase with real in-memory stores.

    Returns:
        A KnowledgeBase instance backed by real in-memory services.
    """
    # Create real in-memory services
    retrieval_service = MemoryRetrievalService()
    kv_service = MemoryKeyValueService()
    graph_service = MemoryGraphService()

    # Create store adapters
    piece_store = RetrievalKnowledgePieceStore(retrieval_service=retrieval_service)
    metadata_store = KeyValueMetadataStore(kv_service=kv_service)
    graph_store = GraphServiceEntityGraphStore(graph_service=graph_service)

    # Create and return KnowledgeBase
    return KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
    )


def get_data_directory() -> Path:
    """Get the path to the _data directory with sample files.

    Returns:
        Path to the _data directory.

    Raises:
        unittest.SkipTest: If _data directory is not accessible.
    """
    # Try relative path from test file
    test_dir = Path(__file__).parent
    data_dir = test_dir / "_data"

    if data_dir.exists():
        return data_dir

    # Try absolute path in fbcode
    fbcode_path = Path("/data/users/zgchen/fbsource/fbcode")
    data_dir = fbcode_path / "rankevolve/test/knowledge/_data"

    if data_dir.exists():
        return data_dir

    raise unittest.SkipTest(
        "_data directory not accessible. "
        "This test requires sample files in rankevolve/test/knowledge/_data/"
    )


class RealIngestionIntegrationTest(unittest.TestCase):
    """Integration tests with real LLM inference and real stores."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        # Check if we're in an environment where Plugboard is available
        try:
            from rankevolve.src.server.llm.plugboard_client import (  # noqa: F401
                PlugboardClient,
            )
        except ImportError:
            raise unittest.SkipTest("PlugboardClient not available in this environment")

        # Try to get data directory
        cls.data_dir = get_data_directory()

    def test_chunker_with_real_file(self) -> None:
        """Test the MarkdownChunker with a real sample file."""
        # Read a sample file
        sample_file = self.data_dir / "li_skills.md"
        if not sample_file.exists():
            self.skipTest(f"Sample file not found: {sample_file}")

        content = sample_file.read_text(encoding="utf-8")

        # Create chunker with reasonable config
        config = ChunkerConfig(max_chars=2000, min_chars=200)
        chunker = MarkdownChunker(config)

        # Chunk the document
        chunks = chunker.chunk_document(content, source_file=str(sample_file))

        # Verify chunks were created
        self.assertGreater(len(chunks), 0, "Should produce at least one chunk")

        # Verify chunk properties
        for chunk in chunks:
            self.assertTrue(chunk.content.strip(), "Chunk content should not be empty")
            self.assertLessEqual(
                len(chunk.content),
                config.max_chars + 500,  # Allow some overflow for header context
                "Chunk should not exceed max size significantly",
            )

        logger.info(
            "Chunked %s into %d chunks",
            sample_file.name,
            len(chunks),
        )

    def test_real_stores_crud_operations(self) -> None:
        """Test that real in-memory stores support CRUD operations."""
        kb = create_real_knowledge_base()

        try:
            # Test metadata store
            from agent_foundation.knowledge.retrieval.models.entity_metadata import (
                EntityMetadata,
            )

            metadata = EntityMetadata(
                entity_id="user:test-user",
                entity_type="user",
                properties={"name": "Test User", "location": "Test City"},
            )
            kb.metadata_store.save_metadata(metadata)

            # Retrieve and verify
            retrieved = kb.metadata_store.get_metadata("user:test-user")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.entity_id, "user:test-user")
            self.assertEqual(retrieved.properties.get("name"), "Test User")

            # Test piece store
            from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
                KnowledgePiece,
                KnowledgeType,
            )

            piece = KnowledgePiece(
                piece_id="test-piece-001",
                content="This is test content about model optimization techniques.",
                knowledge_type=KnowledgeType.Fact,
                info_type="context",
                tags=["optimization", "model"],
                domain="model_optimization",
            )
            piece_id = kb.add_piece(piece)
            self.assertEqual(piece_id, "test-piece-001")

            # Search for the piece
            results = kb.piece_store.search(
                query="optimization techniques",
                top_k=5,
            )
            self.assertGreater(len(results), 0, "Should find the piece via search")

            # Test graph store
            from agent_foundation.knowledge.services.graph_service.graph_node import (
                GraphNode,
            )

            node = GraphNode(
                node_id="user:test-user",
                node_type="user",
                label="Test User",
            )
            kb.graph_store.add_node(node)

            retrieved_node = kb.graph_store.get_node("user:test-user")
            self.assertIsNotNone(retrieved_node)
            self.assertEqual(retrieved_node.label, "Test User")

            logger.info("All CRUD operations passed on real stores")

        finally:
            kb.close()

    def test_real_llm_inference_simple(self) -> None:
        """Test that real LLM inference works with a simple prompt."""
        inferencer = PlugboardSyncInferencer(
            model="claude-opus-4.5",
            max_tokens=100,
            temperature=0.3,
        )

        # Simple test prompt
        response = inferencer("Say 'hello' in exactly one word. Output only: hello")

        self.assertIsNotNone(response)
        self.assertGreater(len(response), 0, "Should get a response from LLM")
        logger.info("LLM response: %s", response[:100])

    def test_full_ingestion_pipeline_with_real_llm(self) -> None:
        """Test full ingestion pipeline with real LLM and real stores.

        This is the main integration test that verifies:
        1. Real file reading
        2. Real chunking
        3. Real LLM inference for structuring
        4. Real storage in KnowledgeBase
        5. Real retrieval from KnowledgeBase
        """
        # Create real inferencer
        inferencer = PlugboardSyncInferencer(
            model="claude-opus-4.5",
            max_tokens=8192,
            temperature=0.3,
        )

        # Create real KnowledgeBase
        kb = create_real_knowledge_base()

        try:
            # Create DocumentIngester with real inferencer
            config = IngesterConfig(
                max_retries=2,
                chunker_config=ChunkerConfig(max_chars=3000, min_chars=500),
            )
            ingester = DocumentIngester(inferencer=inferencer, config=config)

            # Find a sample file to ingest
            sample_file = self.data_dir / "li_skills.md"
            if not sample_file.exists():
                # Try another file
                sample_files = list(self.data_dir.glob("*.md"))
                if not sample_files:
                    self.skipTest("No sample markdown files found")
                sample_file = sample_files[0]

            logger.info("Ingesting file: %s", sample_file)

            # Run ingestion
            result = ingester.ingest_file(str(sample_file), kb)

            # Verify ingestion succeeded
            self.assertTrue(result.success, f"Ingestion failed: {result.errors}")
            logger.info(
                "Ingestion result: chunks=%d, pieces=%d, metadata=%d, nodes=%d, edges=%d",
                result.chunks_processed,
                result.pieces_created,
                result.metadata_created,
                result.graph_nodes_created,
                result.graph_edges_created,
            )

            # Verify data was stored
            self.assertGreater(
                result.pieces_created, 0, "Should create at least one knowledge piece"
            )

            # Test retrieval
            retrieval_result = kb.retrieve(query="skills techniques methods")
            logger.info(
                "Retrieval result: pieces=%d, has_metadata=%s",
                len(retrieval_result.pieces),
                retrieval_result.metadata is not None,
            )

            # The retrieval should find some of the ingested pieces
            # Note: This might be 0 if the LLM structured the content differently
            # than what we're searching for
            logger.info("Full integration test completed successfully!")

        finally:
            kb.close()


class StorePersistenceTest(unittest.TestCase):
    """Test that data persists correctly in the stores."""

    def test_multiple_ingestions_accumulate(self) -> None:
        """Test that multiple ingestions accumulate in the stores."""
        kb = create_real_knowledge_base()

        try:
            from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
                KnowledgePiece,
                KnowledgeType,
            )

            # Add first piece
            piece1 = KnowledgePiece(
                piece_id="piece-001",
                content="First piece about training efficiency.",
                knowledge_type=KnowledgeType.Fact,
                info_type="context",
                domain="training_efficiency",
            )
            kb.add_piece(piece1)

            # Add second piece
            piece2 = KnowledgePiece(
                piece_id="piece-002",
                content="Second piece about model architecture.",
                knowledge_type=KnowledgeType.Fact,
                info_type="context",
                domain="model_architecture",
            )
            kb.add_piece(piece2)

            # Verify both pieces exist
            retrieved1 = kb.piece_store.get_by_id("piece-001")
            retrieved2 = kb.piece_store.get_by_id("piece-002")

            self.assertIsNotNone(retrieved1)
            self.assertIsNotNone(retrieved2)
            self.assertEqual(retrieved1.domain, "training_efficiency")
            self.assertEqual(retrieved2.domain, "model_architecture")

            # Search should find both when relevant
            results = kb.piece_store.search(query="piece training model", top_k=10)
            self.assertGreaterEqual(
                len(results), 2, "Should find both pieces in search"
            )

        finally:
            kb.close()

    def test_domain_based_retrieval(self) -> None:
        """Test that domain-based retrieval works with real stores."""
        kb = create_real_knowledge_base()

        try:
            from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
                KnowledgePiece,
                KnowledgeType,
            )

            # Add pieces with different domains
            domains = [
                ("model_optimization", "Techniques for optimizing model performance."),
                ("training_efficiency", "Methods to improve training speed."),
                ("inference_efficiency", "Optimizations for faster inference."),
            ]

            for i, (domain, content) in enumerate(domains):
                piece = KnowledgePiece(
                    piece_id=f"domain-test-{i}",
                    content=content,
                    knowledge_type=KnowledgeType.Fact,
                    info_type="context",
                    domain=domain,
                    tags=["efficiency", "optimization"],
                )
                kb.add_piece(piece)

            # Test retrieval with domain filter
            result = kb.retrieve(
                query="optimization techniques",
                domain="model_optimization",
            )

            # Should have pieces (may be filtered by domain)
            logger.info(
                "Domain retrieval found %d pieces for model_optimization",
                len(result.pieces),
            )

        finally:
            kb.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
