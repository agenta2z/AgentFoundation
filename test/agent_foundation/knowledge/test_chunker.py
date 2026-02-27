"""Unit tests for the markdown document chunker module."""

import os
import tempfile

import pytest

from agent_foundation.knowledge.ingestion.chunker import (
    ChunkerConfig,
    DocumentChunk,
    MarkdownChunker,
    chunk_markdown_file,
    estimate_tokens,
)


class TestDocumentChunk:
    """Tests for the DocumentChunk dataclass."""

    def test_basic_creation(self):
        chunk = DocumentChunk(content="hello", chunk_index=0, total_chunks=1)
        assert chunk.content == "hello"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 1
        assert chunk.header_context == ""
        assert chunk.source_file is None
        assert chunk.start_line == 0
        assert chunk.end_line == 0

    def test_full_creation(self):
        chunk = DocumentChunk(
            content="text",
            chunk_index=2,
            total_chunks=5,
            header_context="# Intro > ## Setup",
            source_file="doc.md",
            start_line=10,
            end_line=20,
        )
        assert chunk.header_context == "# Intro > ## Setup"
        assert chunk.source_file == "doc.md"
        assert chunk.start_line == 10
        assert chunk.end_line == 20


class TestChunkerConfig:
    """Tests for the ChunkerConfig dataclass."""

    def test_defaults(self):
        config = ChunkerConfig()
        assert config.max_chars == 12000
        assert config.min_chars == 500
        assert config.overlap_chars == 200
        assert config.preserve_code_blocks is True

    def test_custom_values(self):
        config = ChunkerConfig(max_chars=5000, min_chars=100)
        assert config.max_chars == 5000
        assert config.min_chars == 100


class TestMarkdownChunkerEmptyInput:
    """Tests for empty/whitespace input handling (Requirement 10.5)."""

    def test_empty_string(self):
        chunker = MarkdownChunker()
        assert chunker.chunk_document("") == []

    def test_whitespace_only(self):
        chunker = MarkdownChunker()
        assert chunker.chunk_document("   \n\n  \t  ") == []

    def test_newlines_only(self):
        chunker = MarkdownChunker()
        assert chunker.chunk_document("\n\n\n") == []


class TestMarkdownChunkerBasicSplitting:
    """Tests for header-based splitting (Requirement 10.1)."""

    def test_single_section_no_headers(self):
        chunker = MarkdownChunker()
        content = "Just some plain text without headers."
        chunks = chunker.chunk_document(content)
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_splits_by_headers(self):
        content = "# Section 1\nContent 1\n# Section 2\nContent 2"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_document(content)
        assert len(chunks) >= 1
        # All original content should be present across chunks
        combined = " ".join(c.content for c in chunks)
        assert "Content 1" in combined
        assert "Content 2" in combined

    def test_respects_max_chars(self):
        """Every chunk must be <= max_chars (Requirement 10.1)."""
        config = ChunkerConfig(max_chars=200)
        chunker = MarkdownChunker(config)
        content = "# Header\n" + ("A" * 500) + "\n# Header 2\n" + ("B" * 500)
        chunks = chunker.chunk_document(content)
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chars


class TestMarkdownChunkerHeaderContext:
    """Tests for hierarchical header context tracking (Requirement 10.2)."""

    def test_header_context_includes_parent(self):
        content = "# Top\nIntro\n## Sub\nDetails here"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_document(content)
        # At least one chunk should have header context
        contexts = [c.header_context for c in chunks]
        has_context = any("Top" in ctx for ctx in contexts)
        assert has_context

    def test_nested_headers_build_context(self):
        config = ChunkerConfig(max_chars=100)
        content = (
            "# Level 1\nSome text\n"
            "## Level 2\nMore text\n"
            "### Level 3\nDeep text that is long enough to be its own chunk"
        )
        chunker = MarkdownChunker(config)
        chunks = chunker.chunk_document(content)
        # The deepest chunk should have hierarchical context
        deep_chunks = [c for c in chunks if "Level 3" in c.header_context]
        if deep_chunks:
            ctx = deep_chunks[0].header_context
            assert "Level 1" in ctx
            assert "Level 2" in ctx

    def test_sibling_headers_reset_context(self):
        config = ChunkerConfig(max_chars=100)
        content = (
            "# Parent\nIntro\n"
            "## Child A\n" + "A content. " * 20 + "\n"
            "## Child B\n" + "B content. " * 20
        )
        chunker = MarkdownChunker(config)
        chunks = chunker.chunk_document(content)
        b_chunks = [c for c in chunks if "Child B" in c.header_context]
        for bc in b_chunks:
            assert "Child A" not in bc.header_context


class TestMarkdownChunkerParagraphSplitting:
    """Tests for paragraph-boundary splitting with hard-split fallback (Requirement 10.3)."""

    def test_splits_at_paragraph_boundaries(self):
        config = ChunkerConfig(max_chars=200)
        chunker = MarkdownChunker(config)
        # Create content with clear paragraph boundaries
        paragraphs = [f"Paragraph {i}. " * 10 for i in range(10)]
        content = "# Section\n" + "\n\n".join(paragraphs)
        chunks = chunker.chunk_document(content)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chars

    def test_hard_split_fallback(self):
        """When no paragraph boundary exists, falls back to hard split."""
        config = ChunkerConfig(max_chars=100)
        chunker = MarkdownChunker(config)
        # Single long line with no paragraph breaks
        content = "# Header\n" + "x" * 500
        chunks = chunker.chunk_document(content)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chars


class TestMarkdownChunkerMetadata:
    """Tests for chunk metadata (Requirement 10.4)."""

    def test_chunk_index_and_total(self):
        config = ChunkerConfig(max_chars=100)
        chunker = MarkdownChunker(config)
        content = "# A\n" + "word " * 50 + "\n# B\n" + "word " * 50
        chunks = chunker.chunk_document(content)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_source_file_propagated(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk_document("Some content", source_file="test.md")
        assert all(c.source_file == "test.md" for c in chunks)

    def test_start_end_lines_set(self):
        chunker = MarkdownChunker()
        content = "# Header\nLine 1\nLine 2\nLine 3"
        chunks = chunker.chunk_document(content)
        for chunk in chunks:
            assert isinstance(chunk.start_line, int)
            assert isinstance(chunk.end_line, int)
            assert chunk.end_line >= chunk.start_line


class TestChunkMarkdownFile:
    """Tests for the chunk_markdown_file convenience function."""

    def test_reads_and_chunks_file(self):
        content = "# Title\nSome content here."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            chunks = chunk_markdown_file(path)
            assert len(chunks) >= 1
            assert chunks[0].source_file == path
            assert "Some content here" in chunks[0].content
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            chunk_markdown_file("/nonexistent/path/file.md")

    def test_custom_config(self):
        content = "# A\n" + "word " * 200 + "\n# B\n" + "word " * 200
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            config = ChunkerConfig(max_chars=200)
            chunks = chunk_markdown_file(path, config=config)
            for chunk in chunks:
                assert len(chunk.content) <= 200
        finally:
            os.unlink(path)


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("abcd") == 1

    def test_longer_string(self):
        assert estimate_tokens("a" * 100) == 25

    def test_approximation(self):
        text = "Hello world, this is a test."
        assert estimate_tokens(text) == len(text) // 4
