"""
Property-based tests for MarkdownChunker.

Feature: knowledge-module-migration
- Property 14: MarkdownChunker chunk size invariant

**Validates: Requirements 10.1**
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

from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.ingestion.chunker import (
    ChunkerConfig,
    MarkdownChunker,
)


# ── Strategies ────────────────────────────────────────────────────────────────

# Generate markdown header lines at various levels
_header_level = st.integers(min_value=1, max_value=6)
_header_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), blacklist_characters="\n\r\x00#"),
    min_size=1,
    max_size=40,
)


def _markdown_header(level: int, text: str) -> str:
    return f"{'#' * level} {text}"


# Generate a paragraph of body text (no newlines to keep it a single paragraph)
_body_paragraph = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), blacklist_characters="\n\r\x00"),
    min_size=1,
    max_size=300,
)


# A markdown section: optional header + one or more body paragraphs
@st.composite
def _markdown_section(draw):
    """Generate a single markdown section with a header and body paragraphs."""
    level = draw(_header_level)
    title = draw(_header_text)
    num_paragraphs = draw(st.integers(min_value=1, max_value=4))
    paragraphs = [draw(_body_paragraph) for _ in range(num_paragraphs)]
    header_line = _markdown_header(level, title)
    body = "\n\n".join(paragraphs)
    return f"{header_line}\n{body}"


# A full markdown document: one or more sections
@st.composite
def _markdown_document(draw):
    """Generate a non-empty markdown document with headers and content."""
    num_sections = draw(st.integers(min_value=1, max_value=6))
    sections = [draw(_markdown_section()) for _ in range(num_sections)]
    doc = "\n\n".join(sections)
    assume(doc.strip())
    return doc


# ChunkerConfig with max_chars constrained to make the property interesting
_chunker_config = st.builds(
    ChunkerConfig,
    max_chars=st.integers(min_value=100, max_value=5000),
    min_chars=st.just(10),
    overlap_chars=st.just(0),
)


# ── Property 14 ──────────────────────────────────────────────────────────────


class TestMarkdownChunkerChunkSizeInvariant:
    """Property 14: MarkdownChunker chunk size invariant.

    For any non-empty document and ChunkerConfig, every output DocumentChunk
    should have len(content) <= max_chars. The union of all chunk contents
    should cover the original document content (no content loss).

    **Validates: Requirements 10.1**
    """

    @given(document=_markdown_document(), config=_chunker_config)
    @settings(max_examples=100)
    def test_all_chunks_within_max_chars(self, document: str, config: ChunkerConfig):
        """Every chunk's content length must be <= max_chars.

        **Validates: Requirements 10.1**
        """
        chunker = MarkdownChunker(config)
        chunks = chunker.chunk_document(document)

        for chunk in chunks:
            assert len(chunk.content) <= config.max_chars, (
                f"Chunk {chunk.chunk_index} has {len(chunk.content)} chars, "
                f"exceeding max_chars={config.max_chars}"
            )

    @given(document=_markdown_document(), config=_chunker_config)
    @settings(max_examples=100)
    def test_no_content_loss(self, document: str, config: ChunkerConfig):
        """All words from the original document appear in some chunk (no content loss).

        **Validates: Requirements 10.1**
        """
        chunker = MarkdownChunker(config)
        chunks = chunker.chunk_document(document)

        # Collect all words from chunks
        chunk_text = " ".join(c.content for c in chunks)
        chunk_words = set(chunk_text.split())

        # Every non-whitespace word from the original should appear in chunks
        original_words = set(document.split())
        for word in original_words:
            assert word in chunk_words, (
                f"Word '{word[:50]}' from original document not found in any chunk"
            )
