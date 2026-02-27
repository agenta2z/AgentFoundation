"""
Document chunker for long-form knowledge ingestion.

This module provides markdown-aware chunking for long documents, enabling
ingestion of files that exceed LLM context limits. The chunker:
1. Splits by markdown headers to preserve logical structure
2. Respects maximum token/character limits per chunk
3. Maintains header context across chunks for coherent classification
4. Supports configurable overlap for context continuity

Key Features:
- Markdown section detection (# to ######)
- Hierarchical header tracking for context
- Configurable chunk size limits
- Overlap support for context continuity
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DocumentChunk:
    """A chunk of a document with context metadata.

    Attributes:
        content: The chunk text content.
        chunk_index: Zero-based index of this chunk in the document.
        total_chunks: Total number of chunks in the document.
        header_context: Hierarchical header path (e.g., "# Phase 1 > ## Setup").
        source_file: Optional source file path.
        start_line: Starting line number in original document.
        end_line: Ending line number in original document.
    """

    content: str
    chunk_index: int
    total_chunks: int
    header_context: str = ""
    source_file: Optional[str] = None
    start_line: int = 0
    end_line: int = 0


@dataclass
class ChunkerConfig:
    """Configuration for the document chunker.

    Attributes:
        max_chars: Maximum characters per chunk (default 12000 ~= 3000 tokens).
        min_chars: Minimum characters per chunk to avoid tiny chunks (default 500).
        overlap_chars: Characters of overlap between chunks for context (default 200).
        header_pattern: Regex pattern for markdown headers.
        preserve_code_blocks: Whether to keep code blocks intact.
    """

    max_chars: int = 12000
    min_chars: int = 500
    overlap_chars: int = 200
    header_pattern: str = r"^(#{1,6})\s+(.+)$"
    preserve_code_blocks: bool = True


class MarkdownChunker:
    """Markdown-aware document chunker for long-form ingestion.

    Splits documents by markdown headers while respecting character limits.
    Maintains header hierarchy context for each chunk.
    """

    config: ChunkerConfig
    _header_re: "re.Pattern[str]"

    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        """Initialize the chunker with optional configuration.

        Args:
            config: ChunkerConfig instance. Uses defaults if None.
        """
        self.config = config or ChunkerConfig()
        self._header_re = re.compile(self.config.header_pattern, re.MULTILINE)

    def chunk_document(
        self,
        content: str,
        source_file: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Split a document into chunks preserving markdown structure.

        Args:
            content: The full document text.
            source_file: Optional source file path for metadata.

        Returns:
            List of DocumentChunk objects.
        """
        if not content.strip():
            return []

        # Parse into sections by headers
        sections = self._parse_sections(content)

        # Merge small sections and split large ones
        chunks = self._create_chunks(sections)

        # Add metadata
        total = len(chunks)
        result = []
        for i, (chunk_content, header_ctx, start_line, end_line) in enumerate(chunks):
            result.append(
                DocumentChunk(
                    content=chunk_content,
                    chunk_index=i,
                    total_chunks=total,
                    header_context=header_ctx,
                    source_file=source_file,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        return result

    def _parse_sections(self, content: str) -> List[Tuple[int, str, str, int, int]]:
        """Parse document into sections by headers.

        Returns:
            List of (level, header_text, content, start_line, end_line) tuples.
            Level 0 means no header (preamble content).
        """
        lines = content.split("\n")
        sections: List[Tuple[int, str, str, int, int]] = []
        current_level = 0
        current_header = ""
        current_lines: List[str] = []
        current_start = 0

        for i, line in enumerate(lines):
            match = self._header_re.match(line)
            if match:
                # Save previous section if non-empty
                if current_lines or sections:
                    section_content = "\n".join(current_lines)
                    if section_content.strip():
                        sections.append(
                            (
                                current_level,
                                current_header,
                                section_content,
                                current_start,
                                i - 1,
                            )
                        )

                # Start new section
                current_level = len(match.group(1))
                current_header = match.group(2).strip()
                current_lines = [line]
                current_start = i
            else:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            section_content = "\n".join(current_lines)
            if section_content.strip():
                sections.append(
                    (
                        current_level,
                        current_header,
                        section_content,
                        current_start,
                        len(lines) - 1,
                    )
                )

        return sections

    def _create_chunks(
        self,
        sections: List[Tuple[int, str, str, int, int]],
    ) -> List[Tuple[str, str, int, int]]:
        """Create chunks from sections, merging small ones and splitting large ones.

        Returns:
            List of (content, header_context, start_line, end_line) tuples.
        """
        if not sections:
            return []

        chunks: List[Tuple[str, str, int, int]] = []
        header_stack: List[Tuple[int, str]] = []
        current_chunk_parts: List[str] = []
        current_start = 0
        current_end = 0
        current_len = 0

        def get_header_context() -> str:
            """Build header context string from stack."""
            if not header_stack:
                return ""
            return " > ".join(
                f"{'#' * level} {text}" for level, text in header_stack
            )

        def flush_chunk(chunk_start: int, chunk_end: int) -> None:
            """Save current chunk if non-empty."""
            nonlocal current_chunk_parts, current_len
            if current_chunk_parts:
                chunk_content = "\n\n".join(current_chunk_parts)
                if chunk_content.strip():
                    chunks.append(
                        (
                            chunk_content,
                            get_header_context(),
                            chunk_start,
                            chunk_end,
                        )
                    )
            current_chunk_parts = []
            current_len = 0

        for level, header, content, start_line, end_line in sections:
            # Update header stack
            if level > 0:
                # Pop headers at same or higher level
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, header))

            section_len = len(content)

            # If section alone exceeds max, split it
            if section_len > self.config.max_chars:
                flush_chunk(current_start, current_end)
                sub_chunks = self._split_large_section(
                    content, get_header_context(), start_line, end_line
                )
                chunks.extend(sub_chunks)
                current_start = end_line + 1
                continue

            # Account for the "\n\n" separator between parts
            separator_len = 2 if current_chunk_parts else 0

            # If adding this section would exceed max, flush first
            if current_len + separator_len + section_len > self.config.max_chars and current_len > 0:
                flush_chunk(current_start, current_end)
                current_start = start_line
                separator_len = 0

            # Add to current chunk
            if not current_chunk_parts:
                current_start = start_line
            current_chunk_parts.append(content)
            current_end = end_line
            current_len += separator_len + section_len

        # Flush remaining
        flush_chunk(current_start, current_end)

        return chunks

    def _split_large_section(
        self,
        content: str,
        header_context: str,
        start_line: int,
        end_line: int,
    ) -> List[Tuple[str, str, int, int]]:
        """Split a large section into smaller chunks.

        Tries to split at paragraph boundaries. Falls back to hard split.

        Returns:
            List of (content, header_context, start_line, end_line) tuples.
        """
        chunks: List[Tuple[str, str, int, int]] = []

        # Try to split by paragraphs (double newlines)
        paragraphs = re.split(r"\n\n+", content)

        current_parts: List[str] = []
        current_len = 0
        lines_so_far = 0

        for para in paragraphs:
            para_len = len(para)
            para_lines = para.count("\n") + 1

            # If single paragraph exceeds max, hard split it
            if para_len > self.config.max_chars:
                # Flush current
                if current_parts:
                    chunks.append(
                        (
                            "\n\n".join(current_parts),
                            header_context,
                            start_line + lines_so_far,
                            start_line + lines_so_far + para_lines - 1,
                        )
                    )
                    current_parts = []
                    current_len = 0

                # Hard split the paragraph
                hard_chunks = self._hard_split(para, self.config.max_chars)
                for hc in hard_chunks:
                    chunks.append(
                        (
                            hc,
                            header_context,
                            start_line + lines_so_far,
                            start_line + lines_so_far + hc.count("\n"),
                        )
                    )
                lines_so_far += para_lines
                continue

            # Account for the "\n\n" separator between parts
            separator_len = 2 if current_parts else 0

            # If adding would exceed max, flush first
            if current_len + separator_len + para_len > self.config.max_chars and current_len > 0:
                chunk_lines = sum(p.count("\n") + 1 for p in current_parts)
                chunks.append(
                    (
                        "\n\n".join(current_parts),
                        header_context,
                        start_line + lines_so_far - chunk_lines,
                        start_line + lines_so_far - 1,
                    )
                )
                current_parts = []
                current_len = 0
                separator_len = 0

            current_parts.append(para)
            current_len += separator_len + para_len
            lines_so_far += para_lines

        # Flush remaining
        if current_parts:
            chunk_lines = sum(p.count("\n") + 1 for p in current_parts)
            chunks.append(
                (
                    "\n\n".join(current_parts),
                    header_context,
                    end_line - chunk_lines + 1,
                    end_line,
                )
            )

        return chunks

    def _hard_split(self, text: str, max_chars: int) -> List[str]:
        """Hard split text at max_chars boundaries, preferring line breaks.

        Args:
            text: Text to split.
            max_chars: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        chunks: List[str] = []
        remaining = text

        while len(remaining) > max_chars:
            # Try to find a line break near the limit
            split_point = remaining.rfind("\n", 0, max_chars)
            if split_point == -1 or split_point < max_chars // 2:
                # No good line break, try space
                split_point = remaining.rfind(" ", 0, max_chars)
            if split_point == -1 or split_point < max_chars // 2:
                # Hard split at limit
                split_point = max_chars

            chunks.append(remaining[:split_point].strip())
            remaining = remaining[split_point:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks


def chunk_markdown_file(
    file_path: str,
    config: Optional[ChunkerConfig] = None,
) -> List[DocumentChunk]:
    """Convenience function to chunk a markdown file.

    Args:
        file_path: Path to the markdown file.
        config: Optional chunker configuration.

    Returns:
        List of DocumentChunk objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunker = MarkdownChunker(config)
    return chunker.chunk_document(content, source_file=file_path)


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 chars for English).

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // 4
