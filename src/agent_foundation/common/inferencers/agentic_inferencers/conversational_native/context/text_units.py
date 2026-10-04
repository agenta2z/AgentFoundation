"""Text sizes in UTF-16 code units, the unit in which the JavaScript vendors
(Claude Code) measure them: a character outside the BMP is two units."""

from __future__ import annotations


def utf16_len(text: str) -> int:
    """Length of ``text`` in UTF-16 code units."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def utf16_head(text: str, size: int) -> str:
    """The longest start of ``text`` of at most ``size`` UTF-16 units."""
    if size <= 0:
        return ""
    end = min(len(text), size)
    excess = utf16_len(text[:end]) - size
    while excess > 0:
        # A character is one or two units, so fewer than half the excess in
        # characters cannot shed it, and dropping exactly that many never cuts
        # deeper than the longest fitting start.
        end -= (excess + 1) // 2
        excess = utf16_len(text[:end]) - size
    return text[:end]


def utf16_tail(text: str, size: int) -> str:
    """The longest end of ``text`` of at most ``size`` UTF-16 units."""
    if size <= 0:
        return ""
    start = max(len(text) - size, 0)
    excess = utf16_len(text[start:]) - size
    while excess > 0:
        start += (excess + 1) // 2
        excess = utf16_len(text[start:]) - size
    return text[start:]
