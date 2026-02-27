"""
Utility functions for the knowledge module.

Provides entity ID sanitization for safe filesystem usage, entity type parsing
from the `type:name` ID convention, cosine similarity, and token counting.
"""

import math
from typing import List


def sanitize_id(entity_id: str) -> str:
    """Percent-encode entity_id for use as a filename. Reversible via unsanitize_id.

    Encodes characters that are illegal or problematic in filenames:
        '%' → '%25'  (must be first to avoid double-encoding)
        ':' → '%3A'
        '/' → '%2F'
        '\\' → '%5C'

    Examples:
        >>> sanitize_id("user:xinli")
        'user%3Axinli'
        >>> sanitize_id("path/to/thing")
        'path%2Fto%2Fthing'
        >>> sanitize_id("back\\slash")
        'back%5Cslash'
        >>> sanitize_id("already%encoded")
        'already%25encoded'
    """
    return (
        entity_id
        .replace("%", "%25")
        .replace(":", "%3A")
        .replace("/", "%2F")
        .replace("\\", "%5C")
    )


def unsanitize_id(safe_id: str) -> str:
    """Reverse of sanitize_id. Decodes percent-encoded entity ID.

    Decodes in reverse order of encoding to ensure correctness:
        '%5C' → '\\\\'
        '%2F' → '/'
        '%3A' → ':'
        '%25' → '%'  (must be last to avoid premature decoding)

    Examples:
        >>> unsanitize_id("user%3Axinli")
        'user:xinli'
        >>> unsanitize_id("path%2Fto%2Fthing")
        'path/to/thing'
        >>> unsanitize_id("back%5Cslash")
        'back\\\\slash'
        >>> unsanitize_id("already%25encoded")
        'already%encoded'
    """
    return (
        safe_id
        .replace("%5C", "\\")
        .replace("%2F", "/")
        .replace("%3A", ":")
        .replace("%25", "%")
    )


def parse_entity_type(entity_id: str) -> str:
    """Extract entity_type from 'type:name' format.

    Returns 'default' if the entity_id does not contain a colon.

    Examples:
        >>> parse_entity_type("user:xinli")
        'user'
        >>> parse_entity_type("store:costco")
        'store'
        >>> parse_entity_type("plain_id")
        'default'
    """
    return entity_id.split(":")[0] if ":" in entity_id else "default"


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors using pure Python.

    Returns a value in [-1, 1]. Returns 0.0 if either vector has zero magnitude.

    Raises:
        ValueError: If vectors have different dimensions.

    Examples:
        >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
        >>> cosine_similarity([0.0, 0.0], [1.0, 1.0])
        0.0
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimensions do not match: {len(vec_a)} vs {len(vec_b)}"
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def count_tokens(text: str) -> int:
    """Return approximate token count using ~4 characters per token.

    Examples:
        >>> count_tokens("")
        0
        >>> count_tokens("hello world")
        2
        >>> count_tokens("abcd")
        1
    """
    return len(text) // 4

