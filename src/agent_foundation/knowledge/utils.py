"""
Utility functions for the knowledge module.

Provides entity ID sanitization for safe filesystem usage and entity type parsing
from the `type:name` ID convention.
"""


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
