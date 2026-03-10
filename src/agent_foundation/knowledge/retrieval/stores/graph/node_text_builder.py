"""
NodeTextBuilder protocol and default implementation.

Defines a callable protocol for converting GraphNode instances into searchable
text strings for sidecar index ingestion. The default implementation
concatenates node_type, label, and property values into a single string.

If a node has an ``embedding_text`` key in its properties, that value is used
as the Document.embedding_text field (controlling what gets embedded), while
the builder output is used as the Document.content field (for BM25/keyword
search).

Requirements: 2.1, 2.2, 2.3, 2.4
"""

from typing import Protocol

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode


class NodeTextBuilder(Protocol):
    """Callable protocol for converting a GraphNode to searchable text.

    Implementations accept a GraphNode and return a string suitable for
    indexing in a retrieval service. The protocol is satisfied by any
    callable with a matching signature, including plain functions.
    """

    def __call__(self, node: GraphNode) -> str: ...


def default_node_text_builder(node: GraphNode) -> str:
    """Default node text builder: concatenate node_type, label, and property values.

    Iterates over sorted property keys, appending string values directly and
    non-string values as ``key: value`` pairs. The ``embedding_text`` key is
    excluded because it is handled separately as the Document.embedding_text
    field.

    Args:
        node: The GraphNode to convert to searchable text.

    Returns:
        A space-joined string of non-empty parts.
    """
    parts = [node.node_type, node.label]
    for key in sorted(node.properties.keys()):
        val = node.properties[key]
        if key == "embedding_text":
            continue
        if isinstance(val, str):
            parts.append(val)
        else:
            parts.append(f"{key}: {val}")
    return " ".join(p for p in parts if p)
