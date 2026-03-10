from enum import Enum


class SearchMode(str, Enum):
    """Search mode configuration for SemanticGraphStore.

    Controls which search backends are used for graph node search.
    """

    NATIVE = "native"
    SIDECAR = "sidecar"
    BOTH = "both"
