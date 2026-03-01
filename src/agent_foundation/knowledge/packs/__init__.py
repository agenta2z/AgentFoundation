"""
Knowledge Packs — bundled knowledge sets for atomic import/export.

This package provides the Knowledge Pack abstraction for managing collections
of KnowledgePieces as atomic units. Supports importing from ClawHub registry
and loading from local directories or JSON files.
"""

from agent_foundation.knowledge.packs.models import (
    KnowledgePack,
    PackInstallResult,
    PackManagerConfig,
    PackSource,
    PackStatus,
)
from agent_foundation.knowledge.packs.pack_manager import KnowledgePackManager
from agent_foundation.knowledge.packs.clawhub_adapter import (
    ClawhubClient,
    ClawhubPackAdapter,
    parse_skill_md,
)
from agent_foundation.knowledge.packs.local_pack_loader import LocalPackLoader

__all__ = [
    # Models
    "KnowledgePack",
    "PackInstallResult",
    "PackManagerConfig",
    "PackSource",
    "PackStatus",
    # Manager
    "KnowledgePackManager",
    # ClawHub
    "ClawhubClient",
    "ClawhubPackAdapter",
    "parse_skill_md",
    # Local
    "LocalPackLoader",
]
