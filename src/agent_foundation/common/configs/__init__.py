"""AgentFoundation config package — YAML-based object instantiation.

Importing this package eagerly registers all domain target aliases
via ``registered_targets``.
"""

# Ensure targets are registered on first import of this package.
from agent_foundation.common.configs import registered_targets  # noqa: F401

from agent_foundation.common.configs.factories import load_agent, load_inferencer

__all__ = ["load_agent", "load_inferencer"]
