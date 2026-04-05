# pyre-strict
"""
Metamate Clients for AgentFoundation.

This module provides client implementations for interacting with
the Metamate platform, including:
- MetamateClientInterface: Abstract base class
- MetamateClient: Production implementation
- MockMetamateClient: Test implementation
- FallbackClient: Fallback implementation
"""

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.fallback_client import FallbackClient
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.interfaces import (
    FallbackClientInterface,
    MetamateClientInterface,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.metamate_client import MetamateClient
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.mock_metamate_client import (
    MockMetamateClient,
)

__all__ = [
    "MetamateClientInterface",
    "FallbackClientInterface",
    "MetamateClient",
    "MockMetamateClient",
    "FallbackClient",
]
