"""Metamate backend (own Buck target: depends on the Buck-only Metamate SDK)."""

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.metamate.backend import (
    DEFAULT_ENTRY,
    MetamateBackend,
)

__all__ = ["DEFAULT_ENTRY", "MetamateBackend"]
