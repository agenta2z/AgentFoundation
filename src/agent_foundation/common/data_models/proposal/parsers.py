"""Pluggable proposal-parser registry.

AgentFoundation parses ``proposals.json`` directly via
:func:`agent_foundation.common.data_models.proposal.parser.parse_proposal_file`.
This module adds an *optional* indirection so a host application (e.g.
RankEvolve, whose ``parse_proposals`` yields a richer ``ProposalSelectionData``
shape than AF's generic :class:`ProposalIndex`) can register its own parser and
have framework code discover it without importing the host.

Design: a single-parser registry is sufficient for v1 (one producer at a time).
The getter returns ``None`` when nothing is registered, so callers degrade
gracefully (no enrichment) rather than crashing.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ProposalParser(Protocol):
    """Turns a workspace path into proposal data a selection widget understands.

    The return value is intentionally untyped (``Any``): AF consumers treat it
    as opaque widget metadata. ``None`` signals "nothing parseable here".
    """

    def parse(self, workspace: str) -> Optional[Any]: ...


_default_parser: Optional[ProposalParser] = None


def register_proposal_parser(parser: Optional[ProposalParser]) -> None:
    """Register (or, with ``None``, clear) the active proposal parser.

    Last registration wins. A host typically calls this once at startup.
    """
    global _default_parser
    _default_parser = parser


def get_proposal_parser() -> Optional[ProposalParser]:
    """Return the registered parser, or ``None`` if none was registered."""
    return _default_parser
