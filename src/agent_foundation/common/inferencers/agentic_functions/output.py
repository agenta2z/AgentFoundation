# pyre-strict

"""Output carrier and escalation signals for the agentic-function framework."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import attr
from agent_foundation.common.inferencers.agentic_functions.errors import ParseError
from agent_foundation.common.response_parsers.json_block import (
    extract_json_block,
    find_json_block_text,
)


class AgenticOutput:
    """Stage-0 normalized view of an inferencer result.

    ``raw`` is the orchestrator's native return object (e.g. a
    ``DualInferencerResponse`` or a per-worker tuple); ``text`` is the
    defensively-normalized string (see ``extract_result_text``). Parsers receive
    this object so they can reach either the structured ``raw`` or the flat
    ``text``.
    """

    def __init__(self, raw: Any, text: str) -> None:
        self.raw = raw
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"AgenticOutput(text={self.text!r})"

    def parse_block(self, label: str) -> Optional[dict[str, Any]]:
        """Plain labeled-fence decode: a dict, or ``None`` if absent/invalid."""
        return extract_json_block(self.text, label)

    def json(self, label: Optional[str] = None) -> Any:
        """Hardened JSON decode; raises :class:`ParseError` on any problem.

        With ``label`` the body of a ```json <label> fence is decoded; without it
        the whole normalized text is. Duplicate keys and non-finite constants
        (``NaN``/``Infinity``) are rejected, so a malformed or injected reply
        fails closed instead of silently decoding.
        """
        if label is not None:
            raw = find_json_block_text(self.text, label)
            if raw is None:
                raise ParseError(f"no ```json {label} block found")
            source = raw
        else:
            source = self.text
        try:
            return json.loads(
                source,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
        except (json.JSONDecodeError, ValueError) as e:
            where = f" in {label} block" if label is not None else ""
            raise ParseError(f"invalid JSON{where}: {e}") from e


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key: {key!r}")
        seen[key] = value
    return seen


def _reject_nonfinite(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant not allowed: {token}")


class _FromInference:
    """Sentinel default for a mixed body's ``response`` slot.

    On pass-1 (pre-attempt) the slot holds this sentinel — "not yet inferred, run
    the deterministic branch". On pass-2 (parser) the slot holds the stage-1
    output. A body distinguishes them with ``response is FromInference``.
    Constructed once (singleton) so the identity check is stable.
    """

    _instance: "Optional[_FromInference]" = None

    def __new__(cls) -> "_FromInference":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "FromInference"


FromInference: _FromInference = _FromInference()


@attr.s(auto_attribs=True, frozen=True)
class Agentic:
    """Escalation carrier: enrich the escalated prompt and/or pick a parser.

    A body returns this (or the bare :data:`ESCALATE` singleton) to say "I could
    not answer deterministically — run inference". ``feed`` is merged into the
    template feed for this call; ``parser`` overrides the decorator's stage-1
    ``parser=`` for this call only. ``parser`` is typed ``Any`` to avoid a cycle
    with :mod:`parsers`; the pipeline resolves it the same way as ``parser=``.
    """

    feed: Mapping[str, Any] = attr.Factory(dict)
    parser: Any = None


ESCALATE: Agentic = Agentic()


def escalate(*, feed: Optional[Mapping[str, Any]] = None, parser: Any = None) -> Any:
    """Typed escape hatch returning ``Any``.

    ``return escalate(...)`` type-checks in a body with ANY declared return type
    (a strictly-typed ``-> int`` body cannot ``return ESCALATE``, which is an
    :class:`Agentic`). Carries an optional ``feed`` and a per-call ``parser``
    override.
    """
    return Agentic(feed=feed or {}, parser=parser)
