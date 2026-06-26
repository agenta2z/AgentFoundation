"""Output parsers for MultiFlow / MultiFlowDual aggregator + per-flow outputs.

Four small functions, paired with the JSON-fenced response-format addenda
emitted by ``plan/main/_variables/task_response_format/aggregation.jinja2``:

* :func:`parse_winner_tag` — used as ``MultiFlowDual.multi_flow_winner_parser``
  to extract the winning flow index from an aggregator's structured output.
* :func:`parse_decision_stop` — used as a per-flow ``end_condition`` callable
  on each ``flow_configs`` entry, signaling the flow has decided no further
  iteration adds value.
* :func:`parse_finalplan_tag` — used as ``MultiFlowDual.multi_flow_response_parser``
  to extract the final plan content (stripping any wrapper tags) so the
  outer Dual reviewer sees clean prose.
* :func:`parse_ranking_tag` — used as ``MultiFlowDual.multi_flow_ranking_parser``
  to extract the ordered flow ranking from an aggregator's structured output.

Each parser first looks for a JSON-fenced block matching the response-format
template, then falls back to the legacy XML-tag pattern so older outputs and
hand-written test fixtures still parse.

These exist as YAML-instantiable callables via the alias registry — see
``agent_foundation.common.configs.registered_targets`` for ``WinnerParser``,
``DecisionStopParser``, ``FinalPlanParser``, ``RankingParser`` aliases.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional


# Legacy XML tag patterns (fallback path for older outputs / fixtures).
_WINNER_RE = re.compile(r"<Winner>\s*flow_(\d+)\s*</Winner>", re.IGNORECASE)
_DECISION_RE = re.compile(r"<Decision>\s*([\s\S]*?)\s*</Decision>", re.IGNORECASE)
_FINALPLAN_RE = re.compile(r"<FinalPlan>([\s\S]*?)</FinalPlan>", re.IGNORECASE)

# Matches ```json <label> ... ``` (label may be followed by extra text on
# the fence line; we capture the body between the fences).
_JSON_FENCE_TEMPLATE = r"```json\s+{label}\b[^\n]*\n([\s\S]*?)\n\s*```"

_STOP_TOKENS = {"stop", "done", "halt"}


def _extract_json_block(s: str, label: str) -> Optional[dict]:
    """Find ```json <label> ... ``` and parse the body as JSON.

    Returns the decoded dict, or ``None`` if the block is absent or invalid.
    """
    pattern = re.compile(_JSON_FENCE_TEMPLATE.format(label=re.escape(label)))
    m = pattern.search(s)
    if not m:
        return None
    try:
        decoded = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def parse_winner_tag(s: Any) -> Optional[int]:
    """Extract the winning flow index from the aggregator output.

    Primary path: ```json winner_pick`` block with ``"winner_index": K``.
    Fallback: legacy ``<Winner>flow_K</Winner>`` tag.
    Returns ``None`` if neither is present (graceful — dispatch falls back
    to defaults).
    """
    if not isinstance(s, str):
        return None
    block = _extract_json_block(s, "winner_pick")
    if block is not None:
        idx = block.get("winner_index")
        if isinstance(idx, bool):  # bool is subclass of int — exclude
            idx = None
        if isinstance(idx, int):
            return idx
        if isinstance(idx, str) and idx.strip().lstrip("-").isdigit():
            return int(idx.strip())
    m = _WINNER_RE.search(s)
    return int(m.group(1)) if m else None


def parse_decision_stop(state: Any, result: Any) -> bool:
    """End-condition callable: ``True`` iff the model decided to stop iterating.

    Signature matches ``LinearWorkflowInferencer.end_condition`` (the
    ``(state, result)`` callable). Reads from ``result`` only — ``state``
    is ignored but accepted for the standard end_condition interface.

    Primary path: ```json iteration_judgment`` block with
    ``"decision": "stop"`` (case-insensitive; ``done``/``halt`` also accepted).
    Fallback: legacy ``<Decision>stop</Decision>`` tag.
    Anything else (including ``continue``, missing block, malformed JSON)
    → ``False`` (keep iterating).
    """
    text = result if isinstance(result, str) else getattr(result, "output", None)
    if not isinstance(text, str):
        text = str(result) if result is not None else ""
    block = _extract_json_block(text, "iteration_judgment")
    if block is not None:
        decision = block.get("decision")
        if isinstance(decision, str):
            return decision.strip().lower() in _STOP_TOKENS
        return False
    m = _DECISION_RE.search(text)
    if not m:
        return False
    return m.group(1).strip().lower() in _STOP_TOKENS


def parse_finalplan_tag(s: Any) -> Optional[str]:
    """Extract content from ``<FinalPlan>...</FinalPlan>`` with prompt-echo defense.

    Falls back to the original string when no tag is present (graceful — the
    aggregator output is used verbatim if it didn't wrap properly). Used as
    ``MultiFlowDual.multi_flow_response_parser`` so the outer Dual reviewer
    receives the cleaned plan body.

    Three return modes (matching ``DualInferencer._default_response_parser``):
    1. No matches        -> str(s) unchanged (passthrough).
    2. >=1 clean matches -> most-recent clean match's content.
    3. All matches echo  -> None (hard-failure signal).

    See forensics_round_template_echo_bug for the originating incident.
    """
    from agent_foundation.common.response_parsers.delimiter_parser import (
        _PROMPT_ECHO_MARKERS,
    )

    text = str(s) if not isinstance(s, str) else s
    matches = list(_FINALPLAN_RE.finditer(text))
    if not matches:
        return text
    for match in reversed(matches):
        content = match.group(1).strip()
        if all(marker in content for marker in _PROMPT_ECHO_MARKERS):
            continue
        return content
    return None


def parse_ranking_tag(s: Any) -> Optional[List[int]]:
    """Extract the flow ranking from the aggregator output.

    Primary path: ````` json ranking`` block with ``"ranking": [0, 1, ...]``.
    Returns a list of 0-based flow indices ordered best-to-worst, or ``None``
    if the block is absent or invalid (graceful — dispatch falls back to
    declaration order).
    """
    if not isinstance(s, str):
        return None
    block = _extract_json_block(s, "ranking")
    if block is not None:
        ranking = block.get("ranking")
        if isinstance(ranking, list):
            result: List[int] = []
            for idx in ranking:
                if isinstance(idx, bool):
                    continue
                if isinstance(idx, int):
                    result.append(idx)
                elif isinstance(idx, str) and idx.strip().isdigit():
                    result.append(int(idx.strip()))
            if result:
                return result
    return None


# ---------------------------------------------------------------------------
# Factory wrappers for YAML alias registration.
#
# YAML's ``_target_:`` resolves to a class or function and instantiates it
# with the constructor kwargs. Since our parsers are plain functions (not
# classes), we register tiny factories that return the function so the YAML
# can do ``_target_: WinnerParser`` and end up with the function as the
# inferencer's ``end_condition`` / ``winner_parser`` / ``response_parser``.
# ---------------------------------------------------------------------------


def make_winner_parser():
    """Factory returning :func:`parse_winner_tag`."""
    return parse_winner_tag


def make_decision_stop_parser():
    """Factory returning :func:`parse_decision_stop`."""
    return parse_decision_stop


def make_finalplan_parser():
    """Factory returning :func:`parse_finalplan_tag`."""
    return parse_finalplan_tag


def make_ranking_parser():
    """Factory returning :func:`parse_ranking_tag`."""
    return parse_ranking_tag


# Severity ordering for the "never downgrade" merge semantics (§3 Part B).
_SEVERITY_ORDER = {
    "info": 0, "nit": 0, "low": 1, "minor": 1, "medium": 2, "moderate": 2,
    "high": 3, "major": 3, "critical": 4, "blocker": 4,
}


def _normalize_desc(s: Any) -> str:
    """Whitespace/case-normalized description for dedup keys."""
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def merge_reviews(parsed_reviews: List[Any]) -> dict:
    """§3 Part B — deterministically merge N parsed reviews into one review dict.

    Each input is a review dict whose ``issues`` (or ``reviews``) list holds items
    carrying ``location``/``description``/``severity``. The union is **deduped by
    ``(location, normalized description)``**; on a duplicate the **higher severity
    wins (never downgrade)** and ``agreement_count`` is incremented. Output is the
    same review JSON schema (``{"issues": [...]}``) so the existing
    ``_default_check_consensus`` stays correct. Deterministic (first-seen order).
    """
    by_key: dict = {}
    order: list = []
    for review in parsed_reviews or []:
        if not isinstance(review, dict):
            continue
        issues = review.get("issues") or review.get("reviews") or []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            key = (str(issue.get("location", "")), _normalize_desc(issue.get("description")))
            sev = str(issue.get("severity", "medium")).lower()
            if key in by_key:
                existing = by_key[key]
                existing["agreement_count"] = existing.get("agreement_count", 1) + 1
                if _SEVERITY_ORDER.get(sev, 2) > _SEVERITY_ORDER.get(
                    str(existing.get("severity", "medium")).lower(), 2
                ):
                    existing["severity"] = issue.get("severity", existing.get("severity"))
            else:
                merged = dict(issue)
                merged["agreement_count"] = 1
                by_key[key] = merged
                order.append(key)
    return {"issues": [by_key[k] for k in order]}
