# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Pure-function helpers that decide whether a combo can run via the hub's
``${ENABLE_FLAGS}``-based composition (Model A) versus needing a per-combo
gin file (Model B). Sits beside ``hypothesis_implementations`` so the gate
in ``combo_overrides_service`` can fan out by capability without bolting
new concerns onto either the route layer or the implementation-derivation
layer.

Three pure helpers:

  * :func:`hub_supports_flag_composition` — does the hub's launch.json
    even substitute ``${ENABLE_FLAGS}``? (Capability check.)
  * :func:`combo_flag_bindings` — for a combo whose member Hs are all
    implemented, do their hypothesisFlagMap entries resolve to scoped
    gin bindings (e.g. ``hstu_encoder.enable_h17``)? (Quality check.)
  * :func:`hub_flag_map_has_identity` — list any hypothesis IDs whose
    flag-map value is the identity placeholder (``H1 -> "H1"``); used by
    the hub-wide "fix the flag map" banner.

Defense-in-depth context: the runner's overlay writer raises ``RuntimeError``
on bare tokens AND post-parse-verifies bindings. The gate that consumes these
helpers is the FIRST line of defense — it catches bad combos before
submission. The runner remains the LAST line of defense if a combo is somehow
submitted out-of-band.

Ported faithfully from RankEvolve's ``webui.backend.services.combo_capability``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# Strict regex for a scoped gin binding: ``<configurable>.<param>``. Both
# segments must be valid Python identifiers. Identity values like ``"H1"``,
# bare names like ``"enable_h17"``, empty strings, and missing entries all
# fail this match — and are correctly rejected by the runner if they ever
# reach it.
_SCOPED_BINDING_RE: re.Pattern[str] = re.compile(r"^[A-Za-z_][\w]*\.[A-Za-z_][\w]*$")

# Token the runner substitutes at launch time. Hub launch.json files whose
# ``script_args`` contain this literal expect a comma-joined enable_flags
# string at runtime — i.e., the hub IS Model A (flag composition).
_ENABLE_FLAGS_TOKEN: str = "${ENABLE_FLAGS}"


def hub_supports_flag_composition(
    setup_state: dict[str, Any] | None,
    session_dir: Path,
) -> bool:
    """Return True iff the hub's launch.json substitutes ``${ENABLE_FLAGS}``.

    Reads ``setup_state["launchPath"]`` from disk. Returns False (NOT raising)
    on: ``setup_state`` is None/empty; ``launchPath`` missing or not a string;
    file doesn't exist or isn't readable; file isn't valid JSON or lacks a
    ``script_args`` array.

    Returning False on any of these means the gate falls through to legacy
    ``configStatus``-based derivation — backwards-compatible by construction.

    Pure function — no side effects beyond diagnostic log lines.
    """
    if not setup_state:
        return False
    launch_path_str = setup_state.get("launchPath")
    if not isinstance(launch_path_str, str) or not launch_path_str:
        return False
    # session_dir is provided so callers keep a uniform interface even if a
    # future variant of launchPath uses session-relative paths. We accept
    # absolute paths as-is (the writer records absolute paths).
    _ = session_dir
    launch_path = Path(launch_path_str)
    if not launch_path.is_file():
        return False
    try:
        launch_doc = json.loads(launch_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(
            "hub_supports_flag_composition: failed to read/parse %s: %s",
            launch_path,
            e,
        )
        return False
    script_args = launch_doc.get("script_args")
    if not isinstance(script_args, list):
        return False
    return any(isinstance(a, str) and _ENABLE_FLAGS_TOKEN in a for a in script_args)


def combo_flag_bindings(
    combo: dict[str, Any],
    flag_map: dict[str, str],
) -> tuple[bool, list[str], list[str]]:
    """Return ``(all_scoped, scoped_bindings, bare_or_missing)`` for a combo.

    Walks ``combo["selectedItems"]`` in order. For each H ID, looks up
    ``flag_map[H]``. A binding counts as "scoped" iff its value matches
    :data:`_SCOPED_BINDING_RE` (e.g., ``hstu_encoder.enable_h1``). Identity
    values, bare names (``enable_h17`` with no scope), missing entries, empty
    strings, and non-string values all count as bare-or-missing.

    Returns:
      ``(True, [scoped, ...], [])`` when every member resolves to a scoped
      binding — the combo is safe to submit via flag composition.

      ``(False, [scoped, ...], [bare, ...])`` otherwise. The caller (the gate)
      treats this as ``applyState = "pending_flag_map"``. Both lists are
      preserved for diagnostic UX.
    """
    scoped: list[str] = []
    bare: list[str] = []
    for h in combo.get("selectedItems") or []:
        if not isinstance(h, str) or not h:
            continue
        value = flag_map.get(h)
        if (
            not isinstance(value, str)
            or not value
            or not _SCOPED_BINDING_RE.match(value)
        ):
            bare.append(h)
        else:
            scoped.append(value)
    all_scoped = not bare
    return all_scoped, scoped, bare


def hub_flag_map_has_identity(flag_map: dict[str, str]) -> list[str]:
    """Return the list of H IDs whose ``flag_map`` value is the identity
    placeholder (e.g., ``"H1" -> "H1"``).

    Used by the hub-wide UI banner that warns the user once per page about a
    misconfigured flag map, regardless of which combos are in play. An empty
    return means no identity entries -> no banner shown.

    The check is strict equality; bare-name entries like ``"H1" -> "enable_h1"``
    are NOT considered identity (they're bare, a separate problem caught by
    :func:`combo_flag_bindings`).
    """
    out: list[str] = []
    if not isinstance(flag_map, dict):
        return out
    for k, v in flag_map.items():
        if isinstance(k, str) and isinstance(v, str) and k == v:
            out.append(k)
    return out
