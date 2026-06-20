

"""Shared runtime helpers for conversation tools.

ONE source of truth for turning a parsed :class:`ConversationTool` + a UI
response into (a) rendered runtime fields, (b) a validated/serialised published
value, and (c) distinct ``{output_var: value}`` bindings.

Used by the live inline ``ConversationalInferencer`` path — both the single-tool
and compound branches — so the two never diverge (a composite choice binds the
same way whether presented alone or inside a compound widget).

Key concepts
------------
* **Render** templated ``prefix`` fields after parsing (the LLM may echo a raw
  ``{{ session_root_path }}`` that the SOP guidance render did not resolve).
* **Finalize** one collected value: path re-join + explicit serialisation. Never
  ``str(list)`` — a multi value publishes as a reversible JSON array string by
  default (``serialization`` can request ``comma``/``scalar``/``json``).
* **Bindings**: a composite choice produces TWO distinct variables — the
  selected choice value (mode) on the tool's ``output_vars`` AND the entered
  value on the choice's ``InputFieldSpec.name``. Non-composite tools keep the
  legacy single-value aliasing across all declared output vars.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
)

logger = logging.getLogger(__name__)


# --- runtime field rendering (templated prefix) ----------------------------


def render_templated_fields(
    tool: ConversationTool, render: Optional[Callable[[str], str]]
) -> ConversationTool:
    """Render Jinja-templated string fields (``prefix``) on a tool, in place.

    ``render`` maps a template string to its rendered form (already bound to the
    session context); ``None`` skips rendering, leaving an unrendered ``{{ … }}``
    untouched so backend validation fails clearly rather than treating a Jinja
    expression as a filesystem path. Idempotent on non-template strings.
    """
    if render is None:
        return tool

    def _r(s: Any) -> Any:
        if isinstance(s, str) and "{{" in s:
            try:
                return render(s)
            except Exception:
                logger.debug("[conv-runtime] prefix render failed for %.60s", s)
                return s
        return s

    tool.prefix = _r(tool.prefix)
    for c in tool.choices:
        if c.has_input and c.input is not None:
            c.input.prefix = _r(c.input.prefix)
    return tool


# --- path finalisation + serialisation -------------------------------------


def _rejoin_path(value: Any, prefix: str) -> str:
    """Join ``prefix`` + relative value unless the value is absolute / ``~``."""
    v = (value if isinstance(value, str) else str(value)).strip()
    if not v or v.startswith("/") or v.startswith("~"):
        return v
    return str(Path(prefix) / v) if prefix else v


def is_contained(path_str: str, root: str) -> bool:
    """True if ``path_str`` resolves under ``root`` (sibling-prefix safe)."""
    if not root:
        return True
    try:
        Path(os.path.expanduser(path_str)).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _serialize(values: list[str], serialization: str, multiple: bool) -> str:
    if not multiple:
        return values[0] if values else ""
    mode = serialization or "auto"
    if mode == "comma":
        return ",".join(values)
    if mode == "scalar":
        return values[0] if values else ""
    # "json" and "auto" (multi) → reversible, comma-safe JSON array string.
    return json.dumps(values)


def _to_items(raw: Any) -> list[str]:
    """Unwrap a UI value envelope into a flat list of scalar strings."""
    if isinstance(raw, dict):
        for key in ("content", "paths", "value", "values"):
            if key in raw:
                raw = raw[key]
                break
    if isinstance(raw, (list, tuple)):
        return [x if isinstance(x, str) else str(x) for x in raw]
    if raw is None:
        return []
    return [raw if isinstance(raw, str) else str(raw)]


def finalize_input_value(
    raw: Any,
    *,
    expected_input_type: str = "free_text",
    prefix: str = "",
    allow_multiple_input: bool = False,
    serialization: str = "auto",
    session_root: str = "",
    validate: bool = False,
) -> str:
    """Collapse a raw user value into the canonical published string.

    Paths: re-join ``prefix`` for relative entries; when ``validate`` and a
    ``session_root`` is given, reject entries that resolve outside it (raises
    ``ValueError``). Multi-value: serialise per ``serialization`` (``auto`` →
    JSON array string). Never returns ``str(list)``.
    """
    items = _to_items(raw)
    if expected_input_type == "path":
        joined: list[str] = []
        for it in items:
            p = _rejoin_path(it, prefix)
            if validate and session_root and p and not is_contained(p, session_root):
                raise ValueError(
                    f"Path '{p}' is outside the allowed root '{session_root}'"
                )
            joined.append(p)
        items = joined
    return _serialize(items, serialization, allow_multiple_input)


# --- response decode → distinct bindings -----------------------------------


def _bind_all(out_vars: list[str], value: str) -> dict[str, str]:
    """Legacy aliasing: bind one value to every declared output var."""
    return {v: value for v in out_vars}


def decode_tool_bindings(
    tool: ConversationTool,
    response: Any,
    *,
    session_root: str = "",
    validate: bool = False,
) -> dict[str, str]:
    """Decode ONE tool's UI response into distinct ``{output_var: value}`` bindings.

    * ``proposal_selection`` → comma-joined ids (legacy CLI contract preserved).
    * ``confirmation`` → ``{choice}`` value bound to the output vars.
    * single/multiple choice → selected value → output vars (mode); a selected
      choice's embedded ``input`` → its ``input.name`` (distinct value var).
    * clarification / content envelope → finalised (path re-join + serialise).

    Non-composite tools retain the legacy single-value aliasing across all
    declared output vars (so existing multi-``output_vars`` tools are unchanged).
    """
    out_vars = list(tool.output_vars)

    # Bare string / list response → the tool's value.
    if not isinstance(response, dict):
        val = finalize_input_value(
            response,
            expected_input_type=tool.expected_input_type,
            prefix=tool.prefix,
            allow_multiple_input=tool.allow_multiple_input,
            serialization=tool.serialization,
            session_root=session_root,
            validate=validate,
        )
        return _bind_all(out_vars, val) if out_vars else {}

    # proposal_selection — preserve the comma-joined ids contract.
    if tool.tool_type == ConversationToolType.PROPOSAL_SELECTION:
        sel = None
        if isinstance(response.get("selected_proposals"), list):
            sel = response["selected_proposals"]
        elif isinstance(response.get("selected"), list):
            sel = response["selected"]
        elif isinstance(response.get("choice_indices"), list) and tool.choices:
            sel = [
                tool.choices[i].value
                for i in response["choice_indices"]
                if isinstance(i, int) and 0 <= i < len(tool.choices)
            ]
        if sel is not None:
            return _bind_all(out_vars, ",".join(str(s) for s in sel))

    # confirmation — choice value (param_overrides side-effects stay with caller).
    if "choice" in response and "choice_index" not in response:
        return _bind_all(out_vars, str(response["choice"]))

    # multiple_choice — a list of {choice_index|custom_text}; publish comma-joined.
    if isinstance(response.get("selections"), list):
        vals: list[str] = []
        for s in response["selections"]:
            if not isinstance(s, dict):
                continue
            ci = s.get("choice_index")
            if isinstance(ci, int) and tool.choices and 0 <= ci < len(tool.choices):
                vals.append(tool.choices[ci].value)
            elif "custom_text" in s:
                vals.append(str(s["custom_text"]))
        return _bind_all(out_vars, ",".join(vals))

    # single / multiple choice (incl. composite with embedded input).
    if "choice_index" in response or "custom_text" in response or "inputs" in response:
        bindings: dict[str, str] = {}
        idx = response.get("choice_index")
        selected: Optional[ChoiceItem] = None
        if isinstance(idx, int) and tool.choices and 0 <= idx < len(tool.choices):
            selected = tool.choices[idx]
            choice_value = response.get("choice_value") or selected.value
        else:
            choice_value = response.get("custom_text", "")
        for v in out_vars:
            bindings[v] = choice_value
        if selected is not None and selected.has_input and selected.input is not None:
            spec = selected.input
            inputs = response.get("inputs")
            raw: Any = None
            if isinstance(inputs, dict):
                raw = inputs.get(spec.name, next(iter(inputs.values()), None))
            elif "content" in response:
                raw = response.get("content")
            val = finalize_input_value(
                raw,
                expected_input_type=spec.expected_input_type,
                prefix=spec.prefix,
                allow_multiple_input=spec.allow_multiple_input,
                serialization=spec.serialization,
                session_root=session_root,
                validate=validate,
            )
            if spec.name:
                bindings[spec.name] = val
        return bindings

    # clarification / content envelope (single or multi path / free text).
    val = finalize_input_value(
        response,
        expected_input_type=tool.expected_input_type,
        prefix=tool.prefix,
        allow_multiple_input=tool.allow_multiple_input,
        serialization=tool.serialization,
        session_root=session_root,
        validate=validate,
    )
    return _bind_all(out_vars, val) if out_vars else {"input": val}


def primary_output_key(tool: ConversationTool) -> str:
    """The key a compound widget stores this tool's child payload under."""
    return tool.output_vars[0] if tool.output_vars else tool.tool_type


# --- parallel-group partitioning + validation ------------------------------


class GroupValidationError(Exception):
    """Raised when a response's parallel_group layout is invalid.

    See :func:`group_and_validate` for the exact invariants. A single-member
    group is always allowed; the rules only constrain multi-member groups and
    the overall set of distinct group ids in one response.
    """


# Output vars that may NOT belong to a grouped (parallel) tool — they name the
# generic compound/clarification sinks the round-lifecycle core reserves.
_RESERVED_GROUPED_OUTPUT_VARS = frozenset({"values", "user_input"})


def group_and_validate(
    tools: list[ConversationTool],
) -> list[list[ConversationTool]]:
    """Partition ``tools`` into parallel groups and validate the layout.

    Partitioning (preserves order):
      * Consecutive tools sharing the same non-None ``parallel_group`` form one
        group.
      * Consecutive ungrouped (``parallel_group is None``) tools coalesce into
        one implicit group; an ungrouped run only coalesces with adjacent
        ungrouped tools (a grouped tool breaks the run).

    Validation (raises :class:`GroupValidationError`):
      a. A side-effecting tool — ``ConversationToolType.CONFIRMATION`` or
         ``PROPOSAL_SELECTION`` — may not sit inside a multi-member group.
      b. No two tools in one group may share a ``primary_output_key`` (reuses
         :func:`primary_output_key`).
      c. A grouped tool's primary output var may not be in
         ``{"values", "user_input"}``.
      d. At most ONE distinct non-None ``parallel_group`` value may appear
         across the whole response.

    Single-member groups are always allowed (rules a/c apply only when a group
    has more than one member; d/b still apply across/within groups). Returns the
    ordered list of groups.
    """
    # Partition into consecutive runs (same non-None id, or a run of None).
    groups: list[list[ConversationTool]] = []
    prev_key: Any = object()  # sentinel that never equals a real key
    for tool in tools:
        pg = tool.parallel_group
        # Key identity: each distinct non-None id is its own run; None always
        # uses the same key so consecutive ungrouped tools coalesce.
        key: Any = ("none",) if pg is None else ("group", pg)
        if groups and key == prev_key:
            groups[-1].append(tool)
        else:
            groups.append([tool])
        prev_key = key

    # (d) At most one DISTINCT non-None parallel_group across the response.
    distinct_groups = {
        t.parallel_group for t in tools if t.parallel_group is not None
    }
    if len(distinct_groups) > 1:
        raise GroupValidationError(
            "more than one distinct parallel_group present in the response: "
            f"{sorted(distinct_groups)}"
        )

    for group in groups:
        if len(group) <= 1:
            continue  # single-member groups are always allowed
        # (a) no side-effecting tool inside a multi-member group.
        for tool in group:
            if tool.tool_type in (
                ConversationToolType.CONFIRMATION,
                ConversationToolType.PROPOSAL_SELECTION,
            ):
                raise GroupValidationError(
                    f"side-effecting tool '{tool.tool_type}' cannot be a member "
                    "of a multi-tool parallel group"
                )
        # (b) no duplicate primary_output_key within a group, and
        # (c) no reserved primary output var within a MULTI-member group (the
        # direct-map compound payload keys by it, so "values"/"user_input" would
        # be misread by the backend envelope-unwrap; a single-member group never
        # uses the direct-map, so it's exempt).
        seen_keys: set[str] = set()
        for tool in group:
            key = primary_output_key(tool)
            if key in _RESERVED_GROUPED_OUTPUT_VARS:
                raise GroupValidationError(
                    f"grouped tool may not use reserved output var '{key}'"
                )
            if key in seen_keys:
                raise GroupValidationError(
                    f"duplicate primary output key '{key}' within a parallel group"
                )
            seen_keys.add(key)

    return groups


def decode_compound_bindings(
    tools: list[ConversationTool],
    values: dict,
    *,
    session_root: str = "",
    validate: bool = False,
) -> dict[str, str]:
    """Decode a compound widget's ``{output_var: child_payload}`` into merged bindings.

    Reads each child payload by the tool's primary output key, then routes it
    through :func:`decode_tool_bindings` so a composite child's nested ``inputs``
    are bound (not stringified as ``"{'choice_index': 1, …}"``).
    """
    merged: dict[str, str] = {}
    for tool in tools:
        key = primary_output_key(tool)
        if key not in values:
            # Untouched tool — do NOT clobber prior/default variable state with
            # "". (A present-but-empty value still decodes, so a user who
            # explicitly cleared a field can still publish "".)
            continue
        merged.update(
            decode_tool_bindings(
                tool, values[key], session_root=session_root, validate=validate
            )
        )
    return merged
