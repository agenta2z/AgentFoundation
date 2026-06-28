

"""Conversation tool data models.

Defines the structured types for conversation tools that the LLM can invoke
to interact with the user: clarification, single/multiple choice, confirmation,
and tool argument collection.

Author-dialect canonicalisation (hyphenated keys, ``single-choice`` tool names,
string ``output``) lives here in :func:`canonicalize_tool_data` /
:func:`normalize_tool_type` so that BOTH the parser (ToolsToInvoke path) and
``from_dict`` (legacy ``<ConversationTools>`` path) collapse to one canonical
internal schema — a single point of truth, not scattered string fixes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConversationToolType(str, Enum):
    CLARIFICATION = "clarification"
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    CONFIRMATION = "confirmation"
    TOOL_ARGUMENT_FORM = "tool_argument_form"
    # Structured multi-item selection over a ranked/grouped proposal set
    # (richer than multiple_choice: batch/phase grouping, dependencies,
    # impact/complexity). Used by research-propose → Phase 3b review flows.
    PROPOSAL_SELECTION = "proposal_selection"


# --- Author-dialect canonicalisation --------------------------------------

# Tool-type name aliases applied AFTER separators are normalised to underscore.
_TOOL_TYPE_ALIASES: dict[str, str] = {
    "multiple_choices": "multiple_choice",
    "single_choices": "single_choice",
}


def normalize_tool_type(name: Any) -> Any:
    """Normalise a tool-type / tool name to its canonical underscore form.

    Accepts SOP-author dialect: ``single-choice``, ``single choice`` and
    ``single_choices`` all map to ``single_choice``. Non-str passes through.
    """
    if not isinstance(name, str):
        return name
    norm = name.strip().replace("-", "_").replace(" ", "_")
    return _TOOL_TYPE_ALIASES.get(norm, norm)


def _canon_keys(d: Any) -> Any:
    """Rename hyphenated keys to underscore form (e.g. ``expected-input-type``
    → ``expected_input_type``). An explicit underscore key wins when both forms
    are present, so a caller's deliberate underscore is never clobbered.
    """
    if not isinstance(d, dict):
        return d
    out: dict[str, Any] = {}
    for k, v in d.items():
        nk = k.replace("-", "_") if isinstance(k, str) else k
        if nk != k and nk in d:
            continue  # underscore form present elsewhere → it wins
        out[nk] = v
    return out


def _canon_choice(choice: Any) -> Any:
    """Canonicalise a single choice dict, including a nested ``input`` spec."""
    if not isinstance(choice, dict):
        return choice
    choice = _canon_keys(choice)
    inp = choice.get("input")
    if isinstance(inp, dict):
        choice["input"] = _canon_keys(inp)
    return choice


def _coerce_output_list(data: dict[str, Any]) -> None:
    """Coerce a string ``output`` / ``output_vars`` to a list, in place.

    Fixes the latent bug where ``output: "x"`` would later be indexed as
    ``output_vars[0] == "x"[0] == "x"`` / iterated character-by-character.
    """
    for key in ("output", "output_vars"):
        val = data.get(key)
        if isinstance(val, str):
            data[key] = [val] if val else []


def _coerce_choices_list(data: dict[str, Any]) -> None:
    """Coerce a JSON-string ``choices`` to a list, in place.

    Same class of bug as :func:`_coerce_output_list`: the LLM sometimes emits
    ``choices`` as a JSON-encoded STRING (e.g. ``'[{"label": "Auto", ...}]'``)
    instead of an array. Left as a string, the downstream ``for c in choices``
    iterates CHARACTERS — producing one bogus single-character option per
    character (the "every-character-is-a-button" widget bug). Parse it back to a
    list; anything that isn't valid JSON decoding to a list collapses to ``[]``.
    """
    raw = data.get("choices")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        data["choices"] = parsed if isinstance(parsed, list) else []


def coerce_parallel_group(raw: Any) -> int | None:
    """Leniently coerce a raw ``parallel_group`` value to ``int | None``.

    NEVER raises. A genuine ``int`` (rejecting ``bool`` — ``True``/``False`` are
    ints in Python, but never a valid group id) is accepted as-is. Any other
    shape (``None``, ``str``, ``float``, ``bool``, etc.) yields ``None`` so a
    malformed marker degrades to "ungrouped" rather than corrupting grouping.
    Callers that need to surface the bad value record it in metadata.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    return None


def canonicalize_tool_data(data: dict[str, Any]) -> dict[str, Any]:
    """Collapse SOP-author dialect into the canonical conversation-tool schema.

    Handles both wire shapes:
      * ToolsToInvoke: ``{type, name, arguments:{…, choices:[{…, input:{…}}]}, output}``
      * legacy / flat: ``{tool_type, …, choices:[{…, input:{…}}], output}``

    Renames hyphenated keys to underscore (over ``arguments`` / ``choices`` /
    nested ``input``) and coerces a string ``output``/``output_vars`` to a list.
    Idempotent — safe to call more than once.
    """
    if not isinstance(data, dict):
        return data
    data = _canon_keys(dict(data))
    _coerce_output_list(data)

    args = data.get("arguments")
    if isinstance(args, dict):
        args = _canon_keys(args)
        _coerce_choices_list(args)  # stringified choices → list (else char-iterated)
        if isinstance(args.get("choices"), list):
            args["choices"] = [_canon_choice(c) for c in args["choices"]]
        data["arguments"] = args

    # Flat / legacy choices at the top level.
    _coerce_choices_list(data)  # stringified choices → list (else char-iterated)
    if isinstance(data.get("choices"), list):
        data["choices"] = [_canon_choice(c) for c in data["choices"]]

    return data


# --- Models ----------------------------------------------------------------


@dataclass
class InputFieldSpec:
    """A typed input field — used standalone (tool-level) or embedded in a choice.

    When set on a :class:`ChoiceItem`, selecting that choice reveals this input
    and the entered value is bound to ``name`` (distinct from the tool-level
    output, which records *which* choice was selected). ``serialization``
    controls how a collected value is published as an output variable
    (see ``conversation_tool_runtime.finalize_input_value``).
    """

    name: str = ""                          # output variable for the entered value
    expected_input_type: str = "free_text"  # "free_text" | "path" | "url"
    prefix: str = ""                        # base dir for path inputs (autocomplete root)
    allow_multiple_input: bool = False      # collect one-or-more values
    required: bool = False
    placeholder: str = ""
    label: str = ""
    description: str = ""
    serialization: str = "auto"             # auto | scalar | json | comma
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.name:
            d["name"] = self.name
        if self.expected_input_type and self.expected_input_type != "free_text":
            d["expected_input_type"] = self.expected_input_type
        if self.prefix:
            d["prefix"] = self.prefix
        if self.allow_multiple_input:
            d["allow_multiple_input"] = True
        if self.required:
            d["required"] = True
        if self.placeholder:
            d["placeholder"] = self.placeholder
        if self.label:
            d["label"] = self.label
        if self.description:
            d["description"] = self.description
        if self.serialization and self.serialization != "auto":
            d["serialization"] = self.serialization
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputFieldSpec:
        data = _canon_keys(data or {})
        return cls(
            name=data.get("name", ""),
            expected_input_type=data.get("expected_input_type", "free_text"),
            prefix=data.get("prefix", ""),
            allow_multiple_input=bool(data.get("allow_multiple_input", False)),
            required=bool(data.get("required", False)),
            placeholder=data.get("placeholder", ""),
            label=data.get("label", ""),
            description=data.get("description", ""),
            serialization=data.get("serialization", "auto"),
            metadata=data.get("metadata", {}) or {},
        )


@dataclass
class ChoiceItem:
    """A single choice option for single/multiple choice tools.

    A choice may carry an embedded typed ``input``. When present, selecting this
    choice reveals that input and binds the entered value to ``input.name``.
    """

    label: str
    value: str
    description: str = ""
    input: InputFieldSpec | None = None

    @property
    def has_input(self) -> bool:
        return self.input is not None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"label": self.label, "value": self.value}
        if self.description:
            d["description"] = self.description
        if self.input is not None:
            d["input"] = self.input.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChoiceItem:
        data = _canon_keys(data or {})
        inp = data.get("input")
        return cls(
            label=data.get("label", ""),
            value=data.get("value", ""),
            description=data.get("description", ""),
            input=InputFieldSpec.from_dict(inp) if isinstance(inp, dict) else None,
        )


@dataclass
class ConversationTool:
    """A conversation tool invocation parsed from the LLM response.

    Represents the LLM's request to interact with the user in a structured
    way (ask a question, present choices, collect form input, etc.).
    """

    tool_type: str  # One of ConversationToolType constants
    prompt: str = ""
    choices: list[ChoiceItem] = field(default_factory=list)
    allow_custom: bool = True
    expected_input_type: str = "free_text"  # "free_text" | "path" | "url"
    prefix: str = ""  # Path prefix for path input mode
    allow_multiple_input: bool = False  # Standalone multi-value (e.g. multi-path clarification)
    serialization: str = "auto"  # auto | scalar | json | comma (publication format)
    tool_name: str = ""  # For tool_argument_form: which tool
    fields: list[dict[str, Any]] = field(default_factory=list)  # For tool_argument_form
    output_vars: list[str] = field(default_factory=list)  # Variable names to capture
    metadata: dict[str, Any] = field(default_factory=dict)
    # Multiple-choice "Select All" control — passed through to InputModeConfig
    show_select_all: bool = True        # show "All of above" toggle (default: True)
    select_all_text: str = "All of above"  # customisable label
    # Parallel-execution group id (round-lifecycle core). Tools sharing the same
    # non-None value form one group; None = ungrouped. Lenient-parsed (never
    # raises): a bad/str/float/bool value collapses to None and is recorded in
    # ``metadata['parallel_group_invalid']``.
    parallel_group: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tool_type": self.tool_type}
        if self.prompt:
            d["prompt"] = self.prompt
        if self.choices:
            d["choices"] = [c.to_dict() for c in self.choices]
        if not self.allow_custom:
            d["allow_custom"] = False
        if self.expected_input_type and self.expected_input_type != "free_text":
            d["expected_input_type"] = self.expected_input_type
        if self.prefix:
            d["prefix"] = self.prefix
        if self.allow_multiple_input:
            d["allow_multiple_input"] = True
        if self.serialization and self.serialization != "auto":
            d["serialization"] = self.serialization
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.fields:
            d["fields"] = self.fields
        if self.output_vars:
            d["output_vars"] = list(self.output_vars)
        if self.metadata:
            d["metadata"] = self.metadata
        # Only serialise non-default select-all values
        if not self.show_select_all:
            d["show_select_all"] = False
        if self.select_all_text != "All of above":
            d["select_all_text"] = self.select_all_text
        if self.parallel_group is not None:
            d["parallel_group"] = self.parallel_group
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTool:
        data = canonicalize_tool_data(data)
        choices = [
            ChoiceItem.from_dict(c) for c in data.get("choices", [])
        ]
        # Lenient parallel_group parse (never raises): bad value → None, with a
        # structured marker recorded in metadata for diagnosis.
        metadata = dict(data.get("metadata", {}) or {})
        parallel_group = None
        if "parallel_group" in data:
            raw_pg = data["parallel_group"]
            parallel_group = coerce_parallel_group(raw_pg)
            if parallel_group is None and raw_pg is not None:
                metadata["parallel_group_invalid"] = raw_pg
        return cls(
            tool_type=normalize_tool_type(data.get("tool_type", "")),
            prompt=data.get("prompt", ""),
            choices=choices,
            allow_custom=data.get("allow_custom", True),
            expected_input_type=data.get("expected_input_type", "free_text"),
            prefix=data.get("prefix", ""),
            allow_multiple_input=bool(data.get("allow_multiple_input", False)),
            serialization=data.get("serialization", "auto"),
            tool_name=data.get("tool_name", ""),
            fields=data.get("fields", []),
            output_vars=data.get("output_vars", data.get("output", [])),
            metadata=metadata,
            show_select_all=data.get("show_select_all", True),
            select_all_text=data.get("select_all_text", "All of above"),
            parallel_group=parallel_group,
        )
