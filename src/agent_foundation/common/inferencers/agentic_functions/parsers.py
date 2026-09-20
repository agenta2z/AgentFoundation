# pyre-strict

"""Stage-1 parser resolution and return-annotation default coercers.

Stage-1 parsers (the decorator ``parser=`` and an ``Agentic.parser`` override)
receive the :class:`AgenticOutput` and return a decoded value. This module
resolves the polymorphic ``parser=`` spec (callable | registry alias | dotted
path | sequence) and, when neither a ``parser=`` nor a body-as-parser runs,
supplies the default keyed on the function's declared return type.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from typing import Any, Callable, List, Mapping, Optional

from agent_foundation.common.inferencers.agentic_functions.config import _import_symbol
from agent_foundation.common.inferencers.agentic_functions.errors import (
    AgenticFunctionConfigurationError,
    ParseError,
)
from agent_foundation.common.inferencers.agentic_functions.output import AgenticOutput

Parser = Callable[[AgenticOutput], Any]

_BOOL_TRUE = frozenset({"true", "yes", "1"})
_BOOL_FALSE = frozenset({"false", "no", "0"})


def resolve_parser(spec: Any) -> Parser:
    """Turn a ``parser=`` / ``Agentic.parser`` spec into a single callable.

    A sequence is composed **right-to-left** (``compose`` semantics): the LAST
    element receives the :class:`AgenticOutput` first, e.g. ``[normalize, extract]``
    runs ``extract`` then ``normalize``. A ``str`` is resolved via the shared
    target registry (alias) or as a dotted import path.
    """
    if isinstance(spec, str):
        return _resolve_named_parser(spec)
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise AgenticFunctionConfigurationError("parser sequence is empty")
        from rich_python_utils.common_utils.function_helper import compose

        resolved: List[Parser] = [resolve_parser(item) for item in spec]
        if len(resolved) == 1:
            return resolved[0]
        return typing.cast(Parser, compose(*resolved))
    if callable(spec):
        return typing.cast(Parser, spec)
    raise AgenticFunctionConfigurationError(
        f"parser must be a callable, a registry alias/dotted path, or a sequence "
        f"of them; got {type(spec).__name__}"
    )


def _resolve_named_parser(name: str) -> Parser:
    from rich_python_utils.config_utils._registry import resolve_target

    try:
        dotted = resolve_target(name)
    except KeyError as e:
        raise AgenticFunctionConfigurationError(
            f"unknown parser alias {name!r}: {e}"
        ) from e
    return typing.cast(Parser, _import_symbol(dotted))


# ---------------------------------------------------------------------------
# Return-annotation defaults
# ---------------------------------------------------------------------------


def resolve_return_annotation(fn: Callable[..., Any]) -> Any:
    """Return the (best-effort evaluated) return annotation, or ``empty``.

    ``from __future__ import annotations`` in the caller's module leaves
    ``signature().return_annotation`` a *string*; ``get_type_hints`` evaluates it.
    Either form is handled downstream (by identity or by name).
    """
    try:
        return typing.get_type_hints(fn).get("return", inspect.Signature.empty)
    except Exception:
        return inspect.signature(fn).return_annotation


def annotation_kind(annotation: Any) -> str:
    """Classify a return annotation into a default-parser kind.

    Handles both the evaluated type object and the stringified form (checked by
    ``__name__``). ``bool`` is tested before ``int`` (``bool`` is an ``int``
    subclass, so a distinct branch matters). An attrs / dataclass / pydantic
    class is ``"struct"`` (constructed from a decoded JSON object). Anything else
    still unrecognized — including ``Optional[...]`` and typing generics like
    ``List[T]`` — is ``"other"`` (raw text; supply a ``parser=`` or body-as-parser
    for such a type).
    """
    if annotation in (inspect.Signature.empty, None, type(None)):
        return "str"
    name = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", None)
    )
    origin = typing.get_origin(annotation)
    if annotation is AgenticOutput or name == "AgenticOutput":
        return "output"
    if annotation is bool or name == "bool":
        return "bool"
    if annotation is int or name == "int":
        return "int"
    if annotation is float or name == "float":
        return "float"
    if annotation is dict or name in ("dict", "Dict") or origin is dict:
        return "dict"
    if annotation is str or name == "str":
        return "str"
    if _struct_constructor(annotation) is not None:
        return "struct"
    return "other"


def default_parser(fn: Callable[..., Any]) -> Parser:
    """The parser used when neither ``parser=`` nor a body-as-parser runs."""
    annotation = resolve_return_annotation(fn)
    kind = annotation_kind(annotation)
    if kind == "output":
        return lambda out: out
    if kind == "dict":
        return lambda out: out.json()
    if kind == "bool":
        return _coerce_bool
    if kind == "int":
        return _coerce_int
    if kind == "float":
        return _coerce_float
    if kind == "struct":
        return _struct_parser(annotation)
    # "str" and "other": the normalized text (a custom type needs a parser=).
    return lambda out: out.text


def _struct_constructor(
    annotation: Any,
) -> Optional[Callable[[Mapping[str, Any]], Any]]:
    """A canonical constructor for an attrs / dataclass / pydantic class, else None.

    Detection is structural, so importing attrs or pydantic is never forced:
    attrs sets ``__attrs_attrs__``; pydantic v2 exposes a ``model_validate``
    classmethod plus ``model_fields``; pydantic v1 exposes ``parse_obj`` plus
    ``__fields__``. Each branch delegates to that framework's blessed
    build-from-mapping entry point (so field aliases, converters, and validators
    all run). Only a plain class qualifies — a typing generic (``List[T]``,
    ``Optional[T]``) or a stringified annotation (the degraded
    ``get_type_hints`` fallback) returns ``None`` and falls through to raw text.
    """
    if not isinstance(annotation, type):
        return None
    model_validate = getattr(annotation, "model_validate", None)
    if callable(model_validate) and hasattr(annotation, "model_fields"):
        return lambda data: model_validate(data)
    parse_obj = getattr(annotation, "parse_obj", None)
    if callable(parse_obj) and hasattr(annotation, "__fields__"):
        return lambda data: parse_obj(data)
    if hasattr(annotation, "__attrs_attrs__"):
        return lambda data: annotation(**data)
    if dataclasses.is_dataclass(annotation):
        return lambda data: annotation(**data)
    return None


def _struct_parser(annotation: Any) -> Parser:
    """Construct the declared attrs/dataclass/pydantic type from a JSON object.

    The model's normalized text is decoded (hardened; a bad JSON reply already
    raises :class:`ParseError`), required to be an object, then handed to the
    type's canonical constructor. A construction failure (missing/extra field,
    validation error) is re-raised as :class:`ParseError` so it drives the same
    strict parse-retry / fallback path as the scalar coercers — never a silent
    partial object.
    """
    constructor = _struct_constructor(annotation)
    if constructor is None:  # defensive: annotation_kind only routes real structs here
        return lambda out: out.text
    type_name = getattr(annotation, "__name__", str(annotation))

    def parse(out: AgenticOutput) -> Any:
        data = out.json()
        if not isinstance(data, dict):
            raise ParseError(
                f"expected a JSON object to construct {type_name}, "
                f"got {type(data).__name__}"
            )
        try:
            return constructor(data)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"cannot construct {type_name} from {data!r}: {e}") from e

    return parse


def _coerce_int(out: AgenticOutput) -> int:
    text = out.text.strip()
    try:
        return int(text)
    except (TypeError, ValueError) as e:
        raise ParseError(f"cannot parse int from {text!r}") from e


def _coerce_float(out: AgenticOutput) -> float:
    text = out.text.strip()
    try:
        return float(text)
    except (TypeError, ValueError) as e:
        raise ParseError(f"cannot parse float from {text!r}") from e


def _coerce_bool(out: AgenticOutput) -> bool:
    text = out.text.strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    raise ParseError(f"cannot parse bool from {out.text.strip()!r}")


def sequence_is_multi(spec: Any) -> bool:
    """True if ``spec`` is a multi-element parser sequence (for trace labeling)."""
    return isinstance(spec, (list, tuple)) and len(spec) > 1
