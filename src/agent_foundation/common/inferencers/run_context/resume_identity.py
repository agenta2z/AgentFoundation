"""Canonical, secret-free identities for resume verification (plan v8 §5.12).

One pure function family, used for a BTA's input, its definition and its invocation
arguments:

* ``str`` → its UTF-8 bytes; ``bytes`` → themselves;
* anything else → canonical JSON (sorted keys, no whitespace, UTF-8) of its
  identity tree:

  - JSON scalars, lists, tuples, sets (sorted) and mappings, recursively;
  - enums and types by qualified name;
  - an object with a ``resume_identity`` (attribute, or method returning it);
  - an inferencer: its qualified class name and its semantic configuration — the
    constructor fields minus the classes' ``_RESUME_IDENTITY_EXCLUDE`` (placement,
    observability, scheduling, retry policy), every ``Debuggable`` field, and any
    field marked secret (``metadata={"secret": True}``) or named like one;
  - another attrs object: its qualified class name and its constructor fields
    (secret-named ones left out);
  - ``functools.partial``: its function, arguments and keywords;
  - a module-level function or class attribute function: ``module:qualname``.

Anything else — a lambda, a closure, a bound method, a live client — raises
``ResumeIdentityUnavailableError`` naming the value's type and path. ``repr()`` and
memory addresses are never used, and secrets are never hashed.
"""

from __future__ import annotations

import contextvars
import enum
import functools
import hashlib
import inspect
import json
import re
from typing import Any, FrozenSet, Iterable, Mapping
from unittest import mock

import attrs

from .errors import InvocationContractError


class ResumeIdentityUnavailableError(InvocationContractError, TypeError):
    """A value has no stable, secret-free identity. An invocation contract error
    (never retried): a resume that can't be verified fails before any I/O."""


_SECRET_NAME = re.compile(
    r"secret|token|password|passwd|api_?key|private_key|credential|^asap_|^auth$|"
    r"(^|_)env$|^env_vars$"
)


def is_secret_field(field: attrs.Attribute) -> bool:
    """Whether ``field`` is marked secret or named like one; its value is never
    part of an identity."""
    return bool(field.metadata.get("secret")) or bool(
        _SECRET_NAME.search(field.name.lstrip("_").lower())
    )


def qualified_name(obj: Any) -> str:
    return f"{obj.__module__}:{obj.__qualname__}"


def _debuggable_fields() -> FrozenSet[str]:
    from rich_python_utils.common_objects.debuggable import Debuggable

    return frozenset(f.name for f in attrs.fields(Debuggable))


def resume_identity_fields(cls: type) -> tuple[attrs.Attribute, ...]:
    """The constructor fields of inferencer class ``cls`` that make up its identity."""
    excluded = set(_debuggable_fields())
    for klass in cls.__mro__:
        excluded |= vars(klass).get("_RESUME_IDENTITY_EXCLUDE", frozenset())
    return tuple(
        f
        for f in attrs.fields(cls)
        if f.init and f.name not in excluded and not is_secret_field(f)
    )


def _sorted_trees(values: Iterable[Any], path: str) -> list:
    trees = [identity_tree(v, f"{path}[]") for v in values]
    return sorted(trees, key=_canonical_json)


def _mapping_tree(value: Mapping, path: str) -> Any:
    if all(isinstance(k, str) for k in value):
        return {k: identity_tree(v, f"{path}.{k}") for k, v in value.items()}
    items = [
        [identity_tree(k, f"{path}<key>"), identity_tree(v, f"{path}[{k!s}]")]
        for k, v in value.items()
    ]
    return {"items": sorted(items, key=_canonical_json)}


def _attrs_tree(value: Any, path: str) -> dict:
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase

    cls = type(value)
    if isinstance(value, InferencerBase):
        fields = resume_identity_fields(cls)
        kind = "inferencer"
    else:
        fields = tuple(
            f for f in attrs.fields(cls) if f.init and not is_secret_field(f)
        )
        kind = "attrs"
    return {
        kind: qualified_name(cls),
        "fields": {
            f.name: identity_tree(getattr(value, f.name), f"{path}.{f.name}")
            for f in fields
        },
    }


def _callable_tree(value: Any, path: str) -> Any:
    if isinstance(value, functools.partial):
        return {
            "partial": identity_tree(value.func, f"{path}.func"),
            "args": identity_tree(list(value.args), f"{path}.args"),
            "keywords": identity_tree(dict(value.keywords), f"{path}.keywords"),
        }
    qualname = getattr(value, "__qualname__", "")
    if inspect.isfunction(value) and "<" not in qualname:
        return {"callable": qualified_name(value)}
    owner = getattr(value, "__self__", None)
    if inspect.isbuiltin(value) and (owner is None or inspect.ismodule(owner)):
        return {"callable": f"{value.__module__}:{qualname}"}
    if inspect.ismethod(value) and isinstance(owner, type):
        return {"callable": qualified_name(value.__func__)}
    raise ResumeIdentityUnavailableError(
        f"{path}: {type(value).__qualname__} {qualname or ''} has no stable resume "
        f"identity (lambdas, closures and bound methods have none); give it a "
        f"`resume_identity`"
    )


_NOT_SCALAR = object()


def _scalar_tree(value: Any, path: str) -> Any:
    """The identity of a JSON scalar, bytes, an enum or a type; else ``_NOT_SCALAR``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": hashlib.sha256(value).hexdigest()}
    if isinstance(value, enum.Enum):
        return {
            "enum": qualified_name(type(value)),
            "value": identity_tree(value.value, path),
        }
    if isinstance(value, type):
        return {"type": qualified_name(value)}
    return _NOT_SCALAR


_ON_PATH: contextvars.ContextVar[FrozenSet[int]] = contextvars.ContextVar(
    "resume_identity_on_path", default=frozenset()
)


def identity_tree(value: Any, path: str = "value") -> Any:
    """The JSON-compatible identity of ``value`` (see the module docstring)."""
    scalar = _scalar_tree(value, path)
    if scalar is not _NOT_SCALAR:
        return scalar
    on_path = _ON_PATH.get()
    if id(value) in on_path:
        raise ResumeIdentityUnavailableError(
            f"{path}: {type(value).__qualname__} refers back to itself"
        )
    token = _ON_PATH.set(on_path | {id(value)})
    try:
        return _composite_tree(value, path)
    finally:
        _ON_PATH.reset(token)


def _composite_tree(value: Any, path: str) -> Any:
    if isinstance(value, mock.NonCallableMock):
        raise ResumeIdentityUnavailableError(f"{path}: a mock has no resume identity")
    supplied = getattr(value, "resume_identity", None)
    if supplied is not None:
        if callable(supplied):
            supplied = supplied()
        return {
            "identity": qualified_name(type(value)),
            "value": identity_tree(supplied, f"{path}.resume_identity"),
        }
    if isinstance(value, Mapping):
        return _mapping_tree(value, path)
    if isinstance(value, (list, tuple)):
        return [identity_tree(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        return {"set": _sorted_trees(value, path)}
    if attrs.has(type(value)):
        return _attrs_tree(value, path)
    if callable(value):
        return _callable_tree(value, path)
    raise ResumeIdentityUnavailableError(
        f"{path}: {type(value).__module__}.{type(value).__qualname__} has no stable "
        f"resume identity; give it a `resume_identity`"
    )


def _canonical_json(tree: Any) -> str:
    return json.dumps(tree, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identity_bytes(value: Any, path: str = "value") -> bytes:
    """``str`` → UTF-8, ``bytes`` → themselves, else canonical JSON of the tree."""
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes):
        return value
    return _canonical_json(identity_tree(value, path)).encode("utf-8")


def identity_digest(value: Any, path: str = "value") -> str:
    """The SHA-256 hex digest of :func:`identity_bytes`."""
    return hashlib.sha256(identity_bytes(value, path)).hexdigest()
