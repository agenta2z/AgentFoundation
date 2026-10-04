"""Source checks over the invocation-scoped runtime (plan v8 §11 P11 item 4).

Static complements to the purity ratchet, which measures what host calls write:

* **Declared compat fields** — the instance fields a ``RuntimeKey`` declares as the
  bare projection of a result (``compat={...}``) — are written only by the compat
  flush (``publish_result`` / ``_flush_compat``, through ``setattr``) and read only by
  the documented getters and their projections. Any other ``self.<field>`` in the
  sources would make the projection internal state again.
* **Typed node state holds JSON values only**: every field of a registered state class
  is a scalar, ``None``, another state class, or a list / tuple / dict of those, so
  no live object can reach a serialized store. A field typed ``Any`` must say why it
  holds JSON values, here.

Host-call writes outside ``ALLOWED`` and live objects in a host store are measured
by the ratchet (I1–I3) for every certified class.
"""

from __future__ import annotations

import ast
import json
import os
import types
import typing

import agent_foundation
import attrs
import pytest

# Registers the state classes defined outside ``run_context.state``.
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (  # noqa: F401
    breakdown_then_aggregate_inferencer,
)
from agent_foundation.common.inferencers.run_context.state import (
    decode_state,
    InferencerStateBase,
    STATE_REGISTRY,
)

_ROOT = os.path.dirname(agent_foundation.__file__)

# The registered state classes the sources define (test modules register their own).
_STATES = sorted(
    name
    for name, cls in STATE_REGISTRY.items()
    if cls.__module__.startswith("agent_foundation.")
)

# The documented getters, and the projections they read through, by module and
# function: the only places that read a declared compat field.
COMPAT_READERS = {
    ("common/inferencers/templated_inferencer_base.py", "_proposer_task_instructions"),
    (
        "common/inferencers/agentic_inferencers/flow_inferencers/"
        "breakdown_then_aggregate_inferencer.py",
        "_proposer_task_instructions",
    ),
    (
        "common/inferencers/agentic_inferencers/flow_inferencers/"
        "breakdown_then_aggregate_inferencer.py",
        "last_call_summary",
    ),
    # ``get_streaming_result``'s projection; outside an invocation only
    (
        "common/inferencers/terminal_inferencers/terminal_inferencer_base.py",
        "_stream_result",
    ),
    # ``get_final_output``'s projection; outside an invocation only
    (
        "common/inferencers/agentic_inferencers/external/rovodev/rovodev_cli_inferencer.py",
        "_call_output",
    ),
}

# State fields typed ``Any`` (alone or as a container value), each holding JSON
# values by construction.
ANY_FIELDS = {
    # per-flow result records, already plain dicts / strings
    ("BTAState", "latest_per_flow"),
    ("MultiFlowAttemptState", "latest_per_flow"),
    ("MultiFlowAttemptState", "latest_per_flow_path"),
    # the workflow's state dict, saved by its JSON checkpoints too
    ("LinearWorkflowState", "state"),
    # the call arguments a splice re-dispatches with: the workflow's JSON step input
    ("LinearWorkflowState", "splice_orig_args"),
    ("LinearWorkflowState", "splice_orig_kwargs"),
    # task inputs of earlier stores, decoded only (no longer written; X78)
    ("MultiFlowState", "flow_inputs"),
    # role-switch arguments: template names, variables and feed values from config
    ("RoleState", "modes"),
    ("RoleState", "changes"),
    ("RoleState", "template_variables"),
    ("RoleState", "template_extra_feed"),
}


def _sources():
    for directory, _, files in os.walk(_ROOT):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                yield os.path.relpath(path, _ROOT), path


def _parse(path):
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _declared_compat_fields():
    """Every field named in a ``RuntimeKey(..., compat={...})`` literal."""
    fields = set()
    for _, path in _sources():
        for node in ast.walk(_parse(path)):
            if not (isinstance(node, ast.Call) and _name(node.func) == "RuntimeKey"):
                continue
            for keyword in node.keywords:
                if keyword.arg == "compat" and isinstance(keyword.value, ast.Dict):
                    fields |= {k.value for k in keyword.value.keys if k is not None}
    return fields


def _name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _SelfAttributes(ast.NodeVisitor):
    """``self.<name>`` reads and writes with their enclosing function."""

    def __init__(self, names):
        self.names = names
        self.functions = [None]
        self.uses = []  # (function, name, kind)

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in self.names
        ):
            kind = "read" if isinstance(node.ctx, ast.Load) else "write"
            self.uses.append((self.functions[-1], node.attr, kind))
        self.generic_visit(node)


def _compat_uses():
    names = _declared_compat_fields()
    for rel, path in _sources():
        visitor = _SelfAttributes(names)
        visitor.visit(_parse(path))
        for function, name, kind in visitor.uses:
            yield rel, function, name, kind


def test_the_scan_finds_the_known_compat_fields():
    """A scan that found nothing would pass the checks below vacuously."""
    assert _declared_compat_fields() >= {
        "_last_rendered_task_instructions",
        "_last_streaming_output",
        "_last_streaming_stderr",
        "_last_streaming_return_code",
        "_last_clean_output",
        "_last_raw_stdout",
        "_last_call_summary",
        "_worker_task_instructions",
    }


def test_the_attribute_scan_sees_reads_and_writes():
    visitor = _SelfAttributes({"_last_x"})
    visitor.visit(
        ast.parse(
            "class C:\n"
            "    def getter(self):\n        return self._last_x\n"
            "    def leak(self, v):\n        self._last_x = v\n"
        )
    )
    assert visitor.uses == [("getter", "_last_x", "read"), ("leak", "_last_x", "write")]


def test_compat_fields_are_written_only_by_the_compat_flush():
    writes = sorted(
        f"{rel}:{function}: self.{name}"
        for rel, function, name, kind in _compat_uses()
        if kind == "write"
    )
    assert writes == []


def test_compat_fields_are_read_only_by_their_getters():
    reads = {
        (rel, function) for rel, function, _, kind in _compat_uses() if kind == "read"
    }
    assert sorted(reads - COMPAT_READERS) == []
    assert sorted(COMPAT_READERS - reads) == [], "stale reader entries"


_SCALARS = (str, int, float, bool, type(None))


def _json_shaped(tp, any_allowed):
    """Whether values of type ``tp`` are JSON values (``Any`` only if allowed)."""
    if tp is typing.Any:
        return any_allowed
    if isinstance(tp, type) and issubclass(tp, (*_SCALARS, InferencerStateBase)):
        return True
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        return all(_json_shaped(arg, any_allowed) for arg in typing.get_args(tp))
    if origin in (list, tuple, dict) or tp in (list, tuple, dict):
        args = [arg for arg in typing.get_args(tp) if arg is not Ellipsis]
        if origin is dict and args and args[0] not in (str, int):
            return False
        values = args[1:] if origin is dict else args
        return all(_json_shaped(arg, any_allowed) for arg in values) and (
            bool(args) or any_allowed
        )
    return False


def test_the_json_shape_check_rejects_live_and_untyped_values():
    assert _json_shaped(dict[str, list[tuple[int, str | None]]], False)
    assert not _json_shaped(typing.Any, False)
    assert not _json_shaped(list, False)
    assert not _json_shaped(dict[str, object], True)
    assert not _json_shaped(types.SimpleNamespace, True)


@pytest.mark.parametrize("name", _STATES)
def test_typed_state_fields_hold_json_values(name):
    cls = STATE_REGISTRY[name]
    attrs.resolve_types(cls)
    bad = [
        f"{field.name}: {field.type}"
        for field in attrs.fields(cls)
        if not _json_shaped(field.type, (name, field.name) in ANY_FIELDS)
    ]
    assert bad == []


def test_every_any_field_entry_is_still_typed_any():
    """An entry whose field got a precise type (or went away) is stale."""
    stale = []
    for name, field_name in sorted(ANY_FIELDS):
        cls = STATE_REGISTRY.get(name)
        fields = {} if cls is None else {f.name: f for f in attrs.fields(cls)}
        if field_name not in fields or _json_shaped(fields[field_name].type, False):
            stale.append(f"{name}.{field_name}")
    assert stale == []


@pytest.mark.parametrize("name", _STATES)
def test_a_default_typed_state_round_trips_through_json(name):
    cls = STATE_REGISTRY[name]
    required = [f for f in attrs.fields(cls) if f.default is attrs.NOTHING]
    if required:
        state = cls(**{f.name: "" for f in required})
    else:
        state = cls()
    encoded = json.loads(json.dumps(state.to_json()))
    assert decode_state(encoded) == state
