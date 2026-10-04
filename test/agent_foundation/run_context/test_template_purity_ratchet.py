"""P0 host-purity ratchet: invariants I1–I4 of the invocation-scoped runtime plan.

The plan (``AgentFoundation/invocation_scoped_runtime.plan.md``) makes every inferencer
a definition-stable template: under a host-owned ``RunContext`` a call writes nothing
call-specific into its own ``__dict__`` or into any object it borrows. This module
MEASURES how far each class is from that today, and ratchets it:

  * **I1 — owner purity.** Each fixture runs twice, each call under a fresh host root.
    The cold delta (fresh definition → after call 1) may contain only ``ALLOWED``
    residents (lazily built, idempotent definition caches) or the class's
    ``KNOWN_DEBT``. The warm delta (call 1 → call 2) may contain only ``KNOWN_DEBT``: a
    legitimate cache is filled once, so anything that changes again is per-call state.
  * **I2 — descendant purity.** The same rule for every configured child and borrowed
    stage reachable from the fresh definition, plus ``CHILD_DEBT`` rows for writes a
    parent makes into one specific child slot. Objects the call itself creates (factory
    products) are not descendants of the definition and may be initialized freely.
  * **I3 — serializable node state.** After each call the host store goes through
    ``json.dumps`` with no ``default=`` hook, so no live object sits in node state.
  * **I4 — rejection before I/O** lives with its mechanisms: the path claim
    (``test_path_claims.py``), the single-flight guard (``test_single_flight_guard.py``)
    and the checkpoint-root lease (``test_bta_checkpoint_lease.py``).

Rules the tables obey:

  * Classification is debt first (the subject class's ``KNOWN_DEBT``, then the
    ``CHILD_DEBT`` row of that slot), then ``ALLOWED`` (cold delta only), else failure.
    Config fields are NOT blanket-allowed: a config field a call rewrites is per-call
    state wearing a config name (BTA's ``start_nodes`` before P7, LWI's
    ``enable_result_save`` before P9).
  * The subject class of an instance is its first MRO class defined in
    ``agent_foundation``, so the local stubs are charged to the base they exercise.
  * **Declared compat fields are bare-only.** The getter fields a class's
    ``RuntimeKey`` declarations feed (``declared_compat_fields``) are written only by
    non-host invocations, so a host call that changes one fails, whatever the tables
    say.
  * **Every debt entry names the phase that removes it, and a stale entry fails**: an
    entry no fixture observes any more is deleted by the commit that fixed it. The same
    holds for ``ALLOWED``: a resident no fixture writes is no longer an allowance.
  * **Certification both ways.** ``_HOST_PURE_CERTIFIED`` (read by the P6 single-flight
    guard through ``vars(type(owner))``, so never inherited) is set on a measured class
    exactly when its debt is empty — its own ``KNOWN_DEBT`` and the ``CHILD_DEBT`` of
    every fixture it owns — and no class is certified unmeasured. The sources are
    scanned for certifying modules, so an unimported module can't hide one.

The fixtures use local pure stubs rather than the ``_helpers`` mocks: those record calls
on ``self`` by design, which would be charged to the class under test. The provider-leaf
fixtures (``_leaf_fixtures``) are real leaves over fake transports, measured in a third
mode as well, a drained ``ainfer_streaming``.
"""

import asyncio
import importlib
import json
import os
import re
from collections import defaultdict

import agent_foundation
import agent_foundation.common.configs  # noqa: F401
import pytest
from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    declared_compat_fields,
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.run_context.purity import (
    describe_delta,
    diff_vars,
    snapshot_vars,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode

from ._leaf_fixtures import LEAF_FIXTURES


# --- Pure stubs ----------------------------------------------------------------------


@attrs(slots=False)
class _Pure(InferencerBase):
    response = attrib(default="ok")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self.response


class _PureTemplates:
    def __call__(
        self, key, *, active_template_root_space=None, master_version=None, **feed
    ):
        return (
            f"[{active_template_root_space}|{key}|{master_version}] {feed.get('input')}"
        )

    def get_raw_template(self, *a, **k):
        return "x"

    def load_variables(self, *a, **k):
        return {}

    def add_template_root(self, *a, **k):
        pass

    def __deepcopy__(self, memo):
        return self


def _answer(prompt):
    head = prompt[: prompt.index("]") + 1] if prompt.startswith("[") else ""
    if head.startswith("[task_breakdown|"):
        return "1. alpha shard\n2. beta shard"
    if "|aggregation" in head:
        return "AGGREGATED"
    return f"W({prompt})"


@attrs(slots=False)
class _PureTemplated(TemplatedInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return _answer(str(inference_input))

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return _answer(str(inference_input))


@attrs(slots=False)
class _PureStream(StreamingInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "streamed"

    async def _ainfer_streaming(self, inference_input, inference_config=None, **kwargs):
        yield "streamed"


# --- Fixtures: name -> builder(tmp) ----------------------------------------------------


def _templated(**kw):
    return _PureTemplated(
        template_manager=_PureTemplates(),
        template_root_space="plan",
        template_key="initial",
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
        **kw,
    )


def _stop(state, out):
    return True


def _flow(resp, inp):
    return {
        "initial_inferencer": _Pure(response=resp),
        "followup_inferencer": _Pure(response=resp + "-fu"),
        "input": inp,
        "end_condition": _stop,
        "max_dynamic_steps": 1,
    }


def _dual(tmp, **kw):
    return DualInferencer(
        base_inferencer=_Pure(response="base proposal"),
        review_inferencer=_Pure(
            response="## Review\nSeverity: COSMETIC\nApproved: true"
        ),
        fixer_inferencer=None,
        consensus_config=ConsensusConfig(),
        **kw,
    )


def _mfi(tmp):
    return MultiFlowInferencer(
        flow_configs=[_flow("fa", "a"), _flow("fb", "b")], disable_aggregator=True
    )


def _mfdual(tmp, **kw):
    return MultiFlowDualInferencer(
        flow_configs=[_flow("fa", "a"), _flow("fb", "b")],
        multi_flow_disable_aggregator=True,
        reviewer_strategy="all_non_winners",
        **kw,
    )


def _bta(tmp):
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Pure(response="1. Q1\n2. Q2"),
        worker_inferencers=lambda sub_query, index: _Pure(response=f"W{index}"),
        aggregator_inferencer=_Pure(response="agg"),
    )


def _bta_list(tmp):
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Pure(response="1. Q1\n2. Q2"),
        worker_inferencers=[_Pure(response="W0"), _Pure(response="W1")],
        aggregator_inferencer=_Pure(response="agg"),
    )


def _bta_shared_breakdown(tmp):
    """Nested BTAs that a factory builds around the outer BTA's own breakdown."""
    breakdown = _Pure(response="1. Q1")

    def nested(sub_query, index):
        return BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=lambda sub_query, index: _Pure(response="nw"),
            predefined_sub_queries=["n0"],
            disable_aggregator=True,
        )

    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=breakdown,
        worker_inferencers=nested,
        aggregator_inferencer=_Pure(response="agg"),
    )


def _bta_nested(tmp):
    nested = _bta(tmp)
    nested.name = "nested"
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Pure(response="1. Q1"),
        worker_inferencers=[nested],
        aggregator_inferencer=_Pure(response="agg"),
    )


def _lwi(tmp):
    return LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=_Pure(response="one")),
            WorkflowStepConfig(name="s2", inferencer=_Pure(response="two")),
        ],
    )


def _advance_twice(step_input, state):
    if state["iteration"] < 3:
        state["iteration"] += 1
    return state["iteration"]


def _lwi_loop(tmp):
    """A static LWI whose loop advances ``iteration``, so every call re-roots."""
    return LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=_Pure(response="one")),
            WorkflowStepConfig(
                name="advance",
                step_fn=_advance_twice,
                loop_back_to="s1",
                loop_condition=lambda state, result: result < 3,
                enable_result_save=False,
            ),
        ],
        workspace=InferencerWorkspace(root=os.path.join(tmp, "lwi_loop_def")),
    )


def _lwi_dyn(tmp):
    return LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=_Pure(response="init"),
        default_followup_inferencer=_Pure(response="follow"),
        end_condition=_stop,
        max_dynamic_steps=2,
        workspace=InferencerWorkspace(root=os.path.join(tmp, "lwi_dyn_def")),
    )


def _pti(tmp):
    return PlanThenImplementInferencer(
        planner_inferencer=_Pure(response="## Plan\n1. Step one\n2. Step two"),
        executor_inferencer=_Pure(response="Implementation complete."),
        analyzer_inferencer=None,
    )


def _pti_duals(tmp):
    """A PTI whose planner and executor are Duals: child workflows it configures."""
    return PlanThenImplementInferencer(
        planner_inferencer=_dual(tmp),
        executor_inferencer=_dual(tmp),
        analyzer_inferencer=None,
    )


def _lwi_dyn_dual(tmp):
    """A dynamic LWI with a workspace whose initial step is a Dual workflow."""
    return LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=_dual(tmp),
        default_followup_inferencer=_Pure(response="follow"),
        end_condition=_stop,
        max_dynamic_steps=2,
        workspace=InferencerWorkspace(root=os.path.join(tmp, "lwi_dyn_dual_def")),
    )


def _dual_lwi_base(tmp):
    """A Dual whose base is a static LWI workflow."""
    return DualInferencer(
        base_inferencer=_lwi(tmp),
        review_inferencer=_Pure(
            response="## Review\nSeverity: COSMETIC\nApproved: true"
        ),
        fixer_inferencer=None,
        consensus_config=ConsensusConfig(),
    )


def _fanout(tmp):
    return _templated(
        workspace=InferencerWorkspace(root=os.path.join(tmp, "fan_def")),
        bta_inferencer=BreakdownThenAggregateInferencer(
            fallback_mode=FallbackMode.NEVER, max_retry=0
        ),
    )


_FIXTURES = {
    "templated": lambda tmp: _templated(),
    "stream": lambda tmp: _PureStream(),
    "dual": _dual,
    "dual_ckpt": lambda tmp: _dual(tmp, enable_checkpoint=True),
    "mfi": _mfi,
    "mfdual": _mfdual,
    "mfdual_ckpt": lambda tmp: _mfdual(tmp, enable_checkpoint=True),
    "bta": _bta,
    "bta_list": _bta_list,
    "bta_nested": _bta_nested,
    "bta_shared_breakdown": _bta_shared_breakdown,
    "lwi": _lwi,
    "lwi_loop": _lwi_loop,
    "lwi_dyn": _lwi_dyn,
    "pti": _pti,
    "pti_duals": _pti_duals,
    "lwi_dyn_dual": _lwi_dyn_dual,
    "dual_lwi_base": _dual_lwi_base,
    "fanout": _fanout,
}
_MODES = ("sync", "async")
_LEAF_MODES = ("sync", "async", "stream")


def _modes(name):
    return _LEAF_MODES if name in LEAF_FIXTURES else _MODES


_CASES = [
    (name, mode) for name in (*_FIXTURES, *LEAF_FIXTURES) for mode in _modes(name)
]
_CASE_IDS = [f"{name}-{mode}" for name, mode in _CASES]

# Every configured child / borrowed stage the walk must reach: an empty walk would make
# I2 pass vacuously.
EXPECTED_DESCENDANTS = {
    "templated": (),
    "stream": (),
    "dual": ("propose", "review"),
    "dual_ckpt": ("propose", "review"),
    "mfi": ("flow_0_followup", "flow_0_initial", "flow_1_followup", "flow_1_initial"),
    "mfdual": (
        "propose",
        "propose/flow_0_followup",
        "propose/flow_0_initial",
        "propose/flow_1_followup",
        "propose/flow_1_initial",
    ),
    "mfdual_ckpt": (
        "propose",
        "propose/flow_0_followup",
        "propose/flow_0_initial",
        "propose/flow_1_followup",
        "propose/flow_1_initial",
    ),
    "bta": ("aggregator", "breakdown"),
    "bta_list": ("aggregator", "breakdown", "worker_0", "worker_1"),
    "bta_shared_breakdown": ("aggregator", "breakdown"),
    "bta_nested": (
        "aggregator",
        "breakdown",
        "worker_0",
        "worker_0/aggregator",
        "worker_0/breakdown",
    ),
    "lwi": ("steps/s1", "steps/s2"),
    "lwi_loop": ("steps/s1",),
    "lwi_dyn": ("default_followup_inferencer", "default_initial_inferencer"),
    "pti": ("executor", "planner"),
    "pti_duals": (
        "executor",
        "executor/propose",
        "executor/review",
        "planner",
        "planner/propose",
        "planner/review",
    ),
    "lwi_dyn_dual": (
        "default_followup_inferencer",
        "default_initial_inferencer",
        "default_initial_inferencer/propose",
        "default_initial_inferencer/review",
    ),
    "dual_lwi_base": ("propose", "propose/steps/s1", "propose/steps/s2", "review"),
    "fanout": ("bta_inferencer.prototype",),
    **{name: () for name in LEAF_FIXTURES},
}


# --- Tables --------------------------------------------------------------------------

# Lazily built, idempotent definition residents: allowed in the COLD delta only.
ALLOWED = frozenset(
    {
        "_extension_manager_cache",  # pure function of the definition
        "logger",  # logger un-defer: workspace-derived definition cache
        "_logger_awaiting_workspace",  # logger un-defer
        "_resolved_logger_configs",  # logger un-defer
        "_ws_log_relpaths",  # logger un-defer: static {logger: relpath} tag
        "_live_handle_store",  # Tier-3 connection holder
        "_connect_lock",  # lazily created lock
    }
)

# Measured per-call writes on the instance itself: {subject class: {field: removal}}.
KNOWN_DEBT: dict = {}

# Measured parent writes into one child slot: {(fixture, slot path): {field: removal}}.
CHILD_DEBT: dict = {}


# --- Measurement ---------------------------------------------------------------------


@attrs(frozen=True, slots=True)
class _Observed:
    subject = attrib()
    cold = attrib()
    warm = attrib()


@attrs(frozen=True, slots=True)
class _Measurement:
    owner = attrib()
    kids = attrib()  # slot path -> _Observed
    store_errors = attrib()
    outputs = attrib()  # each call's result


def _subject_class(obj):
    return next(
        cls
        for cls in type(obj).__mro__
        if cls.__module__ != __name__ and cls.__module__.startswith("agent_foundation.")
    )


_STAGE_ATTRS = ("breakdown_inferencer", "aggregator_inferencer", "bta_inferencer")
_DEFAULT_STEP_ATTRS = ("default_initial_inferencer", "default_followup_inferencer")


def _named_children(inf):
    out = list(inf._iter_child_slots())
    out += [(attr, getattr(inf, attr, None)) for attr in _STAGE_ATTRS]
    workers = getattr(inf, "worker_inferencers", None)
    if isinstance(workers, (list, tuple)):
        out += [(f"worker_{i}", w) for i, w in enumerate(workers)]
    for step in getattr(inf, "step_configs", None) or ():
        out.append((f"steps/{step.name}", getattr(step, "inferencer", None)))
    out += [(attr, getattr(inf, attr, None)) for attr in _DEFAULT_STEP_ATTRS]
    fanout = getattr(inf, "bta_inferencer", None)
    out.append(("bta_inferencer.prototype", getattr(fanout, "prototype", None)))
    return [(slot, c) for slot, c in out if isinstance(c, InferencerBase)]


def _descendants(root):
    seen = {id(root)}
    out = {}
    stack = [("", root)]
    while stack:
        prefix, node = stack.pop()
        for slot, child in _named_children(node):
            if id(child) in seen:
                continue
            seen.add(id(child))
            path = f"{prefix}/{slot}" if prefix else slot
            out[path] = child
            stack.append((path, child))
    return out


async def _drain(stream):
    return [chunk async for chunk in stream]


def _run_call(inf, mode, ws_root, text):
    root = RunContext.root(workspace=InferencerWorkspace(root=ws_root))
    token = enter_run(root)
    try:
        if mode == "sync":
            output = inf.infer(text)
        elif mode == "async":
            output = asyncio.run(inf.ainfer(text))
        else:
            output = "".join(asyncio.run(_drain(inf.ainfer_streaming(text))))
    finally:
        exit_run(token)
    return root, output


def _store_json_error(root):
    try:
        json.dumps(root._store.to_json())
    except (TypeError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _observe(inf, mode, base):
    """Snapshots the definition and its descendants before, between and after two calls."""
    objs = {"": inf, **_descendants(inf)}
    snaps = [{path: snapshot_vars(o) for path, o in objs.items()}]
    errors, outputs = [], []
    for i in (1, 2):
        root, output = _run_call(
            inf, mode, os.path.join(base, f"call_{i}"), f"task-{i}"
        )
        outputs.append(output)
        snaps.append({path: snapshot_vars(o) for path, o in objs.items()})
        error = _store_json_error(root)
        if error:
            errors.append(f"call {i}: {error}")
    observed = {
        path: _Observed(
            subject=_subject_class(o),
            cold=diff_vars(snaps[0][path], snaps[1][path]),
            warm=diff_vars(snaps[1][path], snaps[2][path]),
        )
        for path, o in objs.items()
    }
    owner = observed.pop("")
    return _Measurement(
        owner=owner, kids=observed, store_errors=tuple(errors), outputs=tuple(outputs)
    )


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Measures each (fixture, mode) lazily, once per module, in its own directory."""
    tmp = str(tmp_path_factory.mktemp("purity_ratchet"))
    cache = {}

    def get(name, mode):
        if (name, mode) not in cache:
            base = os.path.join(tmp, f"{name}_{mode}")
            with pytest.MonkeyPatch.context() as mp:
                inf = (
                    LEAF_FIXTURES[name](base, mp)
                    if name in LEAF_FIXTURES
                    else _FIXTURES[name](base)
                )
                cache[name, mode] = _observe(inf, mode, base)
        return cache[name, mode]

    return get


def _all(measured):
    return [measured(name, mode) for name, mode in _CASES]


def _keys(delta):
    return delta.added.keys() | delta.changed.keys() | delta.removed.keys()


def _debt_visible(obs):
    return (_keys(obs.cold) - ALLOWED) | _keys(obs.warm)


def _violations(obs, debt):
    found = {
        "cold": sorted(_keys(obs.cold) - debt.keys() - ALLOWED),
        "warm": sorted(_keys(obs.warm) - debt.keys()),
    }
    return {phase: keys for phase, keys in found.items() if keys}


def _explain(obs):
    return f"cold: {describe_delta(obs.cold)}; warm: {describe_delta(obs.warm)}"


def _kid_debt(name, path, obs):
    return {**KNOWN_DEBT.get(obs.subject, {}), **CHILD_DEBT.get((name, path), {})}


def _certified(cls):
    return bool(vars(cls).get("_HOST_PURE_CERTIFIED", False))


def _measured_classes(measured):
    return {obs.subject for m in _all(measured) for obs in (m.owner, *m.kids.values())}


def _all_subclasses(cls):
    seen = set()
    stack = [cls]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


_CERTIFICATION = re.compile(
    r"^\s+_HOST_PURE_CERTIFIED\b[^=\n]*=\s*True\b", re.MULTILINE
)


def _source_files(root):
    return [
        os.path.join(d, f)
        for d, _, files in os.walk(root, followlinks=True)
        for f in files
        if f.endswith(".py")
    ]


def _module_name(root, path):
    parts = os.path.splitext(os.path.relpath(path, root))[0].split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["agent_foundation", *parts])


def _declares_certification(path):
    with open(path, encoding="utf-8") as f:
        return _CERTIFICATION.search(f.read()) is not None


def _certifying_modules(root):
    files = _source_files(root)
    assert files, f"no sources under {root}: the certification scan would be vacuous"
    return sorted(_module_name(root, p) for p in files if _declares_certification(p))


# --- I1 / I2 -------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "mode"), _CASES, ids=_CASE_IDS)
def test_i1_owner_delta_is_allowed_or_known_debt(measured, name, mode):
    """Cold ⊆ ALLOWED ∪ debt, warm ⊆ debt, for the definition itself."""
    m = measured(name, mode)
    debt = KNOWN_DEBT.get(m.owner.subject, {})
    assert _violations(m.owner, debt) == {}, (
        f"{m.owner.subject.__name__}: {_explain(m.owner)}"
    )


@pytest.mark.parametrize(("name", "mode"), _CASES, ids=_CASE_IDS)
def test_i2_descendant_deltas_are_known_debt(measured, name, mode):
    """No call writes into a configured child or borrowed stage beyond measured debt."""
    m = measured(name, mode)
    bad = {
        path: (_violations(obs, _kid_debt(name, path, obs)), _explain(obs))
        for path, obs in m.kids.items()
    }
    assert {path: v for path, v in bad.items() if v[0]} == {}


@pytest.mark.parametrize(("name", "mode"), _CASES, ids=_CASE_IDS)
def test_i1_host_calls_never_write_declared_compat_fields(measured, name, mode):
    """Declared compat fields belong to bare invocations only (plan §5.4)."""
    m = measured(name, mode)
    written = {
        path: sorted(
            (_keys(obs.cold) | _keys(obs.warm)) & declared_compat_fields(obs.subject)
        )
        for path, obs in {"<owner>": m.owner, **m.kids}.items()
    }
    assert {path: fields for path, fields in written.items() if fields} == {}


@pytest.mark.parametrize(("name", "mode"), _CASES, ids=_CASE_IDS)
def test_i2_walk_reaches_every_configured_descendant(measured, name, mode):
    """The descendant walk is what makes I2 meaningful; it must not silently shrink."""
    assert sorted(measured(name, mode).kids) == sorted(EXPECTED_DESCENDANTS[name])


def test_every_leaf_fixture_reaches_its_transport(measured):
    """A leaf whose fake transport is never reached would certify vacuously: every
    measured call answers through it."""
    outputs = {
        f"{name}-{mode}": measured(name, mode).outputs
        for name, mode in _CASES
        if name in LEAF_FIXTURES
    }
    silent = {
        case: outs
        for case, outs in outputs.items()
        if not all(str(out).strip() for out in outs)
    }
    assert silent == {}


def test_classifier_flags_a_per_call_write(tmp_path):
    """A stub that stores its input on ``self`` fails both the cold and warm checks."""

    @attrs(slots=False)
    class _Leaky(_Pure):
        def _infer(self, inference_input, inference_config=None, **kwargs):
            self._leak = inference_input
            return self.response

    m = _observe(_Leaky(), "sync", str(tmp_path))
    assert m.owner.subject is InferencerBase
    assert _violations(m.owner, KNOWN_DEBT.get(InferencerBase, {})) == {
        "cold": ["_leak"],
        "warm": ["_leak"],
    }


# --- Stale debt and certification ----------------------------------------------------


def test_every_allowed_entry_is_still_observed(measured):
    """An ALLOWED resident no fixture writes any more is a stale allowance: delete it,
    so a future per-call write of that name can't hide behind it."""
    observed = set()
    for m in _all(measured):
        for obs in (m.owner, *m.kids.values()):
            observed |= _keys(obs.cold) & ALLOWED
    assert sorted(ALLOWED - observed) == []


def test_every_known_debt_entry_is_still_observed(measured):
    """A debt entry no fixture observes was fixed: delete it in the fixing commit."""
    visible = defaultdict(set)
    for m in _all(measured):
        for obs in (m.owner, *m.kids.values()):
            visible[obs.subject] |= _debt_visible(obs)
    stale = {
        f"{cls.__name__}.{field}": tag
        for cls, debt in KNOWN_DEBT.items()
        for field, tag in debt.items()
        if field not in visible[cls]
    }
    assert stale == {}


def _child_debt_problem(measured, name, path, field):
    kids = [measured(name, mode).kids.get(path) for mode in _modes(name)]
    if None in kids:
        return "slot not reached"
    if field in KNOWN_DEBT.get(kids[0].subject, {}):
        return "shadowed by the class's KNOWN_DEBT"
    if not any(field in _debt_visible(kid) for kid in kids):
        return "not observed"
    return None


def test_every_child_debt_entry_is_still_observed(measured):
    """A CHILD_DEBT row must be reached, observed, and not already class debt."""
    problems = {
        f"{name}:{path}.{field}": problem
        for (name, path), debt in CHILD_DEBT.items()
        for field in debt
        if (problem := _child_debt_problem(measured, name, path, field))
    }
    assert problems == {}


def _class_debt(measured, cls):
    """``cls``'s own debt plus the child debt of every fixture whose owner it is: a
    class that still writes into its children is not host-pure either."""
    debt = dict(KNOWN_DEBT.get(cls, {}))
    for (name, path), fields in CHILD_DEBT.items():
        if any(measured(name, mode).owner.subject is cls for mode in _modes(name)):
            debt.update({f"{path}.{field}": phase for field, phase in fields.items()})
    return debt


def test_certification_matches_measured_debt(measured):
    """``_HOST_PURE_CERTIFIED`` is set exactly on the measured classes with no debt."""
    wrong = {
        cls.__name__: sorted(_class_debt(measured, cls))
        for cls in _measured_classes(measured)
        if _certified(cls) == bool(_class_debt(measured, cls))
    }
    assert wrong == {}


def test_no_certified_class_is_unmeasured(measured):
    """Certification without a ratchet fixture would be an unproven claim."""
    for module in _certifying_modules(os.path.dirname(agent_foundation.__file__)):
        importlib.import_module(module)
    classes = _all_subclasses(InferencerBase) | {InferencerBase}
    certified = {cls for cls in classes if _certified(cls)}
    unmeasured = certified - _measured_classes(measured)
    assert sorted(cls.__qualname__ for cls in unmeasured) == []


def test_certification_scan_finds_every_declaration(tmp_path):
    """The scan is what lets an unimported certified class reach the check above."""
    sources = {
        "leaf.py": "class A:\n    _HOST_PURE_CERTIFIED = True\n",
        "pkg/__init__.py": "class B:\n    _HOST_PURE_CERTIFIED: ClassVar[bool] = True\n",
        "off.py": "class C:\n    _HOST_PURE_CERTIFIED = False\n",
        "module_level.py": "_HOST_PURE_CERTIFIED = True\n",
        "mention.py": "# classes set _HOST_PURE_CERTIFIED = True once pure\n",
    }
    for rel, text in sources.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert _certifying_modules(str(tmp_path)) == [
        "agent_foundation.leaf",
        "agent_foundation.pkg",
    ]


# --- I3 ------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "mode"), _CASES, ids=_CASE_IDS)
def test_i3_host_store_is_json_serializable(measured, name, mode):
    """No live object reaches node state: the store dumps with no ``default=`` hook."""
    assert measured(name, mode).store_errors == ()
