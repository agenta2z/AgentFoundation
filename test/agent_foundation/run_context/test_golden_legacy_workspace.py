"""Characterization goldens for loading a pre-refactor save and workspace (I11).

The refactor must keep resuming from state that today's code writes. Two frozen
fixtures under ``goldens/legacy_workspace/fixture/`` were produced once, with
today's code, by the helpers below. They are data: never regenerate them
(``AF_UPDATE_GOLDENS`` rewrites only the JSON goldens).

* ``host_store.json``: ``RunStateStore.save`` of a host root (``_host_store``)
  after a ``MultiFlowInferencer`` call, which leaves typed ``MultiFlowState`` and
  ``MultiFlowAttemptState`` at ``/`` and dict LWI state on the flows, and after a
  context-scoped ``switch_role`` plus a call on a templated leaf under
  ``/reviewer``, which then left a ``RoleState`` in that node's ``call``
  (``from_json`` now migrates it to ``role_state``).
* ``bta/``: the workspace of the stub BTA (``_bta``) after one sync call,
  written at ``GEN_ROOT`` so the absolute paths embedded in it are stable.

Pinned: the rehydrated node states (types, fields, dict key types, creator tags),
the ``to_json`` round trip, what fresh definitions read under a context built on
the rehydrated store, whether today's code still writes the same save and tree,
and what a fresh BTA does when resumed on the checked-in tree.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import attrs as attrs_api
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
    RunStateStore,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode

from ._golden import check_golden, GOLDEN_DIR, Normalizer, workspace_tree

FIXTURE_DIR = GOLDEN_DIR / "legacy_workspace" / "fixture"
STORE_FIXTURE = FIXTURE_DIR / "host_store.json"
BTA_FIXTURE = FIXTURE_DIR / "bta"
GEN_ROOT = "/tmp/af_golden_legacy/bta"


class _Templates:
    def __call__(
        self, key, *, active_template_root_space=None, master_version=None, **feed
    ):
        return f"[{active_template_root_space}|{key}] {feed.get('input')}"

    def get_raw_template(self, *a, **k):
        return "x"

    def load_variables(self, variable_specs=None, root_space="", **k):
        if "task_instructions" not in (variable_specs or {}):
            return {}
        return {"task_instructions": f"Contract<{root_space}> for {{{{ input }}}}"}

    def _resolve_templated_feed(self, feed, root_space=""):
        text = feed["task_instructions"].replace("{{ input }}", str(feed.get("input")))
        return {**feed, "task_instructions": text}

    def add_template_root(self, *a, **k):
        pass

    def __deepcopy__(self, memo):
        return self


@attrs(slots=False)
class _Leaf(TemplatedInferencerBase):
    label = attrib(default="leaf")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"{self.label}({inference_input})"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input)


@attrs(slots=False)
class _Pure(InferencerBase):
    response = attrib(default="ok")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.response(str(inference_input))

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input)


class _Recorder(list):
    def __deepcopy__(self, memo):
        return self


@attrs
class _M(InferencerBase):
    _response = attrib(default="mock")
    _calls = attrib(factory=_Recorder)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._calls.append(self._response)
        return f"{self._response}:{inference_input}"


def _leaf(label: str) -> _Leaf:
    return _Leaf(
        label=label,
        template_manager=_Templates(),
        template_root_space="plan",
        template_key="initial",
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )


def _tag(raw: str, name: str) -> str:
    return raw.split(f"<{name}>", 1)[1].split(f"</{name}>", 1)[0]


def _aggregate(prompt: str) -> str:
    w = 1 if "task-a" in prompt else 0
    return f"<W>{w}</W><R>rev{w}</R><F>fix{w}</F><K>{w},{1 - w}</K>"


def _stop(state, out):
    return True


def _mfi() -> MultiFlowInferencer:
    flows = [
        {"initial_inferencer": _leaf(n), "followup_inferencer": _leaf(n + "-fu")}
        for n in ("fa", "fb")
    ]
    for flow in flows:
        flow.update(end_condition=_stop, max_dynamic_steps=1)
    return MultiFlowInferencer(
        flow_configs=flows,
        aggregator_inferencer=_Pure(response=_aggregate),
        winner_parser=lambda raw: int(_tag(raw, "W")),
        reviewer_alias_parser=lambda raw: _tag(raw, "R"),
        fixer_alias_parser=lambda raw: _tag(raw, "F"),
        ranking_parser=lambda raw: [int(x) for x in _tag(raw, "K").split(",")],
        propagate_runtime_input=True,
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )


def _under(ctx: RunContext, fn: Callable[[], Any]) -> Any:
    token = enter_run(ctx)
    try:
        return fn()
    finally:
        exit_run(token)


def _host_store(root_dir: Path) -> Tuple[RunStateStore, Dict[str, Any]]:
    root = RunContext.root(workspace=InferencerWorkspace(root=str(root_dir / "host")))
    info: Dict[str, Any] = {"mfi": _mfi().infer("task-a", run_context=root)}
    reviewer = _leaf("rev")

    def review() -> str:
        reviewer.switch_role(
            "reviewer", template_key="review", template_root_space="review"
        )
        return reviewer.infer("check task-a")

    info["reviewer"] = _under(root.child("reviewer"), review)
    info["reviewer_instance_role"] = [
        reviewer.template_key,
        reviewer.template_root_space,
    ]
    return root._store, info


def _state_view(value: Any) -> Any:
    if value is None:
        return None
    if not attrs_api.has(type(value)):
        return {"type": type(value).__name__, "value": value}
    out: Dict[str, Any] = {"type": type(value).__name__}
    for field in attrs_api.fields(type(value)):
        item = getattr(value, field.name)
        out[field.name] = _state_view(item) if attrs_api.has(type(item)) else item
        if isinstance(item, dict) and item:
            out[f"{field.name}.key_types"] = sorted({type(k).__name__ for k in item})
    return out


def _node_view(node) -> Dict[str, Any]:
    return {
        "call": _state_view(node.call),
        "attempt": _state_view(node.attempt),
        "role_state": _state_view(node.role_state),
        "conversation": node.conversation,
        "checkpoints": node.checkpoints,
        "provenance": node.provenance,
        "creator": node._creator,
    }


def _mfi_reads(m: MultiFlowInferencer) -> Dict[str, Any]:
    winner = m.get_winner_inferencer()
    return {
        "get_winner_flow_idx": m.get_winner_flow_idx(),
        "get_winner_inferencer": None if winner is None else winner.label,
        "get_chosen_reviewer_alias": m.get_chosen_reviewer_alias(),
        "get_chosen_fixer_alias": m.get_chosen_fixer_alias(),
        "get_ranking": m.get_ranking(),
        "get_runner_up_flow_idx": m.get_runner_up_flow_idx(),
    }


def _role_reads(leaf: _Leaf) -> Dict[str, Any]:
    return {
        "effective_role": list(leaf._effective_role()),
        "active_role_name": leaf._active_role_name(),
        "infer": leaf.infer("again"),
        "_proposer_task_instructions": leaf._proposer_task_instructions(),
    }


def _resumed_reads(store: RunStateStore) -> Dict[str, Any]:
    root = RunContext.root(store=store)
    mfi, reviewer = _mfi(), _leaf("rev")
    return {
        "mfi_no_ctx": _mfi_reads(mfi),
        "mfi_under_root": _under(root, lambda: _mfi_reads(mfi)),
        "reviewer_under_child": _under(
            root.child("reviewer"), lambda: _role_reads(reviewer)
        ),
        "reviewer_instance_role": [reviewer.template_key, reviewer.template_root_space],
    }


def test_host_store_rehydrates(tmp_path):
    """The fixture's ``RoleState`` in ``node.call`` loads as ``node.role_state``,
    so neither the round trip nor a fresh save reproduces the fixture;
    ``MultiFlowAttemptState.latest_per_flow`` is written with ``int`` keys and
    comes back with ``str`` keys."""
    saved = json.loads(STORE_FIXTURE.read_text(encoding="utf-8"))
    store = RunStateStore.from_json(saved)
    fresh, generation = _host_store(tmp_path)
    data = {
        "nodes": {path: _node_view(node) for path, node in store._nodes.items()},
        "roundtrip_equal": store.to_json() == saved,
        "fresh_save_equals_fixture": json.loads(json.dumps(fresh.to_json())) == saved,
        "fresh_root_attempt": _state_view(fresh.peek("/").attempt),
        "fresh_generation": generation,
        "resumed_reads": _resumed_reads(store),
    }
    norm = Normalizer({"<WS>": tmp_path})
    check_golden("legacy_workspace/host_store", norm.value(data))


def _bta(root: Path, calls: Optional[List[str]] = None, **kwargs: Any):
    rec = _Recorder() if calls is None else calls
    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_M(response="bd", calls=rec),
        worker_inferencers=lambda sub_query, index: _M(response=f"w{index}", calls=rec),
        aggregator_inferencer=_M(response="agg", calls=rec),
        checkpoint_mode="jsonfy",
        enable_result_save=True,
        resume_with_saved_results=True,
        predefined_sub_queries=["q0", "q1"],
        **kwargs,
    )
    ws = InferencerWorkspace(root=str(root))
    ws.ensure_dirs()
    bta._workspace = ws
    bta.name = "gbta"
    return bta


def _attempt(fn: Callable[[], Any]) -> Dict[str, Any]:
    try:
        return {"result": fn()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _differing(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def test_bta_workspace_fixture(tmp_path):
    """The checked-in tree predates the resume manifest (P8): a fresh instance
    resumed on it fails closed (``UnverifiedLegacyBtaResumeError``) before any stub
    runs or any checkpoint is read; under ``trust_legacy`` it resumes (rebuilding
    the expansion from the sub-queries its breakdown node saved, B37) and writes no
    manifest. A fresh run's tree differs from the fixture only by the manifest and
    the lease file."""
    legacy = tmp_path / "legacy"
    shutil.copytree(BTA_FIXTURE, legacy)
    norm = Normalizer({"<WS>": legacy}, extra=[(re.escape(GEN_ROOT), "<WS>")])
    fixture_tree = workspace_tree(legacy, norm)
    fresh = tmp_path / "fresh"
    fresh_run = _attempt(lambda: _bta(fresh).infer("task"))
    fresh_tree = workspace_tree(fresh, Normalizer({"<WS>": fresh}))
    calls = _Recorder()
    resume = _attempt(lambda: _bta(legacy, calls).infer("task"))
    after = workspace_tree(legacy, norm)
    trusted_calls = _Recorder()
    trusted = _attempt(
        lambda: _bta(
            legacy, trusted_calls, resume_identity_policy="trust_legacy"
        ).infer("task")
    )
    after_trusted = workspace_tree(legacy, norm)
    data = {
        "fixture_tree": fixture_tree,
        "fresh_run": fresh_run,
        "fresh_tree_differs_from_fixture": _differing(fixture_tree, fresh_tree),
        "resume": resume,
        "resume_calls": list(calls),
        "resume_changed_files": _differing(fixture_tree, after),
        "trusted_resume": trusted,
        "trusted_resume_calls": list(trusted_calls),
        "trusted_resume_changed_files": _differing(fixture_tree, after_trusted),
    }
    check_golden(
        "legacy_workspace/bta_workspace", Normalizer({"<WS>": tmp_path}).value(data)
    )
