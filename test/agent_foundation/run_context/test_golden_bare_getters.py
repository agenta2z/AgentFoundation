"""Characterization goldens for the documented bare getters (invariant I8).

I8 requires bare calls to keep today's output, documented getter values and
host-store node-path set, and requires that a host call never touch a getter's
backing field. Each golden drives one getter family with local stubs and pins,
per step on ONE instance:

* the call result, every getter value and the backing fields after the call;
* for bare calls: two calls with different inputs and one failing call;
* for host calls (a caller-owned ``RunContext.root``): the getters once the
  call returned (what a caller outside the context reads), the getters with the
  host root re-entered, the backing fields before and after, which of them the
  host call touched, and the sorted node paths in the host store;
* where the entry allows it, calls under an explicitly entered legacy root
  (``bridge.mint_root``). Stubs record the context active inside their call as
  ``ctx_seen``: ``null`` is a true no-ctx call (the Claude ``@bridge_entrypoint``
  ``ainfer`` / ``infer``), ``legacy_mint: true`` a minted root (the streaming
  templates, ``infer_streaming`` included).

Every golden carries a ``catalog`` (getter -> class -> backing fields). The
``catalog`` golden lists every documented getter found, including the ones
pinned by other goldens and the ones not drivable with a local stub.
"""

import asyncio
import json
import sys
from typing import Any, Callable, Dict, List, Optional

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    bridge,
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs

# I8's allow-list: every intentional change to what a bare caller observes, tagged
# with the defect it fixes and the commit group that lands it. A golden in this
# module (or another bare golden) changes only together with an entry here.
BARE_EXPECTED_CHANGES = {
    "B4": "P1: a failed call no longer resumes its own broken session on recovery",
    "B5": "P1: the guardrail's empty-fingerprint window spans the call's retries",
    "B11": "P1: devmate's direct streaming no longer flips `dump_output` off",
    "B26": "P2: every streaming entry honours an explicit `run_context`",
    "B26b": "P2: devmate / claude_code sync streaming honour a configured fan-out",
    "B27": "P2: devmate `ainfer()` no longer binds `inference_config` positionally",
    "B30": "P2: OpenClaw `ainfer` runs the base per-call machinery (one attempt)",
    "B34": "P3 c5: each `parallel_infer` / iterator item initializes its call state",
    "X34": "P3 c6: a render-only or claim-rejected call touches no session state",
    "B29": "P3 c7/c9: no run context or leaf ContextVar is visible between yields",
    "B6(b)": "P3 c11: a bare session reset also clears every live branch, so a "
    "failed bare call's recovery no longer resumes a host branch's session",
    "B16": "P5 c1: a role switched under a legacy root renders its master version, "
    "variables, extra feed and modes (it used to render the definition's)",
    "B17": "P5 c2: a bare call of a templated parent no longer writes its feed and "
    "modes into its child instances (they reach the child through the call's ctx)",
    "B18": "P5 c4: BTA no longer writes its graph-reporter observer / interactive "
    "handler onto its stages (they arrive as the stage call's keywords)",
    "B36": "P6 c3: a result no BTA run produced (an external fallback, a "
    "non-exception default) is finalized as a leaf's, and a host BTA publishes its "
    "run summary at its own node (the host store gains the BTA's path)",
    "B1": "P6 c4: each successful BTA / MFI call outside a host ctx relays its "
    "selected worker contract into `_worker_task_instructions`, sync and async "
    "(it was harvested async-only and never reset)",
    "X40": "P6 c4: the task-contract getter fields are compat projections of the "
    "published contract: a host call writes none of them (`_last_rendered_task_"
    "instructions`, `_worker_task_instructions`)",
    "MFI-rerun": "P6 c2: an interactive rerun is a second attempt of the same call, "
    "so MFI post-processes (normalize, dispatch state, `response_parser`) once "
    "per call instead of once per nested call (golden bta/mfi_interactive_rerun)",
}
from rich_python_utils.common_utils.function_helper import FallbackMode

from ._golden import check_golden, live_branch_label, Normalizer

_MFI_FIELDS = {
    "get_winner_flow_idx": ["_last_winner_idx_backing", "MultiFlowState.winner_idx"],
    "get_winner_inferencer": ["_last_winner_idx_backing", "MultiFlowState.winner_idx"],
    "get_chosen_reviewer_alias": [
        "_last_reviewer_alias_backing",
        "MultiFlowState.reviewer_alias",
    ],
    "get_chosen_fixer_alias": [
        "_last_fixer_alias_backing",
        "MultiFlowState.fixer_alias",
    ],
    "get_ranking": ["_last_ranking_backing", "MultiFlowState.ranking"],
    "get_runner_up_flow_idx": ["_last_ranking_backing", "MultiFlowState.ranking"],
    "get_runner_up_inferencer": ["_last_ranking_backing", "MultiFlowState.ranking"],
    "get_first_non_winner_inferencer": [
        "_last_winner_idx_backing",
        "MultiFlowState.winner_idx",
    ],
    "get_non_winner_inferencers": [
        "_last_winner_idx_backing",
        "MultiFlowState.winner_idx",
    ],
}
_TERMINAL_FIELDS = [
    "_last_streaming_output",
    "_last_streaming_return_code",
    "_last_streaming_stderr",
]
_SESSION_FIELDS = ["_session_id", "_live_handle_store[ctx.path].live_session_id"]

CATALOG: Dict[str, Dict[str, List[str]]] = {
    "_proposer_task_instructions": {
        "InferencerBase": [],
        "TemplatedInferencerBase": ["_last_rendered_task_instructions"],
        "MultiFlowInferencer": [
            "MultiFlowState.winner_idx",
            "_last_winner_idx_backing",
            "flow_configs[*].initial_inferencer._proposer_task_instructions()",
        ],
        "BreakdownThenAggregateInferencer": ["_worker_task_instructions"],
        "DualInferencer": [
            "_state['prior_task_instructions'] (node.call | _pending_state_backing)",
            "base_inferencer._proposer_task_instructions()",
        ],
    },
    **{name: {"MultiFlowInferencer": fields} for name, fields in _MFI_FIELDS.items()},
    "active_session_id": {"StreamingInferencerBase": _SESSION_FIELDS},
    "get_final_output": {
        "StreamingInferencerBase": [],
        "RovoDevCliInferencer": ["_last_clean_output", "_last_raw_stdout"],
    },
    "get_streaming_result": {
        "ClaudeCodeCliInferencer": _TERMINAL_FIELDS,
        "DevmateCliInferencer": [
            "_last_streaming_output",
            "_last_streaming_return_code",
        ],
    },
    "get_messages": {"ConversationalInferencer": ["_messages"]},
}

COVERAGE = {
    "driven_here": {
        "_proposer_task_instructions": [
            "TemplatedInferencerBase",
            "MultiFlowInferencer",
            "BreakdownThenAggregateInferencer",
            "DualInferencer",
        ],
        **{name: ["MultiFlowInferencer"] for name in _MFI_FIELDS},
        "active_session_id": ["StreamingInferencerBase", "ClaudeCodeCliInferencer"],
        "get_final_output": ["StreamingInferencerBase"],
        "get_streaming_result": ["ClaudeCodeCliInferencer"],
    },
    "driven_elsewhere": {
        "get_final_output.RovoDevCliInferencer": "test_golden_streaming_rovodev.py",
        "get_streaming_result.DevmateCliInferencer": "test_golden_streaming_devmate.py",
        "get_streaming_result.ClaudeCodeCliInferencer (per entrypoint)": (
            "test_golden_streaming_claude_code.py"
        ),
        "active_session_id (devmate, rovodev, claude_code, openclaw leaves)": (
            "test_golden_streaming_*.py, test_golden_openclaw.py"
        ),
    },
    "not_driven": {
        "get_messages.ConversationalInferencer": (
            "reflects _messages, written only by the server run_agentic_loop; "
            "the public infer/ainfer path renders the real SOP registry"
        ),
        "last_call (agentic_functions/decorator.py)": (
            "a ContextVar trace on an agentic function, not an instance field"
        ),
    },
    "fields_without_documented_getter": [
        "SDK leaves _last_token_count / _last_tool_use_count / _last_usage",
        "ToolAsInferencer._last_response",
        "ClaudeCodeCliInferencer._last_stream_result",
    ],
}


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


def _ctx_view(ctx) -> Optional[Dict[str, Any]]:
    return None if ctx is None else {"path": ctx.path, "legacy_mint": ctx.legacy_mint}


def _see(inf) -> None:
    inf.__dict__.setdefault("seen_ctx", []).append(_ctx_view(active_run_context()))


@attrs(slots=False)
class _Leaf(TemplatedInferencerBase):
    label = attrib(default="leaf")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        _see(self)
        if "boom" in str(inference_input):
            raise RuntimeError("boom")
        return f"{self.label}({inference_input})"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input)


@attrs(slots=False)
class _Pure(InferencerBase):
    response = attrib(default="ok")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return (
            self.response(str(inference_input))
            if callable(self.response)
            else self.response
        )

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input)


@attrs(slots=False)
class _Session(StreamingInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        _see(self)
        return f"sync<{inference_input}>"

    async def _ainfer_streaming(self, prompt, **kwargs):
        _see(self)
        if "boom" in str(prompt):
            raise RuntimeError("boom")
        self.active_session_id = f"sid<{prompt}>"
        yield f"reply<{prompt}>"


class _Claude(ClaudeCodeCliInferencer):
    def construct_command(self, inference_input, **kwargs):
        _see(self)
        return super().construct_command(inference_input, **kwargs)


FAKE_CLAUDE = r"""
import json, sys
argv = sys.argv[1:]
fmt = argv[argv.index("--output-format") + 1] if "--output-format" in argv else None
prompt = sys.stdin.read() if fmt else argv[-1]
sys.stderr.write("err<%s>\n" % prompt)
if fmt == "stream-json":
    delta = {"type": "text_delta", "text": "stream<%s>" % prompt}
    inner = {"type": "content_block_delta", "delta": delta}
    print(json.dumps({"type": "stream_event", "event": inner}))
    sid = "sid-" + prompt.replace(" ", "_")
    res = {"type": "result", "result": "stream<%s>" % prompt, "session_id": sid}
    print(json.dumps(res))
else:
    print("plain<%s>" % prompt)
sys.exit(3 if "fail" in prompt else 0)
"""

Reader = Callable[[Any], Dict[str, Any]]


def _plain(value: Any) -> Any:
    if hasattr(value, "return_code") and hasattr(value, "output"):
        keys = ("output", "return_code", "success", "session_id")
        return {k: getattr(value, k, None) for k in keys}
    try:
        json.dumps(value)
    except TypeError:
        return {"type": type(value).__name__, "str": str(value)}
    return value


def _call(fn: Callable[[], Any]) -> Any:
    try:
        return _plain(fn())
    except Exception as exc:
        return {"error": type(exc).__name__}


def _under(ctx: RunContext, fn: Callable[[], Any]) -> Any:
    token = enter_run(ctx)
    try:
        return fn()
    finally:
        exit_run(token)


def _with_seen(inf, out: Dict[str, Any]) -> Dict[str, Any]:
    seen = inf.__dict__.pop("seen_ctx", None)
    if seen:
        out["ctx_seen"] = seen
    return out


def _bare(
    inf, fn: Callable[[], Any], getters: Reader, fields: Reader
) -> Dict[str, Any]:
    result = _call(fn)
    return _with_seen(
        inf, {"result": result, "getters": getters(inf), "fields": fields(inf)}
    )


def _host(
    inf, run: Callable[[RunContext], Any], getters: Reader, fields: Reader, ws=None
):
    root = RunContext.root(workspace=InferencerWorkspace(root=str(ws)) if ws else None)
    before = fields(inf)
    out = {"result": _call(lambda: run(root))}
    out["store_paths"] = sorted(root._store._nodes)
    out["getters_after"] = getters(inf)
    out["getters_reentered"] = _under(root, lambda: getters(inf))
    after = fields(inf)
    out["fields_before"], out["fields_after"] = before, after
    out["touched"] = sorted(k for k in before if before[k] != after[k])
    return _with_seen(inf, out)


def _golden(
    name: str, tmp_path, getter_names: List[str], steps: Dict[str, Any]
) -> None:
    catalog = {g: CATALOG[g] for g in getter_names}
    data = {"catalog": catalog, "steps": steps}
    check_golden(f"getters/{name}", Normalizer({"<WS>": tmp_path}).value(data))


def _sync(inf, text: str) -> Callable[[], Any]:
    return lambda: inf.infer(text)


def _async(inf, text: str) -> Callable[[], Any]:
    return lambda: asyncio.run(inf.ainfer(text))


def _host_sync(inf, text: str) -> Callable[[RunContext], Any]:
    return lambda root: inf.infer(text, run_context=root)


def _host_async(inf, text: str) -> Callable[[RunContext], Any]:
    return lambda root: asyncio.run(inf.ainfer(text, run_context=root))


def _leaf(label: str, space: str = "plan") -> _Leaf:
    return _Leaf(
        label=label,
        template_manager=_Templates(),
        template_root_space=space,
        template_key="initial",
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )


def _label(inf) -> Optional[str]:
    return None if inf is None else inf.label


def _contract(inf) -> Dict[str, Any]:
    return {"_proposer_task_instructions": inf._proposer_task_instructions()}


def _snapshot_field(inf) -> Dict[str, Any]:
    return {"_last_rendered_task_instructions": inf._last_rendered_task_instructions}


def test_templated_leaf_getter(tmp_path):
    """B10: the snapshot is last-render-on-the-object outside host mode; the host
    call publishes its contract at its node and leaves the field alone (X40)."""
    leaf = _leaf("t")
    steps = {
        "bare1": _bare(leaf, _sync(leaf, "task-a"), _contract, _snapshot_field),
        "bare2": _bare(leaf, _sync(leaf, "task-b"), _contract, _snapshot_field),
        "bare3_fail": _bare(leaf, _sync(leaf, "boom-c"), _contract, _snapshot_field),
        "host": _host(
            leaf, _host_sync(leaf, "task-h"), _contract, _snapshot_field, tmp_path
        ),
    }
    _golden("templated_leaf", tmp_path, ["_proposer_task_instructions"], steps)


def _tag(raw: str, name: str) -> Optional[str]:
    start, end = f"<{name}>", f"</{name}>"
    return raw.split(start, 1)[1].split(end, 1)[0] if start in raw else None


def _aggregate(prompt: str) -> str:
    if "boom" in prompt:
        raise RuntimeError("aggregator boom")
    w = 1 if "task-a" in prompt else 0
    return f"<W>{w}</W><R>rev{w}</R><F>fix{w}</F><K>{w},{1 - w}</K>"


def _stop(state, out):
    return True


def _mfi(ws) -> MultiFlowInferencer:
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
        workspace=ws,
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )


def _mfi_getters(m) -> Dict[str, Any]:
    return {
        "get_winner_flow_idx": m.get_winner_flow_idx(),
        "get_winner_inferencer": _label(m.get_winner_inferencer()),
        "get_chosen_reviewer_alias": m.get_chosen_reviewer_alias(),
        "get_chosen_fixer_alias": m.get_chosen_fixer_alias(),
        "get_ranking": m.get_ranking(),
        "get_runner_up_flow_idx": m.get_runner_up_flow_idx(),
        "get_runner_up_inferencer": _label(m.get_runner_up_inferencer()),
        "get_first_non_winner_inferencer": _label(m.get_first_non_winner_inferencer()),
        "get_non_winner_inferencers": [
            _label(i) for i in m.get_non_winner_inferencers()
        ],
        "_proposer_task_instructions": m._proposer_task_instructions(),
    }


def _mfi_fields(m) -> Dict[str, Any]:
    names = ("winner_idx", "reviewer_alias", "fixer_alias", "ranking")
    out = {f"_last_{n}_backing": m.__dict__.get(f"_last_{n}_backing") for n in names}
    for cfg in m.flow_configs:
        leaf = cfg["initial_inferencer"]
        out[f"{leaf.label}._last_rendered_task_instructions"] = (
            leaf._last_rendered_task_instructions
        )
    return out


def test_multi_flow_getters(tmp_path):
    """After a host call the getters read the backing left by the previous bare
    call; only the re-entered host root shows the host call's dispatch state.
    A failing call resets the dispatch state (B34 covers the other entries). The
    host call leaves the flow leaves' contract fields alone (X40)."""
    m = _mfi(InferencerWorkspace(root=str(tmp_path / "mfi")))
    steps = {
        "bare1": _bare(m, _sync(m, "task-a"), _mfi_getters, _mfi_fields),
        "bare2": _bare(m, _sync(m, "task-b"), _mfi_getters, _mfi_fields),
        "host": _host(
            m, _host_sync(m, "task-a-host"), _mfi_getters, _mfi_fields, tmp_path
        ),
        "bare3_fail": _bare(m, _sync(m, "boom"), _mfi_getters, _mfi_fields),
    }
    _golden(
        "multi_flow", tmp_path, [*_MFI_FIELDS, "_proposer_task_instructions"], steps
    )


def _breakdown(prompt: str) -> str:
    return f"1. {prompt} part1\n2. {prompt} part2"


def _bta(ws) -> BreakdownThenAggregateInferencer:
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Pure(response=_breakdown),
        worker_inferencers=lambda sub_query, index: _leaf(f"w{index}", "impl"),
        aggregator_inferencer=_Pure(response="agg"),
        workspace=ws,
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )


def _bta_field(b) -> Dict[str, Any]:
    return {"_worker_task_instructions": b._worker_task_instructions}


def _bta_steps(tmp_path, mode: str) -> Dict[str, Any]:
    run, host = (_sync, _host_sync) if mode == "sync" else (_async, _host_async)
    b = _bta(InferencerWorkspace(root=str(tmp_path / f"bta_{mode}")))
    return {
        "bare1": _bare(b, run(b, "task-a"), _contract, _bta_field),
        "bare2": _bare(b, run(b, "task-b"), _contract, _bta_field),
        "bare3_fail": _bare(b, run(b, "boom"), _contract, _bta_field),
        "host": _host(
            b, host(b, "task-h"), _contract, _bta_field, tmp_path / f"h_{mode}"
        ),
    }


def test_bta_getter(tmp_path):
    """B1 (fixed in P6 c4): each successful call outside a host ctx relays its
    lowest-index worker's contract, sync and async; a failing call and a host call
    leave the getter field alone. A bare BTA without a workspace fails before any
    worker runs."""
    no_ws = _bta(None)
    steps = {
        "sync": _bta_steps(tmp_path, "sync"),
        "async": _bta_steps(tmp_path, "async"),
        "bare_no_workspace": _bare(
            no_ws, _sync(no_ws, "task-a"), _contract, _bta_field
        ),
    }
    _golden("bta", tmp_path, ["_proposer_task_instructions"], steps)


def _dual() -> DualInferencer:
    return DualInferencer(
        base_inferencer=_leaf("base"),
        review_inferencer=_Pure(
            response="## Review\nSeverity: COSMETIC\nApproved: true"
        ),
        fixer_inferencer=None,
        consensus_config=ConsensusConfig(),
    )


def _dual_getters(d) -> Dict[str, Any]:
    return {**_contract(d), "_prior_task_instructions": d._prior_task_instructions()}


def _dual_fields(d) -> Dict[str, Any]:
    backing = d.__dict__.get("_pending_state_backing")
    prior = (
        backing.get("prior_task_instructions") if isinstance(backing, dict) else None
    )
    return {
        "_pending_state_backing.prior_task_instructions": prior,
        "_pending_state_backing_is_set": backing is not None,
        **{f"base.{k}": v for k, v in _snapshot_field(d.base_inferencer).items()},
    }


def test_dual_getter(tmp_path):
    """Bare calls leave no stored snapshot, so the getter falls back to the base
    leaf's last render; the host snapshot is readable only inside the host root,
    and the host call leaves the base leaf's field alone (X40)."""
    d = _dual()
    steps = {
        "bare1": _bare(d, _sync(d, "task-a"), _dual_getters, _dual_fields),
        "bare2": _bare(d, _sync(d, "task-b"), _dual_getters, _dual_fields),
        "host": _host(
            d, _host_sync(d, "task-h"), _dual_getters, _dual_fields, tmp_path
        ),
        "bare3_fail": _bare(d, _sync(d, "boom"), _dual_getters, _dual_fields),
    }
    _golden("dual", tmp_path, ["_proposer_task_instructions"], steps)


def _store_view(inf) -> Optional[Dict[str, Any]]:
    store = inf.__dict__.get("_live_handle_store")
    if store is None:
        return None
    return {
        live_branch_label(key): dict(h._data)
        for key, h in sorted(store._by_path.items())
    }


def _session_getters(s) -> Dict[str, Any]:
    return {
        "active_session_id": s.active_session_id,
        "get_final_output": s.get_final_output(),
    }


def _session_fields(s) -> Dict[str, Any]:
    return {"_session_id": s._session_id, "_live_handle_store": _store_view(s)}


def test_streaming_session_getters(tmp_path):
    """B4/B6: after a host call the stale bare backing outranks the host's live
    slot. A failing bare call resets the session before recovering, which clears
    the backing and every live branch (B6(b), fixed in P3 c11): the recovery no
    longer resumes the host branch's session (initial attempt, then a plain
    re-run), and later bare reads surface no session until a call sets one."""
    s = _Session(max_retry=0)
    g, f = _session_getters, _session_fields
    legacy = bridge.mint_root(None)
    steps = {
        "bare1": _bare(s, _async(s, "task-a"), g, f),
        "bare2": _bare(s, _async(s, "task-b"), g, f),
        "host": _host(s, _host_async(s, "task-h"), g, f),
        "bare3_fail": _bare(s, _async(s, "boom"), g, f),
        "legacy_entered": _bare(s, lambda: _under(legacy, _async(s, "task-l")), g, f),
        "bare_sync_infer": _bare(s, _sync(s, "task-s"), g, f),
    }
    _golden(
        "streaming_session", tmp_path, ["active_session_id", "get_final_output"], steps
    )


def _claude_getters(c) -> Dict[str, Any]:
    res = c.get_streaming_result()
    return {
        "get_streaming_result": {**_plain(res), "stderr": res.stderr},
        "active_session_id": c.active_session_id,
    }


def _claude_fields(c) -> Dict[str, Any]:
    out = {name: getattr(c, name) for name in _TERMINAL_FIELDS}
    out.update(_session_fields(c))
    return out


def _claude(tmp_path, monkeypatch) -> _Claude:
    monkeypatch.delenv("CLAUDE_CODE_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_MAX_CONCURRENCY", raising=False)
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE, encoding="utf-8")
    return _Claude(
        claude_command=f"{sys.executable} {script}",
        target_path=str(tmp_path),
        max_retry=0,
    )


def _stream(c, text: str) -> Callable[[], Any]:
    return lambda: "".join(c.infer_streaming(text))


def test_claude_cli_stream_result_getter(tmp_path, monkeypatch):
    """B28 (fixed in P10): ``ainfer`` parses its own transport result — its
    stream reports only stderr, so output comes from the stream and the return
    code keeps its default (§23 F1.2) — instead of the output and return code an
    earlier ``infer_streaming`` left; a host call leaves the compat fields alone.
    A bare ``ainfer`` skips the bridge, and its streaming pipeline runs in that
    no-ctx mode."""
    c = _claude(tmp_path, monkeypatch)
    g, f = _claude_getters, _claude_fields
    legacy = bridge.mint_root(None)
    steps = {
        "infer_streaming1": _bare(c, _stream(c, "p1"), g, f),
        "infer_streaming2_fail": _bare(c, _stream(c, "fail p2"), g, f),
        "ainfer_bare": _bare(c, _async(c, "p3"), g, f),
        "host_ainfer": _host(c, _host_async(c, "p4"), g, f),
        "legacy_entered_ainfer": _bare(
            c, lambda: _under(legacy, _async(c, "p5")), g, f
        ),
        "ainfer_bare_fail": _bare(c, _async(c, "fail p6"), g, f),
        "infer_bare": _bare(c, _sync(c, "p7"), g, f),
    }
    names = ["get_streaming_result", "active_session_id"]
    _golden("claude_cli_stream_result", tmp_path, names, steps)


def test_getter_catalog(tmp_path):
    data = {"catalog": CATALOG, "coverage": COVERAGE}
    check_golden("getters/catalog", Normalizer({"<WS>": tmp_path}).value(data))
