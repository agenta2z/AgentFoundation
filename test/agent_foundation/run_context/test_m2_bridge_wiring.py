"""M2: the keyword-only RunContext carrier + bridge wired into InferencerBase.

Verifies the additive, byte-identical wiring:
* ``run_context`` is **keyword-only** on ``infer``/``ainfer`` (can't land in
  ``**_inference_args`` -> the kwarg-leak defense);
* during the call the per-task ``_active_ctx`` is installed (the inferencer can
  read it without it being threaded through ``**_inference_args``);
* ``run_context=None`` legacy-mints a root from the instance workspace;
* a provided ``run_context`` is the active context inside the call.
"""

import asyncio
import inspect

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    active_run_context,
)


@attrs
class CapturingInferencer(InferencerBase):
    """Records the kwargs each ``_infer``/``_ainfer`` saw + the active ctx then."""

    seen_args: dict = attrib(factory=dict, init=False)
    seen_ctx_path: str = attrib(default=None, init=False)

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        self.seen_args = dict(_inference_args)
        ctx = active_run_context()
        self.seen_ctx_path = ctx.path if ctx is not None else None
        return "ok"

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        self.seen_args = dict(_inference_args)
        ctx = active_run_context()
        self.seen_ctx_path = ctx.path if ctx is not None else None
        return "ok"


def test_run_context_is_keyword_only_on_infer_and_ainfer():
    for method in (InferencerBase.infer, InferencerBase.ainfer):
        params = inspect.signature(method).parameters
        assert "run_context" in params
        assert params["run_context"].kind is inspect.Parameter.KEYWORD_ONLY


def test_run_context_does_not_leak_into_inference_args_sync():
    inf = CapturingInferencer()
    inf.infer("x", run_context=None, foo="bar")
    assert "run_context" not in inf.seen_args  # keyword-only -> never in the bag
    assert inf.seen_args.get("foo") == "bar"  # genuine extra kwargs still flow


def test_run_context_does_not_leak_into_inference_args_async():
    inf = CapturingInferencer()
    asyncio.run(inf.ainfer("x", run_context=None, foo="bar"))
    assert "run_context" not in inf.seen_args
    assert inf.seen_args.get("foo") == "bar"


def test_bridge_installs_active_ctx_during_sync_call():
    inf = CapturingInferencer()
    assert active_run_context() is None  # nothing active before
    inf.infer("x")  # legacy None -> mint a root
    assert inf.seen_ctx_path == "/"  # a root was active during the call
    assert active_run_context() is None  # torn down after (finally exit_run)


def test_bridge_installs_active_ctx_during_async_call():
    inf = CapturingInferencer()
    asyncio.run(inf.ainfer("x"))
    assert inf.seen_ctx_path == "/"
    assert active_run_context() is None


def test_explicit_run_context_is_the_active_ctx_inside_the_call():
    inf = CapturingInferencer()
    root = RunContext.root(workspace=None)
    child = root.child("review")
    inf.infer("x", run_context=child)
    assert inf.seen_ctx_path == "/review"  # the explicit ctx was active


def test_explicit_and_legacy_paths_both_succeed_equivalently():
    inf = CapturingInferencer()
    r1 = inf.infer("x")  # legacy mint
    r2 = inf.infer("x", run_context=RunContext.root(workspace=None))  # explicit
    assert r1 == r2 == "ok"


def test_third_positional_arg_is_rejected_keyword_only_protects_the_slot():
    inf = CapturingInferencer()
    # inference_input, inference_config are the only positionals; a 3rd -> TypeError.
    try:
        inf.infer("x", None, "would-be-run-context-positional")
        raised = False
    except TypeError:
        raised = True
    assert raised
