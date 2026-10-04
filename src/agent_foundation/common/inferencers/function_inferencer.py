# pyre-strict

"""``FunctionInferencer`` — adapt a plain callable into an :class:`InferencerBase`.

The reverse direction of ``@agentic_function``: the decorator turns a typed
Python function into an LLM-backed one, while this turns an ordinary
(deterministic) callable into a first-class inferencer node. It can then be
dropped into a topology (Dual / BTA / MultiFlow / …) anywhere an inferencer is
expected, closing the framework under both directions — code → topology node,
and topology → function.

It is a leaf with no template, no network, and no session: ``_infer`` simply
calls the wrapped function with the (already-rendered, upstream) inference input
and returns its result, which then flows through the standard public
``infer()`` / ``ainfer()`` pipeline (retry, fallback, workspace, RunContext). A
coroutine ``func`` is awaited on the async path; calling ``infer()`` on one
raises instead of silently returning an un-awaited coroutine.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrib, attrs


@attrs
class FunctionInferencer(InferencerBase):
    """An :class:`InferencerBase` whose inference delegates to a plain callable.

    ``func`` receives the inference input — in a topology, the upstream node's
    output — and returns this node's output. Registered as alias ``Function``.
    """

    func: Callable[..., Any] = attrib(kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if not callable(self.func):
            raise TypeError(
                f"FunctionInferencer requires a callable `func`; got {self.func!r}"
            )

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args: Any
    ) -> Any:
        if asyncio.iscoroutinefunction(self.func):
            raise TypeError(
                "FunctionInferencer.func is a coroutine function; use ainfer() "
                "(await the inferencer) instead of the synchronous infer()."
            )
        return self.func(inference_input)

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args: Any
    ) -> Any:
        result = self.func(inference_input)
        if inspect.isawaitable(result):
            return await result
        return result
