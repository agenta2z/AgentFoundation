# pyre-strict

"""Shared no-network test doubles for the ``@agentic_function`` tests.

The decorator resolves an ``inferencer=`` spec through
:class:`InferencerProvider`; a plain object with no ``__call__`` is not a valid
spec, so these fakes are injected via a zero-arg factory: ``inferencer=lambda:
fake`` (the provider's callable-factory branch builds and caches it once). Each
fake matches the ``InferencerBase.infer`` / ``ainfer`` surface Agent-verified
from source: ``(inference_input, inference_config=None, *, run_context=None,
**_inference_args)``.
"""

from __future__ import annotations

from typing import Any, Dict, List


class FakeInferencer:
    """A no-network stand-in that records how the decorator called it.

    ``response`` is either a fixed value returned verbatim, or a
    ``callable(call_index: int) -> Any`` so a test can vary the reply per attempt.
    """

    def __init__(self, response: Any = "ok") -> None:
        self._response = response
        self.calls: int = 0
        self.prompts: List[Any] = []
        self.run_contexts: List[Any] = []
        self.infer_kwargs: List[Dict[str, Any]] = []

    def _result(self) -> Any:
        if callable(self._response):
            return self._response(self.calls)
        return self._response

    def infer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls += 1
        self.prompts.append(inference_input)
        self.run_contexts.append(run_context)
        self.infer_kwargs.append(dict(kwargs))
        return self._result()

    async def ainfer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.infer(
            inference_input, inference_config, run_context=run_context, **kwargs
        )


class SessionFake(FakeInferencer):
    """Session-bearing fake: also counts ``reset_session()`` calls.

    Mirrors a ``StreamingInferencerBase`` (``auto_resume=True``) whose reused
    instance would resume the prior call's conversation without a reset.
    """

    def __init__(self, response: Any = "ok") -> None:
        super().__init__(response)
        self.resets: int = 0

    def reset_session(self) -> None:
        self.resets += 1


class AsyncOnlyFake(FakeInferencer):
    """``ainfer`` works; ``infer`` fails — proves the async path never uses sync infer."""

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("sync infer() must not be called on the async path")

    async def ainfer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls += 1
        self.prompts.append(inference_input)
        self.run_contexts.append(run_context)
        self.infer_kwargs.append(dict(kwargs))
        return self._result()


class PreflightFake(FakeInferencer):
    """Fake exposing ``preflight_all()`` (returns a list; does NOT raise)."""

    def __init__(self, problems: Any = (), response: Any = "ok") -> None:
        super().__init__(response)
        self._problems = list(problems)

    async def preflight_all(self) -> List[str]:
        return list(self._problems)
