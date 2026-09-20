# pyre-strict

"""The ``@agentic_function`` decorator — a coding surface on ``InferencerBase``.

Turns a plain typed function into one whose body is (or is fronted by) an LLM
call: the call arguments feed a co-located Jinja2 template, a resolved
inferencer produces the output, and a typed parse pipeline turns that output
into the declared return type. Deterministic and inference-backed code compose
behind the identical call shape (the code-agentic hybrid).

The body's *role* is fixed at decoration from the signature alone (no source /
AST inspection): a body with no ``response`` parameter runs first as a
deterministic pre-attempt (a ``pass`` body is the degenerate always-infer case);
a body with a ``response`` parameter parses the inference output; a body whose
``response`` defaults to :data:`FromInference` does both (on different branches).

The decorator returns a callable object (not a wrapped function) so it can carry
a live introspection surface — ``.inferencer``, ``.last_call``, ``.render()``,
``.with_inference_args()`` — as real properties, and implements ``__get__`` so a
decorated *method* still binds ``self`` like an ordinary function would.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import re
import threading
import time
import typing
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from agent_foundation.common.inferencers.agentic_functions.config import (
    InferencerProvider,
)
from agent_foundation.common.inferencers.agentic_functions.errors import (
    AgenticFunctionConfigurationError,
    ParseError,
)
from agent_foundation.common.inferencers.agentic_functions.output import (
    Agentic,
    AgenticOutput,
    FromInference,
)
from agent_foundation.common.inferencers.agentic_functions.parsers import (
    default_parser,
    resolve_parser,
    resolve_return_annotation,
    sequence_is_multi,
)
from agent_foundation.common.inferencers.agentic_functions.rendering import (
    resolve_template_source,
    TemplateSource,
)
from agent_foundation.common.inferencers.agentic_functions.trace import (
    AgenticFunctionTrace,
    new_trace_var,
)
from agent_foundation.common.inferencers.run_context.bridge import active_run_context
from agent_foundation.common.response_parsers.result_text import extract_result_text
from rich_python_utils.common_utils.async_utils import call_maybe_async

_logger: logging.Logger = logging.getLogger(__name__)


class _Raise:
    """Sentinel: ``fallback`` unset → re-raise the last parse error."""

    def __repr__(self) -> str:
        return "RAISE"


_RAISE: _Raise = _Raise()

_PRE_ATTEMPT = "pre_attempt"
_POST_PARSER = "post_parser"
_MIXED = "mixed"


def agentic_function(
    *,
    inferencer: Any = None,
    inferencer_kwargs: Optional[Mapping[str, Any]] = None,
    template: Any = None,
    template_string: Optional[str] = None,
    template_key: Optional[str] = None,
    template_root_space: Optional[str] = None,
    template_master_version: Optional[str] = None,
    parser: Any = None,
    response_param: str = "response",
    escalate_on_none: Optional[bool] = None,
    parse_max_retries: int = 0,
    retry_on: Any = (ParseError,),
    repair_prompt_builder: Optional[Callable[[str, BaseException], str]] = None,
    fallback: Any = _RAISE,
    redact_arguments: Tuple[str, ...] = (),
    run_context_slot: Any = None,
    infer_kwargs: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Any]:
    """Decorate a typed function so its body is backed by an inference call.

    See the module docstring for the body-role model. ``inferencer`` is
    polymorphic (default ``ClaudeApiInferencer`` | instance | config name |
    ``InferencerConfig`` | inline mapping | zero-arg factory). The prompt comes
    from exactly one of ``template_string`` (inline source), ``template`` as a
    module-relative file path, or ``template`` as a ``TemplateManager`` — which
    ``template_key`` / ``template_root_space`` / ``template_master_version``
    then address. ``parser`` is the reusable stage-1 decode; the loop retries
    ``retry_on`` up to ``parse_max_retries + 1`` times re-rendering a repair
    prompt, then applies ``fallback``.
    """
    opts: Dict[str, Any] = {
        "inferencer": inferencer,
        "inferencer_kwargs": inferencer_kwargs,
        "template": template,
        "template_string": template_string,
        "template_key": template_key,
        "template_root_space": template_root_space,
        "template_master_version": template_master_version,
        "parser": parser,
        "response_param": response_param,
        "escalate_on_none": escalate_on_none,
        "parse_max_retries": parse_max_retries,
        "retry_on": retry_on,
        "repair_prompt_builder": repair_prompt_builder,
        "fallback": fallback,
        "redact_arguments": redact_arguments,
        "run_context_slot": run_context_slot,
        "infer_kwargs": infer_kwargs,
    }

    def _decorate(fn: Callable[..., Any]) -> Any:
        cls = (
            _AsyncAgenticFunction
            if inspect.iscoroutinefunction(fn)
            else _SyncAgenticFunction
        )
        wrapper = cls(fn, opts)
        functools.update_wrapper(wrapper, fn)
        wrapper.__signature__ = wrapper.sig
        return wrapper

    return _decorate


class _AgenticFunctionBase:
    """Resolved configuration + call machinery backing one decorated function.

    The two subclasses differ only in ``__call__`` (sync vs. async); all
    analysis, rendering, inference, parsing, retry, and the introspection surface
    live here so the two paths cannot drift.
    """

    def __init__(self, fn: Callable[..., Any], opts: Mapping[str, Any]) -> None:
        self.fn = fn
        self._opts: Dict[str, Any] = dict(opts)
        self.sig: inspect.Signature = inspect.signature(fn)
        self.is_async: bool = inspect.iscoroutinefunction(fn)

        self.response_param: str = opts["response_param"]
        self.template: Any = opts["template"]
        self.template_string: Optional[str] = opts["template_string"]
        self.template_key: Optional[str] = opts["template_key"]
        self.template_root_space: Optional[str] = opts["template_root_space"]
        self.template_master_version: Optional[str] = opts["template_master_version"]
        self.parser_spec: Any = opts["parser"]
        self.parse_max_retries: int = opts["parse_max_retries"]
        if self.parse_max_retries < 0:
            raise AgenticFunctionConfigurationError(
                f"parse_max_retries must be >= 0, got {self.parse_max_retries}"
            )
        retry_on = opts["retry_on"]
        self.retry_on: Tuple[type, ...] = (
            retry_on if isinstance(retry_on, tuple) else (retry_on,)
        )
        self.repair_prompt_builder: Optional[Callable[[str, BaseException], str]] = (
            opts["repair_prompt_builder"]
        )
        self.fallback: Any = opts["fallback"]
        self.redact_arguments: "frozenset[str]" = frozenset(opts["redact_arguments"])
        self.infer_kwargs: Dict[str, Any] = dict(opts["infer_kwargs"] or {})
        self.provider = InferencerProvider(
            opts["inferencer"], opts["inferencer_kwargs"]
        )
        self.trace_var = new_trace_var(f"agentic::{fn.__module__}.{fn.__qualname__}")

        self._self_param = self._detect_self_param()
        self.role = self._classify_role()
        self.escalate_on_none = self._resolve_escalate_on_none(opts["escalate_on_none"])
        self.feed_params = self._compute_feed_params()
        slot = opts["run_context_slot"]
        self._slot_spec: Any = slot if slot is not None else _sanitize_slot(fn.__name__)

        self._prepare_lock = threading.Lock()
        self._source: Optional[TemplateSource] = None

    # -- decoration-time analysis ------------------------------------------

    def _detect_self_param(self) -> Optional[str]:
        params = list(self.sig.parameters)
        if params and params[0] in ("self", "cls"):
            return params[0]
        return None

    def _classify_role(self) -> str:
        param = self.sig.parameters.get(self.response_param)
        if param is None:
            return _PRE_ATTEMPT
        if param.default is FromInference:
            return _MIXED
        return _POST_PARSER

    def _resolve_escalate_on_none(self, explicit: Optional[bool]) -> bool:
        if explicit is not None:
            return bool(explicit)
        annotation = resolve_return_annotation(self.fn)
        if _annotation_admits_none(annotation):
            raise AgenticFunctionConfigurationError(
                f"{self.fn.__qualname__} returns an Optional/None type, so a bare "
                f"None return is ambiguous (valid value vs. escalation signal); "
                f"pass escalate_on_none=True or =False explicitly"
            )
        return True

    def _compute_feed_params(self) -> "frozenset[str]":
        excluded = {self.response_param}
        if self._self_param is not None:
            excluded.add(self._self_param)
        return frozenset(name for name in self.sig.parameters if name not in excluded)

    # -- lazy first-call setup ---------------------------------------------

    def _prepare(self) -> TemplateSource:
        """Resolve and validate the template once; return the cached source."""
        source = self._source
        if source is not None:
            return source
        with self._prepare_lock:
            source = self._source
            if source is None:
                source = resolve_template_source(
                    self.fn,
                    template=self.template,
                    template_string=self.template_string,
                    template_key=self.template_key,
                    template_root_space=self.template_root_space,
                    template_master_version=self.template_master_version,
                )
                source.validate(self.feed_params)
                self._source = source
            return source

    # -- per-call helpers ---------------------------------------------------

    def _bind(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        # A post-parser body's response slot is filled by inference, never by the
        # caller, so a required (no-default) keyword-only `response` would make
        # sig.bind() reject a normal call. Supply a placeholder for that slot only;
        # _render_feed excludes response_param, so it never reaches the feed, and a
        # genuinely-missing required arg (e.g. the prompt input) still raises.
        param = self.sig.parameters.get(self.response_param)
        if (
            param is not None
            and param.default is inspect.Parameter.empty
            and self.response_param not in call_kwargs
        ):
            call_kwargs[self.response_param] = FromInference
        bound = self.sig.bind(*args, **call_kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def _render_feed(self, bound: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            name: value
            for name, value in bound.items()
            if name in self.feed_params and name not in self.redact_arguments
        }

    def _slot(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        spec = self._slot_spec
        if callable(spec):
            return _sanitize_slot(str(spec(*args, **kwargs)))
        return typing.cast(str, spec)

    def _resolve_run_context(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        caller_rc = self.infer_kwargs.get("run_context")
        if caller_rc is not None:
            return caller_rc
        ctx = active_run_context()
        if ctx is not None:
            return ctx.child(self._slot(args, kwargs))
        return None

    def _call_kwargs(self) -> Dict[str, Any]:
        return {k: v for k, v in self.infer_kwargs.items() if k != "run_context"}

    def _repair(self, prompt: str, err: Optional[BaseException]) -> str:
        if self.repair_prompt_builder is not None:
            return self.repair_prompt_builder(prompt, err)  # type: ignore[arg-type]
        return (
            f"{prompt}\n\n[Retry] The previous response could not be parsed "
            f"({err}). Respond again in exactly the requested format."
        )

    def _stage1(
        self, out: AgenticOutput, parser_override: Any
    ) -> Tuple[Any, bool, Tuple[str, ...]]:
        spec = parser_override if parser_override is not None else self.parser_spec
        if spec is None:
            return out, False, ()
        value = resolve_parser(spec)(out)
        label = "stage1_seq" if sequence_is_multi(spec) else "stage1"
        return value, True, (label,)

    def _decide(self, result: Any) -> Optional[Agentic]:
        """Map a pass-1 body result to an escalation carrier, or None to return it."""
        if isinstance(result, Agentic):
            return result
        if result is None and self.escalate_on_none:
            return Agentic()
        return None

    def _apply_fallback(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        err: Optional[BaseException],
    ) -> Any:
        if self.fallback is _RAISE:
            if err is not None:
                raise err
            raise ParseError(f"{self.fn.__qualname__} produced no parseable output")
        if callable(self.fallback):
            return _call_fallback(self.fallback, args, kwargs)
        return self.fallback

    # -- sync call path -----------------------------------------------------

    def call(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        source = self._prepare()
        bound = self._bind(args, kwargs)
        started = time.monotonic()

        agentic: Agentic
        if self.role == _POST_PARSER:
            agentic = Agentic()
        else:
            pass1_kwargs = dict(kwargs)
            if self.role == _MIXED:
                pass1_kwargs[self.response_param] = FromInference
            try:
                result = self.fn(*args, **pass1_kwargs)
            except NotImplementedError:
                agentic = Agentic()
            else:
                decided = self._decide(result)
                if decided is None:
                    self._publish_precheck(started)
                    return result
                agentic = decided

        feed = {**self._render_feed(bound), **dict(agentic.feed)}
        prompt = source.render(feed)
        rc = self._resolve_run_context(args, kwargs)
        trace = self._new_trace(prompt)
        last_err: Optional[BaseException] = None
        for attempt in range(self.parse_max_retries + 1):
            trace.attempts = attempt + 1
            prompt_i = prompt if attempt == 0 else self._repair(prompt, last_err)
            try:
                inferencer = self.provider.get(feed)
                _reset_session(inferencer)
                raw = inferencer.infer(prompt_i, run_context=rc, **self._call_kwargs())
                out = AgenticOutput(raw, extract_result_text(raw))
                trace.raw_text = out.text
                value, stage1_value, stages = self._parse_sync(
                    out, args, kwargs, agentic.parser
                )
                self._finish_trace(
                    trace,
                    stage1_value,
                    stages,
                    started,
                    True,
                    result=value,
                    rc=rc,
                    feed=feed,
                )
                return value
            except self.retry_on as e:
                last_err = e
                trace.errors = trace.errors + (repr(e),)
        value = self._apply_fallback(args, kwargs, last_err)
        self._finish_trace(
            trace,
            value,
            ("fallback",),
            started,
            False,
            result=value,
            rc=rc,
            feed=feed,
        )
        return value

    def _parse_sync(
        self,
        out: AgenticOutput,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        parser_override: Any,
    ) -> Tuple[Any, Any, Tuple[str, ...]]:
        value, parsed, stages = self._stage1(out, parser_override)
        stage1_value = value
        if self.role in (_POST_PARSER, _MIXED):
            body_kwargs = dict(kwargs)
            body_kwargs[self.response_param] = value
            value = self.fn(*args, **body_kwargs)
            parsed = True
            stages = stages + ("body",)
        if not parsed:
            value = default_parser(self.fn)(out)
            stages = stages + ("default",)
        return value, stage1_value, stages

    # -- async call path ----------------------------------------------------

    async def acall(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        source = self._prepare()
        bound = self._bind(args, kwargs)
        started = time.monotonic()

        agentic: Agentic
        if self.role == _POST_PARSER:
            agentic = Agentic()
        else:
            pass1_kwargs = dict(kwargs)
            if self.role == _MIXED:
                pass1_kwargs[self.response_param] = FromInference
            try:
                result = await call_maybe_async(self.fn, *args, **pass1_kwargs)
            except NotImplementedError:
                agentic = Agentic()
            else:
                decided = self._decide(result)
                if decided is None:
                    self._publish_precheck(started)
                    return result
                agentic = decided

        feed = {**self._render_feed(bound), **dict(agentic.feed)}
        prompt = source.render(feed)
        rc = self._resolve_run_context(args, kwargs)
        trace = self._new_trace(prompt)
        last_err: Optional[BaseException] = None
        for attempt in range(self.parse_max_retries + 1):
            trace.attempts = attempt + 1
            prompt_i = prompt if attempt == 0 else self._repair(prompt, last_err)
            try:
                inferencer = self.provider.get(feed)
                _reset_session(inferencer)
                raw = await inferencer.ainfer(
                    prompt_i, run_context=rc, **self._call_kwargs()
                )
                out = AgenticOutput(raw, extract_result_text(raw))
                trace.raw_text = out.text
                value, stage1_value, stages = await self._parse_async(
                    out, args, kwargs, agentic.parser
                )
                self._finish_trace(
                    trace,
                    stage1_value,
                    stages,
                    started,
                    True,
                    result=value,
                    rc=rc,
                    feed=feed,
                )
                return value
            except self.retry_on as e:
                last_err = e
                trace.errors = trace.errors + (repr(e),)
        value = self._apply_fallback(args, kwargs, last_err)
        self._finish_trace(
            trace,
            value,
            ("fallback",),
            started,
            False,
            result=value,
            rc=rc,
            feed=feed,
        )
        return value

    async def _parse_async(
        self,
        out: AgenticOutput,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        parser_override: Any,
    ) -> Tuple[Any, Any, Tuple[str, ...]]:
        value, parsed, stages = self._stage1(out, parser_override)
        stage1_value = value
        if self.role in (_POST_PARSER, _MIXED):
            body_kwargs = dict(kwargs)
            body_kwargs[self.response_param] = value
            value = await call_maybe_async(self.fn, *args, **body_kwargs)
            parsed = True
            stages = stages + ("body",)
        if not parsed:
            value = default_parser(self.fn)(out)
            stages = stages + ("default",)
        return value, stage1_value, stages

    # -- trace --------------------------------------------------------------

    def _new_trace(self, prompt: str) -> AgenticFunctionTrace:
        return AgenticFunctionTrace(
            function=self.fn.__qualname__, path="agentic", prompt=prompt
        )

    def _finish_trace(
        self,
        trace: AgenticFunctionTrace,
        stage1_value: Any,
        stages: Tuple[str, ...],
        started: float,
        parsed: bool,
        *,
        result: Any,
        rc: Any,
        feed: Mapping[str, Any],
    ) -> None:
        trace.parsed = parsed
        trace.stage1_result = stage1_value
        trace.stages_used = stages
        trace.elapsed_s = time.monotonic() - started
        trace.result = result
        self.trace_var.set(trace)
        self._persist_trace(trace, rc, feed)

    def _persist_trace(
        self, trace: AgenticFunctionTrace, rc: Any, feed: Mapping[str, Any]
    ) -> None:
        """Best-effort JSON dump of the call trace to the run-context workspace.

        The redacted ``feed`` is assembled into the on-disk artifact here and is
        never stored on the trace, so ``AgenticFunctionTrace`` stays structurally
        argument-free (see trace.py) even though the artifact shows the inputs.
        No run-context workspace is a silent no-op, and any I/O error is swallowed
        so a logging failure can never break the inference.
        """
        workspace = getattr(rc, "workspace", None) if rc is not None else None
        if workspace is None:
            return
        try:
            workspace.ensure_dirs()
            path = workspace.artifact_path("agentic_function_trace.json")
            payload = {**trace.to_dict(), "redacted_args": dict(feed)}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str, ensure_ascii=False)
        except Exception:
            _logger.debug("agentic trace persist skipped", exc_info=True)

    def _publish_precheck(self, started: float) -> None:
        self.trace_var.set(
            AgenticFunctionTrace(
                function=self.fn.__qualname__,
                path="precheck",
                elapsed_s=time.monotonic() - started,
            )
        )

    # -- introspection / composition surface -------------------------------

    @property
    def inferencer(self) -> Any:
        """The resolved (built, cached) inferencer — no network call."""
        return self.provider.get({})

    @property
    def last_call(self) -> Optional[AgenticFunctionTrace]:
        """This function's most recent call trace in the current context."""
        return self.trace_var.get()

    def render(self, *args: Any, **kwargs: Any) -> str:
        """Render the prompt for these args without calling the model (dry run)."""
        source = self._prepare()
        bound = self._bind(args, kwargs)
        return source.render(self._render_feed(bound))

    def parse(self, raw: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the parse pipeline over a raw inference result (sync path).

        For an async body-as-parser use the underlying functions directly; this
        drives the synchronous pipeline for testing / offline decode.
        """
        self._prepare()
        out = AgenticOutput(raw, extract_result_text(raw))
        value, _stage1_value, _stages = self._parse_sync(out, args, kwargs, None)
        return value

    def with_inference_args(self, **kw: Any) -> "_AgenticFunctionBase":
        """A clone with extra per-call ``infer_kwargs``, sharing this inferencer."""
        clone = self._make_clone(infer_kwargs={**self.infer_kwargs, **kw})
        clone.provider = self.provider
        return clone

    def with_(self, **overrides: Any) -> "_AgenticFunctionBase":
        """A clone whose default inferencer is rebuilt with merged kwargs."""
        base = self._opts.get("inferencer_kwargs") or {}
        return self._make_clone(inferencer_kwargs={**base, **overrides})

    def _make_clone(self, **changes: Any) -> "_AgenticFunctionBase":
        opts = {**self._opts, **changes}
        clone = type(self)(self.fn, opts)
        functools.update_wrapper(clone, self.fn)
        clone.__signature__ = clone.sig
        return clone

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        """Bind ``self`` for a decorated method (mirrors ``function.__get__``)."""
        if obj is None:
            return self
        return functools.partial(self, obj)


class _SyncAgenticFunction(_AgenticFunctionBase):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(args, kwargs)


class _AsyncAgenticFunction(_AgenticFunctionBase):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.acall(args, kwargs)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def validate_agentic_function(wrapper: Any) -> None:
    """Force resolution with no network call; raise on any misconfiguration.

    Resolves the inferencer, reads + variable-validates the template, and runs
    ``preflight_all()`` (which returns a ``list[str]`` and does NOT raise) —
    surfacing a missing deep dependency as an
    :class:`AgenticFunctionConfigurationError` up front, not at first call.
    """
    if not isinstance(wrapper, _AgenticFunctionBase):
        raise AgenticFunctionConfigurationError(
            f"validate_agentic_function expects an @agentic_function, got "
            f"{type(wrapper).__name__}"
        )
    wrapper._prepare()  # noqa: SLF001 — template read + variable validation
    inferencer = wrapper.inferencer
    preflight = getattr(inferencer, "preflight_all", None)
    if callable(preflight):
        problems = await call_maybe_async(preflight)
        if problems:
            raise AgenticFunctionConfigurationError(
                f"preflight failed for {wrapper.__qualname__}: {problems}"
            )


def _reset_session(inferencer: Any) -> None:
    """Clear a session-bearing inferencer so a reused instance never resumes.

    ``StreamingInferencerBase`` defaults ``auto_resume=True`` and
    ``MetamateSDKInferencer`` mutates its conversation UUIDs; a cached, reused
    instance would otherwise resume the prior call's conversation and bleed
    across independent agentic-function calls.
    """
    reset = getattr(inferencer, "reset_session", None)
    if callable(reset):
        reset()


def _sanitize_slot(name: str) -> str:
    slot = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_")
    return slot or "agentic_fn"


def _annotation_admits_none(annotation: Any) -> bool:
    if annotation in (None, type(None)):
        return True
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        return type(None) in typing.get_args(annotation)
    if isinstance(annotation, str):
        return "None" in annotation or "Optional" in annotation
    return False


def _call_fallback(
    fallback: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Any:
    """Invoke a callable fallback with the call args if its signature accepts them.

    Supports both ``fallback=CodeSearchScope.default`` (zero-arg) and
    ``fallback=lambda task: ...`` (arg-aware) without guessing from a caught
    ``TypeError``.
    """
    try:
        params = inspect.signature(fallback).parameters.values()
        accepts_args = any(
            p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
            for p in params
        )
    except (TypeError, ValueError):
        accepts_args = False
    if accepts_args:
        return fallback(*args, **kwargs)
    return fallback()
