import asyncio
import logging
import os
from abc import ABC, abstractmethod
from contextvars import ContextVar
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, List, Optional, Sequence, Type, Union

from attr import attrib, attrs
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.workflow.common.resumable import Resumable
from rich_python_utils.common_utils import dict_, iter__, resolve_environ
from rich_python_utils.common_utils.function_helper import FallbackMode, execute_with_retry

# Retry prompt mode constants
RETRY_PROMPT_MODES = ("original", "simple_retry", "retry_with_original")

# Module-level ContextVar for per-call fallback state.
# Each asyncio Task from aparallel_infer gets its own context copy — no race.
# Shared with streaming_inferencer_base.py for cache path tracking.
_current_fallback_state: ContextVar[dict | None] = ContextVar("_current_fallback_state", default=None)

_SIMPLE_RETRY_PROMPT = "You got interrupted. Can you retry the above task?"

_logger = logging.getLogger(__name__)


@attrs
class InferencerBase(Debuggable, Resumable, ABC):
    merger__ = """
    Abstract base class for implementing inference logic with retry and fallback recovery.

    This class provides a framework for executing inference tasks with built-in retry mechanisms,
    timeout support, and fallback chain recovery. Subclasses should implement the `_infer` method
    to define specific inference behavior and optionally override other methods to customize prompt
    formatting, output validation, and recovery strategies.

    Retry semantics:
        The default ``fallback_mode`` is ``ON_FIRST_FAILURE``, which means the primary ``_infer``/
        ``_ainfer`` gets one attempt, then ``_infer_recovery``/``_ainfer_recovery`` gets ``max_retry``
        attempts. Total attempts = ``1 + max_retry``. To restore the pre-fallback behavior (all
        attempts call the same function), set ``fallback_mode=FallbackMode.NEVER``.

        Sync path: ``execute_with_retry`` uses ``while True`` + ``attempts >= max_retry: break``
        = initial + max_retry calls per callable.
        Async path: ``async_execute_with_retry`` uses ``for attempt in range(max_retry)``
        = max_retry calls per callable.

    Timeout support:
        ``total_timeout_seconds``: Wall-clock cap for the entire retry+fallback loop (sync and async).
        ``attempt_timeout_seconds``: Per-attempt cap via ``asyncio.wait_for`` (async only).
        Sync path raises ``NotImplementedError`` if ``attempt_timeout_seconds > 0``.

    Attributes:
        model_id (str): The identifier of the model to be used for inference.
        secret_key (str): Secret key for authentication, if needed. Defaults to None.
        max_retry (int): The maximum number of retry attempts. With the default
            ``fallback_mode=ON_FIRST_FAILURE``, total attempts = 1 + max_retry.
            With ``fallback_mode=NEVER``, total attempts = max_retry (async) or 1 + max_retry (sync).
        min_retry_wait (float): Minimum wait time in seconds between retry attempts. Defaults to 0.
        max_retry_wait (float): Maximum wait time in seconds between retry attempts. Defaults to 0.
        default_return_or_raise (Union[Any, Exception]): Value to return or exception to raise on failure.
        total_timeout_seconds (float): Wall-clock cap in seconds. 0 = disabled. Applies to both sync and async.
        attempt_timeout_seconds (float): Per-attempt timeout in seconds. 0.0 = disabled. Async only.
        fallback_inferencer: External fallback inferencer(s). None = self-recovery only.
        fallback_mode (FallbackMode): When to transition to fallback. Default: ON_FIRST_FAILURE.
        default_inference_args (dict): Default arguments passed to ``_infer``.
        input_preprocessor (Callable): Optional input preprocessor.
        response_post_processor (Callable): Optional response post-processor.
        post_response_merger (str, Callable): Optional response merger for iterator inputs.
    """

    model_id: str = attrib(default="")
    _secret_key: Union[str, Sequence[str]] = attrib(default=None)

    # Class-level fire-once warning sets
    _paired_override_warned: set = set()
    _nested_retry_warned: set = set()

    # region retry parameters
    max_retry: int = attrib(default=1)
    min_retry_wait: float = attrib(default=0)
    max_retry_wait: float = attrib(default=0)
    default_return_or_raise: Union[Any, Exception] = attrib(default=None)
    # endregion

    # Total timeout in seconds for the entire retry+fallback loop.
    # 0 = disabled (backward compatible). Applies to both sync (_infer_single)
    # and async (_ainfer_single) paths via the retry helpers.
    total_timeout_seconds: float = attrib(default=0)

    # Per-attempt timeout in seconds. 0.0 = disabled. Float for sub-second granularity.
    # Async only — sync path raises NotImplementedError if > 0.
    attempt_timeout_seconds: float = attrib(default=0.0)

    # External fallback inferencer(s) to try when the primary _infer/_ainfer fails.
    # None = no external fallback (self-recovery via _infer_recovery/_ainfer_recovery still applies).
    fallback_inferencer: Union["InferencerBase", List["InferencerBase"], None] = attrib(default=None)

    # Controls when the retry helper transitions to the next fallback callable.
    fallback_mode: FallbackMode = attrib(default=FallbackMode.ON_FIRST_FAILURE)

    response_types: Sequence[Type] = attrib(default=(str,))
    default_inference_args: dict = attrib(default=None, converter=dict_)
    input_preprocessor: Callable = attrib(default=None)
    response_post_processor: Callable = attrib(default=None)
    post_response_merger: Union[str, Callable] = attrib(default=None)

    # State graph support — optional list of StateGraphTracker instances
    state_graphs: list = attrib(default=None)

    # Whether this inferencer has local file system access (e.g., can write files).
    # False for cloud API inferencers (RovoChat), True for local agents (RovoDevCli).
    has_local_access: bool = attrib(default=False)

    # Default output path — when relative and a workspace is set (via _workspace
    # on flow inferencers), resolves to workspace.outputs_dir/<output_path>.
    # Simple API inferencers leave this as None.
    output_path: Optional[str] = attrib(default=None)

    # === Template-based prompt rendering (opt-in) ===
    # When template_manager is set, inference_input is treated as the raw
    # user query.  Before reaching _infer(), the base class renders a Jinja2
    # template via template_manager, binding the raw input to {{ input }}.
    # This eliminates the need for PromptWrapperInferencer in most cases.
    template_manager: Optional[Any] = attrib(default=None)
    template_key: str = attrib(default="")
    template_root_space: Optional[str] = attrib(default=None)
    template_extra_feed: dict = attrib(factory=dict)

    def __attrs_post_init__(self):
        if isinstance(self.post_response_merger, str):
            if self.post_response_merger == "default":
                from rich_python_utils.mp_utils.common import merge_results

                self.post_response_merger = merge_results
            else:
                from rich_python_utils.mp_utils.common import get_merger

                self.post_response_merger = get_merger(self.post_response_merger)

        # Paired-override warning: detect subclass overriding only one of
        # _ainfer_recovery / _infer_recovery (not both).
        async_overridden = type(self)._ainfer_recovery is not InferencerBase._ainfer_recovery
        sync_overridden = type(self)._infer_recovery is not InferencerBase._infer_recovery
        if async_overridden != sync_overridden:
            cls_name = type(self).__name__
            if cls_name not in InferencerBase._paired_override_warned:
                InferencerBase._paired_override_warned.add(cls_name)
                _logger.warning(
                    f"{cls_name} overrides only "
                    f"{'_ainfer_recovery' if async_overridden else '_infer_recovery'} "
                    f"but not {'_infer_recovery' if async_overridden else '_ainfer_recovery'}. "
                    f"Override both for full sync/async recovery coverage."
                )

        # Nested-retry warning: detect fallback_inferencer with max_retry > 1.
        if self.fallback_inferencer is not None:
            fb_list = (
                self.fallback_inferencer
                if isinstance(self.fallback_inferencer, list)
                else [self.fallback_inferencer]
            )
            cls_name = type(self).__name__
            for fb in fb_list:
                if fb.max_retry > 1 and cls_name not in InferencerBase._nested_retry_warned:
                    InferencerBase._nested_retry_warned.add(cls_name)
                    _logger.warning(
                        f"{cls_name}: fallback_inferencer {type(fb).__name__} has "
                        f"max_retry={fb.max_retry}. This creates multiplicative retries "
                        f"(outer × inner). Consider max_retry=1 for fallbacks."
                    )
                    break

        super().__attrs_post_init__()

    @property
    def secret_key(self) -> str:
        return resolve_environ(self._secret_key)

    @abstractmethod
    def _infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Performs the core inference logic.

        This abstract method should be implemented by subclasses to perform inference based on the provided input and additional arguments.

        Args:
            inference_input (Any): The input data for inference.
            **_inference_args: Additional keyword arguments for customizing the inference process.

        Returns:
            Any: The response of the inference. The exact type and format depend on the specific implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    # -- Template rendering & output finalization -------------------------

    def _build_template_feed(self, inference_input: str) -> dict:
        """Build the template variable feed dict.

        Merges ``template_extra_feed`` with ``{{ input }}`` bound to
        ``inference_input``.  Conditionally includes ``output_path``
        only when the inferencer has local file access.

        Override this method to customize feed construction (e.g., add
        dynamic variables from external sources).
        """
        feed: dict = {"input": inference_input}
        feed.update(self.template_extra_feed)
        resolved = self.resolve_output_path()
        if resolved and os.path.isabs(resolved) and self.has_local_access:
            feed["output_path"] = resolved
        return feed

    @property
    def supports_prompt_rendering(self) -> bool:
        """Whether this inferencer has prompt/template rendering capability.

        Returns True when the inferencer is configured with a template_manager
        (for base class auto-rendering) or when a subclass overrides this to
        indicate its own prompt rendering support.
        """
        return self.template_manager is not None

    def _render_prompt(self, inference_input: Any) -> Any:
        """Render a template-based prompt if template_manager is configured.

        Called by ``_infer_single`` / ``_ainfer_single`` after
        ``input_preprocessor`` and before ``_infer``.  When
        ``template_manager`` is None, returns ``inference_input`` unchanged.
        """
        if self.template_manager is None:
            return inference_input
        feed = self._build_template_feed(inference_input)
        return self.template_manager(
            self.template_key,
            active_template_root_space=self.template_root_space,
            **feed,
        )

    def _finalize_output(self, response: Any) -> Any:
        """Post-process output when template_manager is active.

        When the inferencer lacks local file access but has an
        ``output_path``, extracts ``<Response>``-delimited content from
        the response, writes it to the resolved output path, and returns
        the cleaned text.  This mirrors what PromptWrapperInferencer
        previously did in ``_save_response_if_needed``.

        Called by ``_infer_single`` / ``_ainfer_single`` after ``_infer``
        returns but before ``response_post_processor``.
        """
        if self.template_manager is None:
            return response
        resolved = self.resolve_output_path()
        if not resolved or not os.path.isabs(resolved):
            return response
        if self.has_local_access:
            # Local-access inferencer writes the file itself
            return response
        # Non-local: extract <Response> content and save to file
        from agent_foundation.common.response_parsers import extract_delimited
        cleaned = extract_delimited(str(response))
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(cleaned)
        return cleaned

    # -- Resumable protocol implementation ----------------------------------

    def _get_result_path(self, result_id, *args, **kwargs) -> str:
        """Default checkpoint path using ``output_path`` as base directory.

        Raises ``NotImplementedError`` when ``output_path`` is not configured,
        which is the safe default — workflow callers
        (``WorkGraphNode._should_save_result``) already catch this and skip
        checkpointing gracefully.
        """
        if self.output_path is None:
            raise NotImplementedError(
                f"{type(self).__name__}: no output_path configured. "
                f"Set output_path to enable checkpointing."
            )
        resolved = self.resolve_output_path() or self.output_path
        return os.path.join(resolved, f"{result_id}.pkl")

    def _try_resume_from_cache(self, inference_input, inference_config=None, **kwargs):
        """Sync hook for subclasses to check for resumable cached results.

        Called AFTER preprocessing and template rendering, BEFORE the retry
        loop in ``_infer_single``.  Returns the cached result to short-circuit
        execution, or ``None`` to proceed normally.

        Base implementation returns ``None`` (no caching for plain inferencers).
        ``StreamingInferencerBase`` overrides this with cache discovery logic.
        """
        return None

    async def _atry_resume_from_cache(self, inference_input, inference_config=None, **kwargs):
        """Async hook — mirrors ``_try_resume_from_cache`` for ``_ainfer_single``.

        Base implementation delegates to the sync version.  Streaming inferencers
        override to ``await self._ainfer(augmented)`` for partial-cache recovery.
        """
        return self._try_resume_from_cache(inference_input, inference_config, **kwargs)

    # -- Inference pipeline -------------------------------------------------

    def _infer_single(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Process a single inference input with preprocessing, inference, and post-processing.

        This method handles the complete inference pipeline for a single input:
        1. Preprocesses input via input_preprocessor if provided
        2. Merges default_inference_args with provided _inference_args
        3. Executes _infer() with retry logic via execute_with_retry()
        4. Post-processes the response via response_post_processor if provided
        5. Returns the post-processed inference response

        Args:
            inference_input: Input data for inference (will be preprocessed if input_preprocessor is set)
            inference_config: Optional configuration for the inference run
            **_inference_args: Additional keyword arguments merged with default_inference_args
                and passed to _infer()

        Returns:
            The post-processed inference response. Type depends on the _infer() implementation
            and response_post_processor callable.

        Raises:
            Exception: If all retry attempts fail and default_return_or_raise is set to
                None or an Exception object.
        """
        # Capture original input BEFORE preprocessing for retry_with_original mode
        original_input = inference_input

        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)

        # Template rendering (opt-in: only when template_manager is set)
        inference_input = self._render_prompt(inference_input)

        # Resume hook: check for cached result from a previous session.
        # Placed AFTER preprocessing/rendering so prompt hash matches cache.
        resume_result = self._try_resume_from_cache(
            inference_input, inference_config, **_inference_args
        )
        if resume_result is not None:
            # Run the same post-processing tail as the normal path
            resume_result = self._finalize_output(resume_result)
            if self.state_graphs:
                self.update_state_graphs(resume_result)
            if self.response_post_processor is not None:
                resume_result = self.response_post_processor(resume_result)
            return resume_result

        inference_args = self.default_inference_args.copy()
        if _inference_args:
            inference_args.update(_inference_args)

        # Augment with state graph args
        if self.state_graphs:
            inference_args.update(self.get_inference_args_from_state_graphs())

        # Pop runtime overrides that should not be forwarded to _infer()
        on_retry_callback = inference_args.pop("on_retry_callback", None)
        total_timeout = inference_args.pop(
            "total_timeout_seconds", self.total_timeout_seconds
        )
        attempt_timeout = inference_args.pop(
            "attempt_timeout_seconds", self.attempt_timeout_seconds
        )
        fallback_mode = inference_args.pop("fallback_mode", self.fallback_mode)
        on_fallback_callback = inference_args.pop("on_fallback_callback", None)
        retry_prompt_mode = inference_args.pop("retry_prompt_mode", "original")

        # Per-attempt timeout is async-only — reject in sync path
        if attempt_timeout and attempt_timeout > 0:
            raise NotImplementedError(
                "Per-attempt timeout is async-only. Use total_timeout_seconds "
                "for sync, or call ainfer() instead."
            )

        # Validate retry_prompt_mode
        if retry_prompt_mode not in RETRY_PROMPT_MODES:
            raise ValueError(
                f"Invalid retry_prompt_mode={retry_prompt_mode!r}. "
                f"Must be one of {RETRY_PROMPT_MODES}"
            )

        # Mutable args list — prompt can be swapped by the retry callback
        retry_args = [inference_input]

        # Build internal retry callback (handles prompt transformation + user callback)
        _user_callback = on_retry_callback
        if _user_callback is not None or retry_prompt_mode != "original":

            def _internal_retry_callback(attempt, exception):
                # Forward to user callback with local inference_args
                if _user_callback is not None:
                    _user_callback(attempt, exception, inference_args)
                # Transform prompt based on retry_prompt_mode
                if retry_prompt_mode == "simple_retry":
                    retry_args[0] = _SIMPLE_RETRY_PROMPT
                elif retry_prompt_mode == "retry_with_original":
                    retry_args[0] = (
                        _SIMPLE_RETRY_PROMPT + " The task was:\n" + str(original_input)
                    )
                if retry_prompt_mode != "original":
                    self.log_info(
                        f"Retry prompt ({retry_prompt_mode}): {str(retry_args[0])[:200]}",
                        "RetryPrompt",
                    )

            on_retry_callback = _internal_retry_callback

        self.log_debug(inference_input, "InferenceInput")
        self.log_debug(inference_args, "InferenceArgs")

        # Convert 0 → None for timeout parameters (0 = disabled)
        effective_total_timeout = total_timeout or None

        # -- Build _fallback_state and fallback chain --
        _fallback_state = {"last_exception": None, "partial_output": None, "cache_path": None}

        # Recovery wrapper — reads from closure-captured _fallback_state
        def _recovery_wrapper(inp, **kw):
            return self._infer_recovery(
                inp,
                last_exception=_fallback_state["last_exception"],
                last_partial_output=_fallback_state["partial_output"],
                inference_config=inference_config,
                **inference_args,
            )

        # External fallback wrappers from fallback_inferencer list
        external_wrappers = []
        if self.fallback_inferencer is not None:
            fb_list = (
                self.fallback_inferencer
                if isinstance(self.fallback_inferencer, list)
                else [self.fallback_inferencer]
            )
            external_wrappers = [
                lambda inp, inf=inf, **kw: inf.infer(inp, inference_config, **kw)
                for inf in fb_list
            ]

        # Build fallback chain and mode for the retry helper
        if fallback_mode == FallbackMode.NEVER:
            effective_fallback_func = None
            effective_fallback_mode = FallbackMode.NEVER
        else:
            effective_fallback_func = [_recovery_wrapper] + external_wrappers
            effective_fallback_mode = fallback_mode

        # Transition callback — populates _fallback_state and resets retry_args[0]
        _user_on_fallback = on_fallback_callback

        def _on_transition(from_func, to_func, exception, total_attempts):
            _fallback_state["last_exception"] = exception
            if _fallback_state["cache_path"]:
                try:
                    with open(_fallback_state["cache_path"], "r", encoding="utf-8") as f:
                        raw = f.read()
                    _fallback_state["partial_output"] = raw if raw.strip() else None
                except OSError:
                    _fallback_state["partial_output"] = None
            # Reset retry_args[0] to original input so external fallback
            # inferencers see the original prompt, not the mutated retry prompt
            retry_args[0] = original_input
            # Forward to user-provided on_fallback_callback if present
            if _user_on_fallback is not None:
                _user_on_fallback(from_func, to_func, exception, total_attempts)

        # Set ContextVar for this call (per-thread safe for sync path)
        token = _current_fallback_state.set(_fallback_state)
        try:
            inference_response = execute_with_retry(
                func=partial(self._infer, inference_config=inference_config),
                max_retry=self.max_retry,
                min_retry_wait=self.min_retry_wait,
                max_retry_wait=self.max_retry_wait,
                args=retry_args,
                kwargs=inference_args,
                default_return_or_raise=self.default_return_or_raise,
                on_retry_callback=on_retry_callback,
                total_timeout=effective_total_timeout,
                fallback_func=effective_fallback_func,
                fallback_mode=effective_fallback_mode,
                on_fallback_callback=_on_transition if effective_fallback_func else None,
            )
        except TimeoutError:
            self.log_info(
                f"Total timeout after {total_timeout}s",
                "TotalTimeout",
            )
            raise
        finally:
            _current_fallback_state.reset(token)

        self.log_debug(inference_response, "InferenceResponse")

        # Template output finalization (extract <Response>, save to file)
        inference_response = self._finalize_output(inference_response)

        # Update state graphs from response
        if self.state_graphs:
            self.update_state_graphs(inference_response)

        if self.response_post_processor is not None:
            processed_response = self.response_post_processor(inference_response)
            self.log_debug(processed_response, "PostProcessedResponse")
            return processed_response

        return inference_response

    # -- State graph integration -------------------------------------------

    def attach_state_graph(self, tracker) -> None:
        """Attach a StateGraphTracker to this inferencer."""
        if self.state_graphs is None:
            self.state_graphs = []
        self.state_graphs.append(tracker)

    def get_inference_args_from_state_graphs(self) -> dict:
        """Collect inference args from all attached state graphs.
        Handles None check and iterates over trackers.
        """
        if not self.state_graphs:
            return {}
        merged = {}
        for tracker in self.state_graphs:
            merged.update(self._get_inference_args_from_state_graph(tracker))
        return merged

    def update_state_graphs(self, response) -> None:
        """Update all attached state graphs from inference response.
        Handles None check and iterates over trackers.
        """
        if not self.state_graphs:
            return
        for tracker in self.state_graphs:
            self._update_state_graph(tracker, response)

    def _get_inference_args_from_state_graph(self, tracker) -> dict:
        """Override in subclasses: extract inference args from one tracker.
        Default: empty dict (no-op).
        """
        return {}

    def _update_state_graph(self, tracker, response) -> None:
        """Override in subclasses: update one tracker from response.
        Default: no-op.
        """
        pass

    # -- Workspace output path resolution --

    def resolve_output_path(
        self, runtime_override: Optional[str] = None
    ) -> Optional[str]:
        """Resolve the effective output path.

        Priority: *runtime_override* > ``self.output_path``.

        Resolution:
        - If the path is relative and ``_workspace`` is set (flow inferencers),
          resolve to ``workspace.outputs_dir / path``.
        - If the path is absolute, return as-is.
        - If no workspace, return the path unchanged.

        No side effects (no directory creation, no file I/O).

        Flow inferencers set ``_workspace`` in ``__attrs_post_init__`` or when
        assigned by a parent.  Simple API inferencers never set it, so
        ``getattr`` returns ``None`` and this method returns the raw path.
        """
        path = runtime_override if runtime_override is not None else self.output_path
        if path is None:
            return None
        ws = getattr(self, "_workspace", None)
        if ws is not None and not os.path.isabs(path):
            return ws.output_path(path)
        return path

    def _infer_iterator(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Process an iterator of inputs and yield atomized post-processed results.

        Args:
            inference_input: Iterator of input items to process
            inference_config: Optional configuration for the inference run
            **_inference_args: Additional keyword arguments passed to _infer_single()

        Yields:
            Atomized inference results for each input item
        """
        for _inference_input in inference_input:
            response = self._infer_single(
                _inference_input, inference_config, **_inference_args
            )
            yield from iter__(response, atom_types=self.response_types)

    def infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Execute inference with automatic iterator detection.

        This method routes inference based on input type:
        - If input is an Iterator and post_response_merger is provided:
          Returns a single merged result after processing all items
        - If input is an Iterator and post_response_merger is not provided:
          Returns an Iterator yielding atomized results
        - Otherwise: Returns a single post-processed result

        Args:
            inference_input: Input data for inference. Can be a single input or an Iterator
            inference_config: Optional configuration for the inference run
            **_inference_args: Additional keyword arguments passed to the inference methods

        Returns:
            If input is Iterator with post_response_merger: Returns a single merged result
            If input is Iterator without post_response_merger: Returns an Iterator yielding atomized results
            Otherwise: Returns a single post-processed inference result

        Raises:
            Exception: If inference fails after all retry attempts
        """
        if isinstance(inference_input, Iterator):
            iterator_result = self._infer_iterator(
                inference_input, inference_config, **_inference_args
            )

            if self.post_response_merger is not None:
                all_responses = list(iterator_result)
                merged_response = self.post_response_merger(all_responses)
                self.log_debug(merged_response, "MergedResponse")
                return merged_response
            else:
                return iterator_result
        else:
            return self._infer_single(
                inference_input, inference_config, **_inference_args
            )

    def iter_infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Execute inference and always return an iterator of responses.

        This method wraps infer() to ensure the output is always iterable:
        - If response_types is set: atomizes response using iter__() with response_types
        - If response_types is not set: yields response directly (or from iterator if response is already an iterator)

        Note: If post_response_merger is provided for iterator inputs, the merged response
        will be atomized according to response_types.

        Args:
            inference_input: Input data for inference (single input or iterator)
            inference_config: Optional configuration for the inference run
            **_inference_args: Additional keyword arguments passed to infer()

        Yields:
            Inference responses. Atomization behavior depends on response_types configuration.
        """
        response = self.infer(
            inference_input=inference_input,
            inference_config=inference_config,
            **_inference_args,
        )
        if not self.response_types:
            if isinstance(response, Iterator):
                yield from response
            else:
                yield response
        else:
            yield from iter__(response, atom_types=self.response_types)

    def parallel_infer(
        self,
        inference_inputs: Iterable[Any],
        inference_config: Any = None,
        num_workers: int = None,
        use_threading: bool = True,
        debug: bool = False,
        **_inference_args,
    ) -> list:
        """Process multiple inputs concurrently using thread or process pool.

        Dispatches each input to _infer_single() via parallel_process_by_pool,
        enabling concurrent execution for I/O-bound workloads (API calls, etc.).

        Args:
            inference_inputs: Iterable of inputs to process concurrently.
                Generators are supported (materialized internally).
            inference_config: Optional configuration passed to each _infer_single call.
            num_workers: Number of pool workers. None = auto:
                threading (default): min(len(inputs), 32) for I/O-bound work.
                multiprocessing: get_suggested_num_workers() respecting CPU count.
                Always capped at len(inputs).
            use_threading: True (default) uses ThreadPool — no pickling required.
                False uses multiprocessing.Pool — requires picklable inferencer
                (no lambdas in input_preprocessor, response_post_processor, etc.).
            debug: True runs sequentially in a single process for debugging
                (passed through to parallel_process_by_pool's debug param).
            **_inference_args: Additional keyword arguments merged with
                default_inference_args and passed to _infer_single().

        Returns:
            List of inference results, order-preserving (same index alignment
            as inputs).

        Note:
            post_response_merger is NOT auto-applied. The iterator path in
            infer() atomizes results via iter__() before merging — a different
            input shape. Users can apply their own merging on the returned list.
        """
        from rich_python_utils.mp_utils.mp_target import MPTarget
        from rich_python_utils.mp_utils.parallel_process import parallel_process_by_pool

        inference_inputs = list(inference_inputs)
        if not inference_inputs:
            return []

        num_inputs = len(inference_inputs)

        if num_workers is None:
            if use_threading:
                num_workers = min(num_inputs, 32)
            else:
                from rich_python_utils.mp_utils.common import get_suggested_num_workers

                num_workers = get_suggested_num_workers()
        num_workers = min(num_workers, num_inputs)

        pool_class = None
        if use_threading:
            from multiprocessing.pool import ThreadPool

            pool_class = ThreadPool

        worker = partial(
            self._infer_single,
            inference_config=inference_config,
            **_inference_args,
        )
        mp_target = MPTarget(
            worker,
            pass_pid_to_target=False,
            pass_each_data_item_to_target=True,
        )

        self.log_debug(
            f"{num_inputs} inputs, {num_workers} workers, "
            f"{'threading' if use_threading else 'multiprocessing'}",
            "ParallelInfer",
        )

        results = parallel_process_by_pool(
            num_p=num_workers,
            data_iter=inference_inputs,
            target=mp_target,
            pool_object=pool_class,
            merge_output=True,
            mergers=["list"],
            debug=debug,
        )

        return results

    def __call__(self, inference_input: Any, inference_config: Any = None, **kwargs):
        """
        Allows the instance to be called as a function, invoking the `infer` method.

        This method enables the object to be used like a function, passing the input and any additional keyword arguments directly to the `infer` method.

        Args:
            inference_input (Any): The input data to be used for inference.
            inference_config (Any): The configuration for the current inference.
            **kwargs: Additional keyword arguments to be passed to the `infer` method.

        Returns:
            Any: The response of the inference, as returned by the `infer` method.
        """
        return self.infer(inference_input, inference_config=inference_config, **kwargs)

    # region Async Methods

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """Async version of _infer().

        Default implementation wraps sync _infer() for backwards compatibility.
        Async-native subclasses should override this method directly.

        Args:
            inference_input: Input data for inference.
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments for inference.

        Returns:
            The inference response.
        """
        return self._infer(inference_input, inference_config, **_inference_args)

    def _infer_recovery(
        self,
        inference_input: Any,
        last_exception: Optional[Exception],
        last_partial_output: Optional[str],
        inference_config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Overridable sync recovery method. Default delegates to _infer.

        Called as the fallback function when the primary _infer fails.
        Subclasses (e.g., StreamingInferencerBase) can override to implement
        cache-aware or session-aware recovery strategies.

        Args:
            inference_input: The original inference input.
            last_exception: The exception from the failed primary attempt.
            last_partial_output: Any partial output from the failed attempt, or None.
            inference_config: Optional inference configuration.
            **kwargs: Additional keyword arguments passed through to _infer.
        """
        return self._infer(inference_input, inference_config, **kwargs)

    async def _ainfer_recovery(
        self,
        inference_input: Any,
        last_exception: Optional[Exception],
        last_partial_output: Optional[str],
        inference_config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Overridable async recovery method. Default delegates to _ainfer.

        Called as the fallback function when the primary _ainfer fails.
        Subclasses (e.g., StreamingInferencerBase) can override to implement
        cache-aware or session-aware recovery strategies.

        Args:
            inference_input: The original inference input.
            last_exception: The exception from the failed primary attempt.
            last_partial_output: Any partial output from the failed attempt, or None.
            inference_config: Optional inference configuration.
            **kwargs: Additional keyword arguments passed through to _ainfer.
        """
        return await self._ainfer(inference_input, inference_config, **kwargs)

    async def _ainfer_single(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """Async process a single inference input with preprocessing, inference, and post-processing.

        Async equivalent of _infer_single(). Handles:
        1. Input preprocessing via input_preprocessor
        2. Merging default_inference_args with provided args
        3. Executing _ainfer() with retry logic
        4. Post-processing via response_post_processor

        Args:
            inference_input: Input data for inference.
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments merged with default_inference_args.

        Returns:
            The post-processed inference response.
        """
        from rich_python_utils.common_utils.async_function_helper import (
            async_execute_with_retry,
        )

        # Capture original input BEFORE preprocessing for retry_with_original mode
        original_input = inference_input

        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)

        # Template rendering (opt-in: only when template_manager is set)
        inference_input = self._render_prompt(inference_input)

        # Resume hook (async): check for cached result from a previous session.
        # Uses _atry (async) so streaming override can await self._ainfer().
        resume_result = await self._atry_resume_from_cache(
            inference_input, inference_config, **_inference_args
        )
        if resume_result is not None:
            # Run the same post-processing tail as the normal path
            resume_result = self._finalize_output(resume_result)
            if self.state_graphs:
                self.update_state_graphs(resume_result)
            if self.response_post_processor is not None:
                resume_result = self.response_post_processor(resume_result)
            return resume_result

        inference_args = self.default_inference_args.copy()
        if _inference_args:
            inference_args.update(_inference_args)

        # Augment with state graph args
        if self.state_graphs:
            inference_args.update(self.get_inference_args_from_state_graphs())

        # Pop runtime overrides that should not be forwarded to _ainfer()
        on_retry_callback = inference_args.pop("on_retry_callback", None)
        total_timeout = inference_args.pop(
            "total_timeout_seconds", self.total_timeout_seconds
        )
        attempt_timeout = inference_args.pop(
            "attempt_timeout_seconds", self.attempt_timeout_seconds
        )
        fallback_mode = inference_args.pop("fallback_mode", self.fallback_mode)
        on_fallback_callback = inference_args.pop("on_fallback_callback", None)
        retry_prompt_mode = inference_args.pop("retry_prompt_mode", "original")

        # Validate retry_prompt_mode
        if retry_prompt_mode not in RETRY_PROMPT_MODES:
            raise ValueError(
                f"Invalid retry_prompt_mode={retry_prompt_mode!r}. "
                f"Must be one of {RETRY_PROMPT_MODES}"
            )

        # Mutable args list — prompt can be swapped by the retry callback
        retry_args = [inference_input]

        # Build internal retry callback (handles prompt transformation + user callback)
        _user_callback = on_retry_callback
        if _user_callback is not None or retry_prompt_mode != "original":

            def _internal_retry_callback(attempt, exception):
                # Forward to user callback with local inference_args
                if _user_callback is not None:
                    _user_callback(attempt, exception, inference_args)
                # Transform prompt based on retry_prompt_mode
                if retry_prompt_mode == "simple_retry":
                    retry_args[0] = _SIMPLE_RETRY_PROMPT
                elif retry_prompt_mode == "retry_with_original":
                    retry_args[0] = (
                        _SIMPLE_RETRY_PROMPT + " The task was:\n" + str(original_input)
                    )
                if retry_prompt_mode != "original":
                    self.log_info(
                        f"Retry prompt ({retry_prompt_mode}): {str(retry_args[0])[:200]}",
                        "RetryPrompt",
                    )

            on_retry_callback = _internal_retry_callback

        self.log_debug(inference_input, "InferenceInput")
        self.log_debug(inference_args, "InferenceArgs")

        # Convert 0 → None for timeout parameters (0 = disabled)
        effective_total_timeout = total_timeout or None
        effective_attempt_timeout = attempt_timeout or None

        # -- Build _fallback_state and fallback chain --
        _fallback_state = {"last_exception": None, "partial_output": None, "cache_path": None}

        # Recovery wrapper — reads from closure-captured _fallback_state
        async def _recovery_wrapper(inp, **kw):
            return await self._ainfer_recovery(
                inp,
                last_exception=_fallback_state["last_exception"],
                last_partial_output=_fallback_state["partial_output"],
                inference_config=inference_config,
                **inference_args,
            )

        # External fallback wrappers from fallback_inferencer list
        external_wrappers = []
        if self.fallback_inferencer is not None:
            fb_list = (
                self.fallback_inferencer
                if isinstance(self.fallback_inferencer, list)
                else [self.fallback_inferencer]
            )
            external_wrappers = [
                lambda inp, inf=inf, **kw: inf.ainfer(inp, inference_config, **kw)
                for inf in fb_list
            ]

        # Build fallback chain and mode for the retry helper
        if fallback_mode == FallbackMode.NEVER:
            effective_fallback_func = None
            effective_fallback_mode = FallbackMode.NEVER
        else:
            effective_fallback_func = [_recovery_wrapper] + external_wrappers
            effective_fallback_mode = fallback_mode

        # Transition callback — populates _fallback_state and resets retry_args[0]
        _user_on_fallback = on_fallback_callback

        async def _on_transition(from_func, to_func, exception, total_attempts):
            _fallback_state["last_exception"] = exception
            if _fallback_state["cache_path"]:
                try:
                    with open(_fallback_state["cache_path"], "r", encoding="utf-8") as f:
                        raw = f.read()
                    _fallback_state["partial_output"] = raw if raw.strip() else None
                except OSError:
                    _fallback_state["partial_output"] = None
            # Reset retry_args[0] to original input so external fallback
            # inferencers see the original prompt, not the mutated retry prompt
            retry_args[0] = original_input
            # Forward to user-provided on_fallback_callback if present
            if _user_on_fallback is not None:
                result = _user_on_fallback(from_func, to_func, exception, total_attempts)
                if asyncio.iscoroutine(result):
                    await result

        # Set ContextVar for this call (per-task safe under aparallel_infer)
        token = _current_fallback_state.set(_fallback_state)
        try:
            inference_response = await async_execute_with_retry(
                func=lambda inp: self._ainfer(inp, inference_config, **inference_args),
                max_retry=self.max_retry,
                min_retry_wait=self.min_retry_wait,
                max_retry_wait=self.max_retry_wait,
                args=retry_args,
                default_return_or_raise=self.default_return_or_raise,
                on_retry_callback=on_retry_callback,
                total_timeout=effective_total_timeout,
                attempt_timeout=effective_attempt_timeout,
                fallback_func=effective_fallback_func,
                fallback_mode=effective_fallback_mode,
                on_fallback_callback=_on_transition if effective_fallback_func else None,
            )
        except TimeoutError:
            self.log_info(
                f"Total timeout after {total_timeout}s",
                "TotalTimeout",
            )
            raise
        finally:
            _current_fallback_state.reset(token)

        self.log_debug(inference_response, "InferenceResponse")

        # Template output finalization (extract <Response>, save to file)
        inference_response = self._finalize_output(inference_response)

        # Update state graphs from response
        if self.state_graphs:
            self.update_state_graphs(inference_response)

        if self.response_post_processor is not None:
            processed_response = self.response_post_processor(inference_response)
            self.log_debug(processed_response, "PostProcessedResponse")
            return processed_response

        return inference_response

    async def _ainfer_iterator(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ) -> AsyncIterator:
        """Async process an iterator of inputs and yield atomized post-processed results.

        Async equivalent of _infer_iterator().

        Args:
            inference_input: Iterator of input items to process.
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments passed to _ainfer_single().

        Yields:
            Atomized inference results for each input item.
        """
        for _inference_input in inference_input:
            response = await self._ainfer_single(
                _inference_input, inference_config, **_inference_args
            )
            for item in iter__(response, atom_types=self.response_types):
                yield item

    async def ainfer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """Async version of infer().

        NOTE: For Iterator inputs without post_response_merger, sync infer() returns
        a lazy generator while this method eagerly collects results into a list.
        This is an inherent Python limitation — async generators cannot be returned
        from a regular async def and consumed as a value. All results are computed
        upfront, which uses more memory than the lazy sync path. For large iterator
        inputs where memory is a concern, process items individually with
        await _ainfer_single() in a loop instead.

        Args:
            inference_input: Input data for inference. Can be a single input or an Iterator.
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments passed to the inference methods.

        Returns:
            If input is Iterator with post_response_merger: Returns a single merged result.
            If input is Iterator without post_response_merger: Returns a list of atomized results.
            Otherwise: Returns a single post-processed inference result.
        """
        if isinstance(inference_input, Iterator):
            all_results = []
            async for item in self._ainfer_iterator(
                inference_input, inference_config, **_inference_args
            ):
                all_results.append(item)

            if self.post_response_merger is not None:
                merged = self.post_response_merger(all_results)
                self.log_debug(merged, "MergedResponse")
                return merged
            return all_results
        else:
            return await self._ainfer_single(
                inference_input, inference_config, **_inference_args
            )

    async def aiter_infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ) -> AsyncIterator:
        """Async version of iter_infer().

        Execute inference and always return an async iterator of responses.

        Args:
            inference_input: Input data for inference (single input or iterator).
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments passed to inference.

        Yields:
            Inference responses. Atomization behavior depends on response_types configuration.
        """
        response = await self.ainfer(
            inference_input=inference_input,
            inference_config=inference_config,
            **_inference_args,
        )
        if not self.response_types:
            if isinstance(response, (list, Iterator)):
                for item in response:
                    yield item
            else:
                yield response
        else:
            for item in iter__(response, atom_types=self.response_types):
                yield item

    async def aparallel_infer(
        self,
        inference_inputs: Iterable[Any],
        inference_config: Any = None,
        max_concurrency: int = None,
        debug: bool = False,
        **_inference_args,
    ) -> list:
        """Async process multiple inputs concurrently using asyncio.gather.

        Dispatches each input to _ainfer_single() with optional semaphore-based
        concurrency control to prevent event loop/connection pool exhaustion.

        Args:
            inference_inputs: Iterable of inputs to process concurrently.
                Generators are supported (materialized internally).
            inference_config: Optional configuration passed to each _ainfer_single call.
            max_concurrency: Maximum number of concurrent tasks. None defaults to
                min(len(inputs), 32) to prevent unbounded concurrency.
            debug: True runs sequentially via await loop (parity with sync
                parallel_infer debug mode).
            **_inference_args: Additional keyword arguments merged with
                default_inference_args and passed to _ainfer_single().

        Returns:
            List of inference results, order-preserving (same index alignment
            as inputs).

        Note:
            post_response_merger is NOT auto-applied — same rationale as
            parallel_infer. Users can apply their own merging on the returned list.
        """
        inference_inputs = list(inference_inputs)
        if not inference_inputs:
            return []

        num_inputs = len(inference_inputs)

        if debug:
            self.log_debug(
                f"aparallel_infer debug mode: {num_inputs} inputs",
                "ParallelInfer",
            )
            results = []
            for inp in inference_inputs:
                result = await self._ainfer_single(
                    inp, inference_config, **_inference_args
                )
                results.append(result)
            return results

        if max_concurrency is None:
            max_concurrency = min(num_inputs, 32)

        self.log_debug(
            f"{num_inputs} inputs, max_concurrency={max_concurrency}",
            "ParallelInfer",
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_infer(inp):
            async with semaphore:
                return await self._ainfer_single(
                    inp, inference_config, **_inference_args
                )

        tasks = [_bounded_infer(inp) for inp in inference_inputs]
        results = await asyncio.gather(*tasks)
        return list(results)

    # region Async Lifecycle Methods

    async def aconnect(self, **kwargs):
        """Establish async connection to external service.

        Override this method in subclasses that need persistent connections.
        Default implementation does nothing.

        Args:
            **kwargs: Connection-specific arguments.
        """
        pass

    async def adisconnect(self):
        """Disconnect from external service.

        Override this method in subclasses that need cleanup.
        Default implementation does nothing.
        """
        pass

    async def __aenter__(self):
        """Async context manager entry. Calls aconnect()."""
        await self.aconnect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit. Calls adisconnect()."""
        await self.adisconnect()
        return False

    # endregion
