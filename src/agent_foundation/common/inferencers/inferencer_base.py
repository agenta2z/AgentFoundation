import asyncio
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Sequence, Type, Union

from attr import attrib, attrs
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_utils import dict_, iter__, resolve_environ
from rich_python_utils.common_utils.function_helper import execute_with_retry


@attrs
class InferencerBase(Debuggable, ABC):
    merger__ = """
    Abstract base class for implementing inference logic with retry functionality.

    This class provides a framework for executing inference tasks with built-in retry mechanisms.
    Subclasses should implement the `_infer` method
    to define specific inference behavior and optionally override other methods to customize prompt
    formatting and output validation.

    Attributes:
        model_id (str): The identifier of the model to be used for inference.
        secret_key (str): Secret key for authentication, if needed. Defaults to None.
        max_retry (int): The maximum number of retry attempts for inference if the initial call fails.
            Retry is disabled if this number is <= 1. Defaults to 1.
        min_retry_wait (float): Minimum wait time in seconds between retry attempts. Defaults to 0.
        max_retry_wait (float): Maximum wait time in seconds between retry attempts. If set to 0, no wait is applied.
            Defaults to 0.
        default_return_or_raise (Union[Any, Exception]): The value to return or exception to raise if all retry attempts fail.
            If None, a generic exception will be raised after all retries fail. Defaults to None.
        default_inference_args (dict): A dictionary of default arguments to pass to the `_infer` method during inference.
            This can be overridden or extended by passing additional arguments via `infer` or `__call__`.
        input_preprocessor (Callable): Optional callable for preprocessing input before inference.
            Should take inference_input as parameter and return preprocessed input.
            If None, input passes through unchanged. Defaults to None.
            For multiprocessing compatibility, use module-level functions, not lambdas.
        response_post_processor (Callable): Optional callable for post-processing inference responses.
            Should take inference_response as parameter and return post-processed response.
            If None, response passes through unchanged. Defaults to None.
            For multiprocessing compatibility, use module-level functions, not lambdas.
        post_response_merger (str, Callable): Optional callable that merges multiple post-processed responses into a single response.
            When provided for iterator inputs, all responses are collected, post-processed, and merged instead of yielding.
            Defaults to None. Specify "default" to use a default build-in merger.
    """

    model_id: str = attrib(default="")
    _secret_key: Union[str, Sequence[str]] = attrib(default=None)

    # region retry parameters
    max_retry: int = attrib(default=1)
    min_retry_wait: float = attrib(default=0)
    max_retry_wait: float = attrib(default=0)
    default_return_or_raise: Union[Any, Exception] = attrib(default=None)
    # endregion

    # Total timeout for async inference (applies to _ainfer_single only).
    # 0 = disabled (backward compatible). Caps the entire _ainfer() call
    # including all retries. Sync _infer_single is not wrapped because
    # sync timeout in Python is inherently complex (signal.alarm is
    # Unix/main-thread-only).
    total_timeout_seconds: int = attrib(default=0)

    response_types: Sequence[Type] = attrib(default=(str,))
    default_inference_args: dict = attrib(default=None, converter=dict_)
    input_preprocessor: Callable = attrib(default=None)
    response_post_processor: Callable = attrib(default=None)
    post_response_merger: Union[str, Callable] = attrib(default=None)

    def __attrs_post_init__(self):
        if isinstance(self.post_response_merger, str):
            if self.post_response_merger == "default":
                from rich_python_utils.mp_utils.common import merge_results

                self.post_response_merger = merge_results
            else:
                from rich_python_utils.mp_utils.common import get_merger

                self.post_response_merger = get_merger(self.post_response_merger)
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
        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)

        inference_args = self.default_inference_args.copy()
        if _inference_args:
            inference_args.update(_inference_args)

        self.log_debug(inference_input, "InferenceInput")
        self.log_debug(inference_args, "InferenceArgs")

        inference_response = execute_with_retry(
            func=partial(self._infer, inference_config=inference_config),
            max_retry=self.max_retry,
            min_retry_wait=self.min_retry_wait,
            max_retry_wait=self.max_retry_wait,
            args=[inference_input],
            kwargs=inference_args,
            default_return_or_raise=self.default_return_or_raise,
        )

        self.log_debug(inference_response, "InferenceResponse")

        if self.response_post_processor is not None:
            processed_response = self.response_post_processor(inference_response)
            self.log_debug(processed_response, "PostProcessedResponse")
            return processed_response

        return inference_response

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

        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)

        inference_args = self.default_inference_args.copy()
        if _inference_args:
            inference_args.update(_inference_args)

        self.log_debug(inference_input, "InferenceInput")
        self.log_debug(inference_args, "InferenceArgs")

        async def _do_inference():
            return await async_execute_with_retry(
                func=lambda inp: self._ainfer(inp, inference_config, **inference_args),
                max_retry=self.max_retry,
                min_retry_wait=self.min_retry_wait,
                max_retry_wait=self.max_retry_wait,
                args=[inference_input],
                default_return_or_raise=self.default_return_or_raise,
            )

        if self.total_timeout_seconds > 0:
            try:
                inference_response = await asyncio.wait_for(
                    _do_inference(), timeout=self.total_timeout_seconds
                )
            except asyncio.TimeoutError:
                self.log_info(
                    f"Total timeout after {self.total_timeout_seconds}s",
                    "TotalTimeout",
                )
                raise
        else:
            inference_response = await _do_inference()

        self.log_debug(inference_response, "InferenceResponse")

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
