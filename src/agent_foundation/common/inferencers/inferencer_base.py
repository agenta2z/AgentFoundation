from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Iterator, Sequence, Type, Union

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
