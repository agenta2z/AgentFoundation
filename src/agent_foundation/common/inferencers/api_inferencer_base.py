from typing import Any, Union, Dict, Callable

from attr import attrs, attrib

from science_modeling_tools.common.inferencers.inferencer_base import InferencerBase

import logging

logger = logging.getLogger(__name__)


@attrs
class ApiInferencerBase(InferencerBase):
    """
    A base class for API-based inference, extending `InferencerBase` with remote request functionality.

    This class is designed for sending inference requests to a remote API. It leverages a callable
    `generate_text_api` function to handle the actual API request, and includes a response parsing
    method to format results as required.

    Attributes:
        generate_text_api (Callable[[str, str, ...], Any]): A callable function that accepts the inference prompt,
            the secret key, and other inference parameters, and returns inference results.

    Methods:
        _parse_response(response: Any) -> Union[str, Dict, Any]:
            Parses and formats the API response. This method is intended to be customized by subclasses
            to handle specific response formats, returning either the result as a string, dictionary, or
            other relevant type.

        _infer(prompt: str, **_inference_args) -> Union[str, Dict, Any]:
            Completes the inference process by invoking `generate_text_api` and processing the
            response via `_parse_response`. This method passes additional inference arguments directly
            to the API call as needed.

    Notes:
        - For inherited attributes such as `model_id`, `model_version`, and retry configurations, refer to
        `InferencerBase` for complete descriptions. These attributes support configuring inference
        parameters and retry handling at the base level, enabling robust inference flows.
    """
    _inference_api: Callable[[Any, str, ...], Any] = attrib(default=None)

    def _parse_response(self, response: Any) -> Union[str, Dict, Any]:
        """
        Parses the response received from the remote service.

        This method extracts the relevant inference results from the response data structure
        returned by the remote service. The output can vary depending on the service response
        format and may include strings, dictionaries, or other data types.

        Args:
            response (Any): The raw response data from the remote service.

        Returns:
            Union[str, Dict, Any]: The parsed result of the inference. Typically a string or
            dictionary, but can also be other types depending on the service implementation.
        """
        return response

    def _infer(self, inference_input: str, inference_config: Any = None, **_inference_args) -> str:
        """
        Executes the full inference process by constructing, sending, and parsing a request to a remote service.

        This method integrates the steps of getting the client, constructing the request, sending it,
        and parsing the response. It utilizes the methods defined in the subclass to perform these operations.

        Args:
            inference_input (str): The input prompt for the inference.
            _inference_args: Additional keyword arguments for constructing the request.

        Returns:
            Union[str, Dict, Any]: The parsed inference result from the remote service, which could be
            a string, dictionary, or other types based on the specific service response.
        """
        response = self._inference_api(
            inference_input,
            model=self.model_id,
            api_key=self.secret_key,
            **_inference_args
        )
        logger.debug(f"Response: {response}")
        return self._parse_response(response)
