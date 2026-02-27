import json
from typing import Any, Dict, Union

import requests
from attr import attrib, attrs

from agent_foundation.common.inferencers.inference_args import (
    CommonLlmInferenceArgs,
)
from agent_foundation.common.inferencers.remote_inferencer_base import (
    RemoteInferencerBase,
)
from rich_python_utils.common_utils import dict_

HTTP_REQUEST_SERVICE_URL_PREFIX = "http://"
DEFAULT_HTTP_REQUEST_INFERENCE_ARGS = dict_(
    CommonLlmInferenceArgs(
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=20480,
    ),
    ignore_none_values=True,
)
DEFAULT_BODY_PROMPT_FIELD_NAME = "prompt"
DEFAULT_RESPONSE_FIELD_NAME = "response"
DEFAULT_SECRET_KEY_FIELD_NAME = "api_key"
DEFAULT_MODEL_ID_FIELD_NAME = "model"

@attrs
class HttpRequestInferencer(RemoteInferencerBase):
    """
    An inferencer class for making HTTP POST requests to a remote service for inference tasks.

    This class extends `RemoteInferencerBase` to perform inference via HTTP requests, utilizing common LLM inference
    arguments. It handles constructing the request, sending the request, and parsing the response from the remote service.

    Attributes:
        request_body_prompt_field_name (str): The field name in the request body where the prompt should be placed.
            Defaults to 'prompt'.
        response_field_name (str): The field name in the response where the result is expected. Defaults to 'response'.
        secret_key_field_name (str): The field name in the request body where the secret key should be placed.
            Defaults to 'api_key'.
        model_id_field_name (str): The field name in the request body where the model ID should be placed.
            Defaults to 'model'.
    """

    request_body_prompt_field_name: str = attrib(default=DEFAULT_BODY_PROMPT_FIELD_NAME)
    response_field_name: str = attrib(default=DEFAULT_RESPONSE_FIELD_NAME)
    secret_key_field_name: str = attrib(default=DEFAULT_SECRET_KEY_FIELD_NAME)
    model_id_field_name: str = attrib(default=DEFAULT_MODEL_ID_FIELD_NAME)

    def __attrs_post_init__(self):
        """
        Post-initialization to set default values for the service URL prefix and default inference arguments.

        This method is automatically called after the object is initialized. It ensures that default values
        for `service_url_prefix` and `default_inference_args` are set if they are not provided during initialization.

        Calls:
            super().__attrs_post_init__(): Calls the parent class's post-initialization method to finalize setup.

        Example:
            If `service_url_prefix` is not set, it defaults to 'http://'. Default inference arguments are set
            to typical values for common LLM tasks unless specified otherwise.
        """
        if not self.service_url_prefix:
            self.service_url_prefix = HTTP_REQUEST_SERVICE_URL_PREFIX
        if not self.default_inference_args:
            self.default_inference_args = DEFAULT_HTTP_REQUEST_INFERENCE_ARGS
        super().__attrs_post_init__()

    def get_client(self):
        """
        Returns the HTTP client used for sending requests.

        In this implementation, an HTTP client is not explicitly needed since requests are sent
        directly using the `requests` library. This method returns None, but it could be overridden
        in subclasses if a specific client setup is required.

        Returns:
            None: No client setup required for basic HTTP requests.
        """
        return None

    def _send_request(self, client, request) -> Dict[str, Any]:
        """
        Sends an HTTP POST request to the remote service with robust error handling.

        This method implements proper timeout, status code checking, and JSON parsing
        with comprehensive error handling. It returns structured error responses for
        failures to maintain consistency with the service contract.

        Args:
            client: The client object used to send the request (not used in this implementation).
            request (dict): The request payload, including the prompt and additional inference arguments.

        Returns:
            Dict[str, Any]: The JSON response from the remote service if successful, or a structured
            error response containing 'success', 'error', and 'response' fields.
        """
        try:
            # Make POST request with proper timeout and headers
            response = requests.post(
                self.service_url,
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,  # Use timeout from base class
            )

            # Raise an exception for bad status codes
            response.raise_for_status()

            # Return the JSON response
            return response.json()

        except requests.exceptions.RequestException as e:
            # Return structured error response for HTTP request failures
            return {
                "success": False,
                "error": f"HTTP request failed: {str(e)}",
                "response": None,
            }
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            return {
                "success": False,
                "error": f"Failed to parse JSON response: {str(e)}",
                "response": response.text if "response" in locals() else None,
            }

    def construct_request(self, inference_input: str, **_inference_args) -> dict:
        """
        Constructs the request payload for the HTTP POST request with enhanced parameter handling.

        This method formats the prompt and additional inference arguments into a dictionary that conforms
        to the expected request structure of the remote service. It uses the secret_key and model_id
        from the base InferencerBase class, and includes systematic parameter filtering.

        Args:
            inference_input (str): The input prompt for the inference.
            _inference_args: Additional keyword arguments to include in the request payload. Common
                parameters include:
                - temperature: Temperature for text generation (0.0 to 1.0)
                - max_tokens: Maximum number of tokens to generate
                - max_new_tokens: Maximum number of new tokens to generate
                - top_p: Top-p parameter for nucleus sampling
                - top_k: Top-k parameter for sampling

        Returns:
            dict: The constructed request payload with the prompt and additional arguments.
        """
        # Start with the basic prompt field
        request_data = {self.request_body_prompt_field_name: inference_input}

        # Add secret key from base class if available
        if self.secret_key:
            request_data[self.secret_key_field_name] = self.secret_key

        # Add model ID from base class if available
        if self.model_id:
            request_data[self.model_id_field_name] = self.model_id

        # Add all other inference arguments
        # Filter out None values to keep the request clean
        for key, value in _inference_args.items():
            if value is not None:
                request_data[key] = value

        return request_data

    def _parse_response(self, response) -> Union[str, Dict, Any]:
        """
        Parses the response from the remote service to extract the inference result.

        This method handles both successful responses and structured error responses from the
        improved _send_request method. It checks for success/error indicators and extracts
        the appropriate content.

        Args:
            response (Dict[str, Any]): The JSON response from the remote service, which may
            include structured error information.

        Returns:
            Union[str, Dict, Any]: The parsed response content. For successful responses,
            returns the content from the specified response field. For error responses,
            returns the structured error information. Returns an empty string if the
            response is empty or the expected field is missing.
        """
        if not response:
            return ""

        # Check for structured error response
        if (
            isinstance(response, dict)
            and "success" in response
            and not response.get("success", True)
        ):
            # Return the error information for debugging/logging
            return response

        # Handle successful response
        if isinstance(response, dict) and self.response_field_name in response:
            return response[self.response_field_name]
        elif isinstance(response, dict):
            # If response_field_name is not found, return the whole response
            return response
        else:
            # For non-dict responses, return as-is
            return response
