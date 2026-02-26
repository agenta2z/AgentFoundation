from typing import Any, Callable, Dict, Optional, Tuple, Union

from rich_python_utils.common_utils.arg_utils.param_parse import REQUIRED
from rich_python_utils.service_utils.common import generate_response


def _resolve_llm_timeout(
    timeout: Union[float, Tuple[float, float]] = None,
    connect_timeout: float = None,
    response_timeout: float = None,
):
    if timeout is None:
        if connect_timeout is not None:
            if response_timeout is not None:
                if connect_timeout == response_timeout:
                    timeout = connect_timeout
                else:
                    timeout = (connect_timeout, response_timeout)
            else:
                timeout = (connect_timeout, None)
        else:
            timeout = (None, response_timeout)

    return timeout


# Constants for parameter names
PARAM_PROMPT_OR_MESSAGES = "prompt_or_messages"
PARAM_API_KEY = "api_key"
PARAM_MAX_NEW_TOKENS = "max_new_tokens"
PARAM_TEMPERATURE = "temperature"
PARAM_TOP_P = "top_p"
PARAM_MODEL = "model"


class ModelApiInfo:
    """
    A configuration class for managing model API interactions.

    This class encapsulates API function calls, response processing, model selection,
    and parameter defaults for text generation models. It provides a unified interface
    for generating responses and listing available models, with support for custom parameters.

    The class uses the generic `generate_response` function internally, which provides
    automatic parameter validation, type conversion, enum handling, and error management.

    Attributes:
        api_func (Optional[Callable]): The API function to call for text generation.
        api_response_postprocess_func (Optional[Callable]): Function to process API responses.
        models_enum (Optional[Any]): Enum class containing available model options.
        params (Dict[str, Any]): Parameter specifications including defaults and constraints.
        include_params_in_response (bool): Whether to include parameters in API responses.
        raise_exception (bool): Whether to raise exceptions or return them in response.
        response_field_name (str): Key name for the result in response dictionary.
        success_flag_field_name (str): Key name for the success flag in response dictionary.
        params_field_name (str): Key name for parameters in response dictionary.

    Examples:
        Basic usage with model enum:

        >>> from enum import Enum
        >>>
        >>> class MyModels(Enum):
        ...     GPT_4 = "gpt-4"
        ...     GPT_3 = "gpt-3.5-turbo"
        >>>
        >>> def mock_api_call(prompt_or_messages, model, temperature, **kwargs):
        ...     return f"Response from {model}: {prompt_or_messages[:20]}..."
        >>>
        >>> def mock_postprocess(response_data):
        ...     return {"processed": response_data['result'], "success": response_data['success']}
        >>>
        >>> api_info = ModelApiInfo(
        ...     api_func=mock_api_call,
        ...     api_response_postprocess_func=mock_postprocess,
        ...     models_enum=MyModels,
        ...     default_model="gpt-4",
        ...     default_temperature=0.5
        ... )
        >>>
        >>> # Generate response
        >>> result = api_info.generate_response({"prompt_or_messages": "Hello, world!", "model": "gpt-4"})
        >>> result["success"]
        True
        >>> "Hello" in result["processed"]
        True
        >>>
        >>> # List available models
        >>> models = api_info.list_models()
        >>> len(models)
        2
        >>> models[0]["name"]
        'GPT_4'

        Advanced configuration with custom response handling:

        >>> api_custom = ModelApiInfo(
        ...     api_func=lambda **kwargs: f"Result: {kwargs.get('prompt_or_messages')}",
        ...     default_model="custom-model",
        ...     include_params_in_response=True,
        ...     response_field_name="output",
        ...     custom_param="default_value",
        ...     max_retries=3
        ... )
        >>>
        >>> result = api_custom.generate_response({"prompt_or_messages": "Test"})
        >>> "output" in result
        True
        >>> "params" in result
        True
        >>> result["params"]["custom_param"]
        'default_value'

        Error handling configuration:

        >>> def failing_api(prompt_or_messages, **kwargs):
        ...     raise ValueError("API Error")
        >>>
        >>> # With raise_exception=False (default)
        >>> api_safe = ModelApiInfo(api_func=failing_api, raise_exception=False)
        >>> result = api_safe.generate_response({"prompt_or_messages": "Test"})
        >>> result["success"]
        False
        >>> "API Error" in result["response"]
        True
    """

    def __init__(
        self,
        api_func: Optional[Callable] = None,
        api_response_postprocess_func: Optional[Callable] = None,
        models_enum: Optional[Any] = None,
        default_model: Optional[str] = None,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9,
        include_params_in_response: bool = False,
        raise_exception: bool = False,
        response_field_name: str = "response",
        success_flag_field_name: str = "success",
        params_field_name: str = "params",
        **additional_params,
    ):
        """
        Initialize the ModelApiInfo configuration.

        Args:
            api_func (Optional[Callable], optional): The function that calls the model API.
                Should accept parameters like prompt_or_messages, model, temperature, etc.
                Defaults to None.
            api_response_postprocess_func (Optional[Callable], optional): Function to
                process the response dictionary returned by generate_response. Should accept
                a dict with 'result', 'success', and optionally 'params' keys. Defaults to None.
            models_enum (Optional[Any], optional): An Enum class containing valid model
                options. If provided, enables model validation and listing. Defaults to None.
            default_model (Optional[str], optional): The default model identifier to use
                when none is specified in requests. Defaults to None.
            default_temperature (float, optional): Default temperature parameter for
                controlling randomness in generation (0.0 to 1.0). Defaults to 0.7.
            default_top_p (float, optional): Default top-p (nucleus sampling) parameter
                for controlling diversity (0.0 to 1.0). Defaults to 0.9.
            include_params_in_response (bool, optional): Whether to include all parameters
                used in the response dictionary. Defaults to False.
            raise_exception (bool, optional): Whether to raise exceptions from api_func.
                If False, catches exceptions and returns them in response with success=False.
                Defaults to False.
            response_field_name (str, optional): Key name for the generated result in the
                response dictionary. Defaults to 'result'.
            success_flag_field_name (str, optional): Key name for the success flag in the
                response dictionary. Defaults to 'success'.
            params_field_name (str, optional): Key name for parameters in the response
                dictionary (only used when include_params_in_response=True). Defaults to 'params'.
            **additional_params: Additional custom parameters to include in the params
                dictionary. These can be API-specific parameters like max_retries,
                timeout, frequency_penalty, presence_penalty, etc.

        Examples:
            Basic initialization:

            >>> from enum import Enum
            >>>
            >>> class TestModels(Enum):
            ...     MODEL_A = "model-a"
            ...     MODEL_B = "model-b"
            >>>
            >>> api_info = ModelApiInfo(
            ...     models_enum=TestModels,
            ...     default_model="model-a",
            ...     default_temperature=0.8
            ... )
            >>> api_info.params[PARAM_TEMPERATURE]
            0.8

            With response configuration:

            >>> api_configured = ModelApiInfo(
            ...     default_model="gpt-4",
            ...     include_params_in_response=True,
            ...     response_field_name="ai_output",
            ...     success_flag_field_name="ok",
            ...     params_field_name="request_params"
            ... )
            >>> api_configured.response_field_name
            'ai_output'
            >>> api_configured.include_params_in_response
            True

            With additional custom parameters:

            >>> api_extended = ModelApiInfo(
            ...     default_model="gpt-4",
            ...     default_temperature=0.6,
            ...     frequency_penalty=0.5,
            ...     presence_penalty=0.3,
            ...     stop_sequences=["END", "STOP"],
            ...     max_retries=5
            ... )
            >>> api_extended.params['frequency_penalty']
            0.5
            >>> api_extended.params['max_retries']
            5
        """
        self.api_func = api_func
        self.api_response_postprocess_func = api_response_postprocess_func
        self.models_enum = models_enum

        # Store response configuration
        self.include_params_in_response = include_params_in_response
        self.raise_exception = raise_exception
        self.response_field_name = response_field_name
        self.success_flag_field_name = success_flag_field_name
        self.params_field_name = params_field_name

        # Build parameter specifications with model enum pre-assigned if provided
        self.params = {
            PARAM_PROMPT_OR_MESSAGES: REQUIRED,  # Required
            PARAM_API_KEY: None,  # Optional, None means auto-select
            PARAM_MAX_NEW_TOKENS: None,  # Optional, None means use model default
            PARAM_TEMPERATURE: default_temperature,  # Default temperature
            PARAM_TOP_P: default_top_p,  # Default top_p
            **additional_params,  # Include any additional custom parameters
        }

        # Handle model parameter with or without enum annotation
        if models_enum is not None:
            # Pre-assign model enum to PARAM_MODEL parameter for automatic conversion
            self.params[PARAM_MODEL] = (default_model, models_enum)
        else:
            self.params[PARAM_MODEL] = default_model

    def generate_response(self, request_data: Dict[str, Any]) -> Any:
        """
        Generate a response using the configured API function.

        This method processes the request data according to the parameter specifications,
        calls the API function, and applies any configured post-processing. All parameters
        from the params dictionary (including additional custom parameters) are available
        for use. The method delegates to the generic `generate_response` function which
        handles parameter validation, type conversion, and error management.

        Args:
            request_data (Dict[str, Any]): Dictionary containing generation parameters.
                Must include required parameters (e.g., prompt_or_messages).
                Can override defaults like model, temperature, top_p, and any
                additional custom parameters.

        Returns:
            Any: The processed API response. Type depends on the configured
                api_response_postprocess_func. If no post-processing is configured,
                returns a dictionary with keys determined by response_field_name,
                success_flag_field_name, and optionally params_field_name.

        Examples:
            Basic response generation:

            >>> from enum import Enum
            >>>
            >>> class Models(Enum):
            ...     TEST = "test-model"
            >>>
            >>> def mock_generate(prompt_or_messages, model, temperature, **kwargs):
            ...     return f"Generated: {prompt_or_messages}"
            >>>
            >>> api = ModelApiInfo(
            ...     api_func=mock_generate,
            ...     models_enum=Models,
            ...     default_model="test-model"
            ... )
            >>>
            >>> result = api.generate_response({"prompt_or_messages": "Hello"})
            >>> result["success"]
            True
            >>> "Hello" in result["response"]
            True

            With custom response processing:

            >>> def mock_api(prompt_or_messages, **kwargs):
            ...     return f"Response: {prompt_or_messages}"
            >>>
            >>> def custom_processor(response_data):
            ...     return {
            ...         "text": response_data["response"],
            ...         "ok": response_data["success"],
            ...         "metadata": {"processed": True}
            ...     }
            >>>
            >>> api = ModelApiInfo(
            ...     api_func=mock_api,
            ...     api_response_postprocess_func=custom_processor
            ... )
            >>>
            >>> result = api.generate_response({"prompt_or_messages": "Test"})
            >>> result["ok"]
            True
            >>> result["metadata"]["processed"]
            True

            Including parameters in response:

            >>> api_with_params = ModelApiInfo(
            ...     api_func=lambda **kwargs: "Generated",
            ...     include_params_in_response=True,
            ...     custom_setting="value"
            ... )
            >>>
            >>> result = api_with_params.generate_response({
            ...     "prompt_or_messages": "Test",
            ...     "temperature": 0.9
            ... })
            >>> result["params"]["temperature"]
            0.9
            >>> result["params"]["custom_setting"]
            'value'

            Custom field names:

            >>> api_custom_fields = ModelApiInfo(
            ...     api_func=lambda **kwargs: "Output",
            ...     response_field_name="output",
            ...     success_flag_field_name="ok"
            ... )
            >>>
            >>> result = api_custom_fields.generate_response({"prompt_or_messages": "Hi"})
            >>> result["ok"]
            True
            >>> result["output"]
            'Output'

            Error handling:

            >>> def error_api(prompt_or_messages, **kwargs):
            ...     raise ValueError("Processing failed")
            >>>
            >>> api_safe = ModelApiInfo(
            ...     api_func=error_api,
            ...     raise_exception=False
            ... )
            >>>
            >>> result = api_safe.generate_response({"prompt_or_messages": "Test"})
            >>> result["success"]
            False
            >>> "Processing failed" in result["response"]
            True
        """
        return generate_response(
            request_data=request_data,
            params=self.params,
            generate_response_func=self.api_func,
            response_func=self.api_response_postprocess_func,
            include_params_in_response=self.include_params_in_response,
            raise_exception=self.raise_exception,
            response_field_name=self.response_field_name,
            success_flag_field_name=self.success_flag_field_name,
            params_field_name=self.params_field_name,
        )

    def list_models(self) -> Optional[list]:
        """
        List all available models from the configured models enum.

        Returns:
            Optional[list]: A list of dictionaries containing model information with 'name'
                and 'value' keys if models_enum is configured, None otherwise.
                Each dictionary has:
                - name (str): The enum member name
                - value (str): The enum member value (model identifier)

        Examples:
            With models enum:

            >>> from enum import Enum
            >>>
            >>> class AvailableModels(Enum):
            ...     SMALL = "model-small-v1"
            ...     LARGE = "model-large-v2"
            ...     ULTRA = "model-ultra-v3"
            >>>
            >>> api = ModelApiInfo(models_enum=AvailableModels)
            >>> models = api.list_models()
            >>> len(models)
            3
            >>> models[0]
            {'name': 'SMALL', 'value': 'model-small-v1'}
            >>> models[1]["name"]
            'LARGE'

            Without models enum:

            >>> api_no_enum = ModelApiInfo()
            >>> api_no_enum.list_models() is None
            True

            Iterating over models:

            >>> api = ModelApiInfo(models_enum=AvailableModels)
            >>> models = api.list_models()
            >>> [m["value"] for m in models]
            ['model-small-v1', 'model-large-v2', 'model-ultra-v3']
        """
        if self.models_enum is not None:
            models = [
                {"name": model.name, "value": model.value} for model in self.models_enum
            ]
            return models
        return None
