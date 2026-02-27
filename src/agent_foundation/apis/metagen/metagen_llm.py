import json
import os
import random
from enum import StrEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from metagen import (
    CompletionResponse,
    Message,
    MetaGenKey,
    MetaGenPlatform,
    thrift_platform_factory,
)

from agent_foundation.apis.common import _resolve_llm_timeout

from rich_python_utils.common_utils import get_
from rich_python_utils.console_utils import hprint_message


class MetaGenModels(StrEnum):
    """
    Enumeration for major supported MetaGen models.
    """

    # Claude models
    CLAUDE_4_SONNET = "claude-4-sonnet-genai"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219-us"

    # Llama models
    LLAMA_3_1_405B = "llama3.1-405b-instruct"
    LLAMA_3_3_70B = "llama3.3-70b-instruct"
    LLAMA_3_70B = "llama3-70b-instruct"

    # GPT models
    GPT_5 = "gpt-5-genai"
    GPT_4_1 = "gpt-4-1-genai"
    GPT_4O = "gpt-4o"

    # Gemini models
    GEMINI_2_5_PRO = "gemini-2-5-pro"


# Path to the keys configuration file (located in _config/ relative to this file)
_KEYS_CONFIG_PATH = Path(__file__).parent / "_config" / "keys.json"

# Cached keys data
_cached_keys_data: Optional[Dict] = None


def _get_keys_config_path() -> Path:
    """
    Get the path to the keys configuration file.

    First checks for environment variable METAGEN_KEYS_PATH,
    then falls back to the default location.
    """
    env_path = os.environ.get("METAGEN_KEYS_PATH")
    if env_path:
        return Path(env_path)
    return _KEYS_CONFIG_PATH


def _load_keys_data() -> Dict:
    """
    Load and cache keys data from JSON file.

    Returns:
        Dictionary containing model_to_key_map and default_key.

    Raises:
        FileNotFoundError: If keys.json doesn't exist.
        json.JSONDecodeError: If keys.json is not valid JSON.
    """
    global _cached_keys_data

    if _cached_keys_data is not None:
        return _cached_keys_data

    keys_path = _get_keys_config_path()

    if not keys_path.exists():
        raise FileNotFoundError(
            f"MetaGen keys configuration file not found at {keys_path}. "
            f"Please copy keys.template.json to keys.json and fill in your API keys, "
            f"or set METAGEN_KEYS_PATH environment variable to point to your keys file."
        )

    with open(keys_path, "r") as f:
        _cached_keys_data = json.load(f)

    return _cached_keys_data


def load_model_to_key_map() -> Dict[str, List[str]]:
    """
    Load the model-to-key mapping from configuration.

    Returns:
        Dictionary mapping model IDs to lists of available API keys.
    """
    keys_data = _load_keys_data()
    return keys_data.get("model_to_key_map", {})


def get_default_metagen_key() -> str:
    """
    Get the default MetaGen API key from configuration.

    Returns:
        The default API key string.
    """
    keys_data = _load_keys_data()
    return keys_data.get("default_key", "")


DEFAULT_MAX_TOKENS = {
    f"{MetaGenModels.CLAUDE_4_SONNET}": 10240,
    f"{MetaGenModels.CLAUDE_3_7_SONNET}": 10240,
    f"{MetaGenModels.LLAMA_3_1_405B}": 10240,
    f"{MetaGenModels.LLAMA_3_3_70B}": 10240,
    f"{MetaGenModels.LLAMA_3_70B}": 10240,
    f"{MetaGenModels.GPT_5}": 10240,
    f"{MetaGenModels.GPT_4_1}": 10240,
    f"{MetaGenModels.GPT_4O}": 10240,
    f"{MetaGenModels.GEMINI_2_5_PRO}": 10240,
}


def get_optimal_key_for_model(model: MetaGenModels) -> str:
    """
    Get an optimal MetaGen key for a given model.
    When multiple keys are available for a model, randomly selects one for load balancing.

    Args:
        model: The MetaGen model to get a key for

    Returns:
        A MetaGen key for the specified model (randomly selected if multiple available)

    Examples:
        >>> get_optimal_key_for_model(MetaGenModels.CLAUDE_4_SONNET)
        'mg-api-...'  # Returns one of the available Claude keys

        >>> get_optimal_key_for_model(MetaGenModels.LLAMA_3_1_405B)
        'mg-api-...'  # Returns one of the available Llama keys
    """
    model_to_key_map = load_model_to_key_map()
    default_key = get_default_metagen_key()

    available_keys = model_to_key_map.get(str(model), [default_key])
    if isinstance(available_keys, list) and len(available_keys) > 0:
        return random.choice(available_keys)
    return available_keys if isinstance(available_keys, str) else default_key


def _prepare_completion_params(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
    model: MetaGenModels,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    metagen_key: str,
    timeout: Union[float, Tuple[float, float]],
    connect_timeout: float,
    response_timeout: float,
    verbose: bool,
    **kwargs,
) -> Tuple[MetaGenPlatform, Dict]:
    """
    Prepare common parameters for both sync and async MetaGen completion calls.

    Returns:
        Tuple of (metagen_platform, params_dict)
    """
    # Auto-select optimal key if not provided
    if metagen_key is None:
        metagen_key = get_optimal_key_for_model(model)

    # Build parameters dict
    model_str = f"{model}"
    if not max_new_tokens:
        max_new_tokens = DEFAULT_MAX_TOKENS.get(model_str, 8192)

    messages = _get_messages(prompt_or_messages)

    params = {
        "model": model_str,
        "messages": messages.build(),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    # Handle timeout setting
    timeout = _resolve_llm_timeout(
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout,
    )
    # Note: MetaGen API timeout handling would be implemented here if supported

    params.update(kwargs)  # Add any additional kwargs

    if verbose:
        hprint_message({**params, "Using MetaGen key": metagen_key})

    # Create metagen platform
    metagen_platform: MetaGenPlatform = thrift_platform_factory.create(
        MetaGenKey(key=metagen_key),
        auto_rate_limit=True,
    )

    return metagen_platform, params


def _get_messages(prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]]):
    """
    Convert various input formats to metagen message format.

    Args:
        prompt_or_messages: Input in various formats - string, dict, or sequence

    Returns:
        Message object compatible with metagen API
    """
    messages = Message.message_list()

    if isinstance(prompt_or_messages, str):
        # Single string prompt
        messages.add_user_message(prompt_or_messages)
    elif isinstance(prompt_or_messages, Dict):
        # Single message dictionary
        if prompt_or_messages["role"] == "system":
            messages.add_system_message(prompt_or_messages["content"])
        elif prompt_or_messages["role"] == "user":
            messages.add_user_message(prompt_or_messages["content"])
        elif prompt_or_messages["role"] == "assistant":
            messages.add_ai_message(prompt_or_messages["content"])
        else:
            # Handle any other roles as user messages
            messages.add_user_message(prompt_or_messages["content"])
    elif isinstance(prompt_or_messages, (List, Tuple)):
        if isinstance(prompt_or_messages[0], str):
            # Sequence of strings - alternate between user and assistant
            for i in range(0, len(prompt_or_messages) - 1, 2):
                messages.add_user_message(prompt_or_messages[i])
                if i + 1 < len(prompt_or_messages):
                    messages.add_ai_message(prompt_or_messages[i + 1])
            # If odd number of messages, add the last one as user
            if len(prompt_or_messages) % 2 == 1:
                messages.add_user_message(prompt_or_messages[-1])
        elif isinstance(prompt_or_messages[0], Dict):
            # Sequence of message dictionaries
            for turn in prompt_or_messages:
                if turn["role"] == "system":
                    messages.add_system_message(turn["content"])
                elif turn["role"] == "user":
                    messages.add_user_message(turn["content"])
                elif turn["role"] == "assistant":
                    messages.add_ai_message(turn["content"])
                else:
                    # Handle any other roles as user messages
                    messages.add_user_message(turn["content"])
    else:
        raise ValueError(
            "'prompt_or_messages' must be one of str, Dict, or a sequence of strs or Dicts"
        )

    return messages


def generate_text(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
    model: MetaGenModels = MetaGenModels.CLAUDE_4_SONNET,
    max_new_tokens: int = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    api_key: str = None,
    timeout: Union[float, Tuple[float, float]] = None,
    connect_timeout: float = None,
    response_timeout: float = None,
    return_raw_results: bool = False,
    verbose: bool = False,
    result_array_getter: Union[str, Callable] = None,
    result_processor: Callable = None,
    **kwargs,
) -> str:
    """
    Generate text using MetaGen API (synchronous).

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The MetaGen model to use for generating text.
        max_new_tokens: The maximum number of new tokens to generate (excluding the prompt).
        temperature: Controls the "creativity" of the generated text. A higher temperature will
                    result in more creative responses, while a lower temperature will result in
                    more predictable responses.
        top_p: Nucleus sampling parameter. If set, only tokens with cumulative probability <= top_p
               are considered for sampling. Must be between 0 and 1.
        api_key: Your MetaGen API key. If None, automatically selects optimal key for the model.
        timeout: Request timeout in seconds. Can be either a float (same timeout for connect and read)
                or a tuple of (connect_timeout, read_timeout). If specified, this takes precedence
                over connect_timeout and response_timeout parameters.
        connect_timeout: Maximum time in seconds to wait for establishing connection to the API.
                        Only used if timeout parameter is None. If None, uses the client's default.
        response_timeout: Maximum time in seconds to wait for the complete response stream.
                         Only used if timeout parameter is None. This is the total time allowed
                         for receiving all tokens in the response. If None, uses the client's default.
        return_raw_results: Whether to return the raw results from the API.
        verbose: True to print out parameter values.
        result_array_getter: Custom way to extract results from the API response. Can be either:
                           - A string representing a dot-separated path (e.g., 'choices')
                           - A callable that takes the response object and returns a list of results
                           If None, defaults to extracting from `response.choices`.
        result_processor: Optional callable to process each individual result before returning.
                         Takes a single result object (e.g., a choice from response.choices) and
                         returns the processed value. If None, defaults to extracting the text attribute.

    Returns:
        The generated text, or the raw results returned by the API.

    Examples:
        >>> generate_text(
        ...    prompt_or_messages='hello',
        ...    model=MetaGenModels.CLAUDE_4_SONNET,
        ...    max_new_tokens=1024,
        ...    temperature=0,
        ...    return_raw_results=False
        ... )
        'Hello! How can I help you today?'

        >>> generate_text(
        ...    prompt_or_messages=[
        ...        {"role": "system", "content": "You are a helpful assistant."},
        ...        {"role": "user", "content": "What is the capital of France?"}
        ...    ],
        ...    model=MetaGenModels.LLAMA_3_1_405B,
        ...    temperature=0
        ... )
        'The capital of France is Paris.'
    """
    # Prepare parameters using shared logic
    metagen_platform, params = _prepare_completion_params(
        prompt_or_messages=prompt_or_messages,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        metagen_key=api_key,
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(f"return_raw_results: {return_raw_results}")

    # Synchronous API call
    response: CompletionResponse = metagen_platform.chat_completion(
        messages=params["messages"],
        model=params["model"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=params["max_tokens"],
    )

    if return_raw_results:
        return response

    # Process results using custom getters/processors if provided
    if result_array_getter is not None:
        result = get_(response, result_array_getter)[0]
    else:
        result = response.choices[0]

    if result_processor is not None:
        return result_processor(result)
    else:
        return result.text


async def generate_text_async(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
    model: MetaGenModels = MetaGenModels.CLAUDE_4_SONNET,
    max_new_tokens: int = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    metagen_key: str = None,
    timeout: Union[float, Tuple[float, float]] = None,
    connect_timeout: float = None,
    response_timeout: float = None,
    return_raw_results: bool = False,
    verbose: bool = False,
    result_array_getter: Union[str, Callable] = None,
    result_processor: Callable = None,
    **kwargs,
) -> str:
    """
    Generate text using MetaGen API (asynchronous).

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The MetaGen model to use for generating text.
        max_new_tokens: The maximum number of new tokens to generate (excluding the prompt).
        temperature: Controls the "creativity" of the generated text. A higher temperature will
                    result in more creative responses, while a lower temperature will result in
                    more predictable responses.
        top_p: Nucleus sampling parameter. If set, only tokens with cumulative probability <= top_p
               are considered for sampling. Must be between 0 and 1.
        metagen_key: Your MetaGen API key. If None, automatically selects optimal key for the model.
        timeout: Request timeout in seconds. Can be either a float (same timeout for connect and read)
                or a tuple of (connect_timeout, read_timeout). If specified, this takes precedence
                over connect_timeout and response_timeout parameters.
        connect_timeout: Maximum time in seconds to wait for establishing connection to the API.
                        Only used if timeout parameter is None. If None, uses the client's default.
        response_timeout: Maximum time in seconds to wait for the complete response stream.
                         Only used if timeout parameter is None. This is the total time allowed
                         for receiving all tokens in the response. If None, uses the client's default.
        return_raw_results: Whether to return the raw results from the API.
        verbose: True to print out parameter values.

    Returns:
        The generated text, or the raw results returned by the API.

    Examples:
        >>> await generate_text_async(
        ...    prompt_or_messages='hello',
        ...    model=MetaGenModels.CLAUDE_4_SONNET,
        ...    max_new_tokens=1024,
        ...    temperature=0,
        ...    return_raw_results=False
        ... )
        'Hello! How can I help you today?'

        >>> await generate_text_async(
        ...    prompt_or_messages=[
        ...        {"role": "system", "content": "You are a helpful assistant."},
        ...        {"role": "user", "content": "What is the capital of France?"}
        ...    ],
        ...    model=MetaGenModels.LLAMA_3_1_405B,
        ...    temperature=0
        ... )
        'The capital of France is Paris.'
    """
    # Prepare parameters using shared logic
    metagen_platform, params = _prepare_completion_params(
        prompt_or_messages=prompt_or_messages,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        metagen_key=metagen_key,
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(f"return_raw_results: {return_raw_results}")

    # Asynchronous API call
    response: CompletionResponse = await metagen_platform.chat_completion_async(
        messages=params["messages"],
        model=params["model"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=params["max_tokens"],
    )

    if return_raw_results:
        return response

    # Process results using custom getters/processors if provided
    if result_array_getter is not None:
        result = get_(response, result_array_getter)[0]
    else:
        result = response.choices[0]

    if result_processor is not None:
        return result_processor(result)
    else:
        return result.text


if __name__ == "__main__":
    from rich_python_utils.common_utils.arg_utils.arg_parse import get_parsed_args

    args = get_parsed_args(
        default_prompt="Hello, how are you?",
        default_model="claude-4-sonnet-genai",
        default_max_new_tokens=1024,
        default_temperature=0.7,
        default_top_p=0.9,
        default_return_raw_results=False,
        # region enable these dummy args for bento notebook
        default_log__level=20,
        default_f="",
        # endregion
    )

    _prompt_or_messages = args.prompt
    _model = args.model
    _max_new_tokens = args.max_new_tokens
    _temperature = args.temperature
    _top_p = args.top_p
    _return_raw_results = args.return_raw_results

    _generated_text = generate_text(
        prompt_or_messages=_prompt_or_messages,
        model=_model,
        max_new_tokens=_max_new_tokens,
        temperature=_temperature,
        top_p=_top_p,
        return_raw_results=_return_raw_results,
        verbose=True,
    )

    hprint_message({"response": _generated_text}, title=_model)
