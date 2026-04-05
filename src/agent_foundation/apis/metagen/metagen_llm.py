import json
import os
import random
from enum import StrEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from metagen import (
    CompletionResponse,
    Dialog,
    DialogMessage,
    DialogSource,
    DialogTextContent,
    DialogTextContentDeltaEvent,
    DialogTextContentEndEvent,
    Message,
    MetaGenKey,
    MetaGenPlatform,
    thrift_platform_factory,
)

from agent_foundation.apis.common import _resolve_llm_timeout

from rich_python_utils.common_utils import get_
from rich_python_utils.console_utils import hprint_message


class CompletionMode(StrEnum):
    """
    Completion API mode for MetaGen calls.

    Different models may require different MetaGen API endpoints:
    - CHAT: Uses chat_completion (legacy, has hardcoded temperature=0.6/top_p=0.9 defaults)
    - DIALOG: Uses dialog_completion (modern, nullable temperature/top_p, required for Claude 4.6+)
    - AUTO: Automatically selects based on model name (Claude → DIALOG, others → CHAT)
    """

    CHAT = "chat"
    DIALOG = "dialog"
    AUTO = "auto"


class MetaGenModels(StrEnum):
    """
    Enumeration for major supported MetaGen models.
    """

    # Claude models
    CLAUDE_4_6_OPUS = "claude-4-6-opus-genai"
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


# Models that require dialog_completion (Claude 4.6+ rejects dual temperature+top_p)
_DIALOG_MODE_PREFIXES = {"claude"}

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
    f"{MetaGenModels.CLAUDE_4_6_OPUS}": 128000,
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


def _resolve_completion_mode(model: str, mode: CompletionMode) -> CompletionMode:
    """
    Resolve AUTO completion mode to a concrete CHAT or DIALOG mode.

    Args:
        model: The model string to check.
        mode: The requested completion mode.

    Returns:
        CompletionMode.CHAT or CompletionMode.DIALOG
    """
    if mode != CompletionMode.AUTO:
        return mode
    model_lower = model.lower()
    for prefix in _DIALOG_MODE_PREFIXES:
        if prefix in model_lower:
            return CompletionMode.DIALOG
    return CompletionMode.CHAT


def _build_dialog(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
) -> Dialog:
    """
    Convert various input formats to a MetaGen Dialog object.

    Args:
        prompt_or_messages: Input in various formats - string, dict, or sequence.

    Returns:
        Dialog object compatible with dialog_completion API.
    """
    dialog_messages: list[DialogMessage] = []

    source_map = {
        "system": DialogSource.SYSTEM,
        "user": DialogSource.USER,
        "assistant": DialogSource.ASSISTANT,
    }

    def _make_msg(role: str, content: str) -> DialogMessage:
        source = source_map.get(role, DialogSource.USER)
        return DialogMessage(
            source=source,
            contents=[DialogTextContent(text=content)],
        )

    if isinstance(prompt_or_messages, str):
        dialog_messages.append(_make_msg("user", prompt_or_messages))

    elif isinstance(prompt_or_messages, Dict):
        role = prompt_or_messages.get("role", "user")
        dialog_messages.append(_make_msg(role, prompt_or_messages["content"]))

    elif isinstance(prompt_or_messages, (List, Tuple)):
        if isinstance(prompt_or_messages[0], str):
            for i in range(0, len(prompt_or_messages) - 1, 2):
                dialog_messages.append(_make_msg("user", prompt_or_messages[i]))
                if i + 1 < len(prompt_or_messages):
                    dialog_messages.append(
                        _make_msg("assistant", prompt_or_messages[i + 1])
                    )
            if len(prompt_or_messages) % 2 == 1:
                dialog_messages.append(_make_msg("user", prompt_or_messages[-1]))
        elif isinstance(prompt_or_messages[0], Dict):
            for turn in prompt_or_messages:
                role = turn.get("role", "user")
                dialog_messages.append(_make_msg(role, turn["content"]))
    else:
        raise ValueError(
            "'prompt_or_messages' must be one of str, Dict, or a sequence of strs or Dicts"
        )

    return Dialog(messages=dialog_messages)


def _extract_dialog_response_text(response) -> str:
    """
    Extract text from a DialogCompletionResponse.

    The dialog response structure differs from chat:
      dialog: response.choices[0].dialog.messages[-1].contents[0].text
      chat:   response.choices[0].text
    """
    choice = response.choices[0]
    if hasattr(choice, "dialog") and choice.dialog:
        last_msg = choice.dialog.messages[-1]
        for content in last_msg.contents:
            if isinstance(content, DialogTextContent):
                return content.text
        return str(last_msg.contents[0]) if last_msg.contents else ""
    if hasattr(choice, "text"):
        return choice.text
    return str(choice)


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
    completion_mode: CompletionMode = CompletionMode.AUTO,
    **kwargs,
) -> Tuple[MetaGenPlatform, Dict, CompletionMode]:
    """
    Prepare common parameters for both sync and async MetaGen completion calls.

    Returns:
        Tuple of (metagen_platform, params_dict, resolved_completion_mode)
    """
    # Auto-select optimal key if not provided
    if metagen_key is None:
        metagen_key = get_optimal_key_for_model(model)

    # Build parameters dict
    model_str = f"{model}"
    if not max_new_tokens:
        max_new_tokens = DEFAULT_MAX_TOKENS.get(model_str, 8192)

    resolved_mode = _resolve_completion_mode(model_str, completion_mode)

    params = {
        "model": model_str,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    if resolved_mode == CompletionMode.DIALOG:
        params["dialog"] = _build_dialog(prompt_or_messages)
    else:
        messages = _get_messages(prompt_or_messages)
        params["messages"] = messages.build()

    # Handle timeout setting
    timeout = _resolve_llm_timeout(
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout,
    )

    params.update(kwargs)

    if verbose:
        hprint_message(
            {**params, "Using MetaGen key": metagen_key, "mode": resolved_mode}
        )

    # Create metagen platform
    metagen_platform: MetaGenPlatform = thrift_platform_factory.create(
        MetaGenKey(key=metagen_key),
        auto_rate_limit=True,
    )

    return metagen_platform, params, resolved_mode


def _execute_chat_completion(
    metagen_platform: MetaGenPlatform, params: Dict
) -> CompletionResponse:
    """Execute a chat_completion call (legacy API)."""
    return metagen_platform.chat_completion(
        messages=params["messages"],
        model=params["model"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=params["max_tokens"],
    )


async def _execute_chat_completion_async(
    metagen_platform: MetaGenPlatform, params: Dict
) -> CompletionResponse:
    """Execute a chat_completion_async call (legacy API)."""
    return await metagen_platform.chat_completion_async(
        messages=params["messages"],
        model=params["model"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=params["max_tokens"],
    )


def _execute_dialog_completion(metagen_platform: MetaGenPlatform, params: Dict):
    """
    Execute a dialog_completion call (modern API).

    Only passes temperature OR top_p (not both) since models like
    Claude Opus 4.6 reject having both set simultaneously.
    """
    kwargs = {
        "dialog": params["dialog"],
        "model": params["model"],
        "max_tokens": params["max_tokens"],
    }
    # Only pass temperature (not top_p) to avoid the dual-parameter rejection
    if params.get("temperature") is not None:
        kwargs["temperature"] = params["temperature"]
    elif params.get("top_p") is not None:
        kwargs["top_p"] = params["top_p"]

    return metagen_platform.dialog_completion(**kwargs)


async def _execute_dialog_completion_async(
    metagen_platform: MetaGenPlatform, params: Dict
):
    """
    Execute a dialog_completion_async call (modern API).

    Only passes temperature OR top_p (not both).
    """
    kwargs = {
        "dialog": params["dialog"],
        "model": params["model"],
        "max_tokens": params["max_tokens"],
    }
    if params.get("temperature") is not None:
        kwargs["temperature"] = params["temperature"]
    elif params.get("top_p") is not None:
        kwargs["top_p"] = params["top_p"]

    return await metagen_platform.dialog_completion_async(**kwargs)


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
    completion_mode: CompletionMode = CompletionMode.AUTO,
    **kwargs,
) -> str:
    """
    Generate text using MetaGen API (synchronous).

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The MetaGen model to use for generating text.
        max_new_tokens: The maximum number of new tokens to generate (excluding the prompt).
        temperature: Controls the "creativity" of the generated text.
        top_p: Nucleus sampling parameter.
        api_key: Your MetaGen API key. If None, automatically selects optimal key for the model.
        timeout: Request timeout in seconds.
        connect_timeout: Maximum time in seconds to wait for establishing connection.
        response_timeout: Maximum time in seconds to wait for the complete response stream.
        return_raw_results: Whether to return the raw results from the API.
        verbose: True to print out parameter values.
        result_array_getter: Custom way to extract results from the API response.
        result_processor: Optional callable to process each individual result.
        completion_mode: Which MetaGen API to use. Options:
            - CompletionMode.AUTO (default): Auto-detect based on model name.
              Claude models use dialog_completion; others use chat_completion.
            - CompletionMode.DIALOG: Force dialog_completion (modern API, nullable temp/top_p).
            - CompletionMode.CHAT: Force chat_completion (legacy API, hardcoded defaults).

    Returns:
        The generated text, or the raw results returned by the API.

    Examples:
        >>> generate_text(
        ...    prompt_or_messages='hello',
        ...    model=MetaGenModels.CLAUDE_4_6_OPUS,
        ...    max_new_tokens=1024,
        ...    temperature=0.7,
        ... )
        'Hello! How can I help you today?'

        >>> generate_text(
        ...    prompt_or_messages='hello',
        ...    model=MetaGenModels.LLAMA_3_1_405B,
        ...    completion_mode=CompletionMode.CHAT,
        ... )
        'Hi there!'
    """
    metagen_platform, params, resolved_mode = _prepare_completion_params(
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
        completion_mode=completion_mode,
        **kwargs,
    )

    if verbose:
        print(f"return_raw_results: {return_raw_results}")

    if resolved_mode == CompletionMode.DIALOG:
        response = _execute_dialog_completion(metagen_platform, params)
    else:
        response = _execute_chat_completion(metagen_platform, params)

    if return_raw_results:
        return response

    if result_array_getter is not None:
        result = get_(response, result_array_getter)[0]
    else:
        if resolved_mode == CompletionMode.DIALOG:
            return _extract_dialog_response_text(response)
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
    completion_mode: CompletionMode = CompletionMode.AUTO,
    **kwargs,
) -> str:
    """
    Generate text using MetaGen API (asynchronous).

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The MetaGen model to use for generating text.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: Controls the "creativity" of the generated text.
        top_p: Nucleus sampling parameter.
        metagen_key: Your MetaGen API key. If None, automatically selects optimal key.
        timeout: Request timeout in seconds.
        connect_timeout: Maximum time in seconds for connection.
        response_timeout: Maximum time in seconds for complete response.
        return_raw_results: Whether to return the raw results from the API.
        verbose: True to print out parameter values.
        result_array_getter: Custom way to extract results.
        result_processor: Optional callable to process each result.
        completion_mode: Which MetaGen API to use (AUTO, DIALOG, or CHAT).

    Returns:
        The generated text, or the raw results returned by the API.

    Examples:
        >>> await generate_text_async(
        ...    prompt_or_messages='hello',
        ...    model=MetaGenModels.CLAUDE_4_6_OPUS,
        ...    max_new_tokens=1024,
        ...    temperature=0.7,
        ... )
        'Hello! How can I help you today?'
    """
    metagen_platform, params, resolved_mode = _prepare_completion_params(
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
        completion_mode=completion_mode,
        **kwargs,
    )

    if verbose:
        print(f"return_raw_results: {return_raw_results}")

    if resolved_mode == CompletionMode.DIALOG:
        response = await _execute_dialog_completion_async(metagen_platform, params)
    else:
        response = await _execute_chat_completion_async(metagen_platform, params)

    if return_raw_results:
        return response

    if result_array_getter is not None:
        result = get_(response, result_array_getter)[0]
    else:
        if resolved_mode == CompletionMode.DIALOG:
            return _extract_dialog_response_text(response)
        result = response.choices[0]

    if result_processor is not None:
        return result_processor(result)
    else:
        return result.text


async def generate_text_streaming(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
    model: MetaGenModels = MetaGenModels.CLAUDE_4_SONNET,
    max_new_tokens: int = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    metagen_key: str = None,
    timeout: Union[float, Tuple[float, float]] = None,
    connect_timeout: float = None,
    response_timeout: float = None,
    verbose: bool = False,
    completion_mode: CompletionMode = CompletionMode.AUTO,
    **kwargs,
):
    """Stream text chunks from MetaGen API via dialog_completion_stream_events_async.

    Yields raw text deltas as they arrive. Forces DIALOG completion mode
    since streaming is only available via the dialog API.

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The MetaGen model to use.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Controls creativity of generated text.
        top_p: Nucleus sampling parameter.
        metagen_key: MetaGen API key. If None, auto-selects optimal key.
        timeout: Request timeout in seconds.
        connect_timeout: Maximum time for establishing connection.
        response_timeout: Maximum time for complete response stream.
        verbose: True to print parameter values.
        completion_mode: Ignored — always forced to DIALOG for streaming.

    Yields:
        str: Text chunks as they arrive from the streaming API.
    """
    metagen_platform, params, _ = _prepare_completion_params(
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
        completion_mode=CompletionMode.DIALOG,  # Force DIALOG for streaming
        **kwargs,
    )

    # Build kwargs for the streaming call (same pattern as _execute_dialog_completion)
    stream_kwargs = {
        "dialog": params["dialog"],
        "model": params["model"],
        "max_tokens": params["max_tokens"],
    }
    if params.get("temperature") is not None:
        stream_kwargs["temperature"] = params["temperature"]
    elif params.get("top_p") is not None:
        stream_kwargs["top_p"] = params["top_p"]

    async for event in metagen_platform.dialog_completion_stream_events_async(
        **stream_kwargs
    ):
        if isinstance(event, (DialogTextContentDeltaEvent, DialogTextContentEndEvent)):
            if event.delta:
                yield event.delta


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
