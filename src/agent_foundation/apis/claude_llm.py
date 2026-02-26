from enum import StrEnum
from os import environ, path
from typing import Union, List, Dict, Sequence, Tuple

from anthropic import Anthropic

from science_modeling_tools.apis.common import _resolve_llm_timeout
from rich_python_utils.console_utils import hprint_message

ENV_NAME_CLAUDE_API_KEY = 'ANTHROPIC_API_KEY'


class ClaudeModels(StrEnum):
    """
    Enumeration for supported Claude models.
    See details at https://docs.anthropic.com/claude/docs/models-overview
    """
    CLAUDE_3_OPUS = 'claude-3-opus-20240229'
    CLAUDE_35_SONNET = 'claude-3-5-sonnet-20241022'
    CLAUDE_37_SONNET = 'claude-3-7-sonnet-20250219'
    CLAUDE_40_SONNET = 'claude-sonnet-4-20250514'
    CLAUDE_45_SONNET = 'claude-sonnet-4-5-20250929'
    CLAUDE_41OPUS = 'claude-opus-4-1-20250805'
    CLAUDE_3_HAIKU = 'claude-3-haiku-20240307'

DEFAULT_MAX_TOKENS = {
    f'{ClaudeModels.CLAUDE_3_OPUS}': 4096,
    f'{ClaudeModels.CLAUDE_35_SONNET}': 4096,
    f'{ClaudeModels.CLAUDE_37_SONNET}': 8192,
    f'{ClaudeModels.CLAUDE_45_SONNET}': 8192,
    f'{ClaudeModels.CLAUDE_3_HAIKU}': 4096
}

DEFAULT_CLAUDE_MODEL = ClaudeModels.CLAUDE_45_SONNET


def _get_messages(prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]]):
    """
    Convert various input formats into Claude's message format.
    """
    if isinstance(prompt_or_messages, str):
        if path.exists(prompt_or_messages):
            with open(prompt_or_messages, 'r', encoding='utf-8') as f:
                prompt_or_messages = f.read()
        return [
            {
                'role': 'user',
                'content': prompt_or_messages
            }
        ]
    elif isinstance(prompt_or_messages, Dict):
        return [prompt_or_messages]
    elif isinstance(prompt_or_messages, (List, Tuple)):
        if isinstance(prompt_or_messages[0], str):
            messages = []
            for i in range(0, len(prompt_or_messages) - 1, 2):
                messages.extend(
                    (
                        {
                            'role': 'user',
                            'content': prompt_or_messages[i]
                        },
                        {
                            'role': 'assistant',
                            'content': prompt_or_messages[i + 1]
                        }
                    )
                )
            messages.append(
                {
                    'role': 'user',
                    'content': prompt_or_messages[-1]
                }
            )
            return messages
        elif isinstance(prompt_or_messages[0], Dict):
            return list(prompt_or_messages)
    raise ValueError(
        "'prompt_or_messages' must be one of str, Dict, or a sequence of strs or Dicts"
    )


def generate_text(
        prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
        model: ClaudeModels = DEFAULT_CLAUDE_MODEL,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = None,
        stop: List[str] = None,
        api_key: str = None,
        timeout: Union[float, Tuple[float, float]] = None,
        connect_timeout: float = None,
        response_timeout: float = None,
        return_raw_results: bool = False,
        verbose: bool = False,
        **kwargs
) -> Union[str, List[str], Dict]:
    """
    Generate text using Claude API.

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The Claude model to use for generating text.
        max_new_tokens: The maximum number of new tokens to generate (excluding the prompt).
        temperature: Controls the randomness of the output. Higher values mean more random.
        top_p: Nucleus sampling parameter. If set, only tokens with cumulative probability <= top_p
              are considered for sampling. Must be between 0 and 1.
        stop: List of strings that will stop generation when encountered.
                Note: This is implemented at the application level since Claude API
                doesn't directly support stop sequences.
        api_key: Your Claude API key. If not provided, reads from ENV_NAME_CLAUDE_API_KEY.
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
        **kwargs: Additional parameters to pass to the Claude API.

    Returns:
        Generated text, or raw API response if return_raw_results=True.

    Notes:
        - Claude’s API does not support top_k.
        - Claude's API (including Claude 3.5 Sonnet) does not support generating multiple answers in a single request like OpenAI's n parameter. Each API call to Claude will generate exactly one response.
        - Timeout behavior:
            - If timeout is specified, it takes precedence over connect_timeout and response_timeout
            - If timeout is None, connect_timeout and response_timeout are used to construct the
              timeout tuple (connect_timeout, response_timeout)
            - A TimeoutError will be raised if either the connection establishment or complete
              response exceeds their respective timeouts
            - Response timeout applies to the entire streaming response, not per token
            - Default timeouts are used for any timeout value that remains None

    """
    messages = _get_messages(prompt_or_messages)
    api_key = api_key or environ[ENV_NAME_CLAUDE_API_KEY]

    client = Anthropic(api_key=api_key)

    # region build parameters dict
    model = f'{model}'
    if not max_new_tokens:
        max_new_tokens = DEFAULT_MAX_TOKENS.get(model, 4096)

    params = {
        'model': model,
        'messages': messages,
        'max_tokens': max_new_tokens,
        'temperature': temperature,
    }

    # Add optional sampling parameters
    if top_p is not None:
        params['top_p'] = top_p

    # region Handle timeout setting
    timeout = _resolve_llm_timeout(
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout
    )

    if timeout is not None:
        params['timeout'] = timeout
    # endregion

    params.update(kwargs)  # Add any additional kwargs
    if verbose:
        hprint_message(
            {
                **params,
                'return_raw_results': return_raw_results
            }
        )
    # endregion

    response = client.messages.create(**params)

    if return_raw_results:
        return response

    generated_text = response.content[0].text.strip()

    # Apply stop sequences if provided
    if stop:
        for stop_sequence in stop:
            stop_idx = generated_text.find(stop_sequence)
            if stop_idx != -1:
                generated_text = generated_text[:stop_idx]

    return generated_text.strip()


if __name__ == '__main__':
    from rich_python_utils.common_utils.arg_utils.arg_parse import get_parsed_args

    args = get_parsed_args(
        default_prompt="Hello! What's the capital of France?",
        default_model='claude-3-5-sonnet-20241022',
        default_max_new_tokens=1024,
        default_top_p=0.9,
        default_stop='[]',
        default_temperature=0.7,
        default_return_raw_results=False
    )

    _prompt_or_messages = args.prompt
    _model = args.model
    _max_new_tokens = args.max_new_tokens
    _top_p = args.top_p
    _stop = args.stop
    _temperature = args.temperature
    _return_raw_results = args.return_raw_results

    _generated_text = generate_text(
        prompt_or_messages=_prompt_or_messages,
        model=_model,
        max_new_tokens=_max_new_tokens,
        top_p=_top_p,
        stop=_stop,
        temperature=_temperature,
        return_raw_results=_return_raw_results,
        verbose=True
    )

    hprint_message({'response': _generated_text}, title=_model)
