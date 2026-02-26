from typing import Union, List, Dict, Sequence, Tuple, Callable

import openai
from enum import StrEnum
from os import environ, path

from science_modeling_tools.apis.common import _resolve_llm_timeout
from rich_python_utils.common_utils import get_
from rich_python_utils.console_utils import hprint_message
from rich_python_utils.io_utils.text_io import read_all_text

ENV_NAME_OPENAI_API_KEY = 'OPENAI_APIKEY'


class OpenAIModels(StrEnum):
    """
    Enumeration for major supported ChatGPT models.
    See details at https://platform.openai.com/docs/models/overview
    """
    GPT3 = 'gpt-3.5-turbo'
    GPT3_16K = 'gpt-3.5-turbo-16k'
    GPT4 = "gpt-4"
    GPT4_TURBO = 'gpt-4-turbo'
    GPT4_32K = "gpt-4-32k-0613"
    GPT4O = "gpt-4o"


DEFAULT_MAX_TOKENS = {
    f'{OpenAIModels.GPT4}': 2048,
    f'{OpenAIModels.GPT4_32K}': 3096,
    f'{OpenAIModels.GPT3}': 1024,
    f'{OpenAIModels.GPT3_16K}': 3096,
    f'{OpenAIModels.GPT4O}': 4096
}


def _get_messages(prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]]):
    if isinstance(prompt_or_messages, str):
        if path.exists(prompt_or_messages):
            prompt_or_messages = read_all_text(prompt_or_messages)
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
        prompt_or_messages: str,
        model: OpenAIModels = OpenAIModels.GPT4O,
        max_new_tokens: int = None,
        n: int = 1,
        top_p: float = None,
        stop: str = None,
        temperature: float = 0.7,
        api_key: str = None,
        timeout: Union[float, Tuple[float, float]] = None,
        connect_timeout: float = None,
        response_timeout: float = None,
        return_raw_results: bool = False,
        verbose: bool = False,
        result_array_getter: Union[str, Callable] = None,
        result_processor: Callable = None,
        **kwargs
):
    """

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The OpenAI model to use for generating text.
        max_new_tokens: The maximum number of new tokens to generate (excluding the prompt).
        n: The number of candidate answers to generate.
        top_p: Nucleus sampling parameter. If set, only tokens with cumulative probability <= top_p
              are considered for sampling. Must be between 0 and 1.
        stop: Provide sequences where the API will stop generating further tokens.
        temperature: Controls the "creativity" of the generated text. A higher temperature will result in more creative responses, while a lower temperature will result in more predictable responses.
        verbose: True to print out parameter values.
        api_key: Your OpenAI API key. If not provided, the key will be read from the environment variable `ENV_NAME_OPENAI_API_KEY`.
        timeout: Request timeout in seconds. Can be either a float (same timeout for connect and read)
                or a tuple of (connect_timeout, read_timeout). If specified, this takes precedence
                over connect_timeout and response_timeout parameters.
        connect_timeout: Maximum time in seconds to wait for establishing connection to the API.
                        Only used if timeout parameter is None. If None, uses the client's default.
        response_timeout: Maximum time in seconds to wait for the complete response stream.
                         Only used if timeout parameter is None. This is the total time allowed
                         for receiving all tokens in the response. If None, uses the client's default.
        return_raw_results: Whether to return the raw results from the API.
        result_array_getter: Custom way to extract results from the API response. Can be either:
                           - A string representing a dot-separated path (e.g., 'choices')
                           - A callable that takes the completions object and returns a list of results
                           If None, defaults to extracting from `completions.choices`.
        result_processor: Optional callable to process each individual result before returning.
                         Takes a single result object (e.g., a choice from completions.choices) and
                         returns the processed value. If None, defaults to extracting and stripping
                         the message content. Applied to each result when n > 1.

    Returns:
        The generated text, or the raw results returned by the API.

    Notes:
        - OpenAI’s API does not support top_k.

    Examples:
        >>> generate_text(
        ...    prompt_or_messages='hello',
        ...    model=OpenAIModels.GPT4,
        ...    max_new_tokens=1024,
        ...    n=1,
        ...    stop=None,
        ...    temperature=0,
        ...    api_key=None,
        ...    return_raw_results=False
        ... )
        'Hello! How can I help you today?'

        >>> generate_text(
        ...    prompt_or_messages='What would be a good company name for a company that makes colorful socks?',
        ...    model=OpenAIModels.GPT4,
        ...    max_new_tokens=1024,
        ...    n=2,
        ...    stop=None,
        ...    temperature=0,
        ...    api_key=None,
        ...    return_raw_results=False
        ... )
        ['SockSpectrum', 'SockSpectrum']
    """
    api_key = api_key or environ[ENV_NAME_OPENAI_API_KEY]
    client = openai.OpenAI(api_key=api_key)

    # region build parameters dict
    model = f'{model}'
    if not max_new_tokens:
        max_new_tokens = DEFAULT_MAX_TOKENS.get(model, 4096)

    if not stop:
        stop = None

    messages = _get_messages(prompt_or_messages)

    params = {
        'model': model,
        'messages': messages,
        'max_tokens': max_new_tokens,
        'n': n,
        'stop': stop,
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

    completions = client.chat.completions.create(**params)

    if return_raw_results:
        # An example raw result
        # {
        #     "choices": [
        #         {
        #             "finish_reason": "stop",
        #             "index": 0,
        #             "message": {
        #                 "content": "Hello! How can I help you today?",
        #                 "role": "assistant"
        #             }
        #         }
        #     ],
        #     "created": 1686840573,
        #     "id": "chatcmpl-7RiZxKVtbDTCPJiEOAoKDHB9mJ3mF",
        #     "model": "gpt-4-0314",
        #     "object": "chat.completion",
        #     "usage": {
        #         "completion_tokens": 9,
        #         "prompt_tokens": 8,
        #         "total_tokens": 17
        #     }
        # }
        return completions

    if n == 1:
        if result_array_getter is not None:
            result = get_(completions, result_array_getter)[0]
        else:
            result = completions.choices[0]

        if result_processor is not None:
            return result_processor(result)
        else:
            return result.message.content.strip()
    else:
        if result_array_getter is not None:
            results = get_(completions, result_array_getter)
        else:
            results = completions.choices

        if result_processor is not None:
            return [result_processor(result) for result in results]
        else:
            return [result.message.content.strip() for result in results]


if __name__ == '__main__':
    from rich_python_utils.common_utils.arg_utils.arg_parse import get_parsed_args

    args = get_parsed_args(
        default_prompt='Hello, how are you?',
        default_model='gpt-4o',
        default_max_new_tokens=1024,
        default_n=1,
        default_top_p=0.9,
        default_stop='[]',
        default_temperature=0.7,
        default_return_raw_results=False,
    )

    _prompt_or_messages = args.prompt
    _model = args.model
    _max_new_tokens = args.max_new_tokens
    _n = args.n
    _top_p = args.top_p
    _stop = args.stop
    _temperature = args.temperature
    _return_raw_results = args.return_raw_results

    _generated_text = generate_text(
        prompt_or_messages=_prompt_or_messages,
        model=_model,
        max_new_tokens=_max_new_tokens,
        n=_n,
        top_p=_top_p,
        stop=_stop,
        temperature=_temperature,
        return_raw_results=_return_raw_results,
        verbose=True
    )

    hprint_message({'response': _generated_text}, title=_model)
