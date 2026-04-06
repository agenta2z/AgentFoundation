

"""Plugboard LLM API — standard API module wrapping PlugboardClient.

Provides the same interface as apis/metagen/metagen_llm.py:
- generate_text() — sync
- generate_text_async() — async
- generate_text_streaming() — async generator yielding text chunks

Uses CAT auth (no API key needed). The api_key parameter is accepted
but ignored for ApiInferencerBase compatibility.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence, Tuple, Union

DEFAULT_PIPELINE = "usecase-dev-ai"


def _build_messages(
    prompt_or_messages: Union[str, Dict, List[str], List[Dict]],
    system_prompt: str = "",
) -> Tuple[List[Dict[str, str]], str]:
    """Convert various input formats to (messages_list, system_prompt).

    Handles str, dict, list[str], list[dict]. Extracts system messages
    from the list and appends them to system_prompt.

    Returns:
        Tuple of (messages, system_prompt) where messages is a list of
        {"role": ..., "content": ...} dicts (no system messages).
    """
    messages: List[Dict[str, str]] = []
    system_parts: List[str] = []

    if system_prompt:
        system_parts.append(system_prompt)

    if isinstance(prompt_or_messages, str):
        messages.append({"role": "user", "content": prompt_or_messages})

    elif isinstance(prompt_or_messages, dict):
        role = prompt_or_messages.get("role", "user")
        content = prompt_or_messages.get("content", str(prompt_or_messages))
        if role == "system":
            system_parts.append(content)
        else:
            messages.append({"role": role, "content": content})

    elif isinstance(prompt_or_messages, (list, tuple)):
        if len(prompt_or_messages) == 0:
            pass
        elif isinstance(prompt_or_messages[0], str):
            # Sequence of strings — alternate user/assistant
            for i in range(0, len(prompt_or_messages) - 1, 2):
                messages.append({"role": "user", "content": prompt_or_messages[i]})
                if i + 1 < len(prompt_or_messages):
                    messages.append(
                        {"role": "assistant", "content": prompt_or_messages[i + 1]}
                    )
            if len(prompt_or_messages) % 2 == 1:
                messages.append(
                    {"role": "user", "content": prompt_or_messages[-1]}
                )
        elif isinstance(prompt_or_messages[0], dict):
            for msg in prompt_or_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_parts.append(content)
                else:
                    messages.append({"role": role, "content": content})
    else:
        raise ValueError(
            "'prompt_or_messages' must be str, dict, or sequence of str/dict"
        )

    combined_system = "\n\n".join(system_parts) if system_parts else ""
    return messages, combined_system


async def generate_text_streaming(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
    model: str = "",
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    system_prompt: str = "",
    pipeline: str = DEFAULT_PIPELINE,
    model_pipeline_overrides: Dict[str, str] | None = None,
    **kwargs: Any,
):
    """Stream text chunks from Plugboard.

    Creates a PlugboardClient per call and yields text chunks as they arrive.

    Args:
        prompt_or_messages: Input in various formats.
        model: Model identifier.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        system_prompt: System prompt text.
        pipeline: Plugboard pipeline name.
        model_pipeline_overrides: Per-model pipeline overrides.

    Yields:
        str: Text chunks from the streaming response.
    """
    # TODO: migrate plugboard_client
    from rankevolve.src.server.llm.plugboard_client import PlugboardClient

    messages, system = _build_messages(prompt_or_messages, system_prompt)
    client = PlugboardClient(
        pipeline=pipeline,
        model_pipeline_overrides=model_pipeline_overrides or {},
    )
    async for chunk in client.stream_response(
        messages=messages,
        system=system,
        model=model,
        max_tokens=max_new_tokens,
        temperature=temperature,
    ):
        yield chunk


async def generate_text_async(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
    model: str = "",
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    system_prompt: str = "",
    pipeline: str = DEFAULT_PIPELINE,
    model_pipeline_overrides: Dict[str, str] | None = None,
    api_key: str | None = None,  # Accepted but ignored — Plugboard uses CAT auth
    **kwargs: Any,
) -> str:
    """Generate text from Plugboard by accumulating streaming chunks.

    Args:
        prompt_or_messages: Input in various formats.
        model: Model identifier.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        system_prompt: System prompt text.
        pipeline: Plugboard pipeline name.
        model_pipeline_overrides: Per-model pipeline overrides.
        api_key: Ignored — Plugboard uses CAT auth. Accepted for
            ApiInferencerBase compatibility.

    Returns:
        Complete generated text.
    """
    chunks = []
    async for chunk in generate_text_streaming(
        prompt_or_messages=prompt_or_messages,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        pipeline=pipeline,
        model_pipeline_overrides=model_pipeline_overrides,
    ):
        chunks.append(chunk)
    return "".join(chunks)


def generate_text(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
    model: str = "",
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    system_prompt: str = "",
    pipeline: str = DEFAULT_PIPELINE,
    model_pipeline_overrides: Dict[str, str] | None = None,
    api_key: str | None = None,  # Accepted but ignored — Plugboard uses CAT auth
    **kwargs: Any,
) -> str:
    """Generate text from Plugboard (synchronous wrapper).

    Args:
        prompt_or_messages: Input in various formats.
        model: Model identifier.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        system_prompt: System prompt text.
        pipeline: Plugboard pipeline name.
        model_pipeline_overrides: Per-model pipeline overrides.
        api_key: Ignored — Plugboard uses CAT auth.

    Returns:
        Complete generated text.
    """
    coro = generate_text_async(
        prompt_or_messages=prompt_or_messages,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        pipeline=pipeline,
        model_pipeline_overrides=model_pipeline_overrides,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already in an async context — run in a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
