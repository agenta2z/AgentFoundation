"""OpenAI (GPT) models via Atlassian AI Gateway.

Mirror of ``ai_gateway_claude_llm.py`` but for the OpenAI vendor route. Routes
chat-completion requests through the AI Gateway's OpenAI endpoint
(``/v1/openai/v1/chat/completions``) instead of talking to api.openai.com
directly, so it benefits from centralized auth (SLAuth), use-case allowlisting,
rate limiting, monitoring, and cost tracking.

Supports the same gateway access modes as the Claude backend:
- "direct": shell out to ``atlas slauth token`` CLI, send via httpx (no local
  server needed) — verified working.
- "sdk": use the AI Gateway SDK's typed ``client.openai.v1_chat_completions``
  with a pre-minted SLAuth token (no local slauth server needed).
- "slauth_server": use the SDK with ``SlauthServerAuthFilter`` (needs a local
  ``atlas slauth server``).
- "auto": detect/cascade.

Notes on reasoning models (gpt-5.x / o-series): these spend output budget on an
internal reasoning trace before the visible answer. Always give a generous
``max_new_tokens`` (mapped to ``max_completion_tokens``) or the visible content
may come back empty. ``reasoning_effort`` (none/minimal/low/medium/high) is
forwarded when supplied.
"""

import asyncio
import json
import logging
from enum import StrEnum
from os import environ, path
from typing import AsyncIterator, Dict, List, Sequence, Tuple, Union

import httpx
from ai_gateway.client import AIGatewayClient
from ai_gateway.constants import AIGatewayHeaders
from ai_gateway.models.common import HttpHeaders
from ai_gateway.models.wrapper import RequestWrapper

from agent_foundation.apis.ag.gateway_mode import (
    DEFAULT_AI_GATEWAY_BASE_URL,
    DEFAULT_CLOUD_ID,
    DEFAULT_PROXIMITY_PORT,
    DEFAULT_SLAUTH_GROUPS,
    DEFAULT_SLAUTH_SERVER_URL,
    DEFAULT_USE_CASE_ID,
    GatewayMode,
    build_direct_headers,
    detect_available_mode,
    get_direct_slauth_token,
)
from agent_foundation.apis.common import _resolve_llm_timeout
from rich_python_utils.console_utils import hprint_message

logger = logging.getLogger(__name__)

# Env var names (shared semantics with the Claude backend)
ENV_NAME_AI_GATEWAY_USER_ID = 'AI_GATEWAY_USER_ID'
ENV_NAME_AI_GATEWAY_CLOUD_ID = 'AI_GATEWAY_CLOUD_ID'
ENV_NAME_AI_GATEWAY_USE_CASE_ID = 'AI_GATEWAY_USE_CASE_ID'
ENV_NAME_AI_GATEWAY_BASE_URL = 'AI_GATEWAY_BASE_URL'
ENV_NAME_SLAUTH_SERVER_URL = 'SLAUTH_SERVER_URL'

# The relative path of the OpenAI chat-completions route on the AI Gateway.
# Verified live: POST {base_url}/v1/openai/v1/chat/completions -> 200.
OPENAI_CHAT_COMPLETIONS_PATH = "/v1/openai/v1/chat/completions"


class AIGatewayOpenAIModels(StrEnum):
    """OpenAI models exposed via the AI Gateway (vendor=openai route).

    Only the identifiers below were confirmed available (HTTP 200) for the
    offline-eval use-case at the time of writing; other ids such as ``gpt-4.1``,
    ``gpt-4o``, ``gpt-oss-*`` and the ``o``-series returned
    ``400 "Model Id ... not found in AI Gateway"`` for that use-case and are
    intentionally omitted. Add them back here once the gateway/use-case offers
    them (the request path and code already support any chat model id — you can
    also pass a raw string model id directly to ``generate_text``).
    """
    # GPT-5.x family (reasoning-capable) — confirmed available.
    GPT_55 = 'gpt-5.5-2026-04-23'
    GPT_5 = 'gpt-5-latest'


# Conservative default output budgets. Reasoning models need headroom for the
# internal reasoning trace BEFORE the visible answer, so defaults are generous.
DEFAULT_MAX_TOKENS = {
    f'{AIGatewayOpenAIModels.GPT_55}': 8192,
    f'{AIGatewayOpenAIModels.GPT_5}': 8192,
}

# Models that use the newer reasoning-style request contract: they require
# ``max_completion_tokens`` (NOT ``max_tokens``) and reject a custom
# ``temperature`` (only the default of 1.0 is accepted). Prefix-matched.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o3", "o4", "gpt-oss")

# Cascade order for auto mode. SDK is preferred over slauth_server because it
# does not require a local slauth server process.
_FALLBACK_MODES = [GatewayMode.DIRECT, GatewayMode.SDK, GatewayMode.SLAUTH_SERVER]


def _is_reasoning_model(model_str: str) -> bool:
    """Whether the model uses the reasoning-style request contract."""
    return model_str.startswith(_REASONING_MODEL_PREFIXES)


def _get_messages(
    prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
) -> List[Dict]:
    """Normalise the prompt/messages input into OpenAI chat-message dicts.

    Accepts:
    - a plain string prompt -> [{"role": "user", "content": <str>}]
    - a path to a text file (its contents become the user prompt)
    - a single message dict -> [dict]
    - a sequence of message dicts -> list(dicts)
    - a sequence of strings -> alternating user messages
    """
    if isinstance(prompt_or_messages, str):
        text = prompt_or_messages
        # If it's a path to an existing file, read its contents as the prompt.
        try:
            if path.isfile(prompt_or_messages):
                with open(prompt_or_messages, 'r', encoding='utf-8') as f:
                    text = f.read()
        except (OSError, ValueError):
            text = prompt_or_messages
        return [{"role": "user", "content": text}]

    if isinstance(prompt_or_messages, dict):
        return [prompt_or_messages]

    if isinstance(prompt_or_messages, Sequence):
        items = list(prompt_or_messages)
        if items and all(isinstance(it, dict) for it in items):
            return items
        # Sequence of strings -> user messages.
        return [{"role": "user", "content": str(it)} for it in items]

    raise TypeError(
        f"Unsupported prompt_or_messages type: {type(prompt_or_messages).__name__}"
    )


def _resolve_config(
    user_id: str = None,
    cloud_id: str = None,
    use_case_id: str = None,
    base_url: str = None,
    slauth_server_url: str = None,
    groups: str = None,
) -> dict:
    """Resolve configuration from parameters, env vars, and defaults."""
    return {
        "base_url": base_url or environ.get(ENV_NAME_AI_GATEWAY_BASE_URL, DEFAULT_AI_GATEWAY_BASE_URL),
        "cloud_id": cloud_id or environ.get(ENV_NAME_AI_GATEWAY_CLOUD_ID, DEFAULT_CLOUD_ID),
        "use_case_id": use_case_id or environ.get(ENV_NAME_AI_GATEWAY_USE_CASE_ID, DEFAULT_USE_CASE_ID),
        "slauth_server_url": slauth_server_url or environ.get(ENV_NAME_SLAUTH_SERVER_URL, DEFAULT_SLAUTH_SERVER_URL),
        "user_id": user_id or environ.get(ENV_NAME_AI_GATEWAY_USER_ID) or environ.get('USER', ''),
        "groups": groups or environ.get("AI_GATEWAY_SLAUTH_GROUPS", DEFAULT_SLAUTH_GROUPS),
    }


def _build_request_payload(
    messages: list,
    model_str: str,
    max_new_tokens: int = None,
    temperature: float = 0.7,
    system: str = None,
    stream: bool = False,
    reasoning_effort: str = None,
    **kwargs,
) -> dict:
    """Build the OpenAI chat-completions request payload.

    Handles the reasoning-model contract: ``max_completion_tokens`` instead of
    ``max_tokens`` and dropping a non-default ``temperature`` (those models only
    accept the default). A ``system`` prompt is prepended as a system message.
    """
    if not max_new_tokens:
        max_new_tokens = DEFAULT_MAX_TOKENS.get(model_str, 4096)

    msgs = list(messages)
    if system:
        msgs = [{"role": "system", "content": system}] + msgs

    payload: Dict = {
        "model": model_str,
        "messages": msgs,
    }

    if _is_reasoning_model(model_str):
        # Reasoning models: token cap field is max_completion_tokens; custom
        # temperature is rejected (only default 1.0 allowed) so we omit it.
        payload["max_completion_tokens"] = max_new_tokens
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
    else:
        payload["max_tokens"] = max_new_tokens
        payload["temperature"] = temperature
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

    if stream:
        payload["stream"] = True

    payload.update(kwargs)
    return payload


def _parse_response_data(response_data: dict, stop: List[str] = None) -> str:
    """Extract the assistant message text from an OpenAI chat-completion."""
    choices = response_data.get("choices")
    if not choices:
        raise Exception(f"Unexpected OpenAI response format (no choices): {response_data}")

    message = choices[0].get("message", {})
    generated_text = (message.get("content") or "").strip()

    if stop:
        for stop_sequence in stop:
            idx = generated_text.find(stop_sequence)
            if idx != -1:
                generated_text = generated_text[:idx]

    return generated_text.strip()


def _send_via_direct(model_str: str, request_payload: dict, config: dict, timeout: float = 120) -> dict:
    """Send via the AI Gateway OpenAI route using an atlas-CLI SLAuth token."""
    env = "prod" if "prod" in config["base_url"] else "staging"
    token = get_direct_slauth_token(env=env, groups=config.get("groups"))
    headers = build_direct_headers(
        token=token,
        user_id=config["user_id"],
        cloud_id=config["cloud_id"],
        use_case_id=config["use_case_id"],
    )
    url = f"{config['base_url']}{OPENAI_CHAT_COMPLETIONS_PATH}"

    resp = httpx.post(url, json=request_payload, headers=headers, timeout=timeout)
    if not (200 <= resp.status_code < 300):
        raise Exception(f"Direct mode: AI Gateway returned status {resp.status_code}: {resp.text}")
    return resp.json()


def _send_via_sdk(model_str: str, request_payload: dict, config: dict, timeout: float = 120) -> dict:
    """Send via the AI Gateway SDK's typed ``client.openai.v1_chat_completions``.

    Uses a pre-minted SLAuth token (atlas CLI), so it needs no local slauth
    server. This is the idiomatic SDK path for the OpenAI vendor route.
    """
    from ai_gateway.client.common.models import (
        ClientRequest,
        ClientResponse,
        SyncFilter,
        SyncFilterChain,
    )
    from ai_gateway.models.openai.chat import ChatCompletionRequest

    env = "prod" if "prod" in config["base_url"] else "staging"
    token = get_direct_slauth_token(env=env, groups=config.get("groups"))

    class _PreMintedSlauthFilter(SyncFilter):
        """Stamp a pre-minted SLAuth token onto every outbound request."""

        def __init__(self, t: str):
            if not t:
                raise ValueError("pre-minted SLAuth token is empty")
            self._token = t

        def filter(self, request: ClientRequest, chain: SyncFilterChain) -> ClientResponse:
            tok = self._token
            if not tok.lower().startswith(("slauth ", "bearer ")):
                tok = f"SLAUTH {tok}"
            request.headers["Authorization"] = tok
            return chain.next(request)

    default_headers = HttpHeaders({
        AIGatewayHeaders.USER_ID: config["user_id"],
        AIGatewayHeaders.CLOUD_ID: config["cloud_id"],
        AIGatewayHeaders.USE_CASE_ID: config["use_case_id"],
    })

    client = AIGatewayClient.sync(
        base_url=config["base_url"],
        default_headers=default_headers,
        filters=[_PreMintedSlauthFilter(token)],
    )

    req = ChatCompletionRequest(**request_payload)
    response = client.openai.v1_chat_completions(RequestWrapper(req))

    if not (200 <= response.http_status.code < 300):
        raw_body = getattr(response, "raw_body", None) or response.body
        raw_str = (
            raw_body.decode("utf-8", errors="replace")
            if isinstance(raw_body, (bytes, bytearray))
            else str(raw_body)
        )
        raise Exception(
            f"SDK mode: AI Gateway returned status {response.http_status.code}: {raw_str}"
        )

    body = response.body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        return json.loads(body)
    if hasattr(body, "model_dump"):
        return body.model_dump()
    return body


def _send_via_slauth_server(model_str: str, request_payload: dict, config: dict, timeout: float = 120) -> dict:
    """Send via the SDK + ``SlauthServerAuthFilter`` (needs local slauth server)."""
    from ai_gateway.models.common import HttpMethod

    groups = config.get("groups") or DEFAULT_SLAUTH_GROUPS
    group_set = {g.strip() for g in groups.split(",") if g.strip()}

    from ai_gateway.client.common.filters import SlauthServerAuthFilter

    slauth_filter = SlauthServerAuthFilter(
        sl_auth_server_url=config["slauth_server_url"],
        groups=group_set,
    )

    default_headers = HttpHeaders({
        AIGatewayHeaders.USER_ID: config["user_id"],
        AIGatewayHeaders.CLOUD_ID: config["cloud_id"],
        AIGatewayHeaders.USE_CASE_ID: config["use_case_id"],
    })

    client = AIGatewayClient.sync(
        base_url=config["base_url"],
        default_headers=default_headers,
        filters=[slauth_filter],
    )

    request = RequestWrapper(
        body=json.dumps(request_payload).encode('utf-8'),
        headers=HttpHeaders({'Content-Type': 'application/json'}),
    )
    response = client.raw.http(
        method=HttpMethod.POST,
        uri=OPENAI_CHAT_COMPLETIONS_PATH,
        request=request,
    )

    if not (200 <= response.http_status.code < 300):
        raw_body = getattr(response, 'raw_body', None) or response.body
        raw_str = (
            raw_body.decode('utf-8', errors='replace')
            if isinstance(raw_body, (bytes, bytearray))
            else str(raw_body)
        )
        raise Exception(
            f"SLAuth server mode: AI Gateway returned status {response.http_status.code}: {raw_str}"
        )

    return json.loads(response.body.decode('utf-8'))


def _resolve_mode_order(gateway_mode: str, config: dict, proximity_port: int):
    """Resolve the gateway mode and the ordered list of modes to attempt."""
    resolved_mode = GatewayMode(gateway_mode)
    is_auto = resolved_mode == GatewayMode.AUTO

    if is_auto:
        try:
            detected = detect_available_mode(
                proximity_port=proximity_port,
                slauth_server_url=config["slauth_server_url"],
            )
            # The OpenAI route has no proximity proxy; prefer direct/sdk.
            resolved_mode = (
                detected if detected in (GatewayMode.DIRECT, GatewayMode.SLAUTH_SERVER)
                else GatewayMode.DIRECT
            )
        except RuntimeError:
            resolved_mode = GatewayMode.DIRECT

    if is_auto:
        modes_to_try = [resolved_mode] + [m for m in _FALLBACK_MODES if m != resolved_mode]
    else:
        modes_to_try = [resolved_mode]

    return resolved_mode, is_auto, modes_to_try


def generate_text(
        prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
        model: AIGatewayOpenAIModels = AIGatewayOpenAIModels.GPT_55,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        stop: List[str] = None,
        system: str = None,
        reasoning_effort: str = None,
        user_id: str = None,
        cloud_id: str = None,
        use_case_id: str = None,
        base_url: str = None,
        slauth_server_url: str = None,
        timeout: Union[float, Tuple[float, float]] = None,
        connect_timeout: float = None,
        response_timeout: float = None,
        return_raw_results: bool = False,
        verbose: bool = False,
        gateway_mode: str = "auto",
        proximity_port: int = DEFAULT_PROXIMITY_PORT,
        groups: str = None,
        **kwargs,
) -> Union[str, Dict]:
    """Generate text using an OpenAI model via the AI Gateway.

    Args mirror ``ai_gateway_claude_llm.generate_text`` with the addition of
    ``reasoning_effort`` (none/minimal/low/medium/high) for reasoning models.

    Returns the generated text, or the raw API response if return_raw_results.
    """
    if "api_key" in kwargs:
        kwargs.pop("api_key")
    messages = _get_messages(prompt_or_messages)

    config = _resolve_config(
        user_id=user_id,
        cloud_id=cloud_id,
        use_case_id=use_case_id,
        base_url=base_url,
        slauth_server_url=slauth_server_url,
        groups=groups,
    )

    resolved_mode, is_auto, modes_to_try = _resolve_mode_order(
        gateway_mode, config, proximity_port
    )

    model_str = f"{model}"
    request_payload = _build_request_payload(
        messages=messages,
        model_str=model_str,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system=system,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )

    timeout_value = _resolve_llm_timeout(
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout,
    )
    request_timeout = timeout_value if isinstance(timeout_value, (int, float)) and timeout_value else 120

    if verbose:
        hprint_message(
            {
                "model": model_str,
                "gateway_mode": str(resolved_mode),
                "base_url": config["base_url"],
                "use_case_id": config["use_case_id"],
                "reasoning_effort": reasoning_effort,
                "timeout": timeout_value,
                **kwargs,
            },
            title="AI Gateway OpenAI API Parameters",
        )

    last_error = None
    for mode in modes_to_try:
        try:
            if mode == GatewayMode.DIRECT:
                response_data = _send_via_direct(model_str, request_payload, config, timeout=request_timeout)
            elif mode == GatewayMode.SDK:
                response_data = _send_via_sdk(model_str, request_payload, config, timeout=request_timeout)
            elif mode == GatewayMode.SLAUTH_SERVER:
                response_data = _send_via_slauth_server(model_str, request_payload, config, timeout=request_timeout)
            else:
                raise ValueError(f"Unsupported gateway mode for OpenAI route: {mode}")

            if last_error is not None:
                logger.info("Successfully fell back to gateway mode: %s", mode)
            if return_raw_results:
                return response_data
            return _parse_response_data(response_data, stop=stop)

        except Exception as e:  # noqa: BLE001 - we want to cascade
            last_error = e
            if is_auto and mode != modes_to_try[-1]:
                next_mode = modes_to_try[modes_to_try.index(mode) + 1]
                logger.warning("Gateway mode '%s' failed: %s. Trying '%s'...", mode, e, next_mode)
                continue
            raise


async def generate_text_streaming(
        prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
        model: AIGatewayOpenAIModels = AIGatewayOpenAIModels.GPT_55,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        stop: List[str] = None,
        system: str = None,
        reasoning_effort: str = None,
        user_id: str = None,
        cloud_id: str = None,
        use_case_id: str = None,
        base_url: str = None,
        slauth_server_url: str = None,
        timeout: Union[float, Tuple[float, float]] = None,
        connect_timeout: float = None,
        response_timeout: float = None,
        verbose: bool = False,
        gateway_mode: str = "auto",
        proximity_port: int = DEFAULT_PROXIMITY_PORT,
        groups: str = None,
        **kwargs,
) -> AsyncIterator[str]:
    """Stream text from an OpenAI model via the AI Gateway (SSE).

    Uses the ``direct`` mode SSE stream against the OpenAI chat-completions
    route. Other modes fall back to a single non-streaming call.
    """
    if "api_key" in kwargs:
        kwargs.pop("api_key")
    messages = _get_messages(prompt_or_messages)

    config = _resolve_config(
        user_id=user_id,
        cloud_id=cloud_id,
        use_case_id=use_case_id,
        base_url=base_url,
        slauth_server_url=slauth_server_url,
        groups=groups,
    )

    resolved_mode, is_auto, _ = _resolve_mode_order(gateway_mode, config, proximity_port)

    model_str = f"{model}"

    timeout_value = _resolve_llm_timeout(
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout,
    )
    request_timeout = timeout_value if isinstance(timeout_value, (int, float)) and timeout_value else 300

    # Only the direct mode supports SSE here; otherwise do a single call.
    if resolved_mode == GatewayMode.DIRECT:
        request_payload = _build_request_payload(
            messages=messages,
            model_str=model_str,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system=system,
            stream=True,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
        env = "prod" if "prod" in config["base_url"] else "staging"
        token = await asyncio.to_thread(get_direct_slauth_token, env=env, groups=config.get("groups"))
        headers = build_direct_headers(
            token=token,
            user_id=config["user_id"],
            cloud_id=config["cloud_id"],
            use_case_id=config["use_case_id"],
        )
        url = f"{config['base_url']}{OPENAI_CHAT_COMPLETIONS_PATH}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(request_timeout, connect=10)) as client:
            async with client.stream("POST", url, json=request_payload, headers=headers) as response:
                if not (200 <= response.status_code < 300):
                    error_text = (await response.aread()).decode("utf-8", errors="replace")
                    raise Exception(
                        f"Direct streaming: AI Gateway returned status {response.status_code}: {error_text}"
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    chunk = delta.get("content")
                    if chunk:
                        yield chunk
        return

    # Non-streaming fallback for sdk / slauth_server.
    result = await asyncio.to_thread(
        generate_text,
        prompt_or_messages=prompt_or_messages,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=stop,
        system=system,
        reasoning_effort=reasoning_effort,
        user_id=user_id,
        cloud_id=cloud_id,
        use_case_id=use_case_id,
        base_url=base_url,
        slauth_server_url=slauth_server_url,
        timeout=timeout,
        gateway_mode=str(resolved_mode),
        proximity_port=proximity_port,
        groups=groups,
        **kwargs,
    )
    yield result


async def generate_text_async(
        prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
        model: AIGatewayOpenAIModels = AIGatewayOpenAIModels.GPT_55,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        stop: List[str] = None,
        system: str = None,
        reasoning_effort: str = None,
        user_id: str = None,
        cloud_id: str = None,
        use_case_id: str = None,
        base_url: str = None,
        slauth_server_url: str = None,
        timeout: Union[float, Tuple[float, float]] = None,
        connect_timeout: float = None,
        response_timeout: float = None,
        return_raw_results: bool = False,
        verbose: bool = False,
        gateway_mode: str = "auto",
        proximity_port: int = DEFAULT_PROXIMITY_PORT,
        groups: str = None,
        **kwargs,
) -> Union[str, Dict]:
    """Async wrapper around the sync ``generate_text`` (via a worker thread)."""
    return await asyncio.to_thread(
        generate_text,
        prompt_or_messages=prompt_or_messages,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=stop,
        system=system,
        reasoning_effort=reasoning_effort,
        user_id=user_id,
        cloud_id=cloud_id,
        use_case_id=use_case_id,
        base_url=base_url,
        slauth_server_url=slauth_server_url,
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout,
        return_raw_results=return_raw_results,
        verbose=verbose,
        gateway_mode=gateway_mode,
        proximity_port=proximity_port,
        groups=groups,
        **kwargs,
    )


if __name__ == '__main__':
    _r = generate_text(
        "Hello! Reply with exactly: PONG",
        model=AIGatewayOpenAIModels.GPT_55,
        max_new_tokens=2000,
        user_id=environ.get("USER"),
        verbose=True,
    )
    hprint_message({"response": _r}, title="AI Gateway OpenAI")
