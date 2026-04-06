import asyncio
import json
import logging
import warnings
from enum import StrEnum
from os import environ, path
from typing import AsyncIterator, Dict, List, Sequence, Tuple, Union

import httpx
from ai_gateway.client import AIGatewayClient
from ai_gateway.constants import AIGatewayHeaders
from ai_gateway.models.common import HttpHeaders, HttpMethod
from ai_gateway.models.wrapper import RequestWrapper

from agent_foundation.apis.ag.gateway_mode import (
    DEFAULT_AI_GATEWAY_BASE_URL,
    DEFAULT_CLOUD_ID,
    DEFAULT_PROXIMITY_PORT,
    DEFAULT_SLAUTH_SERVER_URL,
    DEFAULT_USE_CASE_ID,
    GatewayMode,
    bedrock_model_to_anthropic,
    build_direct_headers,
    detect_available_mode,
    get_direct_slauth_token,
)
from agent_foundation.apis.common import _resolve_llm_timeout
from rich_python_utils.console_utils import hprint_message

logger = logging.getLogger(__name__)

# Environment variable names
ENV_NAME_AI_GATEWAY_USER_ID = 'AI_GATEWAY_USER_ID'
ENV_NAME_AI_GATEWAY_CLOUD_ID = 'AI_GATEWAY_CLOUD_ID'
ENV_NAME_AI_GATEWAY_USE_CASE_ID = 'AI_GATEWAY_USE_CASE_ID'
ENV_NAME_AI_GATEWAY_BASE_URL = 'AI_GATEWAY_BASE_URL'
ENV_NAME_SLAUTH_SERVER_URL = 'SLAUTH_SERVER_URL'


class AIGatewayClaudeModels(StrEnum):
    """
    Enumeration for supported Claude models via AI Gateway.
    These correspond to Bedrock model IDs routed through AI Gateway.
    """
    # Claude Sonnet 4.5 (newest)
    CLAUDE_45_SONNET = 'anthropic.claude-sonnet-4-5-20250929-v1:0'

    # Claude Sonnet 4.0
    CLAUDE_40_SONNET = 'anthropic.claude-sonnet-4-20250514-v1:0'

    # Claude Opus 4.6, 4.1 and 4.0
    CLAUDE_46_OPUS = 'anthropic.claude-opus-4-6-v1'
    CLAUDE_41_OPUS = 'anthropic.claude-opus-4-1-20250805-v1:0'
    CLAUDE_40_OPUS = 'anthropic.claude-opus-4-20250514-v1:0'

    # Claude Sonnet 3.7
    CLAUDE_37_SONNET = 'anthropic.claude-3-7-sonnet-20250219-v1:0'

    # Claude Sonnet 3.5 (v2 and v1)
    CLAUDE_35_SONNET_V2 = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
    CLAUDE_35_SONNET_V1 = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

    # Claude Haiku 4.5 and 3.5
    CLAUDE_45_HAIKU = 'anthropic.claude-haiku-4-5-20251001-v1:0'
    CLAUDE_35_HAIKU = 'anthropic.claude-3-5-haiku-20241022-v1:0'


DEFAULT_MAX_TOKENS = {
    f'{AIGatewayClaudeModels.CLAUDE_45_SONNET}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_40_SONNET}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_46_OPUS}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_41_OPUS}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_40_OPUS}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_37_SONNET}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_35_SONNET_V2}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_35_SONNET_V1}': 4096,
    f'{AIGatewayClaudeModels.CLAUDE_45_HAIKU}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_35_HAIKU}': 4096,
}

# Cascade order for auto mode fallback
_FALLBACK_MODES = [GatewayMode.DIRECT, GatewayMode.PROXIMITY, GatewayMode.SLAUTH_SERVER]


def _create_ai_gateway_client(
    base_url: str = None,
    user_id: str = None,
    cloud_id: str = None,
    use_case_id: str = None,
    slauth_server_url: str = None,
) -> AIGatewayClient:
    """
    Create an AI Gateway client with SLAUTH authentication.

    Args:
        base_url: AI Gateway base URL. If None, reads from ENV_NAME_AI_GATEWAY_BASE_URL or uses default.
        user_id: User ID for tracking. If None, reads from ENV_NAME_AI_GATEWAY_USER_ID, then falls back to $USER.
        cloud_id: Cloud ID. If None, reads from ENV_NAME_AI_GATEWAY_CLOUD_ID or uses default.
        use_case_id: Use case ID. If None, reads from ENV_NAME_AI_GATEWAY_USE_CASE_ID or uses default.
        slauth_server_url: SLAUTH server URL. If None, reads from ENV_NAME_SLAUTH_SERVER_URL or uses default.

    Returns:
        Configured AIGatewayClient instance.

    Raises:
        Exception: If user_id is not provided and not found in environment.
    """
    base_url = base_url or environ.get(ENV_NAME_AI_GATEWAY_BASE_URL, DEFAULT_AI_GATEWAY_BASE_URL)
    cloud_id = cloud_id or environ.get(ENV_NAME_AI_GATEWAY_CLOUD_ID, DEFAULT_CLOUD_ID)
    use_case_id = use_case_id or environ.get(ENV_NAME_AI_GATEWAY_USE_CASE_ID, DEFAULT_USE_CASE_ID)
    slauth_server_url = slauth_server_url or environ.get(ENV_NAME_SLAUTH_SERVER_URL, DEFAULT_SLAUTH_SERVER_URL)

    user_id = user_id or environ.get(ENV_NAME_AI_GATEWAY_USER_ID) or environ.get('USER')
    if not user_id:
        raise ValueError(
            f"user_id is required. Set it via parameter, {ENV_NAME_AI_GATEWAY_USER_ID} environment variable, or ensure $USER is set."
        )

    default_headers = HttpHeaders({
        AIGatewayHeaders.USER_ID: user_id,
        AIGatewayHeaders.CLOUD_ID: cloud_id,
        AIGatewayHeaders.USE_CASE_ID: use_case_id
    })

    try:
        from ai_gateway.client.common.filters import SlauthServerAuthFilter

        slauth_filter = SlauthServerAuthFilter(
            sl_auth_server_url=slauth_server_url,
            groups={"atlassian-all"}
        )

        if not slauth_filter:
            raise Exception("Failed to create SLAUTH authentication filter")

    except Exception as e:
        raise Exception(
            f"Failed to create SLAUTH filter: {e}. "
            f"Ensure atlas slauth plugin is installed: atlas plugin install -n slauth. "
            f"Ensure SLAUTH server is running: atlas slauth server --port 5000"
        )

    return AIGatewayClient.sync(
        base_url=base_url,
        default_headers=default_headers,
        filters=[slauth_filter]
    )


def _get_messages(prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]]):
    """
    Convert various input formats into Anthropic's message format.
    This matches the implementation in claude_llm.py for consistency.
    """
    if isinstance(prompt_or_messages, str):
        if path.exists(prompt_or_messages):
            with open(prompt_or_messages, 'r', encoding='utf-8') as f:
                prompt_or_messages = f.read()
        return [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': prompt_or_messages}]
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
                            'content': [{'type': 'text', 'text': prompt_or_messages[i]}]
                        },
                        {
                            'role': 'assistant',
                            'content': [{'type': 'text', 'text': prompt_or_messages[i + 1]}]
                        }
                    )
                )
            messages.append(
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': prompt_or_messages[-1]}]
                }
            )
            return messages
        elif isinstance(prompt_or_messages[0], Dict):
            # Normalize: ensure content is in array-of-blocks format
            # (AI Gateway Bedrock endpoint requires this, especially for streaming)
            normalized = []
            for msg in prompt_or_messages:
                msg = dict(msg)  # shallow copy
                if isinstance(msg.get('content'), str):
                    msg['content'] = [{'type': 'text', 'text': msg['content']}]
                normalized.append(msg)
            return normalized
    raise ValueError(
        "'prompt_or_messages' must be one of str, Dict, or a sequence of strs or Dicts"
    )


def _resolve_config(
    user_id: str = None,
    cloud_id: str = None,
    use_case_id: str = None,
    base_url: str = None,
    slauth_server_url: str = None,
) -> dict:
    """Resolve configuration from parameters, env vars, and defaults."""
    return {
        "base_url": base_url or environ.get(ENV_NAME_AI_GATEWAY_BASE_URL, DEFAULT_AI_GATEWAY_BASE_URL),
        "cloud_id": cloud_id or environ.get(ENV_NAME_AI_GATEWAY_CLOUD_ID, DEFAULT_CLOUD_ID),
        "use_case_id": use_case_id or environ.get(ENV_NAME_AI_GATEWAY_USE_CASE_ID, DEFAULT_USE_CASE_ID),
        "slauth_server_url": slauth_server_url or environ.get(ENV_NAME_SLAUTH_SERVER_URL, DEFAULT_SLAUTH_SERVER_URL),
        "user_id": user_id or environ.get(ENV_NAME_AI_GATEWAY_USER_ID) or environ.get('USER', ''),
    }


def _send_via_direct(model_str: str, request_payload: dict, config: dict, timeout: float = 120) -> dict:
    """Send request directly to AI Gateway using atlas CLI for SLAuth token.

    Args:
        model_str: Bedrock model ID string.
        request_payload: Anthropic/Bedrock request body.
        config: Resolved configuration dict.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.
    """
    env = "prod" if "prod" in config["base_url"] else "staging"
    token = get_direct_slauth_token(env=env)
    headers = build_direct_headers(
        token=token,
        user_id=config["user_id"],
        cloud_id=config["cloud_id"],
        use_case_id=config["use_case_id"],
    )
    url = f"{config['base_url']}/v1/bedrock/model/{model_str}/invoke"

    resp = httpx.post(url, json=request_payload, headers=headers, timeout=timeout)

    if not (200 <= resp.status_code < 300):
        raise Exception(f"Direct mode: AI Gateway returned status {resp.status_code}: {resp.text}")

    return resp.json()


def _send_via_proximity(model_str: str, request_payload: dict, port: int = DEFAULT_PROXIMITY_PORT, timeout: float = 120) -> dict:
    """Send request via the proximity AI gateway proxy.

    The proximity proxy expects Anthropic-native format on the Vertex path.

    Args:
        model_str: Bedrock model ID string (will be converted to Anthropic-native).
        request_payload: Anthropic/Bedrock request body.
        port: Proximity proxy port.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.
    """
    anthropic_model = bedrock_model_to_anthropic(model_str)
    url = f"http://localhost:{port}/vertex/claude/v1/messages?beta=true"

    # Build body for proximity. The proxy's template adds anthropic_version itself,
    # so we must NOT include it — otherwise the merge fails with "duplicate keys".
    body = {
        "model": anthropic_model,
        "messages": request_payload["messages"],
        "max_tokens": request_payload.get("max_tokens", 8192),
    }
    if "temperature" in request_payload:
        body["temperature"] = request_payload["temperature"]
    if "system" in request_payload:
        body["system"] = request_payload["system"]
    # Forward any extra params (top_k, metadata, etc.) but skip anthropic_version
    for key in request_payload:
        if key not in ("anthropic_version", "messages", "max_tokens", "temperature", "system"):
            body[key] = request_payload[key]

    resp = httpx.post(url, json=body, timeout=timeout)

    if not (200 <= resp.status_code < 300):
        raise Exception(f"Proximity mode: proxy returned status {resp.status_code}: {resp.text}")

    return resp.json()


def _send_via_slauth_server(model_str: str, request_payload: dict, config: dict, timeout: float = 120) -> dict:
    """Send request using the AI Gateway SDK with SlauthServerAuthFilter.

    This is the original/existing approach.

    Args:
        model_str: Bedrock model ID string.
        request_payload: Anthropic/Bedrock request body.
        config: Resolved configuration dict.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.
    """
    client = _create_ai_gateway_client(
        base_url=config["base_url"],
        user_id=config["user_id"],
        cloud_id=config["cloud_id"],
        use_case_id=config["use_case_id"],
        slauth_server_url=config["slauth_server_url"],
    )

    request = RequestWrapper(
        body=json.dumps(request_payload).encode('utf-8'),
        headers=HttpHeaders({'Content-Type': 'application/json'})
    )

    response = client.raw.http(
        method=HttpMethod.POST,
        uri=f'/v1/bedrock/model/{model_str}/invoke',
        request=request
    )

    if not (200 <= response.http_status.code < 300):
        error_msg = f"SLAuth server mode: AI Gateway returned status {response.http_status.code}"
        # SDK uses raw_body for error responses (body is None on non-2xx)
        raw_body = getattr(response, 'raw_body', None) or response.body
        if raw_body:
            try:
                raw_str = raw_body if isinstance(raw_body, str) else raw_body.decode('utf-8')
                error_data = json.loads(raw_str)
                if 'message' in error_data:
                    error_msg += f"\nError: {error_data['message']}"
                elif 'upstream' in error_data and 'content' in error_data['upstream']:
                    error_msg += f"\nUpstream error: {json.dumps(error_data['upstream']['content'], indent=2)}"
                else:
                    error_msg += f"\nResponse: {raw_str}"
            except Exception:
                error_msg += f"\nResponse: {raw_body}"
        raise Exception(error_msg)

    return json.loads(response.body.decode('utf-8'))


def _parse_response_data(response_data: dict, stop: List[str] = None) -> str:
    """Parse the Anthropic/Bedrock response and extract generated text.

    Args:
        response_data: Parsed JSON response.
        stop: Optional stop sequences to truncate at.

    Returns:
        Generated text string.
    """
    if 'content' in response_data:
        content_blocks = response_data['content']
        if isinstance(content_blocks, list) and len(content_blocks) > 0:
            first_block = content_blocks[0]
            if isinstance(first_block, dict) and 'text' in first_block:
                generated_text = first_block['text'].strip()
            else:
                generated_text = str(first_block)
        else:
            generated_text = str(content_blocks)
    else:
        raise Exception(f"Unexpected response format: {response_data}")

    if stop:
        for stop_sequence in stop:
            stop_idx = generated_text.find(stop_sequence)
            if stop_idx != -1:
                generated_text = generated_text[:stop_idx]

    return generated_text.strip()


def _build_request_payload(
    messages: list,
    model_str: str,
    max_new_tokens: int = None,
    temperature: float = 0.7,
    system: str = None,
    **kwargs,
) -> dict:
    """Build the Anthropic/Bedrock request payload.

    Args:
        messages: List of message dicts in Anthropic format.
        model_str: Bedrock model ID string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        system: Optional system prompt.
        **kwargs: Additional parameters to include in payload.

    Returns:
        Request payload dict.
    """
    if not max_new_tokens:
        max_new_tokens = DEFAULT_MAX_TOKENS.get(model_str, 8192)

    payload = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': max_new_tokens,
        'messages': messages,
        'temperature': temperature,
    }

    if system:
        payload['system'] = system

    payload.update(kwargs)
    return payload


def generate_text(
        prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
        model: AIGatewayClaudeModels = AIGatewayClaudeModels.CLAUDE_45_SONNET,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        stop: List[str] = None,
        system: str = None,
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
        **kwargs
) -> Union[str, List[str], Dict]:
    """
    Generate text using Claude via AI Gateway.

    Supports three gateway access modes:
    - "direct": Shell out to atlas CLI for SLAuth token, send via httpx (no local server needed)
    - "proximity": Forward to localhost proximity proxy (needs `proximity ai-gateway` running)
    - "slauth_server": Use AI Gateway SDK with SlauthServerAuthFilter (needs `atlas slauth server`)
    - "auto": Auto-detect first available mode, with runtime fallback on failure

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The Claude model to use for generating text (Bedrock model ID).
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: Controls the randomness of the output.
        stop: List of strings that will stop generation when encountered.
        system: System prompt to set context for the conversation.
        user_id: AI Gateway user ID for tracking.
        cloud_id: AI Gateway cloud ID.
        use_case_id: AI Gateway use case ID.
        base_url: AI Gateway base URL.
        slauth_server_url: SLAUTH server URL.
        timeout: Request timeout in seconds.
        connect_timeout: Maximum time to wait for connection.
        response_timeout: Maximum time to wait for response.
        return_raw_results: Whether to return the raw results from the API.
        verbose: True to print out parameter values.
        gateway_mode: Gateway access mode ("direct", "proximity", "slauth_server", "auto").
        proximity_port: Port for proximity proxy (default 29576).
        **kwargs: Additional parameters to pass to the Anthropic API.

    Returns:
        Generated text, or raw API response if return_raw_results=True.
    """
    if "api_key" in kwargs:
        kwargs.pop('api_key')
    messages = _get_messages(prompt_or_messages)

    # Resolve configuration
    config = _resolve_config(
        user_id=user_id,
        cloud_id=cloud_id,
        use_case_id=use_case_id,
        base_url=base_url,
        slauth_server_url=slauth_server_url,
    )

    # Resolve gateway mode
    resolved_mode = GatewayMode(gateway_mode)
    is_auto = resolved_mode == GatewayMode.AUTO

    if is_auto:
        try:
            resolved_mode = detect_available_mode(
                proximity_port=proximity_port,
                slauth_server_url=config["slauth_server_url"],
            )
        except RuntimeError:
            # If detection fails, we'll try all modes during execution
            resolved_mode = GatewayMode.DIRECT

    # Build request payload
    model_str = f'{model}'
    request_payload = _build_request_payload(
        messages=messages,
        model_str=model_str,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system=system,
        **kwargs,
    )

    # Handle timeout
    timeout_value = _resolve_llm_timeout(
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout
    )
    request_timeout = timeout_value if isinstance(timeout_value, (int, float)) and timeout_value else 120

    if True:
        hprint_message(
            {
                'model': model_str,
                'gateway_mode': str(resolved_mode),
                'base_url': config["base_url"],
                'max_tokens': max_new_tokens,
                'temperature': temperature,
                'system': system,
                'timeout': timeout_value,
                'return_raw_results': return_raw_results,
                **kwargs
            },
            title='AI Gateway Claude API Parameters'
        )

    # Build mode execution order
    if is_auto:
        # Try resolved mode first, then fallback through remaining modes
        modes_to_try = [resolved_mode] + [m for m in _FALLBACK_MODES if m != resolved_mode]
    else:
        modes_to_try = [resolved_mode]

    # Execute with fallback
    last_error = None
    for mode in modes_to_try:
        try:
            if mode == GatewayMode.DIRECT:
                response_data = _send_via_direct(model_str, request_payload, config, timeout=request_timeout)
            elif mode == GatewayMode.PROXIMITY:
                response_data = _send_via_proximity(model_str, request_payload, port=proximity_port, timeout=request_timeout)
            elif mode == GatewayMode.SLAUTH_SERVER:
                response_data = _send_via_slauth_server(model_str, request_payload, config, timeout=request_timeout)
            else:
                raise ValueError(f"Unknown gateway mode: {mode}")

            # Success — log if we fell back
            if last_error is not None:
                logger.info(f"Successfully fell back to gateway mode: {mode}")

            if return_raw_results:
                return response_data

            return _parse_response_data(response_data, stop=stop)

        except Exception as e:
            last_error = e
            if is_auto and mode != modes_to_try[-1]:
                next_mode = modes_to_try[modes_to_try.index(mode) + 1]
                warnings.warn(
                    f"Gateway mode '{mode}' failed: {e}. Falling back to '{next_mode}'...",
                    stacklevel=2,
                )
                logger.warning(f"Gateway mode '{mode}' failed: {e}. Trying '{next_mode}'...")
                continue
            raise


async def _send_via_proximity_streaming(
    model_str: str,
    request_payload: dict,
    port: int = DEFAULT_PROXIMITY_PORT,
    timeout: float = 300,
) -> AsyncIterator[str]:
    """Stream text deltas via the proximity AI gateway proxy (SSE format).

    Args:
        model_str: Bedrock model ID string (converted to Anthropic-native).
        request_payload: Anthropic/Bedrock request body.
        port: Proximity proxy port.
        timeout: Read timeout in seconds.

    Yields:
        Text delta strings as they arrive.
    """
    from agent_foundation.apis.ag.stream_parsers import extract_text_delta

    anthropic_model = bedrock_model_to_anthropic(model_str)
    url = f"http://localhost:{port}/vertex/claude/v1/messages?beta=true"

    # Build body for proximity (same as _send_via_proximity but with stream=true)
    body = {
        "model": anthropic_model,
        "messages": request_payload["messages"],
        "max_tokens": request_payload.get("max_tokens", 8192),
        "stream": True,
    }
    if "temperature" in request_payload:
        body["temperature"] = request_payload["temperature"]
    if "system" in request_payload:
        body["system"] = request_payload["system"]
    for key in request_payload:
        if key not in ("anthropic_version", "messages", "max_tokens", "temperature", "system"):
            body[key] = request_payload[key]

    # Use aiohttp instead of httpx for proximity streaming because the
    # proximity proxy sends duplicate Transfer-Encoding headers that
    # httpx's strict HTTP parser rejects.
    import aiohttp

    conn_timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=timeout)
    async with aiohttp.ClientSession(timeout=conn_timeout) as session:
        async with session.post(url, json=body) as response:
            if not (200 <= response.status < 300):
                body_text = await response.text()
                raise Exception(
                    f"Proximity streaming: proxy returned status {response.status}: "
                    f"{body_text}"
                )
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                text = extract_text_delta(event_data)
                if text is not None:
                    yield text


async def _send_via_direct_streaming(
    model_str: str,
    request_payload: dict,
    config: dict,
    timeout: float = 300,
) -> AsyncIterator[str]:
    """Stream text deltas directly via AI Gateway (SSE format).

    Uses the /invoke-with-response-stream endpoint. The AI Gateway returns
    standard SSE (text/event-stream), NOT Bedrock binary event-stream.

    Args:
        model_str: Bedrock model ID string.
        request_payload: Anthropic/Bedrock request body.
        config: Resolved configuration dict.
        timeout: Read timeout in seconds.

    Yields:
        Text delta strings as they arrive.
    """
    from agent_foundation.apis.ag.stream_parsers import extract_text_delta

    env = "prod" if "prod" in config["base_url"] else "staging"
    token = await asyncio.to_thread(get_direct_slauth_token, env=env)
    headers = build_direct_headers(
        token=token,
        user_id=config["user_id"],
        cloud_id=config["cloud_id"],
        use_case_id=config["use_case_id"],
    )
    url = f"{config['base_url']}/v1/bedrock/model/{model_str}/invoke-with-response-stream"

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10)) as client:
        async with client.stream("POST", url, json=request_payload, headers=headers) as response:
            if not (200 <= response.status_code < 300):
                body_text = await response.aread()
                error_text = body_text.decode('utf-8', errors='replace')
                if response.status_code == 404:
                    raise Exception(
                        f"Direct streaming: endpoint not found (404). "
                        f"AI Gateway may not support /invoke-with-response-stream. "
                        f"Response: {error_text}"
                    )
                raise Exception(
                    f"Direct streaming: AI Gateway returned status {response.status_code}: {error_text}"
                )
            # AI Gateway returns standard SSE (text/event-stream)
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                text = extract_text_delta(event_data)
                if text is not None:
                    yield text


async def generate_text_streaming(
        prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
        model: AIGatewayClaudeModels = AIGatewayClaudeModels.CLAUDE_45_SONNET,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        stop: List[str] = None,
        system: str = None,
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
        **kwargs
) -> AsyncIterator[str]:
    """Stream text from Claude via AI Gateway, yielding chunks as they arrive.

    Supports streaming via proximity (SSE) and direct (Bedrock event-stream) modes.
    Falls back to non-streaming for slauth_server mode or if streaming endpoints
    are unavailable.

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The Claude model to use (Bedrock model ID).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Controls randomness of output.
        stop: Stop sequences (applied post-hoc for streaming).
        system: System prompt.
        user_id: AI Gateway user ID.
        cloud_id: AI Gateway cloud ID.
        use_case_id: AI Gateway use case ID.
        base_url: AI Gateway base URL.
        slauth_server_url: SLAUTH server URL.
        timeout: Request timeout in seconds.
        connect_timeout: Maximum time to wait for connection.
        response_timeout: Maximum time to wait for response.
        verbose: Print parameter values.
        gateway_mode: Gateway access mode.
        proximity_port: Port for proximity proxy.
        **kwargs: Additional parameters.

    Yields:
        Text chunks as they arrive from the model.
    """
    if "api_key" in kwargs:
        kwargs.pop('api_key')
    messages = _get_messages(prompt_or_messages)

    config = _resolve_config(
        user_id=user_id,
        cloud_id=cloud_id,
        use_case_id=use_case_id,
        base_url=base_url,
        slauth_server_url=slauth_server_url,
    )

    resolved_mode = GatewayMode(gateway_mode)
    is_auto = resolved_mode == GatewayMode.AUTO

    if is_auto:
        try:
            resolved_mode = detect_available_mode(
                proximity_port=proximity_port,
                slauth_server_url=config["slauth_server_url"],
            )
        except RuntimeError:
            resolved_mode = GatewayMode.DIRECT

    model_str = f'{model}'
    request_payload = _build_request_payload(
        messages=messages,
        model_str=model_str,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system=system,
        **kwargs,
    )

    # Resolve timeout
    timeout_value = _resolve_llm_timeout(
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout
    )
    request_timeout = timeout_value if isinstance(timeout_value, (int, float)) and timeout_value else 300

    if verbose:
        hprint_message(
            {
                'model': model_str,
                'gateway_mode': str(resolved_mode),
                'streaming': True,
                'timeout': request_timeout,
            },
            title='AI Gateway Claude Streaming Parameters'
        )

    # Build mode execution order (streaming-capable modes first)
    if is_auto:
        modes_to_try = [resolved_mode] + [m for m in _FALLBACK_MODES if m != resolved_mode]
    else:
        modes_to_try = [resolved_mode]

    last_error = None
    for mode in modes_to_try:
        try:
            if mode == GatewayMode.PROXIMITY:
                async for chunk in _send_via_proximity_streaming(
                    model_str, request_payload, port=proximity_port, timeout=request_timeout
                ):
                    yield chunk
                return

            elif mode == GatewayMode.DIRECT:
                async for chunk in _send_via_direct_streaming(
                    model_str, request_payload, config, timeout=request_timeout
                ):
                    yield chunk
                return

            elif mode == GatewayMode.SLAUTH_SERVER:
                # slauth_server doesn't support streaming — fall back to non-streaming
                logger.info("slauth_server mode: falling back to non-streaming")
                result = await asyncio.to_thread(
                    generate_text,
                    prompt_or_messages=prompt_or_messages,
                    model=model,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=stop,
                    system=system,
                    user_id=user_id,
                    cloud_id=cloud_id,
                    use_case_id=use_case_id,
                    base_url=base_url,
                    slauth_server_url=slauth_server_url,
                    timeout=timeout,
                    gateway_mode="slauth_server",
                    proximity_port=proximity_port,
                    **kwargs,
                )
                yield result
                return

            else:
                raise ValueError(f"Unknown gateway mode: {mode}")

        except Exception as e:
            last_error = e
            if is_auto and mode != modes_to_try[-1]:
                next_mode = modes_to_try[modes_to_try.index(mode) + 1]
                logger.warning(
                    "Streaming mode '%s' failed: %s. Trying '%s'...", mode, e, next_mode
                )
                continue
            raise

    if last_error:
        raise last_error


async def generate_text_async(
        prompt_or_messages: Union[str, Dict, Sequence[str], Sequence[Dict]],
        model: AIGatewayClaudeModels = AIGatewayClaudeModels.CLAUDE_45_SONNET,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        stop: List[str] = None,
        system: str = None,
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
        **kwargs
) -> Union[str, Dict]:
    """Async generate text using Claude via AI Gateway.

    Wraps the sync generate_text() via asyncio.to_thread() for efficiency
    when streaming is not needed.

    Args:
        Same as generate_text().

    Returns:
        Generated text, or raw API response if return_raw_results=True.
    """
    return await asyncio.to_thread(
        generate_text,
        prompt_or_messages=prompt_or_messages,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=stop,
        system=system,
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
        **kwargs,
    )


if __name__ == '__main__':
    from rich_python_utils.common_utils.arg_utils.arg_parse import get_parsed_args

    args = get_parsed_args(
        default_prompt="Hello! What's the capital of France?",
        default_model='anthropic.claude-sonnet-4-5-20250929-v1:0',
        default_max_new_tokens=1024,
        default_stop='[]',
        default_temperature=0.7,
        default_return_raw_results=False,
        default_user_id=None,
        default_system=None
    )

    _prompt_or_messages = args.prompt
    _model = args.model
    _max_new_tokens = args.max_new_tokens
    _stop = args.stop
    _temperature = args.temperature
    _return_raw_results = args.return_raw_results
    _user_id = args.user_id
    _system = args.system

    call_kwargs = {
        'prompt_or_messages': _prompt_or_messages,
        'model': _model,
        'max_new_tokens': _max_new_tokens,
        'stop': _stop,
        'temperature': _temperature,
        'return_raw_results': _return_raw_results,
        'verbose': True
    }

    if _user_id:
        call_kwargs['user_id'] = _user_id

    if _system:
        call_kwargs['system'] = _system

    _generated_text = generate_text(**call_kwargs)

    hprint_message({'response': _generated_text}, title=f'AI Gateway - {_model}')
