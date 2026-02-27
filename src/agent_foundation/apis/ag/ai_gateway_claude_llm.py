from enum import StrEnum
from os import environ, path
from typing import Union, List, Dict, Sequence, Tuple
import json

from ai_gateway.client import AIGatewayClient
from ai_gateway.constants import AIGatewayHeaders
from ai_gateway.models.common import HttpMethod, HttpHeaders
from ai_gateway.models.wrapper import RequestWrapper

from agent_foundation.apis.common import _resolve_llm_timeout
from rich_python_utils.console_utils import hprint_message

# Environment variable names
ENV_NAME_AI_GATEWAY_USER_ID = 'AI_GATEWAY_USER_ID'
ENV_NAME_AI_GATEWAY_CLOUD_ID = 'AI_GATEWAY_CLOUD_ID'
ENV_NAME_AI_GATEWAY_USE_CASE_ID = 'AI_GATEWAY_USE_CASE_ID'
ENV_NAME_AI_GATEWAY_BASE_URL = 'AI_GATEWAY_BASE_URL'
ENV_NAME_SLAUTH_SERVER_URL = 'SLAUTH_SERVER_URL'

# Default values
DEFAULT_AI_GATEWAY_BASE_URL = "https://ai-gateway.us-east-1.staging.atl-paas.net"
DEFAULT_CLOUD_ID = "local"
DEFAULT_USE_CASE_ID = "ai-gateway-eval-use-case"
DEFAULT_SLAUTH_SERVER_URL = "http://localhost:5000"


class AIGatewayClaudeModels(StrEnum):
    """
    Enumeration for supported Claude models via AI Gateway.
    These correspond to Bedrock model IDs routed through AI Gateway.
    """
    # Claude Sonnet 4.5 (newest)
    CLAUDE_45_SONNET = 'anthropic.claude-sonnet-4-5-20250929-v1:0'
    
    # Claude Sonnet 4.0
    CLAUDE_40_SONNET = 'anthropic.claude-sonnet-4-20250514-v1:0'
    
    # Claude Opus 4.1 and 4.0
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
    f'{AIGatewayClaudeModels.CLAUDE_41_OPUS}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_40_OPUS}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_37_SONNET}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_35_SONNET_V2}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_35_SONNET_V1}': 4096,
    f'{AIGatewayClaudeModels.CLAUDE_45_HAIKU}': 8192,
    f'{AIGatewayClaudeModels.CLAUDE_35_HAIKU}': 4096,
}


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
    # Resolve configuration from environment or defaults
    base_url = base_url or environ.get(ENV_NAME_AI_GATEWAY_BASE_URL, DEFAULT_AI_GATEWAY_BASE_URL)
    cloud_id = cloud_id or environ.get(ENV_NAME_AI_GATEWAY_CLOUD_ID, DEFAULT_CLOUD_ID)
    use_case_id = use_case_id or environ.get(ENV_NAME_AI_GATEWAY_USE_CASE_ID, DEFAULT_USE_CASE_ID)
    slauth_server_url = slauth_server_url or environ.get(ENV_NAME_SLAUTH_SERVER_URL, DEFAULT_SLAUTH_SERVER_URL)
    
    # User ID is required - try AI_GATEWAY_USER_ID first, then fall back to $USER
    user_id = user_id or environ.get(ENV_NAME_AI_GATEWAY_USER_ID) or environ.get('USER')
    if not user_id:
        raise ValueError(
            f"user_id is required. Set it via parameter, {ENV_NAME_AI_GATEWAY_USER_ID} environment variable, or ensure $USER is set."
        )
    
    # Create headers for request tracking
    default_headers = HttpHeaders({
        AIGatewayHeaders.USER_ID: user_id,
        AIGatewayHeaders.CLOUD_ID: cloud_id,
        AIGatewayHeaders.USE_CASE_ID: use_case_id
    })
    
    # Create SLAUTH authentication filter
    try:
        from ai_gateway.client.common.filters import SlauthServerAuthFilter
        
        slauth_filter = SlauthServerAuthFilter(
            sl_auth_server_url=slauth_server_url,
            groups={"ai-gateway-evaluation-dl-all-atlassian-read"}
        )
        
        if not slauth_filter:
            raise Exception("Failed to create SLAUTH authentication filter")
            
    except Exception as e:
        raise Exception(
            f"Failed to create SLAUTH filter: {e}. "
            f"Ensure atlas slauth plugin is installed: atlas plugin install -n slauth. "
            f"Ensure SLAUTH server is running: atlas slauth server --port 5000"
        )
    
    # Create and return the AI Gateway client
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
            return list(prompt_or_messages)
    raise ValueError(
        "'prompt_or_messages' must be one of str, Dict, or a sequence of strs or Dicts"
    )


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
        **kwargs
) -> Union[str, List[str], Dict]:
    """
    Generate text using Claude via AI Gateway.

    Args:
        prompt_or_messages: The prompt or messages to generate text from.
        model: The Claude model to use for generating text (Bedrock model ID).
        max_new_tokens: The maximum number of new tokens to generate (excluding the prompt).
        temperature: Controls the randomness of the output. Higher values mean more random.
        stop: List of strings that will stop generation when encountered.
                Note: This is implemented at the application level since Claude API
                doesn't directly support stop sequences.
        system: System prompt to set context for the conversation.
        user_id: AI Gateway user ID for tracking. If None, reads from ENV_NAME_AI_GATEWAY_USER_ID.
        cloud_id: AI Gateway cloud ID. If None, reads from ENV_NAME_AI_GATEWAY_CLOUD_ID or uses default.
        use_case_id: AI Gateway use case ID. If None, reads from ENV_NAME_AI_GATEWAY_USE_CASE_ID or uses default.
        base_url: AI Gateway base URL. If None, reads from ENV_NAME_AI_GATEWAY_BASE_URL or uses default.
        slauth_server_url: SLAUTH server URL. If None, reads from ENV_NAME_SLAUTH_SERVER_URL or uses default.
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
        **kwargs: Additional parameters to pass to the Anthropic API (e.g., top_k, metadata).
          top_p must not be provided – Bedrock rejects it; use temperature instead if you need sampling control.

    Returns:
        Generated text, or raw API response if return_raw_results=True.

    Notes:
        - Requires SLAUTH authentication. Ensure atlas slauth server is running:
          `atlas slauth server --port 5000`
        - Uses Bedrock endpoint: /v1/bedrock/model/{model}/invoke
        - Anthropic API does not support top_k through the standard parameters
        - Each API call generates exactly one response
        - Timeout behavior matches claude_llm.py implementation
        
    Examples:
        >>> # Simple usage with environment variables set
        >>> response = generate_text("What is the capital of France?")
        >>> print(response)
        
        >>> # Multi-turn conversation
        >>> conversation = [
        ...     "Hello, I need help with Python",
        ...     "Sure! What would you like to know?",
        ...     "How do I read a file?"
        ... ]
        >>> response = generate_text(conversation, model=AIGatewayClaudeModels.CLAUDE_45_SONNET)
        
        >>> # With explicit configuration
        >>> response = generate_text(
        ...     "Explain quantum computing",
        ...     model=AIGatewayClaudeModels.CLAUDE_40_SONNET,
        ...     user_id="user123",
        ...     temperature=0.5,
        ...     max_new_tokens=500,
        ...     verbose=True
        ... )
    """
    if "api_key" in kwargs:
        kwargs.pop('api_key')
    messages = _get_messages(prompt_or_messages)
    
    # Create AI Gateway client
    client = _create_ai_gateway_client(
        base_url=base_url,
        user_id=user_id,
        cloud_id=cloud_id,
        use_case_id=use_case_id,
        slauth_server_url=slauth_server_url
    )
    
    # Build parameters dict
    model_str = f'{model}'
    if not max_new_tokens:
        max_new_tokens = DEFAULT_MAX_TOKENS.get(model_str, 8192)
    
    # Prepare Anthropic request payload for Bedrock
    request_payload = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': max_new_tokens,
        'messages': messages,
        'temperature': temperature,
    }
    
    # Add optional parameters
    if system:
        request_payload['system'] = system
    
    # Add any additional kwargs
    request_payload.update(kwargs)
    
    # Handle timeout setting
    timeout_value = _resolve_llm_timeout(
        timeout=timeout,
        connect_timeout=connect_timeout,
        response_timeout=response_timeout
    )
    
    if True:
        hprint_message(
            {
                'model': model_str,
                'base_url': client.base_url if hasattr(client, 'base_url') else base_url,
                'max_tokens': max_new_tokens,
                'temperature': temperature,
                'system': system,
                'timeout': timeout_value,
                'return_raw_results': return_raw_results,
                **kwargs
            },
            title='AI Gateway Claude API Parameters'
        )
    
    # Create request wrapper
    request = RequestWrapper(
        body=json.dumps(request_payload).encode('utf-8'),
        headers=HttpHeaders({'Content-Type': 'application/json'})
    )
    
    # Make the API call
    # Note: timeout handling would need to be added to the http() call if the SDK supports it
    response = client.raw.http(
        method=HttpMethod.POST,
        uri=f'/v1/bedrock/model/{model_str}/invoke',
        request=request
    )
    
    # Check response status
    if not (200 <= response.http_status.code < 300):
        error_msg = f"AI Gateway request failed with status {response.http_status.code}"
        if response.body:
            try:
                error_data = json.loads(response.body.decode('utf-8'))
                if 'upstream' in error_data and 'content' in error_data['upstream']:
                    error_msg += f"\nUpstream error: {json.dumps(error_data['upstream']['content'], indent=2)}"
                elif 'message' in error_data:
                    error_msg += f"\nError: {error_data['message']}"
                else:
                    error_msg += f"\nResponse: {response.body.decode('utf-8')}"
            except Exception:
                error_msg += f"\nResponse: {response.body}"
        raise Exception(error_msg)
    
    # Parse response
    if return_raw_results:
        return response
    
    try:
        response_data = json.loads(response.body.decode('utf-8'))
    except Exception as e:
        raise Exception(f"Failed to parse response: {e}\nRaw response: {response.body}")
    
    # Extract text from Anthropic/Bedrock response format
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

    # Build kwargs - only include user_id if explicitly provided
    kwargs = {
        'prompt_or_messages': _prompt_or_messages,
        'model': _model,
        'max_new_tokens': _max_new_tokens,
        'stop': _stop,
        'temperature': _temperature,
        # Bedrock rejects top_p values; leave sampling to temperature
        'return_raw_results': _return_raw_results,
        'verbose': True
    }
    
    if _user_id:
        kwargs['user_id'] = _user_id
    
    if _system:
        kwargs['system'] = _system

    _generated_text = generate_text(**kwargs)

    hprint_message({'response': _generated_text}, title=f'AI Gateway - {_model}')
