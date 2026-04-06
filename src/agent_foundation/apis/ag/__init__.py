"""
AI Gateway API clients for various LLM providers.

This package provides API clients that route requests through Atlassian's AI Gateway,
which offers centralized authentication, rate limiting, monitoring, and cost tracking.

Available modules:
- ai_gateway_claude_llm: Claude models via AI Gateway with Bedrock backend
- gateway_mode: Gateway access mode detection, health checks, and fallback
- stream_parsers: SSE and Bedrock event-stream parsing utilities
"""

from .ai_gateway_claude_llm import (
    AIGatewayClaudeModels,
    generate_text,
    generate_text_async,
    generate_text_streaming,
    DEFAULT_MAX_TOKENS,
    ENV_NAME_AI_GATEWAY_USER_ID,
    ENV_NAME_AI_GATEWAY_CLOUD_ID,
    ENV_NAME_AI_GATEWAY_USE_CASE_ID,
    ENV_NAME_AI_GATEWAY_BASE_URL,
    ENV_NAME_SLAUTH_SERVER_URL,
)

from .gateway_mode import (
    GatewayMode,
    check_direct_available,
    check_proximity_available,
    check_slauth_server_available,
    detect_available_mode,
)

__all__ = [
    'AIGatewayClaudeModels',
    'generate_text',
    'generate_text_async',
    'generate_text_streaming',
    'DEFAULT_MAX_TOKENS',
    'ENV_NAME_AI_GATEWAY_USER_ID',
    'ENV_NAME_AI_GATEWAY_CLOUD_ID',
    'ENV_NAME_AI_GATEWAY_USE_CASE_ID',
    'ENV_NAME_AI_GATEWAY_BASE_URL',
    'ENV_NAME_SLAUTH_SERVER_URL',
    'GatewayMode',
    'check_direct_available',
    'check_proximity_available',
    'check_slauth_server_available',
    'detect_available_mode',
]
