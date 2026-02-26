"""
AI Gateway API clients for various LLM providers.

This package provides API clients that route requests through Atlassian's AI Gateway,
which offers centralized authentication, rate limiting, monitoring, and cost tracking.

Available modules:
- ai_gateway_claude_llm: Claude models via AI Gateway with Bedrock backend
"""

from .ai_gateway_claude_llm import (
    AIGatewayClaudeModels,
    generate_text,
    DEFAULT_MAX_TOKENS,
    ENV_NAME_AI_GATEWAY_USER_ID,
    ENV_NAME_AI_GATEWAY_CLOUD_ID,
    ENV_NAME_AI_GATEWAY_USE_CASE_ID,
    ENV_NAME_AI_GATEWAY_BASE_URL,
    ENV_NAME_SLAUTH_SERVER_URL,
)

__all__ = [
    'AIGatewayClaudeModels',
    'generate_text',
    'DEFAULT_MAX_TOKENS',
    'ENV_NAME_AI_GATEWAY_USER_ID',
    'ENV_NAME_AI_GATEWAY_CLOUD_ID',
    'ENV_NAME_AI_GATEWAY_USE_CASE_ID',
    'ENV_NAME_AI_GATEWAY_BASE_URL',
    'ENV_NAME_SLAUTH_SERVER_URL',
]
