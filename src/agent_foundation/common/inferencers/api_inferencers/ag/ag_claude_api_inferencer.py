"""AI Gateway Claude inferencer with streaming support.

Extends StreamingInferencerBase (like PlugboardApiInferencer) so that
streaming capability is built-in via ainfer_streaming() / infer_streaming().
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterable, Optional, Union

from attr import attrib, attrs

from agent_foundation.apis.claude_llm import ClaudeModels
from agent_foundation.apis.ag.ai_gateway_claude_llm import (
    AIGatewayClaudeModels,
    generate_text as ai_gateway_generate_text,
    generate_text_async as ai_gateway_generate_text_async,
    generate_text_streaming as ai_gateway_generate_text_streaming,
)
from agent_foundation.apis.ag.gateway_mode import DEFAULT_PROXIMITY_PORT, GatewayMode
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)

logger = logging.getLogger(__name__)


# Tuple of supported model identifier inputs for quick isinstance checks
SUPPORTED_MODEL_ID_TYPES: Iterable[type] = (
    str,
    ClaudeModels,
    AIGatewayClaudeModels,
)


# Mapping plain Anthropic model identifiers to their AI Gateway equivalents
CLAUDE_TO_AI_GATEWAY_MODEL_MAP = {
    ClaudeModels.CLAUDE_45_SONNET: AIGatewayClaudeModels.CLAUDE_45_SONNET,
    ClaudeModels.CLAUDE_40_SONNET: AIGatewayClaudeModels.CLAUDE_40_SONNET,
    ClaudeModels.CLAUDE_37_SONNET: AIGatewayClaudeModels.CLAUDE_37_SONNET,
    ClaudeModels.CLAUDE_46_OPUS: AIGatewayClaudeModels.CLAUDE_46_OPUS,
    ClaudeModels.CLAUDE_41_OPUS: AIGatewayClaudeModels.CLAUDE_41_OPUS,
    ClaudeModels.CLAUDE_3_OPUS: AIGatewayClaudeModels.CLAUDE_40_OPUS,
    ClaudeModels.CLAUDE_35_SONNET: AIGatewayClaudeModels.CLAUDE_35_SONNET_V2,
    ClaudeModels.CLAUDE_3_HAIKU: AIGatewayClaudeModels.CLAUDE_35_HAIKU,
}


def _resolve_model_id(model_id: Union[str, ClaudeModels, AIGatewayClaudeModels]) -> str:
    """Normalise model identifiers to AI Gateway's Bedrock model strings."""
    if isinstance(model_id, AIGatewayClaudeModels):
        return str(model_id)
    if isinstance(model_id, ClaudeModels):
        gateway_model = CLAUDE_TO_AI_GATEWAY_MODEL_MAP.get(model_id)
        if gateway_model is None:
            raise ValueError(f"Unsupported Claude model for AI Gateway: {model_id}")
        return str(gateway_model)
    if isinstance(model_id, str):
        # Try to interpret as enum string names or values
        try:
            return _resolve_model_id(AIGatewayClaudeModels(model_id))
        except ValueError:
            # Not a gateway enum value; attempt to parse as Claude model value or name
            try:
                return _resolve_model_id(ClaudeModels(model_id))
            except ValueError:
                # Final attempt: maybe the caller passed the enum member name
                try:
                    return _resolve_model_id(ClaudeModels[model_id])
                except (KeyError, ValueError):
                    raise ValueError(
                        f"Model identifier '{model_id}' is not recognised as an Anthropic or AI Gateway model."
                    ) from None

    raise TypeError(
        f"model_id must be one of {SUPPORTED_MODEL_ID_TYPES}, got {type(model_id).__name__}: {model_id}"
    )


@attrs
class AgClaudeApiInferencer(StreamingInferencerBase):
    """
    AI Gateway Claude inferencer with streaming support.

    Extends StreamingInferencerBase (like PlugboardApiInferencer) so that
    streaming is built-in. Supports three modes to reach the AI Gateway:
    - "direct": Shell out to `atlas slauth token` CLI (no local server needed)
    - "proximity": Forward to localhost proximity proxy (needs `proximity ai-gateway`)
    - "slauth_server": Use AI Gateway SDK with SlauthServerAuthFilter (original approach)
    - "auto": Auto-detect first available mode with runtime fallback

    Usage:
        inferencer = AgClaudeApiInferencer(model_id="claude-sonnet-4-20250514")

        # Sync (non-streaming)
        result = inferencer("What is AI?", max_new_tokens=1024)

        # Async streaming
        async for chunk in inferencer.ainfer_streaming("Tell me a story"):
            print(chunk, end="", flush=True)

        # Sync streaming bridge
        for chunk in inferencer.infer_streaming("Tell me a story"):
            print(chunk, end="", flush=True)

    Attributes:
        gateway_mode: Gateway access mode (default "auto").
        proximity_port: Port for the proximity proxy (default 29576).
        system_prompt: System prompt for all requests.
        max_tokens: Maximum tokens to generate (default 8192).
        temperature: Sampling temperature (default 0.7).
    """

    # Gateway configuration
    gateway_mode: str = attrib(default="auto")
    proximity_port: int = attrib(default=DEFAULT_PROXIMITY_PORT)

    # Generation configuration
    system_prompt: str = attrib(default="")
    max_tokens: int = attrib(default=8192)
    temperature: float = attrib(default=0.7)

    # Multi-turn message override (internal)
    _messages_override: Optional[list] = attrib(default=None, init=False)

    def __attrs_post_init__(self):
        super(AgClaudeApiInferencer, self).__attrs_post_init__()

        # AG auth is handled by SLAuth/proximity — dummy key for base class compat
        if not self._secret_key:
            self._secret_key = "ag-slauth-auth"

        if not self.model_id:
            self.model_id = str(AIGatewayClaudeModels.CLAUDE_46_OPUS)
        else:
            self.model_id = _resolve_model_id(self.model_id)

    def set_messages(self, messages: list) -> None:
        """Set explicit API messages for the next inference call.

        When set, _ainfer_streaming passes these directly instead of
        wrapping the prompt string as a single user message.
        Cleared after each streaming call.
        """
        self._messages_override = messages

    def _apply_defaults(self, args: dict) -> dict:
        """Apply instance-level defaults for generation params.

        User-provided values (from call kwargs) take precedence over
        instance attributes. This prevents 'got multiple values for
        keyword argument' errors when the base class merges user kwargs
        into _inference_args.
        """
        args.setdefault('max_new_tokens', self.max_tokens)
        args.setdefault('temperature', self.temperature)
        if self.system_prompt:
            args.setdefault('system', self.system_prompt)
        return args

    def _infer(self, inference_input: str, inference_config: Any = None, **_inference_args) -> str:
        """Execute sync inference, injecting gateway_mode and proximity_port."""
        self._apply_defaults(_inference_args)
        response = ai_gateway_generate_text(
            inference_input,
            model=self.model_id,
            api_key=self.secret_key,
            gateway_mode=self.gateway_mode,
            proximity_port=self.proximity_port,
            **_inference_args
        )
        return response

    async def _ainfer(self, inference_input: Any, inference_config: Any = None, **_inference_args) -> str:
        """Direct async inference via generate_text_async().

        Calls the non-streaming async API for efficiency when streaming
        is not needed.
        """
        self._apply_defaults(_inference_args)
        response = await ai_gateway_generate_text_async(
            inference_input,
            model=self.model_id,
            gateway_mode=self.gateway_mode,
            proximity_port=self.proximity_port,
            **_inference_args,
        )
        logger.debug("AG async response: %s", response[:200] if isinstance(response, str) and response else "")
        return response

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text chunks from AG Gateway streaming API."""
        messages = self._messages_override
        if messages is not None:
            self._messages_override = None
        else:
            messages = prompt  # generate_text_streaming handles str -> messages conversion

        self._apply_defaults(kwargs)
        async for chunk in ai_gateway_generate_text_streaming(
            prompt_or_messages=messages,
            model=self.model_id,
            gateway_mode=self.gateway_mode,
            proximity_port=self.proximity_port,
            **kwargs,
        ):
            yield chunk
