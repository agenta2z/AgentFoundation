"""AI Gateway OpenAI (GPT) inferencer with streaming support.

Mirror of ``ag_claude_api_inferencer.AgClaudeApiInferencer`` but for the OpenAI
vendor route on the AI Gateway. Extends ``StreamingInferencerBase`` so that
streaming is built-in via ``ainfer_streaming()`` / ``infer_streaming()``.

Routes through the AI Gateway's OpenAI chat-completions endpoint, so it uses the
same SLAuth/use-case auth as the Claude inferencer (no raw OpenAI api key).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterable, Optional, Union

from attr import attrib, attrs

from agent_foundation.apis.ag.ai_gateway_openai_llm import (
    AIGatewayOpenAIModels,
    generate_text as ai_gateway_generate_text,
    generate_text_async as ai_gateway_generate_text_async,
    generate_text_streaming as ai_gateway_generate_text_streaming,
)
from agent_foundation.apis.ag.gateway_mode import DEFAULT_PROXIMITY_PORT
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)

logger = logging.getLogger(__name__)


# Tuple of supported model identifier inputs for quick isinstance checks
SUPPORTED_MODEL_ID_TYPES: Iterable[type] = (
    str,
    AIGatewayOpenAIModels,
)


def _resolve_model_id(model_id: Union[str, AIGatewayOpenAIModels]) -> str:
    """Normalise model identifiers to the gateway's OpenAI model string.

    Accepts an ``AIGatewayOpenAIModels`` member, one of its values, or one of
    its member *names*. Any other string is passed through verbatim (the gateway
    will validate it), so newly-offered model ids work without a code change.
    """
    if isinstance(model_id, AIGatewayOpenAIModels):
        return str(model_id)

    if isinstance(model_id, str):
        # Try value first (e.g. "gpt-5.5-2026-04-23").
        try:
            return str(AIGatewayOpenAIModels(model_id))
        except ValueError:
            pass
        # Then try the member name (e.g. "GPT_55").
        try:
            return str(AIGatewayOpenAIModels[model_id])
        except (KeyError, ValueError):
            # Unknown to the enum — pass through; the gateway will validate it.
            return model_id

    raise TypeError(
        f"model_id must be one of {SUPPORTED_MODEL_ID_TYPES}, "
        f"got {type(model_id).__name__}: {model_id}"
    )


@attrs
class AgOpenAIApiInferencer(StreamingInferencerBase):
    """AI Gateway OpenAI (GPT) inferencer with streaming support.

    Extends ``StreamingInferencerBase`` (like ``AgClaudeApiInferencer``) so that
    streaming is built-in. Supports the gateway access modes exposed by the
    backend: "direct", "sdk", "slauth_server", and "auto" (default).

    Usage:
        inferencer = AgOpenAIApiInferencer(model_id="gpt-5.5-2026-04-23")

        # Sync (non-streaming)
        result = inferencer("What is AI?", max_new_tokens=1024)

        # Async streaming
        async for chunk in inferencer.ainfer_streaming("Tell me a story"):
            print(chunk, end="", flush=True)

    Attributes:
        gateway_mode: Gateway access mode (default "auto").
        proximity_port: Port for the proximity proxy (unused by the OpenAI
            route; kept for signature parity with the Claude inferencer).
        system_prompt: System prompt for all requests.
        max_tokens: Maximum tokens to generate (default 8192).
        temperature: Sampling temperature (ignored by reasoning models).
        reasoning_effort: Optional effort tier for reasoning models
            (none/minimal/low/medium/high).
    """

    # Gateway configuration
    gateway_mode: str = attrib(default="auto")
    proximity_port: int = attrib(default=DEFAULT_PROXIMITY_PORT)

    # Generation configuration
    system_prompt: str = attrib(default="")
    max_tokens: int = attrib(default=8192)
    temperature: float = attrib(default=0.7)
    reasoning_effort: Optional[str] = attrib(default=None)

    # Multi-turn message override (internal)
    _messages_override: Optional[list] = attrib(default=None, init=False)

    def __attrs_post_init__(self):
        super(AgOpenAIApiInferencer, self).__attrs_post_init__()

        # AG auth is handled by SLAuth — dummy key for base class compat.
        if not self._secret_key:
            self._secret_key = "ag-slauth-auth"

        if not self.model_id:
            self.model_id = str(AIGatewayOpenAIModels.GPT_55)
        else:
            self.model_id = _resolve_model_id(self.model_id)

    def set_messages(self, messages: list) -> None:
        """Set explicit API messages for the next inference call.

        When set, ``_ainfer_streaming`` passes these directly instead of
        wrapping the prompt string as a single user message. Cleared after each
        streaming call.
        """
        self._messages_override = messages

    def _apply_defaults(self, args: dict) -> dict:
        """Apply instance-level defaults for generation params.

        User-provided values (from call kwargs) take precedence over instance
        attributes, mirroring ``AgClaudeApiInferencer``.
        """
        args.setdefault('max_new_tokens', self.max_tokens)
        args.setdefault('temperature', self.temperature)
        if self.system_prompt:
            args.setdefault('system', self.system_prompt)
        if self.reasoning_effort:
            args.setdefault('reasoning_effort', self.reasoning_effort)
        return args

    def _infer(self, inference_input: str, inference_config: Any = None, **_inference_args) -> str:
        """Execute sync inference via the OpenAI gateway backend."""
        self._apply_defaults(_inference_args)
        return ai_gateway_generate_text(
            inference_input,
            model=self.model_id,
            api_key=self.secret_key,
            gateway_mode=self.gateway_mode,
            proximity_port=self.proximity_port,
            **_inference_args,
        )

    async def _ainfer(self, inference_input: Any, inference_config: Any = None, **_inference_args) -> str:
        """Direct async (non-streaming) inference via the OpenAI gateway backend."""
        self._apply_defaults(_inference_args)
        response = await ai_gateway_generate_text_async(
            inference_input,
            model=self.model_id,
            gateway_mode=self.gateway_mode,
            proximity_port=self.proximity_port,
            **_inference_args,
        )
        logger.debug(
            "AG OpenAI async response: %s",
            response[:200] if isinstance(response, str) and response else "",
        )
        return response

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text chunks from the AG OpenAI streaming API."""
        messages = self._messages_override
        if messages is not None:
            self._messages_override = None
        else:
            messages = prompt  # backend converts str -> messages

        self._apply_defaults(kwargs)
        async for chunk in ai_gateway_generate_text_streaming(
            prompt_or_messages=messages,
            model=self.model_id,
            gateway_mode=self.gateway_mode,
            proximity_port=self.proximity_port,
            **kwargs,
        ):
            yield chunk
