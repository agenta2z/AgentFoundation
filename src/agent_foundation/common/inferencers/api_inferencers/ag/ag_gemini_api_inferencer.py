"""AI Gateway Gemini (Google) inferencer with streaming support.

Mirror of ``ag_openai_api_inferencer.AgOpenAIApiInferencer`` but for the Gemini
(Google) vendor route on the AI Gateway. Extends ``StreamingInferencerBase`` so
streaming is built-in via ``ainfer_streaming()`` / ``infer_streaming()``.

Routes through the AI Gateway's Gemini OpenAI-compatible chat-completions
endpoint, using SLAuth/use-case auth (no raw Google api key).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterable, Optional, Union

from attr import attrib, attrs

from agent_foundation.apis.ag.ai_gateway_gemini_llm import (
    AIGatewayGeminiModels,
    generate_text as ai_gateway_generate_text,
    generate_text_async as ai_gateway_generate_text_async,
    generate_text_streaming as ai_gateway_generate_text_streaming,
)
from agent_foundation.apis.ag.gateway_mode import DEFAULT_PROXIMITY_PORT
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_ID_TYPES: Iterable[type] = (str, AIGatewayGeminiModels)


def _resolve_model_id(model_id: Union[str, AIGatewayGeminiModels]) -> str:
    """Normalise model identifiers to the gateway's Gemini model string.

    Accepts an ``AIGatewayGeminiModels`` member, one of its values, or one of
    its member *names*. Any other string is passed through verbatim (the gateway
    validates it), so newly-offered model ids work without a code change.
    """
    if isinstance(model_id, AIGatewayGeminiModels):
        return str(model_id)

    if isinstance(model_id, str):
        try:
            return str(AIGatewayGeminiModels(model_id))
        except ValueError:
            pass
        try:
            return str(AIGatewayGeminiModels[model_id])
        except (KeyError, ValueError):
            return model_id

    raise TypeError(
        f"model_id must be one of {SUPPORTED_MODEL_ID_TYPES}, "
        f"got {type(model_id).__name__}: {model_id}"
    )


@attrs
class AgGeminiApiInferencer(StreamingInferencerBase):
    """AI Gateway Gemini (Google) inferencer with streaming support.

    Extends ``StreamingInferencerBase`` (like ``AgOpenAIApiInferencer``).
    Supports gateway access modes: "direct", "sdk", "slauth_server", "auto".

    Usage:
        inferencer = AgGeminiApiInferencer(model_id="gemini-3.1-pro-preview")
        result = inferencer("What is AI?", max_new_tokens=1024)

        async for chunk in inferencer.ainfer_streaming("Tell me a story"):
            print(chunk, end="", flush=True)

    Attributes:
        gateway_mode: Gateway access mode (default "auto").
        proximity_port: Kept for signature parity (unused by the Gemini route).
        system_prompt: System prompt for all requests.
        max_tokens: Maximum tokens to generate (default 8192).
        temperature: Sampling temperature (Gemini accepts 0.0).
        top_p: Optional nucleus sampling.
        seed: Optional seed for determinism.
        reasoning_effort: Optional effort tier (low/medium/high).
    """

    gateway_mode: str = attrib(default="auto")
    proximity_port: int = attrib(default=DEFAULT_PROXIMITY_PORT)

    system_prompt: str = attrib(default="")
    max_tokens: int = attrib(default=8192)
    temperature: float = attrib(default=0.7)
    top_p: Optional[float] = attrib(default=None)
    seed: Optional[int] = attrib(default=None)
    reasoning_effort: Optional[str] = attrib(default=None)

    _messages_override: Optional[list] = attrib(default=None, init=False)

    def __attrs_post_init__(self):
        super(AgGeminiApiInferencer, self).__attrs_post_init__()

        if not self._secret_key:
            self._secret_key = "ag-slauth-auth"

        if not self.model_id:
            self.model_id = str(AIGatewayGeminiModels.GEMINI_31_PRO)
        else:
            self.model_id = _resolve_model_id(self.model_id)

    def set_messages(self, messages: list) -> None:
        """Set explicit API messages for the next inference call."""
        self._messages_override = messages

    def _apply_defaults(self, args: dict) -> dict:
        args.setdefault('max_new_tokens', self.max_tokens)
        args.setdefault('temperature', self.temperature)
        if self.top_p is not None:
            args.setdefault('top_p', self.top_p)
        if self.seed is not None:
            args.setdefault('seed', self.seed)
        if self.system_prompt:
            args.setdefault('system', self.system_prompt)
        if self.reasoning_effort:
            args.setdefault('reasoning_effort', self.reasoning_effort)
        return args

    def _infer(self, inference_input: str, inference_config: Any = None, **_inference_args) -> str:
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
        self._apply_defaults(_inference_args)
        response = await ai_gateway_generate_text_async(
            inference_input,
            model=self.model_id,
            gateway_mode=self.gateway_mode,
            proximity_port=self.proximity_port,
            **_inference_args,
        )
        logger.debug(
            "AG Gemini async response: %s",
            response[:200] if isinstance(response, str) and response else "",
        )
        return response

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        messages = self._messages_override
        if messages is not None:
            self._messages_override = None
        else:
            messages = prompt

        self._apply_defaults(kwargs)
        async for chunk in ai_gateway_generate_text_streaming(
            prompt_or_messages=messages,
            model=self.model_id,
            gateway_mode=self.gateway_mode,
            proximity_port=self.proximity_port,
            **kwargs,
        ):
            yield chunk
