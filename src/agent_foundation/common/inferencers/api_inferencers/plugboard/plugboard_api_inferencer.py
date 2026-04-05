# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Plugboard inferencer with native streaming support.

Extends StreamingInferencerBase so that streaming capability is built-in.
Plugboard uses CAT auth (no API key needed).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)

logger = logging.getLogger(__name__)


@attrs
class PlugboardApiInferencer(StreamingInferencerBase):
    """Plugboard inferencer with native streaming support.

    Inherits from StreamingInferencerBase instead of ApiInferencerBase
    so that streaming capability is built-in via ainfer_streaming().
    Plugboard uses CAT auth, so the secret_key is not required.

    Usage:
        inferencer = PlugboardApiInferencer(model_id="claude-4-sonnet-genai")
        result = inferencer.infer("What is the meaning of life?")
        result = await inferencer.ainfer("What is the meaning of life?")

        async for chunk in inferencer.ainfer_streaming("Tell me a story"):
            print(chunk, end="")
    """

    system_prompt: str = attrib(default="")
    max_tokens: int = attrib(default=4096)
    temperature: float = attrib(default=0.7)
    pipeline: str = attrib(default="usecase-dev-ai")
    model_pipeline_overrides: dict = attrib(factory=dict)
    _messages_override: Optional[list] = attrib(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        super(PlugboardApiInferencer, self).__attrs_post_init__()
        # Plugboard uses CAT auth — set a dummy key to avoid AttributeError
        if not self._secret_key:
            self._secret_key = "plugboard-cat-auth"

    def set_messages(self, messages: list) -> None:
        """Set explicit API messages for the next inference call.

        When set, _ainfer_streaming passes these directly instead of
        wrapping the prompt string as a single user message.
        Cleared after each streaming call.
        """
        self._messages_override = messages

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args: Any
    ) -> str:
        """Sync inference via Plugboard generate_text()."""
        from agent_foundation.apis.plugboard import generate_text

        return generate_text(
            inference_input,
            model=self.model_id,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            system_prompt=self.system_prompt,
            pipeline=self.pipeline,
            model_pipeline_overrides=self.model_pipeline_overrides,
            **_inference_args,
        )

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args: Any
    ) -> str:
        """Direct async inference via generate_text_async().

        Calls the non-streaming API directly for efficiency when streaming
        is not needed.
        """
        from agent_foundation.apis.plugboard import generate_text_async

        response = await generate_text_async(
            inference_input,
            model=self.model_id,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            system_prompt=self.system_prompt,
            pipeline=self.pipeline,
            model_pipeline_overrides=self.model_pipeline_overrides,
            **_inference_args,
        )
        logger.debug("Plugboard async response: %s", response[:200] if response else "")
        return response

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text chunks from Plugboard streaming API."""
        from agent_foundation.apis.plugboard import (
            generate_text_streaming,
        )

        messages = self._messages_override
        if messages is not None:
            self._messages_override = None
        else:
            messages = [{"role": "user", "content": prompt}]

        async for chunk in generate_text_streaming(
            prompt_or_messages=messages,
            model=self.model_id,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            system_prompt=self.system_prompt,
            pipeline=self.pipeline,
            model_pipeline_overrides=self.model_pipeline_overrides,
        ):
            yield chunk
