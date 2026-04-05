# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""LLM-based context compressor for AgenticDynamicContext.

Wraps an InferencerBase and uses it to compress accumulated action history
into a shorter summary while preserving key information (tool names, outcomes,
chronological order).
"""

from __future__ import annotations

import logging
from typing import Optional

from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)

logger = logging.getLogger(__name__)

_DEFAULT_COMPRESSION_PROMPT = """\
You are a context compression assistant. Your task is to compress the following \
action history into a concise summary that preserves ALL essential information.

## Rules
1. Preserve the chronological order of actions
2. Keep all tool names and their key outcomes
3. Preserve any paths, IDs, numbers, or other specific values mentioned
4. Remove redundant or verbose descriptions
5. Use bullet points for clarity
6. The compressed output MUST be under {max_length} characters
7. Do NOT add any commentary — output ONLY the compressed summary

## Action History to Compress
{context}

## Compressed Summary:"""


class InferencerContextCompressor:
    """Compresses dynamic context using an LLM inferencer.

    Conforms to ContextCompressorCallable protocol:
        async def __call__(self, context: str, max_length: int) -> str

    Args:
        inferencer: Any InferencerBase instance (e.g., MetagenApiInferencer).
            Can be a lightweight/fast model for cost efficiency.
        prompt_template: Custom compression prompt. Must contain {context}
            and {max_length} placeholders. If None, uses default.
        fallback_on_error: If True, return truncated context on LLM error
            instead of raising. Default True.
    """

    def __init__(
        self,
        inferencer: InferencerBase,
        prompt_template: Optional[str] = None,
        fallback_on_error: bool = True,
    ) -> None:
        self._inferencer = inferencer
        self._prompt_template = prompt_template or _DEFAULT_COMPRESSION_PROMPT
        self._fallback_on_error = fallback_on_error

    async def __call__(self, context: str, max_length: int) -> str:
        """Compress context text to fit within max_length chars."""
        # Skip compression if already within budget
        if len(context) <= max_length:
            return context

        prompt = self._prompt_template.format(
            context=context,
            max_length=max_length,
        )

        try:
            compressed = await self._inferencer.ainfer(prompt)
            compressed = compressed.strip()

            # If LLM output still exceeds max_length, hard-truncate
            if len(compressed) > max_length:
                logger.warning(
                    "Compressed context (%d chars) exceeds max_length (%d), truncating",
                    len(compressed),
                    max_length,
                )
                compressed = compressed[:max_length]

            logger.info(
                "Context compressed: %d -> %d chars (%.0f%% reduction)",
                len(context),
                len(compressed),
                (1 - len(compressed) / len(context)) * 100,
            )
            return compressed

        except Exception as e:
            logger.error("Context compression failed: %s", e)
            if self._fallback_on_error:
                # Fallback: hard-truncate with marker
                return context[: max_length - 20] + "\n... (truncated)"
            raise
