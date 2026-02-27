from attr import attrs

from typing import Iterable, Union

from agent_foundation.apis.claude_llm import ClaudeModels
from agent_foundation.apis.ag.ai_gateway_claude_llm import (
    AIGatewayClaudeModels,
    generate_text as ai_gateway_generate_text,
)
from agent_foundation.common.inferencers.api_inferencer_base import ApiInferencerBase


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
    ClaudeModels.CLAUDE_41OPUS: AIGatewayClaudeModels.CLAUDE_41_OPUS,
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
class AgClaudeApiInferencer(ApiInferencerBase):
    """
    AI Gateway Claude implementation of `ApiInferencerBase` for handling inference requests via AI Gateway.

    This class sets up AI Gateway's Claude API function for generating text, extending `ApiInferencerBase` to handle
    request calls with Claude-specific configurations through the AI Gateway. It uses the `generate_text` function
    from the AI Gateway Claude API module and supports the standard interface for inference calls, leveraging the
    `_infer` and `_parse_response` methods from its base classes.

    Protocol Conformance:
        This class implements the `ReasonerProtocol` interface through its inherited `__call__` method,
        making it compatible with Agent reasoner requirements. The signature:
        __call__(inference_input, inference_config=None, **kwargs) -> response
        matches the ReasonerProtocol expectations.

    Example:
        >>> from agent_foundation.agents.agent import ReasonerProtocol
        >>> inferencer = AgClaudeApiInferencer(
        ...     model_id="claude-sonnet-4-20250514"
        ... )
        >>> # Verify protocol conformance (works without API call)
        >>> assert isinstance(inferencer, ReasonerProtocol)
        >>> # Verify it's callable
        >>> assert callable(inferencer)
        >>> # Actual API usage (requires credentials and network):
        >>> # result = inferencer("What is the capital of Japan?", max_new_tokens=1024, temperature=0.7)
        >>> # print('Tokyo' in result)

    Notes:
        - This class relies on the `generate_text` function from the Claude API to handle the
          actual API calls. Ensure that the required dependencies and access credentials are configured.
        - The `model_id`, `secret_key`, and retry configurations are inherited from `ApiInferencerBase`
          and `InferencerBase`, which provides general mechanism for retrying and error management.
    """

    def __attrs_post_init__(self):
        super(AgClaudeApiInferencer, self).__attrs_post_init__()
        self._inference_api = ai_gateway_generate_text

        if not self.model_id:
            self.model_id = str(AIGatewayClaudeModels.CLAUDE_45_SONNET)
        else:
            self.model_id = _resolve_model_id(self.model_id)