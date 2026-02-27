from attr import attrs

from agent_foundation.apis.claude_llm import ClaudeModels, DEFAULT_CLAUDE_MODEL
from agent_foundation.common.inferencers.api_inferencer_base import ApiInferencerBase


@attrs
class ClaudeApiInferencer(ApiInferencerBase):
    """
    Claude-specific implementation of `ApiInferencerBase` for handling inference requests to the Claude API.

    This class sets up Claude's API function for generating text, extending `ApiInferencerBase` to handle
    request calls with Claude-specific configurations. It uses the `generate_text` function from the Claude API
    module and supports the standard interface for inference calls, leveraging the `_infer` and `_parse_response`
    methods from its base classes.

    Protocol Conformance:
        This class implements the `ReasonerProtocol` interface through its inherited `__call__` method,
        making it compatible with Agent reasoner requirements. The signature:
        __call__(inference_input, inference_config=None, **kwargs) -> response
        matches the ReasonerProtocol expectations.

    Example:
        >>> from agent_foundation.agents.agent import ReasonerProtocol
        >>> inferencer = ClaudeApiInferencer()
        >>> # Verify protocol conformance (works without API call)
        >>> assert isinstance(inferencer, ReasonerProtocol)
        >>> # Verify it's callable
        >>> assert callable(inferencer)
        >>> # Actual API usage (requires credentials and network):
        >>> result = inferencer("What is the capital of Japan?", max_new_tokens=1024, temperature=0.7)
        >>> assert('Tokyo' in result)

    Notes:
        - This class relies on the `generate_text` function from the Claude API to handle the
          actual API calls. Ensure that the required dependencies and access credentials are configured.
        - The `model_id`, `secret_key`, and retry configurations are inherited from `ApiInferencerBase`
          and `InferencerBase`, which provides general mechanism for retrying and error management.
    """

    def __attrs_post_init__(self):
        super(ClaudeApiInferencer, self).__attrs_post_init__()
        from agent_foundation.apis.claude_llm import generate_text, ENV_NAME_CLAUDE_API_KEY
        self._inference_api = generate_text

        if not self._secret_key:
            self._secret_key = ENV_NAME_CLAUDE_API_KEY

        if not self.model_id:
            self.model_id = DEFAULT_CLAUDE_MODEL