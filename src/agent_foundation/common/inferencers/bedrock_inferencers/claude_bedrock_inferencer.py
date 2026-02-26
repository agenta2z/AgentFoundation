from attr import attrs, attrib

from science_modeling_tools.common.inferencers.bedrock_inferencers.bedrock_inferencer import BedrockInferencer
from science_modeling_tools.common.inferencers.bedrock_inferencers.constants import (
    DEFAULT_INFERENCE_ARGS_CLAUDE3,
    MODEL_ID_CLAUDE3_HAIKU,
    BEDROCK_ANTHROPIC_VERSION
)


@attrs
class Claude3BedrockInferencer(BedrockInferencer):
    """
    An inferencer class for performing inference using the Claude 3 model on the Bedrock platform.

    This class extends `BedrockInferencer` and is tailored specifically for Claude 3 inference tasks,
    handling model-specific setup, request construction, and response parsing.

    Attributes:
        model_id (str): The identifier for the Claude 3 model. Defaults to `MODEL_ID_CLAUDE3_HAIKU` if not specified.
        anthropic_version (str): Defaults to `BEDROCK_ANTHROPIC_VERSION` if not specified.
        default_inference_args (dict): Default inference arguments specific to Claude 3. Defaults to `DEFAULT_INFERENCE_ARGS_CLAUDE3`.

    Examples:
        # An example of creating an Claude 3 Sonnet 3.5 inferencer and make inference.
        >>> inferencer = Claude3BedrockInferencer(
        ...        access_key=['key1', 'key2', 'key3'],
        ...        secret_key=['key1', 'key2', 'key3'],
        ...        model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
        ...        region='us-east-1',
        ...        min_retry_wait=3,
        ...        max_retry_wait=10,
        ...        max_retry=5
        ... )
        >>> inferencer('hello')
    """

    anthropic_version: str = attrib(default=BEDROCK_ANTHROPIC_VERSION)

    def __attrs_post_init__(self):
        """
        Post-initialization method that sets the model-specific defaults for Claude 3.

        This method ensures that the correct model ID, version, and default inference arguments are set
        for the Claude 3 model if they are not explicitly provided during initialization.

        Calls:
            super().__attrs_post_init__(): Calls the parent class's post-initialization method to finalize setup.
        """
        if not self.model_id:
            self.model_id = MODEL_ID_CLAUDE3_HAIKU
        if not self.anthropic_version:
            self.anthropic_version = BEDROCK_ANTHROPIC_VERSION
        if not self.default_inference_args:
            self.default_inference_args = DEFAULT_INFERENCE_ARGS_CLAUDE3
        super().__attrs_post_init__()

    def construct_request(self, inference_input: str, **_inference_args):
        """
        Constructs the request payload for the Claude 3 inference task.

        This method formats the prompt and additional inference arguments into a dictionary that matches
        the request structure required by the Claude 3 model on the Bedrock platform. The constructed
        request includes user messages and model-specific parameters.

        Args:
            inference_input (str): The input prompt for the inference.
            _inference_args: Additional keyword arguments to include in the request payload.

        Returns:
            dict: The constructed request payload with the formatted prompt and inference arguments.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": inference_input
                    },
                ],
            }
        ]

        request_body = {
            "anthropic_version": self.anthropic_version,
            "messages": messages,
            **_inference_args
        }

        return request_body

    def _parse_response(self, response) -> str:
        """
        Parses the response from the Claude 3 inference service to extract the generated text.

        This method navigates the response structure to retrieve the relevant inference result, which is
        expected to be contained in a specific content field within the response.

        Args:
            response (dict): The JSON response from the Claude 3 model on the Bedrock platform.

        Returns:
            str: The extracted text content from the response. If the expected fields are missing, this
            method may raise a KeyError or return an empty string, depending on implementation specifics.
        """
        return response.get("content")[0].get("text")
