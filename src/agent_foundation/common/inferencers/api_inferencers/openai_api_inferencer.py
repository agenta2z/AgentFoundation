from attr import attrs

from agent_foundation.common.inferencers.api_inferencer_base import ApiInferencerBase


@attrs
class OpenaiApiInferencer(ApiInferencerBase):
    """
    OpenAI-specific implementation of `ApiInferencerBase` for managing inference requests to OpenAI's API.

    This class initializes the OpenAI API function `generate_text` to facilitate inference, leveraging the
    functionality of `ApiInferencerBase` to handle structured requests and responses. It sets the OpenAI
    API key environment variable as a default for the `secret_key` attribute if not explicitly provided,
    ensuring secure access to OpenAI's models.

    Attributes:
        _generate_text_api (Callable[[str, str, ...], Any]): The OpenAI API function used to process
            inference prompts based on the model, secret key, and other inference parameters.

    Example:
        >>> inferencer = OpenaiApiInferencer(
        ...     model_id="gpt-4o",
        ... )
        >>> prompt = "What is the capital of Japan?"
        >>> result = inferencer(prompt, max_new_tokens=1024, temperature=0.7)
        >>> print('Tokyo' in result)
        True

    Notes:
        - This class relies on the `generate_text` function from OpenAI's API module to perform
          inference requests. Ensure that dependencies are installed and the OpenAI API key
          is set up in the environment.
        - Inherited attributes from `ApiInferencerBase` and `InferencerBase`, such as `model_id`,
          `max_retry`, and retry configurations, provide additional inference customization and error handling.
    """

    def __attrs_post_init__(self):
        from agent_foundation.apis.openai_llm import generate_text, ENV_NAME_OPENAI_API_KEY
        self._inference_api = generate_text

        if not self._secret_key:
            self._secret_key = ENV_NAME_OPENAI_API_KEY