from attr import attrs

from science_modeling_tools.apis.metagen import load_model_to_key_map, MetaGenModels
from science_modeling_tools.common.inferencers.api_inferencer_base import (
    ApiInferencerBase,
)


@attrs
class MetagenApiInferencer(ApiInferencerBase):
    """
    MetaGen-specific implementation of `ApiInferencerBase` for handling inference requests to the MetaGen API.
    If no `secret_key` is provided, a default secret key will be used.
    """

    def __attrs_post_init__(self):
        super(MetagenApiInferencer, self).__attrs_post_init__()
        from science_modeling_tools.apis.metagen import generate_text

        self._inference_api = generate_text

        if not self.model_id:
            self.model_id = MetaGenModels.CLAUDE_4_SONNET

        if not self._secret_key:
            try:
                model_to_key_map = load_model_to_key_map()
                self._secret_key = model_to_key_map[str(self.model_id)][0]
            except Exception:
                raise AttributeError(
                    f"Unable to find default secret key for model '{self.model_id}'"
                )
