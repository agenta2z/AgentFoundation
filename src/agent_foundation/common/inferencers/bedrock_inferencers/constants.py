from science_modeling_tools.common.inferencers.inference_args import CommonLlmInferenceArgs
from rich_python_utils.common_utils import dict_

# region Claude3 constants
BEDROCK_ANTHROPIC_VERSION = 'bedrock-2023-05-31'

MODEL_ID_CLAUDE35_SONNET = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
MODEL_ID_CLAUDE3_SONNET = 'anthropic.claude-3-sonnet-20240229-v1:0'
MODEL_ID_CLAUDE3_HAIKU = 'anthropic.claude-3-haiku-20240307-v1:0'

DEFAULT_INFERENCE_ARGS_CLAUDE3 = dict_(CommonLlmInferenceArgs(), ignore_none_values=True)
# endregion
