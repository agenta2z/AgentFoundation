from typing import List, Optional

from attr import attrs, attrib



@attrs(slots=True)
class CommonLlmInferenceArgs:
    """
    Class for storing common large language model inference arguments.

    Attributes:
        temperature (float): Controls the randomness of predictions by scaling the logits before applying softmax.
            Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.1) make it more deterministic.
            Default is 0.1.
        max_tokens (int): Depending on the model, this pareameter might have slightly different interpretations:
            1) the maximum number of new tokens to generate in the response;
            2) the maximum number of tokens (including input tokens) to process by the model, and then maximum number of tokens is dynamic and is this parameter value minus the actual number of input tokens;
            In both interpretations, this parameter limits the length of the generated output. Default is 2000.
        max_new_tokens (int): Some models use this parameter to specify the maximum number of new tokens to generate in the response. You might only use one of `max_tokens` and `max_new_tokens` and set the other as `None`.
        top_p (float): Nucleus sampling probability threshold. Only the smallest set of tokens with a cumulative
            probability above this threshold are considered. A value of 0.9 means that the model will only consider
            tokens that comprise the top 90% probability mass. Default is 0.9.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k filtering.
            If set to 50, only the top 50 tokens with the highest probabilities are considered for each step.
            Default is 50.
        eos_token_id (Optional[int]): The ID of the end-of-sequence token, if used. If set, the generation stops upon generating this token. Default is None.
        stop_sequences (List[str]): A list of sequences where the model should stop generating further tokens.
            The generation will stop if any of these sequences are generated. Default is an empty list.

    Example:
        >>> from rich_python_utils.common_utils import dict_
        >>> args = CommonLlmInferenceArgs(
        ...     temperature=0.7,
        ...     max_tokens=150,
        ...     top_p=0.95,
        ...     top_k=40,
        ...     stop_sequences=['<|endoftext|>']
        ... )
        >>> dict_(args,ignore_none_values=True)
        {'temperature': 0.7, 'max_tokens': 150, 'top_p': 0.95, 'top_k': 40, 'stop_sequences': ['<|endoftext|>']}
    """
    temperature: float = attrib(default=1.0)
    max_tokens: Optional[int] = attrib(default=2000)
    max_new_tokens: int = attrib(default=None)
    top_p: float = attrib(default=0.9)
    top_k: int = attrib(default=50)
    eos_token_id: Optional[int] = attrib(default=None)
    stop_sequences: Optional[List[str]] = attrib(default=None)
